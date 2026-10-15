import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from datasets import Dataset

from opencompass.configs.datasets.over_editing.over_editing_gen import \
    over_editing_datasets
from opencompass.datasets import (GENERIC_SYSTEM_PROMPT,
                                  OverEditingEvaluator,
                                  create_user_message,
                                  extract_code_from_response)
from opencompass.datasets.over_editing import sandbox
from opencompass.datasets.over_editing.sandbox import \
    isolated_untrusted_check, isolated_untrusted_check_many
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.utils import build_dataset_from_cfg


def _write_fixture(tmp_path):
    questions = [
        {
            'task_id': 'BigCodeBench/2',
            'prompt': 'Return the argument plus one',
            'corrupted_solution': 'def add_one(x):\n    return x',
            'canonical_solution': 'def add_one(x):\n    return x + 1',
            'test_code': 'provenance-only test',
            'mutation_type': 'arithmetic operators',
        },
        {
            'task_id': 'BigCodeBench/7',
            'prompt': 'Return the argument doubled',
            'corrupted_solution': 'def twice(x):\n    return x',
            'canonical_solution': 'def twice(x):\n    return x * 2',
            'test_code': 'provenance-only test',
            'mutation_type': 'arithmetic operators',
        },
    ]
    questions_path = tmp_path / 'questions.jsonl'
    questions_path.write_text(
        ''.join(json.dumps(row) + '\n' for row in questions),
        encoding='utf-8')
    tests = Dataset.from_list([
        {
            'task_id': 'BigCodeBench/2',
            'test': ('class TestCases(unittest.TestCase):\n'
                     '    def test_value(self):\n'
                     '        self.assertEqual(add_one(3), 4)'),
        },
        {
            'task_id': 'BigCodeBench/7',
            'test': ('class TestCases(unittest.TestCase):\n'
                     '    def test_value(self):\n'
                     '        self.assertEqual(twice(3), 6)'),
        },
    ])
    tests_path = tmp_path / 'v0.1.4-00000-of-00001.parquet'
    tests.to_parquet(tests_path)
    return questions_path, tests_path


def _dataset_cfg(tmp_path, selected_ids=None):
    questions_path, tests_path = _write_fixture(tmp_path)
    cfg = copy.deepcopy(over_editing_datasets[0])
    cfg.update(
        path=str(questions_path),
        tests_path=str(tests_path),
        expected_size=2,
        selected_ids=selected_ids,
    )
    return cfg


def test_registered_config_loads_joined_rows_and_native_prompt(tmp_path):
    cfg = _dataset_cfg(tmp_path, ['BigCodeBench/7', 'BigCodeBench/2'])
    benchmark = build_dataset_from_cfg(cfg)

    assert benchmark.test['task_id'] == [
        'BigCodeBench/7', 'BigCodeBench/2'
    ]
    assert benchmark.test[0]['official_test'].startswith('class TestCases')
    template_cfg = copy.deepcopy(cfg['infer_cfg']['prompt_template'])
    template_cfg.pop('type')
    template = PromptTemplate(**template_cfg)
    prompt = template.generate_item(dict(benchmark.test[0]))
    messages = [item for item in prompt if 'role' in item]
    assert messages == [
        {'role': 'SYSTEM', 'prompt': GENERIC_SYSTEM_PROMPT},
        {'role': 'HUMAN', 'prompt': benchmark.test[0]['user_prompt']},
    ]
    assert 'provenance-only test' not in messages[1]['prompt']


def test_prompt_erratum_and_response_extraction():
    generic = create_user_message('Do it', 'def f():\n    pass')
    explicit = create_user_message(
        'Do it', 'def f():\n    pass', is_explicit=True)
    assert GENERIC_SYSTEM_PROMPT == GENERIC_SYSTEM_PROMPT
    assert explicit == generic.replace(
        'What is wrong? Fix and complete my function.',
        'What is wrong? Fix and complete my function but keep as much of '
        'the original code as possible.')
    assert extract_code_from_response('def f():\n    return 1') == \
        'def f():\n    return 1'
    response = ('text\n```python\nx = 1\n```\n```python\n'
                'def f():\n    return 2\n```')
    assert extract_code_from_response(response) == 'def f():\n    return 2'
    assert extract_code_from_response(
        'reason</think>\n```python\ndef f():\n    return 3') == \
        'def f():\n    return 3'
    assert extract_code_from_response('No code here') == ''


def test_passing_only_metrics_and_zero_pass_boundary(tmp_path):
    benchmark = build_dataset_from_cfg(_dataset_cfg(tmp_path))
    evaluator = OverEditingEvaluator(
        execution_timeout=2, execution_workers=2)
    predictions = [
        ('```python\ndef add_one(x):\n'
         '    if True:\n'
         '        return x + 1\n```'),
        ('def twice(x):\n'
         '    if x:\n'
         '        return x * 3\n'
         '    return 0'),
    ]
    result = evaluator.score(predictions, test_set=benchmark.test)
    assert result['Pass@1 ↑'] == 0.5
    assert result['Excess Lev. ↓'] > 0
    assert result['Added CC ↓'] == 1
    assert [item['status'] for item in result['details']] == ['pass', 'fail']

    zero = evaluator.score(['', 'No Python'], test_set=benchmark.test)
    assert zero['Pass@1 ↑'] == 0
    assert zero['Excess Lev. ↓'] is None
    assert zero['Added CC ↓'] is None


def test_generated_code_runs_without_host_workspace_access(tmp_path):
    secret = tmp_path / 'host-secret.txt'
    secret.write_text('not available to generated code', encoding='utf-8')
    status, details = isolated_untrusted_check(
        'def host_secret_is_hidden():\n'
        f'    return not __import__("os").path.exists({str(secret)!r})',
        'class TestCases(unittest.TestCase):\n'
        '    def test_isolated(self):\n'
        '        self.assertTrue(host_secret_is_hidden())',
        timeout=2,
    )
    assert status == 'pass', details


def test_generated_subprocess_output_cannot_corrupt_protocol():
    status, details = isolated_untrusted_check(
        'import subprocess\n'
        'import sys\n'
        'def child_succeeds():\n'
        '    result = subprocess.run([sys.executable, "-c", '
        '"print(\\\"protocol noise\\\")"])\n'
        '    return result.returncode == 0',
        'class TestCases(unittest.TestCase):\n'
        '    def test_child(self):\n'
        '        self.assertTrue(child_succeeds())',
        timeout=2,
    )
    assert status == 'pass', details


def test_packaged_nltk_stopwords_are_mounted_and_preflighted(
        monkeypatch, tmp_path):
    sandbox._check_isolated_dependencies.cache_clear()
    command = sandbox._sandbox_command()
    data_dir = str(sandbox._NLTK_DATA_DIR)
    assert ['--setenv', 'NLTK_DATA', '/runtime/nltk_data'] == command[
        command.index('NLTK_DATA') - 1:command.index('NLTK_DATA') + 2]
    assert ['--ro-bind', data_dir, '/runtime/nltk_data'] == command[
        command.index(data_dir) - 1:command.index(data_dir) + 2]
    sandbox.check_isolated_dependencies({'nltk'})

    sandbox._check_isolated_dependencies.cache_clear()
    monkeypatch.setattr(sandbox, '_NLTK_DATA_DIR', tmp_path / 'missing')
    with pytest.raises(RuntimeError, match='NLTK resources.*stopwords'):
        sandbox.check_isolated_dependencies({'nltk'})


def test_isolated_protocol_serializes_nested_executor_launches(monkeypatch):
    entered = threading.Event()
    second_started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs['input'])
        entered.set()
        release.wait(2)
        return SimpleNamespace(
            returncode=0,
            stdout='{"status": "pass", "details": {}}',
            stderr='',
        )

    monkeypatch.setattr(sandbox, '_sandbox_command', lambda _path=None: [])
    monkeypatch.setattr(sandbox.subprocess, 'run', fake_run)

    def run_second():
        second_started.set()
        return isolated_untrusted_check('second', 'test', 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(isolated_untrusted_check, 'first', 'test', 1)
        assert entered.wait(1)
        second = pool.submit(run_second)
        assert second_started.wait(1)
        time.sleep(0.05)
        assert len(calls) == 1
        release.set()
        assert first.result() == ('pass', {})
        assert second.result() == ('pass', {})
    assert len(calls) == 2


def test_isolated_batch_reuses_worker_and_preserves_verdict_order(
        monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(json.loads(kwargs['input']))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({'items': [
                {'status': 'pass', 'details': {'case': 1}},
                {'status': 'fail', 'details': {'case': 2}},
            ]}),
            stderr='',
        )

    monkeypatch.setattr(sandbox, '_sandbox_command', lambda _path=None: [])
    monkeypatch.setattr(sandbox.subprocess, 'run', fake_run)
    result = isolated_untrusted_check_many([
        ('first', 'test one'), ('second', 'test two')
    ], timeout=3)

    assert result == [
        ('pass', {'case': 1}), ('fail', {'case': 2})
    ]
    assert len(calls) == 1
    assert calls[0] == {
        'mode': 'evaluate_many',
        'samples': [
            {'code': 'first', 'test_code': 'test one'},
            {'code': 'second', 'test_code': 'test two'},
        ],
        'timeout': 3,
    }


def test_loader_rejects_incomplete_test_join(tmp_path):
    cfg = _dataset_cfg(tmp_path)
    Dataset.from_list([{
        'task_id': 'BigCodeBench/2',
        'test': 'class TestCases(unittest.TestCase):\n    pass',
    }]).to_parquet(cfg['tests_path'])
    with pytest.raises(ValueError, match='join is incomplete'):
        build_dataset_from_cfg(cfg)


@pytest.mark.parametrize(('source', 'filename'), [
    ('path', 'missing-questions.jsonl'),
    ('tests_path', 'missing-tests.parquet'),
])
def test_loader_rejects_missing_sources(tmp_path, source, filename):
    cfg = _dataset_cfg(tmp_path)
    cfg[source] = str(tmp_path / filename)
    with pytest.raises(FileNotFoundError, match='requires.*not found'):
        build_dataset_from_cfg(cfg)


@pytest.mark.parametrize(('artifact', 'field', 'remove', 'value', 'message'), [
    ('question', 'prompt', True, None, 'question row 0.*missing.*prompt'),
    ('question', 'canonical_solution', False, None,
     'question row 0.*invalid.*canonical_solution'),
    ('question', 'test_code', False, '',
     'question row 0.*invalid.*test_code'),
    ('test', 'test', True, None,
     'BigCodeBench v0.1.4 test row 0.*missing.*test'),
    ('test', 'test', False, None,
     'BigCodeBench v0.1.4 test row 0.*invalid.*test'),
])
def test_loader_rejects_malformed_required_fields(
        tmp_path, artifact, field, remove, value, message):
    cfg = _dataset_cfg(tmp_path)
    if artifact == 'question':
        with open(cfg['path'], encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if remove:
            rows[0].pop(field)
        else:
            rows[0][field] = value
        with open(cfg['path'], 'w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    else:
        rows = Dataset.from_parquet(cfg['tests_path']).to_list()
        if remove:
            rows[0].pop(field)
        else:
            rows[0][field] = value
        Dataset.from_list(rows).to_parquet(cfg['tests_path'])
    with pytest.raises(ValueError, match=message):
        build_dataset_from_cfg(cfg)
