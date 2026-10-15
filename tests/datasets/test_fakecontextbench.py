import json
from pathlib import Path

import mmengine
import pytest
from mmengine.config import Config, ConfigDict

from opencompass.datasets.fakecontextbench import (
    PROTOCOL, FakeContextBenchDataset, FakeContextBenchEvaluator, classify)
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.tasks import OpenICLEvalTask
from opencompass.utils import build_dataset_from_cfg, get_infer_output_path


def _gold(real='8', context='24', response_format='numeric_or_symbolic',
          tolerance=0):
    return {
        'real_world_answer': real,
        'context_induced_answer': context,
        'response_format': response_format,
        'tolerance': tolerance,
    }


def test_dataset_loads_real_jsonl_shape_and_preserves_messages(tmp_path):
    messages = [{'role': 'user', 'content': 'Untrusted {prompt} text'}]
    row = {
        'case_id': 'case-1',
        'messages': messages,
        'gold': _gold(),
        'domain': 'physics',
        'theory_family': 'family-1',
        'metadata': {'query_parameters': {'mass': 4}},
    }
    second = {
        **row,
        'case_id': 'case-2',
        'metadata': {'query_parameters': ['2', '3']},
        'subdomain': 'algebra',
    }
    data = tmp_path / 'fakecontextbench.jsonl'
    data.write_text(
        json.dumps(row) + '\n' + json.dumps(second) + '\n',
        encoding='utf-8')

    dataset = FakeContextBenchDataset.load(path=str(data))

    assert dataset[0]['case_id'] == 'case-1'
    assert dataset[0]['messages'] == messages
    assert dataset[0]['gold']['context_induced_answer'] == '24'
    assert json.loads(dataset[0]['metadata']['query_parameters']) == {
        'mass': 4}
    assert json.loads(dataset[1]['metadata']['query_parameters']) == ['2', '3']
    assert dataset[0]['subdomain'] is None
    assert dataset[1]['subdomain'] == 'algebra'


def test_loader_rejects_duplicate_ids(tmp_path):
    row = {'case_id': 'same', 'messages': [{'role': 'user', 'content': 'q'}],
           'gold': _gold()}
    data = tmp_path / 'duplicate.jsonl'
    data.write_text('\n'.join((json.dumps(row), json.dumps(row))) + '\n',
                    encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate case_id'):
        FakeContextBenchDataset.load(path=str(data))


def test_native_config_fails_clearly_for_missing_local_file(tmp_path):
    config = Config.fromfile(
        'opencompass/configs/datasets/fakecontextbench/'
        'fakecontextbench_gen_0c1018.py')
    dataset_cfg = config.fakecontextbench_datasets[0]
    dataset_cfg.path = str(tmp_path / 'missing.jsonl')
    dataset_cfg.type = 'FakeContextBenchDataset'

    with pytest.raises(
            FileNotFoundError,
            match=r"Local file .*missing\.jsonl doesn't exist"):
        build_dataset_from_cfg(dataset_cfg)


@pytest.mark.parametrize(
    ('text', 'gold', 'expected', 'method'), [
        ('Answer: 8', _gold(), 'reality_correct',
         'exact_whole_response_reality'),
        ('The value is 24.', _gold(), 'context_follow',
         'full_only_context'),
        ('Neither registered value.', _gold(), 'other',
         'no_complete_target'),
        ('24 is suggested. Final answer: 8', _gold(), 'reality_correct',
         'both_full_tail_reality'),
        ('8 is standard. Conclusion: 24', _gold(), 'context_follow',
         'both_full_tail_context'),
        ('Result 1/2, not 1 or 2', _gold('1/2', '3/4'), 'reality_correct',
         'full_only_reality'),
        ('1.04', _gold('1.0', '2.0', tolerance=.05), 'reality_correct',
         'exact_whole_response_reality'),
        ('The value is 1/2', _gold('0.5', '0.25', 'short_text'), 'other',
         'no_complete_target'),
    ])
def test_official_classification_boundaries(text, gold, expected, method):
    label, _, audit = classify(text, gold)
    assert label == expected
    assert audit['method'] == method


def test_evaluator_aggregates_official_and_paper_metrics():
    evaluator = FakeContextBenchEvaluator()
    result = evaluator.score(['8', '24', 'unknown'], [_gold()] * 3)
    assert result['protocol'] == PROTOCOL
    assert result['n'] == 3
    assert result['counts'] == {
        'reality_correct': 1, 'context_follow': 1, 'other': 1}
    assert result['rates'] == {
        'reality_correct': pytest.approx(1 / 3),
        'context_follow': pytest.approx(1 / 3),
        'other': pytest.approx(1 / 3),
    }
    assert result['Reality'] == pytest.approx(100 / 3)
    assert result['Context'] == pytest.approx(100 / 3)
    assert result['Other'] == pytest.approx(100 / 3)
    assert result['Reality Rate'] == 50
    assert result['R_over_RplusC'] == .5
    assert result['mixed'] == 0
    assert all(d['scoring_audit']['protocol'] == PROTOCOL
               for d in result['details'])


def test_zero_denominator_is_explicitly_unscorable():
    result = FakeContextBenchEvaluator().score(['none'], [_gold()])
    assert result['R_over_RplusC'] is None
    assert result['Reality Rate'] is None


def test_native_config_runs_registered_loader_and_evaluator(tmp_path):
    messages = [{'role': 'user', 'content': 'literal {braces}'}]
    data = tmp_path / 'fakecontextbench.jsonl'
    data.write_text(json.dumps({
        'case_id': 'case-native',
        'messages': messages,
        'gold': _gold(),
        'domain': 'physics',
        'theory_family': 'family-native',
    }) + '\n', encoding='utf-8')

    config = Config.fromfile(
        'opencompass/configs/datasets/fakecontextbench/'
        'fakecontextbench_gen_0c1018.py')
    dataset_cfg = config.fakecontextbench_datasets[0]
    dataset_cfg.path = str(data)
    dataset_cfg.type = 'FakeContextBenchDataset'
    dataset_cfg.eval_cfg.evaluator.type = 'FakeContextBenchEvaluator'

    assert LOAD_DATASET.get(dataset_cfg.type) is FakeContextBenchDataset
    assert (ICL_EVALUATORS.get(dataset_cfg.eval_cfg.evaluator.type)
            is FakeContextBenchEvaluator)
    dataset = build_dataset_from_cfg(dataset_cfg)
    assert dataset.test[0]['case_id'] == 'case-native'
    assert dataset_cfg.reader_cfg.input_columns == ['messages']
    prompt = dataset_cfg.infer_cfg.prompt_template.type(
        **{k: v for k, v in dataset_cfg.infer_cfg.prompt_template.items()
           if k != 'type'})
    assert prompt.generate_item({'messages': messages}) == messages

    model_cfg = ConfigDict(abbr='fake-model')
    prediction_path = Path(get_infer_output_path(
        model_cfg, dataset_cfg, str(tmp_path / 'predictions')))
    prediction_path.parent.mkdir(parents=True)
    mmengine.dump({'0': {'prediction': '8'}}, prediction_path)
    task = OpenICLEvalTask(ConfigDict(
        models=[model_cfg], datasets=[[dataset_cfg]], work_dir=str(tmp_path)))

    task.run()

    result_path = get_infer_output_path(
        model_cfg, dataset_cfg, str(tmp_path / 'results'))
    result = mmengine.load(result_path)
    assert result['protocol'] == [PROTOCOL]
    assert result['Reality'] == 100
