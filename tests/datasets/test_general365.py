import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets.general365 import (
    GENERAL365_JUDGE_SYSTEM_PROMPT,
    GENERAL365_QUESTION_SUFFIX,
    General365Dataset,
    General365Evaluator,
    build_general365_judge_prompt,
    general365_pred_cut,
    parse_general365_judgement,
)
from opencompass.models import OpenAISDK
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.summarizers import DefaultSummarizer


def _row(sample_id='1', answer='2', answer_type='number', question='Q'):
    return dict(
        id=sample_id,
        question=question,
        answer=answer,
        answer_type=answer_type,
        float_round='0',
    )


def test_general365_dependency_lock_matches_official_scorer():
    lock = Path(
        'opencompass/configs/datasets/General365/requirements.lock')
    assert lock.read_text(encoding='utf-8').splitlines() == [
        'antlr4-python3-runtime==4.13.2',
        'datasets==2.18.0',
        'latex2sympy2_extended==1.10.2',
        'openai==1.99.9',
        'sympy==1.14.0',
        'math-verify==0.8.0',
    ]


def test_general365_loads_materialized_parquet_and_validates_shape(tmp_path):
    path = tmp_path / 'general365.parquet'
    Dataset.from_list([_row()]).to_parquet(path)

    loaded = General365Dataset.load(str(path))
    assert loaded.column_names == [
        'id', 'question', 'answer', 'answer_type', 'float_round'
    ]
    assert loaded[0]['id'] == '1'

    bad_path = tmp_path / 'bad.parquet'
    Dataset.from_list([{'id': '1'}]).to_parquet(bad_path)
    with pytest.raises(ValueError, match='missing required columns'):
        General365Dataset.load(str(bad_path))
    with pytest.raises(FileNotFoundError):
        General365Dataset.load(str(tmp_path / 'missing.parquet'))


def test_general365_strict_judge_parser_and_prompt_boundaries():
    assert parse_general365_judgement('{"accuracy": true}') is True
    with pytest.raises(json.JSONDecodeError):
        parse_general365_judgement('```json\n{"accuracy": true}\n```')
    with pytest.raises(KeyError):
        parse_general365_judgement('{"correct": true}')

    question = 'Keep this' + GENERAL365_QUESTION_SUFFIX + ' ignored suffix'
    assert build_general365_judge_prompt(question, 'gold', 'prediction') == (
        'Question: Keep this\nGround truth: gold\nPrediction: prediction')
    long_prompt = build_general365_judge_prompt('q' * 20000, 'a', 'p')
    assert len(long_prompt) == 20007
    assert long_prompt[10000:10007] == '\n\n...\n\n'
    assert general365_pred_cut('answer\\medskip extra\n---more') == 'answer'


def test_general365_math_aggregation_duplicate_overwrite_and_empty_state(
        tmp_path):
    evaluator = General365Evaluator(judge_cfg={})
    evaluator._out_dir = str(tmp_path / 'result')
    rows = Dataset.from_list([
        _row(sample_id='same', answer='2'),
        _row(sample_id='same', answer='3'),
    ])
    result = evaluator.score(
        predictions=['final \\boxed{2}', 'final \\boxed{0}'],
        references=['2', '3'],
        test_set=rows,
    )
    assert result['average_accuracy'] == 0
    assert result['per_question_accuracy'] == {'same': False}
    assert [item['sample_id'] for item in result['details']] == ['same',
                                                                  'same']

    with pytest.raises(ValueError, match='empty response set'):
        evaluator.score([], [], Dataset.from_list([]))


def test_general365_config_is_one_native_dataset_with_official_metric():
    config = Config.fromfile(
        'opencompass/configs/datasets/General365/'
        'general365_rawprompt_cascade_llmjudge_gen.py')
    assert len(config.general365_datasets) == 1
    dataset_cfg = config.general365_datasets[0]
    assert dataset_cfg.abbr == 'General365'
    assert dataset_cfg.type is General365Dataset
    assert dataset_cfg.revision == (
        'e6f063d0a74c51baa101204533823dd54b0c577b')
    assert dataset_cfg.infer_cfg.prompt_template.type is RawPromptTemplate
    assert dataset_cfg.infer_cfg.prompt_template.messages == [
        dict(role='user', content='{question}')
    ]
    assert dataset_cfg.eval_cfg.evaluator.type is General365Evaluator
    judge_cfg = dataset_cfg.eval_cfg.evaluator.judge_cfg
    assert judge_cfg.type is OpenAISDK
    assert judge_cfg.path == 'gpt-4.1-mini'
    assert judge_cfg.max_out_len == 8192
    assert judge_cfg.temperature == 0.0
    assert judge_cfg.retry == 1
    assert judge_cfg.openai_extra_kwargs == {
        'stream': False,
        'response_format': {
            'type': 'json_object'
        },
    }


def test_general365_summarizer_preserves_scalar_average_accuracy():
    config = Config.fromfile(
        'opencompass/configs/summarizers/groups/General365.py')
    assert config.general365_summary_groups == []

    summarizer = DefaultSummarizer.__new__(DefaultSummarizer)
    summarizer.summary_groups = config.general365_summary_groups
    summarizer.model_abbrs = ['test-model']
    raw_results = {
        'test-model': {
            'General365': {
                'average_accuracy': 0.5,
            },
        },
    }
    parsed_results = {
        'test-model': {
            'General365': {
                'average_accuracy': 0.5,
            },
        },
    }
    dataset_metrics = {'General365': ['average_accuracy']}
    dataset_eval_mode = {'General365': 'gen'}

    _, parsed_results, dataset_metrics, _ = (
        summarizer._calculate_group_metrics(
            raw_results, parsed_results, dataset_metrics, dataset_eval_mode))

    metric = parsed_results['test-model']['General365']['average_accuracy']
    assert metric == 0.5
    assert isinstance(metric, float)
    assert dataset_metrics['General365'] == ['average_accuracy']
    assert 'naive_average' not in parsed_results['test-model']['General365']


def test_general365_judge_uses_native_model_request(monkeypatch, tmp_path):
    captured = []

    class _Completions:

        def create(self, **kwargs):
            captured.append(kwargs)
            message = SimpleNamespace(
                content='{"accuracy": true}', reasoning_content=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message,
                                         finish_reason='stop')])

    class _FakeOpenAI:

        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr('openai.OpenAI', _FakeOpenAI)
    evaluator = General365Evaluator(judge_cfg=dict(
        type=OpenAISDK,
        path='fixture-model',
        key='fixture-key',
        openai_api_base='https://example.invalid/v1',
        tokenizer_path='gpt-4',
        max_seq_len=65536,
        max_out_len=8192,
        batch_size=1,
        temperature=0.0,
        retry=1,
        timeout=900,
        max_workers=1,
        query_per_second=1000,
        openai_extra_kwargs=dict(
            stream=False,
            response_format=dict(type='json_object'),
        ),
    ))
    evaluator._out_dir = str(tmp_path / 'judge-result')
    rows = Dataset.from_list([
        _row(answer='A', answer_type='single_choice', question='Choose one')
    ])

    result = evaluator.score(['A'], ['A'], rows)
    assert result['average_accuracy'] == 1
    assert len(captured) == 1
    request = captured[0]
    assert request['messages'] == [
        {
            'role': 'system',
            'content': GENERAL365_JUDGE_SYSTEM_PROMPT,
        },
        {
            'role': 'user',
            'content': 'Question: Choose one\nGround truth: A\nPrediction: A',
        },
    ]
    assert request['max_tokens'] == 8192
    assert request['temperature'] == 0.0
    assert request['stream'] is False
    assert request['response_format'] == {'type': 'json_object'}
    assert request['timeout'] == 900
