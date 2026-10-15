import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import yaml
from mmengine.config import Config

from opencompass.datasets import BackTrendDataset, BackTrendEvaluator
from opencompass.datasets.backtrend.backtrend import (
    build_prompt,
    extract_candidate_signals,
)
from opencompass.registry import LOAD_DATASET


def _write_data(tmp_path):
    topics = {
        'large language models': {'paraphrase': ['LLMs']},
        'epistemic AI': {'paraphrase': []},
    }
    signals = {
        'large language models': {
            'problem': ['hallucination'],
            'solution': ['instruction tuning'],
        },
    }
    topics_path = tmp_path / 'topics.json'
    signals_path = tmp_path / 'weak_signal.json'
    topics_path.write_text(json.dumps(topics), encoding='utf-8')
    signals_path.write_text(json.dumps(signals), encoding='utf-8')
    return topics_path, signals_path


class _ExactMatchJudge:

    def __init__(self):
        self.calls = []

    @staticmethod
    def _block(prompt, start, end):
        value = prompt.split(start, 1)[1]
        if end:
            value = value.split(end, 1)[0]
        return [line.split('. ', 1)[1]
                for line in value.strip().splitlines() if '. ' in line]

    def generate(self, inputs, max_out_len=None, temperature=None):
        self.calls.append((inputs, max_out_len, temperature))
        replies = []
        for messages in inputs:
            assert [message['role'] for message in messages] == [
                'system', 'user']
            prompt = messages[-1]['content']
            if 'matched_pairs' in prompt:
                gt = self._block(prompt, 'Ground truth:\n', '\n\nPredicted:')
                pred = self._block(prompt, 'Predicted:\n', '\n\nReturn ONLY')
                pairs = [dict(gt_index=i, pred_index=j)
                         for i, left in enumerate(gt)
                         for j, right in enumerate(pred) if left == right]
                replies.append(json.dumps(dict(
                    precision=len({x['pred_index'] for x in pairs}) / len(pred),
                    recall=len({x['gt_index'] for x in pairs}) / len(gt),
                    matched_pairs=pairs)))
            else:
                ref = self._block(prompt, 'Reference:\n', '\n\nCandidate:')
                cand = self._block(prompt, 'Candidate:\n', '\n\nReturn ONLY')
                replies.append(json.dumps(
                    {'matches': [int(value in ref) for value in cand]}))
        return replies


def test_backtrend_loader_registry_order_and_prompt_contract(tmp_path):
    topics, signals = _write_data(tmp_path)
    dataset = LOAD_DATASET.build(dict(
        type='BackTrendDataset', path='.', topics_file=str(topics),
        signals_file=str(signals), abbr='backtrend',
        reader_cfg=dict(input_columns=['prompt'], output_column='reference')))
    rows = list(dataset.test)
    assert [row['sample_id'] for row in rows] == [
        'large_language_models--problem',
        'large_language_models--solution',
        'epistemic_ai--problem',
        'epistemic_ai--solution',
    ]
    assert rows[0]['gt_signals'] == ['hallucination']
    assert rows[-1]['gt_signals'] == []
    digests = {
        direction: hashlib.sha256(build_prompt(
            direction, 'Artificial intelligence and machine learning',
            'large language models').encode()).hexdigest()
        for direction in ('problem', 'solution')
    }
    assert digests == {
        'problem': '15118b4e3b7855f90c9a5b9f5547f59bd21d3fe8cc3f42ebb7965e787f3f602e',
        'solution': '7f309c0257d6bb8c165faa891b115be4fedc93d451e0f3ca5ca6000b0b36e4f3',
    }


def test_backtrend_native_config_loads_and_evaluator_constructs():
    project_root = __import__('pathlib').Path(__file__).parents[2]
    config_path = (project_root /
                   'opencompass/configs/datasets/backtrend/backtrend_gen.py')
    cfg = Config.fromfile(config_path)
    dataset_cfg = cfg.backtrend_datasets[0]
    assert dataset_cfg.abbr == 'backtrend'
    assert dataset_cfg.infer_cfg.prompt_template.messages == [
        dict(role='user', content='{prompt}')]
    assert dataset_cfg.eval_cfg.evaluator.type == 'BackTrendEvaluator'
    evaluator = BackTrendEvaluator(**{
        key: value for key, value in dataset_cfg.eval_cfg.evaluator.items()
        if key != 'type'
    })
    assert evaluator.settings == (2, 4, 5)
    catalog = yaml.safe_load(
        (project_root / 'dataset-index.yml').read_text(encoding='utf-8'))
    entry = next(item['backtrend'] for item in catalog if 'backtrend' in item)
    assert entry['configpath'] == (
        'opencompass/configs/datasets/backtrend/backtrend_gen.py')
    assert entry['configpath_llmjudge'] == entry['configpath']


def test_backtrend_loader_rejects_missing_or_malformed_inputs(tmp_path):
    topics, signals = _write_data(tmp_path)
    with pytest.raises(FileNotFoundError, match='Missing BackTrend'):
        BackTrendDataset(path='.', topics_file=str(topics),
                         signals_file=str(tmp_path / 'missing.json'),
                         reader_cfg={})
    signals.write_text(json.dumps({'large language models': {
        'problem': 'not-a-list'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid problem references'):
        BackTrendDataset(path='.', topics_file=str(topics),
                         signals_file=str(signals), reader_cfg={})


def test_backtrend_loader_rejects_unpinned_snapshot(tmp_path):
    construction = tmp_path / 'construction'
    construction.mkdir()
    (construction / 'topics.json').write_text('{}', encoding='utf-8')
    (construction / 'weak_signal.json').write_text('{}', encoding='utf-8')

    with pytest.raises(ValueError, match='snapshot checksum mismatch'):
        BackTrendDataset(path=str(tmp_path), reader_cfg={})


def test_backtrend_prediction_parser_matches_official_fallbacks():
    assert extract_candidate_signals(
        '{"weak_signals":[{"signal":"a"},{"signal":"a"}]}') == ['a', 'a']
    assert extract_candidate_signals(
        'prefix ```json\n{"weak_signals":[{"signal":"b"}]}\n```') == ['b']
    assert extract_candidate_signals(
        'analysis {"weak_signals":[{"signal":"c"}]} trailing') == ['c']
    assert extract_candidate_signals(
        '<think>hidden</think>1. d\n2) d\n3. e') == ['d', 'e']
    assert extract_candidate_signals('<think>unclosed') == []


def test_backtrend_judge_scoring_and_aggregation_boundaries(tmp_path):
    topics, signals = _write_data(tmp_path)
    dataset = BackTrendDataset(
        path='.', topics_file=str(topics), signals_file=str(signals),
        abbr='backtrend',
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'))
    rows = list(dataset.test)
    predictions = [
        '{"weak_signals":[{"signal":"hallucination"}]}',
        '{"weak_signals":[{"signal":"instruction tuning"}]}',
        'not json',
        'not json',
    ]
    judge = _ExactMatchJudge()
    evaluator = BackTrendEvaluator(settings=(2, 4, 5), n_runs=3)
    evaluator.judge_model = judge
    result = evaluator.score(predictions=predictions, test_set=rows)
    assert result['all/set_llm/f1_mean'] == 1.0
    assert result['all/signal_llm/f1_mean'] == 1.0
    assert result['all/coverage_at_10/coverage_mean'] == 1.0
    assert result['all/set_llm/n_total'] == 4
    assert result['all/set_llm/n_scored'] == 2
    assert result['all/set_llm/n_no_gt'] == 2
    assert [detail['status'] for detail in result['details']] == [
        'scored', 'scored', 'no_gt', 'no_gt']
    assert {call[1] for call in judge.calls} == {1024, 1536}
    assert all(call[2] == 1.0 for call in judge.calls)


def test_backtrend_set_judge_uses_unique_indices(tmp_path):
    topics, signals = _write_data(tmp_path)
    dataset = BackTrendDataset(
        path='.', topics_file=str(topics), signals_file=str(signals),
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'))

    class DuplicateJudge:
        def generate(self, inputs, **kwargs):
            return [json.dumps({'precision': 1, 'recall': 1,
                                'matched_pairs': [
                                    {'gt_index': 0, 'pred_index': 0},
                                    {'gt_index': 0, 'pred_index': 0},
                                    {'gt_index': 99, 'pred_index': 0},
                                    {'gt_index': '0', 'pred_index': 0},
                                ]}) for _ in inputs]

    evaluator = BackTrendEvaluator(settings=(2,), n_runs=1)
    evaluator.judge_model = DuplicateJudge()
    result = evaluator.score(
        predictions=['1. candidate', '1. candidate', '', ''],
        test_set=list(dataset.test))
    assert result['all/set_llm/precision_mean'] == 1.0
    assert result['all/set_llm/recall_mean'] == 1.0


def test_backtrend_rejects_invalid_settings():
    with pytest.raises(ValueError, match='Unknown BackTrend settings'):
        BackTrendEvaluator(settings=(0, 2, 6))


@pytest.mark.parametrize(('setting', 'reply', 'message'), [
    (2, 'not-json', 'Could not parse set-judge JSON'),
    (2, json.dumps({'precision': 1.1, 'recall': 0,
                    'matched_pairs': []}), 'must be in \\[0, 1\\]'),
    (4, json.dumps({'matches': []}), 'Expected 1 binary matches'),
    (4, json.dumps({'matches': [2]}), 'Non-binary matches'),
])
def test_backtrend_judge_payloads_fail_closed(tmp_path, setting, reply, message):
    topics, signals = _write_data(tmp_path)
    rows = list(BackTrendDataset(
        path='.', topics_file=str(topics), signals_file=str(signals),
        reader_cfg=dict(input_columns=['prompt'], output_column='reference')).test)

    class ReplyJudge:

        def generate(self, inputs, **kwargs):
            return [reply for _ in inputs]

    evaluator = BackTrendEvaluator(settings=(setting,), n_runs=1)
    evaluator.judge_model = ReplyJudge()
    with pytest.raises(ValueError, match=message):
        evaluator.score(
            predictions=['1. candidate'], test_set=rows[:1])


def test_backtrend_empty_predictions_and_zero_scored_summaries(tmp_path):
    topics, signals = _write_data(tmp_path)
    dataset = BackTrendDataset(
        path='.', topics_file=str(topics), signals_file=str(signals),
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'))
    result = BackTrendEvaluator(settings=(2, 4, 5), n_runs=1).score(
        predictions=['', '', '', ''], test_set=list(dataset.test))
    assert result['all/set_llm/f1_mean'] == 0.0
    assert result['all/signal_llm/f1_mean'] == 0.0
    assert result['all/coverage_at_10/coverage_mean'] == 0.0
    assert result['all/set_llm/n_scored'] == 2
    assert result['details'][0]['set_llm']['note'] == 'empty_pred'

    signals.write_text('{}', encoding='utf-8')
    no_gt = BackTrendDataset(
        path='.', topics_file=str(topics), signals_file=str(signals),
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'))
    empty = BackTrendEvaluator(settings=(2, 4, 5), n_runs=1).score(
        predictions=['', '', '', ''], test_set=list(no_gt.test))
    assert empty['all/set_llm/n_scored'] == 0
    assert empty['all/set_llm/f1_mean'] is None
    assert empty['all/set_llm/f1_std'] is None
    assert 'all/coverage_at_10/coverage_mean' not in empty
    assert 'all/coverage_at_10/coverage_full_mean' not in empty


def test_backtrend_coverage_truncates_and_reuses_short_runs(tmp_path):
    topics, signals = _write_data(tmp_path)
    row = list(BackTrendDataset(
        path='.', topics_file=str(topics), signals_file=str(signals),
        reader_cfg=dict(input_columns=['prompt'], output_column='reference')).test)[:1]

    long_judge = _ExactMatchJudge()
    evaluator = BackTrendEvaluator(
        settings=(5,), n_runs=2, coverage_k=2, batch_size=5)
    evaluator.judge_model = long_judge
    long_result = evaluator.score(predictions=[json.dumps({
        'weak_signals': [
            {'signal': 'unmatched 1'},
            {'signal': 'unmatched 2'},
            {'signal': 'hallucination'},
        ]
    })], test_set=row)
    long_detail = long_result['details'][0]['coverage_at_2']
    assert len(long_judge.calls[0][0]) == 4
    assert long_detail['truncated'] is True
    assert long_detail['coverage'] == 0.0
    assert long_detail['coverage_full'] == 1.0

    short_judge = _ExactMatchJudge()
    evaluator.judge_model = short_judge
    short_result = evaluator.score(predictions=[json.dumps({
        'weak_signals': [{'signal': 'hallucination'}]
    })], test_set=row)
    short_detail = short_result['details'][0]['coverage_at_2']
    assert len(short_judge.calls[0][0]) == 2
    assert short_detail['truncated'] is False
    assert short_detail['coverage'] == short_detail['coverage_full'] == 1.0


def test_backtrend_signal_bertscore_preserves_official_counts(monkeypatch,
                                                               tmp_path):
    topics, signals = _write_data(tmp_path)
    row = list(BackTrendDataset(
        path='.', topics_file=str(topics), signals_file=str(signals),
        reader_cfg=dict(input_columns=['prompt'], output_column='reference')).test)[:1]

    class Scores:

        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    def fake_bert_score(candidates, references, **kwargs):
        assert len(candidates) == len(references) == 2
        return Scores([0.0, 0.0]), Scores([0.0, 0.0]), Scores([0.4, 0.8])

    monkeypatch.setitem(
        sys.modules, 'bert_score', SimpleNamespace(score=fake_bert_score))
    result = BackTrendEvaluator(settings=(3,)).score(
        predictions=[json.dumps({'weak_signals': [
            {'signal': 'candidate 1'}, {'signal': 'candidate 2'}
        ]})], test_set=row)
    detail = result['details'][0]['signal_bertscore']
    assert detail['n_pred'] == 2
    assert detail['n_gt'] == 1
