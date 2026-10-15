import copy
import threading
import time

import pytest
import yaml
from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets.wearableqa import (
    WEARABLEQA_CATEGORY_METRICS, WEARABLEQA_COT_DIRECTIVE,
    WEARABLEQA_DIRECT_DIRECTIVE, WearableQADataset, WearableQAEvaluator,
    build_wearableqa_prompt, wearableqa_answer_postprocess)
from opencompass.registry import (ICL_EVALUATORS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, LOAD_DATASET)
from opencompass.tasks.openicl_infer_concurrent import \
    OpenICLInferConcurrentTask
from opencompass.utils import build_dataset_from_cfg


def _rows():
    rows = []
    for index, category in enumerate(WEARABLEQA_CATEGORY_METRICS):
        rows.append({
            'id': f'sample-{index}',
            'question': f'Question {index}\nA. first\nJ. last',
            'choices': {
                letter: f'choice {letter}' for letter in 'ABCDEFGHIJ'
            },
            'answer': 'A' if index % 2 == 0 else 'B',
            'category': category,
            'reasoning_group': 'data' if index % 2 == 0 else 'health',
            'signal': 'single' if index % 2 == 0 else 'cross',
            'grounding': 'population',
            'representation': 'row',
        })
    return rows


@pytest.fixture()
def wearableqa_parquet(tmp_path):
    path = tmp_path / 'wearableqa.parquet'
    Dataset.from_list(_rows()).to_parquet(path)
    return path


def test_config_loads_and_constructs_registered_components(
        wearableqa_parquet):
    cfg = Config.fromfile(
        'opencompass/configs/datasets/wearableqa/wearableqa_cot_gen.py')
    assert len(cfg.wearableqa_datasets) == 1
    dataset_cfg = copy.deepcopy(cfg.wearableqa_datasets[0])
    dataset_cfg.path = str(wearableqa_parquet)
    dataset_cfg.validate_full = False
    assert dataset_cfg.infer_cfg.inferencer.type.__name__ == 'GenInferencer'

    dataset = build_dataset_from_cfg(dataset_cfg)
    prompt = ICL_PROMPT_TEMPLATES.build(
        dataset_cfg.infer_cfg.prompt_template)
    retriever_cfg = copy.deepcopy(dataset_cfg.infer_cfg.retriever)
    retriever_cfg.dataset = dataset
    retriever = ICL_RETRIEVERS.build(retriever_cfg)
    evaluator = ICL_EVALUATORS.build(dataset_cfg.eval_cfg.evaluator)

    assert LOAD_DATASET.get('WearableQADataset') is WearableQADataset
    assert isinstance(evaluator, WearableQAEvaluator)
    assert retriever.retrieve() == [[] for _ in dataset.test]
    rendered = prompt.generate_item(dataset.test[0])
    assert rendered[1]['prompt'] == dataset.test[0]['input']
    assert dataset.test[0]['input'].endswith(WEARABLEQA_COT_DIRECTIVE)


def test_dataset_index_exposes_native_config():
    with open('dataset-index.yml', encoding='utf-8') as stream:
        index = yaml.safe_load(stream)

    wearableqa = next(item['wearableqa'] for item in index
                      if 'wearableqa' in item)
    assert wearableqa == {
        'name': 'WearableQA',
        'category': 'Knowledge / Medicine',
        'paper': 'https://arxiv.org/abs/2609.05405',
        'configpath':
        'opencompass/configs/datasets/wearableqa/wearableqa_cot_gen.py',
        'configpath_llmjudge': '',
    }


def test_loader_rejects_missing_and_malformed_data(tmp_path):
    with pytest.raises(FileNotFoundError, match='data file not found'):
        WearableQADataset.load(str(tmp_path / 'missing.parquet'))

    malformed = _rows()
    malformed[0]['choices'].pop('J')
    path = tmp_path / 'malformed.parquet'
    Dataset.from_list(malformed).to_parquet(path)
    with pytest.raises(ValueError, match='choices A through J'):
        WearableQADataset.load(str(path), validate_full=False)


def test_prompt_conditions_share_the_released_body():
    body = 'released body ending with J. final option'
    cot = build_wearableqa_prompt(body, 'cot')
    direct = build_wearableqa_prompt(body, 'direct')
    assert cot.removesuffix(WEARABLEQA_COT_DIRECTIVE) == body + '\n\n'
    assert direct.removesuffix(WEARABLEQA_DIRECT_DIRECTIVE) == body + '\n\n'
    with pytest.raises(ValueError, match='prompt_mode'):
        build_wearableqa_prompt(body, 'unsupported')


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [
        ('reasoning\nAnswer: (c)', 'C'),
        ('Answer: B\nA', 'B'),
        ('reasoning\nd', 'D'),
        (' e. leading choice', 'E'),
        ('reasoning mentions A but has no answer', ''),
        ('Answer: Apple', ''),
        ('Answer: (Banana)', ''),
        ('Answer: (C', ''),
        ('Answer: D)', ''),
        ('Answer: (E))', ''),
        ('', ''),
        (None, ''),
    ],
)
def test_answer_parser_precedence(prediction, expected):
    assert wearableqa_answer_postprocess(prediction) == expected


def test_evaluator_emits_all_metrics_details_and_wilson_intervals(
        wearableqa_parquet):
    test_set = WearableQADataset.load(
        str(wearableqa_parquet), validate_full=False)['test']
    references = test_set['target']
    predictions = [f'analysis\nAnswer: {answer}' for answer in references]
    result = WearableQAEvaluator().score(predictions, references, test_set)

    expected_metrics = {
        'Avg', 'Data', 'Health', 'Single', 'Cross',
        *WEARABLEQA_CATEGORY_METRICS.values()
    }
    assert expected_metrics.issubset(result)
    assert all(result[key] == 100.0 for key in expected_metrics)
    assert len(result['details']) == len(test_set)
    assert all(detail['correct'] for detail in result['details'])
    assert 0 <= result['Avg 95% CI low'] <= result['Avg 95% CI high'] <= 100


def test_evaluator_surfaces_zero_denominators():
    test_set = Dataset.from_list([_rows()[0]])
    with pytest.raises(ValueError, match='zero denominator'):
        WearableQAEvaluator().score(['Answer: A'], ['A'], test_set)


class _ConcurrentModel:
    is_api = True
    generation_kwargs = {}

    def __init__(self, capacity):
        self.max_workers = capacity
        self.capacity = capacity
        self.active = 0
        self.peak = 0
        self.started = 0
        self.lock = threading.Lock()
        self.release_first = threading.Event()
        self.replacement_started = threading.Event()

    def parse_template(self, entry, mode):
        return entry

    def generate(self, inputs, max_out_len=None):
        return self.generate_from_template(inputs, max_out_len=max_out_len)

    def generate_from_template(self, inputs, max_out_len=None, **kwargs):
        text = str(inputs[0])
        index = int(text.split('Question ')[1].splitlines()[0])
        with self.lock:
            self.active += 1
            self.started += 1
            self.peak = max(self.peak, self.active)
            if self.started > self.capacity:
                self.replacement_started.set()
        if index == 0:
            assert self.release_first.wait(5)
        else:
            time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return ['Answer: A']


@pytest.mark.parametrize('capacity', [2, 4])
def test_configured_native_scheduler_refills_free_slots(
        wearableqa_parquet, tmp_path, capacity):
    cfg = Config.fromfile(
        'opencompass/configs/datasets/wearableqa/wearableqa_cot_gen.py')
    dataset_cfg = copy.deepcopy(cfg.wearableqa_datasets[0])
    dataset_cfg.path = str(wearableqa_parquet)
    dataset_cfg.validate_full = False
    dataset = build_dataset_from_cfg(dataset_cfg)
    retriever_cfg = copy.deepcopy(dataset_cfg.infer_cfg.retriever)
    retriever_cfg.dataset = dataset
    retriever = ICL_RETRIEVERS.build(retriever_cfg)
    prompt = ICL_PROMPT_TEMPLATES.build(
        dataset_cfg.infer_cfg.prompt_template)
    model = _ConcurrentModel(capacity)
    model_cfg = dict(
        abbr='test-model',
        max_out_len=8,
        max_workers=capacity,
        run_cfg=dict(num_gpus=0, num_procs=1),
    )
    task = OpenICLInferConcurrentTask(
        dict(
            work_dir=str(tmp_path),
            models=[model_cfg],
            datasets=[[dataset_cfg]],
        ))
    inferencer = task._build_inferencer(model, model_cfg, dataset_cfg,
                                        capacity)
    assert type(inferencer).__name__ == 'ParallelGenInferencer'
    inferencer.output_json_filepath = str(tmp_path / f'capacity-{capacity}')
    inferencer.output_json_filename = 'predictions.json'

    thread = threading.Thread(
        target=inferencer.inference,
        kwargs={
            'retriever': retriever,
            'prompt_template': prompt,
        },
    )
    thread.start()
    try:
        assert model.replacement_started.wait(5)
        assert model.peak == capacity
    finally:
        model.release_first.set()
        thread.join(5)
    assert not thread.is_alive()
