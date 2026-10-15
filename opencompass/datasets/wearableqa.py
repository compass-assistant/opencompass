import math
import os.path as osp
import re
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

from .base import BaseDataset


WEARABLEQA_REPO = 'facebook/WearableQA'
WEARABLEQA_REVISION = '38ddf7fd0efbe9ce6da66e75b560a33c50dc6721'
WEARABLEQA_LETTERS = tuple('ABCDEFGHIJ')
WEARABLEQA_COT_DIRECTIVE = (
    'Think step by step. Show your reasoning, then on the LAST line write '
    'ONLY: Answer: X where X is the single letter of your answer.')
WEARABLEQA_DIRECT_DIRECTIVE = (
    'Choose the single best answer. Respond with ONLY the letter (A–J). Do '
    'not explain or output anything else. Answer:')

WEARABLEQA_CATEGORY_METRICS = {
    'signed_correlation': 'Signed Corr.',
    'signal_summary': 'Signal Summary',
    'strongest_pair': 'Strongest Pair',
    'strongest_triplet': 'Strongest Triplet',
    'trend_shape': 'Trend Shape',
    'anomaly_episodes': 'Anomaly Episodes',
    'excursion_count': 'Excursion Count',
    'recovery_time': 'Recovery Time',
    'cross_signal_prediction': 'Cross-Signal Pred.',
    'cohort_relative_risk': 'Risk Assess.',
    'signal_concordance': 'Signal Concord.',
    'multimodal_phenotype': 'Multimodal Phenotype',
    'fitness_prediction': 'Fitness Pred.',
    'differential_diagnosis': 'Differential Diag.',
    'prognostic_prediction': 'Prognostic Pred.',
    'health_recommendation': 'Health Rec.',
}


def build_wearableqa_prompt(question, prompt_mode='cot'):
    """Append the paper's condition-specific directive to a released item."""
    directives = {
        'cot': WEARABLEQA_COT_DIRECTIVE,
        'direct': WEARABLEQA_DIRECT_DIRECTIVE,
    }
    if prompt_mode not in directives:
        raise ValueError("prompt_mode must be either 'cot' or 'direct'")
    return f'{question}\n\n{directives[prompt_mode]}'


def _validate_rows(source, validate_full):
    required = {
        'id', 'question', 'choices', 'answer', 'category', 'reasoning_group',
        'signal', 'grounding', 'representation'
    }
    missing = required.difference(source.column_names)
    if missing:
        raise ValueError(f'WearableQA is missing columns: {sorted(missing)}')

    ids = source['id']
    if len(set(ids)) != len(ids):
        raise ValueError('WearableQA sample IDs must be unique')
    for index, item in enumerate(source):
        choices = item['choices']
        if not isinstance(choices, dict) or set(choices) != set(
                WEARABLEQA_LETTERS) or any(
                    not isinstance(choices[letter], str)
                    for letter in WEARABLEQA_LETTERS):
            raise ValueError(
                f'WearableQA row {index} must contain choices A through J')
        if item['answer'] not in WEARABLEQA_LETTERS:
            raise ValueError(f'WearableQA row {index} has an invalid answer')
        if item['representation'] != 'row':
            raise ValueError(
                f'WearableQA row {index} is not the row representation')

    if not validate_full:
        return
    expected = {
        'reasoning_group': {
            'data': 2724,
            'health': 1360
        },
        'signal': {
            'single': 1682,
            'cross': 2402
        },
        'grounding': {
            'population': 3154,
            'literature': 930
        },
        'representation': {
            'row': 4084
        },
    }
    if len(source) != 4084:
        raise ValueError(
            f'WearableQA row test split must contain 4084 rows, got '
            f'{len(source)}')
    for column, expected_counts in expected.items():
        actual = dict(Counter(source[column]))
        if actual != expected_counts:
            raise ValueError(
                f'WearableQA {column} totals differ: {actual!r}')
    categories = set(source['category'])
    if categories != set(WEARABLEQA_CATEGORY_METRICS):
        raise ValueError('WearableQA category coverage differs from the '
                         'published 16-category taxonomy')


@LOAD_DATASET.register_module()
class WearableQADataset(BaseDataset):
    """The official row-serialized WearableQA test set."""

    @staticmethod
    def load(path=WEARABLEQA_REPO,
             split='test',
             prompt_mode='cot',
             validate_full=True,
             sample_indices=None,
             **kwargs):
        if osp.isabs(path) and not osp.exists(path):
            raise FileNotFoundError(f'WearableQA data file not found: {path}')
        if path != WEARABLEQA_REPO or osp.exists(path):
            path = get_data_path(path)
        if osp.isfile(path):
            source = Dataset.from_parquet(path)
        elif osp.isdir(path):
            filename = osp.join(path, 'test-00000-of-00001.parquet')
            if not osp.isfile(filename):
                filename = osp.join(path, 'data', 'row',
                                    'test-00000-of-00001.parquet')
            if not osp.isfile(filename):
                raise FileNotFoundError(
                    'WearableQA requires the row test Parquet shard at '
                    f'{filename}')
            source = Dataset.from_parquet(filename)
        else:
            source = load_dataset(path,
                                  'row',
                                  revision=WEARABLEQA_REVISION,
                                  split=split)

        _validate_rows(source, validate_full)
        if sample_indices is not None:
            source = source.select(sample_indices)

        rows = []
        for item in source:
            choices = item['choices']
            rows.append({
                'id': item['id'],
                'input': build_wearableqa_prompt(item['question'],
                                                 prompt_mode),
                **{letter: choices[letter] for letter in WEARABLEQA_LETTERS},
                'target': item['answer'],
                'category': item['category'],
                'reasoning_group': item['reasoning_group'],
                'signal': item['signal'],
                'grounding': item['grounding'],
                'representation': item['representation'],
            })
        return DatasetDict({split: Dataset.from_list(rows)})


@TEXT_POSTPROCESSORS.register_module()
def wearableqa_answer_postprocess(prediction):
    """Apply the answer parser order stated in the paper appendix."""
    if not isinstance(prediction, str):
        return ''
    answer = re.search(
        r'(?i)\banswer\s*:\s*'
        r'(?:\(\s*([A-J])\s*\)|([A-J]))'
        r'(?!\w)(?![^\S\r\n]*[()])', prediction)
    if answer:
        return (answer.group(1) or answer.group(2)).upper()
    lines = prediction.splitlines()
    if lines:
        final_line = re.fullmatch(r'\s*([A-J])\s*', lines[-1], re.I)
        if final_line:
            return final_line.group(1).upper()
    leading = re.match(r'\s*([A-J])\b', prediction, re.I)
    return leading.group(1).upper() if leading else ''


def _wilson_interval(correct, total):
    if total <= 0:
        raise ValueError('cannot calculate a Wilson interval for zero items')
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (z / denominator * math.sqrt(
        proportion * (1 - proportion) / total + z * z /
        (4 * total * total)))
    # Floating-point roundoff can otherwise put an endpoint just outside the
    # probability range (for example, 100.00000000000003 at 100% accuracy).
    return [
        max(0.0, 100 * (centre - radius)),
        min(100.0, 100 * (centre + radius)),
    ]


@ICL_EVALUATORS.register_module()
class WearableQAEvaluator(BaseEvaluator):
    """Exact-match accuracies for the published WearableQA taxonomies."""

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references) or len(predictions) != len(
                test_set):
            raise ValueError('predictions, references, and test rows must '
                             'have equal lengths')
        if not predictions:
            raise ValueError('WearableQA has no scorable test items')

        parsed = [wearableqa_answer_postprocess(pred) for pred in predictions]
        correct = [pred == ref for pred, ref in zip(parsed, references)]
        groups = {
            'Avg': list(range(len(test_set))),
            'Data': [
                i for i, value in enumerate(test_set['reasoning_group'])
                if value == 'data'
            ],
            'Health': [
                i for i, value in enumerate(test_set['reasoning_group'])
                if value == 'health'
            ],
            'Single': [
                i for i, value in enumerate(test_set['signal'])
                if value == 'single'
            ],
            'Cross': [
                i for i, value in enumerate(test_set['signal'])
                if value == 'cross'
            ],
        }
        for category, metric in WEARABLEQA_CATEGORY_METRICS.items():
            groups[metric] = [
                i for i, value in enumerate(test_set['category'])
                if value == category
            ]

        result = {}
        for metric, indices in groups.items():
            if not indices:
                raise ValueError(
                    f'WearableQA metric {metric!r} has a zero denominator')
            num_correct = sum(correct[i] for i in indices)
            result[metric] = 100 * num_correct / len(indices)
            lower, upper = _wilson_interval(num_correct, len(indices))
            result[f'{metric} 95% CI low'] = lower
            result[f'{metric} 95% CI high'] = upper
        result['details'] = [{
            'id': test_set[i]['id'],
            'pred': parsed[i],
            'answer': references[i],
            'correct': correct[i],
        } for i in range(len(test_set))]
        return result
