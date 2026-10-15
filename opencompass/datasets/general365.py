import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

# The scorer behavior in this module is adapted from grading.py and utils.py
# at https://github.com/meituan-longcat/General365/tree/6263ee4b995b205aa07f4c3ca5cbeb8e0e5acb43.
# Copyright (c) 2026 Meituan, licensed under the MIT License. The complete
# third-party notice is retained in the General365 configuration README.

from datasets import Dataset, load_dataset

from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


GENERAL365_REVISION = 'e6f063d0a74c51baa101204533823dd54b0c577b'
GENERAL365_TEXT_TYPES = frozenset(
    ('text', 'single_choice', 'multiple_choice'))
GENERAL365_QUESTION_SUFFIX = (
    '\nOutput your final answer at the end of your reply using the following '
    'format:')
GENERAL365_JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for question answering systems.
Your task is to determine if a prediction correctly answers a question based on the ground truth.

Rules:
1. The prediction is correct if it captures all the key information from the ground truth.
2. The prediction is correct even if phrased differently as long as the meaning is the same.
3. The prediction is incorrect if it contains incorrect information or is missing essential details.
4. Do not challenge the correctness of the standard answer.
5. If the standard answer includes multiple possibilities, the prediction must include all and only those possibilities to be considered correct.
Output a JSON object with a single field 'accuracy' whose value is true or false.
"""


@LOAD_DATASET.register_module()
class General365Dataset(BaseDataset):
    """Load the official General365 public test split.

    ``path`` may be the public Hugging Face dataset id or a materialized
    Parquet file. The latter keeps smoke tests and offline runs on the same
    loader and row shape as normal OpenCompass execution.
    """

    @staticmethod
    def load(path: str = 'meituan-longcat/General365_Public',
             split: str = 'test',
             revision: Optional[str] = GENERAL365_REVISION,
             sample_ids: Optional[List[str]] = None,
             **kwargs) -> Dataset:
        source = Path(path).expanduser()
        if source.suffix.lower() == '.parquet' and not source.is_file():
            raise FileNotFoundError(
                f'General365 Parquet input does not exist: {path}')
        if source.is_file():
            if source.suffix.lower() != '.parquet':
                raise ValueError('General365 materialized input must be a '
                                 'Parquet file')
            dataset = load_dataset(
                'parquet', data_files={split: str(source)}, split=split,
                **kwargs)
        else:
            dataset = load_dataset(
                path=path, split=split, revision=revision, **kwargs)

        required = {'id', 'question', 'answer', 'answer_type', 'float_round'}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError('General365 input is missing required columns: '
                             + ', '.join(sorted(missing)))
        if sample_ids is not None:
            positions = {str(row_id): i
                         for i, row_id in enumerate(dataset['id'])}
            missing_ids = [sample_id for sample_id in sample_ids
                           if sample_id not in positions]
            if missing_ids:
                raise KeyError('Unknown General365 sample ids: '
                               + ', '.join(missing_ids))
            dataset = dataset.select([positions[sample_id]
                                      for sample_id in sample_ids])
        return dataset


def parse_general365_judgement(judgement: str):
    """Apply the official scorer's strict JSON/key access semantics."""
    return json.loads(judgement)['accuracy']


def build_general365_judge_prompt(question: str, answer: str,
                                  prediction: str) -> str:
    """Build and character-truncate the official judge user message."""
    question = question.split(GENERAL365_QUESTION_SUFFIX)[0].strip()
    prompt = (f'Question: {question}\nGround truth: {answer}\n'
              f'Prediction: {prediction}')
    if len(prompt) >= 20000:
        prompt = prompt[:10000] + '\n\n...\n\n' + prompt[-10000:]
    return prompt


def general365_pred_cut(prediction: str) -> str:
    """Apply both official fallback delimiters, in their declared order."""
    for delimiter in ('\\medskip', '\n---'):
        prediction = prediction.split(delimiter)[0].strip()
    return prediction


def _general365_grading_math(prediction: str, info: Dict):
    """Execute the released Math-Verify comparison exactly."""
    from math_verify import parse, verify

    pred_parse = parse(prediction)
    gold_parse = parse(info['answer'])
    float_round = int(info['float_round'])

    if float_round > 0:
        verify_result = (
            verify(gold_parse, pred_parse, float_rounding=float_round)
            or verify(pred_parse, gold_parse, float_rounding=float_round))
    else:
        verify_result = (verify(gold_parse, pred_parse)
                         or verify(pred_parse, gold_parse))

    if verify_result is True:
        return verify_result

    if pred_parse and '=' in pred_parse[-1]:
        pred_last_str = pred_parse[-1].split('=')[-1]
        pred_last_str_parse = parse('\\boxed{' + pred_last_str + '}')
        if float_round > 0:
            verify_last_result = (
                verify(gold_parse, pred_last_str_parse,
                       float_rounding=float_round)
                or verify(pred_last_str_parse, gold_parse,
                          float_rounding=float_round))
        else:
            verify_last_result = (verify(gold_parse, pred_last_str_parse)
                                  or verify(pred_last_str_parse, gold_parse))
        verify_result = verify_result or verify_last_result
    return verify_result


def general365_grading_math(prediction: str, info: Dict):
    """Run the initial comparison and the source's cut-and-retry fallback."""
    result = _general365_grading_math(prediction, info)
    if result is True:
        return result
    try:
        return _general365_grading_math(
            general365_pred_cut(prediction), info)
    except Exception:
        return False


@ICL_EVALUATORS.register_module()
class General365Evaluator(BaseEvaluator):
    """Official hybrid General365 evaluator for one ordered response set."""

    def __init__(self, judge_cfg: Dict) -> None:
        super().__init__()
        self.judge_cfg = judge_cfg

    def evaluate(self, k, n, original_dataset: Dataset, **score_kwargs):
        if n != 1:
            raise ValueError('General365 official scoring does not define '
                             'replicated samples')
        return self.score(**score_kwargs)

    def _judge_text(self, predictions: List[str], references: List[str],
                    test_set: Dataset) -> List:
        prompts = [
            build_general365_judge_prompt(row['question'], row['answer'], pred)
            for row, pred in zip(test_set, predictions)
        ]
        judge_set = Dataset.from_dict({'judge_prompt': prompts})
        evaluator = GenericLLMEvaluator(
            prompt_template=dict(
                type=RawPromptTemplate,
                messages=[
                    dict(role='system',
                         content=GENERAL365_JUDGE_SYSTEM_PROMPT),
                    dict(role='user', content='{judge_prompt}'),
                ],
            ),
            judge_cfg=copy.deepcopy(self.judge_cfg),
            dataset_cfg={},
        )
        evaluator._out_dir = self._out_dir + '_judge'
        evaluator._dataset_replica_idx = self._dataset_replica_idx
        output = evaluator.score(predictions, references, test_set=judge_set)
        return [
            parse_general365_judgement(output[str(i)]['prediction'])
            for i in range(len(predictions))
        ]

    def score(self,
              predictions: List[str],
              references: Optional[List[str]] = None,
              test_set: Optional[Dataset] = None) -> Dict:
        if test_set is None:
            raise ValueError('General365 scoring requires the original rows')
        if len(predictions) != len(test_set):
            raise ValueError('predictions and test_set must have equal length')
        if not predictions:
            raise ValueError('General365 cannot score an empty response set')

        text_indices = [
            i for i, row in enumerate(test_set)
            if row['answer_type'] in GENERAL365_TEXT_TYPES
        ]
        text_scores = self._judge_text(
            [predictions[i] for i in text_indices],
            [test_set[i]['answer'] for i in text_indices],
            test_set.select(text_indices),
        ) if text_indices else []
        text_by_index = dict(zip(text_indices, text_scores))

        question_id2acc = {}
        details = []
        for index, (prediction, info) in enumerate(zip(predictions, test_set)):
            if index in text_by_index:
                accuracy = text_by_index[index]
                method = 'judge'
            else:
                accuracy = general365_grading_math(prediction, info)
                method = 'math_verify'
            question_id = info['id']
            question_id2acc[question_id] = accuracy
            details.append({
                'sample_id': question_id,
                'prediction': prediction,
                'answer': info['answer'],
                'answer_type': info['answer_type'],
                'evaluation_method': method,
                'correct': accuracy,
            })

        average_accuracy = (
            sum(question_id2acc.values()) / len(question_id2acc))
        return {
            'average_accuracy': average_accuracy,
            'per_question_accuracy': question_id2acc,
            'details': details,
        }
