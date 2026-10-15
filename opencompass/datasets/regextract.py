# SPDX-License-Identifier: Apache-2.0
"""RegExtract dataset and executable-source evaluator.

The benchmark data and scoring protocol are published at
https://github.com/IBM/PFAS-regulations.  The scorer intentionally keeps
the metric spelling and greedy matching behaviour of the published code.
"""

import ast
import json
import re
from pathlib import Path

from datasets import Dataset
from rouge_score import rouge_scorer

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ['rougeL', 'rougeLsum'], use_stemmer=True)
_REGULATION_FIELDS = ('type', 'date', 'who', 'products')
_METADATA_FIELDS = (
    'title', 'location', 'reference', 'status', 'date', 'pfas_definition')
_JOINT_PROMPT_NAME = 'IE_prompt_schema_design_joint_extraction'
_REGEXTRACT_SAMPLE_IDS = tuple(
    f'document-{index:02d}' for index in range(1, 21))
_REGEXTRACT_METRIC_KEYS = tuple(
    key
    for field in _METADATA_FIELDS
    for key in (f'{field}_value', f'{field}_provenance')
) + (
    'regulations_match_precision',
    'regulations_match_recall',
    'regulations_match_f1',
    'regulations_value_avg_score',
    'regulations_provenance_avg_score',
) + tuple(
    key
    for field in _REGULATION_FIELDS
    for key in (f'regulations_{field}_value_avg_score',
                f'regulations_{field}_provenance_avg_score')
) + (
    'num_pred_regulations',
    'valid_prediction_count',
    'invalid_prediction_count',
)


def _load_joint_prompt(path):
    """Read the pinned prompt as a literal without executing source code."""
    prompt_path = Path(path)
    try:
        tree = ast.parse(prompt_path.read_text(encoding='utf-8'),
                         filename=str(prompt_path))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'RegExtract requires its published prompts.py: {prompt_path}') \
            from error
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name)
               and target.id == _JOINT_PROMPT_NAME
               for target in node.targets):
            prompt = ast.literal_eval(node.value)
            if not isinstance(prompt, str) or not prompt:
                break
            return prompt
    raise ValueError(
        f'{_JOINT_PROMPT_NAME} is missing from {prompt_path}')


def parse_regextract_prediction(text):
    """Apply the official greedy JSON extraction to generated text."""
    match = re.search(r'\{[\s\S]*\}', text)
    if match is None:
        raise ValueError('No JSON object found in generated text.')
    prediction = json.loads(match.group(0))
    if not isinstance(prediction, dict):
        raise ValueError('The extracted RegExtract prediction is not an object.')
    return prediction


@LOAD_DATASET.register_module()
class RegExtractDataset(BaseDataset):
    """Load paired ``document-NN.txt`` and ``document-NN.json`` files."""

    @staticmethod
    def load(path=None,
             manifest_path=None,
             prompt_path=None,
             sample_ids=None,
             **kwargs):
        if manifest_path is not None:
            return RegExtractDataset._load_manifest(
                manifest_path, prompt_path, sample_ids)
        if path is None:
            raise ValueError('RegExtract requires path or manifest_path')
        root = Path(get_data_path(path)).expanduser()
        text_dir = root / 'golden_dataset_txts'
        gold_dir = root / 'golden_dataset_json'
        if not text_dir.is_dir() or not gold_dir.is_dir():
            raise FileNotFoundError(
                'RegExtract expects golden_dataset_txts/ and '
                f'golden_dataset_json/ below {root}')

        text_files = sorted(text_dir.glob('document-*.txt'))
        if not text_files:
            raise FileNotFoundError(
                f'No document-*.txt files found below {text_dir}')

        prompt = _load_joint_prompt(prompt_path or root / 'prompts.py')
        selected = set(sample_ids) if sample_ids is not None else None
        rows = []
        for text_path in text_files:
            sample_id = text_path.stem
            if selected is not None and sample_id not in selected:
                continue
            gold_path = gold_dir / f'{sample_id}.json'
            if not gold_path.is_file():
                raise FileNotFoundError(
                    f'Missing RegExtract reference for {sample_id}: {gold_path}')
            reference = json.loads(gold_path.read_text(encoding='utf-8'))
            if not isinstance(reference, dict):
                raise ValueError(
                    f'RegExtract reference {gold_path} must be a JSON object')
            document = text_path.read_text(encoding='utf-8')
            rows.append(RegExtractDataset._row(
                sample_id, document, reference, prompt))
        RegExtractDataset._validate_selection(rows, sample_ids)
        return Dataset.from_list(rows)

    @staticmethod
    def _row(sample_id, document, reference, prompt):
        return {
            'id': sample_id,
            'document': document,
            'prompt': f'{prompt}\n\nDocument:\n{document}',
            # The published annotations are heterogeneous JSON.  For example,
            # exceptions.value is usually a list but is a boolean in one
            # document.  Arrow has no union type for such nested values, so
            # preserve the document losslessly as JSON and decode it at the
            # evaluator boundary.
            'reference': json.dumps(reference, ensure_ascii=False),
        }

    @staticmethod
    def _validate_selection(rows, sample_ids):
        expected = list(_REGEXTRACT_SAMPLE_IDS if sample_ids is None
                        else sample_ids)
        found = [row['id'] for row in rows]
        if found != expected:
            if sample_ids is None:
                raise ValueError(
                    'RegExtract full load requires exactly document-01 through '
                    f'document-20; expected {expected}, loaded {found}')
            raise ValueError(
                'RegExtract sample_ids must be unique and in dataset order; '
                f'expected {expected}, loaded {found}')

    @staticmethod
    def _load_manifest(manifest_path, prompt_path, sample_ids):
        if prompt_path is None:
            raise ValueError('prompt_path is required with manifest_path')
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        items = manifest.get('items')
        if not isinstance(items, list):
            raise ValueError('RegExtract dataset manifest must contain items')
        inputs = {}
        references = {}
        for item in items:
            item_id = item.get('id', '')
            item_path = Path(item.get('path', ''))
            if not item_path.is_absolute():
                item_path = manifest_file.parent / item_path
            match = re.fullmatch(r'regextract_(input|gold)_(\d{2})', item_id)
            if match is None:
                continue
            sample_id = f'document-{match.group(2)}'
            target = inputs if match.group(1) == 'input' else references
            if sample_id in target:
                raise ValueError(f'Duplicate manifest item for {sample_id}')
            target[sample_id] = item_path
        if set(inputs) != set(references) or not inputs:
            raise ValueError(
                'RegExtract manifest must contain paired input and gold items')
        prompt = _load_joint_prompt(prompt_path)
        selected = set(sample_ids) if sample_ids is not None else None
        rows = []
        for sample_id in sorted(inputs):
            if selected is not None and sample_id not in selected:
                continue
            reference = json.loads(
                references[sample_id].read_text(encoding='utf-8'))
            if not isinstance(reference, dict):
                raise ValueError(
                    f'RegExtract reference for {sample_id} must be an object')
            document = inputs[sample_id].read_text(encoding='utf-8')
            rows.append(RegExtractDataset._row(
                sample_id, document, reference, prompt))
        RegExtractDataset._validate_selection(rows, sample_ids)
        return Dataset.from_list(rows)


def _list_f1(gold, prediction):
    gold_set = {item.lower() if isinstance(item, str) else item
                for item in gold}
    prediction_set = {
        item.lower() if isinstance(item, str) else item
        for item in prediction
    }
    if not gold_set and not prediction_set:
        return 1.0
    true_positive = len(gold_set & prediction_set)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(prediction_set)
    recall = true_positive / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _rouge_similarity(gold, prediction):
    if gold is None and prediction is None:
        return 1.0
    if gold is None or prediction is None:
        return 0.0
    return _ROUGE_SCORER.score(str(gold),
                               str(prediction))['rougeLsum'].fmeasure


def _compare_value(field, gold, prediction):
    if field in ('currently_unavoidable', 'intentionally_added'):
        return float(gold == prediction)
    if field in ('who', 'action', 'products', 'exceptions'):
        return _list_f1(gold or [], prediction or [])
    if field == 'title':
        return _rouge_similarity(gold, prediction)
    if isinstance(gold, str) or isinstance(prediction, str):
        gold = gold.lower() if isinstance(gold, str) else gold
        prediction = (prediction.lower()
                      if isinstance(prediction, str) else prediction)
    return float(gold == prediction)


def _compare_regulation(gold, prediction):
    value_scores = {}
    provenance_scores = {}
    for field in _REGULATION_FIELDS:
        prediction_field = ('reg_type' if field == 'type'
                            and 'type' not in prediction else field)
        gold_item = gold.get(field)
        gold_value = (gold_item.get('value')
                      if isinstance(gold_item, dict) else gold_item)
        prediction_item = prediction.get(prediction_field)
        if not isinstance(prediction_item, dict):
            value_scores[field] = _compare_value(
                field, gold_value, prediction_item)
            provenance_scores[field] = 0.0
            continue
        value_scores[field] = _compare_value(
            field, gold_value, prediction_item.get('value'))
        gold_provenance = (gold_item.get('provenance')
                           if isinstance(gold_item, dict) else None)
        provenance_scores[field] = _rouge_similarity(
            gold_provenance, prediction_item.get('provenance'))
    return value_scores, provenance_scores


def score_regextract_document(gold, prediction):
    """Score one parsed prediction with the repaired official scorer."""
    scores = {}
    for field in _METADATA_FIELDS:
        gold_item = gold.get(field)
        prediction_item = prediction.get(field)
        gold_value = (gold_item.get('value')
                      if isinstance(gold_item, dict) else gold_item)
        prediction_value = (prediction_item.get('value')
                            if isinstance(prediction_item, dict)
                            else prediction_item)
        # Source erratum: the official expression raises on null/missing
        # values.  Only strings participate in its "mpart" normalization.
        if (field == 'pfas_definition'
                and isinstance(prediction_value, str)
                and 'mpart' in prediction_value):
            prediction_value = 'single_fluorinated_carbon'
        scores[f'{field}_value'] = _compare_value(
            field, gold_value, prediction_value)
        if not isinstance(prediction_item, dict):
            scores[f'{field}_provenance'] = 0.0
        else:
            gold_provenance = (gold_item.get('provenance')
                               if isinstance(gold_item, dict) else None)
            scores[f'{field}_provenance'] = _rouge_similarity(
                gold_provenance, prediction_item.get('provenance'))

    gold_regulations = gold.get('regulations', [])
    predicted_regulations = prediction.get('regulations', [])
    matched = set()
    field_value_scores = {field: [] for field in _REGULATION_FIELDS}
    field_provenance_scores = {field: [] for field in _REGULATION_FIELDS}
    regulation_value_scores = []
    regulation_provenance_scores = []
    for gold_regulation in gold_regulations:
        best_score = 0.0
        best_provenance_score = 0.0
        best_index = -1
        best_values = {field: 0.0 for field in _REGULATION_FIELDS}
        best_provenances = {field: 0.0 for field in _REGULATION_FIELDS}
        for index, predicted_regulation in enumerate(predicted_regulations):
            if index in matched:
                continue
            values, provenances = _compare_regulation(
                gold_regulation, predicted_regulation)
            value_score = sum(values.values()) / len(values)
            if value_score > best_score:
                best_score = value_score
                best_provenance_score = (
                    sum(provenances.values()) / len(provenances))
                best_index = index
                best_values = values
                best_provenances = provenances
        if best_index >= 0:
            matched.add(best_index)
        regulation_value_scores.append(best_score)
        regulation_provenance_scores.append(best_provenance_score)
        for field in _REGULATION_FIELDS:
            field_value_scores[field].append(best_values[field])
            field_provenance_scores[field].append(best_provenances[field])

    matched_count = len(matched)
    gold_count = len(gold_regulations)
    prediction_count = len(predicted_regulations)
    precision = matched_count / prediction_count if prediction_count else 0.0
    recall = matched_count / gold_count if gold_count else 0.0
    scores['regulations_match_precision'] = precision
    scores['regulations_match_recall'] = recall
    scores['regulations_match_f1'] = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0)
    scores['regulations_value_avg_score'] = (
        sum(regulation_value_scores) / gold_count if gold_count else 1.0)
    scores['regulations_provenance_avg_score'] = (
        sum(regulation_provenance_scores) / gold_count
        if gold_count else 1.0)
    for field in _REGULATION_FIELDS:
        output_field = 'products' if field == 'products' else field
        scores[f'regulations_{output_field}_value_avg_score'] = (
            sum(field_value_scores[field]) / len(field_value_scores[field])
            if gold_count else 1.0)
        scores[f'regulations_{output_field}_provenance_avg_score'] = (
            sum(field_provenance_scores[field]) /
            len(field_provenance_scores[field]) if gold_count else 1.0)
    scores['num_pred_regulations'] = prediction_count
    return scores


@ICL_EVALUATORS.register_module()
class RegExtractEvaluator(BaseEvaluator):
    """Aggregate official per-document metrics over parse-valid outputs."""

    metric_keys = _REGEXTRACT_METRIC_KEYS

    def score(self, predictions, references, test_set=None):
        if len(predictions) != len(references):
            raise ValueError('predictions and references have different lengths')
        document_scores = []
        details = []
        for index, (prediction, reference) in enumerate(
                zip(predictions, references)):
            if isinstance(reference, str):
                reference = json.loads(reference)
            if not isinstance(reference, dict):
                raise ValueError('RegExtract reference must be a JSON object')
            sample_id = (test_set['id'][index]
                         if test_set is not None and 'id' in test_set
                         else str(index))
            try:
                parsed = parse_regextract_prediction(prediction)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                details.append({
                    'sample_id': sample_id,
                    'valid': False,
                    'error': str(error),
                })
                continue
            sample_scores = score_regextract_document(reference, parsed)
            document_scores.append(sample_scores)
            details.append({
                'sample_id': sample_id,
                'valid': True,
                **sample_scores,
            })

        if not document_scores:
            raise ValueError('RegExtract is unscorable: zero valid documents')
        result = {
            key: sum(item[key] for item in document_scores) /
            len(document_scores)
            for key in document_scores[0]
        }
        result['valid_prediction_count'] = len(document_scores)
        result['invalid_prediction_count'] = len(predictions) - len(document_scores)
        result['details'] = details
        return result
