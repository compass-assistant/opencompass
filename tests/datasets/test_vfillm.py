import csv
import json

from mmengine.config import Config
import pytest

from opencompass.datasets import VFiLLMDataset, VFiLLMEvaluator
from opencompass.datasets.VFiLLM import build_prompt, extract_number, is_correct


CONFIG_PATH = (
    'opencompass/configs/datasets/VFiLLM/VFiLLM_gen.py')


def _write_artifacts(root, include_atom=True):
    questions = root / 'questions.csv'
    atoms = root / 'atoms.json'
    with questions.open('w', newline='', encoding='utf-8') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=['question_id', 'depth', 'question', 'answer',
                        'leaf_1_entity'])
        writer.writeheader()
        writer.writerow({
            'question_id': 'q001',
            'depth': '1',
            'question': 'What is total revenue?',
            'answer': '12.5',
            'leaf_1_entity': 'ACME',
        })
    atom_list = [] if not include_atom else [{
        'entity': 'ACME',
        'period': 'For the Year Ended December 31, 2024',
        'concept': 'total_revenues',
        'label': 'Total revenues',
        'unit': 'USD',
        'value': 12.5,
        'role': 'leaf',
        'parent_concept': '',
    }]
    atoms.write_text(json.dumps(atom_list), encoding='utf-8')
    return questions, atoms


def test_native_config_loads_and_declared_components_are_constructible():
    config = Config.fromfile(CONFIG_PATH)
    entries = config['VFiLLM_datasets']
    assert len(entries) == 1
    entry = entries[0]
    assert entry['type'] is VFiLLMDataset
    assert entry['limit'] == 250
    evaluator_config = entry['eval_cfg']['evaluator']
    assert evaluator_config['type'] is VFiLLMEvaluator
    assert evaluator_config['tolerance'] == 0.1
    evaluator = evaluator_config['type'](
        tolerance=evaluator_config['tolerance'])
    assert evaluator.tolerance == 0.1


def test_loader_preserves_order_and_row_shape(tmp_path):
    questions, atoms = _write_artifacts(tmp_path)
    dataset = VFiLLMDataset.load(str(questions), str(atoms), limit=1)
    assert dataset.column_names == [
        'sample_id', 'prompt', 'question', 'target'
    ]
    row = dataset[0]
    assert row['sample_id'] == 'q001'
    assert row['question'] == 'What is total revenue?'
    assert row['target'] == {
        'answer': '12.5',
        'depth': '1',
        'entity': 'ACME',
        'skip': False,
    }
    assert row['prompt'].startswith(
        'ROLE: Financial analyst. Answer using ONLY the data below.')
    assert '=== DATA ===' in row['prompt']
    assert 'QUESTION: What is total revenue?' in row['prompt']
    assert row['prompt'].endswith('SOLVE NOW:')


def test_loader_missing_files_and_invalid_atoms_fail_clearly(tmp_path):
    questions, atoms = _write_artifacts(tmp_path)
    with pytest.raises(FileNotFoundError, match='questions file not found'):
        VFiLLMDataset.load(str(tmp_path / 'missing.csv'), str(atoms))
    with pytest.raises(FileNotFoundError, match='atoms file not found'):
        VFiLLMDataset.load(str(questions), str(tmp_path / 'missing.json'))
    atoms.write_text('{"not":"an array"}', encoding='utf-8')
    with pytest.raises(ValueError, match='JSON array'):
        VFiLLMDataset.load(str(questions), str(atoms))


def test_parser_preserves_official_answer_and_fallback_semantics():
    assert extract_number('Answer: 1,234.50') == 1234.5
    assert extract_number('answer: True.') is True
    assert extract_number('ANSWER: False;') is False
    assert extract_number('first 1 then 2') == 2.0
    assert extract_number('no numbers here') is None


def test_scorer_preserves_boolean_zero_and_tolerance_boundaries():
    assert is_correct(True, False) is False
    assert is_correct(False, False) is True
    assert is_correct(0.0, '0') is True
    assert is_correct(1e-7, '0') is True
    assert is_correct(110.0, '100', 0.1) is True
    assert is_correct(110.01, '100', 0.1) is False
    assert is_correct(None, '100') is False


def test_evaluator_excludes_skipped_and_uses_official_keys():
    evaluator = VFiLLMEvaluator(tolerance=0.1)
    metrics = evaluator.score(
        ['Answer: 1,100', 'last number 150', 'Answer: True', None],
        [
            {'answer': '1100', 'skip': False},
            {'answer': '100', 'skip': False},
            {'answer': False, 'skip': False},
            {'answer': '10', 'skip': True},
        ],
    )
    assert metrics == {
        'total': 3,
        'skipped': 1,
        'correct': 1,
        'accuracy': 1 / 3,
        'recovered': 0,
    }


def test_empty_non_skipped_denominator_is_unscorable():
    with pytest.raises(ValueError, match='no non-skipped samples'):
        VFiLLMEvaluator.aggregate([
            {'correct': False, 'skip': True},
            {'correct': False, 'skip': True},
        ])
