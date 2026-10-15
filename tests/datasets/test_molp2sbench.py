import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from opencompass.configs.datasets.molp2sbench.molp2sbench_gen import (
    molp2sbench_datasets, molp2sbench_pgr_datasets)
from opencompass.datasets.molp2sbench import (
    MolP2SBenchDataset, MolP2SBenchEvaluator, UnscorableMetricError,
    property_grounded_recovery, score_prediction_rows, validate_rows)
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import build_dataset_from_cfg


def _task(index):
    return {
        'record_id': f'record-{index:03d}',
        'masked_smiles': f'C[*:{index}]',
        'mask': {
            'level': 'functional_group',
            'attachment_count': 1,
            'boundary_bond_orders': ['SINGLE'],
        },
        'properties': {
            'dipole_moment_D': {
                'quantity': 'dipole moment',
                'unit': 'D',
                'value': float(index),
            }
        },
        'candidate_completed_smiles': ['CC', 'CO'],
    }


def _write_release(root, condition='correct', count=136):
    folder = root / 'data' / 'dipole'
    folder.mkdir(parents=True)
    tasks = [_task(index) for index in range(count)]
    labels = [{
        'record_id': task['record_id'],
        'label': index % 2,
    } for index, task in enumerate(tasks)]
    (folder / f'{condition}.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in tasks), encoding='utf-8')
    (folder / f'{condition}.labels.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in labels), encoding='utf-8')


def test_registered_config_loads_real_jsonl_shape(tmp_path):
    _write_release(tmp_path)
    cfg = deepcopy(next(
        cfg for cfg in molp2sbench_datasets
        if cfg['property_name'] == 'dipole'
        and cfg['conditions'] == ['correct']))
    cfg['path'] = str(tmp_path)
    dataset = build_dataset_from_cfg(cfg)

    assert LOAD_DATASET.get('MolP2SBenchDataset') is MolP2SBenchDataset
    assert ICL_EVALUATORS.get(
        'MolP2SBenchEvaluator') is MolP2SBenchEvaluator
    assert len(dataset.test) == 136
    row = dataset.test[0]
    assert row['sample_id'] == 'dipole/correct/record-000'
    assert row['reference']['label'] == 0
    assert '0. CC\n1. CO' in row['user_prompt']
    assert '"unit": "D"' in row['user_prompt']
    assert row['reference']['family_id'] == row['family_id']


def test_catalog_exposes_all_views_and_primary_pgr_configs():
    assert len(molp2sbench_datasets) == 45
    assert len(molp2sbench_pgr_datasets) == 11
    assert all(cfg['complete_families'] for cfg in molp2sbench_pgr_datasets)


def test_literal_prompt_assets_are_pinned():
    prompt_dir = (Path(__file__).parents[2] / 'opencompass' / 'datasets' /
                  'molp2sbench_prompts')
    expected = {
        'single_property_system.txt':
        '57ea0dc54898b772976e57e2839aa8d3cc1cc87a04120ceb9e14e818a5481b5a',
        'single_property_user.txt':
        'f9973c2b463b6794307d5c53cebaab1ff2b99719a2b3f19dac2d5a8bb462f203',
        'property_bundle_system.txt':
        'e01e15fdd3a78cd10fa3d0b34f139767f28675076e20a2c70e33484e8dc98fcf',
        'property_bundle_user.txt':
        'f9973c2b463b6794307d5c53cebaab1ff2b99719a2b3f19dac2d5a8bb462f203',
        'all_properties_system.txt':
        '3b4d2e0e60715fa8428606df377576981a5aac1ebff142a0c7717e1c3f188db0',
        'all_properties_user.txt':
        'f9973c2b463b6794307d5c53cebaab1ff2b99719a2b3f19dac2d5a8bb462f203',
    }
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in prompt_dir.glob('*.txt')
    } == expected


def test_official_view_accounting_keeps_malformed_outputs_in_denominator():
    references = [{
        'sample_id': f'dipole/correct/r{i}',
        'record_id': f'r{i}',
        'family_id': f'f{i}',
        'condition': 'correct',
        'label': 0,
        'n_candidates': 2,
    } for i in range(5)]
    predictions = [
        {'record_id': 'r0', 'selected_candidate_index': 0},
        {'record_id': 'r1', 'selected_candidate_index': 1},
        {'record_id': 'r2', 'invalid': True},
        {'record_id': 'r3', 'selected_candidate_index': 2},
    ]
    result = score_prediction_rows(predictions, references, 'dipole',
                                   'correct')
    assert {key: result[key]
            for key in ('n', 'hits', 'missing', 'invalid', 'out_of_range')} == {
                'n': 5,
                'hits': 1,
                'missing': 1,
                'invalid': 1,
                'out_of_range': 1,
            }
    assert result['accuracy'] == pytest.approx(0.2)


def test_official_prediction_parser_rejects_duplicate_and_boolean_indices():
    reference = [{
        'record_id': 'r0',
        'label': 0,
        'n_candidates': 2,
    }]
    with pytest.raises(ValueError, match='duplicate prediction'):
        score_prediction_rows([
            {'record_id': 'r0', 'selected_candidate_index': 0},
            {'record_id': 'r0', 'selected_candidate_index': 0},
        ], reference, 'dipole', 'correct')
    with pytest.raises(ValueError, match='not an integer'):
        score_prediction_rows([{
            'record_id': 'r0',
            'selected_candidate_index': True,
        }], reference, 'dipole', 'correct')


def test_pgr_uses_equal_source_cf_and_family_weights():
    details = [
        dict(family_id='a', condition='correct', correct=True),
        dict(family_id='a', condition='counterfactual', correct=True),
        dict(family_id='a', condition='counterfactual', correct=False),
        dict(family_id='b', condition='correct', correct=False),
        dict(family_id='b', condition='counterfactual', correct=True),
    ]
    assert property_grounded_recovery(details) == pytest.approx(62.5)
    with pytest.raises(UnscorableMetricError):
        property_grounded_recovery([])


def test_loader_reports_missing_sibling_files(tmp_path):
    with pytest.raises(FileNotFoundError, match='sibling task and label'):
        MolP2SBenchDataset.load(
            property_name='dipole', conditions='correct', path=str(tmp_path))


def test_label_bool_behavior_matches_pinned_executable_source():
    task = _task(0)
    rows = validate_rows([task], [{
        'record_id': task['record_id'],
        'label': True,
    }])
    assert rows[0]['label'] is True
