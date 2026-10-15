import json
from copy import deepcopy
from pathlib import Path

import pytest
from mmengine.config import ConfigDict

from opencompass.configs.datasets.regextract.regextract_gen import (
    regextract_datasets,
)
from opencompass.datasets import (RegExtractDataset, RegExtractEvaluator,
                                  parse_regextract_prediction,
                                  score_regextract_document)
from opencompass.utils import build_dataset_from_cfg


def _field(value, provenance='source text'):
    return {'value': value, 'provenance': provenance}


def _gold():
    return {
        'title': _field('PFAS Act'),
        'location': _field('Example'),
        'reference': _field('HB 1'),
        'status': _field('active'),
        'date': _field('2025'),
        'pfas_definition': _field('single_fluorinated_carbon'),
        'regulations': [{
            'type': _field('ban'),
            'date': _field('2026'),
            'who': _field(['manufacturer']),
            'products': _field(['packaging']),
        }],
    }


def _write_dataset(root, count=20):
    text_dir = root / 'golden_dataset_txts'
    gold_dir = root / 'golden_dataset_json'
    text_dir.mkdir()
    gold_dir.mkdir()
    (root / 'prompts.py').write_text(
        "IE_prompt_schema_design_joint_extraction = 'OFFICIAL PROMPT\\n'\n",
        encoding='utf-8')
    for index in range(1, count + 1):
        name = f'document-{index:02d}'
        (text_dir / f'{name}.txt').write_text(
            f'document {index}', encoding='utf-8')
        (gold_dir / f'{name}.json').write_text(
            json.dumps(_gold()), encoding='utf-8')


def test_registered_config_builds_real_rows_and_prompt(tmp_path):
    _write_dataset(tmp_path)
    cfg = deepcopy(regextract_datasets[0])
    cfg['path'] = str(tmp_path)
    dataset = build_dataset_from_cfg(ConfigDict(cfg))

    assert isinstance(dataset, RegExtractDataset)
    assert dataset.test['id'] == [
        f'document-{index:02d}' for index in range(1, 21)
    ]
    assert dataset.test[0]['document'] == 'document 1'
    assert dataset.test[0]['prompt'] == (
        'OFFICIAL PROMPT\n\n\nDocument:\ndocument 1')
    reference = json.loads(dataset.test[0]['reference'])
    assert reference['title']['value'] == 'PFAS Act'


def test_loader_preserves_heterogeneous_reference_values(tmp_path):
    _write_dataset(tmp_path)
    first_path = (tmp_path / 'golden_dataset_json' / 'document-01.json')
    second_path = (tmp_path / 'golden_dataset_json' / 'document-02.json')
    first = json.loads(first_path.read_text(encoding='utf-8'))
    second = json.loads(second_path.read_text(encoding='utf-8'))
    first['regulations'][0]['exceptions'] = _field(['manufacturer'])
    second['regulations'][0]['exceptions'] = _field(False)
    first_path.write_text(json.dumps(first), encoding='utf-8')
    second_path.write_text(json.dumps(second), encoding='utf-8')

    dataset = RegExtractDataset.load(str(tmp_path))

    assert json.loads(dataset[0]['reference']) == first
    assert json.loads(dataset[1]['reference']) == second


def test_loader_rejects_missing_pair_and_invalid_selection(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / 'golden_dataset_json' / 'document-02.json').unlink()
    with pytest.raises(FileNotFoundError, match='document-02'):
        RegExtractDataset.load(str(tmp_path))

    (tmp_path / 'golden_dataset_json' / 'document-02.json').write_text(
        json.dumps(_gold()), encoding='utf-8')
    with pytest.raises(ValueError, match='dataset order'):
        RegExtractDataset.load(str(tmp_path), sample_ids=['document-02',
                                                          'document-01'])


def test_full_load_rejects_incomplete_paired_corpus(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / 'golden_dataset_txts' / 'document-20.txt').unlink()
    (tmp_path / 'golden_dataset_json' / 'document-20.json').unlink()

    with pytest.raises(ValueError, match='exactly document-01 through'):
        RegExtractDataset.load(str(tmp_path))

    subset = RegExtractDataset.load(
        str(tmp_path), sample_ids=['document-01', 'document-02'])
    assert subset['id'] == ['document-01', 'document-02']


def test_manifest_full_load_rejects_incomplete_paired_corpus(tmp_path):
    _write_dataset(tmp_path)
    items = []
    for index in range(1, 20):
        items.extend([
            {
                'id': f'regextract_input_{index:02d}',
                'path': f'golden_dataset_txts/document-{index:02d}.txt',
            },
            {
                'id': f'regextract_gold_{index:02d}',
                'path': f'golden_dataset_json/document-{index:02d}.json',
            },
        ])
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'items': items}), encoding='utf-8')

    with pytest.raises(ValueError, match='exactly document-01 through'):
        RegExtractDataset.load(
            manifest_path=str(manifest),
            prompt_path=str(tmp_path / 'prompts.py'))

    subset = RegExtractDataset.load(
        manifest_path=str(manifest),
        prompt_path=str(tmp_path / 'prompts.py'),
        sample_ids=['document-01', 'document-02'])
    assert subset['id'] == ['document-01', 'document-02']


def test_config_preserves_published_generation_requirements():
    inferencer = regextract_datasets[0]['infer_cfg']['inferencer']
    assert inferencer['max_out_len'] == 8192

    readme = (Path(__file__).resolve().parents[2]
              / 'opencompass/configs/datasets/regextract/README.md')
    assert '`temperature=0`' in readme.read_text(encoding='utf-8')


def test_parser_is_greedy_like_published_postprocessor():
    assert parse_regextract_prediction('prefix {"a": 1} suffix')['a'] == 1
    with pytest.raises(json.JSONDecodeError):
        parse_regextract_prediction('{"a": 1} between {"b": 2}')
    with pytest.raises(ValueError, match='No JSON object'):
        parse_regextract_prediction('not json')


def test_repaired_missing_pfas_definition_uses_invalid_answer_path():
    prediction = deepcopy(_gold())
    prediction.pop('pfas_definition')
    scores = score_regextract_document(_gold(), prediction)
    assert scores['pfas_definition_value'] == 0.0
    assert scores['pfas_definition_provenance'] == 0.0


def test_mpart_normalization_and_greedy_matching_boundaries():
    prediction = deepcopy(_gold())
    prediction['pfas_definition']['value'] = 'mpart definition'
    prediction['regulations'].append(deepcopy(prediction['regulations'][0]))
    scores = score_regextract_document(_gold(), prediction)
    assert scores['pfas_definition_value'] == 1.0
    assert scores['regulations_match_precision'] == 0.5
    assert scores['regulations_match_recall'] == 1.0
    assert scores['regulations_match_f1'] == pytest.approx(2 / 3)
    assert scores['regulations_products_value_avg_score'] == 1.0
    assert scores['num_pred_regulations'] == 2


def test_evaluator_excludes_only_unparsed_documents_and_reports_coverage():
    valid = json.dumps(_gold())
    result = RegExtractEvaluator().score(
        predictions=['malformed', valid],
        references=[_gold(), _gold()],
        test_set={'id': ['document-01', 'document-02']})
    assert result['invalid_prediction_count'] == 1
    assert result['valid_prediction_count'] == 1
    assert result['regulations_value_avg_score'] == 1.0
    assert [detail['valid'] for detail in result['details']] == [False, True]

    with pytest.raises(ValueError, match='zero valid documents'):
        RegExtractEvaluator().score(['malformed'], [_gold()])
