import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from mmengine.config import Config

from opencompass.datasets.croissantminer import (
    CANONICAL_FIELDS, CroissantMinerDataset, CroissantMinerEvaluator,
    JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT, SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE, build_extraction_messages,
    canonicalize_prediction, resolve_dataset_bundle, resolve_paper_url)
from opencompass.registry import ICL_EVALUATORS, ICL_PROMPT_TEMPLATES
from opencompass.utils import build_dataset_from_cfg


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text_pdf(path: Path, text: str) -> None:
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        (b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
         b'/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>'),
        None,
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    ]
    content = f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode()
    objects[3] = (f'<< /Length {len(content)} >>\nstream\n'.encode()
                  + content + b'\nendstream')
    output = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f'{index} 0 obj\n'.encode() + body + b'\nendobj\n')
    xref = len(output)
    output.extend(f'xref\n0 {len(objects) + 1}\n'.encode())
    output.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        output.extend(f'{offset:010d} 00000 n \n'.encode())
    output.extend(
        f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
        f'startxref\n{xref}\n%%EOF\n'.encode())
    path.write_bytes(output)


def _fixture_manifest(tmp_path: Path) -> Path:
    sample_id = 'sample.with.dots'
    gold = tmp_path / 'gold.parquet'
    pd.DataFrame([
        dict(paper_id=sample_id, field_id='sc:name', gold_value='Example',
             gold_method='unanimous_3of3'),
        dict(paper_id=sample_id, field_id='rai:dataCollection',
             gold_value='Collected by survey',
             gold_method='unanimous_3of3'),
    ]).to_parquet(gold)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'dev': [], 'test': [sample_id]}))
    links = tmp_path / 'links.json'
    links.write_text(json.dumps({sample_id: 'https://example.test/paper.pdf'}))
    paper = tmp_path / 'paper.pdf'
    _write_text_pdf(paper, 'Dataset paper sample')
    files = {
        'gold_annotations': gold,
        'dev_test_split': split,
        'paper_source_manifest': links,
        'paper_sample_with_dots': paper,
    }
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'items': [
            {'id': key, 'path': str(value), 'sha256': _digest(value)}
            for key, value in files.items()
        ]
    }))
    return manifest


def test_registered_dataset_loads_materialized_shape(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    cfg = Config.fromfile(
        'opencompass/configs/datasets/CroissantMiner/croissantminer_gen.py')
    dataset_cfg = cfg.croissantminer_datasets[0].copy()
    dataset_cfg.path = str(manifest)
    dataset_cfg.download = False

    dataset = build_dataset_from_cfg(dataset_cfg)

    assert len(dataset.test) == 1
    assert dataset.test[0]['sample_id'] == 'sample.with.dots'
    assert dataset.test[0]['paper_text'] == 'Dataset paper sample'
    assert json.loads(dataset.test[0]['reference'])['fields'] == {
        'sc:name': 'Example',
        'rai:dataCollection': 'Collected by survey',
    }


def test_configured_prompt_is_the_canonical_zero_shot_chat():
    cfg = Config.fromfile(
        'opencompass/configs/datasets/CroissantMiner/croissantminer_gen.py')
    template = ICL_PROMPT_TEMPLATES.build(
        cfg.croissantminer_datasets[0].infer_cfg.prompt_template)
    assert cfg.croissantminer_datasets[0].infer_cfg.inferencer.max_out_len == 8192
    assert cfg.croissantminer_datasets[0].eval_cfg.evaluator.judge_model_cfg == {}
    rendered = template.generate_item({'paper_text': 'paper',
                                       'sample_id': 'id'})
    assert rendered == build_extraction_messages('paper')
    assert rendered == [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PROMPT_TEMPLATE % 'paper'},
    ]


def test_prediction_parser_and_source_url_erratum():
    raw = 'before```json\n{"sc:name": "Example"}\n```after'
    parsed = canonicalize_prediction(raw, 'sample')
    assert set(parsed) == CANONICAL_FIELDS
    assert parsed['name'] == 'Example'
    assert resolve_paper_url(
        'https://arxiv.org/pdf/cs/0306050') == (
            'https://arxiv.org/pdf/cs/0306050.pdf')
    assert resolve_paper_url('https://aclanthology.org/P16-1144') == (
        'https://aclanthology.org/P16-1144.pdf')
    assert resolve_paper_url(
        'http://www.aclweb.org/anthology/P16-1144') == (
            'https://aclanthology.org/P16-1144.pdf')
    with pytest.raises(ValueError, match='extra fields'):
        canonicalize_prediction({'unexpected': 'value'}, 'bad')


def test_materialized_manifest_rejects_missing_and_corrupt_files(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload['items'][0]['sha256'] = '0' * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='checksum mismatch'):
        resolve_dataset_bundle(manifest, download=False)

    payload['items'][0]['sha256'] = _digest(
        Path(payload['items'][0]['path']))
    Path(payload['items'][-1]['path']).unlink()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(FileNotFoundError, match='paper_sample'):
        resolve_dataset_bundle(manifest, download=False)


class _Judge:

    def __init__(self):
        self.prompts = None

    def generate(self, prompts, max_out_len):
        self.prompts = prompts
        assert max_out_len == 4000
        return ['```json\n{"score": 2, "reason": "partial"}\n```'] * len(
            prompts)


def test_indexed_llmjudge_config_accepts_runtime_judge_and_scores():
    entries = yaml.safe_load(Path('dataset-index.yml').read_text())
    index_entry = next(item['croissantminer'] for item in entries
                       if 'croissantminer' in item)
    cfg = Config.fromfile(index_entry['configpath_llmjudge'])
    evaluator_cfg = cfg.croissantminer_datasets[0].eval_cfg.evaluator.copy()
    evaluator_cfg.judge_model_cfg = dict(type=_Judge)
    evaluator = ICL_EVALUATORS.build(evaluator_cfg)

    prediction = {field: None for field in CANONICAL_FIELDS}
    prediction['name'] = 'Example Dataset'
    prediction['rai:dataCollection'] = 'A survey was used'
    reference = json.dumps({
        'paper_id': 'sample',
        'fields': {
            'sc:name': 'Example',
            'rai:dataCollection': 'Collected by survey',
        },
    })

    result = evaluator.score([prediction], [reference])

    assert result['Core'] == 1.0
    assert result['RAI'] == 0.5


def test_evaluator_runs_rule_judge_and_field_macro_aggregation():
    judge = _Judge()
    prediction = {field: None for field in CANONICAL_FIELDS}
    prediction['name'] = 'Example Dataset'
    prediction['rai:dataCollection'] = 'A survey was used'
    reference = json.dumps({
        'paper_id': 'sample',
        'fields': {
            'sc:name': 'Example',
            'rai:dataCollection': 'Collected by survey',
        },
    })

    result = CroissantMinerEvaluator(judge_model=judge).score(
        [json.dumps(prediction)], [reference])

    assert result['Core'] == 1.0
    assert result['RAI'] == 0.5
    assert result['Composite'] == 0.75
    assert result['Scored Papers'] == 1
    assert result['Scored Cells'] == 2
    assert judge.prompts == [[
        {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
        {'role': 'user', 'content': JUDGE_USER_PROMPT.format(
            field='rai:dataCollection',
            reference='Collected by survey',
            candidate='A survey was used')},
    ]]


def test_missing_metric_stratum_is_explicitly_unsupported():
    prediction = {field: None for field in CANONICAL_FIELDS}
    prediction['name'] = 'Example'
    reference = json.dumps({
        'paper_id': 'sample',
        'fields': {'sc:name': 'Example'},
    })
    with pytest.raises(ValueError, match='metric stratum'):
        CroissantMinerEvaluator(judge_model=_Judge()).score(
            [prediction], [reference])
