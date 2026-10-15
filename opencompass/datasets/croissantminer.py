"""CroissantMiner dataset and evaluator integration.

The prompt, PDF normalization, prediction parser, and field scoring rules in
this module are adapted from CroissantMiner commit
``53a6e2cbb2e666b2d4bba25a03217b500ea0c64f``:
https://github.com/berkearda/croissantminer

MIT License

Copyright (c) 2026 The CroissantMiner Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.registry import (ICL_EVALUATORS, ICL_PROMPT_TEMPLATES,
                                  LOAD_DATASET)
from opencompass.utils import build_model_from_cfg

from .base import BaseDataset

__all__ = [
    'CroissantMinerDataset', 'CroissantMinerEvaluator',
    'CroissantMinerPromptTemplate', 'SYSTEM_PROMPT', 'USER_PROMPT_TEMPLATE',
    'build_extraction_messages', 'canonicalize_prediction',
    'paper_artifact_id', 'strip_reasoning_prefix',
]

UPSTREAM_REVISION = '53a6e2cbb2e666b2d4bba25a03217b500ea0c64f'
UPSTREAM_RAW = (
    'https://raw.githubusercontent.com/berkearda/croissantminer/'
    f'{UPSTREAM_REVISION}')

METADATA_FILES = {
    'gold_annotations': (
        'gold.parquet',
        f'{UPSTREAM_RAW}/data/annotations/gold.parquet',
        '6ccade63fd0b0a358c86c7a2c37c543348ed93b55c91b27093af5958db00f451',
    ),
    'dev_test_split': (
        'dev_test_split.json',
        f'{UPSTREAM_RAW}/data/agentic/dev_test_split.json',
        'cc65cd0d163063fda02ae0228eb345562db8107e5d95accabe5b95ed71c881ee',
    ),
    'paper_source_manifest': (
        'paper_links.json',
        f'{UPSTREAM_RAW}/data/paper_links.json',
        'e697e9a23349b1af1aead22c4e87018a1d80587b0a9e315769aa9b0ee64ec218',
    ),
}

CANONICAL_FIELDS = frozenset({
    'name', 'description', 'url', 'license', 'creator', 'publisher',
    'datePublished', 'inLanguage', 'citeAs', 'isLiveDataset',
    'rai:dataCollection', 'rai:dataCollectionType',
    'rai:dataCollectionMissingData', 'rai:dataCollectionRawData',
    'rai:dataCollectionTimeframe', 'rai:dataImputationProtocol',
    'rai:dataManipulationProtocol', 'rai:dataPreprocessingProtocol',
    'rai:dataAnnotationProtocol', 'rai:dataAnnotationPlatform',
    'rai:dataAnnotationAnalysis', 'rai:annotationsPerItem',
    'rai:annotatorDemographics', 'rai:machineAnnotationTools',
    'rai:dataReleaseMaintenancePlan', 'rai:personalSensitiveInformation',
    'rai:dataSocialImpact', 'rai:dataBiases', 'rai:dataLimitations',
    'rai:dataUseCases',
})

CONSTRAINED_FIELDS = [
    'sc:name', 'sc:license', 'sc:inLanguage', 'cr:isLiveDataset',
    'sc:datePublished', 'sc:publisher', 'sc:url',
]
SHORT_TEXT_FIELDS = ['sc:creator', 'cr:citeAs', 'sc:description']
LONG_TEXT_RAI_FIELDS = [
    'rai:dataCollection', 'rai:dataCollectionType',
    'rai:dataCollectionMissingData', 'rai:dataCollectionRawData',
    'rai:dataCollectionTimeframe', 'rai:dataImputationProtocol',
    'rai:dataManipulationProtocol', 'rai:dataPreprocessingProtocol',
    'rai:dataAnnotationProtocol', 'rai:dataAnnotationPlatform',
    'rai:dataAnnotationAnalysis', 'rai:annotationsPerItem',
    'rai:annotatorDemographics', 'rai:machineAnnotationTools',
    'rai:dataReleaseMaintenancePlan', 'rai:personalSensitiveInformation',
    'rai:dataSocialImpact', 'rai:dataBiases', 'rai:dataLimitations',
    'rai:dataUseCases',
]
TIER1_FIELDS = CONSTRAINED_FIELDS + SHORT_TEXT_FIELDS
ALL_30_FIELDS = TIER1_FIELDS + LONG_TEXT_RAI_FIELDS
DEFAULT_GOLD_METHODS = (
    'unanimous_3of3', 'unanimous_4of4', 'majority_2of3', 'majority_3of4',
    'adjudicated', 'tie_resolved_via_correction', 'adjudicated_berke',
    'audit_corrected_2026-05-04',
)
VALID_VALUE_TYPES = (str, type(None), dict, list, bool, int, float)

SYSTEM_PROMPT = '''You are an expert researcher specializing in "Responsible AI" and metadata extraction for Machine Learning datasets.

**Your Task:**
Extract structured metadata from the provided academic paper according to the MLCommons Croissant RAI schema described below.

**Extraction Guidelines:**
1. **Accuracy First:** Only extract information explicitly stated in the paper.
2. **No Guessing:** If information is not found, return null. Do not fill fields with your own background information or assumptions that are not explicitly given in the paper.
3. **Formatting Rules:**
   - Return ONLY valid JSON.
   - Use null for missing fields.
   - No markdown formatting.

4. **RAI Field-Specific Information:**
   *Use the following official definitions to guide your extraction. Use them as matching criteria when scanning the text:*

   - **rai:dataCollection**: Description of the data collection process.
   - **rai:dataCollectionType**: Define the data collection type(s). Choose one or multiple values from: Surveys, Secondary Data analysis, Physical data collection, Direct measurement, Document analysis, Manual Human Curator, Software Collection, Experiments, Web Scraping, Web API, Focus groups, Self-reporting, Customer feedback data, User-generated content data, Passive Data Collection, Others.
   - **rai:dataCollectionMissingData**: Description of missing data in textual form. Only if missingness is explicitly discussed.
   - **rai:dataCollectionRawData**: Description of the raw data collection (i.e., source of the data).
   - **rai:dataCollectionTimeframe**: Timeframe in terms of start and end date of the collection process.
   - **rai:dataImputationProtocol**: Description of data imputation process if applicable, i.e., how missing or incomplete data were imputed or filled.
   - **rai:dataManipulationProtocol**: Description of data manipulation process if applicable.
   - **rai:dataPreprocessingProtocol**: Description of the steps that were required to bring collected data to a state that can be processed by an ML model/algorithm (e.g., filtering out incomplete entries).
   - **rai:dataAnnotationProtocol**: Description of annotations (labels, ratings), and how these were created or authored.
   - **rai:dataAnnotationPlatform**: Platform, tool, or library used to collect annotations by human annotators.
   - **rai:dataAnnotationAnalysis**: Considerations related to the process of converting the "raw" annotations into the final labels (e.g., uncertainty, disagreement analysis).
   - **rai:dataReleaseMaintenancePlan**: Versioning information in terms of the updating timeframe, the maintainers, and the deprecation policies.
   - **rai:personalSensitiveInformation**: Any sensitive human attribute(s) collected as part of this dataset (e.g., gender, socio-economic status, geography, language, age, culture, experience).
   - **rai:dataSocialImpact**: Discussion of social implications, if applicable.
   - **rai:dataBiases**: Description of biases in the dataset, if applicable.
   - **rai:dataLimitations**: Known limitations (e.g., data generalization limits, quality issues) and non-recommended uses.
   - **rai:dataUseCases**: Dataset use case(s) (e.g., Training, Testing, Validation, Fine Tuning) and Usage Guidelines.
   - **rai:annotationsPerItem**: Number of human labels per dataset item.
   - **rai:annotatorDemographics**: List of demographics specifications about the annotators.
   - **rai:machineAnnotationTools**: List of software used for data annotation (e.g., NER tools, automated labelers).'''

USER_PROMPT_TEMPLATE = '''Extract metadata from the following academic paper by matching text to the field descriptions above. Your output must conform exactly to the following schema:

SCHEMA:
{
  "sc:name": "string",
  "sc:description": "string",
  "sc:url": "string",
  "sc:license": "string",
  "sc:creator": "string",
  "sc:publisher": "string",
  "sc:datePublished": "string",
  "sc:inLanguage": "string",
  "cr:citeAs": "string",
  "cr:isLiveDataset": "string",

  "rai:dataCollection": "string",
  "rai:dataCollectionType": "string (Select from recommended values)",
  "rai:dataCollectionMissingData": "string",
  "rai:dataCollectionRawData": "string",
  "rai:dataCollectionTimeframe": "string",
  "rai:dataImputationProtocol": "string",
  "rai:dataManipulationProtocol": "string",
  "rai:dataPreprocessingProtocol": "string",

  "rai:dataAnnotationProtocol": "string",
  "rai:dataAnnotationPlatform": "string",
  "rai:dataAnnotationAnalysis": "string",
  "rai:annotationsPerItem": "string",
  "rai:annotatorDemographics": "string",
  "rai:machineAnnotationTools": "string",

  "rai:dataReleaseMaintenancePlan": "string",
  "rai:personalSensitiveInformation": "string",
  "rai:dataSocialImpact": "string",
  "rai:dataBiases": "string",
  "rai:dataLimitations": "string",
  "rai:dataUseCases": "string"
}

PAPER TEXT:
%s

Return ONLY valid JSON matching the schema above. Do not include any markdown formatting or explanations.'''

JUDGE_SYSTEM_PROMPT = (
    'You are scoring metadata extractions from ML dataset papers '
    'against a human-validated reference. Output JSON only: '
    '{"score": 1, 2, or 3, "reason": "one short sentence"}')

JUDGE_USER_PROMPT = '''Field: {field}
Reference (human-validated): {reference}
Candidate (model output): {candidate}

Rubric -- rate whether the candidate captures the same factual claim
as the reference. The reference is the authoritative answer for this
paper-field cell, but it is one valid phrasing among many. Phrasing
differences should not lower the score; only differences in factual
content should.

1 = Correct. The candidate captures the essential factual content of
    the reference. The following are EXPLICITLY ACCEPTABLE and should
    NOT lower the score:
      - Paraphrasing, rewording, or restructuring of the same facts
      - Additional accurate detail beyond what the reference contains,
        provided the reference's specific items are still preserved
      - Minor omissions of non-essential information
      - Different level of granularity (compressed vs. expanded)
      - Different ordering of items in a list
    Mark 1 whenever the candidate covers the central factual claim of
    the reference, regardless of phrasing or extra content.

2 = Partially correct. The candidate covers the topic but is missing
    the central factual claim of the reference, OR distorts a key
    fact, OR mixes correct content with clearly unrelated content.

3 = Not correct. The candidate's meaning differs substantially from
    the reference, contradicts it, or describes a different topic
    entirely.

Special case: if the reference is empty / NULL / "[NULL ...]" (paper
does not document this field), score 1 if the candidate is also
empty/null; score 3 if the candidate provides content (hallucination).
The reference is authoritative on whether the paper documents this
field.

Return JSON only.'''

_PREFIX_MAP = {
    'sc:name': 'name',
    'sc:description': 'description',
    'sc:url': 'url',
    'sc:license': 'license',
    'sc:creator': 'creator',
    'sc:publisher': 'publisher',
    'sc:datePublished': 'datePublished',
    'sc:inLanguage': 'inLanguage',
    'cr:citeAs': 'citeAs',
    'cr:isLiveDataset': 'isLiveDataset',
    'sc:citeAs': 'citeAs',
    'sc:isLiveDataset': 'isLiveDataset',
    'cr:name': 'name',
    'cr:description': 'description',
}

_LICENSE_CANON = {}
for _variants, _canonical in [
    (['cc-by-4.0', 'cc by 4.0', 'cc by-4.0', 'creative commons 4.0',
      'creative commons attribution 4.0',
      'creative commons attribution 4.0 license',
      'creative commons attribution 4.0 international',
      'cc-by-4.0 international', 'cc by 4.0 international'], 'cc-by-4.0'),
    (['cc-by-3.0', 'cc by 3.0', 'creative commons attribution 3.0',
      'creative commons attribution 3.0 unported'], 'cc-by-3.0'),
    (['cc-by-sa-4.0', 'cc by-sa 4.0', 'cc by-sa-4.0', 'cc-by-sa 4.0',
      'creative commons attribution-sharealike 4.0',
      'creative commons attribution sharealike 4.0',
      'creative commons attribution-sharealike 4.0 international'],
     'cc-by-sa-4.0'),
    (['cc-by-nc-4.0', 'cc by-nc 4.0', 'cc by-nc-4.0',
      'creative commons attribution-noncommercial 4.0',
      'creative commons attribution noncommercial 4.0',
      'creative commons attribution-noncommercial 4.0 international'],
     'cc-by-nc-4.0'),
    (['cc-by-nc-sa-4.0', 'cc by-nc-sa 4.0',
      'creative commons attribution-noncommercial-sharealike 4.0'],
     'cc-by-nc-sa-4.0'),
    (['cc0', 'cc0 1.0', 'cc0-1.0', 'public domain',
      'creative commons zero', 'cc zero'], 'cc0'),
    (['mit', 'mit license', 'the mit license', 'mit licence'], 'mit'),
    (['apache-2.0', 'apache 2.0', 'apache license 2.0',
      'apache license, version 2.0', 'apache license version 2.0',
      'apache-2.0 license'], 'apache-2.0'),
    (['gpl-3.0', 'gplv3', 'gnu gpl v3',
      'gnu general public license v3', 'gnu general public license v3.0',
      'gpl 3.0', 'gnu general public license version 3'], 'gpl-3.0'),
    (['gpl-2.0', 'gplv2', 'gnu gpl v2', 'gpl 2.0'], 'gpl-2.0'),
    (['bsd-3-clause', 'bsd 3-clause', '3-clause bsd', 'bsd 3 clause',
      'new bsd license', 'modified bsd license'], 'bsd-3-clause'),
    (['bsd-2-clause', 'bsd 2-clause', 'simplified bsd license'],
     'bsd-2-clause'),
    (['custom', 'custom license', 'other', 'proprietary'], 'custom'),
    (['unknown', 'unknown license', 'not specified'], 'unknown'),
]:
    for _variant in _variants:
        _LICENSE_CANON[_variant] = _canonical

_LANG_MAP = {
    'en': 'english', 'eng': 'english', 'english': 'english',
    'de': 'german', 'deu': 'german', 'german': 'german',
    'fr': 'french', 'fra': 'french', 'french': 'french',
    'es': 'spanish', 'spa': 'spanish', 'spanish': 'spanish',
    'it': 'italian', 'ita': 'italian', 'italian': 'italian',
    'pt': 'portuguese', 'por': 'portuguese', 'portuguese': 'portuguese',
    'nl': 'dutch', 'nld': 'dutch', 'dutch': 'dutch',
    'pl': 'polish', 'pol': 'polish', 'polish': 'polish',
    'zh': 'chinese', 'zho': 'chinese', 'chinese': 'chinese',
    'ja': 'japanese', 'jpn': 'japanese', 'japanese': 'japanese',
    'multilingual': 'multilingual',
}

_NULL_GOLD_RE = re.compile(
    r'^(\[null|(null|n/?a|none|unknown)[\s.,;:-]*$|not '
    r'(disclosed|found|mentioned|specified|applicable|available)\b)',
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_download(url: str, destination: Path,
                     expected_sha256: Optional[str] = None,
                     required_marker: Optional[bytes] = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.',
                                     dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, \
                temporary_path.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if required_marker:
            with temporary_path.open('rb') as stream:
                if required_marker not in stream.read(1024):
                    raise ValueError(
                        f'downloaded paper is not a PDF: {url}')
        if expected_sha256 and _sha256(temporary_path) != expected_sha256:
            raise ValueError(f'checksum mismatch for {url}')
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def resolve_paper_url(url: str) -> str:
    """Resolve official paper links, including the documented erratum."""
    arxiv = re.search(r'arxiv\.org/(?:abs|pdf)/([^?#]+)', url)
    if arxiv:
        identifier = arxiv.group(1).removesuffix('.pdf')
        return f'https://arxiv.org/pdf/{identifier}.pdf'
    acl = re.match(
        r'https?://(?:(?:www\.)?aclweb\.org/anthology|aclanthology\.org)'
        r'/([^/?#]+?)(?:\.pdf)?/?$', url)
    if acl:
        return f'https://aclanthology.org/{acl.group(1)}.pdf'
    return url


def paper_artifact_id(paper_id: str) -> str:
    """Map an official paper ID into the dataset-manifest ID namespace."""
    return re.sub(r'[^A-Za-z0-9_-]', '_', f'paper_{paper_id}')


def _materialized_paths(manifest_path: Path) -> tuple[dict[str, Path], list[str]]:
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    items = payload.get('items')
    if not isinstance(items, list) or not items:
        raise ValueError('dataset manifest must contain a non-empty items list')
    paths: dict[str, Path] = {}
    ids: list[str] = []
    for item in items:
        dataset_id = item.get('id')
        raw_path = item.get('path')
        if not isinstance(dataset_id, str) or not isinstance(raw_path, str):
            raise ValueError('each dataset manifest item needs string id/path')
        if dataset_id in paths:
            raise ValueError(f'duplicate dataset manifest id: {dataset_id}')
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f'materialized artifact missing: {dataset_id}')
        expected = item.get('sha256')
        if expected and _sha256(path) != expected:
            raise ValueError(f'checksum mismatch for materialized {dataset_id}')
        paths[dataset_id] = path
        ids.append(dataset_id)
    missing = set(METADATA_FILES) - paths.keys()
    if missing:
        raise ValueError(f'dataset manifest is missing: {sorted(missing)}')
    return paths, ids


def _directory_paths(root: Path, download: bool,
                     split_name: str) -> tuple[dict[str, Path], list[str]]:
    paths = {
        dataset_id: root / filename
        for dataset_id, (filename, _, _) in METADATA_FILES.items()
    }
    for dataset_id, (filename, url, digest) in METADATA_FILES.items():
        path = root / filename
        if not path.is_file():
            if not download:
                raise FileNotFoundError(
                    f'{path} is missing; set download=True to fetch the '
                    'public pinned CroissantMiner artifacts')
            _atomic_download(url, path, digest)
        elif _sha256(path) != digest:
            raise ValueError(f'checksum mismatch for {path}')

    split = json.loads(paths['dev_test_split'].read_text(encoding='utf-8'))
    if split_name not in ('dev', 'test') or not isinstance(
            split.get(split_name), list):
        raise ValueError(
            f'unknown or malformed CroissantMiner split: {split_name}')
    links = json.loads(
        paths['paper_source_manifest'].read_text(encoding='utf-8'))
    paper_dir = root / 'papers'
    for paper_id in sorted(set(split[split_name])):
        if paper_id not in links:
            raise ValueError(f'paper source missing for {paper_id}')
        pdf_path = paper_dir / f'{paper_id}.pdf'
        if not pdf_path.is_file():
            if not download:
                raise FileNotFoundError(f'source PDF missing for {paper_id}')
            _atomic_download(resolve_paper_url(links[paper_id]), pdf_path,
                             required_marker=b'%PDF-')
        else:
            with pdf_path.open('rb') as stream:
                if b'%PDF-' not in stream.read(1024):
                    raise ValueError(f'cached paper is not a PDF: {pdf_path}')
        paths[paper_artifact_id(paper_id)] = pdf_path
    return paths, list(paths)


def resolve_dataset_bundle(path: str | os.PathLike,
                           download: bool = True,
                           split: str = 'test') -> tuple[dict[str, Path],
                                                         list[str]]:
    """Resolve either an Adapter materialization manifest or cache folder."""
    location = Path(path).expanduser()
    if location.is_file():
        return _materialized_paths(location)
    if location.exists() and not location.is_dir():
        raise ValueError(f'dataset path is neither a manifest nor directory: {path}')
    return _directory_paths(location, download=download, split_name=split)


def extract_text_from_pdf(pdf_path: str | os.PathLike) -> str:
    """Apply the released PyPDF2 page extraction behavior."""
    try:
        import PyPDF2
    except ImportError as error:
        raise ImportError(
            'CroissantMiner PDF loading requires PyPDF2==3.0.1') from error
    with Path(pdf_path).open('rb') as stream:
        reader = PyPDF2.PdfReader(stream)
        pages = []
        for page in reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                continue
            if page_text:
                pages.append(page_text)
    return '\n\n[PAGE_BREAK]\n\n'.join(pages)


def clean_text(text: Any) -> str:
    """Apply the released full-PDF text cleanup literally."""
    if not isinstance(text, str):
        return ''
    text = text.replace('[PAGE_BREAK]', '\n')
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    for old, new in [('ﬁ', 'fi'), ('ﬂ', 'fl'), ('ﬀ', 'ff'),
                     ('ﬃ', 'ffi'), ('ﬄ', 'ffl')]:
        text = text.replace(old, new)
    text = text.replace('rn', 'm')
    text = text.replace('vv', 'w')
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_extraction_messages(paper_text: str) -> list[dict[str, str]]:
    """Return the exact canonical zero-shot chat."""
    paper_text = paper_text.encode('utf-8', errors='replace').decode('utf-8')
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PROMPT_TEMPLATE % paper_text},
    ]


@ICL_PROMPT_TEMPLATES.register_module()
class CroissantMinerPromptTemplate(RawPromptTemplate):
    """Native raw-message template for the benchmark's frozen prompt."""

    def __init__(self):
        super().__init__(messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',
             'content': USER_PROMPT_TEMPLATE.replace('%s', '{paper_text}')},
        ])


def parse_json_response(text: str) -> dict:
    """Parse the first Markdown JSON fence, as the single-pass runner does."""
    text = text.strip()
    if '```json' in text:
        text = text.split('```json', 1)[1].split('```', 1)[0].strip()
    elif '```' in text:
        text = text.split('```', 1)[1].split('```', 1)[0].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError('CroissantMiner prediction must decode to an object')
    return parsed


def normalize_field_names(metadata: dict) -> dict:
    return {_PREFIX_MAP.get(key, key): value for key, value in metadata.items()}


def validate_extraction(data: dict, sample_id: str = 'unknown') -> None:
    if not isinstance(data, dict):
        raise ValueError(f'{sample_id}: extraction is not an object')
    fields = set(data)
    missing = CANONICAL_FIELDS - fields
    extra = fields - CANONICAL_FIELDS
    errors = []
    if missing:
        errors.append(f'missing fields: {sorted(missing)}')
    if extra:
        errors.append(f'extra fields: {sorted(extra)}')
    for field, value in data.items():
        if field != field.strip():
            errors.append(f'field {field!r} has surrounding whitespace')
        if not isinstance(value, VALID_VALUE_TYPES):
            errors.append(f'{field}: unsupported {type(value).__name__}')
    if errors:
        raise ValueError(f'{sample_id}: ' + '; '.join(errors))


def canonicalize_prediction(prediction: str | dict,
                            sample_id: str = 'unknown') -> dict:
    if isinstance(prediction, str):
        prediction = parse_json_response(prediction)
    if not isinstance(prediction, dict):
        raise ValueError(f'{sample_id}: prediction is not an object')
    for key in ('extraction', 'fields'):
        nested = prediction.get(key)
        if isinstance(nested, dict):
            prediction = nested
            break
    normalized = normalize_field_names(prediction)
    for field in CANONICAL_FIELDS:
        normalized.setdefault(field, None)
    validate_extraction(normalized, sample_id)
    return normalized


def strip_reasoning_prefix(text: str) -> str:
    """Keep canonical content while retaining reasoning-only responses."""
    if '</think>' in text:
        return text.split('</think>', 1)[-1]
    return text


def _paper_path(paths: dict[str, Path], paper_id: str) -> Path:
    artifact_id = paper_artifact_id(paper_id)
    if artifact_id not in paths:
        raise FileNotFoundError(f'dataset artifact missing: {artifact_id}')
    return paths[artifact_id]


def load_croissantminer_records(
    path: str | os.PathLike,
    split: str = 'test',
    sample_ids: Optional[Iterable[str]] = None,
    download: bool = True,
) -> tuple[list[dict], list[str]]:
    paths, dataset_ids = resolve_dataset_bundle(
        path, download=download, split=split)
    split_data = json.loads(
        paths['dev_test_split'].read_text(encoding='utf-8'))
    if split not in ('dev', 'test') or not isinstance(split_data.get(split), list):
        raise ValueError(f'unknown or malformed CroissantMiner split: {split}')
    official_ids = sorted(split_data[split])
    if len(official_ids) != len(set(official_ids)):
        raise ValueError(f'duplicate sample IDs in {split} split')
    if sample_ids is None:
        ordered_ids = official_ids
    else:
        ordered_ids = list(sample_ids)
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError('sample_ids must be unique')
        unknown = set(ordered_ids) - set(official_ids)
        if unknown:
            raise ValueError(f'sample_ids outside {split}: {sorted(unknown)}')

    gold = pd.read_parquet(paths['gold_annotations'])
    required_columns = {'paper_id', 'field_id', 'gold_value', 'gold_method'}
    missing_columns = required_columns - set(gold.columns)
    if missing_columns:
        raise ValueError(f'gold parquet missing columns: {sorted(missing_columns)}')
    gold = gold[
        gold['gold_method'].isin(DEFAULT_GOLD_METHODS)
        & gold['gold_value'].notna()
        & gold['paper_id'].isin(ordered_ids)
    ]
    grouped = {
        paper_id: dict(zip(rows['field_id'], rows['gold_value']))
        for paper_id, rows in gold.groupby('paper_id', sort=False)
    }

    records = []
    for paper_id in ordered_ids:
        fields = grouped.get(paper_id, {})
        if not fields:
            raise ValueError(f'no settled gold rows for {paper_id}')
        unknown_fields = set(fields) - set(ALL_30_FIELDS)
        if unknown_fields:
            raise ValueError(
                f'unknown gold fields for {paper_id}: {sorted(unknown_fields)}')
        raw = extract_text_from_pdf(_paper_path(paths, paper_id))
        paper_text = clean_text(raw).encode(
            'utf-8', errors='replace').decode('utf-8')
        if not paper_text:
            raise ValueError(f'PDF produced no text for {paper_id}')
        records.append({
            'sample_id': paper_id,
            'paper_text': paper_text,
            'reference': json.dumps({
                'paper_id': paper_id,
                'fields': fields,
            }, ensure_ascii=False),
        })
    return records, dataset_ids


@LOAD_DATASET.register_module()
class CroissantMinerDataset(BaseDataset):
    """Load the public, pinned CroissantMiner split and its source PDFs.

    ``path`` may be either a framework cache directory or a materialized
    dataset-manifest JSON. Missing cache files are downloaded from the pinned
    public repository and the official paper URLs when ``download`` is true.
    """

    @staticmethod
    def load(path: str = 'data/croissantminer',
             split: str = 'test',
             sample_ids: Optional[list[str]] = None,
             download: bool = True,
             **kwargs) -> Dataset:
        records, _ = load_croissantminer_records(
            path, split=split, sample_ids=sample_ids, download=download)
        return Dataset.from_list(records)


def get_pred_value(extraction: dict, field_id: str) -> Any:
    for container in ('extraction', 'fields'):
        nested = extraction.get(container)
        if isinstance(nested, dict):
            extraction = nested
            break
    if field_id in extraction and extraction[field_id] is not None:
        return extraction[field_id]
    short = field_id.split(':')[-1] if ':' in field_id else field_id
    if short in extraction and extraction[short] is not None:
        return extraction[short]
    return None


def is_null_gold(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in (
            'null', 'none', 'n/a', 'na', 'unknown', 'not disclosed',
            'not found', 'not mentioned', 'not specified', 'not applicable',
            'not available'):
        return True
    if _NULL_GOLD_RE.match(lowered):
        return True
    if ('not found in paper' in lowered
            or 'not mentioned in paper' in lowered
            or 'not specified in paper' in lowered):
        return True
    return lowered.startswith(('[null', '[na', '[n/a'))


def _normalize_license(text: str) -> str:
    value = text.strip().lower()
    cc_url = re.search(
        r'creativecommons\.org/licenses?/([\w-]+)/([\d.]+)', value)
    if cc_url:
        return f'cc-{cc_url.group(1)}-{cc_url.group(2)}'
    choose_url = re.search(r'choosealicense\.com/licenses?/([\w.-]+)', value)
    if choose_url:
        slug = choose_url.group(1).rstrip('/')
        return _LICENSE_CANON.get(slug, slug)
    value = re.sub(r'https?://\S+', '', value).strip()
    substitutions = [
        (r'creative\s+commons\s+attribution\s*-?\s*non\s*-?\s*commercial\s*-?\s*share\s*-?\s*alike', 'cc-by-nc-sa'),
        (r'creative\s+commons\s+attribution\s*-?\s*non\s*-?\s*commercial', 'cc-by-nc'),
        (r'creative\s+commons\s+attribution\s*-?\s*share\s*-?\s*alike', 'cc-by-sa'),
        (r'creative\s+commons\s+attribution', 'cc-by'),
        (r'creative\s+commons\s+zero', 'cc0'),
        (r'creative\s+commons', 'cc-by'),
    ]
    for pattern, replacement in substitutions:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r'\s*(licen[sc]e|international|unported)\s*$', '',
                   value).strip()
    value = re.sub(r'^the\s+', '', value).strip()
    value = re.sub(r'\bcc\s+by\b', 'cc-by', value)
    value = re.sub(r'\bcc-by\s+', 'cc-by-', value)
    value = re.sub(r'\bcc-by-sa\s+', 'cc-by-sa-', value)
    value = re.sub(r'\bcc-by-nc\s+', 'cc-by-nc-', value)
    value = re.sub(r'\bcc-by-nc-sa\s+', 'cc-by-nc-sa-', value)
    value = re.sub(r'\s+', '-', value).strip('-')
    if value in _LICENSE_CANON:
        return _LICENSE_CANON[value]
    for known in sorted(_LICENSE_CANON, key=len, reverse=True):
        if value.startswith(known) or known.startswith(value):
            return _LICENSE_CANON[known]
    return value


def _normalize_language(text: str) -> frozenset[str]:
    parts = re.split(r'[,;/]\s*|\s+and\s+|\s+', text.strip().lower())
    return frozenset(_LANG_MAP.get(part.strip().rstrip('.'),
                                   part.strip().rstrip('.'))
                     for part in parts if part.strip().rstrip('.'))


def _normalize_name(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r'\s*\(.*?\)\s*', ' ', value).strip()
    value = re.sub(r'\s*(dataset|benchmark|corpus)\s*$', '', value).strip()
    return value.replace('_', ' ').replace('-', ' ')


def _names_match(prediction: str, gold: str) -> bool:
    pred = prediction.strip().lower()
    ref = gold.strip().lower()
    if pred == ref or pred in ref or ref in pred:
        return True
    pred_normalized, ref_normalized = (_normalize_name(prediction),
                                       _normalize_name(gold))
    if (pred_normalized == ref_normalized
            or pred_normalized in ref_normalized
            or ref_normalized in pred_normalized):
        return True
    for abbreviation in re.findall(r'\(([A-Z]{2,})\)', prediction):
        if abbreviation.lower() in ref:
            return True
    for abbreviation in re.findall(r'\(([A-Z]{2,})\)', gold):
        if abbreviation.lower() in pred:
            return True
    pred_tokens = set(re.findall(r'\w+', pred))
    ref_tokens = set(re.findall(r'\w+', ref))
    return bool((ref_tokens and ref_tokens.issubset(pred_tokens))
                or (pred_tokens and pred_tokens.issubset(ref_tokens)))


def score_constrained(prediction: str, gold: str, field_name: str) -> float:
    pred, ref = prediction.strip(), gold.strip()
    if field_name == 'sc:name':
        matches = _names_match(pred, ref)
    elif field_name == 'sc:license':
        matches = _normalize_license(pred) == _normalize_license(ref)
    elif field_name == 'sc:inLanguage':
        matches = _normalize_language(pred) == _normalize_language(ref)
    elif field_name == 'sc:datePublished':
        pred_year = re.search(r'\b(19|20)\d{2}\b', pred)
        ref_year = re.search(r'\b(19|20)\d{2}\b', ref)
        matches = ((pred_year.group(0) == ref_year.group(0))
                   if pred_year and ref_year else pred.lower() == ref.lower())
    elif field_name == 'cr:isLiveDataset':
        normalize = lambda value: ({'yes': 'true', 'true': 'true', '1': 'true',
                                    'no': 'false', 'false': 'false', '0': 'false'}
                                   .get(value.strip().lower(),
                                        value.strip().lower()))
        matches = normalize(pred) == normalize(ref)
    elif field_name == 'sc:url':
        normalize = lambda value: re.sub(
            r'^https?://', '', value.strip().lower().rstrip('/'))
        matches = normalize(pred) == normalize(ref)
    elif field_name == 'sc:publisher':
        pred_tokens = set(re.findall(r'\w+', pred.lower()))
        ref_tokens = set(re.findall(r'\w+', ref.lower()))
        matches = (len(pred_tokens & ref_tokens)
                   / min(len(pred_tokens), len(ref_tokens)) >= 0.5
                   if pred_tokens and ref_tokens else pred.lower() == ref.lower())
    else:
        matches = pred.lower() == ref.lower()
    return 1.0 if matches else 0.0


def score_token_f1(prediction: str, gold: str) -> float:
    pred_tokens = set(re.findall(r'\w+', prediction.lower()))
    ref_tokens = set(re.findall(r'\w+', gold.lower()))
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def build_judge_messages(field: str, reference: Any,
                         candidate: Any) -> list[dict[str, str]]:
    user = JUDGE_USER_PROMPT.format(
        field=field,
        reference=str(reference)[:8000],
        candidate=str(candidate)[:8000],
    )
    return [
        {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
    ]


def parse_judge_response(text: str) -> tuple[int, str]:
    message = text or ''
    if message.strip().startswith('```'):
        message = re.sub(r'^```(?:json)?\s*|\s*```$', '', message.strip(),
                         flags=re.S)
    if '</think>' in message:
        message = message.split('</think>', 1)[-1]
    match = re.search(r'\{[^{}]*"score"[^{}]*\}', message, re.S)
    if not match:
        raise ValueError(f'judge response has no score JSON: {message[:100]}')
    parsed = json.loads(match.group(0))
    score = int(parsed.get('score', 0))
    if score not in (1, 2, 3):
        raise ValueError(f'judge returned invalid score: {score}')
    return score, str(parsed.get('reason', ''))


def _composite_from_matrix(matrix: np.ndarray) -> float:
    with np.errstate(all='ignore'):
        per_field = np.nanmean(matrix, axis=0)
    per_field = per_field[~np.isnan(per_field)]
    return float('nan') if len(per_field) == 0 else float(np.nanmean(per_field))


def _aggregate_scores(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError('CroissantMiner produced an empty score frame')
    by_field: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_field[row['field_id']].append(float(row['score']))
    field_means = {
        field: float(np.mean(values)) for field, values in by_field.items()
    }
    core_values = [field_means[field] for field in TIER1_FIELDS
                   if field in field_means]
    rai_values = [field_means[field] for field in LONG_TEXT_RAI_FIELDS
                  if field in field_means]
    if not core_values or not rai_values:
        raise ValueError('CroissantMiner has a metric stratum with no scores')

    papers = sorted({row['paper_id'] for row in rows})
    fields = sorted(by_field)
    paper_index = {paper: index for index, paper in enumerate(papers)}
    field_index = {field: index for index, field in enumerate(fields)}
    matrix = np.full((len(papers), len(fields)), np.nan)
    for row in rows:
        matrix[paper_index[row['paper_id']], field_index[row['field_id']]] = \
            row['score']
    composite = _composite_from_matrix(matrix)
    rng = np.random.default_rng(42)
    boots = np.empty(2000)
    for index in range(2000):
        choice = rng.integers(0, len(papers), size=len(papers))
        boots[index] = _composite_from_matrix(matrix[choice])
    boots = boots[~np.isnan(boots)]
    return {
        'Core': float(np.mean(core_values)),
        'RAI': float(np.mean(rai_values)),
        'Composite': composite,
        'Composite CI Low': float(np.quantile(boots, 0.025)),
        'Composite CI High': float(np.quantile(boots, 0.975)),
        'Scored Papers': len(papers),
        'Scored Cells': len(rows),
    }


@ICL_EVALUATORS.register_module()
class CroissantMinerEvaluator(BaseEvaluator):
    """Evaluate all 30 fields with the released v2-min GLM-5 protocol."""

    def __init__(self, judge_model_cfg: Optional[dict] = None,
                 judge_model: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.judge_model_cfg = judge_model_cfg
        self.judge_model = judge_model

    def _build_judge_model(self):
        if self.judge_model is None:
            if not self.judge_model_cfg:
                raise ValueError('judge_model_cfg is required for RAI scoring')
            self.judge_model = build_model_from_cfg(self.judge_model_cfg)
            workers = self.judge_model_cfg.get('max_workers')
            if workers is not None:
                self.judge_model.tokens = threading.Semaphore(
                    max(1, int(workers)))

    def score(self, predictions: list, references: list,
              test_set: Optional[Dataset] = None) -> dict:
        if references is None or len(predictions) != len(references):
            raise ValueError('predictions and references must have equal length')
        if test_set is not None and len(test_set) != len(predictions):
            raise ValueError('test_set and predictions must have equal length')

        parsed_predictions = []
        parsed_references = []
        sample_ids = []
        for index, (prediction, reference) in enumerate(
                zip(predictions, references)):
            if isinstance(reference, str):
                reference = json.loads(reference)
            if not isinstance(reference, dict):
                raise ValueError(f'reference {index} is not an object')
            sample_id = str(reference.get(
                'paper_id', test_set[index]['sample_id']
                if test_set is not None else index))
            fields = reference.get('fields', reference)
            if not isinstance(fields, dict):
                raise ValueError(f'{sample_id}: reference fields are invalid')
            parsed_predictions.append(
                canonicalize_prediction(prediction, sample_id))
            parsed_references.append(fields)
            sample_ids.append(sample_id)

        score_rows: list[dict] = []
        judge_cells: list[tuple[str, str, Any, Any]] = []
        for sample_id, prediction, reference in zip(
                sample_ids, parsed_predictions, parsed_references):
            for field_id, gold_value in reference.items():
                candidate = get_pred_value(prediction, field_id)
                candidate_empty = candidate is None or not str(candidate).strip()
                if is_null_gold(gold_value):
                    if not candidate_empty:
                        score_rows.append({
                            'paper_id': sample_id,
                            'field_id': field_id,
                            'score': 0.0,
                        })
                elif field_id in CONSTRAINED_FIELDS:
                    score_rows.append({
                        'paper_id': sample_id,
                        'field_id': field_id,
                        'score': 0.0 if candidate_empty else score_constrained(
                            str(candidate), str(gold_value), field_id),
                    })
                elif field_id in SHORT_TEXT_FIELDS:
                    score_rows.append({
                        'paper_id': sample_id,
                        'field_id': field_id,
                        'score': 0.0 if candidate_empty else score_token_f1(
                            str(candidate), str(gold_value)),
                    })
                elif field_id in LONG_TEXT_RAI_FIELDS:
                    if candidate_empty:
                        score_rows.append({
                            'paper_id': sample_id,
                            'field_id': field_id,
                            'score': 0.0,
                        })
                    else:
                        judge_cells.append((sample_id, field_id, gold_value,
                                            candidate))
                else:
                    raise ValueError(f'unknown Croissant field: {field_id}')

        if judge_cells:
            self._build_judge_model()
            prompts = [
                build_judge_messages(field, gold, candidate)
                for _, field, gold, candidate in judge_cells
            ]
            responses = self.judge_model.generate(prompts, max_out_len=4000)
            if len(responses) != len(judge_cells):
                raise ValueError('judge returned an incomplete response list')
            for cell, response in zip(judge_cells, responses):
                raw_score, reason = parse_judge_response(response)
                sample_id, field_id, _, _ = cell
                score_rows.append({
                    'paper_id': sample_id,
                    'field_id': field_id,
                    'score': {1: 1.0, 2: 0.5, 3: 0.0}[raw_score],
                    'judge_score': raw_score,
                    'judge_reason': reason,
                })

        metrics = _aggregate_scores(score_rows)
        by_paper: dict[str, dict[str, float]] = defaultdict(dict)
        for row in score_rows:
            by_paper[row['paper_id']][row['field_id']] = float(row['score'])
        details = []
        for sample_id in sample_ids:
            field_scores = by_paper[sample_id]
            details.append({
                'sample_id': sample_id,
                'score': (float(np.mean(list(field_scores.values())))
                          if field_scores else None),
                'field_scores': field_scores,
            })
        return {**metrics, 'details': details}


def finite_metrics(result: dict) -> dict[str, float | int]:
    """Select public numeric metrics and reject undefined values."""
    metrics = {key: value for key, value in result.items()
               if key != 'details' and isinstance(value, (int, float))}
    if not metrics or any(not math.isfinite(float(value))
                          for value in metrics.values()):
        raise ValueError('CroissantMiner metrics must be finite and non-empty')
    return metrics
