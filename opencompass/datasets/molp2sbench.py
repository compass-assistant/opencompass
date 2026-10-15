"""Native dataset and exact-match evaluation support for MolP2SBench.

The public release is pinned to ``UPSTREAM_REVISION``.  A caller can use a
release-layout directory, a materialized dataset manifest, or let the loader
download the two files needed for each configured property/view.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import download_url

from .base import BaseDataset


UPSTREAM_REPOSITORY = 'https://github.com/DUAL-Xiao/MolP2SBench'
UPSTREAM_REVISION = '1ede82bf68f634097b296b1937d66a9b04bbf269'

PROPERTIES = (
    'dipole',
    'gap',
    'infrared',
    'raman',
    'ultraviolet_visible',
    'proton_nmr_peaks',
    'proton_nmr_integration',
    'proton_nmr_atom',
    'carbon_nmr_peaks',
    'carbon_nmr_degeneracy',
    'carbon_nmr_atom',
    'electronic',
    'magnetic_resonance',
    'vibrational',
    'all_properties',
)
SINGLE_PROPERTIES = PROPERTIES[:11]
BUNDLE_PROPERTIES = ('electronic', 'magnetic_resonance', 'vibrational')
CONDITIONS = ('correct', 'shuffled', 'counterfactual')
TASK_FIELDS = (
    'record_id',
    'masked_smiles',
    'mask',
    'properties',
    'candidate_completed_smiles',
)
_COUNTERFACTUAL_240 = {
    'dipole',
    'infrared',
    'raman',
    'ultraviolet_visible',
    'electronic',
    'vibrational',
    'all_properties',
}
_PROMPT_DIR = Path(__file__).with_name('molp2sbench_prompts')

PROPERTY_DESCRIPTIONS = {
    'dipole': 'Dipole moment (Euclidean norm).',
    'gap': ('Gap between the highest occupied and lowest unoccupied '
            'molecular orbitals.'),
    'infrared': 'Infrared spectrum.',
    'raman': 'Raman spectrum.',
    'ultraviolet_visible': 'Ultraviolet-visible excitations.',
    'proton_nmr_peaks': 'Proton NMR, peaks only.',
    'proton_nmr_integration': 'Proton NMR, peaks with integration.',
    'proton_nmr_atom': 'Proton NMR, atom-wise.',
    'carbon_nmr_peaks': 'Carbon NMR, peaks only.',
    'carbon_nmr_degeneracy': 'Carbon NMR, peaks with degeneracy.',
    'carbon_nmr_atom': 'Carbon NMR, atom-wise.',
    'electronic': ('Electronic bundle: dipole norm, HOMO-LUMO gap, and '
                   'ultraviolet-visible excitations.'),
    'magnetic_resonance': ('Magnetic-resonance bundle: proton NMR with '
                           'integration and carbon NMR with degeneracy.'),
    'vibrational': 'Vibrational bundle: infrared and Raman spectra.',
    'all_properties': ('All seven canonical properties: dipole norm, gap, '
                       'infrared, Raman, ultraviolet-visible, proton NMR '
                       'with integration, and carbon NMR with degeneracy.'),
}


class UnscorableMetricError(ValueError):
    """Raised when a requested paper metric has no defined denominator."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load object-valued JSONL records with stable source diagnostics."""
    path = Path(path)
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
            if not isinstance(row, dict) or 'record_id' not in row:
                raise ValueError(f'{path}:{line_number}: missing record_id')
            rows.append(row)
    return rows


def validate_rows(
    tasks: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    source: str = 'dataset',
) -> list[dict[str, Any]]:
    """Validate and combine the task/label sibling streams."""
    task_rows = list(tasks)
    label_rows = list(labels)
    if ([row.get('record_id') for row in task_rows]
            != [row.get('record_id') for row in label_rows]):
        raise ValueError(f'task and label record_id lists differ: {source}')

    combined = []
    for task, label in zip(task_rows, label_rows):
        missing = [field for field in TASK_FIELDS if field not in task]
        if missing:
            raise ValueError(
                f"{source} record {task.get('record_id')} missing {missing}")
        answer = label.get('label')
        # This deliberately follows the executable release scorer, where a
        # bool is an int for label validation. Released labels are ordinary
        # integers; model-selected bools are rejected separately below.
        if not isinstance(answer, int):
            raise ValueError(
                f"label is not an integer: {label.get('record_id')}")
        candidates = task['candidate_completed_smiles']
        if not 0 <= answer < len(candidates):
            raise ValueError(f"label out of range: {label.get('record_id')}")
        combined.append({**task, 'label': answer})
    return combined


def _prompt_family(property_name: str) -> str:
    if property_name in SINGLE_PROPERTIES:
        return 'single_property'
    if property_name in BUNDLE_PROPERTIES:
        return 'property_bundle'
    if property_name == 'all_properties':
        return 'all_properties'
    raise ValueError(f'unknown property: {property_name}')


def prompt_values(task: Mapping[str, Any],
                  property_name: str) -> dict[str, str]:
    """Materialize the documented slots without reordering property data."""
    properties = task['properties']
    mask = task['mask']
    return {
        'record_id': str(task['record_id']),
        'masked_smiles': str(task['masked_smiles']),
        'mask.level': str(mask['level']),
        'mask.attachment_count': str(mask['attachment_count']),
        'mask.boundary_bond_orders': str(mask['boundary_bond_orders']),
        'consider_list': '\n'.join(f'- {name}' for name in properties),
        'property_description': PROPERTY_DESCRIPTIONS[property_name],
        'available_properties_block': json.dumps(
            properties, ensure_ascii=False, indent=2),
        'numbered_candidate_list': '\n'.join(
            f'{index}. {candidate}' for index, candidate in enumerate(
                task['candidate_completed_smiles'])),
    }


def _fill_literal_template(template: str, values: Mapping[str, str]) -> str:
    for name, value in values.items():
        template = template.replace('{' + name + '}', value)
    if any('{' + name + '}' in template for name in values):
        raise ValueError('failed to fill a MolP2SBench prompt slot')
    return template


def render_prompt(task: Mapping[str, Any],
                  property_name: str) -> tuple[str, str]:
    """Render the pinned system/user prompt pair for one released task."""
    family = _prompt_family(property_name)
    system = (_PROMPT_DIR / f'{family}_system.txt').read_text(encoding='utf-8')
    user = (_PROMPT_DIR / f'{family}_user.txt').read_text(encoding='utf-8')
    return system, _fill_literal_template(user,
                                           prompt_values(task, property_name))


def family_id(task: Mapping[str, Any]) -> str:
    """Return a stable identifier for a scaffold/candidate atomic family."""
    payload = json.dumps(
        [task['masked_smiles'], task['candidate_completed_smiles']],
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def sample_id(property_name: str, condition: str, record_id: str) -> str:
    return f'{property_name}/{condition}/{record_id}'


def _manifest_paths(manifest_path: str | Path, property_name: str,
                    condition: str) -> tuple[Path, Path]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    by_id = {item['id']: item for item in manifest.get('items', [])}
    ids = (
        f'{property_name}_{condition}_tasks',
        f'{property_name}_{condition}_labels',
    )
    missing = [name for name in ids if name not in by_id]
    if missing:
        raise FileNotFoundError(
            f'{manifest_path} is missing MolP2SBench artifacts: {missing}')
    paths = tuple(Path(by_id[name]['path']) for name in ids)
    for name, path in zip(ids, paths):
        if not path.is_file():
            raise FileNotFoundError(f'{name} is not a regular file: {path}')
        expected = by_id[name].get('sha256')
        if expected:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f'{name} checksum mismatch: expected {expected}, got '
                    f'{actual}')
    return paths  # type: ignore[return-value]


def _local_paths(root: str | Path, property_name: str,
                 condition: str) -> tuple[Path, Path]:
    root = Path(root).expanduser()
    layouts = (
        (root / 'data' / property_name / f'{condition}.jsonl',
         root / 'data' / property_name / f'{condition}.labels.jsonl'),
        (root / property_name / f'{condition}.jsonl',
         root / property_name / f'{condition}.labels.jsonl'),
        (root / f'{property_name}_{condition}_tasks.jsonl',
         root / f'{property_name}_{condition}_labels.labels.jsonl'),
    )
    for task_path, label_path in layouts:
        if task_path.is_file() and label_path.is_file():
            return task_path, label_path
    required = layouts[0]
    raise FileNotFoundError(
        'MolP2SBench requires sibling task and label JSONL files; expected '
        f'{required[0]} and {required[1]}')


def _download_paths(property_name: str,
                    condition: str) -> tuple[Path, Path]:
    cache_base = os.environ.get('COMPASS_DATA_CACHE')
    if cache_base:
        cache = Path(cache_base).expanduser()
    else:
        cache = Path.home() / '.cache' / 'opencompass'
    target = (cache / 'MolP2SBench' / UPSTREAM_REVISION / 'data' /
              property_name)
    base = ('https://raw.githubusercontent.com/DUAL-Xiao/MolP2SBench/'
            f'{UPSTREAM_REVISION}/data/{property_name}')
    names = (f'{condition}.jsonl', f'{condition}.labels.jsonl')
    for name in names:
        download_url(f'{base}/{name}', str(target), filename=name)
    return target / names[0], target / names[1]


def _pair_paths(property_name: str, condition: str, path: str | None,
                manifest: str | None) -> tuple[Path, Path]:
    if manifest:
        return _manifest_paths(manifest, property_name, condition)
    if path:
        return _local_paths(path, property_name, condition)
    return _download_paths(property_name, condition)


def _expected_count(property_name: str, condition: str) -> int:
    if condition != 'counterfactual':
        return 136
    return 240 if property_name in _COUNTERFACTUAL_240 else 258


def load_native_records(
    property_name: str,
    conditions: str | Sequence[str] = 'correct',
    path: str | None = None,
    manifest: str | None = None,
    selected_ids: Sequence[str] | None = None,
    sample_limit: int | None = None,
    complete_families: bool = False,
) -> list[dict[str, Any]]:
    """Load validated rows in release order and render production prompts."""
    if property_name not in PROPERTIES:
        raise ValueError(f'unknown property: {property_name}')
    if isinstance(conditions, str):
        conditions = (conditions, )
    conditions = tuple(conditions)
    unknown = [item for item in conditions if item not in CONDITIONS]
    if unknown:
        raise ValueError(f'unknown conditions: {unknown}')

    records = []
    seen_ids = set()
    for condition in conditions:
        task_path, label_path = _pair_paths(property_name, condition, path,
                                            manifest)
        rows = validate_rows(
            load_jsonl(task_path),
            load_jsonl(label_path),
            source=f'{property_name} {condition}',
        )
        expected = _expected_count(property_name, condition)
        if len(rows) != expected:
            raise ValueError(
                f'{property_name} {condition} has {len(rows)} records; '
                f'expected {expected}')
        for task in rows:
            stable_id = sample_id(property_name, condition, task['record_id'])
            if stable_id in seen_ids:
                raise ValueError(f'duplicate sample id: {stable_id}')
            seen_ids.add(stable_id)
            atomic_id = family_id(task)
            system_prompt, user_prompt = render_prompt(task, property_name)
            candidates = list(task['candidate_completed_smiles'])
            reference = {
                'sample_id': stable_id,
                'record_id': task['record_id'],
                'property': property_name,
                'condition': condition,
                'family_id': atomic_id,
                'label': task['label'],
                'n_candidates': len(candidates),
            }
            records.append({
                'sample_id': stable_id,
                'record_id': task['record_id'],
                'property': property_name,
                'condition': condition,
                'family_id': atomic_id,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'candidate_completed_smiles': candidates,
                'reference': reference,
            })

    if complete_families:
        present = defaultdict(set)
        for row in records:
            present[row['family_id']].add(row['condition'])
        required = set(conditions)
        complete = {
            key for key, available in present.items() if required <= available
        }
        records = [row for row in records if row['family_id'] in complete]

    if selected_ids is not None:
        selected_ids = list(selected_ids)
        by_id = {row['sample_id']: row for row in records}
        absent = [item for item in selected_ids if item not in by_id]
        if absent:
            raise ValueError(f'unknown selected sample ids: {absent[:3]}')
        records = [by_id[item] for item in selected_ids]

    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError('sample_limit must be positive')
        grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for row in records:
            grouped.setdefault(row['family_id'], []).append(row)
        selected_families = set()
        count = 0
        for atomic_id, rows in grouped.items():
            selected_families.add(atomic_id)
            count += len(rows)
            if count >= sample_limit:
                break
        records = [
            row for row in records if row['family_id'] in selected_families
        ]
    return records


@LOAD_DATASET.register_module()
class MolP2SBenchDataset(BaseDataset):
    """OpenCompass dataset adapter for released MolP2SBench JSONL pairs."""

    @staticmethod
    def load(
        property_name: str,
        conditions: str | Sequence[str] = 'correct',
        path: str | None = None,
        manifest: str | None = None,
        selected_ids: Sequence[str] | None = None,
        sample_limit: int | None = None,
        complete_families: bool = False,
    ) -> DatasetDict:
        rows = load_native_records(
            property_name=property_name,
            conditions=conditions,
            path=path,
            manifest=manifest,
            selected_ids=selected_ids,
            sample_limit=sample_limit,
            complete_families=complete_families,
        )
        return DatasetDict(test=Dataset.from_list(rows))


def normalize_prediction(prediction: Any,
                         expected_record_id: str) -> dict[str, Any]:
    """Convert a model response to the official JSONL prediction shape."""
    parsed = prediction
    if isinstance(prediction, str):
        try:
            parsed = json.loads(prediction)
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, Mapping):
        return {'record_id': expected_record_id, 'invalid': True}
    if parsed.get('invalid') is True or 'selected_candidate_index' not in parsed:
        return {'record_id': expected_record_id, 'invalid': True}
    index = parsed['selected_candidate_index']
    if not isinstance(index, int) or isinstance(index, bool):
        return {'record_id': expected_record_id, 'invalid': True}
    record_id = parsed.get('record_id')
    if not isinstance(record_id, str) or not record_id:
        return {'record_id': expected_record_id, 'invalid': True}
    return {
        'record_id': record_id,
        'selected_candidate_index': index,
    }


def _reference_candidate_count(reference: Mapping[str, Any]) -> int:
    if 'n_candidates' in reference:
        return int(reference['n_candidates'])
    return len(reference['candidate_completed_smiles'])


def score_prediction_rows(
    prediction_rows: Iterable[Mapping[str, Any]],
    references: Iterable[Mapping[str, Any]],
    property_name: str,
    condition: str,
) -> dict[str, Any]:
    """Score normalized prediction objects exactly like ``score_tasks.py``."""
    if property_name not in PROPERTIES or condition not in CONDITIONS:
        raise ValueError(
            f'unknown property or condition: {property_name} {condition}')
    chosen: dict[str, int | None] = {}
    for row in prediction_rows:
        if not isinstance(row, Mapping) or 'record_id' not in row:
            raise ValueError('prediction is missing record_id')
        record_id = row['record_id']
        if record_id in chosen:
            raise ValueError(f'duplicate prediction: {record_id}')
        if row.get('invalid') is True or 'selected_candidate_index' not in row:
            chosen[record_id] = None
            continue
        index = row['selected_candidate_index']
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(
                f'selected_candidate_index is not an integer: {record_id}')
        chosen[record_id] = index

    hits = invalid = missing = out_of_range = 0
    details = []
    references = list(references)
    for reference in references:
        record_id = reference['record_id']
        label = reference['label']
        choice = chosen.get(record_id)
        if record_id not in chosen:
            missing += 1
            status = 'missing'
            correct = False
        elif choice is None:
            invalid += 1
            status = 'invalid'
            correct = False
        elif not 0 <= choice < _reference_candidate_count(reference):
            out_of_range += 1
            status = 'out_of_range'
            correct = False
        else:
            correct = choice == label
            hits += correct
            status = 'correct' if correct else 'incorrect'
        details.append({
            'sample_id': reference.get('sample_id', record_id),
            'record_id': record_id,
            'family_id': reference.get('family_id'),
            'condition': condition,
            'selected_candidate_index': choice,
            'label': label,
            'status': status,
            'correct': bool(correct),
        })
    n = len(references)
    return {
        'property': property_name,
        'condition': condition,
        'n': n,
        'hits': hits,
        'accuracy': hits / n if n else None,
        'missing': missing,
        'invalid': invalid,
        'out_of_range': out_of_range,
        'details': details,
    }


def score_rows(
    predictions: Iterable[Any],
    references: Iterable[Mapping[str, Any]],
    property_name: str,
    condition: str,
) -> dict[str, Any]:
    """Normalize paired model responses, then run the official accounting."""
    predictions = list(predictions)
    references = list(references)
    normalized = [
        normalize_prediction(prediction, reference['record_id'])
        for prediction, reference in zip(predictions, references)
    ]
    # OpenCompass normally supplies one prediction per reference. Preserve the
    # release scorer's missing path if a partial prediction list is supplied.
    return score_prediction_rows(normalized, references, property_name,
                                 condition)


def property_grounded_recovery(details: Iterable[Mapping[str, Any]]) -> float:
    """Compute the paper's equally weighted source/counterfactual PGR (%)."""
    families: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list))
    for detail in details:
        atomic_id = detail.get('family_id')
        condition = detail.get('condition')
        if atomic_id and condition in {'correct', 'counterfactual'}:
            families[str(atomic_id)][str(condition)].append(
                bool(detail['correct']))
    values = []
    for outcomes in families.values():
        source = outcomes.get('correct', [])
        counterfactual = outcomes.get('counterfactual', [])
        if len(source) == 1 and counterfactual:
            values.append(0.5 * float(source[0]) + 0.5 *
                          (sum(counterfactual) / len(counterfactual)))
    if not values:
        raise UnscorableMetricError(
            'PGR requires at least one complete source/counterfactual family')
    return 100.0 * sum(values) / len(values)


@ICL_EVALUATORS.register_module()
class MolP2SBenchEvaluator(BaseEvaluator):
    """Official view scorer plus the paper's complete-family PGR metric."""

    def __init__(self, property_name: str, conditions: Sequence[str] | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        if property_name not in PROPERTIES:
            raise ValueError(f'unknown property: {property_name}')
        self.property_name = property_name
        self.conditions = tuple(conditions) if conditions else None

    def score(self, predictions, references):
        references = list(references)
        predictions = list(predictions)
        conditions = self.conditions or tuple(
            OrderedDict.fromkeys(ref['condition'] for ref in references))
        all_details = []
        view_results = OrderedDict()
        for condition in conditions:
            positions = [
                index for index, reference in enumerate(references)
                if reference['condition'] == condition
            ]
            result = score_rows(
                [predictions[index] for index in positions],
                [references[index] for index in positions],
                self.property_name,
                condition,
            )
            all_details.extend(result.pop('details'))
            if len(conditions) == 1:
                view_results.update(result)
            else:
                for key, value in result.items():
                    view_results[f'{condition}/{key}'] = value
        if {'correct', 'counterfactual'} <= set(conditions):
            view_results['PGR (%)'] = property_grounded_recovery(all_details)
        view_results['details'] = all_details
        return view_results
