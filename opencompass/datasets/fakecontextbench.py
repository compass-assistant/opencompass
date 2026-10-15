"""FakeContextBench-v6 dataset and deterministic whole-answer evaluator.

The scoring behavior follows the public ``fakecontextbench-v6-whole-answer-v2``
implementation at commit ``0c1018016d2e21157ae2b701de48ad4f69c24e9a``.

The scorer-derived portions are Copyright (c) 2026 Pei-lin Li and used under
the MIT License: permission is granted, free of charge, to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies, provided that the
copyright and permission notice are included. The software is provided
"AS IS", without warranty of any kind, express or implied.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from datasets import Dataset, DownloadManager

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

PROTOCOL = 'fakecontextbench-v6-whole-answer-v2'
DATA_URL = (
    'https://raw.githubusercontent.com/peilin717/FakeContext-Bench/'
    '0c1018016d2e21157ae2b701de48ad4f69c24e9a/'
    'fakecontextbench-v6.jsonl')
BASE_NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+'
SIGNED_NUMBER = rf'[-+]?(?:{BASE_NUMBER})(?:[eE][-+]?\d+)?'
NUMERIC_TOKEN = re.compile(
    rf'(?<![\w./]){SIGNED_NUMBER}(?:\s*/\s*{SIGNED_NUMBER})?%?(?![\w./])')
SINGLE_CHOICE = re.compile(r'^[A-Z]$')
ANSWER_CUE = re.compile(
    r'(?i)\b(?:final\s+answer|answer|conclusion)\b|最终答案|答案|所以|因此|故')


def _validate_and_prepare(rows):
    prepared = []
    seen = set()
    for index, row in enumerate(rows):
        case_id = row.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f'row {index} has no non-empty case_id')
        if case_id in seen:
            raise ValueError(f'duplicate case_id: {case_id}')
        seen.add(case_id)

        messages = row.get('messages')
        if (not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], dict)
                or messages[0].get('role') != 'user'
                or not isinstance(messages[0].get('content'), str)):
            raise ValueError(f'{case_id} must contain exactly one user message')

        gold = row.get('gold')
        required = {'real_world_answer', 'context_induced_answer'}
        if not isinstance(gold, dict) or not required.issubset(gold):
            raise ValueError(f'{case_id} has malformed gold answers')
        if (not str(gold['real_world_answer']).strip()
                or not str(gold['context_induced_answer']).strip()
                or gold['real_world_answer'] == gold['context_induced_answer']):
            raise ValueError(f'{case_id} has invalid registered answers')
        prepared.append(dict(row))
    return prepared


@LOAD_DATASET.register_module()
class FakeContextBenchDataset(BaseDataset):
    """Load the released JSONL without altering its messages or order."""

    @staticmethod
    def load(path=DATA_URL, split='train', sample_ids=None, **kwargs):
        del split, kwargs
        local_path = DownloadManager().download(path)
        with open(local_path, encoding='utf-8') as handle:
            loaded = [json.loads(line) for line in handle if line.strip()]
        rows = _validate_and_prepare(loaded)
        if sample_ids is not None:
            by_id = {row['case_id']: row for row in rows}
            missing = [sample_id for sample_id in sample_ids
                       if sample_id not in by_id]
            if missing:
                raise ValueError(f'unknown sample_ids: {missing[:3]}')
            rows = [by_id[sample_id] for sample_id in sample_ids]

        # ``metadata.query_parameters`` is a dict in some released domains
        # and a list in others. Apache Arrow cannot represent that union and
        # its JSON reader segfaults on the pinned artifact. This field is
        # provenance only (prompts and scoring use ``messages`` and ``gold``),
        # so retain it losslessly as canonical JSON in the native Dataset.
        columns = list(dict.fromkeys(key for row in rows for key in row))
        arrow_rows = []
        for row in rows:
            item = {key: row.get(key) for key in columns}
            metadata = item.get('metadata')
            if isinstance(metadata, dict):
                metadata = dict(metadata)
                parameters = metadata.get('query_parameters')
                if parameters is not None:
                    metadata['query_parameters'] = json.dumps(
                        parameters, ensure_ascii=False, sort_keys=True,
                        separators=(',', ':'))
                item['metadata'] = metadata
            arrow_rows.append(item)
        return Dataset.from_list(arrow_rows)


def prepare(text):
    value = unicodedata.normalize('NFKC', str(text or ''))
    return (value.replace('−', '-').replace('–', '-').replace(
        '—', '-').replace('÷', '/').casefold())


def norm(text):
    value = prepare(text).strip()
    value = re.sub(
        r'(?i)^\s*(?:final\s+answer|answer|答案)\s*(?:is|=|:|：)?\s*',
        '', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip('`*_ $[]{}，。,.!！?？:：;；')


def parse_number(text):
    value = prepare(text).strip().replace(' ', '').replace(',', '')
    value = value.strip('$`*_=()[]{}，。,.!！?？:：;；')
    percent = value.endswith('%')
    if percent:
        value = value[:-1]
    try:
        if value.count('/') == 1:
            numerator, denominator = value.split('/', 1)
            result = Fraction(Decimal(numerator)) / Fraction(
                Decimal(denominator))
        elif '/' not in value:
            result = Fraction(Decimal(value))
        else:
            return None
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return result / 100 if percent else result


def numeric_equal(observed, target, tolerance):
    if observed == target:
        return True
    if tolerance <= 0:
        return False
    try:
        return math.isclose(
            float(observed), float(target), abs_tol=tolerance, rel_tol=0.0)
    except (OverflowError, ValueError):
        return False


def choice_hits(text, value):
    letter = value.upper()
    patterns = [
        r'(?i)(?:final\s+answer|answer|option|答案|选项)\s*(?:is|=|:|：)?\s*\(?([A-Z])\)?',
        r'(?<![A-Za-z0-9])\(([A-Z])\)(?![A-Za-z0-9])',
        r'^\s*([A-Z])(?:[.、:)\s]|$)',
    ]
    return [
        match.start(1) for pattern in patterns
        for match in re.finditer(pattern, text, re.MULTILINE)
        if match.group(1).upper() == letter
    ]


def phrase_hits(text, value):
    candidate = norm(value)
    if not candidate:
        return []
    escaped = re.escape(candidate).replace(r'\ ', r'\s+')
    left = r'(?<![\w])' if candidate[0].isalnum() or candidate[0] == '_' else ''
    right = r'(?![\w])' if candidate[-1].isalnum() or candidate[-1] == '_' else ''
    return [m.start() for m in re.finditer(left + escaped + right, prepare(text))]


def candidate_hits(text, value, response_format, tolerance):
    raw_value = str(value).strip()
    if SINGLE_CHOICE.fullmatch(raw_value):
        return choice_hits(text, raw_value), 'choice_whole_token'

    target_number = parse_number(raw_value)
    if target_number is not None and response_format != 'short_text':
        positions = []
        for match in NUMERIC_TOKEN.finditer(prepare(text)):
            observed = parse_number(match.group())
            if observed is not None and numeric_equal(
                    observed, target_number, tolerance):
                positions.append(match.start())
        return positions, 'numeric_whole_token'
    return phrase_hits(text, raw_value), 'phrase_whole_value'


def whole_response_equals(text, value, response_format, tolerance):
    cleaned = norm(text)
    target = norm(value)
    if cleaned == target and target:
        return True
    target_number = parse_number(value)
    observed_number = parse_number(cleaned)
    return (response_format != 'short_text' and target_number is not None
            and observed_number is not None
            and numeric_equal(observed_number, target_number, tolerance))


def final_region(text):
    stripped = text.strip()
    if not stripped:
        return ''
    parts = [
        part.strip()
        for part in re.split(r'(?<=[.!?。！？])\s+|\n+', stripped)
        if part.strip()
    ]
    cue_index = None
    for index, part in enumerate(parts):
        if ANSWER_CUE.search(part):
            cue_index = index
    if cue_index is not None:
        return '\n'.join(parts[cue_index:])
    return '\n'.join(parts[-3:])


def classify(text, gold):
    real = gold['real_world_answer']
    context = gold['context_induced_answer']
    response_format = gold.get('response_format', 'numeric_or_symbolic')
    tolerance = float(gold.get('tolerance', 0) or 0)

    exact_real = whole_response_equals(text, real, response_format, tolerance)
    exact_context = whole_response_equals(text, context, response_format,
                                          tolerance)
    if exact_real and not exact_context:
        return 'reality_correct', real, {
            'method': 'exact_whole_response_reality'
        }
    if exact_context and not exact_real:
        return 'context_follow', context, {
            'method': 'exact_whole_response_context'
        }

    real_hits, real_mode = candidate_hits(text, real, response_format,
                                          tolerance)
    context_hits, context_mode = candidate_hits(text, context,
                                                response_format, tolerance)
    base_audit = {
        'protocol': PROTOCOL,
        'response_format': response_format,
        'real_match_mode': real_mode,
        'context_match_mode': context_mode,
        'full_real_hits': len(real_hits),
        'full_context_hits': len(context_hits),
    }
    if real_hits and not context_hits:
        return 'reality_correct', real, {
            **base_audit, 'method': 'full_only_reality'
        }
    if context_hits and not real_hits:
        return 'context_follow', context, {
            **base_audit, 'method': 'full_only_context'
        }
    if not real_hits and not context_hits:
        return 'other', None, {**base_audit, 'method': 'no_complete_target'}

    tail = final_region(text)
    tail_real, _ = candidate_hits(tail, real, response_format, tolerance)
    tail_context, _ = candidate_hits(tail, context, response_format,
                                     tolerance)
    audit = {
        **base_audit,
        'tail_real_hits': len(tail_real),
        'tail_context_hits': len(tail_context),
    }
    if tail_real and not tail_context:
        return 'reality_correct', real, {
            **audit, 'method': 'both_full_tail_reality'
        }
    if tail_context and not tail_real:
        return 'context_follow', context, {
            **audit, 'method': 'both_full_tail_context'
        }
    if tail_real and tail_context:
        if tail_real[-1] > tail_context[-1]:
            return 'reality_correct', real, {
                **audit, 'method': 'both_tail_last_reality'
            }
        return 'context_follow', context, {
            **audit, 'method': 'both_tail_last_context'
        }
    if real_hits[-1] > context_hits[-1]:
        return 'reality_correct', real, {
            **audit, 'method': 'both_full_last_reality'
        }
    return 'context_follow', context, {
        **audit, 'method': 'both_full_last_context'
    }


@ICL_EVALUATORS.register_module()
class FakeContextBenchEvaluator(BaseEvaluator):
    """Deterministic v2 scorer with paper-table percentage aliases."""

    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError('predictions and references must have equal length')
        counts = Counter()
        methods = Counter()
        details = []
        for prediction_text, gold in zip(predictions, references):
            label, prediction, audit = classify(prediction_text or '', gold)
            audit = {'protocol': PROTOCOL, **audit}
            counts[label] += 1
            methods[audit['method']] += 1
            details.append({
                'label': label,
                'prediction': prediction,
                'scoring_audit': audit,
            })

        n = len(details)
        reality = counts['reality_correct']
        context = counts['context_follow']
        rates = {
            key: counts[key] / n if n else 0
            for key in ('reality_correct', 'context_follow', 'other')
        }
        reality_rate = (reality / (reality + context)
                        if reality + context else None)
        return {
            'protocol': PROTOCOL,
            'n': n,
            'Reality': rates['reality_correct'] * 100,
            'Context': rates['context_follow'] * 100,
            'Other': rates['other'] * 100,
            'Reality Rate': (reality_rate * 100
                             if reality_rate is not None else None),
            'counts': dict(counts),
            'rates': rates,
            'R_over_RplusC': reality_rate,
            'mixed': 0,
            'scoring_methods': dict(methods),
            'validation': {
                'whole_answer_numeric_matching': True,
                'fraction_components_scanned_separately': False,
                'short_text_numeric_fallback': False,
                'unique_case_ids': n,
            },
            'details': details,
        }
