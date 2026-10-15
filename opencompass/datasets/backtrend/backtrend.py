"""BackTrend dataset and evidence-backed evaluators.

The integration follows the executable implementation at
https://github.com/rebeccaz4/BackTrend/tree/fda6a30d52c82e8a10fb377c8ce0ab884f00469d.
"""

from __future__ import annotations

import json
import hashlib
import re
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import build_model_from_cfg, get_data_path

from ..base import BaseDataset

DOMAIN = 'Artificial intelligence and machine learning'
YEAR_RANGE = '2019-2023'
DIRECTIONS = ('problem', 'solution')
DATA_FILE_SHA256 = {
    'topics.json': 'c17732e2dabbcfded2e088bb02deb3b13688efa20e0868e1666efba7bb74ca7b',
    'weak_signal.json': 'd68688783863a8d464cd0e398e80c9ef6a76331ab39024fee3c234292c6bcf58',
}

_INTRO = """You are an expert analyst of frontier {domain} research. Your task is to identify early weak signals that later contributed to a specified mature target topic.

We distinguish two categories of weak signals. One category is the \"{other}-space weak signal\": {other_definition}

However, the category you are asked to identify here is the \"{space}-space weak signal\": {definition}
To be more specific, {space}-space weak signals include {includes} that emerged in the literature during the prediction window, rather than claims tied to a single paper.

Mature target topic:
[{mainframe_topic}]

Prediction window:
[{year_range}]

Retrospective setup:
The mature target topic should be treated as a topic that is already established or prominent by 2024. Your task is not to predict after 2024. Your task is to look backward and identify what this 2024 mature topic looked like in the 2019-2023 literature while it was still emerging.
Use the mature target topic only as 2024 relevance context. The weak signals themselves must be {window_includes} that appeared in [{year_range}]. Do not use 2024-or-later evidence, and do not simply restate the mature target topic unless you name a more specific predecessor formulation.

Question:
What are the early {space}-space weak signals in {domain} that emerged between [{year_range}] and later contributed to the mature target topic [{mainframe_topic}]?

Specificity guidance:
- Use the mature target topic only as relevance context.
- Do not output the target topic itself unless you name a more specific reusable subtopic or predecessor direction.
- Too broad: a whole field (e.g., \"machine learning\", \"computer vision\"), broad model family (e.g., \"deep learning\"), or generic category label (e.g., \"optimization\").
- Too specific: a paper-specific method name, system name, exact dataset, benchmark instance, implementation detail, single experimental finding, or single case study.
- Correct level: {correct_level}
- Focus on the {focus}.
- Do not phrase signals as actions, paper contributions, or problem-solution relationships.
- Avoid \"X for Y\" signal names when X is a method and Y is a problem, goal, task, or desired property. For {space}-space output, keep only the {keep_side} when it is independently reusable and relevant.

Requirements:
- Return ONLY valid JSON, no markdown fences, no explanation.
- Output a JSON object with a \"weak_signals\" array. Order the \"weak_signals\" array from most to least confident: the array order is your ranking, strongest first.
- Output at least 10 weak signals unless you genuinely cannot name that many.
- Each weak signal must have exactly these fields: \"signal\", \"what_it_was\", \"why_weak_signal\".
- Each weak signal must be explicitly tied to the prediction window [{year_range}].
- \"what_it_was\" must include the year or year range within [{year_range}].
- \"why_weak_signal\" must explain why this was a {space}-space weak signal for [{mainframe_topic}].
- Each signal must be conceptually related to [{mainframe_topic}].
- Each signal must be reusable across multiple papers.
- Do not include {exclude}.
- Do not invent evidence or overclaim certainty.

Return only JSON with this schema:
{{
  \"weak_signals\": [
    {{
      \"signal\": \"<short reusable {space}-space weak signal name>\",
      \"what_it_was\": \"<1-2 sentences describing what it was, including the year>\",
      \"why_weak_signal\": \"<1-2 sentences explaining why it was a {space}-space weak signal for [{mainframe_topic}]>\"
    }}
  ]
}}
"""

_PROMPT_ARGUMENTS = {
    'problem': dict(
        other='solution',
        other_definition=('an early research method, technique, or design '
                          'principle that was not yet widely adopted but later '
                          'became an important solution to an already-recognized problem. '),
        definition=('an underrecognized research problem or problem formulation '
                    'that was not yet widely recognized by the research community '
                    'at the time it emerged, but that later became central to the '
                    'mature target topic [{mainframe_topic}].'),
        includes=('research problems, gaps, limitations, risks, bottlenecks, '
                  'evaluation failures, or scientific questions'),
        window_includes=('research problems, gaps, limitations, risks, '
                         'bottlenecks, or scientific questions'),
        correct_level=('a reusable research direction or problem space topic that '
                       'multiple independent papers could study using different '
                       'methods or systems.'),
        focus=('research problem, not on specific implementation details or data '
               'modalities'),
        keep_side='problem side (i.e., Y)',
        exclude='solution methods',
    ),
    'solution': dict(
        other='problem',
        other_definition=('an underrecognized research problem or problem '
                          'formulation that was not yet widely recognized by the '
                          'research community at the time it emerged, but that later '
                          'became central to the mature target topic '
                          '[{mainframe_topic}].'),
        definition=('an early research method, technique, or design principle '
                    'that was not yet widely adopted but later became an important '
                    'solution to an already-recognized problem.'),
        includes=('research methods, method families, system directions, '
                  'evaluation approaches, defenses, or solution directions'),
        window_includes=('research methods, method families, system directions, '
                         'evaluation approaches, defenses, or solution directions'),
        correct_level=('a reusable research method, method family, system direction, '
                       'evaluation approach, defense, or solution direction that '
                       'multiple independent papers could study.'),
        focus=('research method or solution direction, not on specific '
               'implementation details, problem formulations, or data modalities'),
        keep_side='solution side (i.e., X)',
        exclude='problem statements',
    ),
}

JUDGE_SYSTEM_PROMPT = (
    'You are a research topic matcher. Two research topics match only if they '
    'denote the same specific research topic: the same core method or the same '
    'problem, differing at most in wording, phrasing, or abbreviation. Topics '
    'that are merely related, adjacent, complementary, or from the same broad '
    'area do not match.')

SET_USER_TEMPLATE = """Compare the PREDICTED set against the GROUND TRUTH set of research topics.

Precision = fraction of predicted topics that match some ground-truth topic.
Recall = fraction of ground-truth topics that match some predicted topic.

Ground truth:
{gt_block}

Predicted:
{pred_block}

Return ONLY valid JSON:
{{"precision": <float 0-1>, "recall": <float 0-1>, "matched_pairs": [{{"gt_index": <int>, "pred_index": <int>}}]}}
Give one matched_pairs entry per matching (ground-truth, predicted) pair; empty list if none."""

SIGNAL_USER_TEMPLATE = """For each candidate research topic, decide whether it matches any reference research topic.

Reference:
{ref_block}

Candidate:
{cand_block}

Return ONLY valid JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements, 1 if the candidate matches any reference else 0."""


def make_topic_slug(topic: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', topic.lower()).strip('_')


def build_prompt(space: str, domain: str, mainframe_topic: str) -> str:
    if space not in DIRECTIONS:
        raise ValueError(f'Unsupported BackTrend direction: {space!r}')
    values = dict(_PROMPT_ARGUMENTS[space])
    values.update(space=space, domain=domain, mainframe_topic=mainframe_topic,
                  year_range=YEAR_RANGE)
    # Some definition fragments contain the topic placeholder themselves.
    values['definition'] = values['definition'].format(**values)
    values['other_definition'] = values['other_definition'].format(**values)
    return _INTRO.format(**values)


def strip_reasoning_blocks(text: str) -> str:
    if not isinstance(text, str):
        return ''
    if not re.search(r'<think\b[^>]*>', text, flags=re.IGNORECASE):
        return text.strip()
    parts = re.split(r'</think\s*>', text, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else ''


def load_weak_signal_json(text: str) -> Optional[Dict[str, Any]]:
    clean_text = strip_reasoning_blocks(text)
    if not clean_text:
        return None

    def valid(value):
        return isinstance(value, dict) and isinstance(
            value.get('weak_signals'), list)

    try:
        data = json.loads(clean_text)
        if valid(data):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', clean_text)
    if match:
        try:
            data = json.loads(match.group(1))
            if valid(data):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', clean_text):
        try:
            data, _ = decoder.raw_decode(clean_text[match.start():])
        except json.JSONDecodeError:
            continue
        if valid(data):
            return data
    return None


def extract_candidate_signals(text: str) -> List[str]:
    payload = load_weak_signal_json(text)
    if payload is not None:
        return [
            signal for item in payload['weak_signals'] if isinstance(item, dict)
            if (signal := (item.get('signal') or '').strip())
        ]
    clean_text = strip_reasoning_blocks(text)
    if not clean_text:
        return []
    pattern = re.compile(r'["“]signal["”]\s*:\s*["“]([^"“”]+)["”]')
    signals = [value.strip() for value in pattern.findall(clean_text)
               if value.strip()]
    for line in clean_text.splitlines():
        match = re.match(r'^\s*\d+[.)]+\s+(.+?)\s*$', line)
        if match and match.group(1).strip():
            signals.append(match.group(1).strip())
    return list(dict.fromkeys(signals))


def _resolve_data_files(path: str, topics_file: Optional[str],
                        signals_file: Optional[str], local_mode: bool):
    verify_snapshot = not (topics_file or signals_file)
    if topics_file or signals_file:
        if not topics_file or not signals_file:
            raise ValueError('topics_file and signals_file must be provided together')
        topics_path, signals_path = Path(topics_file), Path(signals_file)
    else:
        root = Path(get_data_path(path, local_mode=local_mode))
        if not root.is_dir():
            raise ValueError('BackTrend path must be a directory unless both '
                             'files are provided explicitly')
        topics_path = root / 'construction' / 'topics.json'
        signals_path = root / 'construction' / 'weak_signal.json'
    missing = [str(p) for p in (topics_path, signals_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing BackTrend data file(s): ' + ', '.join(missing))
    if verify_snapshot:
        for data_path in (topics_path, signals_path):
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            expected = DATA_FILE_SHA256[data_path.name]
            if digest != expected:
                raise ValueError(f'BackTrend snapshot checksum mismatch: {data_path}')
    return topics_path, signals_path


@LOAD_DATASET.register_module()
class BackTrendDataset(BaseDataset):
    """Load the canonical 25 topics as 50 ordered direction samples."""

    @staticmethod
    def load(path: str = 'opencompass/backtrend',
             topics_file: Optional[str] = None,
             signals_file: Optional[str] = None,
             local_mode: bool = False,
             sample_ids: Optional[Sequence[str]] = None,
             **kwargs) -> Dataset:
        topics_path, signals_path = _resolve_data_files(
            path, topics_file, signals_file, local_mode)
        topics = json.loads(topics_path.read_text(encoding='utf-8'))
        references = json.loads(signals_path.read_text(encoding='utf-8'))
        if not isinstance(topics, dict) or not isinstance(references, dict):
            raise ValueError('BackTrend data files must contain JSON objects')
        requested = None if sample_ids is None else set(map(str, sample_ids))
        rows = []
        for topic in topics:
            if not isinstance(topic, str):
                raise ValueError('BackTrend topic keys must be strings')
            spaces = references.get(topic, {})
            if not isinstance(spaces, dict):
                raise ValueError(f'Invalid reference object for topic {topic!r}')
            for direction in DIRECTIONS:
                sample_id = f'{make_topic_slug(topic)}--{direction}'
                if requested is not None and sample_id not in requested:
                    continue
                gt = spaces.get(direction, [])
                if not isinstance(gt, list) or not all(
                        isinstance(value, str) for value in gt):
                    raise ValueError(
                        f'Invalid {direction} references for topic {topic!r}')
                rows.append(dict(
                    sample_id=sample_id,
                    topic=topic,
                    topic_slug=make_topic_slug(topic),
                    direction=direction,
                    prompt=build_prompt(direction, DOMAIN, topic),
                    reference=json.dumps(gt, ensure_ascii=False),
                    gt_signals=gt,
                ))
        if requested is not None:
            found = {row['sample_id'] for row in rows}
            unknown = requested - found
            if unknown:
                raise ValueError('Unknown BackTrend sample IDs: ' +
                                 ', '.join(sorted(unknown)))
        return Dataset.from_list(rows)


def _format_block(signals: Sequence[str]) -> str:
    return '\n'.join(f'{i}. {signal}' for i, signal in enumerate(signals))


def build_set_prompt(gt: Sequence[str], pred: Sequence[str]) -> str:
    return SET_USER_TEMPLATE.format(gt_block=_format_block(gt),
                                    pred_block=_format_block(pred))


def build_pairwise_prompt(reference: Sequence[str],
                          candidates: Sequence[str]) -> str:
    return SIGNAL_USER_TEMPLATE.format(
        ref_block=_format_block(reference),
        cand_block=_format_block(candidates),
        n_cand=len(candidates))


def build_judge_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [dict(role='system', content=JUDGE_SYSTEM_PROMPT),
            dict(role='user', content=user_prompt)]


def _safe_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r'\{[\s\S]*\}', text or '')
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _normalize_pairs(raw_pairs: Any, gt: Sequence[str],
                     pred: Sequence[str]) -> List[dict]:
    pairs = []
    if not isinstance(raw_pairs, list):
        return pairs
    for entry in raw_pairs:
        if not isinstance(entry, dict):
            continue
        gi, pi = entry.get('gt_index'), entry.get('pred_index')
        if (isinstance(gi, int) and isinstance(pi, int)
                and 0 <= gi < len(gt) and 0 <= pi < len(pred)):
            pairs.append(dict(gt_index=gi, pred_index=pi,
                              ground_truth=gt[gi], predicted=pred[pi]))
    return pairs


def _parse_set_reply(text: str, gt: Sequence[str], pred: Sequence[str]) -> dict:
    payload = _safe_json(text)
    if payload is None:
        raise ValueError('Could not parse set-judge JSON')
    p_raw, r_raw = payload.get('precision'), payload.get('recall')
    if (not isinstance(p_raw, (int, float))
            or not isinstance(r_raw, (int, float))):
        raise ValueError('Set-judge precision and recall must be numeric')
    if not 0 <= float(p_raw) <= 1 or not 0 <= float(r_raw) <= 1:
        raise ValueError('Set-judge precision and recall must be in [0, 1]')
    pairs = _normalize_pairs(payload.get('matched_pairs'), gt, pred)
    matched_preds = {pair['pred_index'] for pair in pairs}
    matched_gts = {pair['gt_index'] for pair in pairs}
    precision = len(matched_preds) / len(pred) if pred else 0.0
    recall = len(matched_gts) / len(gt) if gt else 0.0
    denom = precision + recall
    return dict(precision=round(precision, 4), manual_p=round(precision, 4),
                recall=round(recall, 4), manual_r=round(recall, 4),
                f1=round(2 * precision * recall / denom if denom else 0.0, 4),
                matched_pairs=pairs)


def _parse_matches(text: str, expected: int) -> List[int]:
    payload = _safe_json(text)
    matches = None if payload is None else payload.get('matches')
    if not isinstance(matches, list) or len(matches) != expected:
        raise ValueError(f'Expected {expected} binary matches, got {matches!r}')
    try:
        result = [int(value) for value in matches]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Non-binary matches: {matches!r}') from exc
    if not all(value in (0, 1) for value in result):
        raise ValueError(f'Non-binary matches: {result!r}')
    return result


def _compute_metrics(precision_matches: Sequence[int],
                     recall_matches: Sequence[int], n_gt: int,
                     n_pred: int) -> dict:
    p_count, r_count = sum(precision_matches), sum(recall_matches)
    precision = p_count / n_pred if n_pred else 0.0
    recall = r_count / n_gt if n_gt else 0.0
    denom = precision + recall
    return dict(precision=round(precision, 4), recall=round(recall, 4),
                f1=round(2 * precision * recall / denom if denom else 0.0, 4),
                precision_matched=p_count, recall_matched=r_count,
                n_ext=n_pred, n_gt=n_gt)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    return (round(statistics.mean(values), 4),
            round(statistics.stdev(values) if len(values) > 1 else 0.0, 4))


def _summarize(records: Sequence[dict]) -> dict:
    scored = [item for item in records if item.get('status') == 'scored']
    result = dict(n_total=len(records), n_scored=len(scored),
                  n_no_gt=sum(item.get('status') == 'no_gt' for item in records))
    for metric in ('precision', 'recall', 'f1', 'coverage', 'coverage_full'):
        values = [item[metric] for item in scored
                  if isinstance(item.get(metric), (int, float))]
        if values:
            result[f'{metric}_mean'], result[f'{metric}_std'] = _mean_std(values)
        elif metric in ('precision', 'recall', 'f1'):
            result[f'{metric}_mean'] = None
            result[f'{metric}_std'] = None
    return result


@ICL_EVALUATORS.register_module()
class BackTrendEvaluator(BaseEvaluator):
    """Official five-setting evaluator with target-native Judge calls."""

    SETTING_NAMES = {
        1: 'set_bertscore', 2: 'set_llm', 3: 'signal_bertscore',
        4: 'signal_llm', 5: 'coverage_at_10'}

    def __init__(self, judge_model_cfg: Optional[dict] = None,
                 settings: Sequence[int] = (1, 2, 3, 4, 5),
                 n_runs: int = 3, coverage_k: int = 10,
                 batch_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        invalid = set(settings) - set(self.SETTING_NAMES)
        if invalid:
            raise ValueError(f'Unknown BackTrend settings: {sorted(invalid)}')
        self.settings = tuple(settings)
        self.n_runs = n_runs
        self.coverage_k = coverage_k
        self.batch_size = batch_size
        self.judge_model_cfg = deepcopy(judge_model_cfg)
        self.judge_model = None

    def _build_judge_model(self):
        if self.judge_model is None:
            if self.judge_model_cfg is None:
                raise ValueError('judge_model_cfg is required for settings 2, 4, and 5')
            self.judge_model = build_model_from_cfg(self.judge_model_cfg)

    def _judge(self, messages: List[List[dict]], max_out_len: int) -> List[str]:
        if not messages:
            return []
        self._build_judge_model()
        return self.judge_model.generate(messages, max_out_len=max_out_len,
                                         temperature=1.0)

    def _score_set_llm(self, work: List[dict]):
        requests, owners = [], []
        for item in work:
            if item['status'] != 'scored' or not item['pred']:
                continue
            for run in range(self.n_runs):
                requests.append(build_judge_messages(build_set_prompt(
                    item['gt'], item['pred'])))
                owners.append((item, run))
        replies = self._judge(requests, 1536)
        grouped = {id(item): [] for item in work}
        for reply, (item, _) in zip(replies, owners):
            grouped[id(item)].append(_parse_set_reply(reply, item['gt'], item['pred']))
        for item in work:
            if item['status'] != 'scored':
                continue
            if not item['pred']:
                item['set_llm'] = dict(setting='set_llm', precision=0.0,
                    manual_p=0.0, recall=0.0, manual_r=0.0, f1=0.0,
                    note='empty_pred', matched_pairs=[])
                continue
            runs = grouped[id(item)]
            p, ps = _mean_std([run['precision'] for run in runs])
            r, rs = _mean_std([run['recall'] for run in runs])
            f, fs = _mean_std([run['f1'] for run in runs])
            item['set_llm'] = dict(setting='set_llm', precision=p, manual_p=p,
                precision_std=ps, recall=r, manual_r=r, recall_std=rs,
                f1=f, f1_std=fs, n_runs=self.n_runs,
                n_pred=len(item['pred']), n_gt=len(item['gt']), runs=runs)

    def _direction_requests(self, work: List[dict], coverage: bool = False):
        requests, owners = [], []
        for item in work:
            if item['status'] != 'scored' or not item['pred']:
                continue
            directions = []
            if coverage:
                directions.append(('topk', item['pred'][:self.coverage_k], item['gt']))
                if len(item['pred']) > self.coverage_k:
                    directions.append(('full', item['pred'], item['gt']))
            else:
                directions.extend((('precision', item['gt'], item['pred']),
                                   ('recall', item['pred'], item['gt'])))
            for run in range(self.n_runs):
                for label, reference, candidates in directions:
                    for start in range(0, len(candidates), self.batch_size):
                        batch = candidates[start:start + self.batch_size]
                        requests.append(build_judge_messages(
                            build_pairwise_prompt(reference, batch)))
                        owners.append((item, run, label, len(batch)))
        return requests, owners

    def _score_signal_llm(self, work: List[dict]):
        requests, owners = self._direction_requests(work)
        replies = self._judge(requests, 1024)
        flags = {}
        for reply, (item, run, label, size) in zip(replies, owners):
            flags.setdefault((id(item), run, label), []).extend(
                _parse_matches(reply, size))
        for item in work:
            if item['status'] != 'scored':
                continue
            if not item['pred']:
                item['signal_llm'] = dict(setting='signal_llm', precision=0.0,
                    recall=0.0, f1=0.0, note='empty_pred')
                continue
            runs = [_compute_metrics(flags[(id(item), run, 'precision')],
                                     flags[(id(item), run, 'recall')],
                                     len(item['gt']), len(item['pred']))
                    for run in range(self.n_runs)]
            p, ps = _mean_std([run['precision'] for run in runs])
            r, rs = _mean_std([run['recall'] for run in runs])
            f, fs = _mean_std([run['f1'] for run in runs])
            item['signal_llm'] = dict(setting='signal_llm', precision=p,
                precision_std=ps, recall=r, recall_std=rs, f1=f, f1_std=fs,
                n_runs=self.n_runs, n_pred=len(item['pred']), n_gt=len(item['gt']))

    def _score_coverage(self, work: List[dict]):
        requests, owners = self._direction_requests(work, coverage=True)
        replies = self._judge(requests, 1024)
        flags = {}
        for reply, (item, run, label, size) in zip(replies, owners):
            flags.setdefault((id(item), run, label), []).extend(
                _parse_matches(reply, size))
        for item in work:
            if item['status'] != 'scored':
                continue
            name = f'coverage_at_{self.coverage_k}'
            if not item['pred']:
                item[name] = dict(setting=name, coverage=0.0,
                                  coverage_full=0.0, note='empty_pred')
                continue
            top_runs = [flags[(id(item), run, 'topk')]
                        for run in range(self.n_runs)]
            if len(item['pred']) > self.coverage_k:
                full_runs = [flags[(id(item), run, 'full')]
                             for run in range(self.n_runs)]
            else:
                full_runs = top_runs
            ratios = lambda values: [sum(run) / len(run) if run else 0.0
                                     for run in values]
            cov, cov_std = _mean_std(ratios(top_runs))
            full, full_std = _mean_std(ratios(full_runs))
            hits = [sum(run[index] for run in top_runs)
                    for index in range(len(item['gt']))]
            item[name] = dict(setting=name, coverage=cov,
                coverage_std=cov_std, coverage_full=full,
                coverage_full_std=full_std, k=self.coverage_k,
                truncated=len(item['pred']) > self.coverage_k,
                n_runs=self.n_runs, n_pred=len(item['pred']),
                n_gt=len(item['gt']), gt_hits_across_runs=hits)

    @staticmethod
    def _score_bertscore(work: List[dict], setting: int):
        try:
            from bert_score import score as bert_score
        except ImportError as exc:
            raise RuntimeError('bert-score is required for BackTrend settings 1 and 3') from exc
        for item in work:
            if item['status'] != 'scored':
                continue
            pred, gt = item['pred'], item['gt']
            name = 'set_bertscore' if setting == 1 else 'signal_bertscore'
            if not pred:
                item[name] = dict(setting=name, precision=0.0, recall=0.0,
                                  f1=0.0, note='empty_pred')
                continue
            if setting == 1:
                p, r, f = bert_score([' [SEP] '.join(pred)],
                    [' [SEP] '.join(gt)], lang='en', model_type='roberta-large',
                    rescale_with_baseline=False, verbose=False)
                values = [p.tolist()[0], r.tolist()[0], f.tolist()[0]]
            else:
                n, m = len(pred), len(gt)
                _, _, flat = bert_score([p for p in pred for _ in gt],
                    [g for _ in pred for g in gt], lang='en',
                    model_type='roberta-large', rescale_with_baseline=False,
                    verbose=False)
                matrix = [flat.tolist()[i * m:(i + 1) * m] for i in range(n)]
                p_val = sum(max(row) for row in matrix) / n
                r_val = sum(max(matrix[i][j] for i in range(n))
                            for j in range(m)) / m
                values = [p_val, r_val,
                          2 * p_val * r_val / (p_val + r_val)
                          if p_val + r_val else 0.0]
            result = dict(setting=name, precision=round(values[0], 4),
                          recall=round(values[1], 4), f1=round(values[2], 4))
            if setting == 3:
                result.update(n_pred=len(pred), n_gt=len(gt))
            item[name] = result

    def score(self, predictions: List[str], references: Optional[List] = None,
              test_set: Optional[Dataset] = None) -> Dict:
        if test_set is None:
            raise ValueError('test_set is required for BackTrend evaluation')
        if len(predictions) != len(test_set):
            raise ValueError('BackTrend prediction count must equal dataset size')
        work = []
        for prediction, row in zip(predictions, test_set):
            gt = row['gt_signals']
            if not isinstance(gt, list):
                gt = json.loads(row['reference'])
            work.append(dict(sample_id=str(row['sample_id']), topic=row['topic'],
                             direction=row['direction'], gt=list(gt),
                             pred=extract_candidate_signals(prediction),
                             raw_prediction=prediction,
                             status='scored' if gt else 'no_gt'))
        if 1 in self.settings:
            self._score_bertscore(work, 1)
        if 2 in self.settings:
            self._score_set_llm(work)
        if 3 in self.settings:
            self._score_bertscore(work, 3)
        if 4 in self.settings:
            self._score_signal_llm(work)
        if 5 in self.settings:
            self._score_coverage(work)

        output = {}
        for split, selected in (
                ('all', work),
                ('problem', [item for item in work if item['direction'] == 'problem']),
                ('solution', [item for item in work if item['direction'] == 'solution'])):
            for setting in self.settings:
                name = (f'coverage_at_{self.coverage_k}' if setting == 5
                        else self.SETTING_NAMES[setting])
                records = []
                for item in selected:
                    record = dict(status=item['status'])
                    if item['status'] == 'scored':
                        record.update(item[name])
                    records.append(record)
                for key, value in _summarize(records).items():
                    output[f'{split}/{name}/{key}'] = value
        details = []
        for item in work:
            detail = dict(sample_id=item['sample_id'], topic=item['topic'],
                          direction=item['direction'], status=item['status'],
                          gt_signals=item['gt'], pred_signals=item['pred'])
            for setting in self.settings:
                name = (f'coverage_at_{self.coverage_k}' if setting == 5
                        else self.SETTING_NAMES[setting])
                if name in item:
                    detail[name] = item[name]
            details.append(detail)
        output['details'] = details
        return output
