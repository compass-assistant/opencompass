"""OpenCompass integration for the Over-Editing benchmark.

The benchmark was released at https://github.com/nreHieW/over-editing
(revision 701ad34febd436729251826d5cbb16fca5e278da).  The prompt, response
extraction, complexity visitor, and execution verdict below follow that
release, with the two published source errata applied to the primary metrics.
"""

import ast
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset
from .sandbox import (check_isolated_dependencies,
                      isolated_untrusted_check_many)

UPSTREAM_REVISION = '701ad34febd436729251826d5cbb16fca5e278da'
QUESTIONS_URL = (
    'https://raw.githubusercontent.com/nreHieW/over-editing/'
    f'{UPSTREAM_REVISION}/data/questions/'
    'corrupted_solutions_manual_easy_400.jsonl')
BIGCODEBENCH_URL = (
    'https://huggingface.co/datasets/bigcode/bigcodebench/resolve/'
    'b74c0d0bf70d2c0bc459be537895cca163007f1a/data/'
    'v0.1.4-00000-of-00001.parquet')

GENERIC_SYSTEM_PROMPT = (
    'You are a Python Expert specializing in code analysis and debugging. '
    'When provided with a problem statement, your task is to fix the code.\n'
    'Do not change the function signature, default arguments, or docstring. '
    'Use the docstring to understand the requirements of the function.')


def create_user_message(problem_statement: str,
                        corrupted_solution: str,
                        is_explicit: bool = False) -> str:
    """Build the zero-shot, hidden-test user message from the release."""
    message = (
        'I am trying to implement a function with the following '
        f'specifications:\n{problem_statement}.\n\n'
        'The function I have written so far is:\n'
        f'{corrupted_solution} \n\n')
    if is_explicit:
        message += ('What is wrong? Fix and complete my function but keep as '
                    'much of the original code as possible.')
    else:
        message += 'What is wrong? Fix and complete my function.'
    return message + 'Wrap your response in ```python and ```'


def extract_code_from_response(response: str) -> str:
    """Apply the release's longest-fenced-block extraction rules."""
    response = response.strip()
    try:
        ast.parse(response)
        return response
    except Exception:
        pass

    if '</think>' in response:
        response = response.split('</think>')[-1]
    if '</thought>' in response:
        response = response.split('</thought>')[-1]
    closed = r'^```(?:python)?\s*\n([\s\S]*?)\n```'
    matches = re.findall(closed, response, re.IGNORECASE | re.MULTILINE)
    if matches:
        return max((code.strip() for code in matches), key=len)
    opened = r'^```(?:python)?\s*\n([\s\S]*)$'
    match = re.search(opened, response, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ''


def standardize_code_formatting(code: str) -> str:
    """Strip comments and normalize formatting using the release fallback."""
    if not code.strip():
        return code
    try:
        return ast.unparse(ast.parse(code))
    except Exception:
        try:
            tokens = [
                token for token in tokenize.generate_tokens(
                    io.StringIO(code).readline)
                if token.type != tokenize.COMMENT
            ]
            return tokenize.untokenize(tokens)
        except Exception:
            lines = code.splitlines()
            result_lines = []
            for line in lines:
                in_string = False
                quote_char = None
                index = 0
                while index < len(line):
                    char = line[index]
                    if not in_string and char in ['"', "'"]:
                        if (index + 2 < len(line)
                                and line[index:index + 3] in ['"""', "'''"]):
                            in_string = True
                            quote_char = line[index:index + 3]
                            index += 3
                            continue
                        in_string = True
                        quote_char = char
                    elif in_string and char == quote_char[0]:
                        if len(quote_char) == 3:
                            if line[index:index + 3] == quote_char:
                                in_string = False
                                quote_char = None
                                index += 3
                                continue
                        elif index == 0 or line[index - 1] != '\\':
                            in_string = False
                            quote_char = None
                    elif not in_string and char == '#':
                        line = line[:index].rstrip()
                        break
                    index += 1
                result_lines.append(line)
            return '\n'.join(result_lines)


def extract_function_body(code: str) -> str:
    """Extract the first function body using the upstream AST fallback."""
    code = standardize_code_formatting(code)
    try:
        tree = ast.parse(code)
        function = next((node for node in ast.walk(tree)
                         if isinstance(node, ast.FunctionDef)), None)
        if function is None:
            return _extract_function_body_regex(code)
        start = function.lineno
        if (function.body and isinstance(function.body[0], ast.Expr)
                and isinstance(function.body[0].value, ast.Constant)
                and isinstance(function.body[0].value.value, str)):
            if len(function.body) == 1:
                return ''
            start = function.body[1].lineno
        return '\n'.join(code.splitlines()[start - 1:])
    except Exception:
        return _extract_function_body_regex(code)


def _extract_function_body_regex(code: str) -> str:
    match = re.search(r'(def\s+[^:]+:\s*)\n', code)
    if not match:
        return code
    body = code[match.end():]
    for pattern in (r'^\s*"""[\s\S]*?"""\s*\n',
                    r"^\s*'''[\s\S]*?'''\s*\n"):
        docstring = re.search(pattern, body, re.DOTALL)
        if docstring:
            return body[docstring.end():]
    return body


def tokenize_code(code: str) -> List[str]:
    try:
        ignored = {
            tokenize.COMMENT, tokenize.NL, tokenize.ENCODING,
            tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT
        }
        return [token.string for token in tokenize.generate_tokens(
            io.StringIO(code).readline) if token.type not in ignored]
    except Exception:
        return code.split()


def _levenshtein(left: List[str], right: List[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1]
                               + (left_item != right_item)))
        previous = current
    return previous[-1]


def normalized_body_levenshtein(reference: str, prediction: str) -> float:
    left = tokenize_code(extract_function_body(reference))
    right = tokenize_code(extract_function_body(prediction))
    denominator = max(len(left), len(right))
    return _levenshtein(left, right) / denominator if denominator else 0.0


def compute_python_complexities(
        code: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Return cyclomatic and cognitive complexity using the source visitor."""
    if not code:
        return None, None
    try:
        tree = ast.parse(standardize_code_formatting(code))
    except Exception:
        try:
            tree = ast.parse(code)
        except Exception:
            return None, None

    class CombinedVisitor(ast.NodeVisitor):

        def __init__(self):
            self.cc = 1
            self.cog = 0
            self.nesting = 0

        def _add_cog(self, base=1):
            self.cog += base + self.nesting

        def _visit_block(self, nodes, inc_nesting=True):
            if not nodes:
                return
            if inc_nesting:
                self.nesting += 1
            for child in nodes:
                self.visit(child)
            if inc_nesting:
                self.nesting -= 1

        def visit_If(self, node):
            self.cc += 1
            self._add_cog()
            self._visit_block(node.body)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self.visit(node.orelse[0])
            else:
                self._visit_block(node.orelse)

        def visit_For(self, node):
            self.cc += 1
            self._add_cog()
            self._visit_block(node.body)
            self._visit_block(node.orelse)

        visit_AsyncFor = visit_For

        def visit_While(self, node):
            self.cc += 1
            self._add_cog()
            self._visit_block(node.body)
            self._visit_block(node.orelse)

        def visit_Try(self, node):
            self.cc += len(getattr(node, 'handlers', []))
            if getattr(node, 'finalbody', []):
                self.cc += 1
            self._visit_block(node.body)
            for handler in getattr(node, 'handlers', []):
                self._add_cog()
                self._visit_block(handler.body)
            self._visit_block(node.orelse)
            self._visit_block(node.finalbody)

        def visit_With(self, node):
            self._visit_block(node.body)

        visit_AsyncWith = visit_With

        def visit_BoolOp(self, node):
            increment = max(0, len(getattr(node, 'values', [])) - 1)
            self.cc += increment
            self.cog += increment
            self.generic_visit(node)

        def visit_IfExp(self, node):
            self.cc += 1
            self._add_cog()
            self.generic_visit(node)

        def visit_comprehension(self, node):
            self.cc += 1 + len(getattr(node, 'ifs', []))
            self._add_cog()
            for _ in getattr(node, 'ifs', []):
                self._add_cog()
            self.generic_visit(node)

        def visit_Assert(self, node):
            self.cc += 1
            self.generic_visit(node)

        def visit_Match(self, node):
            self.cc += len(getattr(node, 'cases', []))
            for case in getattr(node, 'cases', []):
                self._add_cog()
                self._visit_block(case.body)
            self.generic_visit(node)

    visitor = CombinedVisitor()
    visitor.visit(tree)
    return visitor.cc, visitor.cog


def _resolve_source(path: Optional[str], default_url: str,
                    expected_name: str) -> str:
    if path is None:
        return default_url
    path = str(path)
    if path.startswith(('https://', 'http://')):
        return path
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / expected_name
    if not candidate.is_file():
        raise FileNotFoundError(
            f'Over-Editing requires {expected_name}; not found at {candidate}')
    return str(candidate)


def _load_records(source: str, kind: str) -> List[dict]:
    if Path(source).is_file():
        if kind == 'json':
            with open(source, encoding='utf-8') as stream:
                return [json.loads(line) for line in stream if line.strip()]
        return load_dataset('parquet', data_files=source,
                            split='train').to_list()
    return load_dataset(kind, data_files=source, split='train').to_list()


def _validate_string_fields(row: dict, index: int, fields: Iterable[str],
                            record_kind: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f'{record_kind} row {index} must be an object')
    missing = set(fields).difference(row)
    if missing:
        raise ValueError(f'{record_kind} row {index} is missing '
                         f'{sorted(missing)}')
    invalid = [field for field in fields
               if not isinstance(row[field], str) or not row[field].strip()]
    if invalid:
        raise ValueError(f'{record_kind} row {index} has invalid non-empty '
                         f'string fields {invalid}')


def collect_execution_dependencies(rows: Iterable[dict]) -> List[str]:
    """Return imports required by the frozen programs and hidden tests."""
    modules = set()
    for row in rows:
        sample_id = str(row.get('task_id', '<unknown>'))
        for field in ('canonical_solution', 'corrupted_solution',
                      'official_test'):
            try:
                tree = ast.parse(row[field])
            except (KeyError, SyntaxError, TypeError) as error:
                raise ValueError(
                    f'Over-Editing {sample_id} has invalid Python in '
                    f'{field}') from error
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split('.')[0]
                                   for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module.split('.')[0])
    return sorted(modules)


@LOAD_DATASET.register_module()
class OverEditingDataset(BaseDataset):
    """The 400-task Over-Editing set joined to BigCodeBench v0.1.4 tests."""

    @staticmethod
    def load(path: Optional[str] = None,
             tests_path: Optional[str] = None,
             is_explicit: bool = False,
             expected_size: Optional[int] = 400,
             selected_ids: Optional[Iterable[str]] = None):
        question_source = _resolve_source(
            path, QUESTIONS_URL,
            'corrupted_solutions_manual_easy_400.jsonl')
        tests_source = _resolve_source(
            tests_path, BIGCODEBENCH_URL,
            'v0.1.4-00000-of-00001.parquet')
        questions = _load_records(question_source, 'json')
        tests = _load_records(tests_source, 'parquet')

        required = {
            'task_id', 'prompt', 'corrupted_solution', 'canonical_solution',
            'test_code'
        }
        for index, row in enumerate(questions):
            _validate_string_fields(row, index, required, 'question')
        ids = [str(row['task_id']) for row in questions]
        if len(ids) != len(set(ids)):
            raise ValueError('Over-Editing task_id values must be unique')
        if expected_size is not None and len(questions) != expected_size:
            raise ValueError(f'expected {expected_size} tasks, got '
                             f'{len(questions)}')

        for index, row in enumerate(tests):
            _validate_string_fields(row, index, {'task_id', 'test'},
                                    'BigCodeBench v0.1.4 test')
        test_ids = [str(row['task_id']) for row in tests]
        if len(test_ids) != len(set(test_ids)):
            raise ValueError('BigCodeBench v0.1.4 task_id values must be '
                             'unique')
        test_map = {str(row['task_id']): row for row in tests}
        missing_tests = [task_id for task_id in ids if task_id not in test_map]
        if missing_tests:
            raise ValueError('BigCodeBench v0.1.4 join is incomplete; missing '
                             + ', '.join(missing_tests[:5]))

        if selected_ids is not None:
            selected_ids = [str(item) for item in selected_ids]
            if len(selected_ids) != len(set(selected_ids)):
                raise ValueError('selected_ids must be unique')
            rows_by_id = {str(row['task_id']): row for row in questions}
            unknown = [item for item in selected_ids if item not in rows_by_id]
            if unknown:
                raise ValueError(f'unknown selected_ids: {unknown[:5]}')
            questions = [rows_by_id[item] for item in selected_ids]

        joined = []
        for row in questions:
            item = dict(row)
            item['task_id'] = str(item['task_id'])
            item['system_prompt'] = GENERIC_SYSTEM_PROMPT
            item['user_prompt'] = create_user_message(
                item['prompt'], item['corrupted_solution'], is_explicit)
            item['official_test'] = test_map[item['task_id']]['test']
            joined.append(item)
        dataset = Dataset.from_list(joined)
        return DatasetDict(train=dataset, test=dataset)


@ICL_EVALUATORS.register_module()
class OverEditingEvaluator(BaseEvaluator):
    """Functional correctness and passing-only edit-fidelity metrics."""

    def __init__(self,
                 execution_timeout: float = 240.0,
                 execution_workers: int = 8,
                 pred_postprocessor=None):
        super().__init__(pred_postprocessor=pred_postprocessor)
        self.execution_timeout = execution_timeout
        self.execution_workers = execution_workers

    def _score_one(self, prediction: str, row: dict,
                   execution_result=None) -> dict:
        extracted = extract_code_from_response(prediction)
        if extracted:
            if execution_result is None:
                raise RuntimeError('missing execution result')
            status, evidence = execution_result
        else:
            status, evidence = 'fail', {'ALL': 'No Python code extracted'}

        canonical_distance = normalized_body_levenshtein(
            row['corrupted_solution'], row['canonical_solution'])
        model_distance = normalized_body_levenshtein(
            row['corrupted_solution'], extracted)
        _, model_cognitive = compute_python_complexities(extracted)
        _, canonical_cognitive = compute_python_complexities(
            row['canonical_solution'])
        excess = model_distance - canonical_distance
        added = (model_cognitive - canonical_cognitive
                 if model_cognitive is not None
                 and canonical_cognitive is not None else None)
        return {
            'sample_id': str(row['task_id']),
            'status': status,
            'correct': status == 'pass',
            'prediction': prediction,
            'solution': extracted,
            'excess_levenshtein': excess,
            'added_cognitive_complexity': added,
            'details': evidence,
        }

    def score(self, predictions, references=None, test_set=None):
        if test_set is None:
            raise ValueError('OverEditingEvaluator requires test_set rows')
        rows = [dict(row) for row in test_set]
        if len(predictions) != len(rows):
            raise ValueError('predictions and test_set must have equal length')
        # Dependency failures are evaluator setup failures, not failed model
        # repairs. Probe the same isolated runtime before scoring any sample.
        check_isolated_dependencies(collect_execution_dependencies(rows))
        extracted = [extract_code_from_response(item) for item in predictions]
        executable = [
            (solution, row['official_test'])
            for solution, row in zip(extracted, rows) if solution
        ]
        verdicts = iter(isolated_untrusted_check_many(
            executable, self.execution_timeout))
        execution_results = [next(verdicts) if solution else None
                             for solution in extracted]
        details = [
            self._score_one(prediction, row, execution_result)
            for prediction, row, execution_result in zip(
                predictions, rows, execution_results)
        ]

        passing = [detail for detail in details
                   if detail['status'] == 'pass']
        pass_at_one = len(passing) / len(rows) if rows else 0.0
        excess_values = [detail['excess_levenshtein'] for detail in passing]
        added_values = [detail['added_cognitive_complexity']
                        for detail in passing
                        if detail['added_cognitive_complexity'] is not None]
        return {
            'Pass@1 ↑': pass_at_one,
            'Excess Lev. ↓': (sum(excess_values) / len(excess_values)
                                 if excess_values else None),
            'Added CC ↓': (sum(added_values) / len(added_values)
                              if added_values else None),
            'evaluated_problems': len(rows),
            'passed_problems': len(passing),
            'details': details,
        }
