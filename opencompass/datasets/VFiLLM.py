"""V-FiLLM 10-Q dataset, prompt construction, and scoring."""

import csv
import json
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


_BS_SECTION = {
    'cash_and_cash_equivalents': 'Current Assets',
    'short_term_investments': 'Current Assets',
    'accounts_receivable_net': 'Current Assets',
    'inventories': 'Current Assets',
    'prepaid_expenses_and_other': 'Current Assets',
    'long_term_marketable_securities': 'Non-Current Assets',
    'property_plant_and_equipment_net': 'Non-Current Assets',
    'goodwill': 'Non-Current Assets',
    'other_non_current_assets': 'Non-Current Assets',
    'accounts_payable': 'Current Liabilities',
    'deferred_revenue_current': 'Current Liabilities',
    'accrued_expenses_and_other': 'Current Liabilities',
    'current_portion_of_long_term_debt': 'Current Liabilities',
    'long_term_debt': 'Non-Current Liabilities',
    'other_non_current_liabilities': 'Non-Current Liabilities',
    'common_stock_and_additional_paid_in_capital': 'Shareholders Equity',
    'retained_earnings': 'Shareholders Equity',
    'accumulated_other_comprehensive_income_loss': 'Shareholders Equity',
    'total_current_assets': 'Current Assets',
    'total_non_current_assets': 'Non-Current Assets',
    'total_assets': 'Total Assets',
    'total_current_liabilities': 'Current Liabilities',
    'total_non_current_liabilities': 'Non-Current Liabilities',
    'total_liabilities': 'Total Liabilities',
    'total_shareholders_equity': 'Shareholders Equity',
}
_BS_SECTION_ORDER = [
    'Current Assets', 'Non-Current Assets', 'Total Assets',
    'Current Liabilities', 'Non-Current Liabilities', 'Total Liabilities',
    'Shareholders Equity',
]
_BS_ITEM_ORDER = {
    'Current Assets': [
        'cash_and_cash_equivalents', 'short_term_investments',
        'accounts_receivable_net', 'inventories',
        'prepaid_expenses_and_other', 'total_current_assets',
    ],
    'Non-Current Assets': [
        'long_term_marketable_securities',
        'property_plant_and_equipment_net', 'goodwill',
        'other_non_current_assets', 'total_non_current_assets',
    ],
    'Total Assets': ['total_assets'],
    'Current Liabilities': [
        'accounts_payable', 'deferred_revenue_current',
        'accrued_expenses_and_other',
        'current_portion_of_long_term_debt', 'total_current_liabilities',
    ],
    'Non-Current Liabilities': [
        'long_term_debt', 'other_non_current_liabilities',
        'total_non_current_liabilities',
    ],
    'Total Liabilities': ['total_liabilities'],
    'Shareholders Equity': [
        'common_stock_and_additional_paid_in_capital', 'retained_earnings',
        'accumulated_other_comprehensive_income_loss',
        'total_shareholders_equity',
    ],
}

_OPS_SECTION = {
    'total_revenues': 'Revenue',
    'cost_of_sales': 'Cost of Sales',
    'gross_profit': 'Gross Profit',
    'research_and_development': 'Operating Expenses',
    'selling_general_and_administrative': 'Operating Expenses',
    'total_costs_and_expenses': 'Total Costs and Expenses',
    'operating_income': 'Operating Income',
    'other_income_expense_net': 'Other Income / Expense',
    'provision_for_income_taxes': 'Income Tax',
    'net_income': 'Net Income',
    'earnings_per_share_basic': 'Earnings Per Share',
    'earnings_per_share_diluted': 'Earnings Per Share',
    'shares_used_basic': 'Shares Used in Computing EPS',
    'shares_used_diluted': 'Shares Used in Computing EPS',
}
_OPS_SECTION_ORDER = [
    'Revenue', 'Cost of Sales', 'Gross Profit', 'Operating Expenses',
    'Total Costs and Expenses', 'Operating Income',
    'Other Income / Expense', 'Income Tax', 'Net Income',
    'Earnings Per Share', 'Shares Used in Computing EPS',
]
_OPS_ITEM_ORDER = {
    'Gross Profit': ['gross_profit'],
    'Operating Expenses': [
        'research_and_development',
        'selling_general_and_administrative',
    ],
    'Total Costs and Expenses': ['total_costs_and_expenses'],
    'Operating Income': ['operating_income'],
    'Other Income / Expense': ['other_income_expense_net'],
    'Income Tax': ['provision_for_income_taxes'],
    'Net Income': ['net_income'],
    'Earnings Per Share': [
        'earnings_per_share_basic', 'earnings_per_share_diluted',
    ],
    'Shares Used in Computing EPS': ['shares_used_basic', 'shares_used_diluted'],
}

_CF_SECTION = {
    'depreciation_and_amortization': 'Operating Activities',
    'stock_based_compensation': 'Operating Activities',
    'change_in_accounts_receivable': 'Operating Activities',
    'change_in_inventories': 'Operating Activities',
    'change_in_accounts_payable': 'Operating Activities',
    'change_in_other_working_capital': 'Operating Activities',
    'net_cash_from_operating': 'Operating Activities',
    'capital_expenditures': 'Investing Activities',
    'purchases_of_marketable_securities': 'Investing Activities',
    'proceeds_from_maturities_of_securities': 'Investing Activities',
    'net_cash_from_investing': 'Investing Activities',
    'repayments_of_debt': 'Financing Activities',
    'share_repurchases': 'Financing Activities',
    'proceeds_from_stock_option_exercises': 'Financing Activities',
    'dividends_paid': 'Financing Activities',
    'net_cash_from_financing': 'Financing Activities',
    'effect_of_exchange_rate_on_cash': 'Net Change in Cash',
    'opening_cash_and_equivalents': 'Net Change in Cash',
    'net_change_in_cash': 'Net Change in Cash',
}
_CF_SECTION_ORDER = [
    'Operating Activities', 'Investing Activities', 'Financing Activities',
    'Net Change in Cash',
]
_CF_ITEM_ORDER = {
    'Operating Activities': [
        'depreciation_and_amortization', 'stock_based_compensation',
        'change_in_accounts_receivable', 'change_in_inventories',
        'change_in_accounts_payable', 'change_in_other_working_capital',
        'net_cash_from_operating',
    ],
    'Investing Activities': [
        'capital_expenditures', 'purchases_of_marketable_securities',
        'proceeds_from_maturities_of_securities', 'net_cash_from_investing',
    ],
    'Financing Activities': [
        'repayments_of_debt', 'share_repurchases',
        'proceeds_from_stock_option_exercises', 'dividends_paid',
        'net_cash_from_financing',
    ],
    'Net Change in Cash': [
        'effect_of_exchange_rate_on_cash', 'opening_cash_and_equivalents',
        'net_change_in_cash',
    ],
}
_CF_CONCEPTS = set(_CF_SECTION)

_CI_SECTION = {
    'foreign_currency_translation': 'Other Comprehensive Income',
    'unrealized_gains_losses_on_securities': 'Other Comprehensive Income',
    'total_other_comprehensive_income': 'Total OCI',
    'comprehensive_income': 'Total Comprehensive Income',
}
_CI_SECTION_ORDER = [
    'Other Comprehensive Income', 'Total OCI', 'Total Comprehensive Income',
]
_CI_ITEM_ORDER = {
    'Other Comprehensive Income': [
        'foreign_currency_translation',
        'unrealized_gains_losses_on_securities',
    ],
    'Total OCI': ['total_other_comprehensive_income'],
    'Total Comprehensive Income': ['comprehensive_income'],
}
_CI_CONCEPTS = set(_CI_SECTION)


def _parse_period_date(period):
    date_str = period.split('Ended ')[-1].strip(
    ) if 'Ended ' in period else period.strip()
    for date_format in ('%B %d, %Y', '%B %Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, date_format).date()
        except ValueError:
            continue
    return datetime.min.date()


def _atom_section(atom, is_balance_sheet):
    concept = atom['concept']
    parent = atom.get('parent_concept') or ''
    if is_balance_sheet:
        return _BS_SECTION.get(concept, 'Other')
    if concept in _CF_CONCEPTS:
        return _CF_SECTION[concept]
    if concept in _CI_CONCEPTS:
        return _CI_SECTION[concept]
    if concept in _OPS_SECTION:
        return _OPS_SECTION[concept]
    if parent == 'total_revenues':
        return 'Revenue'
    if parent == 'cost_of_sales':
        return 'Cost of Sales'
    return 'Other'


def _sort_section_items(items, section, is_balance_sheet):
    if is_balance_sheet:
        order = _BS_ITEM_ORDER.get(section, [])
    elif section in _CF_ITEM_ORDER:
        order = _CF_ITEM_ORDER[section]
    elif section in _CI_ITEM_ORDER:
        order = _CI_ITEM_ORDER[section]
    elif section in _OPS_ITEM_ORDER:
        order = _OPS_ITEM_ORDER[section]
    elif section == 'Revenue':
        return sorted(
            items,
            key=lambda atom: (atom['concept'] == 'total_revenues',
                              atom.get('label') or atom['concept']))
    elif section == 'Cost of Sales':
        return sorted(
            items,
            key=lambda atom: (atom['concept'] == 'cost_of_sales',
                              atom.get('label') or atom['concept']))
    else:
        order = []

    def sort_key(atom):
        try:
            return (order.index(atom['concept']),
                    atom.get('label') or atom['concept'])
        except ValueError:
            return (len(order), atom.get('label') or atom['concept'])

    return sorted(items, key=sort_key)


def _render_sections(lines, period_atoms, section_order, item_order_map,
                     is_balance_sheet, indent):
    by_section = defaultdict(list)
    for atom in period_atoms:
        by_section[_atom_section(atom, is_balance_sheet)].append(atom)

    ordered = [section for section in section_order if section in by_section]
    remaining = [
        section for section in by_section if section not in section_order
    ]
    for section in ordered + remaining:
        lines.append(f'{indent}[{section}]')
        for atom in _sort_section_items(by_section[section], section,
                                        is_balance_sheet):
            unit = f" ({atom['unit']})" if atom.get('unit') else ''
            display = atom.get('value_display') or (
                atom['value'] if atom.get('value') is not None else '')
            lines.append(
                f"{indent}  {atom.get('label') or atom['concept']}{unit}: "
                f'{display}')


def sheet_to_text_10q(atoms, leaf_only=False):
    """Render atoms in the official 10-Q statement order."""
    entity = None
    ops_by_period = defaultdict(list)
    bs_by_period = defaultdict(list)
    cf_by_period = defaultdict(list)
    ci_by_period = defaultdict(list)

    for atom in atoms:
        if entity is None:
            entity = atom.get('entity')
        if leaf_only and atom.get('role') == 'derived':
            continue
        period = atom['period']
        if 'Ended' not in period:
            bs_by_period[period].append(atom)
        elif atom['concept'] in _CF_CONCEPTS:
            cf_by_period[period].append(atom)
        elif atom['concept'] in _CI_CONCEPTS:
            ci_by_period[period].append(atom)
        else:
            ops_by_period[period].append(atom)

    lines = []
    if entity:
        lines.append(f'  Entity: {entity}')
        lines.append('')
    if ops_by_period:
        lines.append('  === Income Statement ===')
        for period in sorted(ops_by_period, key=_parse_period_date):
            lines.append(f'    {period}:')
            _render_sections(lines, ops_by_period[period],
                             _OPS_SECTION_ORDER, _OPS_ITEM_ORDER, False,
                             '      ')
    if bs_by_period:
        lines.append('')
        lines.append('  === Balance Sheet ===')
        for period in sorted(bs_by_period, key=_parse_period_date):
            lines.append(f'    {period}:')
            _render_sections(lines, bs_by_period[period],
                             _BS_SECTION_ORDER, _BS_ITEM_ORDER, True,
                             '      ')
    if cf_by_period:
        lines.append('')
        lines.append('  === Cash Flow Statement ===')
        for period in sorted(cf_by_period, key=_parse_period_date):
            lines.append(f'    {period}:')
            _render_sections(lines, cf_by_period[period], _CF_SECTION_ORDER,
                             _CF_ITEM_ORDER, False, '      ')
    if ci_by_period:
        lines.append('')
        lines.append('  === Comprehensive Income Statement ===')
        for period in sorted(ci_by_period, key=_parse_period_date):
            lines.append(f'    {period}:')
            _render_sections(lines, ci_by_period[period], _CI_SECTION_ORDER,
                             _CI_ITEM_ORDER, False, '      ')
    return '\n'.join(lines)


def build_prompt(sheet_text, question):
    """Build the exact single-turn V-FiLLM user prompt."""
    return (
        'ROLE: Financial analyst. Answer using ONLY the data below. '
        'No commentary.\n\n'
        '=== DATA ===\n'
        f'{sheet_text}\n'
        '=== END ===\n\n'
        f'QUESTION: {question}\n\n'
        'RULES:\n'
        '- Extract only the numbers you need.\n'
        '- Show calculations in 1–3 lines max if needed.\n'
        '- Last line MUST be: Answer: <value>\n'
        '- <value> is either a number, True, or False. Nothing else.\n'
        '- Do NOT explain, summarize, or add anything after the Answer line.'
        '\n\n'
        '- Do NOT show your thinking process.\n\n'
        'SOLVE NOW:')


def build_10q_sheet_lookup(atoms):
    lookup = {}
    for atom in atoms:
        lookup.setdefault(atom['entity'], []).append(atom)
    return lookup


def extract_number(text):
    """Apply the official Answer-line and last-number parser."""
    match = re.search(r'Answer\s*:\s*(.+)', text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip().rstrip('.,;')
        if raw.lower() == 'true':
            return True
        if raw.lower() == 'false':
            return False
        cleaned = re.sub(r'(\d),(\d)', r'\1\2', raw)
        number_match = re.search(
            r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned)
        if number_match:
            return float(number_match.group())
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
    matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned)
    return float(matches[-1]) if matches else None


def is_correct(predicted, ground_truth, tolerance=0.1):
    """Apply the official boolean, zero, and relative-tolerance rules."""
    if predicted is None or ground_truth is None:
        return False
    if isinstance(predicted, bool) or isinstance(ground_truth, bool):
        return predicted == ground_truth
    if isinstance(ground_truth, str):
        try:
            ground_truth = float(ground_truth.replace(',', ''))
        except ValueError:
            return False
    if ground_truth == 0:
        return abs(predicted) < 1e-6
    return abs(predicted - ground_truth) / abs(ground_truth) <= tolerance


@LOAD_DATASET.register_module()
class VFiLLMDataset(BaseDataset):
    """Load the selected mixed-depth 10-Q V-FiLLM split."""

    @staticmethod
    def load(questions_path, atoms_path, limit=250):
        questions_path = osp.expanduser(questions_path)
        atoms_path = osp.expanduser(atoms_path)
        if not osp.isfile(questions_path):
            raise FileNotFoundError(
                f'V-FiLLM questions file not found: {questions_path}')
        if not osp.isfile(atoms_path):
            raise FileNotFoundError(
                f'V-FiLLM atoms file not found: {atoms_path}')

        with open(questions_path, newline='', encoding='utf-8') as file_handle:
            questions = list(csv.DictReader(file_handle))
        with open(atoms_path, encoding='utf-8') as file_handle:
            atoms = json.load(file_handle)
        if not isinstance(atoms, list):
            raise ValueError('V-FiLLM atoms file must contain a JSON array')
        selected = questions[:limit] if limit else questions
        sheet_lookup = build_10q_sheet_lookup(atoms)

        rows = []
        for index, question in enumerate(selected):
            entity = question.get('leaf_1_entity', '')
            entity_atoms = sheet_lookup.get(entity, [])
            skipped = not entity_atoms
            sheet_text = '' if skipped else sheet_to_text_10q(entity_atoms)
            prompt = '' if skipped else build_prompt(
                sheet_text, question.get('question', ''))
            rows.append({
                'sample_id': question.get('question_id') or str(index + 1),
                'prompt': prompt,
                'question': question.get('question', ''),
                'target': {
                    'answer': question.get('answer'),
                    'depth': question.get('depth'),
                    'entity': entity,
                    'skip': skipped,
                },
            })
        return Dataset.from_list(rows)


@ICL_EVALUATORS.register_module()
class VFiLLMEvaluator(BaseEvaluator):
    """Deterministic V-FiLLM parser and relative-tolerance evaluator."""

    def __init__(self, tolerance=0.1):
        super().__init__()
        self.tolerance = tolerance

    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError(
                'V-FiLLM predictions and references have different lengths')
        details = []
        for prediction, reference in zip(predictions, references):
            if isinstance(reference, dict):
                ground_truth = reference.get('answer')
                skipped = bool(reference.get('skip', False))
            else:
                ground_truth = reference
                skipped = False
            parsed = None if skipped else extract_number(prediction)
            correct = False if skipped else is_correct(
                parsed, ground_truth, self.tolerance)
            details.append({
                'predicted': parsed,
                'ground_truth': ground_truth,
                'correct': correct,
                'skip': skipped,
            })
        return self.aggregate(details)

    @staticmethod
    def aggregate(details):
        evaluated = [detail for detail in details if not detail.get('skip')]
        if not evaluated:
            raise ValueError('V-FiLLM has no non-skipped samples to score')
        correct = sum(detail['correct'] for detail in evaluated)
        return {
            'total': len(evaluated),
            'skipped': len(details) - len(evaluated),
            'correct': correct,
            'accuracy': correct / len(evaluated),
            'recovered': 0,
        }
