"""Zero-shot native OpenCompass configuration for MolP2SBench."""

from opencompass.datasets.molp2sbench import (MolP2SBenchDataset,
                                              MolP2SBenchEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


# Keep iteration inputs local so MMEngine can parse this file as a lazy base
# config while the registered component types remain lazy import references.
_PROPERTIES = (
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
_SINGLE_PROPERTIES = _PROPERTIES[:11]
_CONDITIONS = ('correct', 'shuffled', 'counterfactual')


molp2sbench_reader_cfg = dict(
    input_columns=['system_prompt', 'user_prompt'],
    output_column='reference',
    train_split='test',
    test_split='test',
)

molp2sbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[dict(role='SYSTEM', prompt='{system_prompt}')],
            round=[dict(role='HUMAN', prompt='{user_prompt}')],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


def make_molp2sbench_dataset(property_name,
                             conditions='correct',
                             complete_families=False,
                             **loader_kwargs):
    """Build one registered dataset configuration.

    ``loader_kwargs`` is intentionally available for an operator's local
    release directory, a materialized manifest, or native smoke selection.
    """
    condition_names = ([conditions]
                       if isinstance(conditions, str) else list(conditions))
    suffix = '-'.join(condition_names)
    return dict(
        abbr=f'molp2sbench_{property_name}_{suffix}',
        type=MolP2SBenchDataset,
        property_name=property_name,
        conditions=condition_names,
        complete_families=complete_families,
        reader_cfg=molp2sbench_reader_cfg,
        infer_cfg=molp2sbench_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=MolP2SBenchEvaluator,
                property_name=property_name,
                conditions=condition_names,
            ),
        ),
        **loader_kwargs,
    )


# The normal catalog exposes every independently scored release view.
molp2sbench_datasets = [
    make_molp2sbench_dataset(property_name, condition)
    for property_name in _PROPERTIES for condition in _CONDITIONS
]

# Primary-paper PGR configurations retain only complete source/CF families.
molp2sbench_pgr_datasets = [
    make_molp2sbench_dataset(
        property_name,
        conditions=['correct', 'counterfactual'],
        complete_families=True,
    ) for property_name in _SINGLE_PROPERTIES
]
