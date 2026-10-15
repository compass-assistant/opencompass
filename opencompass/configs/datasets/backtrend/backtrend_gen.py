from opencompass.datasets import BackTrendDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

backtrend_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference',
)

backtrend_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content='{prompt}')],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# The selected paper reproduction reports the three Judge-backed settings.
# BackTrendEvaluator itself defaults to all five official settings.
backtrend_eval_cfg = dict(
    evaluator=dict(
        type='BackTrendEvaluator',
        settings=(2, 4, 5),
        n_runs=3,
        coverage_k=10,
    ),
)

backtrend_datasets = [
    dict(
        abbr='backtrend',
        type=BackTrendDataset,
        path='opencompass/backtrend',
        reader_cfg=backtrend_reader_cfg,
        infer_cfg=backtrend_infer_cfg,
        eval_cfg=backtrend_eval_cfg,
    )
]
