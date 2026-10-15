"""RegExtract full-text Joint Extraction evaluation configuration."""

from opencompass.datasets import RegExtractDataset, RegExtractEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


regextract_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference',
)

regextract_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

regextract_eval_cfg = dict(
    evaluator=dict(type=RegExtractEvaluator),
)

regextract_datasets = [
    dict(
        type=RegExtractDataset,
        abbr='regextract',
        path='opencompass/regextract',
        reader_cfg=regextract_reader_cfg,
        infer_cfg=regextract_infer_cfg,
        eval_cfg=regextract_eval_cfg,
    )
]
