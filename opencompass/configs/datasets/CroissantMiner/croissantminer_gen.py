from opencompass.datasets import (CroissantMinerDataset,
                                  CroissantMinerEvaluator,
                                  CroissantMinerPromptTemplate,
                                  strip_reasoning_prefix)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever


croissantminer_reader_cfg = dict(
    input_columns=['paper_text', 'sample_id'],
    output_column='reference',
)

croissantminer_infer_cfg = dict(
    prompt_template=dict(type=CroissantMinerPromptTemplate),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

# OpenCompass convention is for a top-level experiment config to inject the
# runtime Judge model into this placeholder (see the other ``*_llmjudge``
# dataset configs). Keeping the transport out of the dataset config makes the
# indexed config portable while exposing the normal evaluation route.
croissantminer_judge_model_cfg = dict()

croissantminer_datasets = [
    dict(
        abbr='CroissantMiner-test',
        type=CroissantMinerDataset,
        path='data/croissantminer',
        split='test',
        reader_cfg=croissantminer_reader_cfg,
        infer_cfg=croissantminer_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=CroissantMinerEvaluator,
                judge_model_cfg=croissantminer_judge_model_cfg,
            ),
            pred_postprocessor=dict(type=strip_reasoning_prefix),
            pred_role='BOT',
        ),
    )
]
