from opencompass.datasets import WearableQADataset, WearableQAEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


wearableqa_reader_cfg = dict(
    input_columns=['input'],
    output_column='target',
    train_split='test',
    test_split='test',
)

wearableqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{input}')]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

wearableqa_eval_cfg = dict(
    evaluator=dict(type=WearableQAEvaluator),
)

wearableqa_datasets = [
    dict(
        abbr='wearableqa_cot_row',
        type=WearableQADataset,
        path='facebook/WearableQA',
        split='test',
        prompt_mode='cot',
        validate_full=True,
        reader_cfg=wearableqa_reader_cfg,
        infer_cfg=wearableqa_infer_cfg,
        eval_cfg=wearableqa_eval_cfg,
    )
]
