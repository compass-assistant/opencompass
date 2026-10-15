from opencompass.datasets import OverEditingDataset, OverEditingEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

over_editing_reader_cfg = dict(
    input_columns=['system_prompt', 'user_prompt'],
    output_column='canonical_solution',
)

over_editing_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[dict(role='SYSTEM', prompt='{system_prompt}')],
            round=[dict(role='HUMAN', prompt='{user_prompt}')],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64000),
)

over_editing_eval_cfg = dict(
    evaluator=dict(type=OverEditingEvaluator),
    pred_role='BOT',
)

over_editing_datasets = [
    dict(
        abbr='over_editing_400',
        type=OverEditingDataset,
        path=None,
        tests_path=None,
        is_explicit=False,
        expected_size=400,
        reader_cfg=over_editing_reader_cfg,
        infer_cfg=over_editing_infer_cfg,
        eval_cfg=over_editing_eval_cfg,
    )
]
