from opencompass.datasets.fakecontextbench import (
    DATA_URL, FakeContextBenchDataset, FakeContextBenchEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

fakecontextbench_reader_cfg = dict(
    input_columns=['messages'],
    output_column='gold',
)

fakecontextbench_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(expand_column='messages')],
        format_variables=False,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

fakecontextbench_eval_cfg = dict(
    evaluator=dict(type=FakeContextBenchEvaluator),
)

fakecontextbench_datasets = [
    dict(
        abbr='fakecontextbench-v6',
        type=FakeContextBenchDataset,
        path=DATA_URL,
        reader_cfg=fakecontextbench_reader_cfg,
        infer_cfg=fakecontextbench_infer_cfg,
        eval_cfg=fakecontextbench_eval_cfg,
    )
]
