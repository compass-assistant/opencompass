"""General365 public test split, official zero-shot hybrid evaluation."""

from opencompass.datasets import General365Dataset, General365Evaluator
from opencompass.models import OpenAISDK
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

general365_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)

general365_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content='{question}')],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

general365_judge_cfg = dict(
    type=OpenAISDK,
    # Released scorer default. Paper-result reproduction uses GPT-4.1.
    path='gpt-4.1-mini',
    key='ENV',
    openai_api_base='https://api.openai.com/v1/',
    max_out_len=8192,
    max_seq_len=65536,
    batch_size=1024,
    temperature=0.0,
    retry=1,
    timeout=3600,
    max_workers=16,
    query_per_second=16,
    openai_extra_kwargs=dict(
        stream=False,
        response_format=dict(type='json_object'),
    ),
)

general365_datasets = [
    dict(
        type=General365Dataset,
        abbr='General365',
        path='meituan-longcat/General365_Public',
        revision='e6f063d0a74c51baa101204533823dd54b0c577b',
        split='test',
        reader_cfg=general365_reader_cfg,
        infer_cfg=general365_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=General365Evaluator,
                judge_cfg=general365_judge_cfg,
            ),
        ),
    )
]
