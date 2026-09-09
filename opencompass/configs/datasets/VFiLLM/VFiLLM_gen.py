from opencompass.datasets import VFiLLMDataset, VFiLLMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

VFiLLM_prompt = '{prompt}'

VFiLLM_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content=VFiLLM_prompt)],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

VFiLLM_eval_cfg = dict(
    evaluator=dict(type=VFiLLMEvaluator, tolerance=0.1))

VFiLLM_datasets = [
    dict(
        type=VFiLLMDataset,
        abbr='V-FiLLM-10Q-d1-5',
        questions_path='data/V-FiLLM/random_questions_10q_d1-5.csv',
        atoms_path='data/V-FiLLM/atoms_10q.json',
        limit=250,
        reader_cfg=dict(
            input_columns=['prompt'],
            output_column='target',
            train_split='test',
            test_split='test',
        ),
        infer_cfg=VFiLLM_infer_cfg,
        eval_cfg=VFiLLM_eval_cfg,
    )
]
