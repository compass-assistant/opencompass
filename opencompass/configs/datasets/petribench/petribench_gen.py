"""PetriBench v1 zero-shot generation configuration."""

from opencompass.datasets import (PetriBenchDataset, PetriBenchEvaluator,
                                  petribench_output_postprocessor)
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


def get_petribench_datasets(path=None, difficulties=('easy', 'medium', 'hard'),
                            sample_ids=None):
    """Build native dataset configs, optionally for an operator archive."""
    difficulties = tuple(difficulties)
    reader_cfg = dict(input_columns=['system_prompt', 'prompt'],
                      output_column='answer')
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[dict(role='SYSTEM', prompt='{system_prompt}')],
                round=[dict(role='HUMAN', prompt='{prompt}')],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=ParallelGenInferencer),
    )
    eval_cfg = dict(
        evaluator=dict(type=PetriBenchEvaluator),
        pred_postprocessor=dict(type=petribench_output_postprocessor),
    )
    suffix = difficulties[0] if len(difficulties) == 1 else 'all'
    return [
        dict(
            type=PetriBenchDataset,
            abbr=f'petribench-v1-{suffix}',
            path=path,
            difficulties=difficulties,
            sample_ids=sample_ids,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        )
    ]


petribench_datasets = get_petribench_datasets()
