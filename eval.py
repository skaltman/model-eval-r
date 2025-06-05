from inspect_ai import Task, task                                
import pandas as pd
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_qa
from inspect_ai.dataset import json_dataset, hf_dataset, FieldSpec

ds_1000 = json_dataset(
    "hf://datasets/xlangai/DS-1000/test.jsonl",
    FieldSpec(
        input="prompt",
        target="reference_code",
        metadata=["metadata"]
    )
)

# Only include pandas tasks
ds_pandas = ds_1000.filter(
    lambda sample: (
        isinstance(sample.metadata, dict)
        and isinstance(sample.metadata.get("metadata"), dict)
        and sample.metadata["metadata"].get("library") == "Pandas"
    )
)

@task
def ds_pandas_task():
    return Task(
        dataset=ds_pandas,
        solver=[generate()],
        scorer=model_graded_qa(model="anthropic/claude-sonnet-4-20250514"),
    )

# inspect eval eval.py \
  # --model openai/o3-mini-2025-04-16 \
  # --log-format json