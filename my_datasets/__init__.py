# md/__init__.py
from .basetask import BaseTask
from .mwe_persona import MWEPersona
from .hallucination import Hallucination
target_datasets = {
    "mwe_persona": MWEPersona,
    "hallucination": Hallucination,

}

dataset_dict = {}
dataset_dict.update(target_datasets)

def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    assert dataset in dataset_dict, f"Unknown dataset: {dataset}. Available: {list(dataset_dict.keys())}"
    return dataset_dict[dataset](task_name=dataset, *args, **kwargs)
