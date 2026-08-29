# evaluators/mwe_persona_ab.py
from typing import Any, Dict, Tuple, List
from .base import BaseEvaluator, _letters

class MWEPersonaABEvaluator(BaseEvaluator):
    """
    Evaluator for mwe_persona (A/B). We rely on dataset.apply_template to build prompt.
    """

    def prepare_prompt(self, example: Dict[str, Any], dataset) -> Tuple[str, List[str]]:
        input_str, answer_str_list, _ = dataset.apply_template(example)
        instruct = dataset.get_task_instruction() if getattr(dataset, "use_instruction", False) else ""
        prompt = instruct + "\n" + input_str if instruct else input_str
        letters = _letters(len(answer_str_list))  # ["A","B"]
        return prompt, letters

    def gold_index(self, example: Dict[str, Any]) -> int:
        return int(example["label"])
