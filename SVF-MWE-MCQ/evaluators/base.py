from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseEvaluator(ABC):
    @abstractmethod
    def prepare_prompt(self, example: Dict[str, Any], dataset) -> (str, List[str]):
        raise NotImplementedError

    @abstractmethod
    def gold_index(self, example: Dict[str, Any]) -> int:
        raise NotImplementedError

    def postprocess_prediction(self, raw_pred: str, choices: List[str]) -> int:
        raw = (raw_pred or "").strip()
        if not raw:
            return -1
        import re
        m = re.search(r"([A-Z])", raw)
        if not m:
            return -1
        letter = m.group(1)
        letters = _letters(len(choices))
        return letters.index(letter) if letter in letters else -1


def _letters(n: int) -> List[str]:
    import string
    letters = list(string.ascii_uppercase)
    if n > len(letters):
        raise ValueError(f"Too many options ({n}); max supported {len(letters)}")
    return letters[:n]