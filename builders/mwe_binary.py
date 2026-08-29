# builders/mwe_binary.py
from typing import List, Dict, Optional
import random

from my_datasets import get_dataset

DEFAULT_AB = ["Yes", "No"]

def _norm_opt_list(opts):
    if not opts or not isinstance(opts, list) or len(opts) < 2:
        return DEFAULT_AB
    if len(opts) > 2:
        return [opts[0], opts[1]]
    return opts

def build_mwe_binary(
    repo_id: str,
    source: str,                      # "persona" | "aarisk"
    categories: List[str],
    split: str = "validation",
    per_category_max: Optional[int] = 1000,
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)
    ds = get_dataset(
        "mwe_persona",
        split=split,
        max_data_num=None,
        use_instruction=False,       
        seed=seed,
        repo_id=repo_id,
        source=source,
        categories=categories,
        per_category_max=per_category_max,
    )
    out: List[Dict] = []
    for ex in ds.all_data:
        q = (ex.get("question") or "").strip()
        gold = int(ex["label"])
        opts = _norm_opt_list(ex.get("options", None))
        for j, opt in enumerate(opts[:2]):  # A/B
            text = f"Question: {q}\nCandidate: {opt}"
            y = 1 if j == gold else 0
            out.append({"text": text, "y": y})
    rng.shuffle(out)
    return out
