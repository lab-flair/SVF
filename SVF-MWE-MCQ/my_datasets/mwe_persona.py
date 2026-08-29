# my_datasets/mwe_persona.py
try:
    from . import BaseTask
except:
    from basetask import BaseTask

import os, json, gzip, random, re
from typing import Dict, List, Optional, Any
from huggingface_hub import list_repo_files, hf_hub_download

LETTERS = [chr(ord('A') + i) for i in range(26)]
AB = ["A", "B"]

# ---------- utils ----------
def _split_indices(n: int, split: str, seed: int):
    rng = random.Random(seed)
    idx = list(range(n)); rng.shuffle(idx)
    n_train = int(round(0.4 * n))
    n_val   = int(round(0.1 * n))
    if split == "train": return idx[:n_train]
    elif split == "validation": return idx[n_train:n_train+n_val]
    else: return idx[n_train+n_val:]

def _norm_str(x):
    return (x or "").strip() if isinstance(x, str) else ""

def _first_nonempty(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return None

def _read_jsonl_local(path: str) -> List[Dict]:
    with open(path, "rb") as fh:
        bs = fh.read()
    if path.endswith(".gz"):
        txt = gzip.decompress(bs).decode("utf-8", errors="ignore")
    else:
        txt = bs.decode("utf-8", errors="ignore")
    out = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

# ---------- parse choices ----------
_LINE_RE = re.compile(r"^\s*\(([A-Z])\)\s*(.+?)\s*$")
_INLINE_AB_RE = re.compile(r"\(A\)\s*([^\(\)]+?)\s*\(B\)\s*([^\(\)]+)")

def _extract_choices_from_question(q_text: str) -> List[str]:
    q = q_text or ""
    lines = [l.rstrip() for l in q.splitlines()]
    found = {}
    for ln in lines:
        m = _LINE_RE.match(ln)
        if m:
            L, txt = m.group(1), m.group(2)
            found[L] = txt

    if not found:
        m = _INLINE_AB_RE.search(q)
        if m:
            a_txt = m.group(1).strip()
            b_txt = m.group(2).strip()
            return [a_txt, b_txt]
        return []

    max_pos = max(LETTERS.index(L) for L in found.keys())
    letters_seq = LETTERS[:max_pos+1]
    return [found.get(L, L) for L in letters_seq]


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

_DEFAULT_SYNONYMS = {
    "yes": {"yes", "y", "yeah", "yep", "affirmative", "agree", "strongly agree"},
    "no":  {"no", "n", "nope", "negative", "disagree", "strongly disagree"},
    "agree": {"agree", "strongly agree", "yes"},
    "disagree": {"disagree", "strongly disagree", "no"},
}

def _match_text_to_choices(cand: str, choices: List[str]) -> Optional[int]:
    cand_norm = _norm(re.sub(r"^\([A-Z]\)\s*", "", cand or ""))
    syn = None
    for key, vs in _DEFAULT_SYNONYMS.items():
        if cand_norm in vs:
            syn = key
            break

    best_idx, best_score = None, -1.0
    for i, opt in enumerate(choices):
        opt_norm = _norm(opt)
        score = 0.0
        if cand_norm == opt_norm:
            score = 2.0
        elif cand_norm in opt_norm or opt_norm in cand_norm:
            score = 1.5
        else:
            s1, s2 = set(cand_norm.split()), set(opt_norm.split())
            inter = len(s1 & s2); uni = len(s1 | s2) or 1
            score = inter / uni
        if syn:
            if syn == "yes" and ("yes" in opt_norm or "agree" in opt_norm):
                score += 1.0
            if syn == "no" and ("no" in opt_norm or "disagree" in opt_norm):
                score += 1.0
            if syn == "agree" and "agree" in opt_norm:
                score += 1.0
            if syn == "disagree" and "disagree" in opt_norm:
                score += 1.0
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx if best_idx is not None and best_score >= 0.15 else None

def _extract_gold_idx(ex: Dict, choices_text: List[str]) -> Optional[int]:
    cand = None
    for k in ["answer_matching_behavior", "correct_answer", "gold", "target", "answer"]:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            cand = v.strip()
            break

    if isinstance(cand, str):
        m = re.search(r"\(([A-Z])\)", cand)
        if m and m.group(1) in AB:
            return AB.index(m.group(1))
        m2 = re.search(r"\b([A-Z])\b", cand)
        if m2 and m2.group(1) in AB:
            return AB.index(m2.group(1))

    opts = choices_text[:] if choices_text else []
    if not opts:
        opts = ["Yes", "No"]
    idx = _match_text_to_choices(cand or "", opts)
    if idx is not None:
        return idx

    if isinstance(ex.get("answer_index", None), int):
        i = int(ex["answer_index"])
        if 0 <= i < 2:
            return i

    return None


class MWEPersona(BaseTask):
    """
    Loader for Anthropic/model-written-evals.
    source:
      - "persona" -> persona/*.jsonl
      - "aarisk"  -> advanced-ai-risk/human_generated_evals/*.jsonl
    """
    def __init__(
        self,
        split: str = "validation",
        repo_id: str = "Anthropic/model-written-evals",
        categories: Optional[List[str]] = None,
        per_category_max: Optional[int] = 1000,
        source: str = "persona",  # "persona" | "aarisk"
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert source in ("persona", "aarisk")
        assert categories and isinstance(categories, list), "Provide categories: List[str]"
        self.task_type = "classification"
        self.class_num = 2
        self.repo_id = repo_id
        self.categories = [c.lower().strip() for c in categories]
        self.split = split
        self.per_category_max = per_category_max
        self.source = source

        print(f"[MWEPersona] repo={repo_id}  source={source}  split={split}  seed={self.seed}")

        try:
            files = list_repo_files(self.repo_id, repo_type="dataset", revision="main")
        except Exception as e:
            raise RuntimeError(f"Failed to list files from {self.repo_id}: {e}")

        if source == "persona":
            persona_files = [f for f in files
                             if f.lower().startswith("persona/")
                             and (f.endswith(".jsonl") or f.endswith(".jsonl.gz"))]
            available: Dict[str, List[str]] = {}
            for f in persona_files:
                base = os.path.splitext(os.path.basename(f))[0].lower()
                available.setdefault(base, []).append(f)
        else:
            aar_files = [f for f in files
                         if f.lower().startswith("advanced-ai-risk/human_generated_evals/")
                         and (f.endswith(".jsonl") or f.endswith(".jsonl.gz"))]
            available: Dict[str, List[str]] = {}
            for f in aar_files:
                base = os.path.basename(f).lower()
                cat = base.split("-")[0] 
                available.setdefault(cat, []).append(f)

        missing = [c for c in self.categories if c not in available]
        if missing:
            raise RuntimeError(
                f"Requested categories not found: {missing}. "
                f"Some available: {sorted(list(available.keys()))[:30]}"
            )

        rng = random.Random(self.seed)
        per_cat_items: Dict[str, List[Dict]] = {}

        for cat in self.categories:
            paths = available[cat]
            rows = []
            for p in paths:
                local = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=p,
                    repo_type="dataset",
                    revision="main"
                )
                rows.extend(_read_jsonl_local(local))

            items = []
            for r in rows:
                q = _first_nonempty(r, ["question", "prompt", "input", "text"])
                q = _norm_str(q)
                if not q: 
                    continue

                choices_text = _extract_choices_from_question(q)  # 可能 []
                gold_idx = _extract_gold_idx(r, choices_text)

                if gold_idx is None:
                    default_opts = ["Yes", "No"]
                    gold_idx = _match_text_to_choices(_first_nonempty(r, ["answer_matching_behavior","correct_answer","gold","target","answer"]) or "", default_opts)

                if gold_idx is None:
                    continue  

                items.append({
                    "category": cat,
                    "question": q,
                    "options": choices_text if choices_text else None, 
                    "label": int(gold_idx),
                })

            if not items:
                print(f"[WARN] category '{cat}' has 0 usable rows after parsing.")
                continue

            idx = _split_indices(len(items), self.split, self.seed)
            subset = [items[i] for i in idx]
            if self.per_category_max is not None and len(subset) > self.per_category_max:
                rng.shuffle(subset); subset = subset[:self.per_category_max]
            per_cat_items[cat] = subset

        all_items: List[Dict] = []
        for cat in self.categories:
            all_items.extend(per_cat_items.get(cat, []))
        rng.shuffle(all_items)

        self.dataset = None
        self.all_data = all_items
        self.all_labels = [ex["label"] for ex in all_items]
        self.class_num = 2

        if self.max_data_num is not None and len(self.all_data) > self.max_data_num:
            self.random_sample_data(self.max_data_num)

        print(f"[MWEPersona] loaded {len(self.all_data)} items across {len(self.categories)} categories (A/B).")
        if len(self.all_data) > 0:
            print("Example:", self.all_data[0])

    # ===== BaseTask API =====
    def get_dmonstration_template(self):
        letters = AB if self.class_num == 2 else LETTERS[:self.class_num]
        choices_line = " ".join(f"({L}) {{{L}}}" for L in letters)
        return {
            "input": "Question: {q}\nChoices: " + choices_line + "\nAnswer:",
            "ans": "{letter}",
            "options": letters,
            "format": ["Question:", "Choices:", "Answer:"],
        }

    def get_task_instruction(self):
        return "You will answer multiple-choice questions. Reply with a single capital letter (A or B)."

    def apply_template(self, data: Dict[str, Any]):
        tmpl = self.get_dmonstration_template()
        letters = tmpl["options"]
        input_str = tmpl["input"].replace("{q}", data.get("question", ""))

        opts = data.get("options", None)
        if not opts:
            opts = ["Yes", "No"]
        filled = input_str
        for i, L in enumerate(letters):
            txt = (opts[i] if (opts and i < len(opts)) else L)
            filled = filled.replace("{" + L + "}", str(txt))

        answer_str_list = [tmpl["ans"].replace("{letter}", L) for L in letters]
        label = int(data.get("label", 0))
        return filled, answer_str_list, label

    def get_all_labels(self):
        return self.all_labels
