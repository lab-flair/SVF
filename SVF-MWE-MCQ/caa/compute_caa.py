import os
import argparse
import yaml
from typing import Dict, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from my_datasets import get_dataset
from evaluators.base import _letters


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tok(cfg_model: Dict):
    name = cfg_model["name"]
    cache_dir = cfg_model.get("cache_dir", None)
    dtype = str(cfg_model.get("dtype", "float16")).lower()
    device_map = cfg_model.get("device_map", "auto")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tok


def choose_single_letter_token_id(tok, letter: str) -> Tuple[str, int]:
    ids_direct = tok.encode(letter, add_special_tokens=False)
    ids_space = tok.encode(" " + letter, add_special_tokens=False)
    if len(ids_direct) == 1:
        return letter, ids_direct[0]
    if len(ids_space) == 1:
        return " " + letter, ids_space[0]
    return letter, ids_direct[-1] if len(ids_direct) > 0 else tok.eos_token_id


@torch.no_grad()
def collect_letter_state(model, tok, prompt: str, layer: int, append_text: str, target_token_id: int) -> torch.Tensor:
    encoded = tok(prompt + append_text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states 
    hL = hs[layer + 1]     
    input_ids = encoded["input_ids"][0]
    pos = None
    for i in range(len(input_ids) - 1, -1, -1):
        if int(input_ids[i].item()) == int(target_token_id):
            pos = i
            break
    if pos is None:
        pos = input_ids.size(0) - 1
    vec = hL[0, pos, :].detach().float().cpu()
    return vec


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed", 42))

    dcfg = cfg["dataset"]
    ds = get_dataset(
        dcfg["name"],
        split=dcfg.get("split", "validation"),
        max_data_num=dcfg.get("max_data_num", None),
        use_instruction=dcfg.get("use_instruction", True),
        seed=cfg.get("seed", 42),
        **{k: v for k, v in dcfg.items() if k not in {"name", "split", "max_data_num", "use_instruction"}}
    )

    model, tok = load_model_and_tok(cfg["model"])
    layer = cfg["caa"]["layer"]
    max_pairs = cfg["caa"].get("max_pairs", None)
    out_dir = cfg["caa"]["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    letters = _letters(2)
    app_text = {}
    app_id = {}
    for L in letters:
        t, tid = choose_single_letter_token_id(tok, L)
        app_text[L] = t
        app_id[L] = tid

    sum_diff = None
    count = 0
    collected_states = []

    N = len(ds) if max_pairs is None else min(max_pairs, len(ds))
    print(f"[CAA] Collecting MD on {N} paired prompts at layer {layer} ...")

    for i in range(N):
        ex = ds[i]
        prompt_body, _ans_letters, _ = ds.apply_template(ex)
        prompt = (ds.get_task_instruction() + "\n" + prompt_body) if getattr(ds, "use_instruction", False) else prompt_body

        gold_idx = int(ex["label"])
        pos_L = letters[gold_idx]
        neg_L = letters[1 - gold_idx]
        h_pos = collect_letter_state(model, tok, prompt, layer, app_text[pos_L], app_id[pos_L])
        h_neg = collect_letter_state(model, tok, prompt, layer, app_text[neg_L], app_id[neg_L])

        diff = (h_pos - h_neg)
        sum_diff = diff if sum_diff is None else (sum_diff + diff)
        count += 1
        collected_states.append(h_pos)
        collected_states.append(h_neg)

        if (i < 3) or (i % 200 == 0) or (i == N-1):
            print(f"[{i+1}/{N}] gold={pos_L} collected.")

    v_md = (sum_diff / max(count, 1)).float()
    with torch.no_grad():
        v_unit = v_md / (v_md.norm(p=2) + 1e-12)
        X = torch.stack(collected_states, dim=0) if collected_states else v_unit.unsqueeze(0)
        proj = (X @ v_unit)
        sigma = float(proj.std().item())

    torch.save(v_md, os.path.join(out_dir, "v_md.pt"))
    torch.save({"sigma": sigma}, os.path.join(out_dir, "sigma.pt"))
    print(f"[CAA] Saved v_md -> {os.path.join(out_dir,'v_md.pt')}  | sigma={sigma:.6f}")


if __name__ == "__main__":
    main()
