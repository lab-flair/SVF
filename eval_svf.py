import argparse
import yaml
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from my_datasets import get_dataset
from evaluators import get_evaluator

from vfc.models.shared_boundary import SharedLayerAwareBoundary


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_letters(tok, letters):
    cand_ids, forms = [], []
    for L in letters:
        ids_direct = tok.encode(L, add_special_tokens=False)
        ids_space  = tok.encode(" " + L, add_special_tokens=False)
        if len(ids_direct) == 1:
            cand_ids.append(ids_direct[0]); forms.append(L)
        elif len(ids_space) == 1:
            cand_ids.append(ids_space[0]); forms.append(" " + L)
        else:
            cand_ids.append(ids_direct[-1] if len(ids_direct) > 0 else None); forms.append(L)
    return cand_ids, forms

@torch.no_grad()
def score_next_letter(model, tok, prompt: str, letters):
    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device

    enc = tok(prompt, return_tensors="pt").to(dev)
    out = model(**enc, use_cache=False)
    logits = out.logits[:, -1, :]
    logp = torch.log_softmax(logits, dim=-1)

    cand_ids, _ = map_letters(tok, letters)
    scores = [float("-inf") if tid is None else float(logp[0, tid].item()) for tid in cand_ids]
    pred = int(torch.tensor(scores).argmax().item())
    return pred, scores


class SteeringHook:
    def __init__(self, delta: torch.Tensor):
        self.delta = delta

    def __call__(self, module, inp, out):
        return out + self.delta


def _safe_max_len(tokenizer, model, fallback: int = 4096) -> int:
    ml = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(ml, int) and 0 < ml < 1_000_000:
        return ml
    tml = getattr(tokenizer, "model_max_length", None)
    if isinstance(tml, int) and 0 < tml < 1_000_000:
        return tml
    return fallback

def tokenize_one(tokenizer, model, text: str, device):
    enc = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_safe_max_len(tokenizer, model),
    ).to(device)
    last_idx = enc["attention_mask"].sum(dim=1) - 1  # [1]
    return enc, last_idx

@torch.no_grad()
def extract_last_hidden_multi_layers(model, enc, layers):
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # [0..L]
    last_idx = enc["attention_mask"].sum(dim=1) - 1  # [B]
    out_map = {}
    for l in layers:
        hL = hs[int(l) + 1]  # [B,T,H]
        h_last = hL[torch.arange(hL.size(0), device=hL.device), last_idx, :]
        out_map[int(l)] = h_last
    return out_map

def boundary_logit_and_grad(boundary: SharedLayerAwareBoundary, h_last_1xH: torch.Tensor, layer_id: int):
    boundary.eval()

    # Make dtype consistent with boundary params (R / MLP)
    p = next(boundary.parameters())
    h = h_last_1xH.to(device=p.device, dtype=p.dtype).detach().clone().requires_grad_(True)

    logit = boundary.forward_from_h(h, int(layer_id))  # [1]
    fval = float(logit.detach().cpu().item())

    boundary.zero_grad(set_to_none=True)
    logit.backward()
    grad_h = h.grad.detach().squeeze(0)  # [H], same dtype as h
    return fval, grad_h


def build_delta_for_last_token(enc, last_idx, delta_vec_1xH, device, dtype):
    B, T = enc["input_ids"].shape
    H = delta_vec_1xH.shape[-1]
    delta = torch.zeros((B, T, H), device=device, dtype=dtype)
    delta[0, int(last_idx[0].item()), :] = delta_vec_1xH[0].to(device=device, dtype=dtype)
    return delta


def choose_layers_for_text(
    model, boundary, tokenizer, device,
    text: str,
    candidate_layers: list,
    select_mode: str,
    top_k: int,
):
    top_k = max(1, int(top_k))
    enc, _ = tokenize_one(tokenizer, model, text, device=device)
    h_map = extract_last_hidden_multi_layers(model, enc, candidate_layers)

    if select_mode == "min_abs_margin":
        scored = []
        with torch.no_grad():
            for l in candidate_layers:
                fval = float(boundary.forward_from_h(h_map[int(l)], int(l)).detach().cpu().item())
                scored.append((abs(fval), int(l)))
        scored.sort(key=lambda x: x[0]) 
        return [l for _, l in scored[:top_k]]

    if select_mode == "max_grad_norm":
        scored = []
        for l in candidate_layers:
            _, g = boundary_logit_and_grad(boundary, h_map[int(l)], int(l))
            scored.append((float(g.norm(p=2).detach().cpu().item()), int(l)))
        scored.sort(key=lambda x: -x[0])  
        return [l for _, l in scored[:top_k]]

    raise ValueError(f"Unknown select_mode={select_mode}")

def candidate_scalar_score_from_out(out):
    return float(out.logits[0, -1, :].float().max().detach().cpu().item())



def run_one_shared_vfc_example(
    model, tokenizer, device, model_dtype,
    boundary: SharedLayerAwareBoundary,
    artifact_layers: list,
    ex,
    inject_layers_cfg: list,     
    alpha: float,
    margin_gate: float,
    sigma_g_by_layer: dict,
    sign: float,
    flip: float,
    select_mode: str = "fixed",     
    top_k: int = 1,
    steerability_eps: float = 1e-6, 
):
    q = ex.get("question", "").strip()
    raw_opts = ex.get("options", None)
    if (not raw_opts) or (not isinstance(raw_opts, list)) or (len(raw_opts) < 2):
        opts = ["Yes", "No"]
    else:
        opts = raw_opts[:2]

    textA = f"Question: {q}\nCandidate: {opts[0]}"
    textB = f"Question: {q}\nCandidate: {opts[1]}"

    if select_mode == "fixed":
        layersA = list(inject_layers_cfg)
        layersB = list(inject_layers_cfg)
    else:
        layersA = choose_layers_for_text(
            model, boundary, tokenizer, device,
            text=textA,
            candidate_layers=artifact_layers,
            select_mode=select_mode,
            top_k=top_k,
        )
        layersB = choose_layers_for_text(
            model, boundary, tokenizer, device,
            text=textB,
            candidate_layers=artifact_layers,
            select_mode=select_mode,
            top_k=top_k,
        )

    encA, lastA = tokenize_one(tokenizer, model, textA, device=device)
    encB, lastB = tokenize_one(tokenizer, model, textB, device=device)
    with torch.no_grad():
        base_outA = model(**encA, use_cache=False)
        base_outB = model(**encB, use_cache=False)
    base_scoreA = candidate_scalar_score_from_out(base_outA)
    base_scoreB = candidate_scalar_score_from_out(base_outB)

    hA_map = extract_last_hidden_multi_layers(model, encA, layersA)
    hB_map = extract_last_hidden_multi_layers(model, encB, layersB)

    eff = float(alpha) * float(sign) * float(flip)
    handlesA = []
    try:
        for l in layersA:
            l = int(l)
            h_last = hA_map[l]  # [1,H]
            fval, grad = boundary_logit_and_grad(boundary, h_last, l)

            need = (abs(fval) < margin_gate) if (margin_gate and margin_gate > 0) else True
            if not need:
                continue

            sg = float(sigma_g_by_layer.get(l, 1.0))
            dvec = (eff * (grad / (sg + 1e-9))).unsqueeze(0)  # [1,H]
            delta = build_delta_for_last_token(encA, lastA, dvec, device=device, dtype=model_dtype)

            hdl = model.model.layers[l].register_forward_hook(SteeringHook(delta))
            handlesA.append(hdl)

        with torch.no_grad():
            outA = model(**encA, use_cache=False)
    finally:
        for hdl in handlesA:
            hdl.remove()

    handlesB = []
    try:
        for l in layersB:
            l = int(l)
            h_last = hB_map[l]
            fval, grad = boundary_logit_and_grad(boundary, h_last, l)

            need = (abs(fval) < margin_gate) if (margin_gate and margin_gate > 0) else True
            if not need:
                continue

            sg = float(sigma_g_by_layer.get(l, 1.0))
            dvec = (eff * (grad / (sg + 1e-9))).unsqueeze(0)
            delta = build_delta_for_last_token(encB, lastB, dvec, device=device, dtype=model_dtype)

            hdl = model.model.layers[l].register_forward_hook(SteeringHook(delta))
            handlesB.append(hdl)

        with torch.no_grad():
            outB = model(**encB, use_cache=False)
    finally:
        for hdl in handlesB:
            hdl.remove()
    steer_scoreA = candidate_scalar_score_from_out(outA)
    steer_scoreB = candidate_scalar_score_from_out(outB)
    pred = 0 if steer_scoreA >= steer_scoreB else 1
    gold = int(ex["label"]) 
    if gold == 0:
        base_gap  = base_scoreA  - base_scoreB
        steer_gap = steer_scoreA - steer_scoreB
    else:
        base_gap  = base_scoreB  - base_scoreA
        steer_gap = steer_scoreB - steer_scoreA

    delta = float(steer_gap - base_gap)

    eps = float(steerability_eps)
    if delta > eps:
        cat = "steer"
    elif delta < -eps:
        cat = "anti"
    else:
        cat = "unsteerable"

    return pred, delta, cat



def calibrate_flip(
    all_data, calib_num, seed,
    model, tokenizer, device, model_dtype,
    boundary, artifact_layers,
    inject_layers_cfg,
    alpha, margin_gate,
    sigma_g_by_layer, sign,
    select_mode: str,
    top_k: int,
):
    import random
    n = min(int(calib_num), len(all_data))
    idxs = list(range(len(all_data)))
    random.Random(seed + 12345).shuffle(idxs)
    idxs = idxs[:n]

    def eval_flip(flip):
        corr = 0
        for k in idxs:
            ex = all_data[k]
            gold = int(ex["label"])
            pred, _, _ = run_one_shared_vfc_example(
                model=model, tokenizer=tokenizer, device=device, model_dtype=model_dtype,
                boundary=boundary, artifact_layers=artifact_layers,
                ex=ex,
                inject_layers_cfg=inject_layers_cfg,
                alpha=alpha, margin_gate=margin_gate,
                sigma_g_by_layer=sigma_g_by_layer,
                sign=sign, flip=flip,
                select_mode=select_mode,
                top_k=top_k,
            )
            corr += int(pred == gold)
        return corr / max(1, n), corr, n

    acc_p, c_p, n = eval_flip(+1.0)
    acc_m, c_m, _ = eval_flip(-1.0)

    flip = +1.0 if acc_p >= acc_m else -1.0
    print(f"[Calib] calib_num={n} | flip=+1 acc={acc_p:.4f} ({c_p}/{n}) | flip=-1 acc={acc_m:.4f} ({c_m}/{n}) -> choose flip={flip:+.0f}")
    return float(flip), float(max(acc_p, acc_m))


def main(cfg_path):
    cfg = load_cfg(cfg_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    name = mcfg["name"]
    cache_dir = mcfg.get("cache_dir", None)
    dtype = str(mcfg.get("dtype", "float16")).lower()
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]

    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=mcfg.get("device_map", "auto"),
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    model_dtype = next(model.parameters()).dtype

    dcfg = cfg["dataset"]
    ds = get_dataset(
        dcfg["name"],
        split=dcfg["split"],
        max_data_num=dcfg["max_data_num"],
        use_instruction=dcfg["use_instruction"],
        repo_id=dcfg["repo_id"],
        source=dcfg["source"],
        categories=dcfg["categories"],
    )
    all_data = ds.all_data
    max_examples = int(cfg.get("inference", {}).get("max_examples", len(all_data)))
    all_data = all_data[:max_examples]

    evaluator = get_evaluator(dcfg["name"])

    vcfg = cfg["vfc"]
    use_vfc = bool(vcfg.get("enable", True))

    steerability_eps = float(vcfg.get("steerability_eps", 1e-6))

    if not use_vfc:
        print("[VFC] enable=False -> baseline next-letter scoring")
    else:
        artifact_path = vcfg["shared_boundary_path"]
        art = torch.load(artifact_path, map_location="cpu")

        artifact_layers = [int(x) for x in art["layers"]]
        d = int(art["d"])
        r = int(art["r"])
        norm_type = str(art.get("norm_type", "rms"))
        film_de = int(art.get("film_de", 8))
        layer_id_to_index = {int(k): int(v) for k, v in art["layer_id_to_index"].items()}

        mlp_cfg = cfg.get("mlp", {}) or {}
        boundary = SharedLayerAwareBoundary(
            d=d,
            r=r,
            num_layers=len(artifact_layers),
            layer_id_to_index=layer_id_to_index,
            norm_type=norm_type,
            film_de=film_de,
            mlp_hidden=int(mlp_cfg.get("hidden", 32)),
            mlp_layers=int(mlp_cfg.get("layers", 1)),
            mlp_dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        boundary.load_state_dict(art["state_dict"], strict=True)

        boundary = boundary.to(device=device, dtype=model_dtype)
        boundary.eval()
        alpha = float(vcfg["alpha"])
        margin_gate = float(vcfg.get("margin_gate", 0.0))
        sign = float(art.get("sign", 1.0))

        select_mode = str(vcfg.get("select_layer", "fixed")).strip()
        if select_mode not in ("fixed", "min_abs_margin", "max_grad_norm"):
            raise ValueError(f"vfc.select_layer must be one of fixed|min_abs_margin|max_grad_norm, got {select_mode}")

        if "layers" in vcfg and vcfg["layers"] is not None:
            inject_layers_cfg = [int(x) for x in vcfg["layers"]]
        else:
            inject_layers_cfg = [int(vcfg.get("layer", artifact_layers[len(artifact_layers)//2]))]

        inject_layers_cfg = [l for l in inject_layers_cfg if l in set(artifact_layers)]
        if len(inject_layers_cfg) == 0:
            raise RuntimeError("No valid injection layers after intersecting with artifact layers.")

        top_k = int(vcfg.get("top_k", 1))
        sigma_g_by_layer = {int(k): float(v) for k, v in (art.get("sigma_g_by_layer", {}) or {}).items()}
        sigma_g_global = float(art.get("sigma_g_global", 1.0))
        if not sigma_g_by_layer:
            sigma_g_by_layer = {int(l): sigma_g_global for l in artifact_layers}

        print(f"[SharedVFC] enable=True | alpha={alpha}, margin_gate={margin_gate}, sign={sign:+.0f}")
        print(f"[SharedVFC] artifact_layers={artifact_layers}")
        print(f"[SharedVFC] select_layer={select_mode} | cfg_inject_layers={inject_layers_cfg} | top_k={top_k}")
        print(f"[SharedVFC] steerability_eps={steerability_eps:g}")

        flip = 1.0
        calib_cfg = vcfg.get("calibration", {}) or {}
        calib_enable = bool(calib_cfg.get("enable", True))
        calib_num = int(calib_cfg.get("calib_num", 64))

        if calib_enable and calib_num > 0:
            flip, _ = calibrate_flip(
                all_data=all_data,
                calib_num=calib_num,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_dtype=model_dtype,
                boundary=boundary,
                artifact_layers=artifact_layers,
                inject_layers_cfg=inject_layers_cfg,
                alpha=alpha,
                margin_gate=margin_gate,
                sigma_g_by_layer=sigma_g_by_layer,
                sign=sign,
                select_mode=select_mode,
                top_k=top_k,
            )
        else:
            flip = float(calib_cfg.get("flip", 1.0))
            print(f"[Calib] disabled -> using flip={flip:+.0f}")

        print(f"[SharedVFC] effective direction multiplier sign*flip = {(sign*flip):+.0f}")
    correct = 0
    total = 0
    steer_cnt = 0
    anti_cnt = 0
    un_cnt = 0
    delta_sum = 0.0
    delta_abs_sum = 0.0

    for ex in tqdm(all_data, desc="Evaluating"):
        if not use_vfc:
            prompt, letters = evaluator.prepare_prompt(ex, ds)
            pred_idx, _ = score_next_letter(model, tokenizer, prompt, letters)
            gold_idx = evaluator.gold_index(ex)
            correct += int(pred_idx == gold_idx)
            total += 1
            un_cnt += 1
            continue

        gold = int(ex["label"])
        pred, delta, cat = run_one_shared_vfc_example(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
            boundary=boundary,
            artifact_layers=artifact_layers,
            ex=ex,
            inject_layers_cfg=inject_layers_cfg,
            alpha=float(alpha),
            margin_gate=float(margin_gate),
            sigma_g_by_layer=sigma_g_by_layer,
            sign=float(sign),
            flip=float(flip),
            select_mode=str(select_mode),
            top_k=int(top_k),
            steerability_eps=float(steerability_eps),
        )
        correct += int(pred == gold)
        total += 1

        delta_sum += float(delta)
        delta_abs_sum += abs(float(delta))

        if cat == "steer":
            steer_cnt += 1
        elif cat == "anti":
            anti_cnt += 1
        else:
            un_cnt += 1

    acc = correct / total if total > 0 else 0.0
    tag = "SharedVFC(multi-layer)" if use_vfc else "Baseline(next-letter)"
    print(f"[{tag}] accuracy = {acc:.4f} ({correct}/{total})")

    # ---- print steerability stats (new)
    if total > 0:
        steer_ratio = steer_cnt / total
        anti_ratio = anti_cnt / total
        un_ratio = un_cnt / total
        mean_delta = delta_sum / total
        mean_abs_delta = delta_abs_sum / total

        print(
            f"[Steerability] steer={steer_cnt}/{total} ({steer_ratio:.3%}) | "
            f"anti={anti_cnt}/{total} ({anti_ratio:.3%}) | "
            f"unsteerable={un_cnt}/{total} ({un_ratio:.3%}) | "
            f"eps={steerability_eps:g}"
        )
        print(f"[Steerability] mean_delta={mean_delta:.6g} | mean_abs_delta={mean_abs_delta:.6g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg)
