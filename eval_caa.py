import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from my_datasets import get_dataset
from evaluators import get_evaluator

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(cfg_model: dict):
    name = cfg_model["name"]
    cache_dir = cfg_model.get("cache_dir")
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
    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, tok


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return list(model.base_model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Cannot locate decoder layers for this model.")


class CAAMeanDiffSteerer:
    def __init__(self, model, layer: int, v_md: torch.Tensor, sigma: float, alpha: float, eps: float = 1e-12):
        self.model = model
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.eps = float(eps)

        v = v_md.detach()
        if isinstance(v, dict) and "tensor" in v:
            v = v["tensor"]
        if not isinstance(v, torch.Tensor):
            raise ValueError("v_md must be a torch.Tensor (or a dict with key 'tensor').")
        self.v_scaled_cpu = (v.to(torch.float32).cpu() / (self.sigma + self.eps))

        self.enabled = True
        self.hook_handle = None

        self.disp_sum = 0.0
        self.disp_count = 0

        layers = get_decoder_layers(model)
        if not (0 <= self.layer < len(layers)):
            raise ValueError(f"layer={self.layer} out of range (0..{len(layers)-1})")

        self.hook_handle = layers[self.layer].register_forward_hook(self._hook_fn)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def reset_disp_stats(self):
        self.disp_sum = 0.0
        self.disp_count = 0

    def mean_displacement(self) -> float:
        return float(self.disp_sum / max(1, self.disp_count))

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _hook_fn(self, module, inputs, output):
        if not self.enabled:
            return output

        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None

        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output

        v_scaled = self.v_scaled_cpu.to(device=hs.device, dtype=hs.dtype)  # [d]
        delta_vec = (self.alpha * v_scaled)  # [d]

        disp = float(delta_vec.detach().to(torch.float32).norm(p=2).item())

        hs2 = hs.clone()
        bsz = hs2.size(0)
        hs2[:, -1, :] = hs2[:, -1, :] + delta_vec.view(1, 1, -1)

        self.disp_sum += disp * bsz
        self.disp_count += bsz

        return hs2 if rest is None else (hs2,) + rest


def map_letters(tok, letters):
    cand_ids, forms = [], []
    for L in letters:
        ids_direct = tok.encode(L, add_special_tokens=False)
        ids_space = tok.encode(" " + L, add_special_tokens=False)
        if len(ids_direct) == 1:
            cand_ids.append(ids_direct[0]); forms.append(L)
        elif len(ids_space) == 1:
            cand_ids.append(ids_space[0]); forms.append(" " + L)
        else:
            cand_ids.append(ids_direct[-1] if len(ids_direct) > 0 else None); forms.append(L)
    return cand_ids, forms


@torch.no_grad()
def score_next_letter(model, tok, prompt: str, letters):
    enc = tok(prompt, return_tensors="pt")
    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}

    out = model(**enc, use_cache=False)
    logits = out.logits[:, -1, :]
    logp = torch.log_softmax(logits, dim=-1)

    cand_ids, _ = map_letters(tok, letters)
    scores = [float("-inf") if tid is None else float(logp[0, tid].item()) for tid in cand_ids]
    pred = int(torch.tensor(scores).argmax().item())
    return pred, scores


def gap_gold_minus_other(scores, gold_idx: int):
    if scores is None or len(scores) < 2:
        return 0.0
    g = int(gold_idx)
    if len(scores) == 2:
        o = 1 - g
    else:
        o = max([i for i in range(len(scores)) if i != g], key=lambda i: scores[i])
    return float(scores[g] - scores[o])

def evaluate(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    dcfg = cfg["dataset"]
    ds = get_dataset(
        dcfg["name"],
        split=dcfg.get("split", "validation"),
        max_data_num=dcfg.get("max_data_num", None),
        use_instruction=dcfg.get("use_instruction", True),
        seed=int(cfg.get("seed", 42)),
        **{k: v for k, v in dcfg.items() if k not in {"name", "split", "max_data_num", "use_instruction"}}
    )
    evaluator = get_evaluator(dcfg["name"])

    model, tok = load_model(cfg["model"])

    steer_cfg = cfg.get("steer", None) or {}
    use_steer = bool(steer_cfg.get("enable", False))
    steerability_eps = float(steer_cfg.get("steerability_eps", 1e-6))

    print(f"[Debug] use_steer={use_steer} | steer_cfg keys={list(steer_cfg.keys())}")

    steerer: Optional[CAAMeanDiffSteerer] = None
    if use_steer:
        v_md = torch.load(steer_cfg["v_md_path"])
        if isinstance(v_md, dict) and "tensor" in v_md:
            v_md = v_md["tensor"]

        sigma_obj = torch.load(steer_cfg["sigma_path"])
        sigma = float(sigma_obj["sigma"]) if isinstance(sigma_obj, dict) else float(sigma_obj)

        steerer = CAAMeanDiffSteerer(
            model=model,
            layer=int(steer_cfg["layer"]),
            v_md=v_md,
            sigma=sigma,
            alpha=float(steer_cfg.get("alpha", 1.0)),
        )
        steerer.reset_disp_stats()

    max_examples = int(cfg.get("inference", {}).get("max_examples", len(ds)))
    n_total = min(max_examples, len(ds))
    n_correct = 0

    steer_cnt = 0
    anti_cnt = 0
    un_cnt = 0
    delta_sum = 0.0
    delta_abs_sum = 0.0

    for i in range(n_total):
        ex = ds[i]
        prompt, letters = evaluator.prepare_prompt(ex, ds)
        gold_idx = evaluator.gold_index(ex)

        if steerer is not None:
            steerer.disable()
        base_pred_idx, base_scores = score_next_letter(model, tok, prompt, letters)
        base_gap = gap_gold_minus_other(base_scores, gold_idx)

        if steerer is not None:
            steerer.enable()
        steer_pred_idx, steer_scores = score_next_letter(model, tok, prompt, letters)
        steer_gap = gap_gold_minus_other(steer_scores, gold_idx)

        pred_idx = steer_pred_idx if use_steer else base_pred_idx
        n_correct += int(pred_idx == gold_idx)

        if use_steer:
            delta = float(steer_gap - base_gap)
            delta_sum += delta
            delta_abs_sum += abs(delta)

            if delta > steerability_eps:
                cat = "steer"
                steer_cnt += 1
            elif delta < -steerability_eps:
                cat = "anti"
                anti_cnt += 1
            else:
                cat = "unsteerable"
                un_cnt += 1
        else:
            delta = 0.0
            cat = "unsteerable"
            un_cnt += 1

        if (i < 3) or (i % 50 == 0) or (i == n_total - 1):
            print(
                f"[#{i+1}/{n_total}] Gold={letters[gold_idx]} Pred={letters[pred_idx]} "
                f"Correct={pred_idx==gold_idx} | cat={cat} | delta={delta:.6g}"
            )
    mean_disp = 0.0
    if steerer is not None:
        steerer.disable()
        mean_disp = steerer.mean_displacement()
        steerer.remove()

    acc = n_correct / n_total if n_total > 0 else 0.0

    print("=" * 60)
    print(f"Dataset: {dcfg['name']} | Split: {dcfg.get('split', 'validation')} | N={n_total}")
    if use_steer:
        print(
            f"CAA-MD: enabled | layer={int(steer_cfg['layer'])} "
            f"alpha={float(steer_cfg.get('alpha', 1.0))} "
            f"v_md={steer_cfg.get('v_md_path')} | sigma={sigma:.6g} "
            f"| mean||alpha*(v_md/sigma)||_2={mean_disp:.6g}"
        )
    else:
        print("CAA-MD: disabled")
    print(f"Accuracy: {acc:.4f} ({n_correct}/{n_total})")

    print("-" * 60)
    print(f"[Steerability] total={n_total} | use_steer={use_steer} | eps={steerability_eps:g}")
    if n_total > 0:
        steer_ratio = steer_cnt / n_total
        anti_ratio = anti_cnt / n_total
        un_ratio = un_cnt / n_total

        mean_delta = delta_sum / n_total
        mean_abs_delta = delta_abs_sum / n_total

        print(
            f"[Steerability] steer={steer_cnt}/{n_total} ({steer_ratio:.3%}) | "
            f"anti={anti_cnt}/{n_total} ({anti_ratio:.3%}) | "
            f"unsteerable={un_cnt}/{n_total} ({un_ratio:.3%})"
        )
        print(f"[Steerability] mean_delta={mean_delta:.6g} | mean_abs_delta={mean_abs_delta:.6g}")
    else:
        print("[Steerability] n_total=0 (no examples)")
    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()
    evaluate(args.cfg)
