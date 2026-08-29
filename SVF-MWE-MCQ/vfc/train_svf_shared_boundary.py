import os
import argparse
import yaml
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
#from builders.helpsteer2_binary_gen import build_mwe_binary
#from builders.mwe_binary_gen import build_mwe_binary
from builders.mwe_binary import build_mwe_binary
#from builders.truthfulqa_binary_gen import build_mwe_binary
#from builders.hallucination_binary_gen import build_mwe_binary

from vfc.utils.collect_multi_layer import collect_activations_multi_layer
from vfc.models.shared_boundary import SharedLayerAwareBoundary, norm_hidden


# ----------------- utils -----------------

def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class LayerHDS(Dataset):
    def __init__(self, X_by_layer: Dict[int, torch.Tensor], Y: torch.Tensor, layers: List[int]):
        self.layers = [int(l) for l in layers]
        self.Y = Y
        self.N = Y.numel()
        self.X_by_layer = X_by_layer
        self.M = len(self.layers)

    def __len__(self):
        return self.N * self.M

    def __getitem__(self, idx):
        n = idx % self.N
        m = idx // self.N
        layer_id = self.layers[m]
        h = self.X_by_layer[layer_id][n]
        y = self.Y[n]
        return layer_id, h, y

@torch.no_grad()
def pca_init_R_from_pooled_h(
    X_by_layer: Dict[int, torch.Tensor],
    layers: List[int],
    r: int,
    norm_type: str,
) -> torch.Tensor:
    Xs = []
    for l in layers:
        h = X_by_layer[int(l)] 
        h_hat = norm_hidden(h, norm_type=norm_type).float()
        Xs.append(h_hat)
    X = torch.cat(Xs, dim=0)
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    U, S, V = torch.pca_lowrank(Xc, q=min(r, min(Xc.shape) - 1))
    W = V[:, :r].T.contiguous() 
    return W


@torch.no_grad()
def eval_acc(boundary: SharedLayerAwareBoundary, dl, device) -> float:
    boundary.eval()
    correct, total = 0.0, 0.0
    for layer_id, h, y in dl:
        h = h.to(device)
        y = y.to(device)
        logits = torch.empty_like(y)
        for i in range(h.size(0)):
            logits[i] = boundary.forward_from_h(h[i:i+1], int(layer_id[i].item())).squeeze(0)

        preds = (logits.sigmoid() >= 0.5).float()
        correct += (preds == y).float().sum().item()
        total += y.numel()
    return correct / max(total, 1.0)


@torch.no_grad()
def sign_calibration(
    boundary: SharedLayerAwareBoundary,
    X_by_layer: Dict[int, torch.Tensor],
    Y: torch.Tensor,
    layers: List[int],
    device,
    batch_size: int = 1024,
) -> Tuple[float, float, float]:
    boundary.eval()
    logits_all = []
    Ys_all = []

    for l in layers:
        X = X_by_layer[int(l)]
        for i in range(0, X.size(0), batch_size):
            h = X[i:i+batch_size].to(device)
            li = int(l)
            logits = boundary.forward_from_h(h, li).detach().cpu()
            logits_all.append(logits)
            Ys_all.append(Y[i:i+batch_size].detach().cpu())

    logits_all = torch.cat(logits_all, dim=0)
    Ys_all = torch.cat(Ys_all, dim=0).view(-1)

    y1 = (Ys_all == 1)
    y0 = (Ys_all == 0)
    m1 = float(logits_all[y1].mean()) if y1.any() else 0.0
    m0 = float(logits_all[y0].mean()) if y0.any() else 0.0
    sign = 1.0 if m1 > m0 else -1.0

    if sign < 0:
        last_lin = None
        for m in reversed(boundary.f.net):
            if isinstance(m, nn.Linear):
                last_lin = m
                break
        if last_lin is None:
            raise RuntimeError("Cannot find final Linear in shared boundary MLP for sign calibration.")
        last_lin.weight.data *= -1.0
        if last_lin.bias is not None:
            last_lin.bias.data *= -1.0

    return float(sign), float(m1), float(m0)


def estimate_sigma_g_per_layer(
    boundary: SharedLayerAwareBoundary,
    X_by_layer: Dict[int, torch.Tensor],
    layers: List[int],
    device,
    max_samples_per_layer: int = 2048,
) -> Dict[int, float]:
    boundary.eval()
    sigmas = {}

    for l in layers:
        X = X_by_layer[int(l)]
        n = min(X.size(0), max_samples_per_layer)
        idx = torch.randperm(X.size(0))[:n]
        h = X[idx].to(device).detach().clone().requires_grad_(True)

        logits = boundary.forward_from_h(h, int(l))  # [n]
        grads = []
        for i in range(n):
            boundary.zero_grad(set_to_none=True)
            if h.grad is not None:
                h.grad.zero_()
            logits[i].backward(retain_graph=True)
            grads.append(h.grad[i].detach().float().norm(p=2).item())
        grads = torch.tensor(grads)
        sigmas[int(l)] = float(torch.quantile(grads, 0.5).item())

    return sigmas

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    bcfg = cfg["build"]
    train_items = build_mwe_binary(
        repo_id=bcfg["repo_id"],
        source=bcfg["source"],
        categories=bcfg["categories"],
        split=bcfg.get("train_split", "train"),
        per_category_max=bcfg.get("per_category_max_train", 500),
        seed=int(cfg.get("seed", 42)),
    )
    val_items = build_mwe_binary(
        repo_id=bcfg["repo_id"],
        source=bcfg["source"],
        categories=bcfg["categories"],
        split=bcfg.get("val_split", "validation"),
        per_category_max=bcfg.get("per_category_max_val", 500),
        seed=int(cfg.get("seed", 42)) + 1,
    )

    if len(train_items) == 0:
        raise RuntimeError("No train items.")
    if len(val_items) == 0:
        print("[Warn] No val items; proceeding with train only.")

    mcfg = cfg["model"]
    dtype = str(mcfg.get("dtype", "float16")).lower()
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"], cache_dir=mcfg.get("cache_dir", None))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        cache_dir=mcfg.get("cache_dir", None),
        torch_dtype=torch_dtype,
        device_map=mcfg.get("device_map", "auto"),
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scfg = cfg["shared_vfc"]
    layers = [int(x) for x in scfg["layers"]]
    r = int(scfg["r"])
    norm_type = str(scfg.get("norm_type", "rms")).lower()
    film_de = int(scfg.get("film_de", 8))
    batch_collect = int(scfg.get("batch_collect", 4))
    out_dir = scfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    Xtr_by_layer, Ytr = collect_activations_multi_layer(model, tokenizer, train_items, layers=layers, batch_size=batch_collect)
    if len(val_items) > 0:
        Xva_by_layer, Yva = collect_activations_multi_layer(model, tokenizer, val_items, layers=layers, batch_size=batch_collect)
    else:
        Xva_by_layer, Yva = None, None

    d = next(iter(Xtr_by_layer.values())).size(1)

    layer_id_to_index = {int(l): i for i, l in enumerate(layers)}
    mlp_cfg = cfg["mlp"]
    boundary = SharedLayerAwareBoundary(
        d=d,
        r=r,
        num_layers=len(layers),
        layer_id_to_index=layer_id_to_index,
        norm_type=norm_type,
        film_de=film_de,
        mlp_hidden=int(mlp_cfg.get("hidden", 64)),
        mlp_layers=int(mlp_cfg.get("layers", 1)),
        mlp_dropout=float(mlp_cfg.get("dropout", 0.0)),
    ).to(device)

    if bool(scfg.get("pca_init", False)):
        R0 = pca_init_R_from_pooled_h(Xtr_by_layer, layers, r=r, norm_type=norm_type) 
        with torch.no_grad():
            boundary.R.copy_(R0.to(device))
        print("[Init] R initialized from pooled PCA.")

    train_ds = LayerHDS(Xtr_by_layer, Ytr, layers)
    bs = int(cfg["train"].get("batch_size", 1024))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    if Xva_by_layer is not None:
        val_ds = LayerHDS(Xva_by_layer, Yva, layers)
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    else:
        val_dl = None

    opt = torch.optim.AdamW(
        boundary.parameters(),
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-2)),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = int(cfg["train"].get("epochs", 20))
    patience = int(cfg["train"].get("patience", 5))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    best_score = -1.0
    best_state = None
    bad = 0

    for ep in tqdm(range(1, epochs + 1), desc="Training shared layer-aware boundary"):
        boundary.train()
        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{epochs}", leave=False)
        for layer_id, h, y in pbar:
            h = h.to(device)
            y = y.to(device)
            logits = torch.empty_like(y)
            for i in range(h.size(0)):
                logits[i] = boundary.forward_from_h(h[i:i+1], int(layer_id[i].item())).squeeze(0)

            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(boundary.parameters(), grad_clip)
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        tr_acc = eval_acc(boundary, DataLoader(train_ds, batch_size=bs, shuffle=False), device)
        va_acc = eval_acc(boundary, val_dl, device) if val_dl is not None else float("nan")
        tqdm.write(f"[ep {ep:02d}] train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")

        score = va_acc if va_acc == va_acc else tr_acc 
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in boundary.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if (val_dl is not None) and (bad >= patience):
                tqdm.write(f"[Early Stop] patience={patience} reached.")
                break

    if best_state is not None:
        boundary.load_state_dict(best_state)

    if Xva_by_layer is not None:
        sign, m1, m0 = sign_calibration(boundary, Xva_by_layer, Yva, layers, device=device, batch_size=bs)
        print(f"[Sign Calibration] using VAL pooled logits: mean(y=1)={m1:.4f}, mean(y=0)={m0:.4f}, sign={sign:+.0f}")
    else:
        sign, m1, m0 = sign_calibration(boundary, Xtr_by_layer, Ytr, layers, device=device, batch_size=bs)
        print(f"[Sign Calibration] using TRAIN pooled logits: mean(y=1)={m1:.4f}, mean(y=0)={m0:.4f}, sign={sign:+.0f}")
    sg_cfg = cfg.get("sigma_g", {})
    max_samp = int(sg_cfg.get("max_samples_per_layer", 2048))
    sigma_g_by_layer = estimate_sigma_g_per_layer(boundary, Xtr_by_layer, layers, device=device, max_samples_per_layer=max_samp)
    sigma_g_global = float(torch.tensor(list(sigma_g_by_layer.values())).median().item())
    print(f"[Sigma_g] per-layer median ||grad_h F|| computed; global_median={sigma_g_global:.4f}")
    for l in layers:
        print(f"  layer {l:>2d}: sigma_g={sigma_g_by_layer[int(l)]:.4f}")


    torch.save(
        {
            "cfg": cfg,
            "layers": layers,
            "d": d,
            "r": r,
            "norm_type": norm_type,
            "film_de": film_de,
            "layer_id_to_index": layer_id_to_index,
            "state_dict": {k: v.cpu() for k, v in boundary.state_dict().items()},
            "sign": float(sign),
            "sigma_g_by_layer": sigma_g_by_layer,
            "sigma_g_global": sigma_g_global,
        },
        os.path.join(out_dir, "shared_boundary.pt"),
    )
    print(f"[Shared VFC] saved -> {out_dir}/shared_boundary.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg)
