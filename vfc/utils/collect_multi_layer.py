from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class BinDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _safe_max_len(tokenizer, model, fallback: int = 4096) -> int:
    ml = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(ml, int) and 0 < ml < 1_000_000:
        return ml
    tml = getattr(tokenizer, "model_max_length", None)
    if isinstance(tml, int) and 0 < tml < 1_000_000:
        return tml
    return fallback


@torch.no_grad()
def collect_activations_multi_layer(
    model,
    tokenizer,
    items,
    layers: List[int],
    batch_size: int = 4,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    X_by_layer = {int(l): [] for l in layers}
    Y = []

    dl = DataLoader(BinDS(items), batch_size=batch_size, shuffle=False)
    for batch in tqdm(dl, desc="Collecting multi-layer activations", total=len(dl)):
        texts = batch["text"]
        ys = batch["y"]
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_safe_max_len(tokenizer, model),
        ).to(model.device)

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  
        last_idx = enc["attention_mask"].sum(dim=1) - 1

        for l in layers:
            hL = hs[int(l) + 1]  # (B,T,H)
            gathered = hL[
                torch.arange(hL.size(0), device=hL.device),
                last_idx,
                :
            ]  # (B,H)
            X_by_layer[int(l)].append(gathered.detach().cpu().float())

        Y.extend(ys.tolist() if hasattr(ys, "tolist") else list(ys))

    for l in layers:
        X_by_layer[int(l)] = torch.cat(X_by_layer[int(l)], dim=0)

    Y = torch.tensor(Y, dtype=torch.float32)
    return X_by_layer, Y
