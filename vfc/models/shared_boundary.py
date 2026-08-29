import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_hidden(h: torch.Tensor, norm_type: str = "rms", eps: float = 1e-6) -> torch.Tensor:
    if norm_type == "rms":
        rms = torch.sqrt(torch.mean(h * h, dim=-1, keepdim=True) + eps)
        return h / rms
    elif norm_type == "layernorm":
        mu = torch.mean(h, dim=-1, keepdim=True)
        var = torch.mean((h - mu) ** 2, dim=-1, keepdim=True)
        return (h - mu) / torch.sqrt(var + eps)
    else:
        raise ValueError(f"Unknown norm_type={norm_type}")


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [1]
        mods = []
        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.GELU())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
        mods.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SharedLayerAwareBoundary(nn.Module):
    def __init__(
        self,
        d: int,
        r: int,
        num_layers: int,
        layer_id_to_index: dict,
        norm_type: str,
        film_de: int,
        mlp_hidden: int,
        mlp_layers: int,
        mlp_dropout: float,
    ):
        super().__init__()
        self.d = d
        self.r = r
        self.norm_type = norm_type
        self.layer_id_to_index = layer_id_to_index
        self.index_to_layer_id = {v: k for k, v in layer_id_to_index.items()}
        self.R = nn.Parameter(torch.empty(r, d))
        nn.init.orthogonal_(self.R) 

        self.layer_emb = nn.Embedding(num_layers, film_de)
        nn.init.normal_(self.layer_emb.weight, mean=0.0, std=0.02)
        self.W_gamma = nn.Linear(film_de, r, bias=False)
        self.W_beta = nn.Linear(film_de, r, bias=False)

        self.f = SmallMLP(r, mlp_hidden, mlp_layers, mlp_dropout)

    def forward_from_h(self, h: torch.Tensor, layer_id: int) -> torch.Tensor:
        h_hat = norm_hidden(h, norm_type=self.norm_type)
        u = F.linear(h_hat, self.R)  # [B, r]

        li = self.layer_id_to_index[int(layer_id)]
        li_t = torch.full((u.size(0),), li, device=u.device, dtype=torch.long)
        e = self.layer_emb(li_t)  # [B, de]
        gamma = self.W_gamma(e)   # [B, r]
        beta = self.W_beta(e)     # [B, r]
        u_tilde = (1.0 + gamma) * u + beta

        return self.f(u_tilde)

    def forward_from_u(self, u: torch.Tensor, layer_index: torch.Tensor) -> torch.Tensor:
        e = self.layer_emb(layer_index)
        gamma = self.W_gamma(e)
        beta = self.W_beta(e)
        u_tilde = (1.0 + gamma) * u + beta
        return self.f(u_tilde)
