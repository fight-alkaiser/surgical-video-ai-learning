"""JEPA-style action-conditioned predictor: predict in latent space, not pixels.

Day 62 follow-up. The pixel-space model (model.py) kept losing to a trivial
copy baseline because reconstructing the whole frame from scratch pays a
"blur tax" on the ~95% of the frame that never moves. This model sidesteps
that entirely: there is no decoder, no pixel reconstruction, no pixel loss.

Architecture (standard JEPA/BYOL shape):
- online encoder:  frame_t     -> z_t       (gets gradients)
- target encoder:  frame_{t+H} -> z_{t+H}   (EMA copy of online encoder, no
                    gradients, output detached -- this is what prevents the
                    trivial "collapse to a constant" solution: the target
                    moves slower than the online encoder can chase it, so
                    mapping everything to the same constant can't track it)
- predictor:       (z_t, action) -> predicted z_{t+H}, trained to match the
                    (detached) target via cosine/normalized-MSE loss

No pixels are compared anywhere. If the action carries information about
what happens next, the predictor should be able to use it to get closer to
z_{t+H} than an action-blind guess -- and there's no "static background"
to hide behind, because the whole frame is compressed into one vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """frame (3,64,64) -> latent vector (embed_dim,)"""

    def __init__(self, base_ch: int = 32, embed_dim: int = 64):
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch, stride=2)  # 64 -> 32
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)  # 32 -> 16
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)  # 16 -> 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(base_ch * 4, embed_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = self.enc1(frame)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.pool(x).flatten(1)  # (B, base_ch*4)
        return self.proj(x)  # (B, embed_dim)


class Predictor(nn.Module):
    """(z_t, action) -> predicted z_{t+H}"""

    def __init__(self, embed_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, action], dim=-1))


class JEPAActionModel(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int = 64, base_ch: int = 32, ema_decay: float = 0.99):
        super().__init__()
        self.online_encoder = Encoder(base_ch, embed_dim)
        self.target_encoder = Encoder(base_ch, embed_dim)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(embed_dim, action_dim)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1 - self.ema_decay)

    def forward(self, frame_t: torch.Tensor, action: torch.Tensor, frame_t1: torch.Tensor):
        z_t = self.online_encoder(frame_t)
        pred_z_t1 = self.predictor(z_t, action)
        with torch.no_grad():
            target_z_t1 = self.target_encoder(frame_t1)
        return pred_z_t1, target_z_t1, z_t


def normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2-normalize both vectors before computing MSE (equivalent to 2 - 2*cosine_sim).
    Using normalized MSE, not raw MSE, so the model can't cheat by shrinking output norm."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return ((pred - target) ** 2).sum(dim=-1).mean()


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style anti-collapse term: penalize any embedding dimension whose
    std across the batch falls below `gamma`. EMA + stop-gradient alone did not
    prevent collapse in practice on this small dataset (verified empirically:
    z_std dropped to ~0.002 and loss went to ~0 within 10 epochs without this
    term) -- this term makes the trivial "map everything to a constant"
    solution actively costly, not just theoretically avoidable."""
    std = torch.sqrt(z.var(dim=0) + eps)  # (embed_dim,)
    return F.relu(gamma - std).mean()
