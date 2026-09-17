"""Day106: Conditional Flow Matching action-conditioned latent predictor,
for the stomach-phantom navigation data (see README).

This is a fresh start, not a port of ../action-conditioned-video-prediction/'s
full history -- that project spent Day61-95 discovering (the hard way) that
(a) a from-scratch encoder on ~200 episodes doesn't reliably preserve enough
action-relevant signal, and (b) a frozen, ImageNet-pretrained ResNet18 does,
by a wide margin (Day95/96-98 there). Starting directly from a frozen
ResNet18 encoder here applies that lesson from day one instead of
re-discovering it. Likewise, this file only implements the "flatten the
action window" mode -- the GRU/Transformer/gated action-encoder variants
that project tried (Day79, 86, 88) were explorations of a problem (the real
action actively hurting predictions) that hasn't been established here yet.

Encoder: frozen, ImageNet-pretrained ResNet18 (identical role to
../action-conditioned-video-prediction/cfm_model.py's
PretrainedResNet18Encoder). Same encoder for both the current frame and the
future target frame -- frozen, so there's no online/target EMA pair, no
collapse to prevent.

Predictor: a small MLP velocity field over the same latent space, trained
with the standard Conditional Flow Matching objective (Lipman et al. 2022):
given x0 ~ N(0, I) and x1 = target latent, interpolate z_s = (1-s)x0 + s*x1
and regress the velocity field toward the constant velocity u = x1 - x0.
Sampling integrates dz/ds = v_theta(...) from s=0 to s=1 with Euler steps.
"""

import math
import sys
import types

# This machine's pyenv-built Python 3.11.5 was compiled without the `_lzma` C
# extension (xz was installed via Homebrew after that Python build, not
# before -- see Day106 notes). torchvision's package __init__ unconditionally
# imports torchvision.datasets, which imports an optical-flow dataset loader
# that needs `lzma` purely to read a benchmark file format we never touch.
# Rebuilding the interpreter (`pyenv install --force`) hit an unrelated
# ensurepip segfault tied to a deprecated openssl@1.1 dependency in that
# build recipe, and is shared by other projects in this repo -- not worth
# the risk to fix a module we don't use. Stub it out before the first
# torchvision import instead; nothing here ever calls into lzma.
if "lzma" not in sys.modules:
    _lzma_stub = types.ModuleType("lzma")
    _lzma_stub.open = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("lzma is stubbed out, see Day106 notes"))
    sys.modules["lzma"] = _lzma_stub

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PretrainedResNet18Encoder(nn.Module):
    """Frozen, ImageNet-pretrained ResNet18. embed_dim is fixed at 512
    (ResNet18's pooled feature dimension)."""

    embed_dim = 512

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.resnet = resnet
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # always eval mode: frozen batchnorm/dropout stats

    @torch.no_grad()
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(frame, size=224, mode="bilinear", align_corners=False)
        x = (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)
        return self.resnet(x)


def per_example_normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return ((pred - target) ** 2).sum(dim=-1)


def sinusoidal_time_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device).float() / half)
    args = s[:, None].float() * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityPredictor(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int, time_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.time_dim = time_dim
        in_dim = embed_dim + time_dim + embed_dim + action_dim  # z_s, time_emb, z_t, action
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, z_s: torch.Tensor, s: torch.Tensor, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(s, self.time_dim)
        return self.net(torch.cat([z_s, t_emb, z_t, action], dim=-1))


class CFMActionModel(nn.Module):
    def __init__(self, action_dim_per_step: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        shared = PretrainedResNet18Encoder()
        self.online_encoder = shared
        self.target_encoder = shared  # same frozen module; kept as two names for readability at call sites
        self.embed_dim = shared.embed_dim
        action_dim = action_dim_per_step * horizon
        self.velocity = VelocityPredictor(self.embed_dim, action_dim)

    def encode_action(self, action_window: torch.Tensor) -> torch.Tensor:
        return action_window.reshape(action_window.shape[0], -1)

    def training_step(self, frame_t: torch.Tensor, action_window: torch.Tensor, frame_t1: torch.Tensor):
        z_t = self.online_encoder(frame_t)
        with torch.no_grad():
            x1 = self.target_encoder(frame_t1)
        action = self.encode_action(action_window)
        x0 = torch.randn_like(x1)

        s = torch.rand(x1.shape[0], device=x1.device)
        z_s = (1 - s[:, None]) * x0 + s[:, None] * x1
        u_target = x1 - x0

        v_pred = self.velocity(z_s, s, z_t, action)
        return F.mse_loss(v_pred, u_target), z_t

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, action_window: torch.Tensor, steps: int = 16) -> torch.Tensor:
        action = self.encode_action(action_window)
        z = torch.randn(z_t.shape[0], self.embed_dim, device=z_t.device)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((z_t.shape[0],), i * dt, device=z_t.device)
            v = self.velocity(z, s, z_t, action)
            z = z + v * dt
        return z
