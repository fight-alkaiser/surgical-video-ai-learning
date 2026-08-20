"""Day78: Conditional Flow Matching on top of the Day62 JEPA latent space.

The Day62 JEPA predictor (jepa_model.py) regresses directly to a single
z_{t+H} vector with normalized MSE. That is a deterministic regression: given
the same (z_t, action), the optimal MSE output is the *average* of every
future that could follow. On real action-conditioned surgical video, the
future is not perfectly determined by a 16-dim action vector (small
instrument jitter, camera noise, etc.), so the model has an incentive to
collapse toward "barely change anything" -- which is exactly what happened:
real-action loss tied the do-nothing baseline (see README, Day62 section).

This file keeps the same online/target encoder pair (same collapse-prevention
story: EMA target + stop-gradient + VICReg variance term), but replaces the
single-vector predictor with a velocity field, trained with the standard
Conditional Flow Matching objective (linear/OT interpolant, as in Lipman et
al. 2022 and the Rectified Flow formulation):

    x0 ~ N(0, I)                         (source / prior sample)
    x1 = z_{t+H}                         (target, from the EMA target encoder)
    z_s = (1-s) * x0 + s * x1,  s ~ U(0,1)
    u   = x1 - x0                        (the constant velocity along this path)
    loss = || v_theta(z_s, s, z_t, action) - u ||^2

At inference, integrating dz/ds = v_theta(...) from s=0 (noise) to s=1 with
an ODE solver produces a *sample* of z_{t+H}, conditioned on (z_t, action).
Because it's a sample rather than a single deterministic output, drawing
several samples for the same (z_t, action) and looking at their spread lets
us ask a question the Day62 model could not answer at all: does the action
actually narrow down the distribution of what happens next, or is the model
just as uncertain regardless of which action it's given?
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_model import Encoder, variance_loss  # noqa: F401  (variance_loss re-exported for cfm_train.py)


def per_example_normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Same normalization as jepa_model.normalized_mse_loss, but returns one
    value per row instead of averaging over the batch -- needed for best-of-N
    evaluation, where we take a min *per example* across samples before
    averaging over the batch."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return ((pred - target) ** 2).sum(dim=-1)


def sinusoidal_time_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer/DDPM-style sinusoidal embedding of a scalar in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device).float() / half)
    args = s[:, None].float() * freqs[None, :] * 1000.0  # scale up so low frequencies aren't wasted on [0,1]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityPredictor(nn.Module):
    """(z_s, s, z_t, action) -> velocity, i.e. the flow-matching predictor.

    Replaces jepa_model.py's Predictor (which mapped (z_t, action) -> z_{t+H}
    directly). z_s is the noisy/interpolated latent at flow-time s; z_t and
    action are the conditioning signal, unchanged from the JEPA setup.
    """

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
    def __init__(self, action_dim: int, embed_dim: int = 64, base_ch: int = 32, ema_decay: float = 0.99):
        super().__init__()
        self.embed_dim = embed_dim
        self.online_encoder = Encoder(base_ch, embed_dim)
        self.target_encoder = Encoder(base_ch, embed_dim)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.velocity = VelocityPredictor(embed_dim, action_dim)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1 - self.ema_decay)

    def training_step(self, frame_t: torch.Tensor, action: torch.Tensor, frame_t1: torch.Tensor, source: str = "noise"):
        """One CFM training step. Returns (velocity_loss, z_t) -- z_t is exposed
        so the caller can add the anti-collapse variance_loss on it, exactly as
        jepa_train.py does.

        `source` picks x0, the start point of the flow:
        - "noise": x0 ~ N(0, I), the standard Conditional Flow Matching setup.
          The predictor has to learn both "roughly where z_{t+H} is" and "the
          path from noise to it" at once.
        - "zt": x0 = z_t itself (a Rectified-Flow-style residual formulation).
          The path only has to carry the *change* implied by the action, since
          "no change at all" is already the s=0 point of the path -- structurally
          closer to how model.py's residual/delta prediction fixed the Day61
          blur-tax problem in pixel space.
        """
        z_t = self.online_encoder(frame_t)
        with torch.no_grad():
            x1 = self.target_encoder(frame_t1)  # target latent, detached
        x0 = z_t.detach() if source == "zt" else torch.randn_like(x1)
        s = torch.rand(x1.shape[0], device=x1.device)
        z_s = (1 - s[:, None]) * x0 + s[:, None] * x1
        u_target = x1 - x0
        v_pred = self.velocity(z_s, s, z_t, action)
        velocity_loss = F.mse_loss(v_pred, u_target)
        return velocity_loss, z_t

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, action: torch.Tensor, steps: int = 16, source: str = "noise") -> torch.Tensor:
        """Integrate dz/ds = v_theta(z_s, s, z_t, action) from s=0 to s=1 with a
        simple Euler solver, returning a sample of z_{t+H}. `source` must match
        what the model was trained with (see training_step)."""
        z = z_t.clone() if source == "zt" else torch.randn(z_t.shape[0], self.embed_dim, device=z_t.device)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((z_t.shape[0],), i * dt, device=z_t.device)
            v = self.velocity(z, s, z_t, action)
            z = z + v * dt
        return z
