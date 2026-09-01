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


class ActionSequenceEncoder(nn.Module):
    """Day86: encodes the (H, action_dim_per_step) action window with a GRU,
    instead of flattening it into one H*action_dim_per_step vector.

    Flattening (what this project used through Day85) and CHSS's own action
    chunking both treat the window as an unordered bag of H*D numbers -- the
    network has to rediscover from scratch, if it can, that dims 0:16 come
    before dims 16:32. A GRU is given that order for free: it reads the
    window one timestep at a time and its final hidden state is the action
    embedding. Whether that ordering actually helps this predictor is the
    open question this file exists to test.
    """

    def __init__(self, action_dim_per_step: int, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(action_dim_per_step, hidden, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, action_seq: torch.Tensor) -> torch.Tensor:
        # action_seq: (B, H, action_dim_per_step) -> (B, out_dim)
        _, h_n = self.gru(action_seq)
        return self.out(h_n[-1])


class ActionTransformerEncoder(nn.Module):
    """Day88: encodes the (H, action_dim_per_step) action window with a small
    Transformer encoder, following the design ACT's decoder and pi0's Action
    Expert actually use for chunks -- self-attention across the whole window
    at once, not the step-by-step recurrence the Day86 GRU used. Every
    timestep can attend to every other timestep directly, rather than being
    forced through a single hidden-state bottleneck carried step to step.

    Each step is linearly embedded, given a learned positional embedding
    (order is not implicit here the way it is for a GRU -- attention itself
    is permutation-invariant), passed through a couple of self-attention
    layers, then mean-pooled across the H steps into one action embedding.
    """

    def __init__(
        self,
        action_dim_per_step: int,
        horizon: int,
        hidden: int = 64,
        out_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed = nn.Linear(action_dim_per_step, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, action_seq: torch.Tensor) -> torch.Tensor:
        # action_seq: (B, H, action_dim_per_step) -> (B, out_dim)
        h = self.embed(action_seq) + self.pos_embed
        h = self.transformer(h)
        return self.out(h.mean(dim=1))


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


class GatedVelocityPredictor(nn.Module):
    """Day79 follow-up to Day78's finding: the action pathway reads real signal
    (real action explains the observed transition better than a shuffled one)
    but is still net harmful (using it beats a zeroed-out action -- worse, not
    better, than not conditioning at all). A plain concat+MLP like
    VelocityPredictor can in principle learn to ignore action inputs it
    doesn't trust, but nothing about the architecture makes that easy: the
    action features are mixed into every hidden unit from the first layer
    onward, with no dedicated "how much should I trust this" pathway.

    This version gives the network an explicit, low-friction opt-out: the
    action is embedded separately and injected additively through a learned
    sigmoid gate, conditioned on the current hidden state and the action
    itself. The gate's bias is initialized negative so training starts with
    the action mostly gated off (close to the Day78 "zero action" regime,
    which was the best-performing condition) and has to actively learn to
    open the gate where the action is worth using, rather than starting from
    "always fully mixed in" and having to learn to suppress it.
    """

    def __init__(self, embed_dim: int, action_dim: int, time_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.time_dim = time_dim
        base_in = embed_dim + time_dim + embed_dim  # z_s, time_emb, z_t -- no action here
        self.base = nn.Sequential(nn.Linear(base_in, hidden), nn.GELU())
        self.action_embed = nn.Sequential(nn.Linear(action_dim, hidden), nn.GELU())
        self.gate = nn.Linear(hidden + action_dim, hidden)
        nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2) ~ 0.12: mostly closed at init
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward_with_gate(self, z_s: torch.Tensor, s: torch.Tensor, z_t: torch.Tensor, action: torch.Tensor):
        t_emb = sinusoidal_time_embedding(s, self.time_dim)
        h = self.base(torch.cat([z_s, t_emb, z_t], dim=-1))
        a_emb = self.action_embed(action)
        gate = torch.sigmoid(self.gate(torch.cat([h, action], dim=-1)))
        h = h + gate * a_emb
        return self.out(h), gate

    def forward(self, z_s: torch.Tensor, s: torch.Tensor, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        v, _ = self.forward_with_gate(z_s, s, z_t, action)
        return v


class CFMActionModel(nn.Module):
    def __init__(
        self,
        action_dim_per_step: int,
        horizon: int,
        embed_dim: int = 64,
        base_ch: int = 32,
        ema_decay: float = 0.99,
        gated: bool = False,
        action_mode: str = "flatten",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.action_mode = action_mode
        self.online_encoder = Encoder(base_ch, embed_dim)
        self.target_encoder = Encoder(base_ch, embed_dim)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if action_mode == "sequence":
            self.action_encoder = ActionSequenceEncoder(action_dim_per_step, out_dim=embed_dim)
            action_dim = embed_dim
        elif action_mode == "transformer":
            self.action_encoder = ActionTransformerEncoder(action_dim_per_step, horizon, out_dim=embed_dim)
            action_dim = embed_dim
        else:
            self.action_encoder = None
            action_dim = action_dim_per_step * horizon

        self.velocity = GatedVelocityPredictor(embed_dim, action_dim) if gated else VelocityPredictor(embed_dim, action_dim)
        self.gated = gated
        self.ema_decay = ema_decay

    def encode_action(self, action_window: torch.Tensor) -> torch.Tensor:
        """action_window: (B, H, action_dim_per_step), raw (already-normalized)
        actions -- always this shape regardless of action_mode, so callers
        never need to know whether flattening or the GRU happens inside."""
        if self.action_encoder is not None:
            return self.action_encoder(action_window)
        return action_window.reshape(action_window.shape[0], -1)

    @torch.no_grad()
    def mean_gate(self, z_t: torch.Tensor, action_window: torch.Tensor) -> float:
        """Diagnostic only (gated model): average gate value at a fixed
        reference point (z_s = z_t, s = 0.5), for comparing how open the gate
        is for real vs. shuffled vs. zero action."""
        if not self.gated:
            return float("nan")
        action = self.encode_action(action_window)
        s = torch.full((z_t.shape[0],), 0.5, device=z_t.device)
        _, gate = self.velocity.forward_with_gate(z_t, s, z_t, action)
        return gate.mean().item()

    @torch.no_grad()
    def update_target(self):
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1 - self.ema_decay)

    def training_step(
        self,
        frame_t: torch.Tensor,
        action_window: torch.Tensor,
        frame_t1: torch.Tensor,
        source: str = "noise",
        self_forcing_prob: float = 0.0,
        self_forcing_steps: int = 4,
    ):
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

        Day90: `self_forcing_prob` > 0 trains on *the model's own partially-
        integrated rollouts* some fraction of the time, instead of always on
        exact points on the straight x0-x1 line. Day81-82 found that sampling
        (repeatedly applying the still-imperfect velocity field with an Euler
        integrator) introduces a small, consistent drift that plain CFM
        training can't see, because training only ever shows the model exact
        interpolated points -- points it never actually visits once its own
        field has any error. This is the same exposure-bias story CHSS's Self
        Forcing distillation targets (train the student on its own generated
        rollouts, not ground truth). Here, with probability
        `self_forcing_prob`, z_s is instead produced by running 1..
        self_forcing_steps Euler steps of the model's *current* (detached)
        velocity field starting from x0, and the target velocity is corrected
        to point from wherever that rollout actually landed straight at x1 in
        the remaining time -- (x1 - z_s) / (1 - s) -- rather than the fixed
        x1 - x0 direction of the ideal line, which is generally wrong once
        z_s is already off that line.
        """
        z_t = self.online_encoder(frame_t)
        with torch.no_grad():
            x1 = self.target_encoder(frame_t1)  # target latent, detached
        action = self.encode_action(action_window)
        x0 = z_t.detach() if source == "zt" else torch.randn_like(x1)

        if self_forcing_prob > 0 and torch.rand(()).item() < self_forcing_prob:
            # k stops short of self_forcing_steps so s_val < 1 always -- otherwise
            # (1 - s_val) -> 0 and the corrective target below blows up.
            k = torch.randint(1, self_forcing_steps, (1,)).item()
            dt = 1.0 / self_forcing_steps
            with torch.no_grad():
                z = x0.clone()
                s_val = 0.0
                for i in range(k):
                    s_i = torch.full((x1.shape[0],), s_val, device=x1.device)
                    v = self.velocity(z, s_i, z_t, action)
                    z = z + v * dt
                    s_val += dt
            z_s = z  # the model's own (possibly off-line) rollout state, detached
            s = torch.full((x1.shape[0],), s_val, device=x1.device)
            u_target = (x1 - z_s) / max(1e-3, 1 - s_val)  # velocity that still reaches x1 exactly at s=1
        else:
            s = torch.rand(x1.shape[0], device=x1.device)
            z_s = (1 - s[:, None]) * x0 + s[:, None] * x1
            u_target = x1 - x0

        v_pred = self.velocity(z_s, s, z_t, action)
        velocity_loss = F.mse_loss(v_pred, u_target)
        return velocity_loss, z_t

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, action_window: torch.Tensor, steps: int = 16, source: str = "noise") -> torch.Tensor:
        """Integrate dz/ds = v_theta(z_s, s, z_t, action) from s=0 to s=1 with a
        simple Euler solver, returning a sample of z_{t+H}. `source` must match
        what the model was trained with (see training_step)."""
        action = self.encode_action(action_window)
        z = z_t.clone() if source == "zt" else torch.randn(z_t.shape[0], self.embed_dim, device=z_t.device)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((z_t.shape[0],), i * dt, device=z_t.device)
            v = self.velocity(z, s, z_t, action)
            z = z + v * dt
        return z
