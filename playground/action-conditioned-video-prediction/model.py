"""Minimal action-conditioned next-frame predictor.

Design deliberately mirrors the conditioning mechanism found in
Cosmos-H-Surgical-Simulator's action_conditioned_minimal_v1_lvg_dit.py:
the action vector is embedded by an MLP and used to modulate (scale/shift)
an intermediate feature map, rather than being cross-attended per-token.
This is a FiLM-style conditioning, the same family as the AdaLN mechanism
Cosmos uses -- just applied to a tiny CNN bottleneck instead of a DiT's
timestep embedding.

This is NOT a diffusion model. It is a deterministic regressor trained
with pixel MSE. The goal is to reproduce the *conditioning mechanism*
at toy scale, not the generative model itself.
"""

import torch
import torch.nn as nn


class ActionEmbedder(nn.Module):
    """Action vector -> (scale, shift) for FiLM-style modulation, same shape idea as
    Cosmos's action_embedder_B_D / action_embedder_B_3D pair."""

    def __init__(self, action_dim: int, feature_channels: int):
        super().__init__()
        hidden = feature_channels * 2
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_channels * 2),  # scale + shift
        )
        self.feature_channels = feature_channels

    def forward(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(action)  # (B, 2C)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=stride, padding=1
        )
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ActionConditionedPredictor(nn.Module):
    """frame_t (3,64,64) + action_t (16,) -> predicted frame_{t+1} (3,64,64)."""

    def __init__(self, action_dim: int = 16, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch, stride=2)  # 64 -> 32
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)  # 32 -> 16
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)  # 16 -> 8

        bottleneck_ch = base_ch * 4
        # Day62 fix: the action_embedder now conditions the LAST feature map (after dec3,
        # base_ch channels), not the bottleneck. Modulating right before a GroupNorm let
        # that norm re-standardize the activations and erase the modulation entirely
        # (verified empirically: diff after modulation was ~0.28, diff after the next
        # GroupNorm-containing block was exactly 0.0). Cosmos avoids this by applying its
        # AdaLN modulation AFTER each block's normalization, not before it -- so here the
        # modulation is the last thing to happen before the final conv, with no norm layer
        # left downstream to wash it back out.
        self.action_embedder = ActionEmbedder(action_dim, base_ch)

        self.dec1 = DeconvBlock(bottleneck_ch, base_ch * 2, stride=2)  # 8 -> 16
        self.dec2 = DeconvBlock(base_ch * 2, base_ch, stride=2)  # 16 -> 32
        self.dec3 = DeconvBlock(base_ch, base_ch, stride=2)  # 32 -> 64
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        # zero-init so the model starts as an identity/copy predictor (delta ~ 0)
        # and has to learn to move away from "just copy the input frame"
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.enc1(frame)
        x = self.enc2(x)
        x = self.enc3(x)  # (B, C, 8, 8)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)  # (B, base_ch, 64, 64) -- last normalization already applied

        scale, shift = self.action_embedder(action)  # (B, base_ch) each
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, base_ch, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        x = x * (1 + scale) + shift  # FiLM / AdaLN-style modulation, nothing downstream to undo it

        delta = torch.tanh(self.out_conv(x))  # bounded change in [-1, 1]
        return torch.clamp(frame + delta, 0.0, 1.0)  # residual: copy + learned change
