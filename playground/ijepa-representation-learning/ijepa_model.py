"""Day93: I-JEPA-style self-supervised representation learning, no action involved.

Day78-92 (../action-conditioned-video-prediction/) spent 15 days on a
different question -- does conditioning on the robot action help predict
z_{t+H} -- and landed on "probably a scale limit, not a fixable mistake"
(see that project's README, Day92 section). This is a deliberate pivot away
from that: no actions, no frame pairs across a horizon, just single frames
and the actual I-JEPA (Assran et al., 2023) objective -- predict the
representation of masked-out parts of an image from the visible parts, in
latent space, no pixel reconstruction, no decoder.

Unlike the CNN whole-image encoder used in ../action-conditioned-video-
prediction/jepa_model.py (one vector per frame, no internal structure), this
needs *patch*-level representations for masking to mean anything, so the
encoder here is a small Transformer over image patches instead of a CNN.

Architecture (I-JEPA's shape, kept small):
- patchify: (3, 64, 64) -> 64 patches of 8x8 pixels, each linearly embedded
  + a learned per-position embedding
- context encoder: small Transformer, sees only the *visible* (non-masked)
  patches -- gets gradients
- target encoder: same architecture, EMA copy of the context encoder, sees
  *all* patches (no masking) -- no gradients, output detached. This is what
  prevents collapse: the target moves slower than the context encoder can
  chase it (same story as the CNN version's online/target pair).
- predictor: given the context encoder's output plus learnable mask tokens
  (one per masked position, each carrying that position's embedding), predicts
  the target encoder's representation at every masked position
- loss: normalized MSE between predicted and (detached) target patch
  embeddings, averaged over masked patches -- plus the same VICReg-style
  variance term used throughout this series, as a safety net given this
  project's own history of representation collapse (Day61-62).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 64
PATCH_SIZE = 8
GRID = IMG_SIZE // PATCH_SIZE  # 8x8 = 64 patches
NUM_PATCHES = GRID * GRID
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3


def patchify(frame: torch.Tensor) -> torch.Tensor:
    """(B, 3, 64, 64) -> (B, NUM_PATCHES, PATCH_DIM)"""
    B = frame.shape[0]
    x = frame.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)  # (B,3,GRID,GRID,P,P)
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, NUM_PATCHES, PATCH_DIM)
    return x


class TinyTransformer(nn.Module):
    def __init__(self, embed_dim: int, depth: int, nhead: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, batch_first=True)
        self.net = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, key_padding_mask=None):
        return self.net(x, src_key_padding_mask=key_padding_mask)


class PatchEncoder(nn.Module):
    """patches (B, N, PATCH_DIM) + their position indices -> (B, N, embed_dim)"""

    def __init__(self, embed_dim: int = 64, depth: int = 3, nhead: int = 4):
        super().__init__()
        self.embed = nn.Linear(PATCH_DIM, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, NUM_PATCHES, embed_dim) * 0.02)
        self.transformer = TinyTransformer(embed_dim, depth, nhead)

    def forward(self, patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, PATCH_DIM); positions: (B, N) long indices into pos_embed."""
        h = self.embed(patches) + self.pos_embed[0, positions]
        return self.transformer(h)


class Predictor(nn.Module):
    """context tokens + mask tokens (at target positions) -> predicted target embeddings.

    Narrower than the encoders, as in the original I-JEPA -- the predictor's
    job is small (fill in a representation given context), not to re-derive
    the whole image.
    """

    def __init__(self, embed_dim: int = 64, pred_dim: int = 48, depth: int = 2, nhead: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, pred_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, NUM_PATCHES, pred_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_dim) * 0.02)
        self.transformer = TinyTransformer(pred_dim, depth, nhead)
        self.out_proj = nn.Linear(pred_dim, embed_dim)

    def forward(self, context_tokens: torch.Tensor, context_positions: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        """context_tokens: (B, Nc, embed_dim) from the context encoder.
        context_positions: (B, Nc) long. target_positions: (B, Nt) long.
        Returns predicted target-encoder-space embeddings: (B, Nt, embed_dim)."""
        B, Nt = target_positions.shape
        ctx = self.in_proj(context_tokens) + self.pos_embed[0, context_positions]
        mask = self.mask_token.expand(B, Nt, -1) + self.pos_embed[0, target_positions]
        h = torch.cat([ctx, mask], dim=1)
        h = self.transformer(h)
        pred = h[:, ctx.shape[1] :, :]  # only the (formerly mask-token) target positions
        return self.out_proj(pred)


class IJEPAModel(nn.Module):
    def __init__(self, embed_dim: int = 64, enc_depth: int = 3, pred_depth: int = 2, ema_decay: float = 0.996):
        super().__init__()
        self.context_encoder = PatchEncoder(embed_dim, depth=enc_depth)
        self.target_encoder = PatchEncoder(embed_dim, depth=enc_depth)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(embed_dim)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for online_p, target_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1 - self.ema_decay)

    def forward(self, frame: torch.Tensor, context_positions: torch.Tensor, target_positions: torch.Tensor):
        """context_positions / target_positions: (B, Nc) / (B, Nt) long indices,
        assumed same Nc/Nt across the batch (fixed mask shape per batch)."""
        all_patches = patchify(frame)  # (B, NUM_PATCHES, PATCH_DIM)
        B = frame.shape[0]
        ctx_patches = torch.gather(
            all_patches, 1, context_positions.unsqueeze(-1).expand(-1, -1, PATCH_DIM)
        )
        ctx_tokens = self.context_encoder(ctx_patches, context_positions)
        pred_target = self.predictor(ctx_tokens, context_positions, target_positions)

        with torch.no_grad():
            all_idx = torch.arange(NUM_PATCHES, device=frame.device).unsqueeze(0).expand(B, -1)
            full_tokens = self.target_encoder(all_patches, all_idx)
            true_target = torch.gather(full_tokens, 1, target_positions.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1]))

        return pred_target, true_target, ctx_tokens


def normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return ((pred - target) ** 2).sum(dim=-1).mean()


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Same VICReg-style anti-collapse term as the rest of this series
    (jepa_model.py). z is flattened over any leading dims except the last.

    Day93 note: this checks spread *across the batch*, pooling away any other
    leading dimensions first. For patch tokens (B, N, D), that means it's
    satisfied as long as different *images* produce different embeddings --
    it says nothing about whether different *patches within the same image*
    are distinguishable, which turned out to be exactly the axis this
    project's encoder collapsed on. Use within_image_variance_loss for that."""
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def within_image_variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Day93 fix: penalizes low variance *across patches, within each image*
    (z: (B, N, D)), instead of pooling batch and patch dimensions together.
    Directly targets the collapse mode variance_loss missed here -- different
    images producing different embeddings on average, while every patch
    *within* one image still maps to the same vector regardless of position."""
    std = torch.sqrt(z.var(dim=1) + eps)  # (B, D) -- std across the N patches, per image
    return F.relu(gamma - std).mean()
