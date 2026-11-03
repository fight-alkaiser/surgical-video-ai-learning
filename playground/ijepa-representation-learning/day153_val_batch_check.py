"""Day153 follow-up: is val SIGReg high because val batches are consecutive frames?

The training loop draws val batches as np.arange(i, i+64) over 4 concatenated val
episodes, i.e. 64 consecutive (near-identical) frames of one episode. SIGReg tests
whether a batch looks like an isotropic Gaussian, which assumes i.i.d. samples --
so such a batch should score badly even for a healthy encoder. Re-score the saved
checkpoints with consecutive vs shuffled val batches (and across-image cos sim on
shuffled frames instead of the 8 consecutive probe frames).
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.argv = [sys.argv[0], "--anti-collapse", "sigreg"]  # reuse the train script's data split
import importlib.util
from ijepa_model import IJEPAModel, patchify, sigreg_loss

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
src = open("ijepa_train.py").read()
# only the data-loading prefix of the train script (up to the model definition)
exec(src[: src.index("model = IJEPAModel")])

all_idx = torch.arange(64, device=DEVICE)


@torch.no_grad()
def score(model, order, on):
    sigs, coss = [], []
    for i in range(0, len(order) - 63, 64):
        idx = order[i : i + 64]
        f = to_tensor(val_frames, idx)
        tok = model.context_encoder(patchify(f), all_idx.unsqueeze(0).expand(len(idx), -1))
        z = tok.mean(1) if on == "image" else tok.reshape(-1, tok.shape[-1])
        g = torch.Generator(device=DEVICE).manual_seed(0)
        sigs.append(sigreg_loss(z, num_slices=256, generator=g).item())
        v = F.normalize(tok.mean(1), dim=-1)
        s = v @ v.T
        coss.append(s[~torch.eye(len(idx), dtype=torch.bool, device=DEVICE)].mean().item())
    return np.mean(sigs), np.mean(coss)


rng = np.random.default_rng(0)
consecutive = np.arange(len(val_frames))
shuffled = rng.permutation(len(val_frames))
print(f"{'checkpoint':38s} {'SIGReg consec':>14s} {'SIGReg shuffled':>16s} {'cos consec':>11s} {'cos shuffled':>13s}")
for on in ["image", "patch"]:
    for ep in [0, 10, 20, 29]:
        model = IJEPAModel().to(DEVICE)
        model.load_state_dict(torch.load(f"outputs/checkpoints/model_ijepa_seed0_sigreg-{on}_lam0.05_epoch{ep:04d}.pt", map_location=DEVICE))
        model.eval()
        sc, cc = score(model, consecutive, on)
        ss, cs = score(model, shuffled, on)
        print(f"sigreg-{on} epoch {ep:2d}{'':20s} {sc:14.2f} {ss:16.2f} {cc:11.3f} {cs:13.3f}")
