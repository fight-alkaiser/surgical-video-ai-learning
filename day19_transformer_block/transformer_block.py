import json
import random
from pathlib import Path
from collections import Counter

import numpy as np

# ----------------------------------------
# Parameters
#
# Same 50 videos, same state segmentation, same
# vocabulary, and same train/test split as Day14/16/17/18.
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42

EMBED_DIM = 32
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
FF_DIM = 64
NUM_EPOCHS = 40
LEARNING_RATE = 0.1
GRAD_CLIP = 5.0
LAYER_NORM_EPS = 1e-5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------
# Data prep (identical to Day14/16/17/18)
# ----------------------------------------


def jaccard_similarity(set1, set2):
    if len(set1 | set2) == 0:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def build_frame_data(data):

    instrument_dict = data["categories"]["instrument"]
    verb_dict = data["categories"]["verb"]
    target_dict = data["categories"]["target"]
    phase_dict = data["categories"]["phase"]

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    frame_triplets = {}
    frame_phases = {}

    for frame in frame_ids:

        triplet_set = set()

        for triplet in data["annotations"][str(frame)]:
            if -1 in (triplet[1], triplet[7], triplet[8]):
                continue
            instrument = instrument_dict[str(triplet[1])]
            verb = verb_dict[str(triplet[7])]
            target = target_dict[str(triplet[8])]
            if verb == "null_verb":
                continue
            triplet_set.add((instrument, verb, target))

        frame_triplets[frame] = frozenset(triplet_set)

        phase_id = data["annotations"][str(frame)][0][-1]
        frame_phases[frame] = (
            phase_dict[str(phase_id)] if phase_id != -1 else None
        )

    return frame_ids, frame_triplets, frame_phases


def segment_into_states(frame_ids, frame_triplets, frame_phases, threshold):

    states = [frame_triplets[frame_ids[0]]]
    phase_votes = [Counter([frame_phases[frame_ids[0]]])]

    for frame in frame_ids[1:]:
        similarity = jaccard_similarity(states[-1], frame_triplets[frame])
        if similarity < threshold:
            states.append(frame_triplets[frame])
            phase_votes.append(Counter([frame_phases[frame]]))
        else:
            phase_votes[-1][frame_phases[frame]] += 1

    segment_phases = [v.most_common(1)[0][0] for v in phase_votes]
    return states, segment_phases


def load_state_sequence(video_path):
    with open(video_path, "r") as f:
        data = json.load(f)
    frame_ids, frame_triplets, frame_phases = build_frame_data(data)
    return segment_into_states(
        frame_ids, frame_triplets, frame_phases, SIMILARITY_THRESHOLD
    )


video_paths = sorted(LABELS_DIR.glob("VID*.json"))

video_states = {}
video_segment_phases = {}

for path in video_paths:
    states, segment_phases = load_state_sequence(path)
    video_states[path.stem] = states
    video_segment_phases[path.stem] = segment_phases

signature_to_id = {}
for video_id in sorted(video_states):
    for signature in video_states[video_id]:
        if signature not in signature_to_id:
            signature_to_id[signature] = len(signature_to_id)

vocab_size = len(signature_to_id)

video_id_sequences = {
    video_id: [signature_to_id[s] for s in signatures]
    for video_id, signatures in video_states.items()
}

all_video_ids = sorted(video_id_sequences.keys())
shuffled_ids = all_video_ids[:]
random.seed(RANDOM_SEED)
random.shuffle(shuffled_ids)

num_test = max(1, round(len(shuffled_ids) * TEST_RATIO))
test_ids = sorted(shuffled_ids[:num_test])
train_ids = sorted(shuffled_ids[num_test:])

max_len = max(len(seq) for seq in video_id_sequences.values())

position = np.arange(max_len)[:, None]
div_term = np.exp(np.arange(0, EMBED_DIM, 2) * (-np.log(10000.0) / EMBED_DIM))
positional_encoding = np.zeros((max_len, EMBED_DIM))
positional_encoding[:, 0::2] = np.sin(position * div_term)
positional_encoding[:, 1::2] = np.cos(position * div_term)

# ----------------------------------------
# A single Transformer (decoder) block
#
# Day18 showed that a bare attention layer -- a weighted
# average of value vectors, nothing else -- underperforms
# the RNN. A real Transformer block adds three things on
# top of the attention Day18 already had:
#
#   1. Multiple heads: NUM_HEADS independent attention
#      computations in parallel subspaces, concatenated and
#      mixed by an output projection Wo -- lets the model
#      attend to several different kinds of relationships
#      at once, instead of a single shared attention pattern.
#   2. A position-wise feed-forward network (Linear -> ReLU
#      -> Linear) applied after attention -- the nonlinear
#      processing step Day18's attention lacked entirely.
#   3. Residual connections + Layer Normalization around
#      both the attention and the feed-forward sublayers --
#      what makes it practical to add depth/nonlinearity
#      without the optimization difficulty that plain
#      stacking would introduce.
#
#   MHA_out  = MultiHeadAttention(Xp)
#   Z1       = LayerNorm(Xp + MHA_out)
#   FFN_out  = Linear2(ReLU(Linear1(Z1)))
#   Z2       = LayerNorm(Z1 + FFN_out)
#   logits   = Z2 @ Why.T + by
#
# Forward pass, backward pass (through both LayerNorms, the
# feed-forward network, and multi-head attention), and SGD
# updates are all written out by hand -- no autograd. The
# backward pass was verified against numerical gradients on
# a small synthetic example before running on real data (see
# this day's README).
# ----------------------------------------

E = np.random.randn(vocab_size, EMBED_DIM) / np.sqrt(EMBED_DIM)
Wq = np.random.randn(EMBED_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
Wo = np.random.randn(EMBED_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
gamma1 = np.ones(EMBED_DIM)
beta1 = np.zeros(EMBED_DIM)
W1 = np.random.randn(EMBED_DIM, FF_DIM) / np.sqrt(EMBED_DIM)
b1 = np.zeros(FF_DIM)
W2 = np.random.randn(FF_DIM, EMBED_DIM) / np.sqrt(FF_DIM)
b2 = np.zeros(EMBED_DIM)
gamma2 = np.ones(EMBED_DIM)
beta2 = np.zeros(EMBED_DIM)
Why = np.random.randn(vocab_size, EMBED_DIM) / np.sqrt(EMBED_DIM)
by = np.zeros(vocab_size)


def softmax_rows(x):
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def layer_norm_forward(x, gamma, beta):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    std = np.sqrt(var + LAYER_NORM_EPS)
    xhat = (x - mu) / std
    y = gamma * xhat + beta
    return y, (xhat, std)


def layer_norm_backward(dy, cache, gamma):
    xhat, std = cache
    dbeta = dy.sum(axis=0)
    dgamma = (dy * xhat).sum(axis=0)
    dxhat = dy * gamma
    dx = (1.0 / std) * (
        dxhat
        - dxhat.mean(axis=1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=1, keepdims=True)
    )
    return dx, dgamma, dbeta


def mha_forward(Xp, causal_mask):

    Q = Xp @ Wq
    K = Xp @ Wk
    V = Xp @ Wv

    head_outs = []
    head_caches = []

    for h in range(NUM_HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        Qh, Kh, Vh = Q[:, sl], K[:, sl], V[:, sl]
        scores = (Qh @ Kh.T) / np.sqrt(HEAD_DIM)
        scores = np.where(causal_mask, -1e9, scores)
        weights = softmax_rows(scores)
        out_h = weights @ Vh
        head_outs.append(out_h)
        head_caches.append((Qh, Kh, Vh, weights))

    concat = np.concatenate(head_outs, axis=1)
    mha_out = concat @ Wo

    return mha_out, (Xp, Q, K, V, head_caches, concat)


def mha_backward(dmha_out, cache):

    Xp, Q, K, V, head_caches, concat = cache

    dWo = concat.T @ dmha_out
    dconcat = dmha_out @ Wo.T

    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    for h in range(NUM_HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        Qh, Kh, Vh, weights = head_caches[h]
        dhead_out = dconcat[:, sl]

        dweights = dhead_out @ Vh.T
        dVh = weights.T @ dhead_out

        row_dot = (dweights * weights).sum(axis=1, keepdims=True)
        dscores = weights * (dweights - row_dot)
        dscores /= np.sqrt(HEAD_DIM)

        dQ[:, sl] = dscores @ Kh
        dK[:, sl] = dscores.T @ Qh
        dV[:, sl] = dVh

    dWq = Xp.T @ dQ
    dWk = Xp.T @ dK
    dWv = Xp.T @ dV
    dXp = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    return dXp, dWq, dWk, dWv, dWo


def forward_sequence(sequence):

    n = len(sequence) - 1
    ids = np.array(sequence[:n])
    targets = np.array(sequence[1:n + 1])

    X = E[ids]
    Xp = X + positional_encoding[:n]

    causal_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    mha_out, mha_cache = mha_forward(Xp, causal_mask)
    residual1 = Xp + mha_out
    Z1, ln1_cache = layer_norm_forward(residual1, gamma1, beta1)

    pre_ffn = Z1 @ W1 + b1
    H1 = np.maximum(pre_ffn, 0)
    ffn_out = H1 @ W2 + b2

    residual2 = Z1 + ffn_out
    Z2, ln2_cache = layer_norm_forward(residual2, gamma2, beta2)

    logits = Z2 @ Why.T + by
    probs = softmax_rows(logits)
    loss = -np.log(probs[np.arange(n), targets] + 1e-12).mean()

    cache = (ids, targets, mha_cache, ln1_cache, Z1, pre_ffn, H1,
              ln2_cache, Z2, probs, n)
    return cache, loss


def backward_sequence(cache):

    (ids, targets, mha_cache, ln1_cache, Z1, pre_ffn, H1,
     ln2_cache, Z2, probs, n) = cache

    dlogits = probs.copy()
    dlogits[np.arange(n), targets] -= 1
    dlogits /= n

    dWhy = dlogits.T @ Z2
    dby = dlogits.sum(axis=0)
    dZ2 = dlogits @ Why

    dresidual2, dgamma2, dbeta2 = layer_norm_backward(dZ2, ln2_cache, gamma2)

    dffn_out = dresidual2
    dZ1_a = dresidual2

    dW2 = H1.T @ dffn_out
    db2 = dffn_out.sum(axis=0)
    dH1 = dffn_out @ W2.T
    dpre_ffn = dH1 * (pre_ffn > 0)
    dW1 = Z1.T @ dpre_ffn
    db1 = dpre_ffn.sum(axis=0)
    dZ1_b = dpre_ffn @ W1.T

    dZ1_total = dZ1_a + dZ1_b

    dresidual1, dgamma1, dbeta1 = layer_norm_backward(dZ1_total, ln1_cache, gamma1)

    dmha_out = dresidual1
    dXp_a = dresidual1

    dXp_b, dWq, dWk, dWv, dWo = mha_backward(dmha_out, mha_cache)
    dXp_total = dXp_a + dXp_b

    dE = np.zeros_like(E)
    np.add.at(dE, ids, dXp_total)

    grads = dict(
        E=dE, Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo,
        gamma1=dgamma1, beta1=dbeta1,
        W1=dW1, b1=db1, W2=dW2, b2=db2,
        gamma2=dgamma2, beta2=dbeta2,
        Why=dWhy, by=dby,
    )
    for grad in grads.values():
        np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

    return grads


params = dict(
    E=E, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
    gamma1=gamma1, beta1=beta1,
    W1=W1, b1=b1, W2=W2, b2=b2,
    gamma2=gamma2, beta2=beta2,
    Why=Why, by=by,
)

# ----------------------------------------
# Train with per-video forward/backward + SGD
# ----------------------------------------

loss_history = []

for epoch in range(NUM_EPOCHS):

    epoch_video_ids = train_ids[:]
    random.shuffle(epoch_video_ids)

    total_loss = 0.0
    total_videos = 0

    for video_id in epoch_video_ids:

        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue

        cache, loss = forward_sequence(sequence)
        grads = backward_sequence(cache)

        for name in params:
            params[name] -= LEARNING_RATE * grads[name]

        total_loss += loss
        total_videos += 1

    loss_history.append(total_loss / total_videos)

# ----------------------------------------
# Evaluate next-state accuracy on held-out videos, and
# collect Z2 (final block output) + true phase for the
# visualization / linear probe below.
# ----------------------------------------

correct = 0
total = 0

representations = []
representation_phases = []

for video_id in test_ids:

    sequence = video_id_sequences[video_id]
    if len(sequence) < 2:
        continue

    phases = video_segment_phases[video_id]
    cache, _ = forward_sequence(sequence)
    _, targets, _, _, _, _, _, _, Z2, probs, n = cache

    predicted = probs.argmax(axis=1)
    correct += (predicted == targets).sum()
    total += n

    for t in range(n):
        representations.append(Z2[t])
        representation_phases.append(phases[t + 1])

transformer_accuracy = correct / total

baseline_id = Counter(
    video_id_sequences[v][t + 1]
    for v in train_ids
    for t in range(len(video_id_sequences[v]) - 1)
).most_common(1)[0][0]

baseline_correct = sum(
    1
    for v in test_ids
    for t in range(len(video_id_sequences[v]) - 1)
    if video_id_sequences[v][t + 1] == baseline_id
)
baseline_accuracy = baseline_correct / total

print(f"Vocabulary size: {vocab_size} states")
print(f"Test transitions: {total}")
print()
print(f"Final training loss (per video): {loss_history[-1]:.4f} "
      f"(started at {loss_history[0]:.4f})")
print()
print(f"Transformer block next-state accuracy: {transformer_accuracy:.3f}")
print(f"Baseline accuracy:                     {baseline_accuracy:.3f}")
print()
print("For reference: Day14 Markov table 0.345, "
      "Day16 embedding 0.339/0.352, Day17 RNN 0.405, "
      "Day18 attention 0.331/0.280")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump({
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "ff_dim": FF_DIM,
        "num_test_transitions": int(total),
        "loss_history": loss_history,
        "transformer_accuracy": float(transformer_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
    }, f, indent=2)

# ----------------------------------------
# Plot: final block representation (Z2) colored by true
# current phase (same style as Day17/18, for direct
# comparison).
# ----------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

representations = np.array(representations)
rep_centered = representations - representations.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(rep_centered, full_matrices=False)
rep_2d = rep_centered @ Vt[:2].T

phases_present = sorted(set(representation_phases))
color_map = {
    phase: plt.cm.tab10(i / max(1, len(phases_present) - 1))
    for i, phase in enumerate(phases_present)
}

fig, ax = plt.subplots(figsize=(8, 6))

for phase in phases_present:
    idx = [i for i, p in enumerate(representation_phases) if p == phase]
    ax.scatter(
        rep_2d[idx, 0], rep_2d[idx, 1],
        label=phase, color=color_map[phase], s=10, alpha=0.6
    )

ax.set_title(
    "Transformer block output (PCA to 2D) on held-out test videos,\n"
    "colored by true current phase (phase never shown in training)"
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(output_dir / "representation_by_phase.png", dpi=150)
print(f"\nSaved plot to {output_dir / 'representation_by_phase.png'}")
