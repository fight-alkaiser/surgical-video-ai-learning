import json
import random
from pathlib import Path
from collections import Counter

import numpy as np

# ----------------------------------------
# Parameters
#
# Same 50 videos, same state segmentation, same
# vocabulary, and same train/test split as Day14/16/17.
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42

EMBED_DIM = 16
KEY_DIM = 16
VALUE_DIM = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.2
GRAD_CLIP = 5.0
USE_POSITIONAL_ENCODING = True

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------
# Data prep (identical to Day17's state_rnn.py)
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

    frame_ids = sorted(
        int(frame)
        for frame in data["annotations"].keys()
    )

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

        similarity = jaccard_similarity(
            states[-1],
            frame_triplets[frame]
        )

        if similarity < threshold:
            states.append(frame_triplets[frame])
            phase_votes.append(Counter([frame_phases[frame]]))
        else:
            phase_votes[-1][frame_phases[frame]] += 1

    segment_phases = [
        votes.most_common(1)[0][0] for votes in phase_votes
    ]

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

# ----------------------------------------
# Fixed sinusoidal positional encoding (Vaswani et al.,
# 2017) -- no learned parameters, added to each state's
# embedding before attention. Self-attention has no
# built-in notion of order (unlike an RNN, whose recurrence
# makes position implicit): without this, shuffling a
# video's states would not change a single attention
# weight. This is tested directly below (USE_POSITIONAL_ENCODING).
# ----------------------------------------

position = np.arange(max_len)[:, None]
div_term = np.exp(
    np.arange(0, EMBED_DIM, 2) * (-np.log(10000.0) / EMBED_DIM)
)
positional_encoding = np.zeros((max_len, EMBED_DIM))
positional_encoding[:, 0::2] = np.sin(position * div_term)
positional_encoding[:, 1::2] = np.cos(position * div_term)

# ----------------------------------------
# Attention model
#
# Where Day17's RNN compressed all history into one
# fixed-size hidden vector h_t, updated one step at a time,
# this model instead lets every position t look directly
# back at every earlier position s <= t and decide, via a
# learned compatibility score, how much to weight it:
#
#   Q = Xp @ Wq,  K = Xp @ Wk,  V = Xp @ Wv
#   scores[t, s]  = (Q[t] . K[s]) / sqrt(KEY_DIM)   for s <= t, else -inf
#   weights[t, :] = softmax(scores[t, :])
#   context[t]    = sum_s weights[t, s] * V[s]
#   logits[t]     = context[t] @ Why.T + by
#
# There is no recurrence at all: context[t] is a direct,
# weighted combination of the value vectors at every
# earlier position, computed in one shot per video rather
# than one step at a time. Everything (forward pass,
# softmax-over-scores backward, causal masking) is written
# out by hand with numpy -- no autograd.
# ----------------------------------------

E = np.random.randn(vocab_size, EMBED_DIM) / np.sqrt(EMBED_DIM)
Wq = np.random.randn(EMBED_DIM, KEY_DIM) / np.sqrt(EMBED_DIM)
Wk = np.random.randn(EMBED_DIM, KEY_DIM) / np.sqrt(EMBED_DIM)
Wv = np.random.randn(EMBED_DIM, VALUE_DIM) / np.sqrt(EMBED_DIM)
Why = np.random.randn(vocab_size, VALUE_DIM) / np.sqrt(VALUE_DIM)
by = np.zeros(vocab_size)


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def forward_sequence(sequence):
    """Run causal self-attention over one full video's state
    sequence. Only positions 0..n-1 are used as queries/keys
    (n = len(sequence) - 1), since only those have a next-state
    target -- exactly the same convention as Day17's RNN.
    """

    n = len(sequence) - 1
    ids = np.array(sequence[:n])
    targets = np.array(sequence[1:n + 1])

    X = E[ids]
    if USE_POSITIONAL_ENCODING:
        Xp = X + positional_encoding[:n]
    else:
        Xp = X

    Q = Xp @ Wq
    K = Xp @ Wk
    V = Xp @ Wv

    scores = (Q @ K.T) / np.sqrt(KEY_DIM)
    causal_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(causal_mask, -1e9, scores)

    weights = softmax_rows(scores)
    context = weights @ V

    logits = context @ Why.T + by
    probs = softmax_rows(logits)

    loss = -np.log(probs[np.arange(n), targets] + 1e-12).mean()

    cache = (ids, targets, Xp, Q, K, V, weights, context, probs, n)
    return cache, loss


def backward_sequence(cache):

    ids, targets, Xp, Q, K, V, weights, context, probs, n = cache

    dlogits = probs.copy()
    dlogits[np.arange(n), targets] -= 1
    dlogits /= n

    dWhy = dlogits.T @ context
    dby = dlogits.sum(axis=0)
    dcontext = dlogits @ Why

    dweights = dcontext @ V.T
    dV = weights.T @ dcontext

    row_dot = (dweights * weights).sum(axis=1, keepdims=True)
    dscores = weights * (dweights - row_dot)
    dscores /= np.sqrt(KEY_DIM)

    dQ = dscores @ K
    dK = dscores.T @ Q

    dWq = Xp.T @ dQ
    dWk = Xp.T @ dK
    dWv = Xp.T @ dV

    dXp = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    dE = np.zeros_like(E)
    np.add.at(dE, ids, dXp)

    grads = (dE, dWq, dWk, dWv, dWhy, dby)
    for grad in grads:
        np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

    return grads


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
        dE, dWq, dWk, dWv, dWhy, dby = backward_sequence(cache)

        E -= LEARNING_RATE * dE
        Wq -= LEARNING_RATE * dWq
        Wk -= LEARNING_RATE * dWk
        Wv -= LEARNING_RATE * dWv
        Why -= LEARNING_RATE * dWhy
        by -= LEARNING_RATE * dby

        total_loss += loss
        total_videos += 1

    loss_history.append(total_loss / total_videos)

# ----------------------------------------
# Evaluate next-state accuracy on held-out videos, and
# collect context vectors (+ true phase) for the
# visualization / linear probe below.
# ----------------------------------------

correct = 0
total = 0

context_vectors = []
context_phases = []

for video_id in test_ids:

    sequence = video_id_sequences[video_id]
    if len(sequence) < 2:
        continue

    phases = video_segment_phases[video_id]
    cache, _ = forward_sequence(sequence)
    _, targets, _, _, _, _, _, context, probs, n = cache

    predicted = probs.argmax(axis=1)
    correct += (predicted == targets).sum()
    total += n

    for t in range(n):
        context_vectors.append(context[t])
        context_phases.append(phases[t + 1])

attention_accuracy = correct / total

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
print(f"Attention next-state accuracy: {attention_accuracy:.3f}")
print(f"Baseline accuracy:             {baseline_accuracy:.3f}")
print()
print("For reference: Day14 Markov table 0.345, "
      "Day16 embedding model 0.339/0.352, Day17 RNN 0.405")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump({
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "key_dim": KEY_DIM,
        "value_dim": VALUE_DIM,
        "use_positional_encoding": USE_POSITIONAL_ENCODING,
        "num_test_transitions": int(total),
        "loss_history": loss_history,
        "attention_accuracy": float(attention_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
    }, f, indent=2)

# ----------------------------------------
# Plot: attention context vectors colored by true current
# phase (same style as Day17's hidden-state plot, for a
# direct visual comparison).
# ----------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

context_vectors = np.array(context_vectors)
context_centered = context_vectors - context_vectors.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(context_centered, full_matrices=False)
context_2d = context_centered @ Vt[:2].T

phases_present = sorted(set(context_phases))
color_map = {
    phase: plt.cm.tab10(i / max(1, len(phases_present) - 1))
    for i, phase in enumerate(phases_present)
}

fig, ax = plt.subplots(figsize=(8, 6))

for phase in phases_present:
    idx = [i for i, p in enumerate(context_phases) if p == phase]
    ax.scatter(
        context_2d[idx, 0], context_2d[idx, 1],
        label=phase, color=color_map[phase], s=10, alpha=0.6
    )

ax.set_title(
    "Attention context vectors (PCA to 2D) on held-out test videos,\n"
    "colored by true current phase (phase never shown in training)"
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(output_dir / "context_by_phase.png", dpi=150)
print(f"\nSaved plot to {output_dir / 'context_by_phase.png'}")
