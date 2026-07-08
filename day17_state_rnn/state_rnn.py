import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# ----------------------------------------
# Parameters
#
# Same 50 videos, same state segmentation, same
# vocabulary, and same train/test split as Day14/16, so
# accuracy is directly comparable across all three models.
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42

EMBED_DIM = 16
HIDDEN_DIM = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.2
GRAD_CLIP = 5.0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------
# Jaccard similarity (same as Day12-14)
# ----------------------------------------


def jaccard_similarity(set1, set2):

    if len(set1 | set2) == 0:
        return 1.0

    return len(set1 & set2) / len(set1 | set2)


# ----------------------------------------
# Build a triplet set and a phase label for every frame
# of one video (same as Day16). Phase is only used to
# label the hidden-state visualization -- it never enters
# training.
# ----------------------------------------


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
    phases = [frame_phases[frame_ids[0]]]
    phase_votes = [Counter([frame_phases[frame_ids[0]]])]

    for frame in frame_ids[1:]:

        similarity = jaccard_similarity(
            states[-1],
            frame_triplets[frame]
        )

        if similarity < threshold:
            states.append(frame_triplets[frame])
            phases.append(frame_phases[frame])
            phase_votes.append(Counter([frame_phases[frame]]))
        else:
            phase_votes[-1][frame_phases[frame]] += 1

    # Majority-vote phase per finished segment (used only for
    # the hidden-state visualization).
    segment_phases = [
        votes.most_common(1)[0][0] for votes in phase_votes
    ]

    return states, segment_phases


def load_state_sequence(video_path):

    with open(video_path, "r") as f:
        data = json.load(f)

    frame_ids, frame_triplets, frame_phases = build_frame_data(data)

    return segment_into_states(
        frame_ids,
        frame_triplets,
        frame_phases,
        SIMILARITY_THRESHOLD
    )


# ----------------------------------------
# Step 1-2: state sequences + shared vocabulary
# (identical to Day14/16)
# ----------------------------------------

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
    video_id: [
        signature_to_id[signature]
        for signature in signatures
    ]
    for video_id, signatures in video_states.items()
}

# ----------------------------------------
# Step 3: same train/test video split as Day14/16
# ----------------------------------------

all_video_ids = sorted(video_id_sequences.keys())

shuffled_ids = all_video_ids[:]
random.seed(RANDOM_SEED)
random.shuffle(shuffled_ids)

num_test = max(1, round(len(shuffled_ids) * TEST_RATIO))

test_ids = sorted(shuffled_ids[:num_test])
train_ids = sorted(shuffled_ids[num_test:])

# ----------------------------------------
# Step 4: the RNN
#
# Day16's embedding model conditioned each prediction on
# only the current state -- exactly as limited as the
# Markov table. Here, a hidden vector h_t is carried
# forward across an entire video, so in principle the
# prediction at step t can depend on everything the model
# has seen since frame 0, not just state t:
#
#   x_t     = E[state_t]                          (D,)
#   h_t     = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh) (H,)
#   logits  = Why @ h_t + by                       (V,)
#   probs   = softmax(logits)
#
# Trained with full backpropagation through time (BPTT)
# per video, written out by hand -- no autograd. Gradients
# are clipped (a well-known practical necessity for RNNs:
# repeatedly multiplying by Whh over many steps can make
# gradients explode).
# ----------------------------------------

# Xavier-style init: scaled by 1/sqrt(fan_in) rather than a
# small fixed constant (e.g. 0.01). A too-small fixed scale
# was tried first and silently failed -- see the Reflection
# in this day's README for what that failure looked like and
# why it happens.
E = np.random.randn(vocab_size, EMBED_DIM) / np.sqrt(EMBED_DIM)
Wxh = np.random.randn(HIDDEN_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
Whh = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) / np.sqrt(HIDDEN_DIM)
bh = np.zeros(HIDDEN_DIM)
Why = np.random.randn(vocab_size, HIDDEN_DIM) / np.sqrt(HIDDEN_DIM)
by = np.zeros(vocab_size)


def softmax(logits):
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def forward_sequence(sequence):
    """Run the RNN over one full video's state sequence.

    Returns per-step hidden states, embeddings, probs, and
    the loss, for use in both training (BPTT) and evaluation.
    """

    h = np.zeros(HIDDEN_DIM)
    xs, hs, probs_list = [], [h], []
    total_loss = 0.0

    for t in range(len(sequence) - 1):

        current_id = sequence[t]
        next_id = sequence[t + 1]

        x = E[current_id]
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        logits = Why @ h + by
        probs = softmax(logits)

        xs.append(x)
        hs.append(h)
        probs_list.append(probs)

        total_loss += -np.log(probs[next_id] + 1e-12)

    return xs, hs, probs_list, total_loss


def bptt(sequence, xs, hs, probs_list):

    dE = np.zeros_like(E)
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)
    dWhy = np.zeros_like(Why)
    dby = np.zeros_like(by)

    dh_next = np.zeros(HIDDEN_DIM)

    num_steps = len(sequence) - 1

    for t in reversed(range(num_steps)):

        next_id = sequence[t + 1]

        dlogits = probs_list[t].copy()
        dlogits[next_id] -= 1

        dWhy += np.outer(dlogits, hs[t + 1])
        dby += dlogits

        dh = Why.T @ dlogits + dh_next
        dhraw = (1 - hs[t + 1] ** 2) * dh  # tanh'(z) = 1 - tanh(z)^2

        dbh += dhraw
        dWxh += np.outer(dhraw, xs[t])
        dWhh += np.outer(dhraw, hs[t])

        dx = Wxh.T @ dhraw
        dE[sequence[t]] += dx

        dh_next = Whh.T @ dhraw

    for grad in (dE, dWxh, dWhh, dbh, dWhy, dby):
        np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

    return dE, dWxh, dWhh, dbh, dWhy, dby


# ----------------------------------------
# Step 5: train with per-video BPTT + SGD
# ----------------------------------------

loss_history = []

for epoch in range(NUM_EPOCHS):

    epoch_video_ids = train_ids[:]
    random.shuffle(epoch_video_ids)

    total_loss = 0.0
    total_steps = 0

    for video_id in epoch_video_ids:

        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue

        xs, hs, probs_list, loss = forward_sequence(sequence)
        dE, dWxh, dWhh, dbh, dWhy, dby = bptt(sequence, xs, hs, probs_list)

        num_steps = len(sequence) - 1
        E -= LEARNING_RATE * dE / num_steps
        Wxh -= LEARNING_RATE * dWxh / num_steps
        Whh -= LEARNING_RATE * dWhh / num_steps
        bh -= LEARNING_RATE * dbh / num_steps
        Why -= LEARNING_RATE * dWhy / num_steps
        by -= LEARNING_RATE * dby / num_steps

        total_loss += loss
        total_steps += num_steps

    loss_history.append(total_loss / total_steps)

# ----------------------------------------
# Step 6: evaluate next-state accuracy on held-out videos,
# and collect hidden states (with their true current phase)
# for the visualization below.
# ----------------------------------------

correct = 0
total = 0

hidden_states = []
hidden_state_phases = []

for video_id in test_ids:

    sequence = video_id_sequences[video_id]
    if len(sequence) < 2:
        continue

    phases = video_segment_phases[video_id]

    xs, hs, probs_list, _ = forward_sequence(sequence)

    for t in range(len(sequence) - 1):
        predicted_id = int(np.argmax(probs_list[t]))
        actual_id = sequence[t + 1]

        total += 1
        if predicted_id == actual_id:
            correct += 1

        hidden_states.append(hs[t + 1])
        hidden_state_phases.append(phases[t + 1])

rnn_accuracy = correct / total

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

# ----------------------------------------
# Results
# ----------------------------------------

print(f"Vocabulary size: {vocab_size} states")
print(f"Test transitions: {total}")
print()
print(f"Final training loss (per step): {loss_history[-1]:.4f} "
      f"(started at {loss_history[0]:.4f})")
print()
print(f"RNN next-state accuracy:  {rnn_accuracy:.3f}")
print(f"Baseline accuracy:        {baseline_accuracy:.3f}")
print()
print("For reference: Day14 Markov table 0.345, "
      "Day16 embedding model 0.339/0.352 (seen states)")

output_dir = Path(__file__).parent

with open(output_dir / "results.json", "w") as f:
    json.dump({
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_test_transitions": total,
        "loss_history": loss_history,
        "rnn_accuracy": float(rnn_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
    }, f, indent=2)

# ----------------------------------------
# Plot: RNN hidden states colored by the TRUE current
# phase at that timestep (not the state id's global
# dominant phase, since the same state id can now produce
# different hidden vectors depending on context). Compare
# this to Day16's embedding_by_phase.png.
# ----------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

hidden_states = np.array(hidden_states)
hidden_centered = hidden_states - hidden_states.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(hidden_centered, full_matrices=False)
hidden_2d = hidden_centered @ Vt[:2].T

phases_present = sorted(set(hidden_state_phases))
color_map = {
    phase: plt.cm.tab10(i / max(1, len(phases_present) - 1))
    for i, phase in enumerate(phases_present)
}

fig, ax = plt.subplots(figsize=(8, 6))

for phase in phases_present:
    idx = [
        i for i, p in enumerate(hidden_state_phases) if p == phase
    ]
    ax.scatter(
        hidden_2d[idx, 0], hidden_2d[idx, 1],
        label=phase, color=color_map[phase], s=10, alpha=0.6
    )

ax.set_title(
    "RNN hidden states (PCA to 2D) on held-out test videos,\n"
    "colored by true current phase (phase never shown in training)"
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(output_dir / "hidden_states_by_phase.png", dpi=150)
print(f"\nSaved plot to {output_dir / 'hidden_states_by_phase.png'}")
