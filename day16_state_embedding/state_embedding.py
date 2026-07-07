import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# ----------------------------------------
# Parameters
#
# Same 50 videos, same state segmentation, same
# vocabulary, and same train/test split (same random
# seed) as Day14, so the embedding model's accuracy is
# directly comparable to the Markov table's 34.5%.
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42

EMBED_DIM = 16
NUM_EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------
# Jaccard similarity
# (same as Day12 / Day13 / Day14)
# ----------------------------------------


def jaccard_similarity(set1, set2):

    if len(set1 | set2) == 0:
        return 1.0

    return len(set1 & set2) / len(set1 | set2)


# ----------------------------------------
# Build a triplet set and a phase label for every frame
# of one video. Phase is only used later, to color the
# embedding visualization -- it never enters training.
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


# ----------------------------------------
# Compress consecutive similar frames into states
# (same segmentation as Day12 / Day13 / Day14), and
# record the majority phase covered by each state segment.
# ----------------------------------------


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

    return states, phase_votes


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
# Step 1: compute a state sequence (+ phase votes) for
# every video.
# ----------------------------------------

video_paths = sorted(LABELS_DIR.glob("VID*.json"))

video_states = {}
video_phase_votes = {}

for path in video_paths:
    states, phase_votes = load_state_sequence(path)
    video_states[path.stem] = states
    video_phase_votes[path.stem] = phase_votes

# ----------------------------------------
# Step 2: build one shared vocabulary across all videos
# (same "tokenizer" as Day14), and tally which phase each
# vocabulary id is most often seen in.
# ----------------------------------------

signature_to_id = {}
id_phase_votes = defaultdict(Counter)

for video_id in sorted(video_states):
    for signature, votes in zip(
        video_states[video_id], video_phase_votes[video_id]
    ):
        if signature not in signature_to_id:
            signature_to_id[signature] = len(signature_to_id)
        state_id = signature_to_id[signature]
        id_phase_votes[state_id].update(votes)

vocab_size = len(signature_to_id)

id_to_dominant_phase = {
    state_id: votes.most_common(1)[0][0]
    for state_id, votes in id_phase_votes.items()
}

video_id_sequences = {
    video_id: [
        signature_to_id[signature]
        for signature in signatures
    ]
    for video_id, signatures in video_states.items()
}

# ----------------------------------------
# Step 3: split videos into train / test
# (identical split to Day14 / Day15: same seed, same
# ratio, applied to the same sorted video id list).
# ----------------------------------------

all_video_ids = sorted(video_id_sequences.keys())

shuffled_ids = all_video_ids[:]
random.seed(RANDOM_SEED)
random.shuffle(shuffled_ids)

num_test = max(1, round(len(shuffled_ids) * TEST_RATIO))

test_ids = sorted(shuffled_ids[:num_test])
train_ids = sorted(shuffled_ids[num_test:])

# ----------------------------------------
# Step 4: build (current_id, next_id) training pairs
# ----------------------------------------


def build_pairs(video_ids):

    current_list = []
    next_list = []

    for video_id in video_ids:
        sequence = video_id_sequences[video_id]
        for i in range(len(sequence) - 1):
            current_list.append(sequence[i])
            next_list.append(sequence[i + 1])

    return np.array(current_list), np.array(next_list)


train_current, train_next = build_pairs(train_ids)
test_current, test_next = build_pairs(test_ids)

# States that were never a "current" state during training:
# the Markov table (Day14) simply has no entry for them.
# The embedding model *will* still produce a prediction for
# them (via their randomly-initialized, never-updated
# embedding row) -- this is tracked separately below.
train_seen_current_ids = set(train_current.tolist())

# ----------------------------------------
# Step 5: the embedding model
#
# This replaces the Markov count table (Day13/14) with a
# small neural network:
#
#   embedding = E[current_id]            (D,)
#   logits    = W @ embedding + b        (V,)
#   probs     = softmax(logits)
#
# E is a (vocab_size, EMBED_DIM) lookup table: exactly the
# same idea as a word embedding, except the "words" here
# are surgical triplet-states. Instead of directly counting
# "what followed this exact state", the model must squeeze
# every state's identity through an EMBED_DIM-wide
# bottleneck (8 numbers, versus 358 one-hot dimensions)
# before it can predict what comes next. Everything below
# (forward pass, loss, gradients, SGD update) is written by
# hand with numpy -- no autograd -- so every step of
# backpropagation is visible.
# ----------------------------------------

E = np.random.randn(vocab_size, EMBED_DIM) * 0.01
W = np.random.randn(vocab_size, EMBED_DIM) * 0.01
b = np.zeros(vocab_size)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def forward(current_ids):
    embeddings = E[current_ids]                # (batch, D)
    logits = embeddings @ W.T + b               # (batch, V)
    probs = softmax(logits)
    return embeddings, probs


def train_step(current_ids, next_ids, learning_rate):

    batch_size = len(current_ids)

    embeddings, probs = forward(current_ids)

    # Cross-entropy loss gradient w.r.t. logits:
    # softmax_output - one_hot(target), averaged over batch.
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), next_ids] -= 1
    dlogits /= batch_size

    dW = dlogits.T @ embeddings                 # (V, D)
    db = dlogits.sum(axis=0)                    # (V,)
    dembeddings = dlogits @ W                   # (batch, D)

    dE = np.zeros_like(E)
    np.add.at(dE, current_ids, dembeddings)

    W_local = W - learning_rate * dW
    b_local = b - learning_rate * db
    E_local = E - learning_rate * dE

    loss = -np.log(
        probs[np.arange(batch_size), next_ids] + 1e-12
    ).mean()

    return loss, W_local, b_local, E_local


# ----------------------------------------
# Step 6: train with mini-batch SGD
# ----------------------------------------

num_train = len(train_current)
loss_history = []

for epoch in range(NUM_EPOCHS):

    order = np.random.permutation(num_train)
    epoch_losses = []

    for start in range(0, num_train, BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        current_ids = train_current[batch_idx]
        next_ids = train_next[batch_idx]

        loss, W, b, E = train_step(
            current_ids, next_ids, LEARNING_RATE
        )
        epoch_losses.append(loss)

    loss_history.append(float(np.mean(epoch_losses)))

# ----------------------------------------
# Step 7: evaluate next-state prediction accuracy on the
# held-out test videos, same metric as Day14.
# ----------------------------------------

_, test_probs = forward(test_current)
predicted_next = test_probs.argmax(axis=1)

correct = (predicted_next == test_next)
overall_accuracy = correct.mean()

seen_mask = np.array([
    cid in train_seen_current_ids for cid in test_current
])
seen_accuracy = correct[seen_mask].mean()
unseen_count = int((~seen_mask).sum())

baseline_id = Counter(train_next.tolist()).most_common(1)[0][0]
baseline_accuracy = (test_next == baseline_id).mean()

# ----------------------------------------
# Step 8: 2D projection of the learned embedding table via
# plain SVD-based PCA (no sklearn), colored by each state's
# dominant phase. Phase was never used in training -- this
# only checks whether phase structure emerges anyway.
# ----------------------------------------

E_centered = E - E.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(E_centered, full_matrices=False)
E_2d = E_centered @ Vt[:2].T

# ----------------------------------------
# Results
# ----------------------------------------

print(f"Vocabulary size: {vocab_size} states")
print(f"Train transitions: {num_train}, Test transitions: {len(test_current)}")
print()
print(f"Final training loss: {loss_history[-1]:.4f} "
      f"(started at {loss_history[0]:.4f})")
print()
print(f"Embedding model accuracy (overall):        {overall_accuracy:.3f}")
print(f"Embedding model accuracy (seen states only): {seen_accuracy:.3f}")
print(f"Unseen-in-training current states in test:  {unseen_count}")
print(f"Baseline (always predict most common next):  {baseline_accuracy:.3f}")
print()
print("Day14 Markov table for reference: 0.345 (overall), baseline 0.121")

output_dir = Path(__file__).parent

with open(output_dir / "results.json", "w") as f:
    json.dump({
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "num_train_transitions": num_train,
        "num_test_transitions": len(test_current),
        "loss_history": loss_history,
        "overall_accuracy": float(overall_accuracy),
        "seen_accuracy": float(seen_accuracy),
        "unseen_current_state_count": unseen_count,
        "baseline_accuracy": float(baseline_accuracy),
    }, f, indent=2)

# ----------------------------------------
# Plot: embedding space colored by dominant phase
# ----------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

phases_present = sorted(set(
    id_to_dominant_phase.get(i, "unknown") for i in range(vocab_size)
))
color_map = {
    phase: plt.cm.tab10(i / max(1, len(phases_present) - 1))
    for i, phase in enumerate(phases_present)
}

fig, ax = plt.subplots(figsize=(8, 6))

for phase in phases_present:
    idx = [
        i for i in range(vocab_size)
        if id_to_dominant_phase.get(i, "unknown") == phase
    ]
    ax.scatter(
        E_2d[idx, 0], E_2d[idx, 1],
        label=phase, color=color_map[phase], s=18, alpha=0.8
    )

ax.set_title(
    "Learned state embeddings (PCA to 2D), colored by phase\n"
    "(phase was never shown to the model during training)"
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(output_dir / "embedding_by_phase.png", dpi=150)
print(f"\nSaved plot to {output_dir / 'embedding_by_phase.png'}")
