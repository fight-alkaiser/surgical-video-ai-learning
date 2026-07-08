import json
import random
from pathlib import Path
from collections import Counter

import numpy as np

# ----------------------------------------
# Linear probe: how much phase information is linearly
# readable from the RNN's hidden states?
#
# state_rnn.py showed a PCA plot where hidden states
# visually group by phase. A 2D PCA projection is a human-
# friendly summary, but it only shows the two directions of
# largest variance in the full 32-dim hidden space -- it
# could be hiding or exaggerating how much phase information
# is really there. A linear probe is the standard, more
# rigorous check: freeze the trained RNN entirely (no
# gradient flows into it here), then fit the *simplest
# possible* classifier -- a single linear layer + softmax --
# on top of its hidden states to predict phase. If a linear
# probe can recover phase well, phase is linearly encoded in
# the hidden space (not just visually suggestive in 2D). If
# it can't, the PCA plot may have been optimistic.
#
# This script re-trains the same RNN as state_rnn.py (same
# data, same architecture, same seed) so it is fully
# self-contained, then adds the probe on top.
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

PROBE_EPOCHS = 200
PROBE_LEARNING_RATE = 0.5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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


# ----------------------------------------
# Data prep + train/test split (identical to state_rnn.py)
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

# ----------------------------------------
# RNN (identical architecture/training to state_rnn.py)
# ----------------------------------------

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
        dhraw = (1 - hs[t + 1] ** 2) * dh

        dbh += dhraw
        dWxh += np.outer(dhraw, xs[t])
        dWhh += np.outer(dhraw, hs[t])

        dx = Wxh.T @ dhraw
        dE[sequence[t]] += dx

        dh_next = Whh.T @ dhraw

    for grad in (dE, dWxh, dWhh, dbh, dWhy, dby):
        np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

    return dE, dWxh, dWhh, dbh, dWhy, dby


for epoch in range(NUM_EPOCHS):

    epoch_video_ids = train_ids[:]
    random.shuffle(epoch_video_ids)

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

print("RNN training done.")

# ----------------------------------------
# Collect frozen hidden states + true phase, split by the
# same train/test video split as the RNN itself (so the
# probe is evaluated on videos neither the RNN nor the probe
# has seen).
# ----------------------------------------

phase_names = sorted({
    phase
    for phases in video_segment_phases.values()
    for phase in phases
    if phase is not None
})
phase_to_id = {phase: i for i, phase in enumerate(phase_names)}
num_phases = len(phase_names)


def collect_hidden_states(video_ids):

    hidden_list = []
    phase_id_list = []

    for video_id in video_ids:

        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue

        phases = video_segment_phases[video_id]
        _, hs, _, _ = forward_sequence(sequence)

        for t in range(len(sequence) - 1):
            phase = phases[t + 1]
            if phase is None:
                continue
            hidden_list.append(hs[t + 1])
            phase_id_list.append(phase_to_id[phase])

    return np.array(hidden_list), np.array(phase_id_list)


train_hidden, train_phase_ids = collect_hidden_states(train_ids)
test_hidden, test_phase_ids = collect_hidden_states(test_ids)

print(f"Probe training examples: {len(train_hidden)}, "
      f"test examples: {len(test_hidden)}, phases: {num_phases}")

# ----------------------------------------
# The probe itself: a single linear layer + softmax,
# trained on the FROZEN hidden states above (no gradient
# ever flows back into E/Wxh/Whh/Why/by from here on).
# ----------------------------------------

Wp = np.random.randn(num_phases, HIDDEN_DIM) / np.sqrt(HIDDEN_DIM)
bp = np.zeros(num_phases)

num_train = len(train_hidden)
probe_loss_history = []

for epoch in range(PROBE_EPOCHS):

    logits = train_hidden @ Wp.T + bp
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    loss = -np.log(
        probs[np.arange(num_train), train_phase_ids] + 1e-12
    ).mean()
    probe_loss_history.append(float(loss))

    dlogits = probs.copy()
    dlogits[np.arange(num_train), train_phase_ids] -= 1
    dlogits /= num_train

    dWp = dlogits.T @ train_hidden
    dbp = dlogits.sum(axis=0)

    Wp -= PROBE_LEARNING_RATE * dWp
    bp -= PROBE_LEARNING_RATE * dbp

# ----------------------------------------
# Evaluate the probe on held-out test videos
# ----------------------------------------

test_logits = test_hidden @ Wp.T + bp
predicted_phase_ids = test_logits.argmax(axis=1)
probe_accuracy = (predicted_phase_ids == test_phase_ids).mean()

baseline_phase_id = Counter(train_phase_ids.tolist()).most_common(1)[0][0]
baseline_accuracy = (test_phase_ids == baseline_phase_id).mean()

print()
print(f"Linear probe accuracy (phase from frozen RNN hidden state): "
      f"{probe_accuracy:.3f}")
print(f"Baseline (always predict most common phase):                "
      f"{baseline_accuracy:.3f}")

output_dir = Path(__file__).parent
with open(output_dir / "linear_probe_results.json", "w") as f:
    json.dump({
        "num_phases": num_phases,
        "num_train_examples": int(num_train),
        "num_test_examples": int(len(test_hidden)),
        "probe_loss_history": probe_loss_history,
        "probe_accuracy": float(probe_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
    }, f, indent=2)
