import json
import random
from pathlib import Path
from collections import Counter

import numpy as np

# ----------------------------------------
# Linear probe: how much phase information is linearly
# readable from the attention model's context vectors --
# run once with positional encoding, once without, to
# quantify how much of any phase structure comes from
# absolute position (which happens to correlate with phase,
# since surgical phases proceed in a roughly fixed clinical
# order) versus from content-based attention over past
# states. See state_attention.py and this day's README for
# the reasoning; this script is self-contained (re-trains
# both attention variants from scratch, same data/split/seed).
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

PROBE_EPOCHS = 200
PROBE_LEARNING_RATE = 0.5


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

phase_names = sorted({
    phase
    for phases in video_segment_phases.values()
    for phase in phases
    if phase is not None
})
phase_to_id = {phase: i for i, phase in enumerate(phase_names)}
num_phases = len(phase_names)


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_attention(use_positional_encoding, seed):

    rng = np.random.RandomState(seed)

    E = rng.randn(vocab_size, EMBED_DIM) / np.sqrt(EMBED_DIM)
    Wq = rng.randn(EMBED_DIM, KEY_DIM) / np.sqrt(EMBED_DIM)
    Wk = rng.randn(EMBED_DIM, KEY_DIM) / np.sqrt(EMBED_DIM)
    Wv = rng.randn(EMBED_DIM, VALUE_DIM) / np.sqrt(EMBED_DIM)
    Why = rng.randn(vocab_size, VALUE_DIM) / np.sqrt(VALUE_DIM)
    by = np.zeros(vocab_size)

    def forward_sequence(sequence):

        n = len(sequence) - 1
        ids = np.array(sequence[:n])
        targets = np.array(sequence[1:n + 1])

        X = E[ids]
        Xp = X + positional_encoding[:n] if use_positional_encoding else X

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

        return (ids, targets, Xp, Q, K, V, weights, context, probs, n), loss

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

    for epoch in range(NUM_EPOCHS):

        epoch_video_ids = train_ids[:]
        random.Random(seed * 1000 + epoch).shuffle(epoch_video_ids)

        for video_id in epoch_video_ids:

            sequence = video_id_sequences[video_id]
            if len(sequence) < 2:
                continue

            cache, _ = forward_sequence(sequence)
            dE, dWq, dWk, dWv, dWhy, dby = backward_sequence(cache)

            E -= LEARNING_RATE * dE
            Wq -= LEARNING_RATE * dWq
            Wk -= LEARNING_RATE * dWk
            Wv -= LEARNING_RATE * dWv
            Why -= LEARNING_RATE * dWhy
            by -= LEARNING_RATE * dby

    return forward_sequence


def collect_context_vectors(forward_sequence, video_ids):

    context_list = []
    phase_id_list = []

    for video_id in video_ids:

        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue

        phases = video_segment_phases[video_id]
        cache, _ = forward_sequence(sequence)
        _, _, _, _, _, _, _, context, _, n = cache

        for t in range(n):
            phase = phases[t + 1]
            if phase is None:
                continue
            context_list.append(context[t])
            phase_id_list.append(phase_to_id[phase])

    return np.array(context_list), np.array(phase_id_list)


def run_linear_probe(train_features, train_labels, test_features, test_labels):

    num_train = len(train_features)
    feature_dim = train_features.shape[1]

    rng = np.random.RandomState(RANDOM_SEED)
    Wp = rng.randn(num_phases, feature_dim) / np.sqrt(feature_dim)
    bp = np.zeros(num_phases)

    for epoch in range(PROBE_EPOCHS):

        logits = train_features @ Wp.T + bp
        probs = softmax_rows(logits)

        dlogits = probs.copy()
        dlogits[np.arange(num_train), train_labels] -= 1
        dlogits /= num_train

        Wp -= PROBE_LEARNING_RATE * (dlogits.T @ train_features)
        bp -= PROBE_LEARNING_RATE * dlogits.sum(axis=0)

    test_logits = test_features @ Wp.T + bp
    predicted = test_logits.argmax(axis=1)
    return (predicted == test_labels).mean()


results = {}

for use_pe in (True, False):

    label = "with_positional_encoding" if use_pe else "without_positional_encoding"
    print(f"Training attention model ({label})...")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    forward_sequence = train_attention(use_pe, seed=RANDOM_SEED)

    train_features, train_labels = collect_context_vectors(forward_sequence, train_ids)
    test_features, test_labels = collect_context_vectors(forward_sequence, test_ids)

    probe_accuracy = run_linear_probe(
        train_features, train_labels, test_features, test_labels
    )
    baseline_accuracy = (
        test_labels == Counter(train_labels.tolist()).most_common(1)[0][0]
    ).mean()

    results[label] = {
        "probe_accuracy": float(probe_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
    }
    print(f"  Linear probe phase accuracy: {probe_accuracy:.3f} "
          f"(baseline {baseline_accuracy:.3f})")

output_dir = Path(__file__).parent
with open(output_dir / "linear_probe_results.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print("Summary:")
for label, r in results.items():
    print(f"  {label}: {r['probe_accuracy']:.3f} (baseline {r['baseline_accuracy']:.3f})")
