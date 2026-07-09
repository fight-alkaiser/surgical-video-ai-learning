import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# ----------------------------------------
# Does next-state prediction actually need long history, or
# does a short window already capture most of what Day17's
# RNN (full history, 40.5%) and Day18's attention (full
# history, 33.1%) achieved?
#
# Three methods, each re-run at several context-window sizes
# k, so accuracy-vs-k can be read directly:
#
#   1. k-th order Markov table: pure counting, no learning at
#      all -- the (context of last k states -> most likely
#      next state) generalization of Day13/14's 1st-order table.
#   2. Windowed RNN: same architecture as Day17, but the
#      hidden state is reset to zero every k steps, so no
#      prediction can depend on more than k-1 prior states.
#      (An approximation of a true sliding window: context
#      length varies from 0 to k-1 within a reset chunk,
#      rather than always being exactly k. Simpler to
#      implement than re-running the recurrence from scratch
#      at every single position, and a standard truncated-BPTT
#      style approximation of "bounded context.")
#   3. Windowed attention: same architecture as Day18, but the
#      causal mask additionally blocks any key more than k-1
#      steps behind the query -- a true sliding window, exact
#      for every position (no approximation needed here, since
#      attention has no persistent state to reset).
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42

WINDOW_SIZES = [1, 2, 3, 5, 10]

EMBED_DIM = 16
HIDDEN_DIM = 32
KEY_DIM = 16
VALUE_DIM = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.2
GRAD_CLIP = 5.0

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

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    frame_triplets = {}

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

    return frame_ids, frame_triplets


def segment_into_states(frame_ids, frame_triplets, threshold):

    states = [frame_triplets[frame_ids[0]]]

    for frame in frame_ids[1:]:
        similarity = jaccard_similarity(states[-1], frame_triplets[frame])
        if similarity < threshold:
            states.append(frame_triplets[frame])

    return states


def load_state_sequence(video_path):
    with open(video_path, "r") as f:
        data = json.load(f)
    frame_ids, frame_triplets = build_frame_data(data)
    return segment_into_states(frame_ids, frame_triplets, SIMILARITY_THRESHOLD)


video_paths = sorted(LABELS_DIR.glob("VID*.json"))
video_states = {path.stem: load_state_sequence(path) for path in video_paths}

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


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ----------------------------------------
# Method 1: k-th order Markov table (pure counting)
# ----------------------------------------


def evaluate_kth_order_markov(k):

    transition_counts = defaultdict(Counter)

    for video_id in train_ids:
        sequence = video_id_sequences[video_id]
        for t in range(k - 1, len(sequence) - 1):
            context = tuple(sequence[t - k + 1:t + 1])
            next_id = sequence[t + 1]
            transition_counts[context][next_id] += 1

    most_likely_next = {
        context: counts.most_common(1)[0][0]
        for context, counts in transition_counts.items()
    }

    correct = 0
    total = 0

    for video_id in test_ids:
        sequence = video_id_sequences[video_id]
        for t in range(k - 1, len(sequence) - 1):
            context = tuple(sequence[t - k + 1:t + 1])
            actual_next = sequence[t + 1]
            total += 1
            predicted = most_likely_next.get(context)
            if predicted == actual_next:
                correct += 1

    return correct / total, total


# ----------------------------------------
# Method 2: windowed RNN (hidden state reset every k steps)
# ----------------------------------------


def train_and_evaluate_windowed_rnn(k, seed):

    rng = np.random.RandomState(seed)

    E = rng.randn(vocab_size, EMBED_DIM) / np.sqrt(EMBED_DIM)
    Wxh = rng.randn(HIDDEN_DIM, EMBED_DIM) / np.sqrt(EMBED_DIM)
    Whh = rng.randn(HIDDEN_DIM, HIDDEN_DIM) / np.sqrt(HIDDEN_DIM)
    bh = np.zeros(HIDDEN_DIM)
    Why = rng.randn(vocab_size, HIDDEN_DIM) / np.sqrt(HIDDEN_DIM)
    by = np.zeros(vocab_size)

    def forward_sequence(sequence):

        h = np.zeros(HIDDEN_DIM)
        xs, hs, probs_list = [], [h], []
        total_loss = 0.0

        for t in range(len(sequence) - 1):

            if t % k == 0:
                h = np.zeros(HIDDEN_DIM)

            current_id = sequence[t]
            next_id = sequence[t + 1]

            x = E[current_id]
            h = np.tanh(Wxh @ x + Whh @ h + bh)
            logits = Why @ h + by
            probs = softmax_rows(logits[None, :])[0]

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

            # A reset severs the recurrence: no gradient (and
            # no hidden state) crosses a chunk boundary.
            if t % k == 0:
                dh_next = np.zeros(HIDDEN_DIM)
            else:
                dh_next = Whh.T @ dhraw

        for grad in (dE, dWxh, dWhh, dbh, dWhy, dby):
            np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

        return dE, dWxh, dWhh, dbh, dWhy, dby

    for epoch in range(NUM_EPOCHS):

        epoch_video_ids = train_ids[:]
        random.Random(seed * 1000 + epoch).shuffle(epoch_video_ids)

        for video_id in epoch_video_ids:

            sequence = video_id_sequences[video_id]
            if len(sequence) < 2:
                continue

            xs, hs, probs_list, _ = forward_sequence(sequence)
            dE, dWxh, dWhh, dbh, dWhy, dby = bptt(sequence, xs, hs, probs_list)

            num_steps = len(sequence) - 1
            E -= LEARNING_RATE * dE / num_steps
            Wxh -= LEARNING_RATE * dWxh / num_steps
            Whh -= LEARNING_RATE * dWhh / num_steps
            bh -= LEARNING_RATE * dbh / num_steps
            Why -= LEARNING_RATE * dWhy / num_steps
            by -= LEARNING_RATE * dby / num_steps

    correct = 0
    total = 0

    for video_id in test_ids:
        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue
        _, _, probs_list, _ = forward_sequence(sequence)
        for t in range(len(sequence) - 1):
            predicted = int(np.argmax(probs_list[t]))
            total += 1
            if predicted == sequence[t + 1]:
                correct += 1

    return correct / total


# ----------------------------------------
# Method 3: windowed attention (sliding-window causal mask)
# ----------------------------------------


def train_and_evaluate_windowed_attention(k, seed):

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
        Xp = X + positional_encoding[:n]

        Q = Xp @ Wq
        K = Xp @ Wk
        V = Xp @ Wv

        scores = (Q @ K.T) / np.sqrt(KEY_DIM)

        t_idx = np.arange(n)[:, None]
        s_idx = np.arange(n)[None, :]
        # A position may only attend to itself and up to
        # (k - 1) positions before it -- a sliding window,
        # exact at every position (unlike the RNN's chunked
        # reset approximation above).
        blocked = (s_idx > t_idx) | (s_idx <= t_idx - k)
        scores = np.where(blocked, -1e9, scores)

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

    correct = 0
    total = 0

    for video_id in test_ids:
        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue
        cache, _ = forward_sequence(sequence)
        _, targets, _, _, _, _, _, _, probs, n = cache
        predicted = probs.argmax(axis=1)
        correct += (predicted == targets).sum()
        total += n

    return correct / total


# ----------------------------------------
# Run all three methods across window sizes
# ----------------------------------------

results = {"markov": {}, "rnn": {}, "attention": {}}

print("k-th order Markov table:")
for k in WINDOW_SIZES:
    acc, n_test = evaluate_kth_order_markov(k)
    results["markov"][k] = acc
    print(f"  k={k:2d}: accuracy={acc:.3f} (N={n_test})")

print("\nWindowed RNN (hidden state reset every k steps):")
for k in WINDOW_SIZES:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    acc = train_and_evaluate_windowed_rnn(k, seed=RANDOM_SEED)
    results["rnn"][k] = acc
    print(f"  k={k:2d}: accuracy={acc:.3f}")

print("\nWindowed attention (sliding-window causal mask):")
for k in WINDOW_SIZES:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    acc = train_and_evaluate_windowed_attention(k, seed=RANDOM_SEED)
    results["attention"][k] = acc
    print(f"  k={k:2d}: accuracy={acc:.3f}")

print("\nFor reference (full/unbounded history):")
print("  Day14 Markov table (k=1, same as k=1 above): 0.345")
print("  Day17 RNN (full history):       0.405")
print("  Day18 Attention (full history):  0.331")

output_dir = Path(__file__).parent
with open(output_dir / "window_size_results.json", "w") as f:
    json.dump({
        "window_sizes": WINDOW_SIZES,
        "results": results,
        "full_history_reference": {
            "rnn": 0.405,
            "attention": 0.331,
        },
    }, f, indent=2)
