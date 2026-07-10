import json
import random
from pathlib import Path
from collections import Counter

import numpy as np

# ----------------------------------------
# Linear probe: how much phase information is linearly
# readable from the Transformer block's final output (Z2)?
# Same frozen-features + single-linear-layer method as
# Day17/18, for direct comparison. Self-contained: re-trains
# the block from scratch (same data/split/seed as
# transformer_block.py).
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

params = dict(
    E=E, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
    gamma1=gamma1, beta1=beta1,
    W1=W1, b1=b1, W2=W2, b2=b2,
    gamma2=gamma2, beta2=beta2,
    Why=Why, by=by,
)


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


for epoch in range(NUM_EPOCHS):

    epoch_video_ids = train_ids[:]
    random.shuffle(epoch_video_ids)

    for video_id in epoch_video_ids:

        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue

        cache, _ = forward_sequence(sequence)
        grads = backward_sequence(cache)

        for name in params:
            params[name] -= LEARNING_RATE * grads[name]

print("Transformer block training done.")


def collect_representations(video_ids):

    rep_list = []
    phase_id_list = []

    for video_id in video_ids:

        sequence = video_id_sequences[video_id]
        if len(sequence) < 2:
            continue

        phases = video_segment_phases[video_id]
        cache, _ = forward_sequence(sequence)
        _, _, _, _, _, _, _, _, Z2, _, n = cache

        for t in range(n):
            phase = phases[t + 1]
            if phase is None:
                continue
            rep_list.append(Z2[t])
            phase_id_list.append(phase_to_id[phase])

    return np.array(rep_list), np.array(phase_id_list)


train_reps, train_phase_ids = collect_representations(train_ids)
test_reps, test_phase_ids = collect_representations(test_ids)

print(f"Probe training examples: {len(train_reps)}, "
      f"test examples: {len(test_reps)}, phases: {num_phases}")

Wp = np.random.randn(num_phases, EMBED_DIM) / np.sqrt(EMBED_DIM)
bp = np.zeros(num_phases)

num_train = len(train_reps)

for epoch in range(PROBE_EPOCHS):

    logits = train_reps @ Wp.T + bp
    probs = softmax_rows(logits)

    loss_grad = probs.copy()
    loss_grad[np.arange(num_train), train_phase_ids] -= 1
    loss_grad /= num_train

    Wp -= PROBE_LEARNING_RATE * (loss_grad.T @ train_reps)
    bp -= PROBE_LEARNING_RATE * loss_grad.sum(axis=0)

test_logits = test_reps @ Wp.T + bp
predicted_phase_ids = test_logits.argmax(axis=1)
probe_accuracy = (predicted_phase_ids == test_phase_ids).mean()

baseline_phase_id = Counter(train_phase_ids.tolist()).most_common(1)[0][0]
baseline_accuracy = (test_phase_ids == baseline_phase_id).mean()

print()
print(f"Linear probe accuracy (phase from frozen Z2): {probe_accuracy:.3f}")
print(f"Baseline (always predict most common phase):  {baseline_accuracy:.3f}")

output_dir = Path(__file__).parent
with open(output_dir / "linear_probe_results.json", "w") as f:
    json.dump({
        "num_phases": num_phases,
        "num_train_examples": int(num_train),
        "num_test_examples": int(len(test_reps)),
        "probe_accuracy": float(probe_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
    }, f, indent=2)
