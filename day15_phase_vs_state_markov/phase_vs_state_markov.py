import json
import random
from pathlib import Path
from collections import defaultdict, Counter

# ----------------------------------------
# Parameters
#
# Same 50 videos, same train/test split (same
# random seed) as Day14, so the two accuracies are
# directly comparable.
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

TEST_RATIO = 0.2
RANDOM_SEED = 42


# ----------------------------------------
# Build the phase sequence for one video
#
# Unlike Day12-14, there is no similarity threshold
# here: CholecT50 already provides a phase label for
# every frame, so a "state" is just a maximal run of
# consecutive frames sharing the same phase.
# ----------------------------------------


def load_phase_sequence(video_path):

    with open(video_path, "r") as f:
        data = json.load(f)

    phase_names = data["categories"]["phase"]

    frame_ids = sorted(
        int(frame)
        for frame in data["annotations"].keys()
    )

    raw_phases = []

    for frame in frame_ids:

        phase_id = data["annotations"][str(frame)][0][-1]

        if phase_id == -1:
            continue

        raw_phases.append(phase_names[str(phase_id)])

    phase_sequence = [raw_phases[0]]

    for phase in raw_phases[1:]:
        if phase != phase_sequence[-1]:
            phase_sequence.append(phase)

    return phase_sequence


def evaluate_markov_model(train_sequences, test_sequences):
    """
    Fit a 1-step Markov model (most common next state per
    current state) on train_sequences, then measure its
    next-state prediction accuracy on test_sequences.
    Also reports a baseline that ignores the current state.
    """

    transition_counts = defaultdict(Counter)

    for sequence in train_sequences:
        for i in range(len(sequence) - 1):
            transition_counts[sequence[i]][sequence[i + 1]] += 1

    most_likely_next = {
        current: next_counts.most_common(1)[0][0]
        for current, next_counts in transition_counts.items()
    }

    overall_next_counts = Counter()
    for next_counts in transition_counts.values():
        overall_next_counts.update(next_counts)

    baseline_state = overall_next_counts.most_common(1)[0][0]

    total = 0
    correct = 0
    baseline_correct = 0
    unseen = 0

    for sequence in test_sequences:
        for i in range(len(sequence) - 1):

            current = sequence[i]
            actual_next = sequence[i + 1]

            total += 1

            if current in most_likely_next:
                if most_likely_next[current] == actual_next:
                    correct += 1
            else:
                unseen += 1

            if baseline_state == actual_next:
                baseline_correct += 1

    return {
        "total": total,
        "accuracy": correct / total,
        "baseline_accuracy": baseline_correct / total,
        "unseen_rate": unseen / total,
    }


# ----------------------------------------
# Step 1: load a phase sequence for every video
# ----------------------------------------

video_paths = sorted(LABELS_DIR.glob("VID*.json"))

video_phase_sequences = {
    path.stem: load_phase_sequence(path)
    for path in video_paths
}

# ----------------------------------------
# Step 2: same train/test split as Day14
# (same video list, same seed, same ratio)
# ----------------------------------------

all_video_ids = sorted(video_phase_sequences.keys())

shuffled_ids = all_video_ids[:]
random.seed(RANDOM_SEED)
random.shuffle(shuffled_ids)

num_test = max(1, round(len(shuffled_ids) * TEST_RATIO))

test_ids = sorted(shuffled_ids[:num_test])
train_ids = sorted(shuffled_ids[num_test:])

train_sequences = [video_phase_sequences[v] for v in train_ids]
test_sequences = [video_phase_sequences[v] for v in test_ids]

results = evaluate_markov_model(train_sequences, test_sequences)

# ----------------------------------------
# Results
# ----------------------------------------

print("=" * 60)
print("Phase-level Markov prediction (macro level)")
print("=" * 60)

print(f"Videos found          : {len(all_video_ids)}")
print(f"Train videos           : {len(train_ids)}")
print(f"Test videos            : {len(test_ids)}")
print(f"Test predictions made  : {results['total']}")
print(f"Phase-level accuracy   : {results['accuracy']:.3f}")
print(f"Baseline accuracy      : {results['baseline_accuracy']:.3f}")
print(f"Unseen current-state rate: {results['unseen_rate']:.3f}")

print()
print("Test videos:", ", ".join(test_ids))

print()
print("=" * 60)
print("Comparison with Day14 (triplet-state / micro level)")
print("=" * 60)
print(f"{'Level':<28}{'Accuracy':>10}{'Baseline':>12}")
print(f"{'Phase (macro, today)':<28}{results['accuracy']:>10.3f}{results['baseline_accuracy']:>12.3f}")
print(f"{'Triplet-state (micro, Day14)':<28}{0.345:>10.3f}{0.121:>12.3f}")
