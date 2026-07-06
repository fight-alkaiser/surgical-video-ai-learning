import json
import random
from pathlib import Path
from collections import defaultdict, Counter

# ----------------------------------------
# Question this script checks
#
# Day15 showed phase-level transitions are far more
# predictable (98.2%) than triplet-state (S) transitions
# (34.5%, from Day14). But phases move in a fixed clinical
# order, so that result is expected, not surprising.
#
# This script asks a sharper question: even if we only
# credit the Markov model on S-to-S transitions that
# happen to cross a phase boundary (the more
# procedurally-constrained moments), does accuracy get
# anywhere close to the phase-level number? Or does S
# alone stay insufficient either way?
#
# The expected answer, going in: S alone should stay
# insufficient in both cases, because what a surgeon does
# next depends on context and anatomy that a single
# current state cannot capture - not on whether a phase
# label happens to change at that moment.
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42


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
    frame_phase = {}

    for frame in frame_ids:

        annotations = data["annotations"][str(frame)]

        phase_id = annotations[0][-1]
        frame_phase[frame] = (
            phase_dict[str(phase_id)] if phase_id != -1 else None
        )

        triplet_set = set()

        for triplet in annotations:

            if -1 in (triplet[1], triplet[7], triplet[8]):
                continue

            instrument = instrument_dict[str(triplet[1])]
            verb = verb_dict[str(triplet[7])]
            target = target_dict[str(triplet[8])]

            if verb == "null_verb":
                continue

            triplet_set.add((instrument, verb, target))

        frame_triplets[frame] = frozenset(triplet_set)

    return frame_ids, frame_triplets, frame_phase


def segment_into_states(frame_ids, frame_triplets, frame_phase, threshold):
    """
    Same Jaccard-threshold segmentation as Day12-14, but
    also records the phase of each state's first frame, so
    later we can tell whether a state-to-state transition
    crosses a phase boundary.
    """

    signatures = [frame_triplets[frame_ids[0]]]
    phases = [frame_phase[frame_ids[0]]]

    for frame in frame_ids[1:]:

        similarity = jaccard_similarity(
            signatures[-1],
            frame_triplets[frame]
        )

        if similarity < threshold:
            signatures.append(frame_triplets[frame])
            phases.append(frame_phase[frame])

    return signatures, phases


def load_video(video_path):

    with open(video_path, "r") as f:
        data = json.load(f)

    frame_ids, frame_triplets, frame_phase = build_frame_data(data)

    return segment_into_states(
        frame_ids, frame_triplets, frame_phase, SIMILARITY_THRESHOLD
    )


# ----------------------------------------
# Step 1: load every video's (state signatures, state phases)
# ----------------------------------------

video_paths = sorted(LABELS_DIR.glob("VID*.json"))

video_data = {
    path.stem: load_video(path)
    for path in video_paths
}

# ----------------------------------------
# Step 2: same train/test split as Day14/15
# ----------------------------------------

all_video_ids = sorted(video_data.keys())

shuffled_ids = all_video_ids[:]
random.seed(RANDOM_SEED)
random.shuffle(shuffled_ids)

num_test = max(1, round(len(shuffled_ids) * TEST_RATIO))

test_ids = sorted(shuffled_ids[:num_test])
train_ids = sorted(shuffled_ids[num_test:])

# ----------------------------------------
# Step 3: fit the Markov model on train videos only
# (keyed directly by triplet-set signature, same idea as
# Day14, just without the integer-ID bookkeeping)
# ----------------------------------------

transition_counts = defaultdict(Counter)

for video_id in train_ids:

    signatures, _phases = video_data[video_id]

    for i in range(len(signatures) - 1):
        transition_counts[signatures[i]][signatures[i + 1]] += 1

most_likely_next = {
    current: next_counts.most_common(1)[0][0]
    for current, next_counts in transition_counts.items()
}

# ----------------------------------------
# Step 4: evaluate on test videos, split into
# "crosses a phase boundary" vs "stays within the same phase"
# ----------------------------------------

groups = {
    "boundary": {"total": 0, "correct": 0, "unseen": 0},
    "within_phase": {"total": 0, "correct": 0, "unseen": 0},
    "overall": {"total": 0, "correct": 0, "unseen": 0},
}

for video_id in test_ids:

    signatures, phases = video_data[video_id]

    for i in range(len(signatures) - 1):

        current_sig = signatures[i]
        actual_next_sig = signatures[i + 1]

        crosses_boundary = phases[i] != phases[i + 1]
        group_name = "boundary" if crosses_boundary else "within_phase"

        for group in (groups[group_name], groups["overall"]):

            group["total"] += 1

            if current_sig in most_likely_next:
                if most_likely_next[current_sig] == actual_next_sig:
                    group["correct"] += 1
            else:
                group["unseen"] += 1

# ----------------------------------------
# Results
# ----------------------------------------

print("=" * 60)
print("S-level Markov accuracy: phase-boundary vs within-phase")
print("=" * 60)

print(f"Train videos: {len(train_ids)}   Test videos: {len(test_ids)}")
print()
print(f"{'Group':<16}{'N':>8}{'Accuracy':>12}{'Unseen rate':>14}")

for name in ("overall", "boundary", "within_phase"):
    g = groups[name]
    accuracy = g["correct"] / g["total"] if g["total"] else float("nan")
    unseen_rate = g["unseen"] / g["total"] if g["total"] else float("nan")
    print(f"{name:<16}{g['total']:>8}{accuracy:>12.3f}{unseen_rate:>14.3f}")

print()
print("For reference:")
print(f"  Day15 phase-level accuracy   : 0.982")
print(f"  Day14 triplet-state accuracy : 0.345 (should match 'overall' above)")
