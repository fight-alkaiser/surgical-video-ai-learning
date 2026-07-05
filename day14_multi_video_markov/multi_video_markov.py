import json
import random
from pathlib import Path
from collections import defaultdict, Counter

# ----------------------------------------
# Parameters
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ----------------------------------------
# Jaccard similarity
# (same as Day12 / Day13)
# ----------------------------------------


def jaccard_similarity(set1, set2):

    if len(set1 | set2) == 0:
        return 1.0

    return len(set1 & set2) / len(set1 | set2)


# ----------------------------------------
# Build a triplet set for every frame of one video
# (same logic as Day12 / Day13, wrapped in a function
# so it can be reused for all 50 videos instead of
# copy-pasting it 50 times)
# ----------------------------------------


def build_frame_triplets(data):

    instrument_dict = data["categories"]["instrument"]
    verb_dict = data["categories"]["verb"]
    target_dict = data["categories"]["target"]

    frame_ids = sorted(
        int(frame)
        for frame in data["annotations"].keys()
    )

    frame_triplets = {}

    for frame in frame_ids:

        triplet_set = set()

        for triplet in data["annotations"][str(frame)]:

            # Skip missing labels
            if -1 in (triplet[1], triplet[7], triplet[8]):
                continue

            instrument = instrument_dict[str(triplet[1])]
            verb = verb_dict[str(triplet[7])]
            target = target_dict[str(triplet[8])]

            # Ignore waiting tools
            if verb == "null_verb":
                continue

            triplet_set.add(
                (instrument, verb, target)
            )

        frame_triplets[frame] = frozenset(triplet_set)

    return frame_ids, frame_triplets


# ----------------------------------------
# Compress consecutive similar frames into states
# (same segmentation as Day12 / Day13)
# ----------------------------------------


def segment_into_states(frame_ids, frame_triplets, threshold):

    states = [frame_triplets[frame_ids[0]]]

    for frame in frame_ids[1:]:

        similarity = jaccard_similarity(
            states[-1],
            frame_triplets[frame]
        )

        if similarity < threshold:
            states.append(frame_triplets[frame])

    return states


def load_state_sequence(video_path):

    with open(video_path, "r") as f:
        data = json.load(f)

    frame_ids, frame_triplets = build_frame_triplets(data)

    return segment_into_states(
        frame_ids,
        frame_triplets,
        SIMILARITY_THRESHOLD
    )


# ----------------------------------------
# Step 1: compute a state sequence for every video
#
# Day13 used only VID01. With a single video, most
# transitions are only ever observed once, so the
# "probabilities" are not really statistics yet. Using
# all 50 videos lets the transition counts reflect real
# variation across different patients and operations.
# ----------------------------------------

video_paths = sorted(LABELS_DIR.glob("VID*.json"))

video_state_sequences = {
    path.stem: load_state_sequence(path)
    for path in video_paths
}

# ----------------------------------------
# Step 2: build one shared vocabulary across all videos
#
# Every distinct triplet-set signature seen in any video
# gets a single, stable integer ID. This is the "tokenizer"
# for this project: from now on, a surgical video is just
# a sequence of these integer IDs. This is exactly the kind
# of input a sequence model (RNN, Transformer) expects.
# ----------------------------------------

signature_to_id = {}

for video_id in sorted(video_state_sequences):
    for signature in video_state_sequences[video_id]:
        if signature not in signature_to_id:
            signature_to_id[signature] = len(signature_to_id)


def label_for(signature):

    if len(signature) == 0:
        return "(idle / no active triplet)"

    return ", ".join(
        "|".join(triplet)
        for triplet in sorted(signature)
    )


video_id_sequences = {
    video_id: [
        signature_to_id[signature]
        for signature in signatures
    ]
    for video_id, signatures in video_state_sequences.items()
}

# ----------------------------------------
# Step 3: split videos into train / test
#
# This is the same idea used to evaluate any machine
# learning model: fit the model only on the training
# videos, then check how well it predicts videos it has
# never seen. A fixed random seed makes the split
# reproducible.
# ----------------------------------------

all_video_ids = sorted(video_id_sequences.keys())

shuffled_ids = all_video_ids[:]
random.seed(RANDOM_SEED)
random.shuffle(shuffled_ids)

num_test = max(1, round(len(shuffled_ids) * TEST_RATIO))

test_ids = sorted(shuffled_ids[:num_test])
train_ids = sorted(shuffled_ids[num_test:])

# ----------------------------------------
# Step 4: build the transition (Markov) model
# using only the training videos
# ----------------------------------------

transition_counts = defaultdict(Counter)

for video_id in train_ids:

    sequence = video_id_sequences[video_id]

    for i in range(len(sequence) - 1):
        current_id = sequence[i]
        next_id = sequence[i + 1]
        transition_counts[current_id][next_id] += 1

most_likely_next = {
    current_id: next_counts.most_common(1)[0][0]
    for current_id, next_counts in transition_counts.items()
}

# Baseline: ignore the current state entirely and always
# predict whichever state most often comes next overall.
# A useful model should beat this baseline.
overall_next_counts = Counter()

for next_counts in transition_counts.values():
    overall_next_counts.update(next_counts)

baseline_state_id = overall_next_counts.most_common(1)[0][0]

# ----------------------------------------
# Step 5: evaluate next-state prediction accuracy
# on the held-out test videos
# ----------------------------------------

total_predictions = 0
correct_predictions = 0
baseline_correct = 0
unseen_state_count = 0

for video_id in test_ids:

    sequence = video_id_sequences[video_id]

    for i in range(len(sequence) - 1):

        current_id = sequence[i]
        actual_next_id = sequence[i + 1]

        total_predictions += 1

        if current_id in most_likely_next:
            predicted_id = most_likely_next[current_id]
            if predicted_id == actual_next_id:
                correct_predictions += 1
        else:
            unseen_state_count += 1

        if baseline_state_id == actual_next_id:
            baseline_correct += 1

accuracy = correct_predictions / total_predictions
baseline_accuracy = baseline_correct / total_predictions
unseen_rate = unseen_state_count / total_predictions

# ----------------------------------------
# Results
# ----------------------------------------

print("=" * 60)
print("Multi-Video State Vocabulary and Markov Prediction")
print("=" * 60)

print(f"Videos found            : {len(all_video_ids)}")
print(f"Train videos             : {len(train_ids)}")
print(f"Test videos              : {len(test_ids)}")
print(f"Vocabulary size          : {len(signature_to_id)} states")

print()
print(f"Test predictions made    : {total_predictions}")
print(f"Markov model accuracy    : {accuracy:.3f}")
print(f"Baseline accuracy        : {baseline_accuracy:.3f}")
print(
    f"Unseen current-state rate: {unseen_rate:.3f} "
    f"({unseen_state_count}/{total_predictions})"
)

print()
print("Test videos:", ", ".join(test_ids))

# ----------------------------------------
# Save the vocabulary so later days can reuse it
# instead of recomputing it from scratch
# ----------------------------------------

vocabulary_export = [
    {"id": state_id, "label": label_for(signature)}
    for signature, state_id in signature_to_id.items()
]

vocabulary_export.sort(key=lambda entry: entry["id"])

with open("state_vocabulary.json", "w") as f:
    json.dump(vocabulary_export, f, indent=2, ensure_ascii=False)

print()
print("Saved vocabulary to state_vocabulary.json")
