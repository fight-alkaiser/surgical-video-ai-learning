import json
from collections import defaultdict, Counter

# ----------------------------------------
# Load annotation file
# ----------------------------------------

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json",
    "r"
) as f:
    data = json.load(f)

# ----------------------------------------
# Load label dictionaries
# ----------------------------------------

instrument_dict = data["categories"]["instrument"]
verb_dict = data["categories"]["verb"]
target_dict = data["categories"]["target"]

# ----------------------------------------
# Jaccard similarity
# ----------------------------------------


def jaccard_similarity(set1, set2):

    if len(set1 | set2) == 0:
        return 1.0

    return len(set1 & set2) / len(set1 | set2)


# ----------------------------------------
# Build triplet set for every frame
# (same as Day12)
# ----------------------------------------

frame_triplets = {}

SIMILARITY_THRESHOLD = 0.6

frame_ids = sorted(
    int(frame)
    for frame in data["annotations"].keys()
)

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

# ----------------------------------------
# Compress consecutive frames into states
# (same segmentation as Day12)
# ----------------------------------------

state_signatures = []

current_state = frame_triplets[frame_ids[0]]

for frame in frame_ids[1:]:

    similarity = jaccard_similarity(
        current_state,
        frame_triplets[frame]
    )

    if similarity < SIMILARITY_THRESHOLD:
        state_signatures.append(current_state)
        current_state = frame_triplets[frame]

state_signatures.append(current_state)

# ----------------------------------------
# Assign a state ID to each distinct signature
#
# The same triplet set (for example, only the
# irrigator aspirating fluid) can reappear at
# different points in the video. Here we give it
# the same ID every time it reappears, instead of
# treating every segment as a brand new state.
#
# This turns the video into a sequence of IDs,
# for example: 0, 1, 2, 1, 3, 1, 4, ...
# Later (Day14+) this integer sequence is exactly
# what gets fed into a sequence model as "tokens".
# ----------------------------------------

signature_to_id = {}
state_labels = []

for signature in state_signatures:

    if signature not in signature_to_id:
        new_id = len(signature_to_id)
        signature_to_id[signature] = new_id

        if len(signature) == 0:
            label = "(idle / no active triplet)"
        else:
            label = ", ".join(
                "|".join(triplet)
                for triplet in sorted(signature)
            )

        state_labels.append(label)

state_sequence = [
    signature_to_id[signature]
    for signature in state_signatures
]

# ----------------------------------------
# Count state transitions
#
# For every consecutive pair (current -> next) in
# the state sequence, count how often that pair
# occurs. This is the core idea of a Markov chain:
# we assume the next state depends only on the
# current state, not on the full history before it.
# ----------------------------------------

transition_counts = defaultdict(Counter)

for i in range(len(state_sequence) - 1):

    current_id = state_sequence[i]
    next_id = state_sequence[i + 1]

    transition_counts[current_id][next_id] += 1

# ----------------------------------------
# Summary
# ----------------------------------------

print("=" * 60)
print("State Transition Matrix")
print("=" * 60)

print(f"Frames : {len(frame_ids)}")
print(f"State occurrences (sequence length) : {len(state_sequence)}")
print(f"Unique states (vocabulary size) : {len(signature_to_id)}")

# ----------------------------------------
# State vocabulary
# ----------------------------------------

print()
print("=" * 60)
print("State Vocabulary")
print("=" * 60)

for state_id, label in enumerate(state_labels):
    print(f"S{state_id:<3} : {label}")

# ----------------------------------------
# Transition probabilities for each state
# ----------------------------------------

print()
print("=" * 60)
print("Transition Probabilities")
print("=" * 60)

deterministic_count = 0
branching_count = 0

for current_id in sorted(transition_counts.keys()):

    next_counts = transition_counts[current_id]
    total = sum(next_counts.values())

    if len(next_counts) == 1:
        deterministic_count += 1
    else:
        branching_count += 1

    print()
    print(f"From S{current_id} : {state_labels[current_id]}")
    print(f"  (seen as a starting state {total} time(s))")

    ranked_next = sorted(
        next_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for next_id, count in ranked_next:
        probability = count / total
        print(
            f"    -> S{next_id:<3} "
            f"P={probability:.2f} "
            f"({count}/{total})  "
            f"{state_labels[next_id]}"
        )

# ----------------------------------------
# Observation
# ----------------------------------------

print()
print("=" * 60)
print("Observation")
print("=" * 60)

print(
    f"{deterministic_count} state(s) always move to exactly "
    f"one next state (fully predictable)."
)
print(
    f"{branching_count} state(s) branch into two or more "
    f"possible next states (uncertain)."
)
