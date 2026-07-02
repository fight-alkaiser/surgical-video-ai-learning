import json

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
# Compress consecutive identical states
# ----------------------------------------

states = []

current_state = frame_triplets[frame_ids[0]]
start_frame = frame_ids[0]

for frame in frame_ids[1:]:

    similarity = jaccard_similarity(
        current_state,
        frame_triplets[frame]
    )

    if similarity < SIMILARITY_THRESHOLD:

        states.append(
            {
                "start": start_frame,
                "end": frame - 1,
                "duration": frame - start_frame,
                "triplets": current_state,
            }
        )

        current_state = frame_triplets[frame]
        start_frame = frame

states.append(
    {
        "start": start_frame,
        "end": frame_ids[-1],
        "duration": frame_ids[-1] - start_frame + 1,
        "triplets": current_state,
    }
)

# ----------------------------------------
# Statistics
# ----------------------------------------

print("=" * 60)
print("State Segmentation")
print("=" * 60)

print(f"Similarity Threshold : {SIMILARITY_THRESHOLD:.2f}")
print(f"Frames : {len(frame_ids)}")
print(f"States : {len(states)}")

compression = len(frame_ids) / len(states)

print(f"Compression Ratio : {compression:.2f}x")

# ----------------------------------------
# Display all states
# ----------------------------------------

print()
print("=" * 60)
print("State List")
print("=" * 60)

for i, state in enumerate(states, start=1):

    print()

    print(f"State {i}")

    print(
        f"Frames : {state['start']} - {state['end']}"
    )

    print(
        f"Duration : {state['duration']} frames"
    )

    if len(state["triplets"]) == 0:

        print("Triplets")

        print("  (None)")

    else:

        print("Triplets")

        for triplet in sorted(state["triplets"]):

            instrument, verb, target = triplet

            print(
                f"  {instrument:10} "
                f"{verb:12} "
                f"{target}"
            )

    print("-" * 60)
