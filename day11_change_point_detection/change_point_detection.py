import json
import csv

# ----------------------------------------
# Parameters
# ----------------------------------------

SIMILARITY_THRESHOLD = 0.5

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
# Build triplet set for each frame
# ----------------------------------------

frame_triplets = {}

frame_ids = sorted(
    int(frame)
    for frame in data["annotations"].keys()
)

for frame in frame_ids:

    triplet_set = set()

    for triplet in data["annotations"][str(frame)]:

        if -1 in (
            triplet[1],
            triplet[7],
            triplet[8],
        ):
            continue

        instrument = instrument_dict[str(triplet[1])]
        verb = verb_dict[str(triplet[7])]
        target = target_dict[str(triplet[8])]

        # Ignore idle instruments
        if verb == "null_verb":
            continue

        triplet_set.add(
            (
                instrument,
                verb,
                target,
            )
        )

    frame_triplets[frame] = triplet_set

# ----------------------------------------
# Detect change points
# ----------------------------------------

change_points = []

for i in range(len(frame_ids) - 1):

    frame1 = frame_ids[i]
    frame2 = frame_ids[i + 1]

    set1 = frame_triplets[frame1]
    set2 = frame_triplets[frame2]

    if len(set1) == 0 or len(set2) == 0:
        continue

    similarity = len(set1 & set2) / len(set1 | set2)

    if similarity < SIMILARITY_THRESHOLD:

        removed = sorted(set1 - set2)
        added = sorted(set2 - set1)

        change_points.append(
            {
                "frame1": frame1,
                "frame2": frame2,
                "similarity": similarity,
                "removed": removed,
                "added": added,
            }
        )

# ----------------------------------------
# Display results
# ----------------------------------------

print("=" * 60)
print("Detected Change Points")
print("=" * 60)

for cp in change_points:

    print()

    print(
        f"Frame {cp['frame1']} -> {cp['frame2']}"
    )

    print(
        f"Similarity : {cp['similarity']:.3f}"
    )

    print()

    print("Removed")

    if len(cp["removed"]) == 0:
        print("  None")
    else:
        for triplet in cp["removed"]:
            print(
                " ",
                " | ".join(triplet)
            )

    print()

    print("Added")

    if len(cp["added"]) == 0:
        print("  None")
    else:
        for triplet in cp["added"]:
            print(
                " ",
                " | ".join(triplet)
            )

    print("-" * 60)

# ----------------------------------------
# Save CSV
# ----------------------------------------

with open(
    "change_points.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Frame1",
            "Frame2",
            "Similarity",
            "Removed",
            "Added",
        ]
    )

    for cp in change_points:

        removed = "; ".join(
            [
                " | ".join(t)
                for t in cp["removed"]
            ]
        )

        added = "; ".join(
            [
                " | ".join(t)
                for t in cp["added"]
            ]
        )

        writer.writerow(
            [
                cp["frame1"],
                cp["frame2"],
                round(cp["similarity"], 3),
                removed,
                added,
            ]
        )

print()
print("=" * 60)
print(f"Detected {len(change_points)} change points.")
print("CSV saved as change_points.csv")
print("=" * 60)
