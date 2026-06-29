import json
from collections import defaultdict
import matplotlib.pyplot as plt

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
# Count recurrence of each triplet
# ----------------------------------------

active_triplets = {}
recurrence_count = defaultdict(int)

frame_ids = sorted(int(frame) for frame in data["annotations"].keys())

for frame in frame_ids:

    frame_data = data["annotations"][str(frame)]

    current_triplets = set()

    for triplet in frame_data:

        # Skip invalid labels
        if -1 in (triplet[1], triplet[7], triplet[8]):
            continue

        instrument = instrument_dict[str(triplet[1])]
        verb = verb_dict[str(triplet[7])]
        target = target_dict[str(triplet[8])]

        triplet_name = (instrument, verb, target)

        current_triplets.add(triplet_name)

        if triplet_name not in active_triplets:
            active_triplets[triplet_name] = frame

    disappeared = []

    for triplet_name in active_triplets:

        if triplet_name not in current_triplets:

            recurrence_count[triplet_name] += 1
            disappeared.append(triplet_name)

    for triplet_name in disappeared:
        del active_triplets[triplet_name]

# ----------------------------------------
# Remaining triplets
# ----------------------------------------

for triplet_name in active_triplets:
    recurrence_count[triplet_name] += 1

# ----------------------------------------
# Display recurrence ranking
# ----------------------------------------

print("=" * 50)
print("Triplet Recurrence Ranking")
print("=" * 50)

sorted_triplets = sorted(
    recurrence_count.items(),
    key=lambda x: x[1],
    reverse=True,
)

for triplet, count in sorted_triplets[:20]:

    instrument, verb, target = triplet

    print(
        f"{instrument:10} "
        f"{verb:15} "
        f"{target:20} "
        f"{count:3d} occurrences"
    )

# ----------------------------------------
# Summary
# ----------------------------------------

counts = list(recurrence_count.values())

from collections import Counter

counter = Counter(counts)

print("\nRecurrence Summary")

for occurrence in sorted(counter):

    print(
        f"{occurrence:2d} occurrence(s): "
        f"{counter[occurrence]:2d}"
    )

print()

print(f"Unique triplets : {len(recurrence_count)}")

print(f"Total events    : {sum(counts)}")

print(f"Average recurrence : {sum(counts)/len(counts):.2f}")

# ----------------------------------------
# Histogram
# ----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    counts,
    bins=range(1, max(counts) + 2),
    align="left",
    edgecolor="black",
)

plt.title("Triplet Recurrence Distribution")
plt.xlabel("Number of Occurrences")
plt.ylabel("Number of Unique Triplets")

plt.tight_layout()

plt.savefig("recurrence_histogram.png", dpi=300)

plt.show()
