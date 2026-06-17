import json
from collections import Counter

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json",
    "r"
) as f:
    data = json.load(f)

triplets = data["categories"]["triplet"]

counter = Counter()

for frame, annotations in data["annotations"].items():

    for ann in annotations:

        triplet_id = ann[0]

        if triplet_id == -1:
            continue

        counter[triplet_id] += 1

print("Top 10 Triplets in VID01")
print()

for triplet_id, count in counter.most_common(10):

    print(
        count,
        triplets[str(triplet_id)]
    )