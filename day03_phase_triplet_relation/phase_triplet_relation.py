import json
from collections import Counter

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json",
    "r"
) as f:
    data = json.load(f)

triplet_names = data["categories"]["triplet"]
phase_names = data["categories"]["phase"]

phase_triplets = {}

for frame, annotations in data["annotations"].items():

    phase_id = annotations[0][-1]

    if phase_id not in phase_triplets:
        phase_triplets[phase_id] = Counter()

    for ann in annotations:

        triplet_id = ann[0]

        if triplet_id == -1:
            continue

        phase_triplets[phase_id][triplet_id] += 1

for phase_id in sorted(phase_triplets.keys()):

    print()
    print("=" * 50)
    print("PHASE:", phase_names[str(phase_id)])
    print("=" * 50)

    for triplet_id, count in phase_triplets[phase_id].most_common(10):

        print(
            count,
            triplet_names[str(triplet_id)]
        )
