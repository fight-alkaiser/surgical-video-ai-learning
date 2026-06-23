import json
from collections import Counter, defaultdict

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json"
) as f:
    data = json.load(f)

triplet_names = data["categories"]["triplet"]
phase_names = data["categories"]["phase"]

triplet_phase_counts = defaultdict(Counter)

for frame, annotations in data["annotations"].items():

    phase_id = annotations[0][-1]

    if phase_id == -1:
        continue

    phase_name = phase_names[str(phase_id)]

    for ann in annotations:

        triplet_id = ann[0]

        if triplet_id == -1:
            continue

        triplet_name = triplet_names[str(triplet_id)]

        triplet_phase_counts[triplet_name][phase_name] += 1

results = []

for triplet, counts in triplet_phase_counts.items():

    total = sum(counts.values())

    if total < 20:
        continue

    dominant_phase, dominant_count = counts.most_common(1)[0]

    ratio = dominant_count / total

    results.append(
        (
            ratio,
            total,
            dominant_phase,
            triplet
        )
    )

results.sort(reverse=True)

print()
print("Most Phase-Specific Triplets")
print("=" * 60)

for ratio, total, phase, triplet in results[:20]:

    print(
        f"{ratio:.2f}",
        total,
        phase,
        triplet
    )
