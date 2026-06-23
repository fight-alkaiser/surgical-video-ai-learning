import json
from collections import defaultdict

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json"
) as f:
    data = json.load(f)

triplet_names = data["categories"]["triplet"]
phase_names = data["categories"]["phase"]

triplet_phases = defaultdict(set)

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

        triplet_phases[triplet_name].add(
            phase_name
        )

for triplet, phases in sorted(
    triplet_phases.items(),
    key=lambda x: len(x[1])
):

    print(
        len(phases),
        triplet
    )
