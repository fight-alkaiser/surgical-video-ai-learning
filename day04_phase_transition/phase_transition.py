import json

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json"
) as f:
    data = json.load(f)

phase_names = data["categories"]["phase"]
triplet_names = data["categories"]["triplet"]

# frame -> phase
phases = {}

for frame, annotations in data["annotations"].items():
    phases[int(frame)] = annotations[0][-1]

# phase transition points
current = None
transitions = []

for frame in sorted(phases.keys()):
    phase = phases[frame]

    if phase != current:
        transitions.append((frame, phase))
        current = phase

print("Phase Transitions")
print("=" * 50)

for frame, phase_id in transitions:
    print(frame, phase_names[str(phase_id)])

print()
print("=" * 50)
print("Transition Analysis")
print("=" * 50)

for transition_frame, phase_id in transitions:

    phase_name = phase_names[str(phase_id)]

    print()
    print("=" * 60)
    print(
        f"Transition at frame {transition_frame}: "
        f"{phase_name}"
    )
    print("=" * 60)

    start = max(0, transition_frame - 10)
    end = transition_frame + 10

    for frame in range(start, end + 1):

        if str(frame) not in data["annotations"]:
            continue

        annotations = data["annotations"][str(frame)]

        triplets = []

        for ann in annotations:

            triplet_id = ann[0]

            if triplet_id == -1:
                continue

            triplets.append(
                triplet_names[str(triplet_id)]
            )

        print(frame, triplets)
