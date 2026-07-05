import json

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json",
    "r"
) as f:
    data = json.load(f)

phases = {}

for frame, values in data["annotations"].items():
    phases[int(frame)] = values[0][-1]

phase_names = data["categories"]["phase"]

current = None

for frame in sorted(phases.keys()):
    phase = phases[frame]

    if phase != current:
        print(frame, phase_names[str(phase)])
        current = phase

transitions = []

current = None

for frame in sorted(phases.keys()):
    phase = phases[frame]

    if phase != current:
        transitions.append((frame, phase))
        current = phase

for i in range(len(transitions)):
    start_frame, phase = transitions[i]

    if i < len(transitions) - 1:
        end_frame = transitions[i + 1][0]
    else:
        end_frame = max(phases.keys())

    duration = end_frame - start_frame

    print(
        phase_names[str(phase)],
        duration,
        "seconds"
    )


