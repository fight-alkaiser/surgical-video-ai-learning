import json

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json"
) as f:
    data = json.load(f)

phase_names = data["categories"]["phase"]
triplet_names = data["categories"]["triplet"]

# phase transitions取得
transitions = []

current_phase = None

for frame in sorted(
    map(int, data["annotations"].keys())
):

    annotations = data["annotations"][str(frame)]

    phase_id = annotations[0][-1]

    if phase_id != current_phase:

        transitions.append(
            (frame, phase_names[str(phase_id)])
        )

        current_phase = phase_id

print("Transition Trigger Candidates")
print("=" * 60)

for transition_frame, phase_name in transitions:

    print()
    print("=" * 60)
    print(
        f"Transition at frame {transition_frame}"
    )
    print(
        f"Phase: {phase_name}"
    )
    print("=" * 60)

    before_triplets = set()
    after_triplets = set()

    # 遷移前10フレーム
    for frame in range(
        max(0, transition_frame - 10),
        transition_frame
    ):

        if str(frame) not in data["annotations"]:
            continue

        for ann in data["annotations"][str(frame)]:

            triplet_id = ann[0]

            if triplet_id == -1:
                continue

            before_triplets.add(
                triplet_names[str(triplet_id)]
            )

    # 遷移後10フレーム
    for frame in range(
        transition_frame,
        transition_frame + 10
    ):

        if str(frame) not in data["annotations"]:
            continue

        for ann in data["annotations"][str(frame)]:

            triplet_id = ann[0]

            if triplet_id == -1:
                continue

            after_triplets.add(
                triplet_names[str(triplet_id)]
            )

    new_triplets = (
        after_triplets - before_triplets
    )

    if len(new_triplets) == 0:

        print("No new triplets")

    else:

        for triplet in sorted(new_triplets):

            print(triplet)
