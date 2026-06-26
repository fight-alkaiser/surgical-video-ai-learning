import json

with open(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json"
) as f:
    data = json.load(f)

triplet_names = data["categories"]["triplet"]

MIN_DURATION = 20

active = {}
printed = set()

print("Triplet Lifetimes")
print("=" * 60)

for frame in sorted(data["annotations"].keys(), key=int):

    frame = int(frame)

    current = set()

    for ann in data["annotations"][str(frame)]:

        triplet_id = ann[0]

        if triplet_id == -1:
            continue

        current.add(
            triplet_names[str(triplet_id)]
        )

    # newly appeared
    for triplet in current:

        if triplet not in active:

            active[triplet] = frame

    # disappeared
    for triplet in list(active.keys()):

        if triplet not in current:

            start = active[triplet]
            duration = frame - start

            if duration >= MIN_DURATION:

                print()
                print(f"Frame {start}")
                print(f"+ {triplet}")
                print(
                    f"- {triplet} "
                    f"(duration: {duration} frames)"
                )

                printed.add(triplet)

            del active[triplet]