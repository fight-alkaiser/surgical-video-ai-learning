import json
import statistics
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
# Calculate triplet lifetimes
# ----------------------------------------

active_triplets = {}
lifetime_records = []

frame_ids = sorted(int(frame) for frame in data["annotations"].keys())

for frame in frame_ids:

    frame_data = data["annotations"][str(frame)]

    current_triplets = set()

    for triplet in frame_data:
        if triplet[1] == -1 or triplet[7] == -1 or triplet[8] == -1:
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

            start_frame = active_triplets[triplet_name]
            lifetime = frame - start_frame

            lifetime_records.append(
                {
                    "triplet": triplet_name,
                    "start": start_frame,
                    "end": frame - 1,
                    "lifetime": lifetime,
                }
            )

            disappeared.append(triplet_name)

    for triplet_name in disappeared:
        del active_triplets[triplet_name]

# ----------------------------------------
# Handle triplets remaining in final frame
# ----------------------------------------

last_frame = frame_ids[-1]

for triplet_name, start_frame in active_triplets.items():

    lifetime = last_frame - start_frame + 1

    lifetime_records.append(
        {
            "triplet": triplet_name,
            "start": start_frame,
            "end": last_frame,
            "lifetime": lifetime,
        }
    )

# ----------------------------------------
# Lifetime statistics
# ----------------------------------------

lifetimes = [record["lifetime"] for record in lifetime_records]

print("=" * 50)
print("Triplet Lifetime Statistics")
print("=" * 50)

print(f"Number of lifetime records : {len(lifetimes)}")
print(f"Mean lifetime              : {statistics.mean(lifetimes):.2f}")
print(f"Median lifetime            : {statistics.median(lifetimes):.2f}")
print(f"Minimum lifetime           : {min(lifetimes)}")
print(f"Maximum lifetime           : {max(lifetimes)}")

if len(lifetimes) > 1:
    print(f"Standard deviation         : {statistics.stdev(lifetimes):.2f}")

# ----------------------------------------
# Longest lifetime events
# ----------------------------------------

print("\nTop 10 longest lifetime events\n")

sorted_records = sorted(
    lifetime_records,
    key=lambda x: x["lifetime"],
    reverse=True,
)

for record in sorted_records[:10]:

    instrument, verb, target = record["triplet"]

    print(
        f"{instrument:10} "
        f"{verb:15} "
        f"{target:20} "
        f"{record['lifetime']:4d} frames "
        f"({record['start']}–{record['end']})"
    )

print("\nLifetime Summary")

for threshold in [5,10,20,30,60]:

    count = sum(l <= threshold for l in lifetimes)

    print(
        f"<= {threshold:2d} frames : "
        f"{count:3d} "
        f"({count/len(lifetimes)*100:5.1f}%)"
    )

# ----------------------------------------
# Histogram
# ----------------------------------------

plt.figure(figsize=(8,5))

plt.hist(
    lifetimes,
    bins=30,
    edgecolor="black"
)

plt.axvline(
    statistics.mean(lifetimes),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {statistics.mean(lifetimes):.1f}"
)

plt.axvline(
    statistics.median(lifetimes),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median = {statistics.median(lifetimes):.1f}"
)

plt.xlim(0,60)

plt.title("Triplet Lifetime Distribution")
plt.xlabel("Lifetime (frames)")
plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()

plt.savefig("histogram.png", dpi=300)

plt.show()
