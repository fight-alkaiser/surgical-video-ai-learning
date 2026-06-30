import json
import statistics
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
# Build triplet set for each frame
# ----------------------------------------

frame_triplets = {}

frame_ids = sorted(int(frame) for frame in data["annotations"].keys())

for frame in frame_ids:

    triplet_set = set()

    for triplet in data["annotations"][str(frame)]:

        if -1 in (triplet[1], triplet[7], triplet[8]):
            continue

        instrument = instrument_dict[str(triplet[1])]
        verb = verb_dict[str(triplet[7])]
        target = target_dict[str(triplet[8])]

        if verb == "null_verb":
            continue
        if target == "null_target":
            continue

        triplet_set.add(
            (instrument, verb, target)
        )

    frame_triplets[frame] = triplet_set

# ----------------------------------------
# Calculate Jaccard Similarity
# ----------------------------------------

similarities = []

for i in range(len(frame_ids)-1):

    frame1 = frame_ids[i]
    frame2 = frame_ids[i+1]

    set1 = frame_triplets[frame1]
    set2 = frame_triplets[frame2]

    if len(set1)==0 or len(set2)==0:
        continue

    union = set1 | set2

    if len(union) == 0:
        similarity = 1.0
    else:
        similarity = len(set1 & set2) / len(union)

    similarities.append(
        {
            "frame1": frame1,
            "frame2": frame2,
            "similarity": similarity
        }
    )

# ----------------------------------------
# Statistics
# ----------------------------------------

values = [x["similarity"] for x in similarities]

print("=" * 50)
print("Frame-to-Frame Similarity")
print("=" * 50)

print(f"Number of comparisons : {len(values)}")
print(f"Mean similarity       : {statistics.mean(values):.3f}")
print(f"Median similarity     : {statistics.median(values):.3f}")
print(f"Minimum similarity    : {min(values):.3f}")
print(f"Maximum similarity    : {max(values):.3f}")

# ----------------------------------------
# Lowest similarity frames
# ----------------------------------------

print("\nLowest Similarity Transitions\n")

lowest = sorted(
    similarities,
    key=lambda x: x["similarity"]
)

for item in lowest[:10]:

    print(
        f"{item['frame1']:4d} -> {item['frame2']:4d}"
        f" : {item['similarity']:.3f}"
    )

print("\n")
print("=" * 60)
print("Detailed Lowest Similarity Frames")
print("=" * 60)

for item in lowest[:5]:

    f1 = item["frame1"]
    f2 = item["frame2"]

    print(f"\nFrame {f1}")

    if len(frame_triplets[f1]) == 0:
        print("  (No triplets)")
    else:
        for t in sorted(frame_triplets[f1]):
            print(" ", t)

    print("\nFrame", f2)

    if len(frame_triplets[f2]) == 0:
        print("  (No triplets)")
    else:
        for t in sorted(frame_triplets[f2]):
            print(" ", t)

    print(f"\nSimilarity = {item['similarity']:.3f}")
    print("-"*60)

# ----------------------------------------
# Histogram
# ----------------------------------------

plt.figure(figsize=(8,5))

plt.hist(
    values,
    bins=20,
    edgecolor="black"
)

plt.title("Frame-to-Frame Jaccard Similarity")
plt.xlabel("Similarity")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("frame_similarity_histogram.png", dpi=300)

plt.show()


# ----------------------------------------
# Show added / removed triplets
# ----------------------------------------

print()
print("=" * 60)
print("Triplet Changes at Lowest Similarity Frames")
print("=" * 60)

for item in lowest[:5]:

    frame1 = item["frame1"]
    frame2 = item["frame2"]
    similarity = item["similarity"]

    set1 = frame_triplets[frame1]
    set2 = frame_triplets[frame2]

    removed = set1 - set2
    added = set2 - set1

    print()
    print(f"{frame1} -> {frame2}")
    print(f"Similarity = {similarity:.3f}")

    print()
    print("Removed")

    if removed:
        for t in sorted(removed):
            print(" ", t)
    else:
        print("  None")

    print()
    print("Added")

    if added:
        for t in sorted(added):
            print(" ", t)
    else:
        print("  None")

    print("-" * 60)