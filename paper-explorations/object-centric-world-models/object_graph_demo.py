import json
from pathlib import Path

# ----------------------------------------
# What this script demonstrates
#
# The paper "Object-Centric World Models Meet Monte Carlo
# Tree Search" (Vakhitov et al., 2026) represents a scene as
# a graph: each Object is a Node, and the relationships
# between Objects are Edges. The world's Dynamics is then
# just how this graph changes over time.
#
# This script does NOT reproduce the paper's GNN or MCTS.
# It only demonstrates that the same Node/Edge idea can be
# built directly from data this project already has: a
# CholecT50 triplet (instrument, verb, target) is exactly
# an (Object) -[Interaction]-> (Object) edge.
#
# We reuse the same frame-triplet and state segmentation
# logic as Day12/13/14 to pick out three consecutive states
# around a meaningful moment (clipping the cystic duct), and
# print each one as a small Mermaid graph. Watching the graph
# change from state to state is a toy version of "Dynamics".
# ----------------------------------------

VIDEO_PATH = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json"
)

SIMILARITY_THRESHOLD = 0.6


def jaccard_similarity(set1, set2):

    if len(set1 | set2) == 0:
        return 1.0

    return len(set1 & set2) / len(set1 | set2)


def build_frame_triplets(data):

    instrument_dict = data["categories"]["instrument"]
    verb_dict = data["categories"]["verb"]
    target_dict = data["categories"]["target"]

    frame_ids = sorted(
        int(frame)
        for frame in data["annotations"].keys()
    )

    frame_triplets = {}

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

            triplet_set.add((instrument, verb, target))

        frame_triplets[frame] = frozenset(triplet_set)

    return frame_ids, frame_triplets


def segment_into_states(frame_ids, frame_triplets, threshold):

    states = [frame_triplets[frame_ids[0]]]

    for frame in frame_ids[1:]:

        similarity = jaccard_similarity(
            states[-1],
            frame_triplets[frame]
        )

        if similarity < threshold:
            states.append(frame_triplets[frame])

    return states


def print_mermaid_graph(title, triplet_set):

    print(f"### {title}")
    print()
    print("```mermaid")
    print("graph LR")

    if len(triplet_set) == 0:
        print("    idle[\"(idle)\"]")
    else:
        for instrument, verb, target in sorted(triplet_set):
            instrument_node = instrument.replace(" ", "_")
            target_node = target.replace(" ", "_")
            print(
                f"    {instrument_node}[\"{instrument}\"] "
                f"-- \"{verb}\" --> "
                f"{target_node}[\"{target}\"]"
            )

    print("```")
    print()


# ----------------------------------------
# Load VID01 and reuse Day12/13/14 segmentation
# ----------------------------------------

with open(VIDEO_PATH, "r") as f:
    data = json.load(f)

frame_ids, frame_triplets = build_frame_triplets(data)
states = segment_into_states(frame_ids, frame_triplets, SIMILARITY_THRESHOLD)

# ----------------------------------------
# Find the clipping event
# (clipper, clip, cystic_duct) and take the state
# right before it, the clipping state itself, and the
# state right after it
# ----------------------------------------

clip_triplet = ("clipper", "clip", "cystic_duct")

clip_index = next(
    i for i, state in enumerate(states)
    if clip_triplet in state
)

window = states[clip_index - 1: clip_index + 2]
titles = ["Before clipping", "During clipping", "After clipping"]

print("=" * 60)
print("Object-centric graph view of a clipping event (VID01)")
print("=" * 60)
print()

for title, triplet_set in zip(titles, window):
    print_mermaid_graph(title, triplet_set)
