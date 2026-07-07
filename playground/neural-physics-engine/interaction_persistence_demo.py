import json
import random
from pathlib import Path
from collections import defaultdict, Counter

# ----------------------------------------
# What this script demonstrates
#
# The Neural Physics Engine (Chang, Ullman, Torralba,
# Tenenbaum, ICLR 2017) predicts Dynamics by applying ONE
# shared pairwise Interaction function to every Object pair
# in a scene, then combining the effects. Because the same
# function is reused for every pair, it generalizes to scenes
# with a different number of Objects than it was trained on.
#
# This script does NOT train a neural network. It builds the
# simplest possible stand-in for that shared pairwise function
# -- an empirical lookup table -- and checks whether the same
# "reuse across object count" property shows up in real
# CholecT50 data.
#
# Object   = an instrument or an anatomical target
# Edge     = an (instrument, target) pair currently interacting
#            (the verb is dropped; we only ask "are these two
#            Objects interacting right now?")
# Interaction function f(instrument, target) = P(this edge is
#            still present in the next state | present now),
#            estimated once from training videos and then
#            applied identically to every edge, in any state,
#            regardless of how many other edges are present.
# Dynamics = the predicted next state's edge set, obtained by
#            applying f to every edge of the current state
#            independently.
#
# We reuse the same frame-triplet and state segmentation logic
# as Day12/13/14, and the same train/test video split idea as
# Day14.
# ----------------------------------------

LABELS_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels"
)

SIMILARITY_THRESHOLD = 0.6
TEST_RATIO = 0.2
RANDOM_SEED = 42


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


def edges_of(state):
    """Drop the verb: an Object-pair either is or isn't interacting."""
    return frozenset((instrument, target) for instrument, verb, target in state)


def load_edge_sequence(video_path):

    with open(video_path, "r") as f:
        data = json.load(f)

    frame_ids, frame_triplets = build_frame_triplets(data)
    states = segment_into_states(frame_ids, frame_triplets, SIMILARITY_THRESHOLD)

    return [edges_of(state) for state in states]


# ----------------------------------------
# Step 1: build an edge-state sequence for every video
# ----------------------------------------

video_paths = sorted(LABELS_DIR.glob("VID*.json"))

video_edge_sequences = {
    path.stem: load_edge_sequence(path)
    for path in video_paths
}

# ----------------------------------------
# Step 2: train / test split (same recipe as Day14)
# ----------------------------------------

all_video_ids = sorted(video_edge_sequences.keys())

shuffled_ids = all_video_ids[:]
random.seed(RANDOM_SEED)
random.shuffle(shuffled_ids)

num_test = max(1, round(len(shuffled_ids) * TEST_RATIO))

test_ids = sorted(shuffled_ids[:num_test])
train_ids = sorted(shuffled_ids[num_test:])

# ----------------------------------------
# Step 3: fit the shared pairwise interaction function
# on training videos only.
#
# persist_count[edge] / total_count[edge] approximates
# NPE's per-pair interaction function: given this Object
# pair is interacting now, how likely is it still
# interacting one state later? This single table is reused
# for every edge, in every state, no matter how many total
# edges that state has.
# ----------------------------------------

persist_count = Counter()
total_count = Counter()

for video_id in train_ids:

    sequence = video_edge_sequences[video_id]

    for i in range(len(sequence) - 1):

        current_edges = sequence[i]
        next_edges = sequence[i + 1]

        for edge in current_edges:
            total_count[edge] += 1
            if edge in next_edges:
                persist_count[edge] += 1

interaction_fn = {
    edge: persist_count[edge] / total_count[edge]
    for edge in total_count
}

# Baseline: ignore which specific edge it is, always use the
# overall persistence rate across all training edges.
overall_persist_rate = sum(persist_count.values()) / sum(total_count.values())

# ----------------------------------------
# Step 4: evaluate on held-out test videos.
#
# For every edge present in a test state, predict whether it
# persists into the next state using the shared interaction
# function, and compare against a baseline that only knows the
# overall rate (no per-pair structure at all).
# ----------------------------------------

total_predictions = 0
correct_predictions = 0
baseline_correct = 0
unseen_edge_count = 0

for video_id in test_ids:

    sequence = video_edge_sequences[video_id]

    for i in range(len(sequence) - 1):

        current_edges = sequence[i]
        next_edges = sequence[i + 1]

        for edge in current_edges:

            total_predictions += 1
            actual_persists = edge in next_edges

            if edge in interaction_fn:
                predicted_persists = interaction_fn[edge] >= 0.5
            else:
                unseen_edge_count += 1
                predicted_persists = overall_persist_rate >= 0.5

            if predicted_persists == actual_persists:
                correct_predictions += 1

            baseline_predicts_persists = overall_persist_rate >= 0.5
            if baseline_predicts_persists == actual_persists:
                baseline_correct += 1

accuracy = correct_predictions / total_predictions
baseline_accuracy = baseline_correct / total_predictions
unseen_edge_rate = unseen_edge_count / total_predictions

# ----------------------------------------
# Step 5: the point of the paper -- compositional
# generalization across object count.
#
# A whole-STATE model (like Day14's Markov model over full
# triplet-set signatures) can only predict a transition it has
# seen in that *exact* combination before. An edge-level model
# only needs to have seen each *pair* before, no matter how many
# other pairs happen to share that state. Since a handful of
# instrument-target pairs recombine into many different whole
# states, edge-level coverage should be far higher than
# whole-state coverage on unseen test videos.
# ----------------------------------------

train_whole_states = set()
for video_id in train_ids:
    train_whole_states.update(video_edge_sequences[video_id])

test_whole_state_total = 0
test_whole_state_seen = 0

for video_id in test_ids:
    for edges in video_edge_sequences[video_id]:
        test_whole_state_total += 1
        if edges in train_whole_states:
            test_whole_state_seen += 1

whole_state_coverage = test_whole_state_seen / test_whole_state_total
edge_coverage = 1 - unseen_edge_rate

# ----------------------------------------
# Results
# ----------------------------------------

print("=" * 60)
print("NPE-style pairwise interaction function on CholecT50")
print("=" * 60)
print()
print(f"Train videos: {len(train_ids)}, Test videos: {len(test_ids)}")
print(f"Distinct (instrument, target) edges learned: {len(interaction_fn)}")
print()
print(f"Edge persistence accuracy (shared interaction fn): {accuracy:.3f}")
print(f"Edge persistence accuracy (rate-only baseline):    {baseline_accuracy:.3f}")
print(f"Edges in test data never seen in training:         {unseen_edge_rate:.1%}")
print()
print("Compositional generalization check:")
print(f"  Whole-state signatures seen before (Day14-style): {whole_state_coverage:.1%}")
print(f"  Individual edges seen before (this script):       {edge_coverage:.1%}")
print()

# ----------------------------------------
# Step 6: concrete demo -- same function, different object
# counts. Show one test state with 1 edge and one with 3
# edges, and apply the identical interaction_fn to each.
# ----------------------------------------


def describe(edges):
    if len(edges) == 0:
        return "(idle)"
    return ", ".join(f"{i} -> {t}" for i, t in sorted(edges))


demo_states = []
for video_id in test_ids:
    for edges in video_edge_sequences[video_id]:
        if len(edges) in (1, 3) and all(e in interaction_fn for e in edges):
            demo_states.append((video_id, edges))

print("Same shared function applied to states with a different")
print("number of Objects present (no retraining in between):")
print()

shown_counts = set()
for video_id, edges in demo_states:
    if len(edges) in shown_counts:
        continue
    shown_counts.add(len(edges))
    print(f"[{video_id}] state with {len(edges)} edge(s): {describe(edges)}")
    for edge in sorted(edges):
        print(f"    f{edge} = {interaction_fn[edge]:.2f} persistence probability")
    print()
    if shown_counts == {1, 3}:
        break
