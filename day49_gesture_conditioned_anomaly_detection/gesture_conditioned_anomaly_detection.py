import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ----------------------------------------
# Day48 closed the trajectory-forecasting arc: six structurally
# different methods all converged on the same ceiling, converging
# evidence that forecasting a human-teleoperated instrument's exact
# future path is close to an information limit, not a modeling gap.
# The owner chose to pivot to anomaly/deviation detection: instead of
# guessing an undetermined future coordinate, ask a well-posed
# descriptive question -- is the CURRENT motion typical for its
# context, or not?
#
# "Context" here is the active gesture (Day41's transcriptions).
# "Typical" is defined using EXPERT trials only (self-proclaimed
# skill == 'E' in the meta file) as the reference for normal motion --
# per gesture, per motion-state dimension. Novice/intermediate motion
# is then scored for how far it deviates from that expert-defined
# norm.
#
# Feature choice: translational velocity (3), rotational velocity (3),
# and gripper angle (1) -- NOT absolute position. Absolute tooltip
# position depends on where a trial happens to be set up in the
# workspace and isn't comparable across trials; motion-state variables
# (how fast/how the tool is moving and opening) are.
#
# Validation: this deviation score is built with ZERO access to skill
# labels. If it captures something real, it should still correlate
# with the independent skill labels (self-proclaimed N/I/E and the
# continuous GRS score) that were never used to fit it -- an honest
# external check, not a training target.
# ----------------------------------------

JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
TASK = "Suturing"
KINEMATICS_DIR = JIGSAWS_ROOT / TASK / "kinematics" / "AllGestures"
TRANSCRIPTIONS_DIR = JIGSAWS_ROOT / TASK / "transcriptions"
META_PATH = JIGSAWS_ROOT / TASK / f"meta_file_{TASK}.txt"

SLAVE_RIGHT_VEL_COLS = [69, 70, 71]
SLAVE_RIGHT_ROT_VEL_COLS = [72, 73, 74]
SLAVE_RIGHT_GRIPPER_COL = 75
FEATURE_COLS = SLAVE_RIGHT_VEL_COLS + SLAVE_RIGHT_ROT_VEL_COLS + [SLAVE_RIGHT_GRIPPER_COL]
FEATURE_NAMES = ["vel_x", "vel_y", "vel_z", "rot_vel_x", "rot_vel_y", "rot_vel_z", "gripper_angle"]

MIN_FRAMES_PER_GESTURE_FOR_NORM = 100  # need enough expert frames to estimate mean/std

# ----------------------------------------
# Load meta file: skill labels + GRS scores.
# ----------------------------------------

meta = {}
for line in META_PATH.read_text().splitlines():
    if not line.strip():
        continue
    parts = line.split()
    trial_name = parts[0]
    skill = parts[1]
    grs = int(parts[2])
    meta[trial_name] = {"skill": skill, "grs": grs}

trial_names = list(meta.keys())
print(f"Trials: {len(trial_names)}, skill distribution: "
      f"{ {s: sum(1 for m in meta.values() if m['skill'] == s) for s in ['N', 'I', 'E']} }")


def load_trial_features(trial_name):
    path = KINEMATICS_DIR / f"{trial_name}.txt"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 76:
            continue
        rows.append([float(parts[i]) for i in FEATURE_COLS])
    return np.array(rows) if rows else None


def load_trial_gestures(trial_name, num_frames):
    labels = np.full(num_frames, -1, dtype=np.int64)
    path = TRANSCRIPTIONS_DIR / f"{trial_name}.txt"
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        start, end, gesture = int(parts[0]), int(parts[1]), int(parts[2].lstrip("G"))
        end = min(end, num_frames - 1)
        if start <= end:
            labels[start:end + 1] = gesture
    return labels


trial_features = {}
trial_gestures = {}
for name in trial_names:
    features = load_trial_features(name)
    if features is None:
        continue
    trial_features[name] = features
    trial_gestures[name] = load_trial_gestures(name, len(features))

def fit_gesture_stats(frame_list):
    """frame_list: list of (feature_vector, gesture_id). Returns
    {gesture_id: (mean, std, n)}."""
    by_gesture = defaultdict(list)
    for feat, gesture in frame_list:
        if gesture != -1:
            by_gesture[gesture].append(feat)
    stats = {}
    for gesture, feats in by_gesture.items():
        arr = np.stack(feats)
        stats[gesture] = (arr.mean(axis=0), arr.std(axis=0) + 1e-6, len(arr))
    return stats


def deviation_scores(features, gestures, stats, usable_gestures):
    """Per-frame deviation score = sum of squared z-scores across
    feature dimensions, using the normal-motion stats for that
    frame's gesture. Frames with no gesture label, or a gesture not
    in `usable_gestures`, are skipped (returns NaN for those)."""
    scores = np.full(len(features), np.nan)
    for i, (feat, gesture) in enumerate(zip(features, gestures)):
        if gesture not in stats or gesture not in usable_gestures:
            continue
        mean, std, _ = stats[gesture]
        z = (feat - mean) / std
        scores[i] = float(np.sum(z ** 2))
    return scores


def run_reference_group(reference_trials, label):
    """Fits the normal-motion model from `reference_trials` (leave-
    one-trial-out for scoring a reference trial itself; the full
    reference-fit stats are used as-is for every trial NOT in the
    reference set) and scores every trial in the dataset against it."""

    print(f"\n{'=' * 70}\nReference group: {label} (n={len(reference_trials)})\n"
          f"  -> {reference_trials}\n{'=' * 70}")

    reference_frames_by_trial = {}
    all_reference_frames = []
    for name in reference_trials:
        frames = list(zip(trial_features[name], trial_gestures[name]))
        reference_frames_by_trial[name] = frames
        all_reference_frames.extend(frames)

    global_stats = fit_gesture_stats(all_reference_frames)
    usable_gestures = {g for g, (_, _, n) in global_stats.items()
                       if n >= MIN_FRAMES_PER_GESTURE_FOR_NORM}
    print(f"Gestures with enough reference frames (>= {MIN_FRAMES_PER_GESTURE_FOR_NORM}): "
          f"{sorted(usable_gestures)}")

    trial_results = {}
    for name in trial_names:
        if name not in trial_features:
            continue
        if name in reference_trials:
            loo_frames = [f for n, frames in reference_frames_by_trial.items() if n != name
                          for f in frames]
            stats = fit_gesture_stats(loo_frames)
        else:
            stats = global_stats

        scores = deviation_scores(trial_features[name], trial_gestures[name], stats,
                                   usable_gestures)
        valid = ~np.isnan(scores)
        trial_results[name] = {
            "skill": meta[name]["skill"],
            "grs": meta[name]["grs"],
            "mean_deviation": float(np.nanmean(scores)) if valid.any() else None,
            "num_scored_frames": int(valid.sum()),
            "num_total_frames": len(scores),
        }

    by_skill = defaultdict(list)
    for r in trial_results.values():
        if r["mean_deviation"] is not None:
            by_skill[r["skill"]].append(r["mean_deviation"])

    print(f"\nMean deviation score by self-proclaimed skill group ({label} reference):")
    for skill in ["N", "I", "E"]:
        vals = np.array(by_skill[skill])
        print(f"  {skill}: n={len(vals)}, mean={vals.mean():.2f}, median={np.median(vals):.2f}")

    grs_values = np.array([r["grs"] for r in trial_results.values() if r["mean_deviation"] is not None])
    deviation_values = np.array([r["mean_deviation"] for r in trial_results.values()
                                  if r["mean_deviation"] is not None])
    correlation = float(np.corrcoef(grs_values, deviation_values)[0, 1])
    print(f"Correlation (deviation vs. GRS): r = {correlation:.3f} "
          f"(negative expected if higher deviation <-> lower GRS)")

    return {
        "reference_trials": reference_trials,
        "usable_gestures": [int(g) for g in sorted(usable_gestures)],
        "trial_results": trial_results,
        "mean_deviation_by_skill": {s: float(np.mean(v)) for s, v in by_skill.items()},
        "correlation_deviation_vs_grs": correlation,
    }


# ----------------------------------------
# Reference group A: self-proclaimed experts (the original plan).
# Reference group B: top-GRS trials (same size as A, for a clean
# comparison), chosen after A revealed self-proclaimed skill doesn't
# track GRS well in this dataset (self-proclaimed 'E' mean GRS 16.3,
# actually the LOWEST of the three groups -- 'I' mean GRS 25.1 is the
# highest). If self-proclaimed skill is a poor proxy for actual
# technique, defining "normal" by measured performance (GRS) instead
# should be a more valid reference.
# ----------------------------------------

expert_trials = [n for n in trial_features if meta[n]["skill"] == "E"]
num_reference = len(expert_trials)

grs_ranked = sorted(trial_names, key=lambda n: meta[n]["grs"], reverse=True)
top_grs_trials = [n for n in grs_ranked if n in trial_features][:num_reference]

results = {
    "feature_names": FEATURE_NAMES,
    "self_proclaimed_expert_reference": run_reference_group(expert_trials, "self-proclaimed expert"),
    "top_grs_reference": run_reference_group(top_grs_trials, "top-GRS"),
}

# Sanity check on the diagnosis itself: mean GRS per self-proclaimed
# skill group.
grs_by_skill = defaultdict(list)
for name in trial_names:
    grs_by_skill[meta[name]["skill"]].append(meta[name]["grs"])
results["mean_grs_by_self_proclaimed_skill"] = {
    s: float(np.mean(v)) for s, v in grs_by_skill.items()
}
print("\n" + "=" * 70)
print("Mean GRS by self-proclaimed skill group (context for the two references above)")
print("=" * 70)
for s in ["N", "I", "E"]:
    print(f"  {s}: mean GRS = {results['mean_grs_by_self_proclaimed_skill'][s]:.1f}")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {output_dir / 'results.json'}")
