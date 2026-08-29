import json
from pathlib import Path

import numpy as np

# ----------------------------------------
# Day41 set this arc's direction: short-horizon forecasting of a
# surgical instrument's future trajectory from its past kinematic
# state, entirely in the 76-dim numeric kinematics space (no video,
# no pixels -- clarified with the owner as a time-series/tabular
# forecasting problem, not a computer-vision one, since JIGSAWS'
# kinematics come directly from the da Vinci robot's own joint
# encoders, not from image analysis).
#
# Before any learned model, this day establishes what naive,
# zero-training baselines achieve -- the same discipline this project
# used throughout the CholecT50 series (e.g. Day01/20's trivial
# baselines, Day26's before/after comparison). Any future learned
# model must be checked against these numbers, not against "does the
# loss go down."
#
# Task: Suturing. Signal: the slave-right tooltip (the instrument
# actually moving in the workspace, columns 58-60 of the 76-dim
# kinematics vector per readme.txt) -- position xyz plus gripper
# angle (column 76). One arm only, to keep this first day small;
# both arms and the other two tasks are natural extensions.
#
# Anti-fabrication rule (from Day41, binding for this whole arc):
# every "prediction" here uses only the input window (frames before
# t); nothing at or after t is read when computing a prediction for
# time t. This holds trivially for the two closed-form baselines
# below, but is stated explicitly since later, learned days must
# satisfy it too.
# ----------------------------------------

JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
TASK = "Suturing"
KINEMATICS_DIR = JIGSAWS_ROOT / TASK / "kinematics" / "AllGestures"
META_PATH = JIGSAWS_ROOT / TASK / f"meta_file_{TASK}.txt"

FPS = 30.0
DT = 1.0 / FPS
INPUT_FRAMES = 30   # 1 second of past context
OUTPUT_FRAMES = 30  # 1 second forecast horizon
HORIZON_CHECKPOINTS_S = [0.1, 0.3, 0.5, 1.0]

# Slave-right block: columns 58-76 (1-indexed, readme.txt), i.e.
# 0-indexed columns 57-75. Within that block: xyz (57-59),
# rotation matrix (60-68), trans_vel (69-71), rot_vel (72-74),
# gripper angle (75).
SLAVE_RIGHT_XYZ_COLS = [57, 58, 59]
SLAVE_RIGHT_VEL_COLS = [69, 70, 71]
SLAVE_RIGHT_GRIPPER_COL = 75

trial_names = [
    line.split()[0] for line in META_PATH.read_text().splitlines() if line.strip()
]

# ----------------------------------------
# Load per-trial kinematics for the signal we care about: xyz (3),
# trans_vel (3), gripper angle (1) -- 7 columns total, one row per
# frame.
# ----------------------------------------


def load_trial_signal(trial_name):
    path = KINEMATICS_DIR / f"{trial_name}.txt"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 76:
            continue
        values = [float(parts[i]) for i in
                  SLAVE_RIGHT_XYZ_COLS + SLAVE_RIGHT_VEL_COLS + [SLAVE_RIGHT_GRIPPER_COL]]
        rows.append(values)
    return np.array(rows) if rows else None


trial_signals = {}
for name in trial_names:
    signal = load_trial_signal(name)
    if signal is not None and len(signal) > INPUT_FRAMES + OUTPUT_FRAMES:
        trial_signals[name] = signal

print(f"Loaded {len(trial_signals)} trials with usable length "
      f"(need > {INPUT_FRAMES + OUTPUT_FRAMES} frames)")

# ----------------------------------------
# Sliding windows: every valid (input, target) pair across every
# trial. Non-overlapping stride of OUTPUT_FRAMES keeps the number of
# windows manageable and avoids near-duplicate windows dominating the
# average.
# ----------------------------------------

STRIDE = OUTPUT_FRAMES

windows = []  # list of (trial_name, t) where t is the first target frame
for name, signal in trial_signals.items():
    num_frames = len(signal)
    t = INPUT_FRAMES
    while t + OUTPUT_FRAMES <= num_frames:
        windows.append((name, t))
        t += STRIDE

print(f"Total evaluation windows: {len(windows)}")

# ----------------------------------------
# Two zero-training baselines, both using only frames strictly
# before t (the anti-fabrication rule).
# ----------------------------------------


def predict_last_position_held(signal, t):
    """Baseline A: nothing moves -- future position = last observed
    position, future gripper angle = last observed angle."""
    last_xyz = signal[t - 1, 0:3]
    last_gripper = signal[t - 1, 6]
    xyz_pred = np.tile(last_xyz, (OUTPUT_FRAMES, 1))
    gripper_pred = np.full(OUTPUT_FRAMES, last_gripper)
    return xyz_pred, gripper_pred


def predict_constant_velocity(signal, t):
    """Baseline B: extrapolate using the robot's own last-observed
    translational velocity (already present in kinematics -- not
    finite-differenced, so it isn't noise-amplified by differencing)."""
    last_xyz = signal[t - 1, 0:3]
    last_vel = signal[t - 1, 3:6]
    last_gripper = signal[t - 1, 6]
    steps = (np.arange(1, OUTPUT_FRAMES + 1) * DT).reshape(-1, 1)
    xyz_pred = last_xyz + last_vel * steps
    gripper_pred = np.full(OUTPUT_FRAMES, last_gripper)
    return xyz_pred, gripper_pred


BASELINES = {
    "last_position_held": predict_last_position_held,
    "constant_velocity": predict_constant_velocity,
}

# ----------------------------------------
# Evaluate: Euclidean displacement error (meters) between predicted
# and actual tooltip xyz, at each horizon checkpoint, averaged over
# all windows.
# ----------------------------------------

horizon_frame_indices = [int(round(s * FPS)) - 1 for s in HORIZON_CHECKPOINTS_S]

results = {"task": TASK, "num_windows": len(windows), "baselines": {}}

for baseline_name, predict_fn in BASELINES.items():

    errors_by_checkpoint = {s: [] for s in HORIZON_CHECKPOINTS_S}
    full_horizon_errors = []

    for name, t in windows:
        signal = trial_signals[name]
        xyz_pred, _ = predict_fn(signal, t)
        xyz_true = signal[t:t + OUTPUT_FRAMES, 0:3]

        displacement_error = np.linalg.norm(xyz_pred - xyz_true, axis=1)
        full_horizon_errors.append(displacement_error.mean())

        for s, idx in zip(HORIZON_CHECKPOINTS_S, horizon_frame_indices):
            errors_by_checkpoint[s].append(displacement_error[idx])

    summary = {
        "mean_error_full_horizon_m": float(np.mean(full_horizon_errors)),
        "mean_error_by_checkpoint_m": {
            str(s): float(np.mean(errors_by_checkpoint[s])) for s in HORIZON_CHECKPOINTS_S
        },
    }
    results["baselines"][baseline_name] = summary

    print(f"\n=== {baseline_name} ===")
    print(f"Mean error averaged over full 1s horizon: "
          f"{summary['mean_error_full_horizon_m'] * 1000:.2f} mm")
    for s in HORIZON_CHECKPOINTS_S:
        err_mm = summary["mean_error_by_checkpoint_m"][str(s)] * 1000
        print(f"  at +{s:.1f}s: {err_mm:.2f} mm")

# ----------------------------------------
# Save a handful of example windows (predictions + ground truth) for
# the 3D trajectory plot, chosen from different trials for variety.
# ----------------------------------------

example_windows = []
seen_trials = set()
for name, t in windows:
    if name in seen_trials:
        continue
    seen_trials.add(name)
    signal = trial_signals[name]
    xyz_input = signal[t - INPUT_FRAMES:t, 0:3]
    xyz_true = signal[t:t + OUTPUT_FRAMES, 0:3]
    xyz_pred_a, _ = predict_last_position_held(signal, t)
    xyz_pred_b, _ = predict_constant_velocity(signal, t)
    example_windows.append({
        "trial": name, "t": t,
        "input_xyz": xyz_input.tolist(),
        "true_future_xyz": xyz_true.tolist(),
        "last_position_held_pred_xyz": xyz_pred_a.tolist(),
        "constant_velocity_pred_xyz": xyz_pred_b.tolist(),
    })
    if len(example_windows) >= 4:
        break

results["example_windows"] = example_windows

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_dir / 'results.json'}")
