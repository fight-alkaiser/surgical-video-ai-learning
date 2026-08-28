import json
import subprocess
from collections import Counter
from pathlib import Path

# ----------------------------------------
# Day41 starts a new dataset (JIGSAWS) after closing the CholecT50
# series at Day40. Before any hypothesis-driven analysis, this day
# does plain exploratory data analysis (EDA): what is actually in
# these files, how are they structured, what labels exist, and do
# video and kinematics agree with each other and with the dataset's
# own documentation (readme.txt). No modeling, no probes -- just
# looking, the same role Day01 played for CholecT50.
# ----------------------------------------

JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
TASKS = ["Suturing", "Needle_Passing", "Knot_Tying"]

# Standard JIGSAWS gesture vocabulary (Gao et al., 2014, "JHU-ISI
# Gesture and Skill Assessment Working Set (JIGSAWS): A Surgical
# Activity Dataset for Human Motion Modeling"). Not derivable from
# the data files themselves (which only give numeric gesture IDs) --
# recorded here for readability, cited to the paper rather than
# re-derived.
GESTURE_NAMES = {
    1: "Reaching for needle with right hand",
    2: "Positioning needle",
    3: "Pushing needle through tissue",
    4: "Transferring needle from left to right",
    5: "Moving to center with needle in grip",
    6: "Pulling suture with left hand",
    7: "Pulling suture with right hand",
    8: "Orienting needle",
    9: "Using right hand to help tighten suture",
    10: "Loosening more suture",
    11: "Dropping suture and moving to end points",
    12: "Reaching for needle with left hand",
    13: "Making C loop around right hand",
    14: "Reaching for suture with right hand",
    15: "Pulling suture with both hands",
}

results = {}

for task in TASKS:

    task_dir = JIGSAWS_ROOT / task
    print(f"\n{'=' * 60}\n{task}\n{'=' * 60}")

    # --- readme.txt (kinematic column layout, gesture vocab used) ---
    readme_path = task_dir / "readme.txt"
    readme_text = readme_path.read_text()

    # --- meta file: skill levels ---
    meta_path = task_dir / f"meta_file_{task}.txt"
    meta_lines = [
        line.split() for line in meta_path.read_text().splitlines() if line.strip()
    ]
    trial_names = [row[0] for row in meta_lines]
    self_proclaimed = [row[1] for row in meta_lines]
    grs_scores = [int(row[2]) for row in meta_lines]

    # Trial names look like "Suturing_B001" or "Needle_Passing_B001" --
    # the subject+trial code is always the LAST underscore-separated
    # part (e.g. "B001"), regardless of how many underscores are in
    # the task name itself.
    subjects = sorted(set(name.split("_")[-1][0] for name in trial_names))
    self_proclaimed_counts = Counter(self_proclaimed)

    print(f"Trials: {len(trial_names)}")
    print(f"Subjects (by letter code): {subjects} ({len(subjects)} total)")
    print(f"Self-proclaimed skill counts: {dict(self_proclaimed_counts)}")
    print(f"GRS score range: {min(grs_scores)}-{max(grs_scores)}, "
          f"mean {sum(grs_scores) / len(grs_scores):.1f}")

    # --- transcriptions: gesture vocabulary actually used ---
    transcription_dir = task_dir / "transcriptions"
    gesture_counter = Counter()
    segment_lengths = []
    for trial_name in trial_names:
        trans_path = transcription_dir / f"{trial_name}.txt"
        if not trans_path.exists():
            continue
        for line in trans_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            start, end, gesture = int(parts[0]), int(parts[1]), parts[2]
            gesture_id = int(gesture.lstrip("G"))
            gesture_counter[gesture_id] += 1
            segment_lengths.append(end - start)

    gestures_used = sorted(gesture_counter.keys())
    print(f"Gesture vocabulary used: {gestures_used} ({len(gestures_used)} distinct)")
    print("Gesture frequency (with standard names):")
    for gid in gestures_used:
        name = GESTURE_NAMES.get(gid, "?")
        print(f"  G{gid:<3d} {name:45s} count={gesture_counter[gid]}")
    print(f"Segment length (frames): min={min(segment_lengths)}, "
          f"max={max(segment_lengths)}, mean={sum(segment_lengths)/len(segment_lengths):.0f}")

    # --- kinematics: dimensionality and frame count, sanity-checked
    # against readme's documented 76 columns and against video frame
    # count for a sample of trials ---
    kinematics_dir = task_dir / "kinematics" / "AllGestures"
    kinematics_dims = set()
    kinematics_frame_counts = {}
    for trial_name in trial_names:
        kin_path = kinematics_dir / f"{trial_name}.txt"
        if not kin_path.exists():
            continue
        lines = kin_path.read_text().splitlines()
        if not lines:
            continue
        num_cols = len(lines[0].split())
        kinematics_dims.add(num_cols)
        kinematics_frame_counts[trial_name] = len(lines)

    print(f"Kinematics column counts observed: {kinematics_dims} "
          f"(readme documents 76)")

    # --- video: resolution/fps/frame count, sampled trials, checked
    # against kinematics frame count for the same trials ---
    video_dir = task_dir / "video"
    sample_trials = trial_names[:3]
    video_info = {}
    for trial_name in sample_trials:
        video_path = video_dir / f"{trial_name}_capture1.avi"
        if not video_path.exists():
            continue
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
             "-of", "json", str(video_path)],
            capture_output=True, text=True,
        )
        info = json.loads(probe.stdout)["streams"][0]
        kin_frames = kinematics_frame_counts.get(trial_name)
        video_info[trial_name] = {
            "width": info.get("width"), "height": info.get("height"),
            "fps": info.get("r_frame_rate"), "video_frames": info.get("nb_frames"),
            "duration_s": info.get("duration"), "kinematics_frames": kin_frames,
        }
        print(f"  {trial_name}: {info.get('width')}x{info.get('height')} "
              f"@ {info.get('r_frame_rate')}, video_frames={info.get('nb_frames')}, "
              f"kinematics_frames={kin_frames}, duration={info.get('duration')}s")

    results[task] = {
        "num_trials": len(trial_names),
        "subjects": subjects,
        "self_proclaimed_skill_counts": dict(self_proclaimed_counts),
        "grs_min": min(grs_scores), "grs_max": max(grs_scores),
        "grs_mean": sum(grs_scores) / len(grs_scores),
        "gestures_used": gestures_used,
        "gesture_frequency": dict(gesture_counter),
        "segment_length_min": min(segment_lengths),
        "segment_length_max": max(segment_lengths),
        "segment_length_mean": sum(segment_lengths) / len(segment_lengths),
        "kinematics_column_counts_observed": sorted(kinematics_dims),
        "sample_video_info": video_info,
    }

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nSaved results to {output_dir / 'results.json'}")
