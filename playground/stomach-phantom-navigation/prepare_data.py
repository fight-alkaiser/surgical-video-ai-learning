"""Extract frames from episode videos and align them with action vectors.

Source: Open-H Dataset (nvidia/PhysicalAI-Robotics-Open-H-Embodiment, CC-BY-4.0)
        Endoscopy/cuhk/openh_dataset_full/find_greater_curvature
        (custom 2-motor soft robotic endoscope, silicone stomach phantom,
        task: "Search the greater curvature for a white oval suspicious region.")

Unlike ../action-conditioned-video-prediction/'s dVRK data (16-dim action),
this dataset's action space is just 2 continuous dims (antagonistic
cable-driven motor speeds) -- see this project's README.

Output: data/episodes/episode_NNNNNN_frames.npy   (N, 64, 64, 3) uint8
        data/episodes/episode_NNNNNN_actions.npy  (N, 2) float32
        data/episode_lengths.json                 frame count per episode
"""

import glob
import json
import os

import cv2
import numpy as np
import pandas as pd

FRAME_SIZE = 64

os.makedirs("data/episodes", exist_ok=True)

parquet_files = sorted(glob.glob("data/raw/episode_*.parquet"))
episode_lengths = {}

for pq_path in parquet_files:
    ep_id = os.path.basename(pq_path).replace(".parquet", "")  # e.g. episode_000000
    mp4_path = f"data/raw/{ep_id}_endo.mp4"
    if not os.path.exists(mp4_path):
        print(f"skip {ep_id}: no video")
        continue

    df = pd.read_parquet(pq_path)
    actions = np.stack(df["action"].values).astype(np.float32)

    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    cap.release()
    frames = np.stack(frames).astype(np.uint8)

    if len(frames) != len(actions):
        print(f"skip {ep_id}: frame/action mismatch ({len(frames)} vs {len(actions)})")
        continue

    np.save(f"data/episodes/{ep_id}_frames.npy", frames)
    np.save(f"data/episodes/{ep_id}_actions.npy", actions)
    episode_lengths[ep_id] = len(frames)
    print(f"{ep_id}: {len(frames)} frames")

with open("data/episode_lengths.json", "w") as f:
    json.dump(episode_lengths, f, indent=2)

print(f"\ntotal episodes: {len(episode_lengths)}, total frames: {sum(episode_lengths.values())}")
