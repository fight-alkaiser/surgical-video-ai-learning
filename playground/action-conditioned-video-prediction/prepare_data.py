"""Extract frames from multiple episode videos and align them with action vectors.

Source: Open-H Dataset (nvidia/PhysicalAI-Robotics-Open-H-Embodiment, CC-BY-4.0)
        Surgical/hamlyn/peg_transfer/episode_000000..000019 (dVRK, peg transfer task)

Output: data/episodes/ep{NN}_frames.npy   (N, H, W, 3) uint8, resized square frames
        data/episodes/ep{NN}_actions.npy  (N, 16) float32, raw action vector per frame
        data/episode_lengths.json         frame count per episode, for splitting
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
    mp4_path = f"data/raw/{ep_id}_color.mp4"
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
