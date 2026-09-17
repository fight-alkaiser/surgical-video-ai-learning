#!/bin/bash
# Download Open-H stomach-phantom episodes into data/raw/.
# Source: nvidia/PhysicalAI-Robotics-Open-H-Embodiment (CC-BY-4.0),
# Endoscopy/cuhk/openh_dataset_full/find_greater_curvature -- a custom
# 2-motor soft robotic endoscope navigating a silicone stomach phantom,
# task: "Search the greater curvature for a white oval suspicious region."
# (462 episodes total, chunk-000 only). See Day105/106 notes.
# Usage: ./download_episodes.sh <start_ep> <end_ep_inclusive>
set -e
cd "$(dirname "$0")"
mkdir -p data/raw

START="${1:?usage: download_episodes.sh <start_ep> <end_ep_inclusive>}"
END="${2:?usage: download_episodes.sh <start_ep> <end_ep_inclusive>}"

BASE="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/resolve/main/Endoscopy/cuhk/openh_dataset_full/find_greater_curvature"
# the camera key contains a non-ASCII character in the source repo (endo三);
# curl needs it percent-encoded.
CAM_KEY="observation.images.endo%E4%B8%89"

for i in $(seq "$START" "$END"); do
    ep=$(printf "episode_%06d" "$i")
    pq="data/raw/${ep}.parquet"
    mp4="data/raw/${ep}_endo.mp4"

    if [ ! -f "$pq" ]; then
        curl -sL --fail "${BASE}/data/chunk-000/${ep}.parquet" -o "$pq" || { echo "FAILED parquet $ep"; rm -f "$pq"; continue; }
    fi
    if [ ! -f "$mp4" ]; then
        curl -sL --fail "${BASE}/videos/chunk-000/${CAM_KEY}/${ep}.mp4" -o "$mp4" || { echo "FAILED mp4 $ep"; rm -f "$mp4"; continue; }
    fi
    echo "done $ep"
done
echo "all downloads finished"
du -sh data/raw
