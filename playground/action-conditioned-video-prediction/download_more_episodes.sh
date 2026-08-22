#!/bin/bash
# Day80: download additional Open-H peg_transfer episodes (episode_000020..000099)
# to test the data-scale hypothesis from Day79. Source: nvidia/PhysicalAI-Robotics-Open-H-Embodiment
# (CC-BY-4.0), Surgical/hamlyn/peg_transfer, same task/camera as the existing 20 episodes.
set -e
cd "$(dirname "$0")"
mkdir -p data/raw

BASE="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/resolve/main/Surgical/hamlyn/peg_transfer"

for i in $(seq 20 99); do
    ep=$(printf "episode_%06d" "$i")
    pq="data/raw/${ep}.parquet"
    mp4="data/raw/${ep}_color.mp4"

    if [ ! -f "$pq" ]; then
        curl -sL --fail "${BASE}/data/chunk-000/${ep}.parquet" -o "$pq" || { echo "FAILED parquet $ep"; rm -f "$pq"; continue; }
    fi
    if [ ! -f "$mp4" ]; then
        curl -sL --fail "${BASE}/videos/chunk-000/observation.images.color/${ep}.mp4" -o "$mp4" || { echo "FAILED mp4 $ep"; rm -f "$mp4"; continue; }
    fi
    echo "done $ep"
done
echo "all downloads finished"
du -sh data/raw
