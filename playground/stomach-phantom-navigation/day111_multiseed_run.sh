#!/bin/bash
# Day111 prep: check whether Day109/110's bias/variance orderings (flatten:
# real < zero < shuffled; sequence: shuffled < zero < real) reproduce across
# seeds, or were seed0-only noise. Runs seed1 and seed2 for both action
# modes at 50 epochs (half of Day107-110's 100, to fit four runs in one
# sitting -- Day107 found 50 epochs already showed the same qualitative
# picture as 100), each followed immediately by the bias/variance
# decomposition on that checkpoint. Sequential, not parallel -- Day96-97
# noted two runs in parallel maxed out this Mac mini's 8GB RAM.
set -e
cd "$(dirname "$0")"
source ../action-conditioned-video-prediction/.venv/bin/activate

for mode in flatten sequence; do
  for seed in 1 2; do
    tag="h20_n200_seed${seed}"
    if [ "$mode" != "flatten" ]; then
      tag="${tag}_${mode}"
    fi
    echo "=== training action-mode=${mode} seed=${seed} ==="
    python3 train.py --epochs 50 --batch-size 128 --seed "$seed" --action-mode "$mode" \
      > "outputs/day111_train_${tag}.log" 2>&1
    echo "=== bias/variance eval action-mode=${mode} seed=${seed} ==="
    python3 cfm_eval_distribution.py --checkpoint "outputs/model_cfm_${tag}.pt" --action-mode "$mode" \
      > "outputs/day111_distribution_${tag}.log" 2>&1
    # cfm_eval_distribution.py writes fixed filenames (day109_distribution_<tag>.json,
    # day109_sample_distribution_pca_<tag>.png) regardless of which day invokes it --
    # rename to day111 so they don't collide with Day109/110's seed0 artifacts.
    mv "outputs/day109_distribution_${tag}.json" "outputs/day111_distribution_${tag}.json"
    mv "outputs/day109_sample_distribution_pca_${tag}.png" "outputs/day111_sample_distribution_pca_${tag}.png"
  done
done
echo "all four runs finished"
