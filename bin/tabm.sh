#!/bin/bash

run_experiment() {
    local dataset=$1
    local numshot=$2
    local gpu=$3
    
    echo "Running ${dataset} ${numshot}-shot on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    python src/tabm_eval.py --dataset "$dataset" --numshot "$numshot"
}

datasets=(breastcancer breastcancer2 chemotherapy coloncancer diabetes heart respiratory)
numshots=(4 8 16 32 64 128 256 all)

experiments=()
for dataset in "${datasets[@]}"; do
    for numshot in "${numshots[@]}"; do
        experiments+=("$dataset $numshot")
    done
done

# Run experiments in parallel across 4 GPUs
gpu_count=4
for i in "${!experiments[@]}"; do
    gpu=$((i % gpu_count))
    experiment=(${experiments[i]})
    dataset=${experiment[0]}
    numshot=${experiment[1]}
    
    run_experiment "$dataset" "$numshot" "$gpu" &
    
    if (( (i + 1) % gpu_count == 0 )); then
        wait  # Wait for current batch to complete before starting next batch
    fi
done

wait
echo "All TabM experiments completed!"