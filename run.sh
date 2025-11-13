#!/bin/bash

# chmod +x run.sh

EPOCHS=300
BATCH_SIZE=1024
NUM_WORKERS=24

DATASETS=("COURSERA" "PATENT" "IMDB" "ARXIV") # "PATENT" "IMDB" "COURSERA" "ARXIV"
# DATASETS=("PATENT" "ARXIV") # "PATENT" "IMDB" "COURSERA" "ARXIV"
MODES=("baseline" "nodes" "node_semantic_node_structure" "node_edges" "full")
n=10
for ((i = 0; i < n; i++)); do 
  for dataset in "${DATASETS[@]}"; do
    if [[ $dataset == "IMDB" ]]; then
      BATCH_SIZE=1024
    elif [[ $dataset == "PATENT" ]]; then
      BATCH_SIZE=128
    elif [[ $dataset == "COURSERA" ]]; then
      BATCH_SIZE=128
    elif [[ $dataset == "ARXIV" ]]; then
      BATCH_SIZE=1024
    fi
    for mode in "${MODES[@]}"; do
      echo "======== Running: $dataset - $mode ========"
      python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --experiment $i
      echo ""
    done
  done
done
