#!/bin/bash
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  $PROJECT_DIR/containers/TRACE.sif \
  python /workspace/set_transformer/run.py \
    --mode train \
    --gpu 0 \
    --run_name TAPES_test_run \
    --num_steps 5000 \
    --test_freq 500 \
    --save_freq 1000