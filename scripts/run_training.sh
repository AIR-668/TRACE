#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  $PROJECT_DIR/containers/TRACE.sif \
  python /workspace/set_transformer/run.py