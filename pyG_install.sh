#!/usr/bin/env bash
TORCH=1.13.1
PYG_TORCH=1.13.0  # PyG wheel index for the torch 1.13.x series
CUDA=$1  # Supply as command line cpu or cu117
REQS="$(dirname "$0")/requirements.txt"  # used as constraints, so pip keeps conda's numpy / scipy
pip install torch==${TORCH}+${CUDA} --extra-index-url https://download.pytorch.org/whl/${CUDA} -c "$REQS"
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-${PYG_TORCH}+${CUDA}.html -c "$REQS"
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-${PYG_TORCH}+${CUDA}.html -c "$REQS"
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-${PYG_TORCH}+${CUDA}.html -c "$REQS"
pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-${PYG_TORCH}+${CUDA}.html -c "$REQS"
pip install torch-geometric==1.6.3 -c "$REQS"

# torch>=1.9 removed these aliases from torch._six, but torch-geometric 1.6.3 and data/data_loading.py
# still import them. Restore them in the installed torch.
SIX="$(python -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('torch').origin), '_six.py'))")"
grep -q "container_abcs" "$SIX" || cat >> "$SIX" <<'EOF'

# Added by pyG_install.sh: aliases removed in torch 1.9, needed by torch-geometric 1.6.3
import collections.abc as container_abcs
int_classes = int
EOF
