#!/usr/bin/env bash

# These are solved together: scikit-sparse (for py-raccoon[exact]) needs scipy>=1.14, the code needs numpy<1.24,
# and graph-tool pulls in matplotlib, which therefore must come from conda as well.
conda install -c conda-forge --override-channels -y python=3.10 graph-tool==2.44 scikit-sparse==0.5.0 \
    numpy==1.23.5 scipy==1.14.1 matplotlib-base==3.5.3

# The pip torch wheel bundles an old libgomp that breaks graph-tool if torch is imported first.
# Preload the environment's libgomp instead (takes effect after re-activating the environment).
conda env config vars set LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
