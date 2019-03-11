source activate gym

export TF_CPP_MIN_LOG_LEVEL=3
export OPENBLAS_NUM_THREADS=1

# train single agent
# PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH  xvfb-run -s "-screen 0 1400x900x24" -a python train.py

# Apex training
PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH  python distributed_train.py
