source activate gym

export TF_CPP_MIN_LOG_LEVEL=3
export OPENBLAS_NUM_THREADS=1

# Apex training
PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH  python run/off_policy/train.py --algorithm=td3 --distributed=true
