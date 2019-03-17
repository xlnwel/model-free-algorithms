source activate gym

export TF_CPP_MIN_LOG_LEVEL=3
export OPENBLAS_NUM_THREADS=1

# train single agent for off policy agent
PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH xvfb-run -s "-screen 0 1400x900x24" -a python train/off_policy/train.py --algorithm=td3
# PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH python train/off_policy/train.py --algorithm=sac
