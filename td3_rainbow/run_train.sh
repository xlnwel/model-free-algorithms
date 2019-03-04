source activate gym

# train multiple agent
export OPENBLAS_NUM_THREADS=1
PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH  xvfb-run -s "-screen 0 1400x900x24" -a python distributed_train.py
