source activate gym

# python train.py

PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH  xvfb-run -s "-screen 0 1400x900x24" -a python train.py
