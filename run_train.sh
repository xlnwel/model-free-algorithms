source activate gym

export TF_CPP_MIN_LOG_LEVEL=3
export OPENBLAS_NUM_THREADS=1

# train single agent for off policy agent
python run/train.py --algorithm=td3 --trials=3
# python train/train.py --algorithm=sac
# python run/train.py --algorithm=td3 --distributed=true

# xvfb-run -s "-screen 0 1400x900x24" -a python run/train.py --algorithm=ppo --distributed=false --trials=3
