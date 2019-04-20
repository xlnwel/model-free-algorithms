source activate gym

export TF_CPP_MIN_LOG_LEVEL=3
export OPENBLAS_NUM_THREADS=1

# train single agent for off policy agent
# python run/train.py -a=td3 -d=false -t=3
# python run/train.py -a=sac -d=false -t=1
python run/train.py -a=ppo -d=false -t=1
