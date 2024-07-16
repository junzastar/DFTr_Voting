#!/bin/bash
n_gpu=2  # number of gpu to use
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port 5235 train_mp6d.py --gpus=$n_gpu --gpu='0,1' #-checkpoint="path to/train_log/MP6D/checkpoints/FFB6D.pth.tar"
