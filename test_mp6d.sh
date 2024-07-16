#!/bin/bash
tst_mdl= path to checkpoint  # checkpoint to test.
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 14152 train_mp6d.py --gpu '3' -eval_net -checkpoint $tst_mdl -test -test_pose # -debug
