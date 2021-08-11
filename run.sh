#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=1 distributed_training.py
