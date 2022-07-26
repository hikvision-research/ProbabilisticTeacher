#!/bin/bash

#cd /data1/projects/PT

CUDA_VISIBLE_DEVICES=0 \
python train_net.py \
     --num-gpus 1 \
     --config configs/pt/final_c2f.yaml \
      MODEL.ANCHOR_GENERATOR.NAME "DifferentiableAnchorGenerator" \
      UNSUPNET.EFL True \
      UNSUPNET.EFL_LAMBDA [0.5,0.5] \
      UNSUPNET.TAU [0.5,0.5]