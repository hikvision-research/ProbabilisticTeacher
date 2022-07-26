# Getting Started
This page provides basic tutorials about the usage of PT.

## To reproduce the main results
```shell
bash train.sh
```
You can change ```--config configs/pt/final_c2f.yaml``` to other configs in ```configs/pt``` to reproduce the main results of other tasks.
## Resume training
```
CUDA_VISIBLE_DEVICES=0 \
python train_net.py \
     --num-gpus 1 \
     --resume \
     --config configs/pt/final_c2f.yaml \
      MODEL.ANCHOR_GENERATOR.NAME "DifferentiableAnchorGenerator" \
      UNSUPNET.EFL True \
      UNSUPNET.EFL_LAMBDA [0.5,0.5] \
      UNSUPNET.TAU [0.5,0.5] \
      MODEL.WEIGHTS /path/to/model_weights
```
## Other ablations
- Ablation study w/o anchor adapation, please run:
```bash
CUDA_VISIBLE_DEVICES=0 \
python train_net.py \
     --num-gpus 1 \
     --config configs/pt/final_c2f.yaml \
      MODEL.ANCHOR_GENERATOR.NAME "DefaultAnchorGenerator" \
      UNSUPNET.EFL True \
      UNSUPNET.EFL_LAMBDA [0.5,0.5] \
      UNSUPNET.TAU [0.5,0.5]
```
- Ablation study without EFL, please run:
```bash
CUDA_VISIBLE_DEVICES=0 \
python train_net.py \
     --num-gpus 1 \
     --config configs/pt/final_c2f.yaml \
      MODEL.ANCHOR_GENERATOR.NAME "DefaultAnchorGenerator" \
      UNSUPNET.EFL False \
      UNSUPNET.EFL_LAMBDA [0.5,0.5] \
      UNSUPNET.TAU [0.5,0.5]
```
- Also, you are free to adjust the temperature of classification and localization, as well as the hyper-parameter in EFL:
```bash
CUDA_VISIBLE_DEVICES=0 \
python train_net.py \
     --num-gpus 1 \
     --config configs/pt/final_c2f.yaml \
      MODEL.ANCHOR_GENERATOR.NAME "DefaultAnchorGenerator" \
      UNSUPNET.EFL True \
      UNSUPNET.EFL_LAMBDA [classification,localization] \
      UNSUPNET.TAU [classification,localization]
```

## Multi-GPU training
We conduct all the exps in the paper with a single V100 GPU with 32G memory. Yet, we also test this code with 2 Gefore 3090 GPUs with 24G memory.
```
CUDA_VISIBLE_DEVICES=0,1 \
python train_net.py \
     --num-gpus 2 \
     --config configs/pt/final_c2f.yaml \
      MODEL.ANCHOR_GENERATOR.NAME "DifferentiableAnchorGenerator" \
      UNSUPNET.EFL True \
      UNSUPNET.EFL_LAMBDA [0.5,0.5] \
      UNSUPNET.TAU [0.5,0.5]
```
