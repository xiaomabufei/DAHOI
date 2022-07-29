# DAHOI
DAHOI:Dynamic Anchor for Human-Object Interaction Detection

<img src="img/overall architecture of DAHOI.png"  width="1000"/>

## Requirements

- PyTorch >= 1.5.1
- torchvision >= 0.6.1
- loguru (log training process and env info)
  - tabulate (format log info)

```bash
pip install -r requirements.txt
```

- Compiling CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# test
python test.py
```

## Dataset Preparation

### HICO-DET

Please follow the HICO-DET dataset preparation of [GGNet](https://github.com/SherlockHolmes221/GGNet).

After preparation, the `data/hico_20160224_det` folder as follows:

```bash
data
├── hico_20160224_det
|   ├── images
|   |   ├── test2015
|   |   └── train2015
|   └── annotations
|       ├── anno_list.json
|       ├── corre_hico.npy
|       ├── file_name_to_obj_cat.json
|       ├── hoi_id_to_num.json
|       ├── hoi_list_new.json
|       ├── test_hico.json
|       └── trainval_hico.json
```

### V-COCO

Please follow the installation of [V-COCO](https://github.com/s-gupta/v-coco).

For evaluation, please put `vcoco_test.ids` and `vcoco_test.json` into `data/v-coco/data` folder.

After preparation, the `data/v-coco` folder as follows:

```bash
data
├── v-coco
|   ├── prior.pickle
|   ├── images
|   |   ├── train2014
|   |   └── val2014
|   ├── data
|   |   ├── instances_vcoco_all_2014.json
|   |   ├── vcoco_test.ids
|   |   └── vcoco_test.json
|   └── annotations
|       ├── corre_vcoco.npy
|       ├── test_vcoco.json
|       └── trainval_vcoco.json
```
#**********************************
## Evaluation

We currently provide results on HICO-DET.

Download the model to `params` folder.
- We test the model with NVIDIA RTX 3090 GPU, Pytorch 1.10.0, Python 3.8 and CUDA 11.2.

| Model | Full (def) | Rare (def) | None-Rare (def) | Full (ko) | Rare (ko) | None-Rare (ko) | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| VFDA-Tiny | 31.09 | 24.52 | 33.05 | 33.25 | 26.38 | 35.30 | [model](https://drive.google.com/file/d/1CLcR_upML6gakF8u11Vxdv_K6xZ0k3Dm/view?usp=sharing) |
| VFDA-Base | 32.72 | 24.68 | 34.65 | 34.57 | 26.84 | 36.04 | [model](https://drive.google.com/file/d/1jfw304N3KYvj25-94-T2a-SfLNkmcdj0/view?usp=sharing) |
| VFDA-Base*+ | 36.24 | 28.64 | 38.42 | 38.04 | 28.79 | 40.22 | [model](https://drive.google.com/file/d/1ve97e1foCxvsmllOWcFRx8kkYEDO882m/view?usp=sharing) |

Evaluating the model by running the following command.

`--eval_extra` to evaluate the spatio contribution.

`mAP_default.json` and `mAP_ko.json` will save in current folder.

- Swin-Tiny

```bash
python main.py --resume params/QAHOI_swin_tiny_mul3.pth --backbone swin_tiny --num_feature_levels 3 --use_nms --eval
```

- Swin-Base*+

```bash
python main.py --resume params/QAHOI_swin_base_384_22k_mul3.pth --backbone swin_base_384 --num_feature_levels 3 --use_nms --eval
```

- Swin-Large*+

```bash
python main.py --resume params/QAHOI_swin_large_384_22k_mul3.pth --backbone swin_large_384 --num_feature_levels 3 --use_nms --eval
```

## Training

- `--no_obj`: BCE loss for the object label

### HICO-DET

Download the pre-trained swin-tiny model from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) to `params` folder.

Training QAHOI with Swin-Tiny from scratch.

```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_tiny \
        --pretrained params/swin_tiny_patch4_window7_224.pth \
        --output_dir logs/swin_tiny_mul3 \
        --epochs 150 \
        --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --use_nms
```

Training QAHOI with Swin-Base*+ from scratch.

```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_base_384 \
        --pretrained params/swin_base_patch4_window7_224_22k.pth \
        --output_dir logs/swin_base_384_22k_mul3 \
        --epochs 150 \
        --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --use_nms
```
Training QAHOI with Swin-Large*+ from scratch.

```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone swin_large_384 \
        --pretrained params/swin_large_patch4_window12_384_22k.pth \
        --output_dir logs/swin_large_384_22k_mul3 \
        --epochs 150 \
        --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --use_nms
```

### V-COCO

```bash
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone [backbone_name] \
        --output_dir logs/[log_path] \
        --epochs 150 --lr_drop 120 \
        --num_feature_levels 3 \
        --num_queries 300 \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --use_nms [--no_obj]
```

- Train ResNet-50

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --backbone swin_tiny --pretrained params/swin_tiny_patch4_window7_224.pth --output_dir logs/swin_tiny_mul3_vcoco --epochs 150 --lr_drop 120 --num_feature_levels 3 --num_queries 300 --dataset_file vcoco --hoi_path data/v-coco --num_obj_classes 81 --num_verb_classes 29 --use_nms --no_obj
```

- Evaluation of V-COCO

Please generate the detection at first.

```bash
python generate_vcoco_official.py \
        --resume [checkpoint.pth] \
        --save_path vcoco.pickle \
        --hoi_path data/v-coco \
        --dataset_file vcoco \
        --backbone [backbone_name] \
        --num_feature_level 3 \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --use_nms [--no_obj]
```

Then, using the official code to evaluate.

```bash
python vsrl_eval.py --vcoco_path data/v-coco --detections vcoco.pickle
```

## Citation

~~~
@article{cjw,
  title={QAHOI: Query-Based Anchors for Human-Object Interaction Detection},
  author={Junwen Chen and Keiji Yanai},
  journal={arXiv preprint arXiv:2112.08647},
  year={2021}
}
~~~
