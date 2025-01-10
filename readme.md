# GRAMSSAT
Code of the paper GRAMSSAT: An Efficient Label Inference Attack against Two-party Split Learning based on Gradient Matching and Semi-supervised Learning.

## Environment Setup
./env_setup.sh

## Dataset Download
Download the following datasets to './datasets/Datasets'.

- CIFAR-10 or CIFAR-100:
        Use pytorch built-in classes.

- Tiny ImageNet:
        https://www.kaggle.com/c/tiny-imagenet

## Quick Start
### normal split learning
CUDA_VISIBLE_DEVICES=0 python sl_train.py -d CIFAR100 --path-dataset ./datasets/Datasets/CIFAR100  --k 5 --epochs 100 --n-labeled-per-class-for-ssl 4 --idx 1

### train surrogate top model
CUDA_VISIBLE_DEVICES=0 python train_surrogate.py --dataset-name CIFAR100 --dataset-path ./datasets/Datasets/CIFAR100 --total-epochs 100 --epochs 10 --lr 0.001 --alpha 0.0 --beta 1.0 --ssl-method MixMatch --idx 1

### validate attack performance
CUDA_VISIBLE_DEVICES=0 python validate.py --dataset CIFAR100 --dataset-path ./datasets/Datasets/CIFAR100 --resume-name-bottom CIFAR100_saved_framework_lr=0.1_n=4_idx=1.pth --resume-name-surrogate CIFAR100_saved_framework_lr=0.001_alpha=0.0_beta=1.0_num_layer=4_activation_func=None_use_bn=True_idx=1.pth

### baseline:model_completion
CUDA_VISIBLE_DEVICES=0 python model_completion.py --dataset-name CIFAR100 --dataset-path ./datasets/Datasets/CIFAR100 --n-labeled 400 --k 5 --resume-sub-dir surrogate_test --resume-name CIFAR100_saved_framework_lr=0.1_n=4_idx=1.pth --print-to-txt 1 --epochs 50 --idx 1

### Acknowledgements
This project is based on the [label-inference-attacks](https://github.com/FuChong-cyber/label-inference-attacks.git) by FuChong-cyber.