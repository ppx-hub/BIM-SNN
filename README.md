# Training Script

## 1. Baseline

```bash
CUDA_VISIBLE_DEVICES='4' python main.py --alpha 0.8 --ckpt_path ckpt/ --train --tensorboard_path ./ --use_tensorboard True --batch_size 64 --modulation Normal
```



## 2. Inverse

```bash
CUDA_VISIBLE_DEVICES='4' python main.py --alpha 0.8 --ckpt_path ckpt/ --train --tensorboard_path ./ --inverse --use_tensorboard True --batch_size 64 --modulation Normal
```



## 3. Metamodal

```bash
CUDA_VISIBLE_DEVICES='4' python main_metamodal.py --alpha 0.8 --ckpt_path ckpt/ --train --tensorboard_path ./
```



## 4. Inverse + Metamodal

```bash
CUDA_VISIBLE_DEVICES='4' python main_inverse_metamodal.py --alpha 0.8 --ckpt_path ckpt3/ --train --tensorboard_path ./ --meta_ratio 0.1 --inverse --use_tensorboard True
```



## 5. Unimodal_Finetune

```bash
CUDA_VISIBLE_DEVICES='4' python main_unimodalFintune.py --train --ckpt_path ckpt_unimodal/ --train --ckpt_path /home/hexiang/OGM-GE_CVPR2022/ckpt2/Normal_inverse_False_alpha_0.8_bs_64_metaratio_0.2_epoch_33_acc_0.6344086021505376.pth
```



部分代码copy from：

https://github.com/GeWu-Lab/OGM-GE_CVPR2022
