# 11-785 Project: Towards Interpretable Speech Enhancement with Acoustic-phonetic Alignment

## Data Preparation

We arrange all data as follows.
```
- CaPAAP
  - data
    - acoustics
    - ph_logits_aligned
- PAAP
  - data
    - DNS-Challenge
```

## Training for Acoustic-Phonetic Prediction

```sh
cd CaPAAP
# train a FCN
python train.py --exp conv
# train a CapsNet
python train.py --exp caps
```

## Fine-tuning for Speech Enhancement

```sh
cd PAAP
# fine-tune FullSubNet with LR weights for PAAP loss
python train.py finetune=fullsubnet
# fine-tune FullSubNet with FCN logits for PAAP loss
python train.py finetune=fullsubnet-convnet
# fine-tune FullSubNet with CapsNet logits for PAAP loss
python train.py finetune=fullsubnet-capsnet-full
# fine-tune FullSubNet with CapsNet coupling coefficients for PAAP loss
python train.py finetune=fullsubnet-capsnet-weight
```

The Evaluation is done during fine-tuning.