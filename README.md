# Leveraging LLM Embeddings for Cross-Dataset Label Alignment and Zero-Shot Music Emotion Prediction

<div align="center">
<a href="https://arxiv.org/pdf/2410.11522v1">Paper</a> |
<a href="https://huggingface.co/datasets/amaai-lab/cross-dataset-emotion-splits">Dataset Splits</a>
<br/><br/>

[![Hugging Face Dataset Splits](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/amaai-lab/cross-dataset-emotion-splits) [![arXiv](https://img.shields.io/badge/arXiv-2406.02255-brightgreen.svg)](https://arxiv.org/abs/XXXX.XXXX)

</div>

This repository contains the implementation of our novel approach to music emotion recognition across multiple datasets using Large Language Model (LLM) embeddings. Our method involves:

1. Computing LLM embeddings for emotion labels
2. Clustering similar labels across datasets
3. Mapping music features (MERT) to the LLM embedding space
4. Introducing alignment regularization to improve cluster dissociation and enhance generalization to unseen datasets for zero-shot classification

## Table of Contents

- [Leveraging LLM Embeddings for Cross-Dataset Label Alignment and Zero-Shot Music Emotion Prediction](#leveraging-llm-embeddings-for-cross-dataset-label-alignment-and-zero-shot-music-emotion-prediction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
    - [Segment-Level Results](#segment-level-results)
    - [Song-Level Results (Majority Voting)](#song-level-results-majority-voting)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AMAAI-Lab/cross-dataset-emotion-alignment.git
   cd cross-dataset-emotion-alignment
   ```

2. Set up a Conda environment:

   ```bash
   conda env create -f environment.yaml
   conda activate cross-dataset-emotion-alignment
   ```

## Data Preparation

1. Option A: Download the MTG-Jamendo, CAL500, and Emotify datasets and place them in the `data/` directory.

   Option B: Use the preprocessed data from [Hugging Face](https://huggingface.co/datasets/amaai-lab/cross-dataset-emotion-splits).

2. If using Option A, preprocess the data and compute MERT features:

   ```bash
   python src/data/preprocess.py --audio_duration 10
   ```

## Training

Configure your training with various options:

1. Baseline dataset training:

   ```bash
   # Single dataset with label augmentation
   python src/train.py data.combine_train_datasets=["mtg"] version=your_version +experiment=label_aug

   # Multi-dataset without label augmentation
   python src/train.py data.combine_train_datasets=["mtg", "CAL500"] version=your_version
   ```

2. Label clustering (includes label augmentation by default):

   ```bash
   python src/train.py +cluster=mtg_CAL500 data.train_combine_mode=max_size_cycle
   ```

3. Alignment regularization:

   ```bash
   python src/train.py +cluster=reg_mtg_CAL500 model.regularization_alpha=2.5 data.train_combine_mode=max_size_cycle
   ```

Available options:

- Datasets: `mtg`, `CAL500`, `emotify`
- Clusters: `mtg_CAL500`, `mtg_emotify`, `CAL500_emotify`
- Regularizations: `reg_mtg_CAL500`, `reg_mtg_emotify`, `reg_CAL500_emotify`

## Evaluation

To evaluate a trained model, replace `train.py` with `eval.py` in the command and specify the checkpoint path. You can also use a wandb artifact to load the model by uncommenting the wandb-related configs in `conf/eval.yaml` and providing your wandb project, entity, and artifact name. By default, evaluation is performed on all three datasets. To evaluate on other datasets, modify `data/multi.yaml`.

Example commands:

1. Evaluate on a single dataset:

   ```bash
   python src/eval.py data.combine_train_datasets=["mtg"] ckpt_path=path/to/checkpoint.ckpt
   ```

2. Evaluate with clustering:

   ```bash
   python src/eval.py +cluster=mtg_CAL500 data.train_combine_mode=max_size_cycle ckpt_path=path/to/checkpoint.ckpt
   ```

3. Evaluate with alignment regularization:

   ```bash
   python src/eval.py +cluster=reg_mtg_CAL500 model.regularization_alpha=2.5 data.train_combine_mode=max_size_cycle ckpt_path=path/to/checkpoint.ckpt
   ```

## Results

We present our results for both segment-level and song-level predictions. Song-level results are obtained through majority voting of segment-level predictions.

### Segment-Level Results

| Phases                   | Train-MTG+CAL, Test-EMO | Train-CAL+EMO, Test-MTG | Train-EMO+MTG, Test-CAL |
| ------------------------ | ----------------------- | ----------------------- | ----------------------- |
| Baseline 1               | 0.324                   | 0.0215                  | 0.255                   |
| Baseline 2 (Clustering)  | 0.341                   | 0.0240                  | 0.262                   |
| Alignment Regularisation | **0.402** (λ = 2.5)     | **0.0248** (λ = 2.5)    | **0.262** (λ = 1)       |

### Song-Level Results (Majority Voting)

| Phases                   | Train-MTG+CAL, Test-EMO | Train-CAL+EMO, Test-MTG | Train-EMO+MTG, Test-CAL |
| ------------------------ | ----------------------- | ----------------------- | ----------------------- |
| Baseline 1               | 0.315                   | 0.0129                  | 0.252                   |
| Baseline 2 (Clustering)  | 0.346                   | 0.0175                  | **0.267**               |
| Alignment Regularisation | **0.400** (λ = 2.5)     | **0.0202** (λ = 2.5)    | 0.229 (λ = 1)           |

These results demonstrate the effectiveness of our approach, particularly the alignment regularization technique, in improving cross-dataset generalization for music emotion recognition tasks.

______________________________________________________________________
