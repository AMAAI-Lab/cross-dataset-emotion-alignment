# Leveraging LLM Embeddings for Cross Dataset Label Alignment and Zero Shot Music Emotion Prediction
<div align="center">
<a href="https://arxiv.org/abs/XXXX.XXXX">Paper</a>,
<a href="https://huggingface.co/datasets/amaai-lab/cross-dataset-emotion-splits">Dataset Splits</a>
<br/><br/>
  
[![Hugging Face Dataset Splits](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/amaai-lab/cross-dataset-emotion-splits) [![arXiv](https://img.shields.io/badge/arXiv-2406.02255-brightgreen.svg)](https://arxiv.org/abs/XXXX.XXXX)
</div>


In this repository, we present the code used to leverage Large Language Model (LLM) embeddings for music emotion recognition across multiple datasets. The approach involves computing LLM embeddings for emotion labels, clustering similar labels across datasets, and mapping music features (MERT) to the LLM embedding space. We also introduce alignment regularization to better dissociate different clusters, improving generalization to unseen datasets for zero shot classification.

