# Sentiment Analysis with LLMs

This repository contains some demo code on how to build a sentiment analysis application using (self-hosted) LLMs, 
in particular, using these two open source models:

- **M1)** [Qwen2.5 1.5B, instruction-tuned](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), and
- **M2)** [Qwen2.5 500M, instruction-tuned](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

For more information on the Qwen models, click [here](https://qwenlm.github.io/blog/qwen2.5/).

As dataset, we use the [IMDB review dataset](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) from HuggingFace, of which we randownly sample 500 reviews (50% positive, 50% negative) from its **test set**.

## Overview

In this demonstration, we use PyTorch and HuggingFace to implement various methods for classifying reviews, without having to train M1 or M2:

1. Zero-shot inference (with greedy decoding).
2. Log-likelihood queries (with greedy decoding).
3. In-context learning a.k.a. k-shot learning.
4. Sampling with optimized $\text{top}_k$ and temperature $\tau$ parameters.  

Method 2. is derived from the original GPT-3 paper, wherein a technique that emulates a traditional softmax classification layer using LLMs is described.

Since this is classification task, we use traditional classification performance metrics to compare methods. 
For each experiment, we measure precision, recall, F1-score and accuracy. We also measure throughput (i.e. number of words processed per second by each model).

In general, **we can observe that method 2. performs best**, followed by method 3.

For a **detailed** description of the experiments (with numbers, links to the GPT-3 paper, detailed comments and plots), please refer to the notebook under `notebook/analysis_notebook.ipynb`.

## Installation and Execution

These demo was implemeted on Python 3.10.12, on a MacBook Air M2 machine, and tested addtionally on a Tesla T4 running on CUDA 12.2. We assume the availability of Anaconda, concretely, that you create
a virtual environment, and install the dependencies as follows
```bash
conda create -n sent_analysis_w_quen python=3.10.12
conda activate sent_analysis_w_quen
conda install -c conda-forge jupyterlab
pip install -f requirements.txt
```

The demo runs on three different architectures: a) CUDA devices / NViDIA GPUs, b) MPS devices / Apple Silicon GPUs and c) x86 devices / Intel CPUs. We recommend using
either a) or b) with ~10-16GB of GPU RAM in order to execute with sufficient speed. It is implemeted using HuggingFace and PyTorch libraries, which has the advantage of allowing for the use of multiple inference methods (or as people call them, decoding algorithms) alongside the usual sampling decoding customary for LLMs. 

To run the notebook, type
```bash
cd notebook
jupyter lab
```
and... you'll be ready to roll!

For readibility the functions used are also delivered as a (procedural) Python library (see `sentiment_code/sentiment_analysis.py`).
Additonally, a CLI script, demonstrating how to reproduce on CLI the zero-shot experiment can be executed via:
```bash
python sentiment_code/main.py
```
It will display the confusion matrixes, and write the scores to CSV files in `results/`

The models and their BPE tokenizers will be downloaded from the HuggingFace hub automatically.

## Notes

- We **randonmly sample** 500 reviews from the IMDB dataset. Some performance figures may thus vary across repeated runs, but the ranking of the models and models should not very much.
- The Qwen models have a large input context window of 32,768 (model) tokens, and hence IMDB reviews don't need to
be truncated.
- In the notebook we include some profiling figures. Such numbers will vay based on the host you use, but the proportions across models and inference methods should still hold.

## To Do

- Implement logging
- Implement exception handling
- Add OOP where useful

It will display confusion matrixes for models M1 and M2 above, and save the precision, recall, F1-score and accuracy figures on the `results` directory.