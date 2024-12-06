# Sentiment Analysis with LLMs

This repository contains some demo code on how to build a sentiment analysis application using (self-hosted) LLMs, in particular, using these two Alibaba open source models:

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

Method 2. is derived from the [original GPT-3 paper](https://arxiv.org/abs/2005.14165), wherein a technique that emulates a traditional softmax classification layer using LLMs is described.

Since this is classification task, we use traditional classification performance metrics to compare methods. 
For each experiment, we measure precision, recall, F1-score and accuracy. We also measure throughput (i.e. number of words processed per second by each model).

In general, **we can observe that method 2. performs best**, followed by method 3.

For a **detailed** description of the experiments (with numbers, links to the GPT-3 paper, detailed comments and plots), please refer to the notebook under `notebook/analysis_notebook.ipynb`.

## Installation and Execution

These demo was implemeted on Python 3.10.12, on a MacBook Air M2 machine, and tested addtionally on a Tesla T4 running on CUDA 12.2. We **assume you have Python and Anaconda** installed on your host, and running a **Bash terminal** session. To recreate the environment used, type (terminal):
```bash
conda create -n sent_analysis_w_quen python=3.10.12
conda activate sent_analysis_w_quen
pip install -r requirements.txt
```

The demo runs on three different architectures: a) CUDA devices / NViDIA GPUs, b) MPS devices / Apple Silicon GPUs and c) x86 devices / Intel CPUs. We recommend using either a) or b) with ~10-16GB of GPU RAM in order to execute with sufficient speed. It is implemeted using the HuggingFace and PyTorch libraries, which has the advantage of allowing for the use of multiple inference methods (or as people call them, decoding algorithms) alongside the usual sampling decoding customary for LLMs. 

For readibility the functions used are also delivered as a (procedural) Python library (see `sentiment_code/sentiment_analysis.py`). Additonally, a CLI script, demonstrating how to reproduce on CLI each experiment can be executed via (terminal):
```bash
python sentiment_code/run_<exp_name>.py
```
It will run the experiment and save all results in `results/` (CSV files for the performance scores, PNG files
for the confusion matrixes, and a log file with inference latencies and other information). Each experiment takes
between 15-30 mins on a MacBook Air M2.

To run the notebook, you'll need additionally to install JupyterLab and link a kernel to `sent_analysis_w_quen` (terminal):
```bash
pip install ipykernel
python -m ipykernel install --name sent_analysis_w_quen --user
pip install jupyterlab
cd notebook
jupyter lab
```
Please note that it includes somewhat simpler versions of the functions used.

The models and their BPE tokenizers will be in all cases downloaded from the HuggingFace hub automatically.

## Notes

- We **randonmly sample** 500 reviews from the IMDB dataset. Some performance figures may thus vary across repeated runs, but the ranking of the models and models should not very much.
- The Qwen models have a large input context window of 32,768 (model) tokens, and hence IMDB reviews don't need to
be truncated.
- In the notebook and logs we include some profiling figures. Such numbers will vay based on the host you use, but the proportions across models and inference methods should still hold.

## To Do

- Implement exception handling
- Add OOP where useful