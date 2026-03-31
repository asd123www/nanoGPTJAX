# Lab 1: LLM Training Profiling on a Single-TPU

## Lab1 Overview

In this lab, we will become familiar with large language model (LLM) training and the infrastructure we will use in the upcoming labs. In particular, we will profile the training process of a GPT-2 style language model on a single accelerator, with the focus on its compute characteristics and memory footprint. Understanding the compute and memory behavior of LLM training will help you understand the key challenges of scaling LLMs, and will motivate the systems techniques we discuss in class as well as in later labs. Lab 1 also covers some core concepts and prepares you for the following labs.

We will build our training pipeline in JAX, a Python library that supports automatic differentiation for machine learning and provides efficient scaling through just-in-time (JIT) compilation. To improve training throughput, we will use a TPU accelerator, an application-specific integrated circuit (ASIC) developed by Google for efficient neural network computation. The model we study is a GPT-2-style model, which includes many of the key components used in modern LLMs, such as Transformer blocks with self-attention.


## Lab 1 Prep Work

We need access to a TPU server to run our training script. We will use Colab, a free cloud-based Jupyter Notebook environment provided by Google. It serves as an website interface where we can write and execute Python on different runtime backend, including servers with TPU acclerators.

We will start from the same Colab environment and install the dependencies on top of that. Follow the instructions below to build the environment and run our training script.

1. Change runtime type. Open [Google Colab](https://colab.research.google.com/) and create a new notebook. Click the dropdown arrow next to `Connect`, select "Change runtime type", use `Python3, v5e-1 TPU, and 2025.07` runtime version.

2. Check your TPU-info. Click "Terminal" in lower left corner, type `tpu-info`, make sure you have 1 TPU v5e acclerator.

3. Setup Lab 1 environment. Download our repo via `git clone https://github.com/asd123www/nanoGPTJAX.git`. In the root directory, run `python3.11 -m pip install -r requirements.txt` to install the dependencies. Also download the dataset and profile tools.
   ```
   cd nanogpt
   python3.11 download_fineweb_tokens.py

   pip install tensorboard tensorboard-plugin-profile
   ```
4. Verify the training process. Run training process with `python3.11 nanogpt/train.py --config configs/project1-part2.yaml`. You should be able to see the model config printed and the training process run to complete.
5. Add Colab notebook script. This allows you to visualize the profiling result in the notebook.
   ```
   %cd /content/nanoGPTJAX
   %run nanogpt/train.py --config configs/project1-part2.yaml
   ```
   ```
   %load_ext tensorboard
   %tensorboard --logdir profiles/ --port 6006
   ```

## The GPT-2 model training

At a high level, this codebase is organized around a single training entrypoint, `nanogpt/train.py`, which wires together the data pipeline, model definition, optimizer, checkpointing, and profiling. Most of the core logic lives in the `nanogpt/` directory: `fineweb_dataloader.py` loads token shards, `model.py` defines the GPT2-style model, `optim.py` builds the optimizer, `config.py` turns config YAML files in `configs/` into runtime configs, and `checkpoint_utils.py` handles state checkpoint.

### DataLoading and Tokenization

We train on the pretokenized FineWeb dataset. `download_fineweb_tokens.py` downloads these cached GPT-2 token files from Hugging Face so we can skip the tokenization step and start training immediately.

**Token** is the unit of text processing in LLM. Tokens allow the model to process language as discrete elements rather than raw texts. The conversion between text and tokens is handled by the **tokenizer**, which defines a mapping from text to token IDs and back. Specifically, the tokenizer encodes raw text into a sequence of tokens before training, and decodes output tokens back into text. Tokenization is essential for both LLM capability and efficiency. GPT-2 adopts Byte Pair Encoding (BPE), a subword-based tokenization method [BPE impl]{https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py}. 

The loading logic lives in `fineweb_dataloader.py`. `LoadShardTokens` reads each shard into CPU memory, validates the file header, and extracts the token array together with the positions of the beginning-of-sequence token. `BOSFinder` then builds efficient `(start, end)` ranges for each batch so the training loop can pack contiguous sequences of length `seqlen + 1` without re-scanning the entire shard every step. In `train.py`, these ranges are assembled into input tokens `x` and next-token labels `y`, then loaded to the TPU.

### Transformer Model

The GPT2 model is defined in `model.py`. It begins with a token embedding layer, then applies a sequence of Transformer blocks, and ends with a language-model head that projects hidden states back to the vocabulary for next-token prediction.

Each Transformer block contains multi-head-query causal self-attention and an MLP, both wrapped with RMSNorm and residual connections. Check this [slides](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf) for the attention and transformers. The attention path supports two implementations: a native XLA attention path and a Pallas flash-attention kernel (`nanogpt/pallas/flash_attention.py`) for fused execution. The model supports [activation checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html) to reduce activation memory usage.

### Optimizer

An optimizer is the algorithm that updates the model weights during training. After the forward pass computes the loss and the backward pass computes gradients, the optimizer uses those gradients to decide how each weight should change in order to reduce the loss.

In this lab, we use AdamW, a widely adopted optimizer for training neural networks. The optimizer is built in `optim.py` using Optax. AdamW maintains additional state for each parameter, mainly the first moment and second moment, which track moving averages of the gradients and squared gradients. In the training script, the optimizer is also combined with global gradient clipping and can optionally support gradient accumulation through `optax.MultiSteps` when the target global batch size exceeds the per-step TPU batch size.

### Config files

Experiment settings are stored in YAML files under `configs/`, such as `configs/project1-part2.yaml`. These files specify the model shape, sequence length, attention implementation, batch sizes, learning-rate schedule, etc. In this Lab, we will focus on `model:` fields.

`config.py` parses each YAML file into typed dataclasses (`ModelConfig`, `HyperParams`, `CheckpointConfig`, and `ProfileConfig`). This gives the training script a structured configuration object, while still keeping experiments easy to modify. In practice, this means you can switch model size, enable or disable activation checkpointing, change attention backend, or turn profiling on and off by editing a config file rather than modifying training code.

## Assignment(100 points)


### Part1(20 points)

### Part2(60 points)

### Part3(20 points)



## Submission


