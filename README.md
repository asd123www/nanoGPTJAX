# Lab 1: LLM Training Profiling on a Single-TPU


## Lab1 Overview

In this lab, we will get familiar with large lauguage model training and the infrastructure that we will use in upcoming projects. In particular, we will profile the training process of a GPT-2 style language model on a single accelerator, focusing on compute characteristic and memory footprint. Understanding the memory and compute characteristic of LLM training can help you understand the key challenges in scaling LLM training, better motivate the techniques we will discuss in the class and also the following labs.

The model we will use is a gpt-2 style transformer model, that encompasses a range of key components in modern advanced LLms, such as transformer block with self-attention, 

about different restaurants, reviews, and making
reservations


## Lab 1 Prep Work

We need access to a TPU server to run our training script. We will use Colab, a free cloud-based Jupyter Notebook environment provided by Google. It serves as an website interface where we can write and execute Python on different runtime backend, including servers with TPU acclerators.

We will start from the same Colab environment and install the dependencies on top of that. Follow the instructions below to build the environment and run our training script.

1. Change runtime type. Open [Google Colab](https://colab.research.google.com/) and create a new notebook. Click the dropdown arrow next to `Connect`, select "Change runtime type", use Python3, v5e-1 TPU, and 2025.07 runtime version.

2. Check your TPU-info. Click "Terminal" in lower left corner, type `tpu-info`, make sure you have 1 TPU v5e chip.

3. Setup Lab 1 environment. Download our repo via `git clone https://github.com/asd123www/nanoGPTJAX.git`. In the root directory, run `python3.11 -m pip install -r requirements.txt` to install the dependencies.
   ```
   cd nanogpt
   python3.11 download_fineweb_tokens.py

   pip install tensorboard tensorboard-plugin-profile
   ```
4. Verify the training process. Run training process with `python3.11 nanogpt/train.py --config configs/project1-part2.yaml`. You should be able to see the model config printed and the training process run to complete.
5. Add notebook script.
   ```
   %cd /content/nanoGPTJAX
   %run nanogpt/train.py --config configs/project1-part2.yaml
   ```
   ```
   %load_ext tensorboard
   %tensorboard --logdir profiles/ --port 6006
   ```

## The GPT-2 style model training

At a high level, this codebase is organized around a single training entrypoint, `nanogpt/train.py`, which wires together the data pipeline, model definition, optimizer, checkpointing, and profiling. Most of the core logic lives in the `nanogpt/` package: `fineweb_dataloader.py` loads token shards, `model.py` defines the GPT-style network, `optim.py` builds the optimizer, `config.py` turns YAML files into structured runtime configs, and helper files such as `checkpoint_utils.py` and `utils.py` handle state restoration, parameter initialization, sharding, and logging.

### DataLoader

We train on pretokenized FineWeb shards stored as binary `.bin` files. `download_fineweb_tokens.py` downloads these cached GPT-2 token files from Hugging Face so the lab can skip the expensive tokenization step and start training immediately.

The loading logic lives in `fineweb_dataloader.py`. `LoadShardTokens` reads each shard into memory, validates the file header, and extracts the token array together with the positions of the beginning-of-sequence token. `BOSFinder` then builds efficient `(start, end)` ranges for each batch so the training loop can pack contiguous sequences of length `seqlen + 1` without re-scanning the entire shard every step. In `train.py`, these ranges are assembled into input tokens `x` and next-token labels `y`, then copied to the TPU with the expected batch sharding.

### Transformer Model

The GPT-style model is defined in `model.py`. It begins with a token embedding layer, applies a stack of repeated Transformer blocks, and ends with a linear language-model head that projects hidden states back to the vocabulary for next-token prediction.

Each Transformer block contains grouped-query causal self-attention and an MLP, both wrapped with RMSNorm and residual connections. The attention path supports two implementations: a Pallas flash-attention kernel for faster fused execution and an XLA attention path for easier inspection and profiling. The model also applies rotary positional embeddings (RoPE), supports activation checkpointing to reduce memory usage, and uses a final RMSNorm plus soft logit capping before the loss is computed.

### Adam Optimizer

The optimizer is built in `optim.py` using Optax. Instead of assigning one identical update rule to every parameter, the code uses `optax.multi_transform` to give the token embedding, output projection, and all remaining parameters different learning-rate schedules. This makes the training setup closer to modern LLM recipes, where embeddings and unembeddings are often tuned more conservatively than the rest of the network.

In the training script, the optimizer is wrapped with global gradient clipping and can optionally use gradient accumulation through `optax.MultiSteps` when the desired global batch size is larger than the per-step TPU batch size. For the main model weights, the optimizer combines AdamW with an additional cautious weight-decay rule that only decays matrix-shaped parameters when the update direction and parameter direction agree.

### Config files

Experiment settings are stored in YAML files under `configs/`, such as `configs/project1-part2.yaml`. These files specify the model shape, sequence length, attention implementation, batch sizes, learning-rate schedule, checkpoint frequency, profiling window, and dataset directory.

`config.py` parses each YAML file into typed dataclasses (`ModelConfig`, `HyperParams`, `CheckpointConfig`, and `ProfileConfig`). This gives the training script a structured configuration object instead of a raw dictionary, while still keeping experiments easy to modify. In practice, this means you can switch model size, enable or disable activation checkpointing, change optimizer settings, or turn profiling on and off by editing a config file rather than modifying training code.

## Assignment(100 points)


### Part1(20 points)

### Part2(60 points)

### Part3(20 points)

## Submission


