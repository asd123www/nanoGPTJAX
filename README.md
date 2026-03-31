# Project 1: LLM Training Profiling on a Single-TPU

**Turnin:** Online via Canvas  
**Teams:** Individual or teams of 2  
**Due:** TBD






## Project Overview

In this project, we will use a small GPT2-style training codebase to study how LLM training works on a single accelerator. Using JAX as the deep learning library and TPU as the accelerator, we will profile LLM training in Colab, understand the memory and latency characteristic.

Before start you will setup the Colab environment and clone the codebase for running the experiment.For part 1, you will study the gpt-2 model code in JAX and anslysis the model structure and computation. For part 2, you will run the training script profiling compute and memory traces, answering the related questions. For part 3, you will breakdown the attention and mlp kernel latency with varying sequence length.








## Background

### LLM Training

Large language model (LLM) have become a transformative technology in everyday application. 

### JAX

### TPU

### Colab








## Setup

How are we going to setup the environment in Colab?



Requires Python 3.11+.

```bash
python3.11 -m pip install -r requirements.txt
```

## Prepare Data

Download the FineWeb10B tokenized dataset before training:

```bash
cd nanogpt
python3.11 download_fineweb_tokens.py
```

## Training

```bash
python3.11 nanogpt/train.py --config configs/small.yaml
```

Edit `configs/small.yaml` to change model size, hyperparameters, checkpoint paths, and data directory.

## Profiling

```bash
pip install tensorboard tensorboard-plugin-profile
~/.local/bin/tensorboard --logdir profiles/ --port 6006
```






## Assignment

## Part 1: Understand LLM Training (20 points)

This part is write-up only. No experiments are required.

### Tasks

1. Use the `model` section of `configs/project1.yaml` to explain how the architecture is defined.
2. For the model in `configs/project1.yaml`, list the shapes and dtypes of the major model parameters.
3. Explain the shapes of the inputs, hidden states, and logits during a forward pass for a batch of token IDs.
4. Explain the computation performed in one transformer layer, including attention, the MLP, residual connections, and the layer output.
5. Briefly explain what additional work is required during backward propagation beyond the forward pass.

### Deliverables

- A short write-up in `report.pdf`
- Your Part 1 answer should be clear enough that a reader unfamiliar with the repo could reconstruct the model's major tensor shapes

### Grading

- 8 points: correct interpretation of the config and architecture
- 6 points: correct tensor shapes and dtypes for major weights and outputs
- 6 points: clear explanation of transformer-layer computation

## Part 2: Memory Profiling of LLM Training (60 points)

## Part 2.1: Memory Accounting (10 points)

Before running experiments, categorize memory into persistent model states and residual states.

### Tasks

1. Identify the persistent model states.
2. Identify the residual states and temporary buffers that appear during training.
3. For each category, list representative shapes and dtypes.
4. Include at least one attention-related temporary buffer and one MLP-related temporary buffer.
5. You may summarize repeated tensors by template and multiplicity rather than exhaustively listing every single layer instance.

### Deliverables

- A table or structured list in `report.pdf` separating model states from residual states
- A short paragraph explaining which states scale with model size, which scale with sequence length, and which scale with both

### Grading

- 4 points: correct model-state accounting
- 4 points: correct residual-state accounting
- 2 points: clear discussion of scaling behavior

## Part 2.2: Baseline Profiling (20 points)

The baseline configuration for this part is:

- Attention implementation: `xla`
- Activation checkpointing: disabled

### Tasks

1. Set `model.attn_impl: xla` in the config.
2. Disable activation checkpointing by removing or commenting out the `@jax.checkpoint` decorator on `_checkpointed_block_forward` in `nanogpt/model.py`.
3. Run training long enough to capture one profiler window.
4. Open the TensorBoard profile and inspect both the trace view and the memory view.
5. Explain the main memory spikes and major trends across one training step.
6. Relate the observed memory behavior to specific computations such as attention, MLPs, activations kept for backward, and temporary buffers.

### Deliverables

- One screenshot of the memory view
- One screenshot of the trace view
- A short analysis in `report.pdf`
- A note describing exactly how you disabled activation checkpointing

### Grading

- 8 points: correct experiment setup and profiler capture
- 6 points: accurate explanation of memory spikes and trends
- 6 points: correct mapping from profiler events to model computations

## Part 2.3: Activation Checkpointing (15 points)

For this part, keep `attn_impl: xla` and re-enable activation checkpointing.


## Learning Objectives

After completing this project, students should be able to:

- Explain how a GPT-style model is defined by a configuration file.
- Infer parameter shapes, activation shapes, and output shapes from model code.
- Describe the computation performed by one transformer layer.
- Distinguish persistent model states from residual and temporary memory.
- Use JAX and TensorBoard profiling tools to interpret traces and memory timelines.
- Compare baseline attention, activation checkpointing, and flash attention.
- Reason about how sequence length affects training latency and memory footprint.


### Tasks

1. Re-enable the `@jax.checkpoint` decorator on `_checkpointed_block_forward` in `nanogpt/model.py`.
2. Rerun the same experiment as in Part 2.2.
3. Compare the memory footprint with the baseline.
4. Compare the step latency with the baseline.
5. Explain why activation checkpointing changes memory usage and latency.

### Deliverables

- One screenshot of the memory view
- One screenshot of the trace view or step summary
- A comparison paragraph against Part 2.2
- A short explanation of recomputation and why it changes the tradeoff

### Grading

- 5 points: correct experiment setup
- 5 points: accurate memory comparison
- 5 points: accurate latency and recomputation explanation

## Part 2.4: Flash Attention (15 points)

For this part, keep activation checkpointing enabled and switch the attention implementation to `flash_attn`.

### Tasks

1. Set `model.attn_impl: flash_attn` in the config.
2. Rerun the profiling experiment.
3. Compare the result against the checkpointed XLA-attention run from Part 2.3.
4. Explain how flash attention changes the memory footprint and why.
5. If your environment does not support the flash-attention path, document the failure clearly and discuss the expected effect using the code structure and any instructor-provided reference trace.

### Deliverables

- One screenshot of the memory view
- One screenshot of the trace view
- A comparison paragraph against Part 2.3
- A short explanation of why flash attention changes the attention-memory pattern

### Grading

- 5 points: correct experiment setup
- 5 points: accurate analysis of memory differences
- 5 points: clear explanation of the underlying systems reason

## Part 3: Latency Profiling and Sequence Length (20 points)

In this part, you will study how sequence length affects training latency on a single device.

### Required Setting

- Use `flash_attn`
- Keep activation checkpointing enabled
- Keep all model dimensions fixed except for `seqlen`
- If memory requires it, you may reduce batch size, but you must report any such change clearly

### Tasks

1. Run the model at several sequence lengths. Recommended values: `256`, `512`, `1024`, and `2048`.
2. For each run, measure or estimate the latency of the following four quantities:
   - flash-attention forward
   - flash-attention backward
   - MLP forward
   - MLP backward
3. Create a plot with sequence length on the x-axis and latency on the y-axis.
4. Explain the trend you observe.
5. Explain the implications of your results for long-context LLM training.

### Deliverables

- One plot containing the required latency curves
- A short paragraph explaining the trend
- A short paragraph explaining the practical implication for long-context training

### Grading

- 8 points: correct and readable plot
- 6 points: correct discussion of latency scaling
- 6 points: thoughtful discussion of implications for long-context training

## Submission

Submit a single archive containing the following:

- `report.pdf` with all write-ups, tables, screenshots, and comparisons
- Any config files you changed or duplicated for your experiments
- A short `changes.txt` or appendix section listing any code changes you made, even if the change was only toggling one decorator
- Your final latency plot as a standalone image file in addition to the copy inside `report.pdf`

## Suggested Report Organization

To make grading easier, organize your report as follows:

- Part 1
- Part 2.1
- Part 2.2
- Part 2.3
- Part 2.4
- Part 3
- Appendix: exact config changes and any one-line code toggles

## Hints

- The model definition is controlled primarily by the `model` section of `configs/project1.yaml`.
- The training loop starts and stops profiler tracing in `nanogpt/train.py`.
- Activation checkpointing is currently implemented in `nanogpt/model.py`.
- Attention implementation is selected by `model.attn_impl`.
- For memory accounting, it is often easier to describe one transformer block carefully and then multiply by `num_layers`.
- For latency analysis, you may aggregate multiple low-level launches if the trace viewer splits one logical operation into several kernels. If you do so, explain your aggregation rule clearly.
