# nanoGPTJAX

A GPT-2-style language model built in pure JAX, designed to run on TPUs and GPUs.

## Setup

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

## Inference

```bash
python3.11 nanogpt/inference.py --config configs/small.yaml
```

This starts an interactive prompt where you type text and the model generates a completion. Make sure the checkpoint path in your config file points to a valid trained checkpoint.

## Acknowledgements

This project is based on [nanoGPTJAX](https://github.com/AakashKumarNain/nanoGPTJAX) by Aakash Kumar Nain. Thanks to the original project for providing a clean, from-scratch JAX implementation of GPT training and inference.

## License

Apache-2.0
