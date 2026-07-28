# P2G Ambiguity in Streaming ASR

This repository contains the code accompanying the paper **A Cross-Linguistic
Study of Context and the Accuracy–Latency Trade-off in Streaming ASR**, which
examines how phoneme-to-grapheme (P2G) ambiguity and available acoustic context
affect streaming ASR across languages.

Phoneme-to-grapheme ambiguity is the uncertainty involved in mapping speech
sounds to written forms. This relationship differs across languages: some have
relatively direct sound-to-spelling mappings, while others allow the same sound
to be written in several ways.

The experiments use Conformer Transducer models. The code supports offline and
streaming recognition, configurable chunk size and left context, and
attention-based or attention-free encoders.

## Repository structure

```text
.
├── tokenizer/
│   ├── common_voice_prepare.py  # Creates dataset manifests
│   ├── train.py                 # Trains the BPE tokenizer
│   ├── hparams/                 # Tokenizer configuration
│   └── utils/                   # Random and speaker-based sampling
├── transducer/
│   ├── train.py                 # Trains and evaluates the ASR model
│   ├── hparams/                 # Experiment and streaming configuration
│   └── models/                  # Encoder and transducer components
└── requirements.txt             # Shared Python dependencies
```

The `tokenizer` directory prepares Common Voice manifests, optionally creates
fixed-duration random or speaker-balanced subsets, and trains the SentencePiece
BPE tokenizer.

The `transducer` directory contains the Conformer Transducer model,
hyperparameters, training pipeline, streaming configuration, checkpointing, and
WER evaluation.

The expected workflow moves from data preparation and tokenizer training to
ASR model training and, finally, offline or streaming evaluation. Each
directory has its own README with the settings and commands for that stage.

## Environment setup

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for model training.

## Next steps

Follow the workflows in this order:

1. [Prepare the data and train the tokenizer](tokenizer/README.md).
2. [Train and evaluate the Conformer Transducer](transducer/README.md).

The folder-specific READMEs contain the commands and configuration options for
each stage.
