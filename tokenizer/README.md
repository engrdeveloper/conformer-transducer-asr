# Tokenizer

This directory prepares Common Voice-style datasets and trains a SentencePiece
BPE tokenizer for the ASR model.

The sampling utilities can be used to create smaller training subsets before
training the tokenizer. Development and test splits are kept unchanged.

## Expected dataset structure

The input directory must contain a Common Voice-style dataset:

```text
/path/to/common-voice/language/
├── train.tsv
├── dev.tsv
├── test.tsv
└── clips/
    └── audio files...
```

## First run: prepare the data and train the tokenizer

Run the command from this `tokenizer` directory:

```bash
python train.py hparams/tokenizer.yaml \
  --data_folder=/path/to/common-voice/language \
  --language=en \
  --output_folder=trained/en \
  --skip_prep=False
```

This performs two steps:

1. Reads the Common Voice TSV files and creates SpeechBrain manifests in
   `trained/en/save/`:

   ```text
   train.csv
   dev.csv
   test.csv
   ```

2. Reads the training manifest and trains the SentencePiece BPE tokenizer in
   `trained/en/`.

The default vocabulary size is 1,000 tokens. It can be changed with:

```bash
python train.py hparams/tokenizer.yaml \
  --data_folder=/path/to/common-voice/language \
  --language=en \
  --output_folder=trained/en \
  --token_output=2000
```

## Create a smaller training subset

Two utilities are provided:

- `utils/random_sampling.py` randomly selects utterances.
- `utils/speaker_based_sampling.py` approximately preserves the original
  speaker distribution.

In the selected script, set `TOKENIZER_PATH` to the prepared dataset directory
and choose the target number of training hours:

```python
TARGET_TRAIN_HOURS = 100
TOKENIZER_PATH = "/path/to/tokenizer/trained/en"
```

The source directory must contain:

```text
/path/to/tokenizer/trained/en/save/
├── train.csv
├── dev.csv
└── test.csv
```

Run the desired utility:

```bash
python utils/random_sampling.py
```

or:

```bash
python utils/speaker_based_sampling.py
```

For a 100-hour target, the output is written to:

```text
/path/to/tokenizer/trained/en_100h_random/save/
```

or:

```text
/path/to/tokenizer/trained/en_100h_spkdist/save/
```

Only the training manifest is sampled. The development and test manifests are
copied unchanged. The script stops with an error if `TOKENIZER_PATH` does not
exist.

## What `skip_prep` means

Data preparation and tokenizer training are separate operations.

- `skip_prep=False` allows the program to generate `train.csv`, `dev.csv`, and
  `test.csv` from the original TSV files.
- `skip_prep=True` completely skips TSV-to-CSV preparation and immediately
  trains the tokenizer using the CSV files configured by `train_csv`,
  `valid_csv`, and `test_csv`.

Use `skip_prep=True` when the manifests already exist, for example after
creating a random or speaker-balanced training subset:

```bash
python train.py hparams/tokenizer.yaml \
  --language=en_100h_random \
  --output_folder=trained/en_100h_random \
  --skip_prep=True
```

Before using `True`, verify that the following files exist:

```text
trained/en_100h_random/save/train.csv
trained/en_100h_random/save/dev.csv
trained/en_100h_random/save/test.csv
```

If these files are missing, tokenizer training will fail because it has no
prepared transcript manifest to read.

When `skip_prep=False`, preparation is also skipped automatically if all three
CSV files already exist. To regenerate them after changing the source data,
splits, paths, or normalization settings, remove or relocate the old manifests
first and run again with `skip_prep=False`.
