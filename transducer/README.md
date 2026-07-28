# Conformer Transducer

This directory trains and evaluates a SpeechBrain-based Conformer Transducer
for Common Voice-style datasets.

## Required inputs

Run the tokenizer workflow first. The model expects:

```text
tokenizer/trained/<language>/
├── *.model
├── *.vocab
└── save/
    ├── train.csv
    ├── dev.csv
    └── test.csv
```

The CSV manifests use the following columns:

```text
ID,duration,wav,spk_id,wrd
```

The original Common Voice audio directory must remain available because the
manifests reference its audio files.

## Run training

Run from this `transducer` directory:

```bash
python train.py hparams/conformer_transducer.yaml \
  --data_folder=/path/to/common-voice/en \
  --tokenizer_path=/absolute/path/to/tokenizer/trained/en \
  --language=en \
  --output_folder=results/en/online/conformer_transducer
```

The YAML currently contains example `/workdir/...` paths. Either edit them or
override `data_folder`, `tokenizer_path`, and `output_folder` as shown above.

The script automatically restores compatible checkpoints from `save/`, trains
the model, averages the best checkpoints, and evaluates the configured test
manifest.



## Troubleshooting the Numba RNN-T loss

When `use_torchaudio: False`, the RNN-T loss uses Numba. If training reports
`RuntimeError: Missing libdevice file`, first check whether `libdevice` was
installed in the active virtual environment:

```bash
find "$VIRTUAL_ENV" -path "*cuda_nvcc*" -name "libdevice*.bc"
```

If nothing is printed, install the requirements again:

```bash
python -m pip install -r requirements.txt
```

If the file exists, point Numba to its CUDA Toolkit directory:

```bash
export CUDA_HOME="$(python -c 'import site; print(site.getsitepackages()[0] + "/nvidia/cuda_nvcc")')"
```

Run training again from the same terminal so that it retains `CUDA_HOME`.

## Offline and streaming modes

For offline training:

```bash
python train.py hparams/conformer_transducer.yaml \
  --data_folder=/path/to/common-voice/en \
  --tokenizer_path=/absolute/path/to/tokenizer/trained/en \
  --output_folder=results/en/offline/conformer_transducer \
  --streaming=False \
  --mode=offline
```

For streaming training:

```bash
python train.py hparams/conformer_transducer.yaml \
  --data_folder=/path/to/common-voice/en \
  --tokenizer_path=/absolute/path/to/tokenizer/trained/en \
  --output_folder=results/en/online/conformer_transducer \
  --streaming=True \
  --mode=online
```

When streaming is enabled, dynamic chunk training samples chunk sizes and
available left context during training.

Fixed chunk size and left context are commented out in the default YAML. With
no fixed `test_config`, evaluation uses full context.

To test a particular streaming configuration, uncomment and edit these lines
in `hparams/conformer_transducer.yaml`:

```yaml
test_chunk_size: 16
test_left_context: 8

dynchunktrain_config_sampler:
   # Keep the existing training options above this block.
   test_config: !new:speechbrain.utils.dynamic_chunk_training.DynChunkTrainConfig
      chunk_size: !ref <test_chunk_size>
      left_context_size: !ref <test_left_context>
```

One encoder frame corresponds to roughly 40 ms of audio after the convolutional
frontend. A chunk size of 16 therefore represents roughly 640 ms.

Change the two values to evaluate other latency and context combinations. Use
the same `output_folder` to load the existing checkpoint rather than starting a
new experiment.

## Run testing

Pass `--test` with the language code to evaluate an existing experiment:

```bash
python train.py hparams/conformer_transducer.yaml \
  --test \
  --language=tr
```

Replace `tr` with the language to evaluate and use the same configuration and
output folder as its saved checkpoint.

## Attention-free experiments

The standard Conformer uses self-attention:

```bash
--use_attention=True
```

To remove self-attention while retaining the feed-forward and convolution
modules:

```bash
--use_attention=False \
--model_name=attentionfree_conformer_transducer
```

## Outputs

The configured `output_folder` contains:

```text
output_folder/
├── hyperparams.yaml
├── save/                         # Checkpoints
├── train_log.txt
├── train_tensor_log/             # TensorBoard events
└── test_wer_results_with_cs_lc/  # Detailed WER reports
```

View training curves with:

```bash
tensorboard --logdir /path/to/output_folder/train_tensor_log
```

## Multi-GPU training

SpeechBrain distributed training can be launched with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py hparams/conformer_transducer.yaml \
  --data_folder=/path/to/common-voice/en \
  --tokenizer_path=/absolute/path/to/tokenizer/trained/en \
  --output_folder=results/en/online/conformer_transducer \
  --distributed_launch \
  --find_unused_parameters
```
