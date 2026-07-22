# Voice-Cloner YouTube Training Pipeline

This repository now supports training Piper voices from a YouTube-only CSV input.

## Input CSV

Use [audio-samples-in/sample.csv](audio-samples-in/sample.csv) as the example input file.

Header:

```csv
voice_name,youtube_url,start_timestamp,end_timestamp
```

Delimiter:

- Comma-delimited CSV is supported.
- Tab-delimited files are also supported (your current sample file format).

- `voice_name`: name of the output voice/model group.
- `youtube_url`: full YouTube URL.
- `start_timestamp`: clip start in `HH:MM:SS`.
- `end_timestamp`: clip end in `HH:MM:SS`.

You can include multiple rows and multiple `voice_name` values. The pipeline trains one voice per `voice_name`.

## Prerequisites

Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build ffmpeg espeak-ng
```

Install Python dependencies in your environment:

```bash
pip install -r requirements.txt
```

Install Piper training code as described in the upstream guide:

```bash
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[train]'
./build_monotonic_align.sh
python3 setup.py build_ext --inplace
```

## Run

```bash
chmod +x train_voice.sh
./train_voice.sh \
  --input-csv audio-samples-in/sample.csv \
  --espeak-voice en-us \
  --language-code en \
  --sample-rate 22050 \
  --batch-size 32 \
  --checkpoint-path /path/to/piper-medium.ckpt \
  --trainer-accelerator gpu \
  --trainer-devices 1
```

### Dry run

```bash
./train_voice.sh --input-csv audio-samples-in/sample.csv --dry-run
```

## Checkpoints

Using `--checkpoint-path` is strongly recommended by the Piper training guide because it usually:

- Converges faster than training from scratch.
- Produces better voice quality with less data.
- Reduces the risk of unstable training early in the run.

Where to find checkpoints:

- Piper checkpoint dataset: https://huggingface.co/datasets/rhasspy/piper-checkpoints
- Choose a `medium` checkpoint unless you plan to tune model settings for other sizes.

Example:

```bash
./train_voice.sh \
  --input-csv audio-samples-in/sample.csv \
  --checkpoint-path /models/en_US-medium.ckpt
```

## GPU Checks And Usage

Check whether your machine has an NVIDIA GPU and driver:

```bash
nvidia-smi
```

If this shows a GPU table, CUDA driver is available. If command is missing/fails, use CPU mode or install drivers.

Run training on GPU explicitly:

```bash
./train_voice.sh \
  --input-csv audio-samples-in/sample.csv \
  --trainer-accelerator gpu \
  --trainer-devices 1
```

How to confirm GPU is being used:

- The pipeline prints selected device config at startup.
- `piper.train` command in logs includes `--trainer.accelerator gpu`.
- During training, watch GPU utilization in another terminal:

```bash
watch -n 1 nvidia-smi
```

If GPU utilization stays at 0% for the full training run, verify your PyTorch/CUDA environment and retry.

## Output

For each `voice_name`, the pipeline writes files under:

- `voice-files-out/<voice_name>/metadata.csv`
- `voice-files-out/<voice_name>/config.json`
- `voice-files-out/<voice_name>/train/...`
- `voice-files-out/<voice_name>/exports/<language>-<voice_name>-medium.onnx`
- `voice-files-out/<voice_name>/exports/<language>-<voice_name>-medium.onnx.json`

## Notes

- `--checkpoint-path` is optional, but strongly recommended by Piper for faster convergence.
- Timestamps must be valid and `end_timestamp` must be greater than `start_timestamp`.
- Clips are normalized to mono WAV at the selected sample rate.
