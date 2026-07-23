# Voice-Cloner

Clone a voice from YouTube videos. Point this tool at one or more YouTube clips, and it downloads, trims, and trains a text-to-speech voice model you can use offline.

---

## Overview

You provide a CSV file listing YouTube URLs and timestamps. The pipeline:

1. Downloads each clip from YouTube
2. Trims it to the timestamps you specified
3. Transcribes the audio
4. Trains a [Piper](https://github.com/OHF-Voice/piper1-gpl) voice model
5. Exports a `.onnx` voice file you can load into any Piper-compatible TTS app

---

## One-Time Setup

Do these steps once before you use the tool for the first time.

### Step 1 — Install system packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build ffmpeg espeak-ng
```

### Step 2 — Install Python dependencies for this project

Run this from inside the Voice-Cloner folder:

```bash
pip install -r requirements.txt
```

### Step 3 — Install the Piper training tool (outside this folder)

Piper is a separate training engine. Install it in your home directory so it does not clutter this project:

```bash
cd ~
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[train]'
./build_monotonic_align.sh
python3 setup.py build_ext --inplace
```

> After this, `~/piper1-gpl` holds the Piper engine. You do not need to touch it again.

### Step 4 — (Recommended) Download a checkpoint

A checkpoint is a pre-trained starting point that makes training faster and produces better quality voices. Without one, training from scratch takes much longer.

- Browse checkpoints: https://huggingface.co/datasets/rhasspy/piper-checkpoints
- Download a `medium` checkpoint for your language (e.g. `en_US-medium.ckpt`) and save it somewhere easy to find, such as `~/piper-checkpoints/`.

---

## Prepare Your Input CSV

Edit [audio-samples-in/sample.csv](audio-samples-in/sample.csv) with your YouTube links.

**Column reference:**

| Column | What to put |
|---|---|
| `voice_name` | A short name for the voice you are training (e.g. `alice`) |
| `youtube_url` | Full YouTube URL |
| `start_timestamp` | Where to start the clip — format `HH:MM:SS` |
| `end_timestamp` | Where to end the clip — format `HH:MM:SS` |

You can add as many rows as you want. Rows that share the same `voice_name` are combined into one voice model.

Both comma-delimited and tab-delimited CSV files are supported.

---

## Run

### Try a dry run first (no downloads, no training)

This checks your CSV and prints what the pipeline would do, without doing anything:

```bash
./train_voice.sh --input-csv audio-samples-in/sample.csv --dry-run
```

### Run training

```bash
./train_voice.sh \
  --input-csv audio-samples-in/sample.csv \
  --checkpoint-path ~/piper-checkpoints/en_US-medium.ckpt
```

That is the minimum you need. The pipeline uses sensible defaults for everything else.

### All options

| Option | Default | Description |
|---|---|---|
| `--input-csv` | *(required)* | Path to your CSV file |
| `--checkpoint-path` | *(none)* | Path to a Piper checkpoint — strongly recommended |
| `--trusted-checkpoint` | *(off)* | Use `--checkpoint-path` as a trusted warmstart checkpoint for compatibility with older Piper checkpoints |
| `--espeak-voice` | `en-us` | eSpeak voice code for your language |
| `--language-code` | `en` | Language code used in output filenames |
| `--sample-rate` | `22050` | Audio sample rate in Hz |
| `--batch-size` | `32` | Training batch size — lower this if you run out of memory |
| `--max-epochs` | `100` | Maximum training epochs — use `-1` for unlimited |
| `--checkpoint-every-n-epochs` | `10` | Save an extra periodic checkpoint every N epochs (set `0` to disable) |
| `--checkpoint-keep-last-n` | `5` | Number of periodic checkpoints to retain (older periodic checkpoints are pruned by the pipeline) |
| `--trainer-accelerator` | `cpu` | `cpu` or `gpu` |
| `--trainer-devices` | `1` | Number of GPUs to use |
| `--voice-name` | *(latest checkpoint voice)* | Voice name used by `--export-only`; if omitted, exports the most recently updated checkpoint across voices |
| `--export-only` | *(off)* | Skip download/transcribe/train and export ONNX from latest checkpoint |
| `--dry-run` | *(off)* | Preview actions without running them |

---

## GPU Usage

If your machine has an NVIDIA GPU, training will be significantly faster. First check that the driver is working:

```bash
nvidia-smi
```

If a GPU table appears, add these flags to your run command:

```bash
./train_voice.sh \
  --input-csv audio-samples-in/sample.csv \
  --checkpoint-path ~/piper-checkpoints/en_US-medium.ckpt \
  --trainer-accelerator gpu \
  --trainer-devices 1
```

To watch GPU utilization while training runs (in a second terminal):

```bash
watch -n 1 nvidia-smi
```

---

## Output

Finished voice files are written to the `voice-files-out/` folder inside this project. Nothing is written outside of this folder.

For each `voice_name` in your CSV:

```
voice-files-out/
  <voice_name>/
    exports/
      <language>-<voice_name>-medium.onnx        ← the voice model
      <language>-<voice_name>-medium.onnx.json   ← config for the model
    metadata.csv
    config.json
    train/
```

The two files in `exports/` are what you load into a Piper-compatible TTS application.

---

## Voice Tester

After you have at least one exported voice in `voice-files-out/<voice>/exports/`, run the tiny local tester:

```bash
./test_voice.sh
```

It opens a local browser page where you can pick a voice, type text, and press Play to hear the result.

The app auto-discovers any `*.onnx` file under `voice-files-out/*/exports/` and uses the matching `*.onnx.json` config when present.
