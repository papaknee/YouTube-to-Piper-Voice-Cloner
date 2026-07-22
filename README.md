# Voice-Cloner YouTube Training Pipeline

This repository now supports training Piper voices from a YouTube-only CSV input.

## Input CSV

Create `youtube-inputs.csv` with this exact header:

```csv
voice_name,youtube_url,start_timestamp,end_timestamp
```

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
  --input-csv youtube-inputs.csv \
  --espeak-voice en-us \
  --language-code en \
  --sample-rate 22050 \
  --batch-size 32 \
  --checkpoint-path /path/to/piper-medium.ckpt
```

### Dry run

```bash
./train_voice.sh --input-csv youtube-inputs.csv --dry-run
```

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
