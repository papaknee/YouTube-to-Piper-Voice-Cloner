#!/usr/bin/env python3
"""YouTube to Piper training pipeline.

Input CSV schema:
voice_name,youtube_url,start_timestamp,end_timestamp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TIMESTAMP_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
CANONICAL_COLUMNS = ("voice_name", "youtube_url", "start_timestamp", "end_timestamp")
INPUT_COLUMN_ALIASES = {
    "voice_name": "voice_name",
    "youtube_url": "youtube_url",
    "yotuube_url": "youtube_url",
    "start_timestamp": "start_timestamp",
    "end_timestamp": "end_timestamp",
    "end_timstamp": "end_timestamp",
}


@dataclass
class ClipRequest:
    voice_name: str
    youtube_url: str
    start_seconds: int
    end_seconds: int


@dataclass
class ClipOutput:
    voice_name: str
    wav_path: Path
    transcript: str


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


class PipelineError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Piper voices from YouTube segments")
    parser.add_argument("--input-csv", default="audio-samples-in/sample.csv", help="Input CSV path")
    parser.add_argument("--espeak-voice", default="en-us", help="Piper espeak voice")
    parser.add_argument("--language-code", default="en", help="Language code for exported model name")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Piper training batch size")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs passed to Piper's trainer (use -1 for unlimited)")
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=10,
        help="Save an extra training checkpoint every N epochs (set 0 to disable periodic extra checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-keep-last-n",
        type=int,
        default=5,
        help="How many periodic checkpoints to keep when periodic checkpointing is enabled",
    )
    parser.add_argument("--checkpoint-path", default="", help="Optional Piper checkpoint path (recommended for faster convergence)")
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help=(
            "Treat --checkpoint-path as a trusted warmstart checkpoint for compatibility with older Piper checkpoints. "
            "Use only with trusted checkpoint sources."
        ),
    )
    parser.add_argument("--whisper-model", default="base", help="faster-whisper model name")
    parser.add_argument("--transcribe-device", default="auto", choices=["auto", "cpu", "cuda"], help="faster-whisper device")
    parser.add_argument(
        "--trainer-accelerator",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="PyTorch Lightning accelerator passed to piper.train",
    )
    parser.add_argument("--trainer-devices", default="1", help="PyTorch Lightning devices value passed to piper.train")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Maximum clip duration in seconds")
    parser.add_argument(
        "--auto-split-long-clips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically split clips longer than --split-max-seconds into multiple snippets",
    )
    parser.add_argument(
        "--split-min-seconds",
        type=float,
        default=10.0,
        help="Target minimum duration for auto-split clip parts",
    )
    parser.add_argument(
        "--split-max-seconds",
        type=float,
        default=30.0,
        help="Target maximum duration for auto-split clip parts",
    )
    parser.add_argument(
        "--split-tail-min-seconds",
        type=float,
        default=7.0,
        help="Allowed minimum tail duration when no perfect 10-30s segmentation is possible",
    )
    parser.add_argument("--root-dir", default=".", help="Repository root")
    parser.add_argument("--piper-dir", default="~/piper1-gpl", help="Path to the piper1-gpl install (contains .venv/)")
    parser.add_argument("--voice-name", default="", help="Voice name to export in --export-only mode (defaults to latest available voice checkpoint)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without running commands")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export from latest checkpoint (use --voice-name to target a specific voice)",
    )
    return parser.parse_args()


def resolve_command_path(command: str, root_dir: Path | None = None) -> str | None:
    if command.startswith("/"):
        return command

    if root_dir is not None:
        local_bin = root_dir / ".venv" / "bin" / command
        if local_bin.exists() and os.access(local_bin, os.X_OK):
            return str(local_bin)

    if sys.prefix and sys.prefix != sys.base_prefix:
        venv_bin = Path(sys.prefix) / "bin" / command
        if venv_bin.exists() and os.access(venv_bin, os.X_OK):
            return str(venv_bin)

    return shutil.which(command)


def ensure_command(command: str, root_dir: Path | None = None) -> str:
    resolved = resolve_command_path(command, root_dir)
    if resolved is None:
        raise PipelineError(f"Required command not found: {command}")
    return resolved


def parse_timestamp(raw_value: str) -> int:
    match = TIMESTAMP_PATTERN.match(raw_value)
    if not match:
        raise PipelineError(f"Invalid timestamp '{raw_value}'. Expected HH:MM:SS")

    hours, minutes, seconds = (int(value) for value in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise PipelineError(f"Invalid timestamp '{raw_value}'. Minutes/seconds must be < 60")

    return (hours * 3600) + (minutes * 60) + seconds


def parse_requests(csv_path: Path) -> List[ClipRequest]:
    if not csv_path.exists():
        raise PipelineError(f"Input CSV not found: {csv_path}")

    requests: List[ClipRequest] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        sample = file_handle.read(4096)
        file_handle.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        reader = csv.DictReader(file_handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise PipelineError("CSV file is empty")

        raw_fieldnames = [name.strip() for name in reader.fieldnames]
        normalized_fieldnames = [name.lower() for name in raw_fieldnames]

        mapped_columns: Dict[str, str] = {}
        for original, normalized in zip(raw_fieldnames, normalized_fieldnames):
            canonical = INPUT_COLUMN_ALIASES.get(normalized)
            if canonical is None:
                raise PipelineError(
                    "Unexpected CSV column '"
                    + original
                    + "'. Expected: voice_name,youtube_url,start_timestamp,end_timestamp"
                )
            if canonical in mapped_columns:
                raise PipelineError(f"Duplicate CSV meaning detected for column: {canonical}")
            mapped_columns[canonical] = original

        missing = [column for column in CANONICAL_COLUMNS if column not in mapped_columns]
        if missing:
            raise PipelineError(
                "Missing CSV columns: " + ",".join(missing)
            )

        for index, row in enumerate(reader, start=2):
            voice_name = (row.get(mapped_columns["voice_name"]) or "").strip()
            youtube_url = (row.get(mapped_columns["youtube_url"]) or "").strip()
            start_text = (row.get(mapped_columns["start_timestamp"]) or "").strip()
            end_text = (row.get(mapped_columns["end_timestamp"]) or "").strip()

            if not voice_name:
                raise PipelineError(f"Line {index}: voice_name is required")
            if not youtube_url:
                raise PipelineError(f"Line {index}: youtube_url is required")

            start_seconds = parse_timestamp(start_text)
            end_seconds = parse_timestamp(end_text)
            duration = end_seconds - start_seconds

            if duration <= 0:
                raise PipelineError(f"Line {index}: end_timestamp must be greater than start_timestamp")

            requests.append(
                ClipRequest(
                    voice_name=voice_name,
                    youtube_url=youtube_url,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )

    if not requests:
        raise PipelineError("Input CSV has no data rows")

    return requests


def run_command(
    cmd: Sequence[str],
    *,
    dry_run: bool,
    cwd: Path | None = None,
    log_file: Path | None = None,
    stream_output: bool = False,
) -> str:
    resolved_cmd = list(cmd)
    executable = resolved_cmd[0] if resolved_cmd else ""
    if executable and not executable.startswith("/") and "/" not in executable:
        resolved_executable = resolve_command_path(executable, cwd)
        if resolved_executable is not None:
            resolved_cmd[0] = resolved_executable

    printable = " ".join(resolved_cmd)
    print(f"$ {printable}")
    if dry_run:
        return ""

    if stream_output:
        # Let stdout/stderr flow directly to the terminal so the user can
        # monitor progress (e.g. training loss). Output is not captured for
        # the log file in this mode.
        if log_file:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n$ {printable}\n")
        result = subprocess.run(
            resolved_cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
        )
        if result.returncode != 0:
            raise PipelineError(f"Command failed ({result.returncode}): {printable}")
        return ""

    process = subprocess.run(
        resolved_cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )

    if log_file:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ {printable}\n")
            handle.write(process.stdout)
            handle.write(process.stderr)

    if process.returncode != 0:
        raise PipelineError(
            f"Command failed ({process.returncode}): {printable}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )

    return process.stdout.strip()


def download_audio(url: str, raw_dir: Path, dry_run: bool, log_file: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_template = raw_dir / "%(id)s.%(ext)s"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio",
        "--print",
        "after_move:filepath",
        "-o",
        str(output_template),
        url,
    ]
    output = run_command(cmd, dry_run=dry_run, log_file=log_file)
    if dry_run:
        return raw_dir / "dry-run-audio.m4a"

    audio_path = Path(output.splitlines()[-1].strip())
    if not audio_path.exists():
        raise PipelineError(f"yt-dlp did not produce an audio file for URL: {url}")
    return audio_path


def seconds_to_ffmpeg_time(value: int) -> str:
    hours = value // 3600
    minutes = (value % 3600) // 60
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def seconds_to_ffmpeg_time_precise(value: float) -> str:
    safe_value = max(0.0, value)
    whole_seconds = int(safe_value)
    milliseconds = int(round((safe_value - whole_seconds) * 1000))
    if milliseconds >= 1000:
        whole_seconds += 1
        milliseconds = 0

    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def clip_audio(
    source_audio: Path,
    output_wav: Path,
    start_seconds: int,
    end_seconds: int,
    sample_rate: int,
    dry_run: bool,
    log_file: Path,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        seconds_to_ffmpeg_time(start_seconds),
        "-to",
        seconds_to_ffmpeg_time(end_seconds),
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav),
    ]
    run_command(cmd, dry_run=dry_run, log_file=log_file)


def validate_clip_duration(wav_path: Path, min_duration: float, max_duration: float, dry_run: bool, log_file: Path) -> None:
    duration = probe_clip_duration(wav_path, dry_run=dry_run, log_file=log_file)
    if dry_run:
        return

    if duration < min_duration or duration > max_duration:
        raise PipelineError(
            f"Clip duration out of range for {wav_path.name}: {duration:.2f}s "
            f"(allowed {min_duration:.2f}-{max_duration:.2f}s)"
        )


def probe_clip_duration(wav_path: Path, dry_run: bool, log_file: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    output = run_command(cmd, dry_run=dry_run, log_file=log_file)
    if dry_run:
        return 0.0

    try:
        return float(output.strip())
    except ValueError as exc:
        raise PipelineError(f"Could not read duration for clip: {wav_path}") from exc


def sanitize_transcript(text: str) -> str:
    sanitized = " ".join(text.replace("|", " ").replace("\n", " ").split())
    return sanitized.strip()


def transcribe_clip(model, wav_path: Path) -> str:
    transcript, _ = transcribe_clip_with_segments(model, wav_path)
    return transcript


def transcribe_clip_with_segments(model, wav_path: Path) -> Tuple[str, List[TranscriptSegment]]:
    segments, _ = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
    parsed_segments: List[TranscriptSegment] = []
    text_fragments: List[str] = []

    for segment in segments:
        raw_text = (segment.text or "").strip()
        if not raw_text:
            continue

        safe_start = max(0.0, float(segment.start))
        safe_end = max(safe_start, float(segment.end))
        parsed_segments.append(
            TranscriptSegment(
                start_seconds=safe_start,
                end_seconds=safe_end,
                text=raw_text,
            )
        )
        text_fragments.append(raw_text)

    return sanitize_transcript(" ".join(text_fragments)), parsed_segments


def pick_split_point(
    candidates: List[float],
    lower_bound: float,
    upper_bound: float,
    fallback: float,
) -> float:
    bounded_candidates = [point for point in candidates if lower_bound <= point <= upper_bound]
    if not bounded_candidates:
        return fallback
    return max(bounded_candidates)


def plan_split_ranges(
    transcript_segments: List[TranscriptSegment],
    total_duration: float,
    min_seconds: float,
    max_seconds: float,
    tail_min_seconds: float,
) -> List[Tuple[float, float]]:
    if total_duration <= max_seconds:
        return [(0.0, total_duration)]

    candidate_boundaries = sorted(
        {
            max(0.0, min(total_duration, segment.end_seconds))
            for segment in transcript_segments
            if segment.end_seconds > 0.0 and segment.end_seconds < total_duration
        }
    )

    ranges: List[Tuple[float, float]] = []
    cursor = 0.0
    epsilon = 1e-6

    while total_duration - cursor > epsilon:
        remaining = total_duration - cursor
        if remaining <= max_seconds + epsilon:
            ranges.append((cursor, total_duration))
            break

        lower_bound = cursor + min_seconds
        upper_bound = min(cursor + max_seconds, total_duration)
        chosen = pick_split_point(
            candidate_boundaries,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fallback=upper_bound,
        )
        if chosen <= cursor + epsilon:
            chosen = upper_bound

        ranges.append((cursor, chosen))
        cursor = chosen

    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        last_duration = last_end - last_start
        if last_duration < min_seconds:
            prev_start, prev_end = ranges[-2]
            prev_duration = prev_end - prev_start

            if prev_duration + last_duration <= max_seconds:
                ranges[-2] = (prev_start, last_end)
                ranges.pop()
            elif last_duration < tail_min_seconds:
                target_boundary = max(
                    prev_start + min_seconds,
                    last_end - min_seconds,
                )
                shifted_boundary = pick_split_point(
                    candidate_boundaries,
                    lower_bound=prev_start + min_seconds,
                    upper_bound=min(prev_start + max_seconds, last_end - min_seconds),
                    fallback=target_boundary,
                )
                shifted_boundary = max(prev_start + min_seconds, shifted_boundary)
                shifted_boundary = min(last_end - min_seconds, shifted_boundary)
                if shifted_boundary > prev_start and shifted_boundary < last_end:
                    ranges[-2] = (prev_start, shifted_boundary)
                    ranges[-1] = (shifted_boundary, last_end)

    normalized: List[Tuple[float, float]] = []
    for start, end in ranges:
        safe_start = max(0.0, min(total_duration, start))
        safe_end = max(safe_start, min(total_duration, end))
        if safe_end - safe_start > epsilon:
            normalized.append((safe_start, safe_end))

    if not normalized:
        return [(0.0, total_duration)]
    return normalized


def split_clip_by_ranges(
    source_wav: Path,
    ranges: List[Tuple[float, float]],
    sample_rate: int,
    dry_run: bool,
    log_file: Path,
) -> List[Path]:
    if len(ranges) <= 1:
        return [source_wav]

    part_paths: List[Path] = []
    stem = source_wav.stem

    for index, (start_seconds, end_seconds) in enumerate(ranges, start=1):
        part_path = source_wav.with_name(f"{stem}_part{index:02d}.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            seconds_to_ffmpeg_time_precise(start_seconds),
            "-to",
            seconds_to_ffmpeg_time_precise(end_seconds),
            "-i",
            str(source_wav),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(part_path),
        ]
        run_command(cmd, dry_run=dry_run, log_file=log_file)
        part_paths.append(part_path)

    return part_paths


def find_latest_checkpoint(search_dir: Path) -> Path:
    checkpoints = sorted(search_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise PipelineError(f"No checkpoint found under: {search_dir}")
    return checkpoints[-1]


# Generator-architecture hyperparameters that must match between a checkpoint's
# saved weights and the model built for this run. Piper's Lightning CLI
# normally restores these automatically from --ckpt_path, but that recovery
# silently fails for older checkpoints containing hyperparameter keys the
# current CLI no longer accepts (e.g. "sample_bytes"), so the Trainer falls
# back to default ("medium") architecture and then fails to load the
# checkpoint's weights with a shape mismatch. Reading the checkpoint directly
# and passing the architecture explicitly avoids relying on that recovery.
CHECKPOINT_ARCHITECTURE_KEYS = (
    "resblock",
    "resblock_kernel_sizes",
    "resblock_dilation_sizes",
    "upsample_rates",
    "upsample_initial_channel",
    "upsample_kernel_sizes",
    "inter_channels",
    "hidden_channels",
    "filter_channels",
    "n_heads",
    "n_layers",
    "kernel_size",
    "p_dropout",
    "n_layers_q",
    "use_spectral_norm",
    "gin_channels",
    "use_sdp",
    "segment_size",
    "filter_length",
    "hop_length",
    "win_length",
    "mel_channels",
    "mel_fmin",
    "mel_fmax",
    "num_symbols",
)

# Keys that live under the `--data.*` CLI group instead of `--model.*`.
CHECKPOINT_DATA_GROUP_KEYS = {"num_symbols"}


def read_checkpoint_architecture(piper_python: Path, checkpoint_path: Path) -> Dict[str, object]:
    """Read generator-architecture hyperparameters saved inside a checkpoint.

    Runs in the Piper training environment (via piper_python) since torch is
    not installed in this pipeline's own environment.
    """
    snippet = (
        "import json, torch\n"
        f"ckpt = torch.load({str(checkpoint_path)!r}, map_location='cpu', weights_only=False)\n"
        "hp = ckpt.get('hyper_parameters', {})\n"
        f"keys = {CHECKPOINT_ARCHITECTURE_KEYS!r}\n"
        "print(json.dumps({k: hp[k] for k in keys if k in hp}))\n"
    )
    try:
        result = subprocess.run(
            [str(piper_python), "-c", snippet],
            check=True,
            text=True,
            capture_output=True,
        )
        return json.loads(result.stdout.strip() or "{}")
    except Exception as exc:
        print(
            f"Warning: could not read architecture hyperparameters from checkpoint {checkpoint_path} ({exc}). "
            "Falling back to CLI-restored/default architecture."
        )
        return {}


def checkpoint_architecture_cli_args(hparams: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in hparams.items():
        flag = f"--data.{key}" if key in CHECKPOINT_DATA_GROUP_KEYS else f"--model.{key}"
        if isinstance(value, bool):
            args.extend([flag, "true" if value else "false"])
        elif value is None:
            args.extend([flag, "null"])
        elif isinstance(value, str):
            # Passed directly (no shell involved): avoid json.dumps here, which
            # would wrap the value in literal quote characters.
            args.extend([flag, value])
        else:
            args.extend([flag, json.dumps(value)])
    return args


def find_latest_voice_checkpoint(root_dir: Path) -> tuple[str, Path]:
    voice_root = root_dir / "voice-files-out"
    candidates: List[tuple[str, Path]] = []

    if voice_root.exists():
        for voice_dir in voice_root.iterdir():
            if not voice_dir.is_dir() or voice_dir.name == "logs":
                continue
            train_root = voice_dir / "train"
            if not train_root.exists():
                continue
            for checkpoint in train_root.rglob("*.ckpt"):
                candidates.append((voice_dir.name, checkpoint))

    if not candidates:
        raise PipelineError(f"No checkpoint found under: {voice_root}")

    voice_name, checkpoint_path = max(candidates, key=lambda item: item[1].stat().st_mtime)
    return voice_name, checkpoint_path


def save_checkpoint_copy(checkpoint_path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination)
    return destination


def build_metadata_file(outputs: Iterable[ClipOutput], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    for item in outputs:
        if not item.transcript:
            raise PipelineError(f"Empty transcript for clip: {item.wav_path.name}")
        rows.append(f"{item.wav_path.name}|{item.transcript}")

    metadata_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def create_periodic_checkpoint_config(args: argparse.Namespace, root_dir: Path) -> Path | None:
    if args.checkpoint_every_n_epochs <= 0:
        return None

    config_text = (
        "trainer:\n"
        "  callbacks:\n"
        "    - class_path: lightning.pytorch.callbacks.ModelCheckpoint\n"
        "      init_args:\n"
        "        monitor: null\n"
        "        save_top_k: -1\n"
        "        save_last: true\n"
        "        auto_insert_metric_name: false\n"
        f"        every_n_epochs: {args.checkpoint_every_n_epochs}\n"
        "        filename: \"periodic-epoch={epoch}-step={step}\"\n"
    )

    logs_dir = root_dir / "voice-files-out" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="periodic-checkpoint-",
        suffix=".yaml",
        dir=str(logs_dir),
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(config_text)
        return Path(handle.name)


def prune_periodic_checkpoints(train_root: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return

    periodic_files = sorted(
        train_root.rglob("periodic-*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if len(periodic_files) <= keep_last_n:
        return

    for path in periodic_files[:-keep_last_n]:
        path.unlink(missing_ok=True)


def train_and_export_voice(
    voice_name: str,
    clips: List[ClipOutput],
    args: argparse.Namespace,
    root_dir: Path,
    log_file: Path,
    checkpoint_override: Path | None = None,
) -> None:
    voice_out_dir = root_dir / "voice-files-out" / voice_name
    voice_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = voice_out_dir / "metadata.csv"
    config_path = voice_out_dir / "config.json"
    train_root = voice_out_dir / "train"
    export_dir = voice_out_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    piper_python = Path(args.piper_dir).expanduser() / ".venv" / "bin" / "python"
    if not piper_python.exists():
        raise PipelineError(
            f"Piper python not found at {piper_python}. "
            "Install piper1-gpl per the README or pass --piper-dir."
        )

    if not args.export_only:
        if not clips:
            raise PipelineError(f"No clips available for voice: {voice_name}")

        clips_dir = clips[0].wav_path.parent
        build_metadata_file(clips, metadata_path)

        train_cmd = [
            str(piper_python),
            "-m",
            "piper.train",
            "fit",
            "--data.voice_name",
            voice_name,
            "--data.csv_path",
            str(metadata_path),
            "--data.audio_dir",
            str(clips_dir),
            "--model.sample_rate",
            str(args.sample_rate),
            "--data.espeak_voice",
            args.espeak_voice,
            "--data.cache_dir",
            str(voice_out_dir / "cache"),
            "--data.config_path",
            str(config_path),
            "--data.batch_size",
            str(min(args.batch_size, len(clips))),
            "--data.num_test_examples",
            "0",
            "--trainer.default_root_dir",
            str(train_root),
            "--trainer.accelerator",
            args.trainer_accelerator,
            "--trainer.devices",
            str(args.trainer_devices),
            "--trainer.max_epochs",
            str(args.max_epochs),
        ]

        periodic_config_path = create_periodic_checkpoint_config(args, root_dir)
        if periodic_config_path is not None:
            train_cmd.extend(["-c", str(periodic_config_path)])

        if args.checkpoint_path:
            checkpoint_hparams = read_checkpoint_architecture(piper_python, Path(args.checkpoint_path))
            if checkpoint_hparams:
                print(
                    "Applying generator architecture from checkpoint hyperparameters: "
                    f"{checkpoint_hparams}"
                )
                train_cmd.extend(checkpoint_architecture_cli_args(checkpoint_hparams))

            if args.trusted_checkpoint:
                # Older checkpoints can fail CLI hyperparameter parsing when used as
                # ckpt_path. Warmstart loads model weights and skips resume parsing.
                train_cmd.extend(["--model.warmstart_ckpt", args.checkpoint_path])
                print(
                    "Warning: trusted checkpoint mode enabled. "
                    "Using Piper warmstart checkpoint loading for compatibility with older checkpoints. "
                    "Use only for trusted checkpoint sources."
                )
            else:
                train_cmd.extend(["--ckpt_path", args.checkpoint_path])
        else:
            if args.trusted_checkpoint:
                print("Warning: --trusted-checkpoint was set, but --checkpoint-path is empty. Flag will be ignored.")
            print("Warning: --checkpoint-path not provided. Piper recommends fine-tuning from an existing checkpoint.")

        run_command(
            train_cmd,
            dry_run=args.dry_run,
            cwd=root_dir,
            log_file=log_file,
            stream_output=True,
        )

        if not args.dry_run and args.checkpoint_every_n_epochs > 0:
            prune_periodic_checkpoints(train_root, args.checkpoint_keep_last_n)
    else:
        print("Export-only mode: skipping training and reusing the latest checkpoint from the existing run.")

    if args.dry_run:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("/tmp/dry-run.ckpt")
    elif checkpoint_override is not None:
        checkpoint_path = checkpoint_override
    else:
        checkpoint_path = find_latest_checkpoint(train_root)
        stable_checkpoint_path = save_checkpoint_copy(checkpoint_path, voice_out_dir / "latest-checkpoint.ckpt")
        checkpoint_path = stable_checkpoint_path

    model_name = f"{args.language_code}-{voice_name}-medium.onnx"
    model_path = export_dir / model_name
    export_cmd = [
        str(piper_python),
        str(root_dir / "scripts" / "piper_export_compat.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--output-file",
        str(model_path),
    ]
    run_command(export_cmd, dry_run=args.dry_run, cwd=root_dir, log_file=log_file)

    if not args.dry_run:
        if not config_path.exists():
            raise PipelineError(f"Expected config file not found: {config_path}")
        target_config = model_path.with_suffix(".onnx.json")
        shutil.copy2(config_path, target_config)
        if not model_path.exists() or not target_config.exists():
            raise PipelineError(f"Export artifacts missing for voice: {voice_name}")


def group_by_voice(requests: Iterable[ClipRequest]) -> Dict[str, List[ClipRequest]]:
    by_voice: Dict[str, List[ClipRequest]] = defaultdict(list)
    for item in requests:
        by_voice[item.voice_name].append(item)
    return by_voice


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()

    if args.split_min_seconds <= 0 or args.split_max_seconds <= 0:
        raise SystemExit("--split-min-seconds and --split-max-seconds must be positive")
    if args.split_min_seconds > args.split_max_seconds:
        raise SystemExit("--split-min-seconds must be less than or equal to --split-max-seconds")
    if args.split_tail_min_seconds <= 0:
        raise SystemExit("--split-tail-min-seconds must be positive")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = root_dir / "voice-files-out" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline-{timestamp}.log"

    try:
        if not args.export_only:
            ensure_command("yt-dlp", root_dir)
            ensure_command("ffmpeg", root_dir)
            ensure_command("ffprobe", root_dir)

        if args.trainer_accelerator == "gpu" and shutil.which("nvidia-smi") is None:
            raise PipelineError(
                "--trainer-accelerator gpu was requested, but nvidia-smi was not found. "
                "Install NVIDIA drivers/CUDA or run with --trainer-accelerator cpu."
            )

        print(
            "Training device configuration: "
            f"accelerator={args.trainer_accelerator}, devices={args.trainer_devices}, "
            f"whisper_device={args.transcribe_device}"
        )

        if args.export_only:
            if args.voice_name:
                voice_name = args.voice_name.strip()
                train_root = root_dir / "voice-files-out" / voice_name / "train"
                checkpoint_path = find_latest_checkpoint(train_root)
            else:
                voice_name, checkpoint_path = find_latest_voice_checkpoint(root_dir)

            print(f"Export-only mode: exporting voice '{voice_name}' from checkpoint {checkpoint_path}")
            stable_checkpoint_path = save_checkpoint_copy(
                checkpoint_path,
                root_dir / "voice-files-out" / voice_name / "latest-checkpoint.ckpt",
            )
            train_and_export_voice(
                voice_name,
                [],
                args,
                root_dir,
                log_file,
                checkpoint_override=stable_checkpoint_path,
            )
        else:
            requests = parse_requests((root_dir / args.input_csv).resolve())
            grouped = group_by_voice(requests)

            model = None
            if not args.dry_run:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise PipelineError("Missing dependency: faster-whisper. Install with pip install -r requirements.txt") from exc

                model = WhisperModel(args.whisper_model, device=args.transcribe_device)

            download_cache: Dict[str, Path] = {}
            clips_by_voice: Dict[str, List[ClipOutput]] = defaultdict(list)

            for voice_name, voice_rows in grouped.items():
                clip_dir = root_dir / "audio-samples-in" / "clips" / voice_name
                for index, request in enumerate(voice_rows, start=1):
                    if request.youtube_url not in download_cache:
                        download_cache[request.youtube_url] = download_audio(
                            request.youtube_url,
                            root_dir / "audio-samples-in" / "raw-youtube",
                            args.dry_run,
                            log_file,
                        )

                    source_audio = download_cache[request.youtube_url]
                    clip_name = f"{voice_name}_{index:04d}.wav"
                    wav_path = clip_dir / clip_name
                    clip_audio(
                        source_audio,
                        wav_path,
                        request.start_seconds,
                        request.end_seconds,
                        args.sample_rate,
                        args.dry_run,
                        log_file,
                    )
                    clip_duration = probe_clip_duration(wav_path, args.dry_run, log_file)
                    split_paths: List[Path] = [wav_path]
                    first_pass_transcript = ""

                    if (
                        args.auto_split_long_clips
                        and not args.dry_run
                        and clip_duration > args.split_max_seconds
                    ):
                        first_pass_transcript, initial_segments = transcribe_clip_with_segments(model, wav_path)
                        planned_ranges = plan_split_ranges(
                            initial_segments,
                            clip_duration,
                            min_seconds=args.split_min_seconds,
                            max_seconds=args.split_max_seconds,
                            tail_min_seconds=args.split_tail_min_seconds,
                        )
                        split_paths = split_clip_by_ranges(
                            wav_path,
                            planned_ranges,
                            args.sample_rate,
                            args.dry_run,
                            log_file,
                        )

                    for split_index, split_path in enumerate(split_paths, start=1):
                        validate_clip_duration(
                            split_path,
                            args.min_duration,
                            args.max_duration,
                            args.dry_run,
                            log_file,
                        )

                        transcript = "dry run transcript"
                        if not args.dry_run:
                            if len(split_paths) == 1 and first_pass_transcript:
                                transcript = first_pass_transcript
                            else:
                                transcript = transcribe_clip(model, split_path)

                        transcript = sanitize_transcript(transcript)
                        if not transcript:
                            part_label = ""
                            if len(split_paths) > 1:
                                part_label = f" (part {split_index})"
                            raise PipelineError(f"Transcript is empty for {split_path.name}{part_label}")

                        clips_by_voice[voice_name].append(
                            ClipOutput(voice_name=voice_name, wav_path=split_path, transcript=transcript)
                        )

            for voice_name, clips in clips_by_voice.items():
                train_and_export_voice(voice_name, clips, args, root_dir, log_file)

        print("Pipeline completed successfully.")
        print(f"Log file: {log_file}")
        return 0
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Log file: {log_file}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
