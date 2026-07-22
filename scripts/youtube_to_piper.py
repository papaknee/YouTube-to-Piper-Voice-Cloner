#!/usr/bin/env python3
"""YouTube to Piper training pipeline.

Input CSV schema:
voice_name,youtube_url,start_timestamp,end_timestamp
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


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


class PipelineError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Piper voices from YouTube segments")
    parser.add_argument("--input-csv", default="youtube-inputs.csv", help="Input CSV path")
    parser.add_argument("--espeak-voice", default="en-us", help="Piper espeak voice")
    parser.add_argument("--language-code", default="en", help="Language code for exported model name")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Piper training batch size")
    parser.add_argument("--checkpoint-path", default="", help="Optional Piper checkpoint path")
    parser.add_argument("--whisper-model", default="base", help="faster-whisper model name")
    parser.add_argument("--transcribe-device", default="auto", choices=["auto", "cpu", "cuda"], help="faster-whisper device")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Maximum clip duration in seconds")
    parser.add_argument("--root-dir", default=".", help="Repository root")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without running commands")
    return parser.parse_args()


def ensure_command(command: str) -> None:
    if shutil.which(command) is None:
        raise PipelineError(f"Required command not found: {command}")


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
        reader = csv.DictReader(file_handle)
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


def run_command(cmd: Sequence[str], *, dry_run: bool, cwd: Path | None = None, log_file: Path | None = None) -> str:
    printable = " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return ""

    process = subprocess.run(
        cmd,
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
        return

    try:
        duration = float(output.strip())
    except ValueError as exc:
        raise PipelineError(f"Could not read duration for clip: {wav_path}") from exc

    if duration < min_duration or duration > max_duration:
        raise PipelineError(
            f"Clip duration out of range for {wav_path.name}: {duration:.2f}s "
            f"(allowed {min_duration:.2f}-{max_duration:.2f}s)"
        )


def sanitize_transcript(text: str) -> str:
    sanitized = " ".join(text.replace("|", " ").replace("\n", " ").split())
    return sanitized.strip()


def transcribe_clip(model, wav_path: Path) -> str:
    segments, _ = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return sanitize_transcript(text)


def find_latest_checkpoint(search_dir: Path) -> Path:
    checkpoints = sorted(search_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise PipelineError(f"No checkpoint found under: {search_dir}")
    return checkpoints[-1]


def build_metadata_file(outputs: Iterable[ClipOutput], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    for item in outputs:
        if not item.transcript:
            raise PipelineError(f"Empty transcript for clip: {item.wav_path.name}")
        rows.append(f"{item.wav_path.name}|{item.transcript}")

    metadata_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def train_and_export_voice(
    voice_name: str,
    clips: List[ClipOutput],
    args: argparse.Namespace,
    root_dir: Path,
    log_file: Path,
) -> None:
    voice_out_dir = root_dir / "voice-files-out" / voice_name
    voice_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = voice_out_dir / "metadata.csv"
    config_path = voice_out_dir / "config.json"
    train_root = voice_out_dir / "train"
    export_dir = voice_out_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    clips_dir = clips[0].wav_path.parent
    build_metadata_file(clips, metadata_path)

    train_cmd = [
        sys.executable,
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
        str(args.batch_size),
        "--trainer.default_root_dir",
        str(train_root),
    ]

    if args.checkpoint_path:
        train_cmd.extend(["--ckpt_path", args.checkpoint_path])
    else:
        print("Warning: --checkpoint-path not provided. Piper recommends fine-tuning from an existing checkpoint.")

    run_command(train_cmd, dry_run=args.dry_run, cwd=root_dir, log_file=log_file)

    if args.dry_run:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("/tmp/dry-run.ckpt")
    else:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else find_latest_checkpoint(train_root)

    model_name = f"{args.language_code}-{voice_name}-medium.onnx"
    model_path = export_dir / model_name
    export_cmd = [
        sys.executable,
        "-m",
        "piper.train.export_onnx",
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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = root_dir / "voice-files-out" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline-{timestamp}.log"

    try:
        ensure_command("yt-dlp")
        ensure_command("ffmpeg")
        ensure_command("ffprobe")

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
                validate_clip_duration(
                    wav_path,
                    args.min_duration,
                    args.max_duration,
                    args.dry_run,
                    log_file,
                )

                transcript = "dry run transcript"
                if not args.dry_run:
                    transcript = transcribe_clip(model, wav_path)
                transcript = sanitize_transcript(transcript)
                if not transcript:
                    raise PipelineError(f"Transcript is empty for {wav_path.name}")

                clips_by_voice[voice_name].append(
                    ClipOutput(voice_name=voice_name, wav_path=wav_path, transcript=transcript)
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
