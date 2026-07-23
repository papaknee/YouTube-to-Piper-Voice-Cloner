#!/usr/bin/env python3
"""YouTube to Piper training pipeline.

Input CSV schema:
voice_name,youtube_url,start_timestamp,end_timestamp
"""

from __future__ import annotations

import argparse
import csv
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
