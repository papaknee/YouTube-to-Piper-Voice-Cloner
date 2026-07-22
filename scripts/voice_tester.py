#!/usr/bin/env python3
"""Tiny local voice tester for exported Piper voices.

This scans voice-files-out/*/exports/*.onnx and lets a user pick a voice,
type text, and play synthesized speech in the browser.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import gradio as gr
from piper import PiperVoice, SynthesisConfig


@dataclass(frozen=True)
class VoiceModel:
    label: str
    model_path: Path
    config_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview exported Piper voices in a local browser app")
    parser.add_argument("--root-dir", default=".", help="Repository root")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window automatically")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA when loading Piper voices")
    return parser.parse_args()


def discover_voices(root_dir: Path) -> list[VoiceModel]:
    export_root = root_dir / "voice-files-out"
    if not export_root.exists():
        return []

    voices: list[VoiceModel] = []
    for model_path in sorted(export_root.glob("*/exports/*.onnx")):
        config_path = model_path.with_suffix(".onnx.json")
        if not config_path.exists():
            config_path = None

        voice_name = model_path.parent.parent.name
        label = f"{voice_name} / {model_path.stem}"
        voices.append(VoiceModel(label=label, model_path=model_path, config_path=config_path))

    return voices


def voice_choices(voices: Iterable[VoiceModel]) -> list[tuple[str, str]]:
    return [(voice.label, str(voice.model_path)) for voice in voices]


def load_config(config_path: Path | None) -> SynthesisConfig:
    if config_path is None:
        return SynthesisConfig()

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return SynthesisConfig()

    inference = config_data.get("inference", {})
    return SynthesisConfig(
        length_scale=inference.get("length_scale"),
        noise_scale=inference.get("noise_scale"),
        noise_w_scale=inference.get("noise_w"),
        normalize_audio=True,
    )


@lru_cache(maxsize=8)
def load_voice(model_path_str: str, config_path_str: str | None, use_cuda: bool) -> PiperVoice:
    model_path = Path(model_path_str)
    config_path = Path(config_path_str) if config_path_str else None
    return PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)


def synthesize(model_path_str: str, text: str, use_cuda: bool) -> tuple[str | None, str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return None, "Enter some text to synthesize."

    model_path = Path(model_path_str)
    config_path = model_path.with_suffix(".onnx.json")
    if not config_path.exists():
        config_path = None

    if not model_path.exists():
        return None, f"Missing model file: {model_path}"

    try:
        voice = load_voice(str(model_path), str(config_path) if config_path else None, use_cuda)
        syn_config = load_config(config_path)

        with tempfile.NamedTemporaryFile(prefix="voice-test-", suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        with wave.open(str(temp_path), "wb") as wav_file:
            voice.synthesize_wav(cleaned_text, wav_file, syn_config=syn_config)

        return str(temp_path), f"Played from {model_path.parent.parent.name}."
    except Exception as exc:
        return None, f"Synthesis failed: {exc}"


def build_app(root_dir: Path, use_cuda: bool) -> gr.Blocks:
    voices = discover_voices(root_dir)
    choices = voice_choices(voices)
    default_value = choices[0][1] if choices else None
    status_text = (
        "No exported voices found yet. Train a voice first, then reopen this app."
        if not choices
        else ""
    )

    with gr.Blocks(title="Voice Tester") as demo:
        gr.Markdown(
            "# Voice Tester\n"
            "Pick an exported Piper voice, type text, and press Play. "
            "This scans `voice-files-out/*/exports/*.onnx` automatically."
        )

        voice_dropdown = gr.Dropdown(
            label="Voice",
            choices=choices,
            value=default_value,
            interactive=True,
        )
        text_input = gr.Textbox(
            label="Text",
            lines=4,
            placeholder="Type something to hear it spoken...",
        )
        play_button = gr.Button("Play", variant="primary")
        audio_output = gr.Audio(label="Playback", type="filepath")
        status_output = gr.Textbox(label="Status", value=status_text, interactive=False)

        def on_play(selected_model_path: str, text: str) -> tuple[str | None, str]:
            if not selected_model_path:
                return None, "Choose a voice first."

            return synthesize(selected_model_path, text, use_cuda)

        play_button.click(
            fn=on_play,
            inputs=[voice_dropdown, text_input],
            outputs=[audio_output, status_output],
        )

    return demo


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    app = build_app(root_dir, args.cuda)
    app.launch(server_name=args.host, server_port=args.port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()