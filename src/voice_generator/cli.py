"""voice-gen CLI.

Examples (Windows, in the Chatterbox venv):
  voice-gen doctor
  voice-gen check-ref refs\\me_10s.wav
  voice-gen synth --script script.txt --ref refs\\me_10s.wav --out output\\ep1.wav
  voice-gen synth --text "Hello, this is my digital twin." --ref refs\\me_10s.wav --out output\\hello.wav --backend mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voice_generator.audio import check_reference, wav_info
from voice_generator.backends import available_backends
from voice_generator.pipeline import synthesize_script


def _doctor() -> int:
    import platform

    print(f"python   {platform.python_version()}  ({sys.executable})")
    try:
        import torch

        print(f"torch    {torch.__version__}")
        if torch.cuda.is_available():
            print(f"cuda     {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"vram     {total:.1f} GB")
        else:
            print("cuda     not available — real backends need an Nvidia GPU (mock still works)")
    except ImportError:
        print("torch    not installed in this venv (expected unless this is a backend venv)")
    try:
        import chatterbox  # noqa: F401

        print("chatterbox  installed")
    except ImportError:
        print('chatterbox  not installed — pip install -e ".[chatterbox]" in its venv')
    import shutil

    print(f"ffmpeg   {shutil.which('ffmpeg') or 'not found on PATH (needed later for Stage 2)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice-gen", description="Local FOSS voice cloning")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check python / torch / CUDA / VRAM / ffmpeg")
    sub.add_parser("backends", help="list backends available in this build")

    p_ref = sub.add_parser("check-ref", help="validate a reference voice clip")
    p_ref.add_argument("ref", type=Path)

    p = sub.add_parser("synth", help="synthesize a script (or one line of text) to a WAV")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="a single line / short passage")
    src.add_argument("--script", type=Path, help="path to a .txt script file (chunked automatically)")
    p.add_argument("--ref", type=Path, default=None, help="reference voice clip (.wav, ~10 s for Chatterbox)")
    p.add_argument("--out", type=Path, required=True, help="output .wav path")
    p.add_argument("--backend", default="chatterbox", choices=available_backends())
    p.add_argument("--variant", default="turbo", choices=["original", "turbo", "multilingual"])
    p.add_argument("--language", default="en")
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg", type=float, default=0.5, dest="cfg_weight")
    p.add_argument("--max-chars", type=int, default=300)

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _doctor()
    if args.command == "backends":
        print("\n".join(available_backends()))
        print("\nplanned (separate venvs, not yet wired): cosyvoice (CosyVoice 3), qwen3 (Qwen3-TTS)")
        return 0
    if args.command == "check-ref":
        info = wav_info(args.ref)
        print(f"{args.ref}: {info.duration_s:.1f}s, {info.sample_rate} Hz, {info.channels}ch")
        warnings = check_reference(args.ref)
        for w in warnings:
            print(f"  ! {w}")
        if not warnings:
            print("  OK — good reference clip")
        return 0
    if args.command == "synth":
        text = args.text if args.text else args.script.read_text(encoding="utf-8")
        result = synthesize_script(
            text=text,
            reference_wav=args.ref,
            out_wav=args.out,
            backend_name=args.backend,
            variant=args.variant,
            language=args.language,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            max_chars=args.max_chars,
        )
        print(f"\nDone: {result.wav_path}  ({result.duration_s:.1f}s, {result.chunks} chunk(s))")
        print(f"Manifest: {result.manifest_path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
