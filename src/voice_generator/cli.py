"""voice-gen CLI.

Examples (Windows, in the Chatterbox venv):
  voice-gen doctor
  voice-gen check-ref refs\\me_10s.wav
  voice-gen synth --script script.txt --ref refs\\me_10s.wav --out output\\ep1.wav
  voice-gen synth --text "Hello, this is my digital twin." \
      --ref refs\\me_10s.wav --out output\\hello.wav --backend mock

Error contract: expected failures (missing files, bad input, a backend
not installed in this venv) print a one-line ``Error: ...`` to stderr and
exit 1, instead of dumping a traceback — the common cases are user
errors with a known fix, and the fix is printed with the error where
one exists. Unexpected exceptions still propagate with a traceback.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import wave
from pathlib import Path

from voice_generator.audio import check_reference, wav_info
from voice_generator.backends import BACKEND_IMPORTS, available_backends
from voice_generator.pipeline import synthesize_script

# Errors the CLI converts to a clean message instead of a traceback.
# wave.Error covers malformed WAVs; the rest are the documented failure
# modes of the pipeline and backends.
EXPECTED_ERRORS = (FileNotFoundError, ValueError, RuntimeError, wave.Error)


def _probe(module: str) -> bool:
    """True if ``module`` imports in this environment, without side effects."""
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _doctor() -> int:
    """Print environment diagnostics: Python, torch/CUDA/VRAM, backends, ffmpeg."""
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
    # One probe per backend instead of three hand-written try/except
    # blocks that drifted apart (the old code probed chatterbox/qwen-tts
    # by import and cosyvoice by import too, with different messages).
    hints = {
        "chatterbox": 'not installed — pip install -e ".[chatterbox]" in its venv',
        "qwen3": 'not installed — pip install -e ".[qwen3]" in its venv',
        "cosyvoice": "not importable here — it lives in its own Conda env + repo clone",
    }
    for name in available_backends():
        module = BACKEND_IMPORTS[name]
        status = "installed" if _probe(module) else hints[name]
        print(f"{name:<11} {status}")
    home = os.environ.get("COSYVOICE_HOME")
    print(f"cosy-home  {home or 'unset (set COSYVOICE_HOME or pass --cosyvoice-home)'}")
    print(f"ffmpeg   {shutil.which('ffmpeg') or 'not found on PATH (needed later for Stage 2)'}")
    return 0


def _synth(args: argparse.Namespace) -> int:
    text = args.text if args.text else args.script.read_text(encoding="utf-8")
    ref_text = args.ref_text
    if args.ref_text_file is not None:
        ref_text = args.ref_text_file.read_text(encoding="utf-8").strip()
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
        ref_text=ref_text,
        x_vector_only=args.x_vector_only,
        model_size=args.model_size,
        attn=args.attn,
        cosyvoice_home=args.cosyvoice_home,
        cosy_model_dir=args.cosy_model_dir,
        cosy_mode=args.cosy_mode,
        cosy_version=args.cosy_version,
    )
    print(f"\nDone: {result.wav_path}  ({result.duration_s:.1f}s, {result.chunks} chunk(s))")
    print(f"Manifest: {result.manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="voice-gen", description="Local FOSS voice cloning")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check python / torch / CUDA / VRAM / ffmpeg")
    sub.add_parser("backends", help="list backends available in this build")

    p_ref = sub.add_parser("check-ref", help="validate a reference voice clip")
    p_ref.add_argument("ref", type=Path)

    p = sub.add_parser("synth", help="synthesize a script (or one line of text) to a WAV")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="a single line / short passage")
    src.add_argument(
        "--script", type=Path, help="path to a .txt script file (chunked automatically)"
    )
    p.add_argument(
        "--ref", type=Path, default=None, help="reference voice clip (.wav, ~10 s for Chatterbox)"
    )
    p.add_argument("--out", type=Path, required=True, help="output .wav path")
    p.add_argument("--backend", default="chatterbox", choices=available_backends())
    p.add_argument("--variant", default="turbo", choices=["original", "turbo", "multilingual"])
    p.add_argument("--language", default="en")
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg", type=float, default=0.5, dest="cfg_weight")
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument(
        "--ref-text", default=None, help="exact transcript of the reference clip (Qwen3)"
    )
    p.add_argument(
        "--ref-text-file", type=Path, default=None, help="file containing that transcript"
    )
    p.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Qwen3: clone from speaker embedding only, no transcript needed (lower quality)",
    )
    p.add_argument("--model-size", default="1.7b", choices=["1.7b", "0.6b"], help="Qwen3 Base size")
    p.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Qwen3 attention implementation (sdpa is the Windows-friendly default)",
    )
    p.add_argument(
        "--cosyvoice-home", default=None, help="CosyVoice: path to the cloned CosyVoice repo"
    )
    p.add_argument("--cosy-model-dir", default=None, help="CosyVoice: model snapshot directory")
    p.add_argument(
        "--cosy-mode",
        default="zero_shot",
        choices=["zero_shot", "cross_lingual"],
        help="CosyVoice clone mode (cross_lingual needs no reference transcript)",
    )
    p.add_argument("--cosy-version", default="3", choices=["3", "2"], help="CosyVoice version")

    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "backends":
            print("\n".join(available_backends()))
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
            return _synth(args)
    except EXPECTED_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
