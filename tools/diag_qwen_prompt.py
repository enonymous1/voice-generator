r"""Minimal upstream repro for a Qwen3 clone-prompt stall.

Bypasses voice-generator entirely: calls qwen-tts directly with
timestamps, so you can see exactly which call hangs. It found the
2026-11-09 stall's real cause — a CPU-only torch build in the venv
(``cuda available: False``), which made prompt creation grind for
12+ minutes instead of failing loudly.

Run from the project root, in the Qwen3 venv:

    python tools\diag_qwen_prompt.py [--ref refs\me_10s.wav] [--full]

Without --full, the script stops after the CUDA verdict on a CPU-only
box — that verdict is usually the whole answer, and the full run
would grind for 10+ minutes to prove it again. Importing this module
is safe (it loads nothing); the model loads only inside main().
"""

import argparse
import time
from pathlib import Path

# The reference clip, relative to the project root (a Windows-style
# backslash literal here would be a literal filename character on
# Linux/macOS, so this is a Path, not a string).
REF = Path("refs") / "me_10s.wav"


def log(msg: str) -> None:
    """Print a timestamped step line — timing each call is the point of this script."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the Qwen3 clone-prompt stall against qwen-tts directly."
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=REF,
        help="reference clip (default: refs/me_10s.wav)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the full repro (model load + generation) even on a CPU-only box",
    )
    args = parser.parse_args(argv)

    log("import torch + qwen_tts ...")
    # Imports are deliberately after the log lines: the point of this
    # script is timestamping each step, including the import.
    import torch
    from qwen_tts import Qwen3TTSModel

    log(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"device: {torch.cuda.get_device_name(0)}")
    elif not args.full:
        log("CPU-only: prompt creation would grind for 10+ minutes on CPU —")
        log("this script already answered the CUDA question. Re-run with")
        log("--full to grind through it anyway.")
        return 0
    else:
        log("CPU-only and --full given: prompt creation will take 10+ minutes;")
        log("Ctrl+C if you only needed the CUDA verdict.")

    log("loading ref audio with torchaudio ...")
    import torchaudio

    wav, sr = torchaudio.load(str(args.ref))
    log(f"ref audio: {tuple(wav.shape)} @ {sr} Hz")

    log("loading 1.7B Base model ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    log("model loaded")

    log("create_voice_clone_prompt (x-vector only) ...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(args.ref),
        ref_text=None,
        x_vector_only_mode=True,
    )
    log("prompt created")

    log("generate_voice_clone ...")
    wavs, sr = model.generate_voice_clone(
        text="Hello, this is a test of my cloned voice.",
        language="English",
        voice_clone_prompt=prompt,
    )
    log(f"generated: sr={sr}, samples={len(wavs[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
