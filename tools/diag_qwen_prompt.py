r"""Minimal upstream repro for a Qwen3 clone-prompt stall.

Bypasses voice-generator entirely: calls qwen-tts directly with
timestamps, so you can see exactly which call hangs. It found the
2026-11-09 stall's real cause — a CPU-only torch build in the venv
(``cuda available: False``), which made prompt creation grind for
12+ minutes instead of failing loudly.

Run from the project root, in the Qwen3 venv:

    python tools\diag_qwen_prompt.py
"""

import time


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("import torch + qwen_tts ...")
# Imports are deliberately after the log lines (noqa E402): the point
# of this script is timestamping each step, including the import.
import torch  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402

log(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"device: {torch.cuda.get_device_name(0)}")

log("loading ref audio with torchaudio ...")
import torchaudio  # noqa: E402

wav, sr = torchaudio.load(r"refs\me_10s.wav")
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
    ref_audio=r"refs\me_10s.wav",
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
