"""The backend contract every TTS engine implements.

One interface, interchangeable engines: that is what lets Stage 1 swap
Chatterbox for CosyVoice 3 / Qwen3-TTS without touching the CLI, the
chunking, the manifest, or Stage 2 (which only ever sees a WAV file).

Backends must keep heavy imports (torch, the engine package) inside
methods, never at module top level: the registry imports every backend
module lazily, but a top-level torch import in any one of them would
break the torch-free core CLI and the GPU-less test suite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SynthesisRequest:
    """Everything a backend needs to render one chunk of text.

    Fields that only some engines use (``exaggeration``, ``ref_text``) are
    carried for all of them on purpose: the pipeline builds one request
    shape and every backend ignores what it does not support, instead of
    the pipeline branching per backend. ``reference_wav`` is ``None`` only
    for backends with a built-in default voice (currently none of the real
    ones — the mock backend ignores it).
    """

    text: str
    reference_wav: Path | None  # None = backend's default voice, if it has one
    language: str = "en"
    exaggeration: float = 0.5  # Chatterbox-style emotion intensity; ignored if unsupported
    cfg_weight: float = 0.5
    ref_text: str | None = None  # exact transcript of the reference clip (Qwen3 ICL cloning)
    x_vector_only: bool = False  # speaker-embedding-only cloning where supported (Qwen3)


@dataclass(frozen=True)
class SynthesisResult:
    """What a backend returns for one rendered chunk.

    ``settings`` is the backend's own record of the knobs that produced
    this chunk. The pipeline copies it into the run manifest, so a render
    stays reproducible from the manifest alone — a backend that leaves
    this empty is hiding provenance.
    """

    wav_path: Path
    backend: str
    duration_s: float
    settings: dict[str, object] = field(default_factory=dict)


def require_reference(request: SynthesisRequest, backend_name: str) -> Path:
    """Return the reference clip, or raise the same error every backend used to repeat.

    Centralised here so the message (and the CLI flag it points at) cannot
    drift between backends. Callers must call this before loading a model:
    a missing reference is a user error and should fail before a multi-GB
    model download starts.
    """
    if request.reference_wav is None:
        raise ValueError(f"{backend_name} needs a reference clip (--ref your_ref_clip.wav)")
    return request.reference_wav


def resolve_device(preferred: str) -> str:
    """Resolve a backend's device request to something torch can actually use.

    ``"auto"`` (the default) means CUDA when torch reports it, else CPU —
    so a CPU-only machine gets a clean, slow render (or a clear engine
    error) instead of a cryptic CUDA failure from deep inside a model
    load. An explicit ``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` is honoured
    verbatim. torch itself is imported lazily and its absence (core env,
    tests) resolves to ``"cpu"``.

    The fallback is loud on purpose: on 2026-11-09 a venv whose torch
    was a CPU-only build made a Qwen3 render sit with no progress for
    12+ minutes after model load, looking hung, because nothing said
    the GPU was not in use. When torch is present but CUDA is not, a
    stderr warning names the cause and the fix.
    """
    if preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch present but broken/uninitialised: fail soft to CPU
        print(
            "Warning: torch is installed but CUDA could not be initialised — "
            "synthesis will run on CPU and be very slow on large models.",
            file=sys.stderr,
        )
        return "cpu"
    print(
        "Warning: torch is installed but CUDA is not available — synthesis "
        "will run on CPU. Large models (Qwen3 1.7B) can take 10+ minutes per "
        "step on CPU. If this machine has an Nvidia GPU, install a CUDA "
        "torch build: pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121",
        file=sys.stderr,
    )
    return "cpu"


class BaseBackend:
    """Base class for TTS backends. Subclasses implement ``synthesize`` only.

    ``name`` and ``license`` are part of the contract, not decoration:
    both are copied into every run manifest.
    """

    name: str = "base"
    license: str = "unknown"
    min_reference_s: float = 3.0

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        """Render ``request.text`` in the cloned voice to ``out_wav`` (PCM-16 WAV).

        Backends write via ``voice_generator.audio.write_pcm16_wav`` (or an
        equivalent PCM-16 writer) so the pipeline's stdlib concatenation
        always works. Implementations must validate their own inputs
        (reference clip, transcript) before loading a model.
        """
        raise NotImplementedError
