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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from voice_generator.audio import wav_info


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


class BackendNotInstalledError(RuntimeError):
    """A backend's engine package is not installed in this environment.

    Raised (instead of a bare ``RuntimeError``) when a backend cannot even
    import its engine, so the CLI can tell "not installed — here is the fix"
    apart from a real engine failure (CUDA OOM, shape mismatch, genuine bug),
    which must still propagate with its traceback. See the CLI's
    ``EXPECTED_ERRORS`` contract.
    """


def require_reference(
    request: SynthesisRequest, backend_name: str, min_reference_s: float = 0.0
) -> Path:
    """Return the reference clip, or raise the same error every backend used to repeat.

    Centralised here so the message (and the CLI flag it points at) cannot
    drift between backends. Callers must call this before loading a model:
    a missing reference is a user error and should fail before a multi-GB
    model download starts.

    When ``min_reference_s`` is positive, each backend passes its own
    ``min_reference_s`` and the clip's actual duration is read from its
    WAV header: a clip shorter than the minimum is rejected naming the
    backend, the minimum, and the actual duration, because a 1 s clip
    clones badly with no signal otherwise. (The pipeline already rejects
    non-WAV references before any backend runs, so a ``wave.Error`` here
    only fires for direct library use — and its message carries the
    ffmpeg conversion fix.) Backends that do not clone from a reference
    (the mock) do not call this at all.
    """
    if request.reference_wav is None:
        raise ValueError(f"{backend_name} needs a reference clip (--ref your_ref_clip.wav)")
    ref = request.reference_wav
    if not ref.exists():
        raise FileNotFoundError(f"reference clip not found: {ref}")
    if min_reference_s > 0:
        info = wav_info(ref)
        if info.duration_s < min_reference_s:
            raise ValueError(
                f"{backend_name} needs a reference clip of at least {min_reference_s:.0f}s "
                f"(--ref your_ref_clip.wav), got {info.duration_s:.1f}s"
            )
    return ref


def resolve_device(preferred: str) -> str:
    """Resolve a backend's device request to something torch can actually use.

    ``"auto"`` (the default) means CUDA when torch reports it, else MPS on
    Apple Silicon, else CPU — so a CPU-only machine gets a clean, slow
    render (or a clear engine error) instead of a cryptic CUDA failure
    from deep inside a model load. An explicit ``"cuda"`` / ``"cuda:0"`` /
    ``"cpu"`` / ``"mps"`` is honoured verbatim, except an explicit
    ``"cuda*"`` request is validated against the real hardware first: a
    ``"cuda"`` request on a CPU-only box (or ``"cuda:1"`` on a single-GPU
    box) raises ``ValueError`` naming the request and the hardware,
    instead of failing deep inside ``from_pretrained``. When torch itself
    is absent there is nothing to validate against, so the request passes
    through and the engine import fails first, as before.

    torch itself is imported lazily and its absence (core env, tests)
    resolves to ``"cpu"``.

    The fallback is loud on purpose: on 2026-11-09 a venv whose torch
    was a CPU-only build made a Qwen3 render sit with no progress for
    12+ minutes after model load, looking hung, because nothing said
    the GPU was not in use. When torch is present but neither CUDA nor
    MPS is available, a stderr warning names the cause and the fix — the
    CUDA-torch install hint only on platforms that can plausibly have an
    Nvidia GPU (on macOS there is no CUDA torch build to install, so the
    warning says MPS/CPU instead).
    """
    if preferred != "auto":
        if preferred.startswith("cuda"):
            try:
                import torch
            except ImportError:
                return preferred  # nothing to validate against; engine import fails first
            try:
                cuda_ok = torch.cuda.is_available()
            except Exception as exc:
                raise ValueError(
                    f"device {preferred!r} requested but CUDA could not be "
                    f"initialised ({exc})"
                ) from exc
            if not cuda_ok:
                raise ValueError(
                    f"device {preferred!r} requested but CUDA is not available "
                    "on this machine — torch reports no CUDA device; use "
                    '"auto" (CPU/MPS) or install a CUDA torch build'
                )
            if ":" in preferred:
                try:
                    index = int(preferred.split(":", 1)[1])
                except ValueError:
                    raise ValueError(
                        f"device {preferred!r} has a non-numeric index — "
                        'use "cuda" or "cuda:<n>"'
                    ) from None
                count = torch.cuda.device_count()
                if index < 0 or index >= count:
                    raise ValueError(
                        f"device {preferred!r} requested but this machine has "
                        f"only {count} CUDA device(s) (indices 0–{count - 1})"
                    )
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
    try:
        # Apple Silicon has no CUDA; without this branch every backend
        # silently fell back to CPU even though the engines would use MPS.
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        # Older torch without backends.mps, or a broken MPS backend:
        # treat as unavailable and fall through to the CPU warning.
        pass
    if sys.platform == "darwin":
        print(
            "Warning: torch is installed but neither CUDA nor MPS is available — "
            "synthesis will run on CPU. Large models (Qwen3 1.7B) can take "
            "10+ minutes per step on CPU.",
            file=sys.stderr,
        )
    else:
        print(
            "Warning: torch is installed but CUDA is not available — synthesis "
            "will run on CPU. Large models (Qwen3 1.7B) can take 10+ minutes per "
            "step on CPU. If this machine has an Nvidia GPU, install a CUDA "
            "torch build: pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121",
            file=sys.stderr,
        )
    return "cpu"


def flatten_mono_samples(samples: object) -> Sequence[float]:
    """Flatten one model-output chunk to mono float samples.

    Model outputs arrive in several shapes — a torch tensor ``(1, N)`` or
    ``(N, 1)``, a numpy array, a nested list ``[[...]]`` — and the writer
    only takes a flat 1-D sequence. This is the one place that conversion
    lives, so its edge behaviour is the same for every backend (it used to
    be three near-identical helpers with divergent failure modes).

    The result is the cheapest 1-D view available (a numpy/tensor row or
    the input list itself), NOT a fresh Python list of floats: the writer
    iterates it directly, so building a list here would be a second copy
    of every chunk for nothing. Callers may use ``len()`` on the result
    but must not mutate it.

    Raises ``ValueError`` on multi-channel ``(2, N)`` output — taking
    channel 0 silently would ship a clone from half the audio — and
    ``TypeError`` on an unrecognised type.
    """
    obj = samples
    for step in ("detach", "cpu", "numpy"):
        if hasattr(obj, step):
            obj = getattr(obj, step)()
    shape = getattr(obj, "shape", None)
    if shape is not None:
        # Array-like (numpy array, torch tensor): validate by shape and
        # return a view — no copy, no Python list.
        dims = tuple(shape)
        if len(dims) == 1:
            return obj  # type: ignore[return-value]
        if len(dims) == 2:
            if dims[0] == 1:
                return obj[0]  # type: ignore[index, return-value]
            if dims[1] == 1:
                return obj[:, 0]  # type: ignore[index, return-value]
            raise ValueError(
                "expected mono model output (1, N) or (N, 1), got "
                f"{dims[0]} channels of {dims[1]} samples — "
                "multi-channel output is not supported"
            )
        raise ValueError(f"expected 1-D or 2-D model output, got shape {dims}")
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        return _flatten_list(obj)
    if getattr(obj, "ndim", 1) == 2:
        return _flatten_2d(obj)
    if isinstance(obj, (str, bytes)) or not hasattr(obj, "__iter__"):
        raise TypeError(f"unsupported model output type: {type(samples).__name__}")
    return obj  # type: ignore[return-value]


def _flatten_list(obj: list | tuple) -> Sequence[float]:
    """Mono-ify a plain (possibly nested) list of samples."""
    if obj and isinstance(obj[0], (list, tuple)):
        if len(obj) == 1:
            return obj[0]  # (1, N)
        if all(len(row) == 1 for row in obj):
            return [row[0] for row in obj]  # (N, 1): squeeze the column
        raise ValueError(
            "expected mono model output (1, N) or (N, 1), got "
            f"{len(obj)} channels of {len(obj[0])} samples — "
            "multi-channel output is not supported"
        )
    return obj


def _flatten_2d(obj: object) -> Sequence[float]:
    """Mono-ify a 2-D tensor-like with no ``shape`` and no ``tolist``.

    Only minimal tensor-likes (test fakes) reach here: real tensors and
    numpy arrays are handled by shape above. Length is used to validate
    when it exists; without it there is nothing to validate against, so
    the single row is taken as before — the row conversion below still
    fails loudly on a non-iterable row.
    """
    try:
        n_rows = len(obj)  # type: ignore[arg-type]
    except TypeError:
        return _as_sequence(obj[0])  # type: ignore[index]
    if n_rows == 1:
        return _as_sequence(obj[0])  # type: ignore[index]
    first = obj[0]  # type: ignore[index]
    try:
        first_len = len(first)
    except TypeError:
        raise ValueError("expected mono model output (1, N) or (N, 1)") from None
    if first_len == 1:
        return [float(obj[i][0]) for i in range(n_rows)]  # type: ignore[index]
    raise ValueError(
        "expected mono model output (1, N) or (N, 1), got "
        f"{n_rows} channels of {first_len} samples — "
        "multi-channel output is not supported"
    )


def _as_sequence(row: object) -> Sequence[float]:
    """Return one row of model output as a 1-D sequence of floats."""
    if hasattr(row, "tolist"):
        return row.tolist()  # type: ignore[attr-defined]
    if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
        raise TypeError(f"unsupported model output row type: {type(row).__name__}")
    return row  # type: ignore[return-value]


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
