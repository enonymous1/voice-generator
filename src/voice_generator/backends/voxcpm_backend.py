"""VoxCPM2 backend (OpenBMB) — wired in v0.4, at the user's choice after a
survey of fast open voice-cloning models (2026-11-10).

Verified against the upstream package docs on 2026-11-10 (PyPI
`voxcpm` 2.0.3 README + readthedocs Quick Start / Usage Guide):
package `voxcpm` on PyPI, Python 3.10–<3.13, PyTorch >= 2.5, CUDA >= 12.
Weights and code are Apache-2.0. Model: ``openbmb/VoxCPM2`` (2B,
30 languages, 48 kHz out, accepts 16 kHz reference audio).

API facts that shape this code:
- ``VoxCPM.from_pretrained(model_id, load_denoiser=False, device="auto",
  optimize=...)`` — ``device`` accepts "auto" (prefers cuda -> mps ->
  cpu), "cuda", "cuda:0", "cpu", "mps". ``optimize`` controls
  ``torch.compile``; the docs say to try ``optimize=False`` on
  platform-specific compile issues, and the CLI ships ``--no-optimize``
  for the same reason. This backend defaults ``optimize`` on (the
  vendor's RTF ~0.30 on a 4090 was measured with it) and falls back to
  off once, automatically, when a load or a first generate fails with a
  compile/Triton-shaped error — the Windows failure mode the survey
  flagged. If an installed voxcpm does not accept the ``optimize``
  keyword at all, the backend retries without it instead of failing.
- ``generate(text, reference_wav_path=..., prompt_wav_path=...,
  prompt_text=..., cfg_value=2.0, inference_timesteps=10,
  normalize=False, retry_badcase=True)`` returns a numpy waveform at
  ``model.tts_model.sample_rate`` (48 kHz). Two clone modes:
  *controllable* (default) — reference clip only, no transcript, timbre
  cloned; *ultimate* — the same clip passed as both ``prompt_wav_path``
  and ``reference_wav_path`` plus its exact transcript as
  ``prompt_text``, for maximum similarity. (Upstream calls the second
  mode "Hi-Fi Cloning" in the Usage Guide and "Ultimate Cloning" on
  PyPI; this backend calls it ``ultimate``.)
- Reference audio: 5–30 s is the practical range; any format torchaudio
  supports. Upstream also documents a streaming API
  (``generate_streaming``) and a Voice Design mode (no reference) —
  neither is wired here: the pipeline is chunk-based, and every real
  backend in this project clones from a reference clip.
- The denoiser is NOT loaded (``load_denoiser=False``): it only
  enhances noisy prompt audio, and the reference clips this project
  targets are clean recordings. There is no denoise knob in v0.4.0.
- Long text: upstream warns it can buzz, speed up, or OOM, and
  recommends splitting into segments — which is exactly what this
  project's chunker already does (default 300 chars).
"""

from __future__ import annotations

import sys
from pathlib import Path

from voice_generator.audio import write_pcm16_wav
from voice_generator.backends.base import (
    BackendNotInstalledError,
    BaseBackend,
    SynthesisRequest,
    SynthesisResult,
    flatten_mono_samples,
    require_reference,
    resolve_device,
)

_DEFAULT_MODEL = "openbmb/VoxCPM2"
_DEFAULT_SAMPLE_RATE = 48000  # VoxCPM2 outputs 48 kHz; used only if the model won't say so

_SETUP_HINT = (
    "voxcpm is not installed in this environment. Install it in its own venv "
    "(upstream: Python 3.10–<3.13, PyTorch >= 2.5, CUDA >= 12):\n"
    "  py -3.12 -m venv .venv-voxcpm\n"
    "  .venv-voxcpm\\Scripts\\activate   (Windows)\n"
    "  pip install torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu121\n"
    '  pip install -e ".[voxcpm]"\n'
    'Then verify with: python -c "import torch; print(torch.cuda.is_available())"'
)


def _looks_like_compile_error(exc: BaseException) -> bool:
    """True if an exception smells like a torch.compile / Triton failure.

    String-matched on purpose: upstream surfaces these as RuntimeError
    (or a wrapped error) with the failing subsystem in the message, and
    there is no stable exception type to catch. Kept narrow — the tokens
    name the compile subsystems (Triton, TorchDynamo, TorchInductor,
    CUDA graphs), so a download failure or a bad model id must NOT
    trigger the optimize fallback, or a real error would be retried and
    re-reported as a different one. A bare "compile" substring used to
    be in the list and matched unrelated errors (a tokenizer or
    model-file message mentioning compilation), triggering a full
    multi-GB model reload for nothing — it was removed for that reason.
    """
    text = str(exc).lower()
    return any(
        token in text
        for token in ("triton", "torch.compile", "torchdynamo", "inductor", "cudagraph")
    )


class VoxCPM2Backend(BaseBackend):
    """VoxCPM2 voice cloning in controllable or ultimate mode.

    Controllable mode clones from the reference clip alone; ultimate
    mode passes the same clip as both prompt and reference plus its
    exact transcript, for maximum similarity. ``optimize`` controls
    ``torch.compile`` and defaults on (the vendor's speed figures were
    measured with it); when a load or a first generate fails with a
    compile/Triton-shaped error, the backend reloads unoptimized and
    retries once. A second full model load is expensive, so the "retry
    once" policy is load-bearing — the fallback never retries twice.
    """

    name = "voxcpm"
    license = "Apache-2.0"
    min_reference_s = 5.0  # upstream: 5–30 s is the practical range

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        mode: str = "controllable",
        device: str = "auto",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        optimize: bool = True,
        normalize: bool = False,
    ) -> None:
        if mode not in ("controllable", "ultimate"):
            raise ValueError(f"mode must be controllable or ultimate, got {mode!r}")
        if cfg_value <= 0:
            raise ValueError(f"cfg_value must be positive, got {cfg_value!r}")
        if inference_timesteps < 1:
            raise ValueError(f"inference_timesteps must be >= 1, got {inference_timesteps!r}")
        self.model_id = model_id
        self.mode = mode
        self.device = device
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.optimize = optimize
        self.normalize = normalize
        self._model: object | None = None
        # False once an installed voxcpm proves it does not accept the
        # optimize keyword (older/newer API drift); the load then omits it.
        self._optimize_kwarg_supported = True

    def _load(self) -> object:
        """Import the engine and load the model, once, with two fallbacks.

        First, an installed voxcpm that rejects the ``optimize`` keyword
        is retried without it. Second, a compile/Triton-shaped failure
        is retried once with ``optimize=False`` — but only with the
        keyword when this install accepts it, or the retry would raise
        ``TypeError`` and mask the original compile failure. Either
        fallback's retry can trigger the other, so both run inside one
        loop; each fires at most once, so the load is attempted at most
        three times. Anything else (bad model id, download failure)
        re-raises untouched. Raises ``BackendNotInstalledError`` when
        the engine package is missing, so the CLI can tell "not
        installed" apart from a real engine failure.
        """
        if self._model is not None:
            return self._model
        try:
            from voxcpm import VoxCPM
        except ImportError as exc:
            raise BackendNotInstalledError(_SETUP_HINT) from exc
        kwargs: dict[str, object] = {
            "load_denoiser": False,
            "device": resolve_device(self.device),
        }
        if self._optimize_kwarg_supported:
            kwargs["optimize"] = self.optimize
        while True:
            try:
                self._model = VoxCPM.from_pretrained(self.model_id, **kwargs)
                return self._model
            except TypeError as exc:
                # The installed voxcpm does not know the optimize keyword:
                # retry once without it instead of failing on a kwarg.
                if self._optimize_kwarg_supported and "optimize" in str(exc):
                    self._optimize_kwarg_supported = False
                    kwargs.pop("optimize")
                    continue
                raise
            except Exception as exc:
                # A compile/Triton failure at load: retry once unoptimized.
                # Anything else (bad model id, download failure) re-raises
                # untouched — retrying those would only waste a download.
                if self.optimize and _looks_like_compile_error(exc):
                    print(
                        "VoxCPM torch.compile failed at load "
                        f"({exc}); retrying with optimize=False.",
                        file=sys.stderr,
                    )
                    self.optimize = False
                    if self._optimize_kwarg_supported:
                        kwargs["optimize"] = False
                    continue
                raise

    def _generate(self, request: SynthesisRequest, ref: Path) -> object:
        """One generate() call, with the same optimize fallback as _load.

        torch.compile is lazy — a Triton failure can surface on the
        first forward pass (here) rather than at load. On that error the
        model is reloaded unoptimized and the chunk is retried once.
        """
        model = self._load()
        kwargs: dict[str, object] = {
            "text": request.text,
            "reference_wav_path": str(ref),
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
            "normalize": self.normalize,
            "retry_badcase": True,
        }
        if self.mode == "ultimate":
            kwargs["prompt_wav_path"] = str(ref)
            kwargs["prompt_text"] = request.ref_text
        try:
            return model.generate(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            if self.optimize and _looks_like_compile_error(exc):
                print(
                    "VoxCPM torch.compile failed during synthesis "
                    f"({exc}); retrying this chunk with optimize=False.",
                    file=sys.stderr,
                )
                self.optimize = False
                self._model = None
                model = self._load()
                return model.generate(**kwargs)  # type: ignore[attr-defined]
            raise

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        """Render one chunk in the selected clone mode.

        Ultimate mode needs the reference transcript (it is the prompt);
        controllable mode needs only the clip. The reference is validated
        (exists, at least ``min_reference_s``) before the model loads.
        """
        ref = require_reference(request, "voxcpm", self.min_reference_s)
        if self.mode == "ultimate" and not request.ref_text:
            raise ValueError(
                "voxcpm ultimate mode needs the reference transcript: pass --ref-text \"...\" "
                "(or --ref-text-file). For transcript-free cloning use the default "
                "controllable mode (omit --voxcpm-mode)."
            )
        wav = self._generate(request, ref)
        samples = flatten_mono_samples(wav)
        if len(samples) == 0:
            raise RuntimeError("voxcpm produced no audio for this chunk")
        model = self._model
        tts_model = getattr(model, "tts_model", None)
        sample_rate = int(getattr(tts_model, "sample_rate", None) or _DEFAULT_SAMPLE_RATE)
        write_pcm16_wav(out_wav, samples, sample_rate)
        return SynthesisResult(
            wav_path=out_wav,
            backend="voxcpm",
            duration_s=len(samples) / sample_rate,
            settings={
                "model": self.model_id,
                "mode": self.mode,
                "cfg_value": self.cfg_value,
                "inference_timesteps": self.inference_timesteps,
                "optimize": self.optimize if self._optimize_kwarg_supported else "unsupported",
                "normalize": self.normalize,
                "has_ref_text": request.ref_text is not None,
                "reference": str(request.reference_wav),
            },
        )
