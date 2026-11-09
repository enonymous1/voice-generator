"""Chatterbox (Resemble AI) backend — the v0.1 real engine.

Why first: MIT license (cleanest in the field), `pip install chatterbox-tts`,
~10 s reference clip, runs in ~1.5–6 GB VRAM (community figures) on an
RTX 3060 12 GB, and it is the most Windows-friendly of the researched models.
Caveat, disclosed in the README: every output carries Resemble's imperceptible
PerTh watermark.

torch / chatterbox are imported lazily inside synthesize(), never at module
import time, so the rest of the package works without them installed.
"""

from __future__ import annotations

from pathlib import Path

from voice_generator.audio import wav_info, write_pcm16_wav
from voice_generator.backends.base import (
    BaseBackend,
    SynthesisRequest,
    SynthesisResult,
    require_reference,
    resolve_device,
)

_VARIANTS = ("original", "turbo", "multilingual")


def _to_mono_samples(wav: object) -> list[float]:
    """Flatten a model output tensor (1, N) to a mono float sample list.

    The tensor may live on CUDA, so detach + move to CPU first. Plain
    nested lists (as in tests) pass through the same path.
    """
    tensor = wav.detach().cpu() if hasattr(wav, "detach") else wav  # type: ignore[attr-defined]
    if isinstance(tensor, (list, tuple)):
        # Plain nested list [[...]] (tests / non-torch callers): unwrap the
        # single channel the same way ndim == 2 does for tensors below.
        if tensor and isinstance(tensor[0], (list, tuple)):
            tensor = tensor[0]
        return [float(s) for s in tensor]
    if getattr(tensor, "ndim", 1) == 2:
        tensor = tensor[0]  # type: ignore[index]
    if hasattr(tensor, "tolist"):
        return [float(s) for s in tensor.tolist()]  # type: ignore[attr-defined]
    return [float(s) for s in tensor]  # type: ignore[union-attr]


class ChatterboxBackend(BaseBackend):
    name = "chatterbox"
    license = "MIT (outputs carry a PerTh watermark)"
    min_reference_s = 5.0

    def __init__(self, variant: str = "turbo", device: str = "auto") -> None:
        # "auto" (new default, was hardcoded "cuda") resolves at load time:
        # a CPU-only machine previously died inside from_pretrained with a
        # CUDA error; now it renders on CPU or fails with a clear message.
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.device = device
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            import chatterbox  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "chatterbox is not installed in this environment. Install it in its own venv:\n"
                '  python -m venv .venv-chatterbox\n'
                '  .venv-chatterbox\\Scripts\\activate   (Windows)\n'
                '  pip install -e ".[chatterbox]"'
            ) from exc
        device = resolve_device(self.device)
        if self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self._model = ChatterboxTurboTTS.from_pretrained(device=device)
        elif self.variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=device)
        return self._model

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        require_reference(request, "chatterbox")
        model = self._load()
        kwargs: dict[str, object] = {
            "exaggeration": request.exaggeration,
            "cfg_weight": request.cfg_weight,
        }
        if self.variant == "multilingual":
            kwargs["language_id"] = request.language
        wav = model.generate(  # type: ignore[attr-defined]
            request.text, audio_prompt_path=str(request.reference_wav), **kwargs
        )
        # Write PCM-16 via the shared writer, NOT torchaudio.save: torchaudio
        # writes a float32 tensor as a 32-bit float WAV (format tag 3), and
        # this project's stdlib WAV reader (wave.open) only reads integer
        # PCM — so the very next line (wav_info) failed with the cryptic
        # "unknown format: 3" on the first real GPU render (2026-11-09),
        # after the model had already rendered the audio. Qwen3/CosyVoice
        # already used the shared writer; Chatterbox was the odd one out.
        write_pcm16_wav(out_wav, _to_mono_samples(wav), model.sr)  # type: ignore[attr-defined]
        return SynthesisResult(
            wav_path=out_wav,
            backend=f"chatterbox:{self.variant}",
            duration_s=wav_info(out_wav).duration_s,
            settings={
                "variant": self.variant,
                "language": request.language,
                "exaggeration": request.exaggeration,
                "cfg_weight": request.cfg_weight,
                "reference": str(request.reference_wav),
            },
        )
