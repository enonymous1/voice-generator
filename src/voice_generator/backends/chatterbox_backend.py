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

from voice_generator.audio import wav_info
from voice_generator.backends.base import BaseBackend, SynthesisRequest, SynthesisResult

_VARIANTS = ("original", "turbo", "multilingual")


class ChatterboxBackend(BaseBackend):
    name = "chatterbox"
    license = "MIT (outputs carry a PerTh watermark)"
    min_reference_s = 5.0

    def __init__(self, variant: str = "turbo", device: str = "cuda") -> None:
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
        if self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        elif self.variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        else:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=self.device)
        return self._model

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        if request.reference_wav is None:
            raise ValueError("chatterbox needs a reference clip (--ref your_10s_ref_clip.wav)")
        model = self._load()
        import torchaudio

        kwargs: dict[str, object] = {
            "exaggeration": request.exaggeration,
            "cfg_weight": request.cfg_weight,
        }
        if self.variant == "multilingual":
            kwargs["language_id"] = request.language
        wav = model.generate(  # type: ignore[attr-defined]
            request.text, audio_prompt_path=str(request.reference_wav), **kwargs
        )
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_wav), wav, model.sr)  # type: ignore[attr-defined]
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
