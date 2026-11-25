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

_VARIANTS = ("original", "turbo", "multilingual")


class ChatterboxBackend(BaseBackend):
    """Chatterbox TTS in one of its three variants.

    ``turbo`` (default) is the fast path; ``original`` honours the
    ``exaggeration`` / ``cfg_weight`` knobs; ``multilingual`` adds a
    ``language_id``. Turbo ignores both knobs, so they are neither passed
    to its ``generate`` nor recorded as effective settings — a manifest
    claiming "rendered with exaggeration 0.7" for a Turbo render would
    assert reproducibility it does not have.
    """

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
        """Import the engine and load the model for this variant, once.

        The first call downloads multi-GB weights (``from_pretrained``)
        and caches the model on ``self._model``; later calls return the
        cached model. Raises ``BackendNotInstalledError`` (not a bare
        ``RuntimeError``) when the engine package is missing, so the CLI
        can tell "not installed — here is the fix" apart from a real
        engine failure.
        """
        if self._model is not None:
            return self._model
        try:
            import chatterbox  # noqa: F401
        except ImportError as exc:
            raise BackendNotInstalledError(
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
        """Render one chunk with the selected variant.

        The reference clip is validated (exists, at least
        ``min_reference_s``) before the model loads, so a bad reference
        fails before a multi-GB download starts. Turbo renders ignore
        ``exaggeration`` / ``cfg_weight`` — see the class docstring.
        """
        require_reference(request, "chatterbox", self.min_reference_s)
        model = self._load()
        kwargs: dict[str, object] = {}
        if self.variant != "turbo":
            # Only original (and, as shipped, multilingual) honour these
            # knobs; passing them to Turbo would be a lie in the manifest.
            kwargs["exaggeration"] = request.exaggeration
            kwargs["cfg_weight"] = request.cfg_weight
        if self.variant == "multilingual":
            kwargs["language_id"] = request.language
        wav = model.generate(  # type: ignore[attr-defined]
            request.text, audio_prompt_path=str(request.reference_wav), **kwargs
        )
        # Write PCM-16 via the shared writer, NOT torchaudio.save: torchaudio
        # writes a float32 tensor as a 32-bit float WAV (format tag 3), and
        # this project's stdlib WAV reader (wave.open) only reads integer
        # PCM — so the first real GPU render (2026-11-09) failed with the
        # cryptic "unknown format: 3" after the model had already rendered
        # the audio. Qwen3/CosyVoice already used the shared writer;
        # Chatterbox was the odd one out.
        samples = flatten_mono_samples(wav)
        sample_rate = int(model.sr)  # type: ignore[attr-defined]
        write_pcm16_wav(out_wav, samples, sample_rate)
        settings: dict[str, object] = {
            "variant": self.variant,
            "language": request.language,
            "reference": str(request.reference_wav),
        }
        if self.variant == "turbo":
            settings["exaggeration"] = None
            settings["cfg_weight"] = None
            settings["note"] = "turbo ignores exaggeration/cfg_weight"
        else:
            settings["exaggeration"] = request.exaggeration
            settings["cfg_weight"] = request.cfg_weight
        return SynthesisResult(
            wav_path=out_wav,
            backend=f"chatterbox:{self.variant}",
            duration_s=len(samples) / sample_rate,
            settings=settings,
        )
