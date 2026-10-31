"""Qwen3-TTS backend (Alibaba Qwen) — wired in v0.2, at the user's choice.

Verified against the upstream README on 2026-10-22
(github.com/QwenLM/Qwen3-TTS): package `qwen-tts` on PyPI, Python 3.12
recommended, cloning lives in the *Base* checkpoints
(Qwen/Qwen3-TTS-12Hz-1.7B-Base and ...-0.6B-Base, Apache-2.0).

Voice-clone specifics that shape this code:
- Full (ICL) cloning needs the reference clip AND its exact transcript
  (`ref_text`). Without a transcript you must use `x_vector_only_mode=True`,
  which clones from the speaker embedding only and is lower quality.
- The clone prompt is built once per reference with
  `create_voice_clone_prompt(...)` and reused across chunks via
  `generate_voice_clone(..., voice_clone_prompt=...)` — the pipeline creates
  one backend instance per run, so the prompt is cached here and the feature
  extraction cost is paid once per script, not once per chunk.
- `generate_voice_clone` takes full language names ("English"), not ISO
  codes, so the CLI's codes are mapped below.
- Upstream recommends FlashAttention 2, but that is painful on Windows;
  the default attention here is "sdpa", with "flash_attention_2" opt-in.
- Output WAVs are written as PCM-16 with the stdlib (soundfile not needed):
  our WAV concatenation requires PCM, and the model's sample rate is used.
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

_MODELS = {
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    "auto": "Auto",
}


def map_language(language: str) -> str:
    return _LANGUAGES.get(language.lower(), language[:1].upper() + language[1:])


class Qwen3Backend(BaseBackend):
    name = "qwen3"
    license = "Apache-2.0"
    min_reference_s = 3.0

    def __init__(
        self,
        size: str = "1.7b",
        device: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
    ) -> None:
        if size not in _MODELS:
            raise ValueError(f"size must be one of {sorted(_MODELS)}, got {size!r}")
        # Fail fast on a bad dtype instead of a bare KeyError inside _load,
        # where the real cause would be buried under a model-load traceback.
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"dtype must be bfloat16, float16 or float32, got {dtype!r}")
        self.size = size
        self.model_id = _MODELS[size]
        self.device = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self._model: object | None = None
        self._prompt_cache: dict[tuple[str, str | None, bool], object] = {}

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "qwen-tts is not installed in this environment. Install it in its own venv "
                "(upstream recommends Python 3.12):\n"
                "  py -3.12 -m venv .venv-qwen3\n"
                "  .venv-qwen3\\Scripts\\activate   (Windows)\n"
                '  pip install -e ".[qwen3]"'
            ) from exc
        import torch

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype_name]
        # "auto" resolves here, at load time (not in __init__), so a
        # backend constructed in a test or a CPU-only env still works and
        # the device decision sees the real torch of this environment.
        kwargs: dict[str, object] = {"device_map": resolve_device(self.device), "dtype": dtype}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
        return self._model

    def _prompt(self, request: SynthesisRequest) -> object:
        # Explicit check, not assert: asserts vanish under python -O and
        # this is a user-input validation, not an internal invariant.
        ref = require_reference(request, "qwen3")
        key = (str(ref), request.ref_text, request.x_vector_only)
        if key not in self._prompt_cache:
            model = self._load()
            self._prompt_cache[key] = model.create_voice_clone_prompt(  # type: ignore[attr-defined]
                ref_audio=str(request.reference_wav),
                ref_text=request.ref_text,
                x_vector_only_mode=request.x_vector_only,
            )
        return self._prompt_cache[key]

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        require_reference(request, "qwen3")
        if not request.x_vector_only and not request.ref_text:
            raise ValueError(
                "qwen3 full cloning needs the reference transcript: pass --ref-text \"...\" "
                "(or --ref-text-file). Without a transcript, use --x-vector-only "
                "(speaker-embedding only, lower quality)."
            )
        model = self._load()
        prompt = self._prompt(request)
        wavs, sr = model.generate_voice_clone(  # type: ignore[attr-defined]
            text=request.text,
            language=map_language(request.language),
            voice_clone_prompt=prompt,
        )
        write_pcm16_wav(out_wav, wavs[0], int(sr))
        return SynthesisResult(
            wav_path=out_wav,
            backend=f"qwen3:{self.size}",
            duration_s=wav_info(out_wav).duration_s,
            settings={
                "model": self.model_id,
                "language": map_language(request.language),
                "x_vector_only": request.x_vector_only,
                "has_ref_text": request.ref_text is not None,
                "reference": str(request.reference_wav),
            },
        )
