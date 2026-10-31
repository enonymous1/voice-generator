"""CosyVoice backend (FunAudioLLM) — wired in v0.3: CosyVoice 3 by default,
CosyVoice 2 selectable.

Verified against the upstream repo on 2026-10-22 (README + example.py,
github.com/FunAudioLLM/CosyVoice). CosyVoice is NOT a pip package, and this
backend is honest about that:

- Upstream install is a recursive git clone + a Conda env (Python 3.10) +
  `pip install -r requirements.txt`, plus sox on the system. This project is
  then installed INTO that env (`pip install -e .`, no extra).
- The backend finds the clone via --cosyvoice-home / $COSYVOICE_HOME, adds
  the repo root and its `third_party/Matcha-TTS` submodule to sys.path
  (exactly as upstream's example.py does), and loads
  `cosyvoice.cli.cosyvoice.AutoModel(model_dir=...)`.
- Model weights are a separate snapshot download, e.g.
  huggingface_hub.snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
  local_dir='pretrained_models/Fun-CosyVoice3-0.5B').

API quirks that shape this code, all from upstream example.py:
- Zero-shot cloning needs the reference transcript, like Qwen3. There is no
  transcript-free mode in the upstream API — the fallback here is
  cross-lingual mode (`inference_cross_lingual`), selected explicitly.
- For CosyVoice 3, the zero-shot `prompt_text` in the official example is
  NOT the bare transcript: it is
  "You are a helpful assistant.<|endofprompt|>" + transcript.
  CosyVoice 2 takes the bare transcript. This backend applies the v3 format
  for v3 models unless the transcript already contains <|endofprompt|>.
- Inference is a generator of {"tts_speech": tensor} segments at
  model.sample_rate; segments are concatenated and written as PCM-16 via
  the shared stdlib writer (not torchaudio, whose float WAVs the stdlib
  concatenation cannot read).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from voice_generator.audio import wav_info, write_pcm16_wav
from voice_generator.backends.base import (
    BaseBackend,
    SynthesisRequest,
    SynthesisResult,
    require_reference,
)

_DEFAULT_MODEL_DIRS = {
    "3": "pretrained_models/Fun-CosyVoice3-0.5B",
    "2": "pretrained_models/CosyVoice2-0.5B",
}

_SETUP_HINT = (
    "CosyVoice was not found. It is a repo + Conda env, not a pip package:\n"
    "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
    "  conda create -n cosyvoice -y python=3.10 && conda activate cosyvoice\n"
    "  cd CosyVoice && pip install -r requirements.txt   (plus sox on the system)\n"
    "  python -c \"from huggingface_hub import snapshot_download; "
    "snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', "
    "local_dir='pretrained_models/Fun-CosyVoice3-0.5B')\"\n"
    "  pip install -e <path-to-ss-voiceGenerator>   (into the cosyvoice env)\n"
    "Then pass --cosyvoice-home <CosyVoice clone> (or set COSYVOICE_HOME)."
)


def _flatten_samples(segment: object) -> list[float]:
    """Turn one tts_speech value (torch tensor (1, N), numpy array, or nested
    lists) into a flat list of mono float samples, torch-free."""
    obj = segment
    if hasattr(obj, "detach"):
        obj = obj.detach()  # type: ignore[attr-defined]
    if hasattr(obj, "cpu"):
        obj = obj.cpu()  # type: ignore[attr-defined]
    if hasattr(obj, "numpy"):
        obj = obj.numpy()  # type: ignore[attr-defined]
    if hasattr(obj, "tolist"):
        obj = obj.tolist()  # type: ignore[attr-defined]
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (list, tuple)):
            return [float(s) for s in obj[0]]  # first channel
        return [float(s) for s in obj]
    raise TypeError(f"unsupported tts_speech type: {type(obj).__name__}")


class CosyBackend(BaseBackend):
    name = "cosyvoice"
    license = "Apache-2.0"
    min_reference_s = 3.0

    def __init__(
        self,
        version: str = "3",
        cosyvoice_home: str | Path | None = None,
        model_dir: str | Path | None = None,
        mode: str = "zero_shot",
    ) -> None:
        if version not in _DEFAULT_MODEL_DIRS:
            raise ValueError(
                f"version must be one of {sorted(_DEFAULT_MODEL_DIRS)}, got {version!r}"
            )
        if mode not in ("zero_shot", "cross_lingual"):
            raise ValueError(f"mode must be zero_shot or cross_lingual, got {mode!r}")
        self.version = version
        self.mode = mode
        # Initialised here (it used to be created implicitly in _load via
        # getattr, which hid a real AttributeError-shaped bug class and
        # confused every reader about the object's state).
        self._model: object | None = None
        home = cosyvoice_home or os.environ.get("COSYVOICE_HOME")
        self.cosyvoice_home = Path(home) if home else None
        model = model_dir or os.environ.get("COSYVOICE_MODEL_DIR")
        if model:
            self.model_dir = Path(model)
        elif self.cosyvoice_home:
            self.model_dir = self.cosyvoice_home / _DEFAULT_MODEL_DIRS[version]
        else:
            self.model_dir = Path(_DEFAULT_MODEL_DIRS[version])

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        if self.cosyvoice_home is not None:
            # Upstream example.py does exactly this; the guard keeps a
            # second backend instance in the same process from duplicating
            # entries. Note this mutates global sys.path for the process
            # lifetime — acceptable for a CLI run, a known limitation for
            # library embedding (see REVIEW.md).
            for extra in (self.cosyvoice_home, self.cosyvoice_home / "third_party" / "Matcha-TTS"):
                if extra.exists() and str(extra) not in sys.path:
                    sys.path.insert(0, str(extra))
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise RuntimeError(_SETUP_HINT) from exc
        self._model = AutoModel(model_dir=str(self.model_dir))
        return self._model

    def _prompt_text(self, request: SynthesisRequest) -> str:
        # Explicit check, not assert (see Qwen3 backend): user input, and
        # asserts vanish under python -O.
        if request.ref_text is None:
            raise ValueError("cosyvoice prompt needs a reference transcript")
        if self.version == "3" and "<|endofprompt|>" not in request.ref_text:
            return f"You are a helpful assistant.<|endofprompt|>{request.ref_text}"
        return request.ref_text

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        require_reference(request, "cosyvoice")
        model = self._load()
        if self.mode == "zero_shot":
            if not request.ref_text:
                raise ValueError(
                    "cosyvoice zero-shot needs the reference transcript: pass --ref-text \"...\" "
                    "(or --ref-text-file). CosyVoice has no transcript-free clone mode; "
                    "for transcript-free cloning use --cosy-mode cross_lingual."
                )
            segments = model.inference_zero_shot(  # type: ignore[attr-defined]
                request.text,
                self._prompt_text(request),
                str(request.reference_wav),
                stream=False,
            )
        else:
            segments = model.inference_cross_lingual(  # type: ignore[attr-defined]
                request.text, str(request.reference_wav)
            )
        samples: list[float] = []
        for seg in segments:
            samples.extend(_flatten_samples(seg["tts_speech"]))
        if not samples:
            raise RuntimeError("cosyvoice produced no audio segments")
        sample_rate = int(model.sample_rate)  # type: ignore[attr-defined]
        write_pcm16_wav(out_wav, samples, sample_rate)
        return SynthesisResult(
            wav_path=out_wav,
            backend=f"cosyvoice:{self.version}",
            duration_s=wav_info(out_wav).duration_s,
            settings={
                "version": self.version,
                "mode": self.mode,
                "model_dir": str(self.model_dir),
                "has_ref_text": request.ref_text is not None,
                "reference": str(request.reference_wav),
            },
        )
