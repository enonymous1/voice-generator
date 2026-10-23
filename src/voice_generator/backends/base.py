"""The backend contract every TTS engine implements.

One interface, interchangeable engines: that is what lets Stage 1 swap
Chatterbox for CosyVoice 3 / Qwen3-TTS later without touching the CLI,
the chunking, the manifest, or Stage 2 (which only ever sees a WAV file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    reference_wav: Path | None  # None = backend's default voice, if it has one
    language: str = "en"
    exaggeration: float = 0.5  # Chatterbox-style emotion intensity; ignored if unsupported
    cfg_weight: float = 0.5
    ref_text: str | None = None  # exact transcript of the reference clip (Qwen3 ICL cloning)
    x_vector_only: bool = False  # speaker-embedding-only cloning where supported (Qwen3)


@dataclass(frozen=True)
class SynthesisResult:
    wav_path: Path
    backend: str
    duration_s: float
    settings: dict[str, object] = field(default_factory=dict)


class BaseBackend:
    name: str = "base"
    license: str = "unknown"
    min_reference_s: float = 3.0

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        raise NotImplementedError
