"""Mock backend: writes a real, valid WAV (a soft tone) with a duration
estimated from the text length. It is NOT speech.

Its job is plumbing: verify the CLI, chunking, concatenation and manifest
end-to-end on any machine — including one with no GPU — and hand Stage 2 a
known-duration WAV to test lip-sync timing against.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from voice_generator.audio import wav_info
from voice_generator.backends.base import BaseBackend, SynthesisRequest, SynthesisResult

_SAMPLE_RATE = 24000
_CHARS_PER_SECOND = 14.0  # ~average English speaking rate


class MockBackend(BaseBackend):
    name = "mock"
    license = "MIT (project code; generates a tone, not speech)"

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        duration = max(0.4, len(request.text) / _CHARS_PER_SECOND)
        frames = int(duration * _SAMPLE_RATE)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            pcm = bytearray()
            for i in range(frames):
                sample = int(6000 * math.sin(2 * math.pi * 220 * i / _SAMPLE_RATE))
                pcm += struct.pack("<h", sample)
            w.writeframes(bytes(pcm))
        return SynthesisResult(
            wav_path=out_wav,
            backend=self.name,
            duration_s=wav_info(out_wav).duration_s,
            settings={"note": "tone only — not speech", "language": request.language},
        )
