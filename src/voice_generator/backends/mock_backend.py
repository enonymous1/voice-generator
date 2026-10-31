"""Mock backend: writes a real, valid WAV (a soft tone) with a duration
estimated from the text length. It is NOT speech.

Its job is plumbing: verify the CLI, chunking, concatenation and manifest
end-to-end on any machine — including one with no GPU — and hand Stage 2 a
known-duration WAV to test lip-sync timing against. It writes through the
shared PCM-16 writer like every real backend, so the mock exercises the
exact joining path production renders use.
"""

from __future__ import annotations

import math
from pathlib import Path

from voice_generator.audio import wav_info, write_pcm16_wav
from voice_generator.backends.base import BaseBackend, SynthesisRequest, SynthesisResult

_SAMPLE_RATE = 24000
_CHARS_PER_SECOND = 14.0  # ~average English speaking rate


class MockBackend(BaseBackend):
    name = "mock"
    license = "MIT (project code; generates a tone, not speech)"

    def synthesize(self, request: SynthesisRequest, out_wav: Path) -> SynthesisResult:
        """Write a 220 Hz tone lasting ``len(text) / 14`` seconds (min 0.4 s)."""
        duration = max(0.4, len(request.text) / _CHARS_PER_SECOND)
        frames = int(duration * _SAMPLE_RATE)
        samples = [0.18 * math.sin(2 * math.pi * 220 * i / _SAMPLE_RATE) for i in range(frames)]
        write_pcm16_wav(out_wav, samples, _SAMPLE_RATE)
        return SynthesisResult(
            wav_path=out_wav,
            backend=self.name,
            duration_s=wav_info(out_wav).duration_s,
            settings={"note": "tone only — not speech", "language": request.language},
        )
