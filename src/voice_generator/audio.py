"""Small WAV helpers built only on the stdlib, so the core package stays light."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WavInfo:
    path: Path
    channels: int
    sample_width: int  # bytes per sample
    sample_rate: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0


def wav_info(path: str | Path) -> WavInfo:
    p = Path(path)
    with wave.open(str(p), "rb") as w:
        return WavInfo(
            path=p,
            channels=w.getnchannels(),
            sample_width=w.getsampwidth(),
            sample_rate=w.getframerate(),
            frames=w.getnframes(),
        )


def check_reference(path: str | Path, *, min_s: float = 3.0, ideal_s: float = 10.0) -> list[str]:
    """Validate a voice-clone reference clip. Returns human-readable warnings.

    Raises FileNotFoundError / ValueError on hard failures; soft problems
    (too short, stereo, odd sample rate) come back as warnings so the user
    decides, because backends differ: Chatterbox wants ~10 s, CosyVoice /
    Qwen3-TTS can clone from ~3 s.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference clip not found: {p}")
    if p.suffix.lower() != ".wav":
        raise ValueError(f"reference clip must be a .wav file, got: {p.suffix}")
    info = wav_info(p)
    warnings: list[str] = []
    if info.duration_s < min_s:
        warnings.append(
            f"reference is only {info.duration_s:.1f}s "
            f"(< {min_s:.0f}s minimum) — cloning will be poor"
        )
    elif info.duration_s < ideal_s:
        warnings.append(
            f"reference is {info.duration_s:.1f}s — usable, "
            f"but ~{ideal_s:.0f}s gives a better clone"
        )
    if info.duration_s > 60:
        warnings.append(f"reference is {info.duration_s:.0f}s — trim to the cleanest 10–30 s")
    if info.channels > 1:
        warnings.append(
            "reference is stereo — most backends expect mono; it will be downmixed or rejected"
        )
    if info.sample_rate < 16000:
        warnings.append(f"reference sample rate is {info.sample_rate} Hz — 16 kHz+ recommended")
    return warnings


def write_pcm16_wav(path: str | Path, samples: object, sample_rate: int) -> None:
    """Write mono float samples (-1..1) as a PCM-16 WAV, numpy-free.

    Backends use this instead of torchaudio/soundfile so every chunk lands in
    exactly the same format and the stdlib WAV concatenation always works.
    Samples are clipped to [-1, 1] (a model overshooting slightly must not
    wrap around into loud clicks) and packed in one bulk ``struct.pack``
    call: the previous per-sample loop was the slowest code in the
    pipeline for long renders (millions of samples).
    """
    import struct

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clipped = [max(-1.0, min(1.0, float(s))) for s in samples]  # type: ignore[union-attr, arg-type]
    pcm = struct.pack(f"<{len(clipped)}h", *(int(v * 32767) for v in clipped))
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def concat_wavs(parts: list[Path], out: Path) -> WavInfo:
    """Concatenate PCM WAVs that share channels / width / rate. Stdlib only.

    Each part is read fully into memory in turn (not all at once); a
    10-minute mono 24 kHz render is ~28 MB, so this is fine for script
    lengths this tool targets, but it is not a streaming join.
    """
    if not parts:
        raise ValueError("no WAV parts to concatenate")
    first = wav_info(parts[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(first.channels)
        dst.setsampwidth(first.sample_width)
        dst.setframerate(first.sample_rate)
        for part in parts:
            info = wav_info(part)
            if (info.channels, info.sample_width, info.sample_rate) != (
                first.channels,
                first.sample_width,
                first.sample_rate,
            ):
                raise ValueError(f"WAV format mismatch in {part} — cannot concatenate")
            with wave.open(str(part), "rb") as src:
                dst.writeframes(src.readframes(src.getnframes()))
    return wav_info(out)
