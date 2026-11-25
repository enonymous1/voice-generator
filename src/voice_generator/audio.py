"""Small WAV helpers built only on the stdlib, so the core package stays light."""

from __future__ import annotations

import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WavInfo:
    """Header facts for one WAV file: format plus frame count.

    ``duration_s`` derives from ``frames / sample_rate``; read the header
    once and pass the info around rather than re-opening the file per
    question — several callers already know the values and construct this
    directly instead of re-reading.
    """

    path: Path
    channels: int
    sample_width: int  # bytes per sample
    sample_rate: int
    frames: int

    @property
    def duration_s(self) -> float:
        """Seconds of audio in this file (``frames / sample_rate``)."""
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0


def wav_info(path: str | Path) -> WavInfo:
    """Read a WAV header. Raises the stdlib's wave.Error, with help attached.

    The stdlib `wave` module only reads integer PCM. The error it raises
    for anything else — `unknown format: 3` for a 32-bit float WAV, which
    is what torchaudio writes for float32 tensors and what some recorders
    produce — names a number, not a cause. The re-raised message keeps
    the original text and adds the conversion command, so the user sees a
    fix instead of a format tag.
    """
    p = Path(path)
    try:
        with wave.open(str(p), "rb") as w:
            return WavInfo(
                path=p,
                channels=w.getnchannels(),
                sample_width=w.getsampwidth(),
                sample_rate=w.getframerate(),
                frames=w.getnframes(),
            )
    except wave.Error as exc:
        raise wave.Error(
            f"{p}: {exc} — this tool only reads PCM WAVs; "
            f"convert first, e.g. ffmpeg -i {p} -c:a pcm_s16le {p.with_stem(p.stem + '_pcm16')}"
        ) from exc


def check_reference(
    path: str | Path, *, min_s: float = 3.0, ideal_s: float = 10.0, info: WavInfo | None = None
) -> list[str]:
    """Validate a voice-clone reference clip. Returns human-readable warnings.

    Raises FileNotFoundError / ValueError on hard failures; soft problems
    (too short, stereo, odd sample rate) come back as warnings so the user
    decides, because backends differ: Chatterbox wants ~10 s, CosyVoice /
    Qwen3-TTS can clone from ~3 s. Pass ``info`` when the header was already
    read (e.g. by the CLI, which prints it) so the file is not opened twice.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference clip not found: {p}")
    if p.suffix.lower() != ".wav":
        raise ValueError(f"reference clip must be a .wav file, got: {p.suffix}")
    if info is None:
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


_WRITE_BLOCK = 65_536  # samples packed per struct.pack call (see write_pcm16_wav)


def write_pcm16_wav(path: str | Path, samples: Iterable[float], sample_rate: int) -> None:
    """Write mono float samples (-1..1) as a PCM-16 WAV, numpy-free.

    Backends use this instead of torchaudio/soundfile so every chunk lands in
    exactly the same format and the stdlib WAV concatenation always works.
    Samples are clipped to [-1, 1] (a model overshooting slightly must not
    wrap around into loud clicks). Any iterable of numbers is accepted,
    including numpy arrays and torch tensors — ``float(s)`` is applied per
    element, so no conversion step is needed before calling this.

    Samples are packed in blocks of ``_WRITE_BLOCK`` instead of one bulk
    ``struct.pack`` over the whole render: the bulk version built a list of
    every clipped sample plus a star-unpacked tuple of ints, holding ~30x
    the output size in transient objects for a long render. Blocking keeps
    peak memory at one block plus the output bytes, with identical output.
    """
    import struct

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        block: list[int] = []
        for value in samples:
            block.append(int(max(-1.0, min(1.0, float(value))) * 32767))
            if len(block) == _WRITE_BLOCK:
                w.writeframes(struct.pack(f"<{_WRITE_BLOCK}h", *block))
                block.clear()
        if block:
            w.writeframes(struct.pack(f"<{len(block)}h", *block))


def trim_wav(src: str | Path, dst: str | Path, *, start_s: float, end_s: float) -> WavInfo:
    """Keep the audio between ``start_s`` and ``end_s`` (seconds), sample-accurate.

    Stdlib only, and the slice keeps the source's channels / width / rate,
    so a trimmed reference needs no conversion — intake already normalized
    it to 16 kHz mono PCM-16. Raises ValueError on a bad range.
    """
    p = Path(src)
    info = wav_info(p)  # raises wave.Error (with help) on an unreadable file
    if start_s < 0:
        raise ValueError(f"trim start must be >= 0, got {start_s}")
    if end_s <= start_s:
        raise ValueError(f"trim end ({end_s}s) must be after start ({start_s}s)")
    if end_s > info.duration_s:
        raise ValueError(
            f"trim end ({end_s}s) is past the end of the clip ({info.duration_s:.1f}s)"
        )
    start_frame = int(start_s * info.sample_rate)
    end_frame = int(end_s * info.sample_rate)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "rb") as src_w:
        src_w.setpos(start_frame)
        frames = src_w.readframes(end_frame - start_frame)
    with wave.open(str(dst_p), "wb") as dst_w:
        dst_w.setnchannels(info.channels)
        dst_w.setsampwidth(info.sample_width)
        dst_w.setframerate(info.sample_rate)
        dst_w.writeframes(frames)
    # The output's format is already known — re-reading the file just
    # written would only recover what this function computed.
    return WavInfo(
        path=dst_p,
        channels=info.channels,
        sample_width=info.sample_width,
        sample_rate=info.sample_rate,
        frames=end_frame - start_frame,
    )


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
        for index, part in enumerate(parts):
            # The first part's header was already read above; opening it
            # again would only recover the same values.
            info = first if index == 0 else wav_info(part)
            if (info.channels, info.sample_width, info.sample_rate) != (
                first.channels,
                first.sample_width,
                first.sample_rate,
            ):
                raise ValueError(f"WAV format mismatch in {part} — cannot concatenate")
            with wave.open(str(part), "rb") as src:
                dst.writeframes(src.readframes(src.getnframes()))
    return wav_info(out)
