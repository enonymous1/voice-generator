"""Tests for the stdlib WAV helpers in voice_generator.audio."""

import math

import pytest

from voice_generator.audio import trim_wav, wav_info, write_pcm16_wav


def _tone_wav(path, seconds: float = 12.0, rate: int = 16000) -> None:
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
    write_pcm16_wav(path, samples, rate)


def test_trim_wav_keeps_slice(tmp_path):
    src = tmp_path / "src.wav"
    _tone_wav(src)
    out = tmp_path / "trim.wav"
    info = trim_wav(src, out, start_s=2.0, end_s=5.0)
    assert info.duration_s == pytest.approx(3.0, abs=0.01)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.sample_width == 2


def test_trim_wav_first_ten_seconds(tmp_path):
    src = tmp_path / "src.wav"
    _tone_wav(src)
    out = tmp_path / "trim.wav"
    info = trim_wav(src, out, start_s=0.0, end_s=10.0)
    assert info.duration_s == pytest.approx(10.0, abs=0.01)


def test_trim_wav_rejects_bad_ranges(tmp_path):
    src = tmp_path / "src.wav"
    _tone_wav(src)
    out = tmp_path / "trim.wav"
    with pytest.raises(ValueError, match=">= 0"):
        trim_wav(src, out, start_s=-1.0, end_s=5.0)
    with pytest.raises(ValueError, match="must be after start"):
        trim_wav(src, out, start_s=5.0, end_s=2.0)
    with pytest.raises(ValueError, match="past the end"):
        trim_wav(src, out, start_s=0.0, end_s=13.0)
    assert not out.exists()  # no partial file on a failed trim


def test_trim_wav_source_stays_untouched(tmp_path):
    src = tmp_path / "src.wav"
    _tone_wav(src)
    before = wav_info(src)
    trim_wav(src, tmp_path / "trim.wav", start_s=0.0, end_s=10.0)
    assert wav_info(src).frames == before.frames
