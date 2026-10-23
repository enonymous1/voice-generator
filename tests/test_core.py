import json
import struct
import wave
from pathlib import Path

import pytest

from voice_generator.audio import check_reference, concat_wavs, wav_info
from voice_generator.pipeline import synthesize_script
from voice_generator.text import chunk_script


def _make_wav(path: Path, seconds: float = 1.0, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return path


def test_chunking_respects_max_and_keeps_sentences():
    text = "One two three. Four five six! Seven eight nine? " * 10
    chunks = chunk_script(text.strip(), max_chars=60)
    assert chunks and all(len(c) <= 60 for c in chunks)
    assert "One two three." in chunks[0]


def test_chunking_empty():
    assert chunk_script("   ") == []


def test_check_reference_warns_on_short(tmp_path: Path):
    ref = _make_wav(tmp_path / "short.wav", seconds=1.0)
    warnings = check_reference(ref)
    assert any("only 1.0s" in w for w in warnings)


def test_check_reference_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        check_reference(tmp_path / "nope.wav")


def test_concat_wavs(tmp_path: Path):
    a = _make_wav(tmp_path / "a.wav", seconds=0.5)
    b = _make_wav(tmp_path / "b.wav", seconds=0.25)
    out = concat_wavs([a, b], tmp_path / "both.wav")
    assert out.duration_s == pytest.approx(0.75, abs=0.01)
    assert out.frames == 18000
    assert wav_info(out.path).frames == 18000


def test_mock_pipeline_end_to_end(tmp_path: Path):
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    out = tmp_path / "out" / "ep1.wav"
    result = synthesize_script(
        text="Hello world. This is a plumbing test, not real speech.",
        reference_wav=ref,
        out_wav=out,
        backend_name="mock",
        max_chars=30,
    )
    assert out.exists() and result.chunks >= 2 and result.duration_s > 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "mock"
    assert len(manifest["chunks"]) == result.chunks
