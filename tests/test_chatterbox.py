"""Chatterbox backend tests with a fake `chatterbox` module — no GPU,
no downloads, no torch. The fake returns a torch-like tensor so we can
prove the backend writes PCM-16 (not the float WAV torchaudio.save
produced, which this project's stdlib reader rejects with
"unknown format: 3" — first seen on the real GPU run, 2026-11-09)."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from voice_generator.audio import wav_info
from voice_generator.backends.base import SynthesisRequest
from voice_generator.backends.chatterbox_backend import (
    ChatterboxBackend,
    _to_mono_samples,
)


class _FakeRow:
    ndim = 1

    def __init__(self, samples: list[float]) -> None:
        self._samples = samples

    def tolist(self) -> list[float]:
        return list(self._samples)


class _FakeTensor:
    ndim = 2

    def __init__(self, samples: list[float]) -> None:
        self._samples = samples

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def __getitem__(self, idx: int) -> _FakeRow:
        assert idx == 0
        return _FakeRow(self._samples)


class _FakeModel:
    instances: ClassVar[list["_FakeModel"]] = []

    def __init__(self, device: str) -> None:
        self.device = device
        self.sr = 24000
        self.generate_calls: list[dict[str, object]] = []
        _FakeModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, device: str) -> "_FakeModel":
        return cls(device)

    def generate(self, text: str, audio_prompt_path: str, **kwargs: object) -> _FakeTensor:
        self.generate_calls.append({"text": text, "ref": audio_prompt_path, **kwargs})
        return _FakeTensor([0.0] * 2400)  # 0.1 s of silence at 24 kHz


@pytest.fixture()
def fake_chatterbox(monkeypatch):
    _FakeModel.instances.clear()
    pkg = types.ModuleType("chatterbox")
    turbo_mod = types.ModuleType("chatterbox.tts_turbo")
    turbo_mod.ChatterboxTurboTTS = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chatterbox", pkg)
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", turbo_mod)
    return _FakeModel


def _req(tmp_path: Path, **over) -> SynthesisRequest:
    base = {
        "text": "Hello there.",
        "reference_wav": tmp_path / "ref.wav",
        "language": "en",
    }
    base.update(over)
    return SynthesisRequest(**base)


def test_to_mono_samples_handles_tensors_and_lists():
    assert _to_mono_samples(_FakeTensor([0.5, -0.5])) == [0.5, -0.5]
    assert _to_mono_samples([[0.25, 0.25]]) == [0.25, 0.25]
    assert _to_mono_samples([0.5]) == [0.5]


def test_synthesize_writes_pcm16_stdlib_can_read(fake_chatterbox, tmp_path: Path):
    backend = ChatterboxBackend()
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    info = wav_info(tmp_path / "out.wav")
    assert info.sample_width == 2  # PCM-16, not the float WAV torchaudio wrote
    assert info.sample_rate == 24000
    assert result.duration_s == pytest.approx(0.1, abs=0.01)
    assert result.settings["variant"] == "turbo"
    # "auto" device resolves to CPU here (no torch in the test env).
    assert fake_chatterbox.instances[0].device == "cpu"


def test_missing_reference_rejected(fake_chatterbox, tmp_path: Path):
    backend = ChatterboxBackend()
    with pytest.raises(ValueError, match="reference clip"):
        backend.synthesize(_req(tmp_path, reference_wav=None), tmp_path / "z.wav")
