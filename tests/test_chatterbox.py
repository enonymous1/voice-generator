"""Chatterbox backend tests with a fake `chatterbox` module — no GPU,
no downloads, no torch. The fake returns a torch-like tensor so we can
prove the backend writes PCM-16 (not the float WAV torchaudio.save
produced, which this project's stdlib reader rejects with
"unknown format: 3" — first seen on the real GPU run, 2026-11-09).

The fakes have STRICT generate signatures (no **kwargs catch-all): the
old catch-all hid signature mismatches with the real engine, so each
variant's fake accepts exactly the arguments the backend passes for
that variant — Turbo takes only text + reference, original adds the two
knobs, multilingual adds language_id plus the knobs."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from voice_generator.audio import wav_info, write_pcm16_wav
from voice_generator.backends.base import SynthesisRequest, flatten_mono_samples
from voice_generator.backends.chatterbox_backend import ChatterboxBackend


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


class _FakeTurbo:
    instances: ClassVar[list["_FakeTurbo"]] = []

    def __init__(self, device: str) -> None:
        self.device = device
        self.sr = 24000
        self.generate_calls: list[dict[str, object]] = []
        _FakeTurbo.instances.append(self)

    @classmethod
    def from_pretrained(cls, device: str) -> "_FakeTurbo":
        return cls(device)

    def generate(self, text: str, audio_prompt_path: str) -> _FakeTensor:
        self.generate_calls.append({"text": text, "ref": audio_prompt_path})
        return _FakeTensor([0.0] * 2400)  # 0.1 s of silence at 24 kHz


class _FakeOriginal:
    instances: ClassVar[list["_FakeOriginal"]] = []

    def __init__(self, device: str) -> None:
        self.device = device
        self.sr = 24000
        self.generate_calls: list[dict[str, object]] = []
        _FakeOriginal.instances.append(self)

    @classmethod
    def from_pretrained(cls, device: str) -> "_FakeOriginal":
        return cls(device)

    def generate(
        self,
        text: str,
        audio_prompt_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> _FakeTensor:
        self.generate_calls.append(
            {
                "text": text,
                "ref": audio_prompt_path,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
        )
        return _FakeTensor([0.0] * 2400)


class _FakeMultilingual:
    instances: ClassVar[list["_FakeMultilingual"]] = []

    def __init__(self, device: str) -> None:
        self.device = device
        self.sr = 24000
        self.generate_calls: list[dict[str, object]] = []
        _FakeMultilingual.instances.append(self)

    @classmethod
    def from_pretrained(cls, device: str) -> "_FakeMultilingual":
        return cls(device)

    def generate(
        self,
        text: str,
        audio_prompt_path: str,
        language_id: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> _FakeTensor:
        self.generate_calls.append(
            {
                "text": text,
                "ref": audio_prompt_path,
                "language_id": language_id,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
        )
        return _FakeTensor([0.0] * 2400)


@pytest.fixture()
def fake_chatterbox(monkeypatch):
    _FakeTurbo.instances.clear()
    _FakeOriginal.instances.clear()
    _FakeMultilingual.instances.clear()
    pkg = types.ModuleType("chatterbox")
    turbo_mod = types.ModuleType("chatterbox.tts_turbo")
    turbo_mod.ChatterboxTurboTTS = _FakeTurbo  # type: ignore[attr-defined]
    orig_mod = types.ModuleType("chatterbox.tts")
    orig_mod.ChatterboxTTS = _FakeOriginal  # type: ignore[attr-defined]
    mtl_mod = types.ModuleType("chatterbox.mtl_tts")
    mtl_mod.ChatterboxMultilingualTTS = _FakeMultilingual  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chatterbox", pkg)
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", turbo_mod)
    monkeypatch.setitem(sys.modules, "chatterbox.tts", orig_mod)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl_mod)
    return _FakeTurbo


def _req(tmp_path: Path, **over) -> SynthesisRequest:
    # require_reference enforces the backend's minimum (5 s for
    # Chatterbox), so the fixture clip is a real 10 s WAV.
    ref = tmp_path / "ref.wav"
    write_pcm16_wav(ref, [0.0] * (10 * 16000), 16000)
    base = {
        "text": "Hello there.",
        "reference_wav": ref,
        "language": "en",
    }
    base.update(over)
    return SynthesisRequest(**base)


def test_flatten_mono_samples_handles_tensors_and_lists():
    assert list(flatten_mono_samples(_FakeTensor([0.5, -0.5]))) == [0.5, -0.5]
    assert list(flatten_mono_samples([[0.25, 0.25]])) == [0.25, 0.25]
    assert list(flatten_mono_samples([0.5])) == [0.5]


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


def test_turbo_does_not_pass_or_record_ignored_knobs(fake_chatterbox, tmp_path: Path):
    # Turbo ignores exaggeration/cfg_weight: they must reach neither the
    # engine call (the strict fake would raise TypeError) nor the
    # manifest settings, which would claim an effect they do not have.
    backend = ChatterboxBackend(variant="turbo")
    result = backend.synthesize(
        _req(tmp_path, exaggeration=0.7, cfg_weight=0.3), tmp_path / "out.wav"
    )
    assert fake_chatterbox.instances[0].generate_calls[0] == {
        "text": "Hello there.",
        "ref": str(tmp_path / "ref.wav"),
    }
    assert result.settings["exaggeration"] is None
    assert result.settings["cfg_weight"] is None
    assert "turbo ignores" in str(result.settings["note"])


def test_original_passes_and_records_knobs(fake_chatterbox, tmp_path: Path):
    backend = ChatterboxBackend(variant="original")
    result = backend.synthesize(
        _req(tmp_path, exaggeration=0.7, cfg_weight=0.3), tmp_path / "out.wav"
    )
    call = _FakeOriginal.instances[0].generate_calls[0]
    assert call["exaggeration"] == 0.7
    assert call["cfg_weight"] == 0.3
    assert result.settings["exaggeration"] == 0.7
    assert result.settings["cfg_weight"] == 0.3


def test_multilingual_passes_language_and_knobs(fake_chatterbox, tmp_path: Path):
    backend = ChatterboxBackend(variant="multilingual")
    result = backend.synthesize(
        _req(tmp_path, language="es", exaggeration=0.7, cfg_weight=0.3),
        tmp_path / "out.wav",
    )
    call = _FakeMultilingual.instances[0].generate_calls[0]
    assert call["language_id"] == "es"
    assert call["exaggeration"] == 0.7
    assert call["cfg_weight"] == 0.3
    assert result.settings["exaggeration"] == 0.7
    assert result.settings["cfg_weight"] == 0.3
