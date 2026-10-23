"""Qwen3 backend tests with a fake `qwen_tts` module — no GPU, no downloads,
no torch. The fake records calls so we can prove the prompt is built once
and reused, and that the transcript requirement is enforced."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from voice_generator.audio import wav_info
from voice_generator.backends.base import SynthesisRequest
from voice_generator.backends.qwen_backend import Qwen3Backend, map_language


class _FakeModel:
    instances: ClassVar[list["_FakeModel"]] = []

    def __init__(self) -> None:
        self.prompt_calls = 0
        self.generate_calls = 0
        _FakeModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        model = cls()
        model.model_id = model_id
        model.kwargs = kwargs
        return model

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.prompt_calls += 1
        return {"ref_audio": ref_audio, "ref_text": ref_text, "x": x_vector_only_mode}

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        self.generate_calls += 1
        # Upstream shape: ([waveform, ...], sample_rate) — wavs[0] is one waveform.
        return ([[0.0] * 2400], 24000)  # 0.1 s of silence at 24 kHz


@pytest.fixture()
def fake_qwen(monkeypatch):
    _FakeModel.instances.clear()
    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    # The backend imports torch only to map the dtype; stub it out.
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    torch.float32 = "fp32"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return _FakeModel


def _req(tmp_path: Path, **over) -> SynthesisRequest:
    base = {
        "text": "Hello there.",
        "reference_wav": tmp_path / "ref.wav",
        "language": "en",
        "ref_text": "The exact transcript of the reference clip.",
    }
    base.update(over)
    return SynthesisRequest(**base)


def test_language_mapping():
    assert map_language("en") == "English"
    assert map_language("es") == "Spanish"
    assert map_language("English") == "English"


def test_synthesize_writes_pcm_wav(fake_qwen, tmp_path: Path):
    backend = Qwen3Backend(size="0.6b")
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert result.backend == "qwen3:0.6b"
    assert result.duration_s == pytest.approx(0.1, abs=0.01)
    assert wav_info(tmp_path / "out.wav").sample_rate == 24000
    assert fake_qwen.instances[0].model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def test_prompt_built_once_and_reused(fake_qwen, tmp_path: Path):
    backend = Qwen3Backend()
    backend.synthesize(_req(tmp_path), tmp_path / "a.wav")
    backend.synthesize(_req(tmp_path, text="Second chunk."), tmp_path / "b.wav")
    model = fake_qwen.instances[0]
    assert model.prompt_calls == 1
    assert model.generate_calls == 2


def test_transcript_required_unless_x_vector(fake_qwen, tmp_path: Path):
    backend = Qwen3Backend()
    with pytest.raises(ValueError, match="transcript"):
        backend.synthesize(_req(tmp_path, ref_text=None), tmp_path / "x.wav")
    result = backend.synthesize(
        _req(tmp_path, ref_text=None, x_vector_only=True), tmp_path / "y.wav"
    )
    assert result.settings["x_vector_only"] is True


def test_missing_reference_rejected(fake_qwen, tmp_path: Path):
    backend = Qwen3Backend()
    with pytest.raises(ValueError, match="reference clip"):
        backend.synthesize(_req(tmp_path, reference_wav=None), tmp_path / "z.wav")
