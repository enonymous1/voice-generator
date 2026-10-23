"""CosyVoice backend tests with a fake `cosyvoice` module chain — no Conda
env, no repo clone, no torch. The fake records the prompt text so we can
prove the CosyVoice 3 <|endofprompt|> format is applied (and not for v2),
the transcript requirement is enforced, and segments are concatenated."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from voice_generator.audio import wav_info
from voice_generator.backends.base import SynthesisRequest
from voice_generator.backends.cosy_backend import CosyBackend, _flatten_samples


class _FakeCosyModel:
    instances: ClassVar[list["_FakeCosyModel"]] = []

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.sample_rate = 24000
        self.zero_shot_calls: list[tuple[str, str, str]] = []
        self.cross_lingual_calls: list[tuple[str, str]] = []
        _FakeCosyModel.instances.append(self)

    def inference_zero_shot(self, text, prompt_text, prompt_wav, stream=False):
        self.zero_shot_calls.append((text, prompt_text, prompt_wav))
        yield {"tts_speech": [[0.0] * 1200]}
        yield {"tts_speech": [[0.0] * 1200]}

    def inference_cross_lingual(self, text, prompt_wav):
        self.cross_lingual_calls.append((text, prompt_wav))
        yield {"tts_speech": [[0.0] * 2400]}


@pytest.fixture()
def fake_cosy(monkeypatch):
    _FakeCosyModel.instances.clear()
    pkg = types.ModuleType("cosyvoice")
    cli_pkg = types.ModuleType("cosyvoice.cli")
    cli_mod = types.ModuleType("cosyvoice.cli.cosyvoice")
    cli_mod.AutoModel = _FakeCosyModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cosyvoice", pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", cli_pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", cli_mod)
    return _FakeCosyModel


def _req(tmp_path: Path, **over) -> SynthesisRequest:
    base = {
        "text": "Hello there.",
        "reference_wav": tmp_path / "ref.wav",
        "language": "en",
        "ref_text": "The exact transcript of the reference clip.",
    }
    base.update(over)
    return SynthesisRequest(**base)  # type: ignore[arg-type]


def test_flatten_samples_handles_channels_and_lists():
    assert _flatten_samples([[0.5, -0.5]]) == [0.5, -0.5]
    assert _flatten_samples([0.25, 0.25]) == [0.25, 0.25]


def test_v3_prompt_format_and_segment_concat(fake_cosy, tmp_path: Path):
    backend = CosyBackend(version="3", model_dir=tmp_path / "model")
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    model = fake_cosy.instances[0]
    assert model.zero_shot_calls[0][1] == (
        "You are a helpful assistant.<|endofprompt|>The exact transcript of the reference clip."
    )
    # two 1200-sample segments at 24 kHz = 0.1 s total
    assert result.duration_s == pytest.approx(0.1, abs=0.01)
    assert wav_info(tmp_path / "out.wav").sample_rate == 24000


def test_v2_prompt_is_bare_transcript(fake_cosy, tmp_path: Path):
    backend = CosyBackend(version="2", model_dir=tmp_path / "model")
    backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert fake_cosy.instances[0].zero_shot_calls[0][1] == "The exact transcript of the reference clip."


def test_zero_shot_requires_transcript(fake_cosy, tmp_path: Path):
    backend = CosyBackend(model_dir=tmp_path / "model")
    with pytest.raises(ValueError, match="transcript"):
        backend.synthesize(_req(tmp_path, ref_text=None), tmp_path / "x.wav")


def test_cross_lingual_needs_no_transcript(fake_cosy, tmp_path: Path):
    backend = CosyBackend(mode="cross_lingual", model_dir=tmp_path / "model")
    result = backend.synthesize(_req(tmp_path, ref_text=None), tmp_path / "y.wav")
    assert result.settings["mode"] == "cross_lingual"
    assert len(fake_cosy.instances[0].cross_lingual_calls) == 1


def test_missing_package_gives_setup_hint(tmp_path: Path, monkeypatch):
    monkeypatch.delitem(sys.modules, "cosyvoice.cli.cosyvoice", raising=False)
    monkeypatch.delitem(sys.modules, "cosyvoice.cli", raising=False)
    monkeypatch.delitem(sys.modules, "cosyvoice", raising=False)
    backend = CosyBackend(model_dir=tmp_path / "model")
    with pytest.raises(RuntimeError, match="git clone --recursive"):
        backend.synthesize(_req(tmp_path), tmp_path / "z.wav")
