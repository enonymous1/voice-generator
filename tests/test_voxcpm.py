"""VoxCPM2 backend tests with a fake `voxcpm` module — no GPU, no
downloads, no torch. The fake records calls so we can prove the two
clone modes pass the right arguments, and that the torch.compile /
optimize fallbacks behave (load-time and generate-time)."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from voice_generator.audio import wav_info
from voice_generator.backends.base import SynthesisRequest
from voice_generator.backends.voxcpm_backend import VoxCPM2Backend


class _FakeModel:
    instances: ClassVar[list["_FakeModel"]] = []
    load_calls: ClassVar[int] = 0
    # Class-level switches so a test can make one fake misbehave.
    fail_load_once_with: ClassVar[BaseException | None] = None
    fail_generate_once_with: ClassVar[BaseException | None] = None

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.optimize: bool | None = None
        self.tts_model = types.SimpleNamespace(sample_rate=48000)
        _FakeModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.load_calls += 1
        if cls.fail_load_once_with is not None:
            exc, cls.fail_load_once_with = cls.fail_load_once_with, None
            raise exc
        model = cls()
        model.model_id = model_id
        model.kwargs = kwargs
        model.optimize = kwargs.get("optimize")
        return model

    def generate(self, **kwargs):
        if type(self).fail_generate_once_with is not None:
            exc, type(self).fail_generate_once_with = (
                type(self).fail_generate_once_with,
                None,
            )
            raise exc
        self.generate_calls.append(kwargs)
        return [0.0] * 4800  # 0.1 s of silence at 48 kHz


@pytest.fixture()
def fake_voxcpm(monkeypatch):
    _FakeModel.instances.clear()
    _FakeModel.load_calls = 0
    _FakeModel.fail_load_once_with = None
    _FakeModel.fail_generate_once_with = None
    module = types.ModuleType("voxcpm")
    module.VoxCPM = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voxcpm", module)
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


def test_synthesize_writes_pcm_wav_at_48k(fake_voxcpm, tmp_path: Path):
    backend = VoxCPM2Backend()
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert result.backend == "voxcpm"
    assert result.duration_s == pytest.approx(0.1, abs=0.01)
    assert wav_info(tmp_path / "out.wav").sample_rate == 48000
    model = fake_voxcpm.instances[0]
    assert model.model_id == "openbmb/VoxCPM2"
    assert model.kwargs["load_denoiser"] is False
    assert model.kwargs["device"] == "cpu"  # no torch in this venv -> resolve_device -> cpu
    assert result.settings["model"] == "openbmb/VoxCPM2"
    assert result.settings["mode"] == "controllable"
    assert result.settings["cfg_value"] == 2.0
    assert result.settings["inference_timesteps"] == 10
    assert result.settings["optimize"] is True


def test_controllable_mode_needs_no_transcript(fake_voxcpm, tmp_path: Path):
    backend = VoxCPM2Backend()
    result = backend.synthesize(_req(tmp_path, ref_text=None), tmp_path / "out.wav")
    call = fake_voxcpm.instances[0].generate_calls[0]
    assert call["reference_wav_path"] == str(tmp_path / "ref.wav")
    assert "prompt_wav_path" not in call
    assert "prompt_text" not in call
    assert result.settings["has_ref_text"] is False


def test_ultimate_mode_passes_prompt_and_requires_transcript(fake_voxcpm, tmp_path: Path):
    backend = VoxCPM2Backend(mode="ultimate")
    with pytest.raises(ValueError, match="transcript"):
        backend.synthesize(_req(tmp_path, ref_text=None), tmp_path / "x.wav")
    result = backend.synthesize(_req(tmp_path), tmp_path / "y.wav")
    call = fake_voxcpm.instances[0].generate_calls[0]
    assert call["prompt_wav_path"] == str(tmp_path / "ref.wav")
    assert call["prompt_text"] == "The exact transcript of the reference clip."
    assert call["reference_wav_path"] == str(tmp_path / "ref.wav")
    assert result.settings["mode"] == "ultimate"
    assert result.settings["has_ref_text"] is True


def test_missing_reference_rejected(fake_voxcpm, tmp_path: Path):
    backend = VoxCPM2Backend()
    with pytest.raises(ValueError, match="reference clip"):
        backend.synthesize(_req(tmp_path, reference_wav=None), tmp_path / "z.wav")


def test_bad_mode_and_values_rejected_up_front(fake_voxcpm):
    with pytest.raises(ValueError, match="mode"):
        VoxCPM2Backend(mode="design")
    with pytest.raises(ValueError, match="cfg_value"):
        VoxCPM2Backend(cfg_value=0)
    with pytest.raises(ValueError, match="inference_timesteps"):
        VoxCPM2Backend(inference_timesteps=0)


def test_load_compile_error_retries_unoptimized(fake_voxcpm, tmp_path: Path):
    _FakeModel.fail_load_once_with = RuntimeError("Triton kernel failed to launch")
    backend = VoxCPM2Backend()
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert fake_voxcpm.load_calls == 2  # first load failed, second succeeded
    assert fake_voxcpm.instances[0].optimize is False
    assert result.settings["optimize"] is False


def test_load_non_compile_error_is_not_retried(fake_voxcpm, tmp_path: Path):
    _FakeModel.fail_load_once_with = RuntimeError("model id not found on the hub")
    backend = VoxCPM2Backend()
    with pytest.raises(RuntimeError, match="not found on the hub"):
        backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert len(fake_voxcpm.instances) == 0


def test_generate_compile_error_reloads_unoptimized_and_retries_chunk(
    fake_voxcpm, tmp_path: Path
):
    _FakeModel.fail_generate_once_with = RuntimeError("torch.compile inductor error")
    backend = VoxCPM2Backend()
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert len(fake_voxcpm.instances) == 2  # reloaded after the failed generate
    assert fake_voxcpm.instances[1].optimize is False
    assert result.settings["optimize"] is False
    assert result.duration_s == pytest.approx(0.1, abs=0.01)


def test_unknown_optimize_kwarg_retries_without_it(fake_voxcpm, tmp_path: Path):
    # An installed voxcpm that does not accept `optimize`: the backend
    # retries the load without the keyword instead of failing.
    orig = _FakeModel.__dict__["from_pretrained"]

    def _no_optimize(cls, model_id, **kwargs):
        if "optimize" in kwargs:
            raise TypeError("from_pretrained() got an unexpected keyword argument 'optimize'")
        return orig.__func__(cls, model_id, **kwargs)

    _FakeModel.from_pretrained = classmethod(_no_optimize)  # type: ignore[method-assign]
    try:
        backend = VoxCPM2Backend()
        result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    finally:
        _FakeModel.from_pretrained = orig  # type: ignore[method-assign]
    assert result.settings["optimize"] == "unsupported"
    assert "optimize" not in fake_voxcpm.instances[0].kwargs
