"""Tests pinning the 2026-11-25 review fixes for the backends.

Each test names the defect it guards, mirroring test_review.py: a
future refactor that reopens one of them fails loudly here instead of
silently on a user's machine. All fakes are local to this file so the
fix tests do not depend on the other backend test files' fixtures.
"""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from voice_generator.audio import wav_info, write_pcm16_wav
from voice_generator.backends import BACKENDS, get_backend
from voice_generator.backends.base import (
    SynthesisRequest,
    flatten_mono_samples,
    require_reference,
    resolve_device,
)
from voice_generator.backends.chatterbox_backend import ChatterboxBackend
from voice_generator.backends.cosy_backend import CosyBackend
from voice_generator.backends.mock_backend import MockBackend
from voice_generator.backends.qwen_backend import Qwen3Backend, map_language
from voice_generator.backends.voxcpm_backend import VoxCPM2Backend, _looks_like_compile_error


def _wav(tmp_path: Path, name: str, seconds: float) -> Path:
    """A real WAV of the given duration (silence, 16 kHz mono)."""
    p = tmp_path / name
    write_pcm16_wav(p, [0.0] * int(seconds * 16000), 16000)
    return p


def _req(tmp_path: Path, **over) -> SynthesisRequest:
    ref = _wav(tmp_path, "ref.wav", 10.0)
    base = {
        "text": "Hello there.",
        "reference_wav": ref,
        "language": "en",
        "ref_text": "The exact transcript of the reference clip.",
    }
    base.update(over)
    return SynthesisRequest(**base)


# --- resolve_device: MPS (fix 1) ---------------------------------------


def _fake_torch(*, cuda: bool, mps: bool):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )


def test_resolve_device_auto_returns_mps_on_apple_silicon(monkeypatch, capsys):
    # Old code never returned "mps": Apple Silicon silently fell back to
    # CPU even though the engines would use MPS.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    assert resolve_device("auto") == "mps"
    assert capsys.readouterr().err == ""


def test_resolve_device_auto_warns_without_cuda_or_mps(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    monkeypatch.setattr(sys, "platform", "linux")
    assert resolve_device("auto") == "cpu"
    err = capsys.readouterr().err
    assert "CUDA is not available" in err
    assert "cu121" in err


def test_resolve_device_warning_on_macos_names_mps_not_cuda(monkeypatch, capsys):
    # The old warning told a Mac user to install a CUDA torch build,
    # which is impossible on Apple Silicon.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert resolve_device("auto") == "cpu"
    err = capsys.readouterr().err
    assert "MPS" in err
    assert "cu121" not in err


# --- resolve_device: explicit cuda validation (fix 3) --------------------


def _fake_torch_cuda(available: bool, count: int = 1):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: available, device_count=lambda: count
        )
    )


def test_resolve_device_explicit_cuda_rejected_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_cuda(False))
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda:0")


def test_resolve_device_explicit_cuda_index_checked(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_cuda(True, count=1))
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("cuda") == "cuda"
    with pytest.raises(ValueError, match="only 1 CUDA device"):
        resolve_device("cuda:1")
    with pytest.raises(ValueError, match="non-numeric index"):
        resolve_device("cuda:x")


def test_resolve_device_explicit_cuda_passes_through_without_torch(monkeypatch):
    # Nothing to validate against: the engine import fails first, as before.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert resolve_device("cuda:0") == "cuda:0"


# --- require_reference: min_reference_s enforcement (fix 2) --------------


def test_require_reference_rejects_short_clip(tmp_path: Path):
    ref = _wav(tmp_path, "short.wav", 2.0)
    req = SynthesisRequest(text="Hello.", reference_wav=ref)
    with pytest.raises(ValueError) as excinfo:
        require_reference(req, "chatterbox", 5.0)
    message = str(excinfo.value)
    assert "chatterbox" in message and "5s" in message and "2.0s" in message


def test_require_reference_accepts_long_enough_clip(tmp_path: Path):
    ref = _wav(tmp_path, "long.wav", 10.0)
    req = SynthesisRequest(text="Hello.", reference_wav=ref)
    assert require_reference(req, "chatterbox", 5.0) == ref


def test_require_reference_missing_file_is_clear(tmp_path: Path):
    req = SynthesisRequest(text="Hello.", reference_wav=tmp_path / "nope.wav")
    with pytest.raises(FileNotFoundError, match="reference clip not found"):
        require_reference(req, "chatterbox", 5.0)


def test_chatterbox_rejects_short_ref_before_model_load(tmp_path: Path):
    # The ValueError must fire before _load: no chatterbox is installed
    # in this venv, so reaching _load would raise BackendNotInstalledError.
    ref = _wav(tmp_path, "short.wav", 2.0)
    backend = ChatterboxBackend()
    with pytest.raises(ValueError, match="at least 5s"):
        backend.synthesize(SynthesisRequest(text="Hello.", reference_wav=ref), tmp_path / "o.wav")


def test_mock_backend_is_exempt_from_minimum():
    # The tone is synthetic; no reference is needed or used.
    assert MockBackend.min_reference_s == 0.0


# --- get_backend: reflection errors (fix 4) ------------------------------


def test_get_backend_misregistered_class_is_named(monkeypatch):
    monkeypatch.setitem(
        BACKENDS, "mock", "voice_generator.backends.mock_backend:NoSuchClass"
    )
    with pytest.raises(RuntimeError) as excinfo:
        get_backend("mock")
    message = str(excinfo.value)
    assert "misregistered" in message
    assert "NoSuchClass" in message and "mock_backend" in message


def test_get_backend_rejects_unknown_option():
    with pytest.raises(RuntimeError) as excinfo:
        get_backend("mock", bogus_option=1)
    message = str(excinfo.value)
    assert "mock" in message and "bogus_option" in message and "valid options" in message


def test_get_backend_accepts_valid_options():
    backend = get_backend("qwen3", size="0.6b")
    assert isinstance(backend, Qwen3Backend)
    assert backend.size == "0.6b"


# --- flatten_mono_samples: one helper, one behaviour (fixes 7, 18) ------


def test_flatten_rejects_stereo_list():
    with pytest.raises(ValueError, match="multi-channel"):
        flatten_mono_samples([[0.1] * 4, [0.2] * 4])


def test_flatten_squeezes_single_column():
    assert list(flatten_mono_samples([[0.1], [0.2], [0.3]])) == [0.1, 0.2, 0.3]


def test_flatten_single_row_and_flat_lists():
    assert list(flatten_mono_samples([[0.1, 0.2]])) == [0.1, 0.2]
    assert list(flatten_mono_samples([0.1, 0.2])) == [0.1, 0.2]


def test_flatten_rejects_non_audio():
    with pytest.raises(TypeError):
        flatten_mono_samples(42)
    with pytest.raises(TypeError):
        flatten_mono_samples("hello")


class _FakeArray:
    """Minimal numpy-like: shape + row indexing, no tolist."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(rows[0])) if rows else (0,)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            _, col = idx
            return [row[col] for row in self._rows]
        return self._rows[idx]


def test_flatten_returns_array_row_view_without_copying():
    row = [0.5, -0.5, 0.25]
    arr = _FakeArray([row])
    assert flatten_mono_samples(arr) is row  # (1, N): the row itself, no copy


def test_flatten_array_single_column_and_stereo():
    assert list(flatten_mono_samples(_FakeArray([[0.1], [0.2]]))) == [0.1, 0.2]
    with pytest.raises(ValueError, match="multi-channel"):
        flatten_mono_samples(_FakeArray([[0.1, 0.2], [0.3, 0.4]]))


class _FakeTensor:
    """Minimal torch-like: detach/cpu/numpy chain, no tolist."""

    def __init__(self, arr: _FakeArray) -> None:
        self._arr = arr

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> _FakeArray:
        return self._arr


def test_flatten_tensor_chain_reaches_array_row():
    row = [0.5, -0.5]
    assert flatten_mono_samples(_FakeTensor(_FakeArray([row]))) is row


# --- VoxCPM: compile tokens (fix 15) -------------------------------------


def test_looks_like_compile_error_tokens():
    assert _looks_like_compile_error(RuntimeError("Triton kernel failed to launch"))
    assert _looks_like_compile_error(RuntimeError("torch.compile inductor error"))
    assert _looks_like_compile_error(RuntimeError("torchdynamo failed"))
    assert _looks_like_compile_error(RuntimeError("cudagraph capture failed"))
    # A bare "compile" word is NOT a compile-subsystem error: the old
    # token retried (and reloaded the whole model) on unrelated failures.
    assert not _looks_like_compile_error(RuntimeError("failed to compile the tokenizer"))
    assert not _looks_like_compile_error(RuntimeError("model id not found on the hub"))


# --- VoxCPM: kwarg-rejecting install + compile error (fix 16) ------------


class _FakeVoxModel:
    """Rejects the optimize kwarg, then fails once with a Triton error."""

    instances: ClassVar[list["_FakeVoxModel"]] = []
    load_calls: ClassVar[int] = 0

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.tts_model = types.SimpleNamespace(sample_rate=48000)
        _FakeVoxModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> "_FakeVoxModel":
        cls.load_calls += 1
        if "optimize" in kwargs:
            raise TypeError(
                "from_pretrained() got an unexpected keyword argument 'optimize'"
            )
        if cls.load_calls == 2:
            raise RuntimeError("Triton kernel failed to launch")
        model = cls(**kwargs)
        model.model_id = model_id
        return model

    def generate(self, **kwargs: object) -> list[float]:
        return [0.0] * 4800  # 0.1 s of silence at 48 kHz


@pytest.fixture()
def fake_voxcpm_combined(monkeypatch):
    _FakeVoxModel.instances.clear()
    _FakeVoxModel.load_calls = 0
    module = types.ModuleType("voxcpm")
    module.VoxCPM = _FakeVoxModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voxcpm", module)
    return _FakeVoxModel


def test_voxcpm_kwarg_reject_then_compile_error_retries_without_optimize(
    fake_voxcpm_combined, tmp_path: Path
):
    # Load 1 rejects `optimize` (TypeError) -> retry without it. Load 2
    # then fails with a Triton error -> compile fallback must retry
    # WITHOUT re-adding `optimize`, or the user sees "unexpected keyword
    # argument" instead of the compile failure.
    backend = VoxCPM2Backend()
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert fake_voxcpm_combined.load_calls == 3
    assert "optimize" not in fake_voxcpm_combined.instances[0].kwargs
    assert result.settings["optimize"] == "unsupported"
    assert result.duration_s == pytest.approx(0.1, abs=0.01)


# --- VoxCPM: sample_rate None (fix 17) -----------------------------------


class _FakeVoxModelNoneSr:
    instances: ClassVar[list["_FakeVoxModelNoneSr"]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.tts_model = types.SimpleNamespace(sample_rate=None)
        _FakeVoxModelNoneSr.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> "_FakeVoxModelNoneSr":
        return cls(**kwargs)

    def generate(self, **kwargs: object) -> list[float]:
        return [0.0] * 4800


@pytest.fixture()
def fake_voxcpm_none_sr(monkeypatch):
    _FakeVoxModelNoneSr.instances.clear()
    module = types.ModuleType("voxcpm")
    module.VoxCPM = _FakeVoxModelNoneSr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voxcpm", module)
    return _FakeVoxModelNoneSr


def test_voxcpm_none_sample_rate_uses_default(fake_voxcpm_none_sr, tmp_path: Path):
    # tts_model.sample_rate present but None: the old getattr default
    # only applied when the attribute was missing, so int(None) raised
    # TypeError after a successful render.
    backend = VoxCPM2Backend()
    result = backend.synthesize(_req(tmp_path), tmp_path / "out.wav")
    assert wav_info(tmp_path / "out.wav").sample_rate == 48000
    assert result.duration_s == pytest.approx(0.1, abs=0.01)


# --- CosyVoice: path validation (fixes 12, 14) ---------------------------


def _model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    return d


def _clone_home(tmp_path: Path, *, with_matcha: bool) -> Path:
    home = tmp_path / "CosyVoice"
    if with_matcha:
        (home / "third_party" / "Matcha-TTS").mkdir(parents=True)
    else:
        home.mkdir(parents=True)
    return home


def test_cosy_missing_home_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        CosyBackend(cosyvoice_home=tmp_path / "nope")


def test_cosy_missing_matcha_submodule_rejected(tmp_path: Path):
    home = _clone_home(tmp_path, with_matcha=False)
    with pytest.raises(ValueError) as excinfo:
        CosyBackend(cosyvoice_home=home)
    assert "Matcha-TTS" in str(excinfo.value)
    assert "submodule" in str(excinfo.value)


def test_cosy_missing_model_dir_rejected(tmp_path: Path):
    with pytest.raises(ValueError) as excinfo:
        CosyBackend(model_dir=tmp_path / "no-model")
    message = str(excinfo.value)
    assert "model dir does not exist" in message
    assert "--cosy-model-dir" in message


def test_cosy_home_with_matcha_and_model_dir_constructs(tmp_path: Path):
    home = _clone_home(tmp_path, with_matcha=True)
    model = home / "pretrained_models" / "Fun-CosyVoice3-0.5B"
    model.mkdir(parents=True)
    backend = CosyBackend(cosyvoice_home=home)
    assert backend.model_dir == model


# --- CosyVoice: ImportError narrowing (fix 13) ---------------------------


def test_cosy_non_cosyvoice_import_error_reraised(tmp_path: Path, monkeypatch):
    # A missing `matcha` (broken submodule) must NOT be rewritten as the
    # "CosyVoice was not found" setup hint.
    pkg = types.ModuleType("cosyvoice")
    cli_pkg = types.ModuleType("cosyvoice.cli")
    cli_mod = types.ModuleType("cosyvoice.cli.cosyvoice")

    def _getattr(name: str):
        raise ModuleNotFoundError("No module named 'matcha'", name="matcha")

    cli_mod.__getattr__ = _getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cosyvoice", pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", cli_pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", cli_mod)
    backend = CosyBackend(model_dir=_model_dir(tmp_path))
    with pytest.raises(ModuleNotFoundError) as excinfo:
        backend.synthesize(_req(tmp_path), tmp_path / "z.wav")
    assert "matcha" in str(excinfo.value)
    assert "git clone" not in str(excinfo.value)


# --- Qwen: attn validation, map_language, empty output (fixes 9, 10, 11) -


def test_qwen_attn_implementation_rejected_up_front():
    with pytest.raises(ValueError, match="attn_implementation"):
        Qwen3Backend(attn_implementation="flash-attention-2")
    with pytest.raises(ValueError, match="attn_implementation"):
        Qwen3Backend(attn_implementation="flash_attention")
    assert Qwen3Backend(attn_implementation=None).attn_implementation is None
    assert Qwen3Backend(attn_implementation="eager").attn_implementation == "eager"


def test_map_language_rejects_unknown_code():
    with pytest.raises(ValueError) as excinfo:
        map_language("pt-BR")
    assert "supported codes" in str(excinfo.value)
    with pytest.raises(ValueError, match="supported codes"):
        map_language("xx")
    # A full language name is still accepted, case-insensitively.
    assert map_language("English") == "English"
    assert map_language("SPANISH") == "Spanish"


class _FakeQwenModel:
    instances: ClassVar[list["_FakeQwenModel"]] = []
    generate_return: ClassVar[tuple] = ([[0.0] * 2400], 24000)

    def __init__(self) -> None:
        _FakeQwenModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> "_FakeQwenModel":
        model = cls()
        model.model_id = model_id
        return model

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        return {"ref_audio": ref_audio, "ref_text": ref_text, "x": x_vector_only_mode}

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        return type(self).generate_return


@pytest.fixture()
def fake_qwen2(monkeypatch):
    _FakeQwenModel.instances.clear()
    _FakeQwenModel.generate_return = ([[0.0] * 2400], 24000)
    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = _FakeQwenModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    torch.float32 = "fp32"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return _FakeQwenModel


@pytest.mark.parametrize("empty", [([], 24000), ([[]], 24000)])
def test_qwen_empty_output_raises(fake_qwen2, tmp_path: Path, empty):
    # CosyVoice and VoxCPM guard this; Qwen was the odd one out — an
    # empty list raised a bare IndexError, an empty waveform wrote a
    # 0-second WAV reported as success.
    _FakeQwenModel.generate_return = empty
    backend = Qwen3Backend()
    with pytest.raises(RuntimeError, match="produced no audio"):
        backend.synthesize(_req(tmp_path), tmp_path / "x.wav")
