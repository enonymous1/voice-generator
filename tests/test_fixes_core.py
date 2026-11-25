"""Tests for the core fixes from the 2026-11-25 code review.

Each test pins one fix in cli.py / pipeline.py / text.py / audio.py so the
regression it closed cannot come back silently.
"""

import argparse
import builtins
import importlib.util
import json
import math
import struct
import sys
import tracemalloc
import types
import wave
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

import voice_generator.cli as cli
import voice_generator.pipeline as pipeline
from voice_generator.audio import (
    WavInfo,
    check_reference,
    trim_wav,
    wav_info,
    write_pcm16_wav,
)
from voice_generator.backends.base import BackendNotInstalledError
from voice_generator.pipeline import RunResult, SynthConfig, _write_atomic, synthesize_script
from voice_generator.text import chunk_script


def _make_wav(path: Path, seconds: float = 10.0, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return path


# --- Fix 1: empty --text "" is a clean error, not a traceback ------------


def test_cli_empty_text_is_clean_error_not_traceback(tmp_path: Path, capsys):
    code = cli.main(
        ["synth", "--backend", "mock", "--text", "", "--out", str(tmp_path / "x.wav")]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "script is empty" in captured.err
    assert "Traceback" not in captured.err


# --- Fix 2: doctor hints cover every backend; _probe never raises -------


def test_doctor_mock_hint_when_mock_unimportable(monkeypatch, capsys):
    """The mock entry used to be missing from ``hints`` — a broken mock
    import made doctor itself crash with KeyError."""
    monkeypatch.setattr(cli, "available_backends", lambda: ["mock"])
    monkeypatch.setattr(cli, "_probe", lambda module: False)
    assert cli._doctor() == 0
    assert "mock" in capsys.readouterr().out


@pytest.mark.parametrize("error", [ModuleNotFoundError, ValueError])
def test_probe_returns_false_when_find_spec_raises(monkeypatch, error):
    def _boom(name):
        raise error("broken parent package")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert cli._probe("voice_generator.backends.mock_backend") is False


# --- Fix 3: doctor survives torch import/probe failures ------------------


def test_doctor_tolerates_oserror_from_torch(monkeypatch, capsys):
    """A torch install with missing CUDA DLLs raises OSError ([WinError 126])
    on import, not ImportError — doctor must report it, not traceback."""

    class _BrokenTorch:
        def __getattr__(self, name):
            raise OSError("[WinError 126] The specified module could not be found")

    monkeypatch.setitem(sys.modules, "torch", _BrokenTorch())
    assert cli._doctor() == 0
    assert "not installed" in capsys.readouterr().out


def test_doctor_tolerates_cuda_probe_failure(monkeypatch, capsys):
    def _boom():
        raise RuntimeError("boom")

    fake_torch = types.SimpleNamespace(
        __version__="2.5.0",
        cuda=types.SimpleNamespace(is_available=_boom),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cli._doctor() == 0
    assert "cuda     not available" in capsys.readouterr().out


# --- Fix 4: BackendNotInstalledError is clean; bare RuntimeError is not --


def test_backend_not_installed_error_is_clean(monkeypatch, tmp_path: Path, capsys):
    def _raise(name, **kwargs):
        raise BackendNotInstalledError('chatterbox not installed — pip install -e ".[chatterbox]"')

    monkeypatch.setattr(pipeline, "get_backend", _raise)
    code = cli.main(
        ["synth", "--backend", "chatterbox", "--text", "Hello.", "--out", str(tmp_path / "x.wav")]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: chatterbox not installed" in captured.err
    assert "Traceback" not in captured.err


def test_plain_runtime_error_propagates_with_traceback(monkeypatch, tmp_path: Path):
    """A genuine engine failure (CUDA OOM, shape mismatch, real bug) must
    keep its traceback — it is not a user error with a known fix."""

    def _raise(name, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "get_backend", _raise)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        cli.main(
            [
                "synth",
                "--backend",
                "chatterbox",
                "--text",
                "Hello.",
                "--out",
                str(tmp_path / "x.wav"),
            ]
        )


# --- Fix 5: _freeze/_serve only blame the web extra when it is missing ---


def _fake_import_raising(name_to_fail, exc_name):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == name_to_fail:
            raise ImportError(f"No module named {exc_name!r}", name=exc_name)
        return real_import(name, *args, **kwargs)

    return _fake_import


def test_freeze_deep_import_error_propagates(monkeypatch):
    """An ImportError from inside the web package is a real bug, not a
    missing extra — it must propagate, not print the install hint."""
    monkeypatch.setattr(
        builtins, "__import__", _fake_import_raising("voice_generator.web.freeze", "some_dep")
    )
    with pytest.raises(ImportError, match="some_dep"):
        cli._freeze(SimpleNamespace(project=Path("."), out=Path("site"), seed_mock=False))


def test_freeze_missing_flask_is_clean_error(monkeypatch, capsys):
    monkeypatch.setattr(
        builtins, "__import__", _fake_import_raising("voice_generator.web.freeze", "flask")
    )
    assert cli._freeze(SimpleNamespace(project=Path("."), out=Path("site"), seed_mock=False)) == 1
    assert 'pip install -e ".[web]"' in capsys.readouterr().err


def test_serve_missing_flask_is_clean_error(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setattr(
        builtins, "__import__", _fake_import_raising("voice_generator.web.app", "flask")
    )
    args = SimpleNamespace(
        host="127.0.0.1",
        port=5000,
        refs_dir=tmp_path / "refs",
        output_dir=tmp_path / "output",
        chatterbox_python=None,
        qwen_python=None,
        cosy_python=None,
        voxcpm_python=None,
    )
    assert cli._serve(args) == 1
    assert 'pip install -e ".[web]"' in capsys.readouterr().err


def test_serve_deep_import_error_propagates(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        builtins, "__import__", _fake_import_raising("voice_generator.web.app", "some_dep")
    )
    args = SimpleNamespace(
        host="127.0.0.1",
        port=5000,
        refs_dir=tmp_path / "refs",
        output_dir=tmp_path / "output",
        chatterbox_python=None,
        qwen_python=None,
        cosy_python=None,
        voxcpm_python=None,
    )
    with pytest.raises(ImportError, match="some_dep"):
        cli._serve(args)


# --- Fix 6: --exaggeration / --cfg are validated to [0, 1] ----------------


def test_unit_float_accepts_boundaries():
    assert cli._unit_float("0") == 0.0
    assert cli._unit_float("1") == 1.0
    assert cli._unit_float("0.5") == 0.5
    with pytest.raises(argparse.ArgumentTypeError):
        cli._unit_float("1.01")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._unit_float("-0.1")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._unit_float("abc")


@pytest.mark.parametrize("flag", ["--exaggeration", "--cfg"])
@pytest.mark.parametrize("value", ["-0.1", "1.1", "500", "abc"])
def test_cli_rejects_out_of_range_unit_floats(tmp_path: Path, flag, value):
    """argparse rejects the value with usage + exit 2, before any render."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "synth",
                "--backend",
                "mock",
                "--text",
                "Hello.",
                flag,
                value,
                "--out",
                str(tmp_path / "x.wav"),
            ]
        )
    assert excinfo.value.code == 2


# --- Fix 7: check-ref reads the WAV header once --------------------------


def test_check_ref_reads_header_once(monkeypatch, tmp_path: Path, capsys):
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    calls: list[Path] = []
    real_wav_info = cli.wav_info

    def _counting(path):
        calls.append(Path(path))
        return real_wav_info(path)

    monkeypatch.setattr(cli, "wav_info", _counting)
    seen: dict = {}
    real_check = cli.check_reference

    def _capture(path, **kwargs):
        seen.update(kwargs)
        return real_check(path, **kwargs)

    monkeypatch.setattr(cli, "check_reference", _capture)
    assert cli.main(["check-ref", str(ref)]) == 0
    assert len(calls) == 1
    assert seen.get("info") is not None


def test_check_reference_accepts_pre_read_info(tmp_path: Path):
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    info = wav_info(ref)
    assert check_reference(ref, info=info) == []
    # The path checks still run even with a pre-read header.
    with pytest.raises(FileNotFoundError):
        check_reference(tmp_path / "nope.wav", info=info)


# --- Fix 9: chunk work dir is cleaned up on success, kept otherwise ------


def test_successful_render_removes_auto_parts_dir(tmp_path: Path):
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    out = tmp_path / "out" / "ep1.wav"
    result = synthesize_script(
        text="Hello world. This is a plumbing test, not real speech.",
        reference_wav=ref,
        out_wav=out,
        backend_name="mock",
        max_chars=30,
    )
    assert result.chunks >= 2
    assert not (tmp_path / "out" / ".ep1.parts").exists()


def test_explicit_work_dir_is_kept_after_success(tmp_path: Path):
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    work = tmp_path / "work"
    synthesize_script(
        text="Hello world. Plumbing test.",
        reference_wav=ref,
        out_wav=tmp_path / "ep1.wav",
        backend_name="mock",
        work_dir=work,
    )
    assert work.is_dir()
    assert list(work.glob("chunk_*.wav"))


def test_failed_render_keeps_parts_dir(monkeypatch, tmp_path: Path):
    """A failed chunk stays inspectable — the dir is only removed on success."""
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    out = tmp_path / "out" / "ep1.wav"

    def _failing_backend(name, **kwargs):
        backend = SimpleNamespace(
            license="mock",
            synthesize=lambda request, part: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        return backend

    monkeypatch.setattr(pipeline, "get_backend", _failing_backend)
    with pytest.raises(RuntimeError, match="boom"):
        synthesize_script(
            text="Hello world. Plumbing test.",
            reference_wav=ref,
            out_wav=out,
            backend_name="mock",
        )
    assert (tmp_path / "out" / ".ep1.parts").is_dir()


# --- Fix 10: synthesize_script forwards every SynthConfig field ----------


def test_synthesize_script_accepts_every_synth_config_field(tmp_path: Path):
    """Every SynthConfig field must be reachable through synthesize_script.

    The old signature listed all fields by hand; a new option could be
    added to the dataclass and silently dropped by the entry point. The
    **overrides forwarding makes that impossible — this test pins it by
    deriving the expected kwargs from the dataclass itself.
    """
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    out = tmp_path / "ep1.wav"
    kwargs: dict[str, object] = {}
    for f in fields(SynthConfig):
        if f.name in ("text", "reference_wav", "out_wav"):
            continue
        kwargs[f.name] = f.default
    kwargs["backend_name"] = "mock"
    kwargs["max_chars"] = 30
    kwargs["work_dir"] = tmp_path / "work"
    result = synthesize_script(
        text="Hello world. Plumbing test.", reference_wav=ref, out_wav=out, **kwargs
    )
    assert result.chunks >= 1
    # The explicitly passed work dir is kept (see fix 9).
    assert (tmp_path / "work").is_dir()


def test_synthesize_script_rejects_unknown_kwargs(tmp_path: Path):
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        synthesize_script(
            text="Hello.",
            reference_wav=ref,
            out_wav=tmp_path / "ep1.wav",
            backend_name="mock",
            typo_field=1,
        )


# --- Fix 11: _write_atomic uses a unique temp name + fsync ----------------


def test_write_atomic_roundtrip_and_no_temp_left(tmp_path: Path):
    path = tmp_path / "m.json"
    _write_atomic(path, '{"a": 1}')
    assert path.read_text(encoding="utf-8") == '{"a": 1}'
    assert list(tmp_path.glob("*.tmp")) == []
    _write_atomic(path, '{"a": 2}')  # replace: second write wins
    assert path.read_text(encoding="utf-8") == '{"a": 2}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_atomic_failure_cleans_temp(tmp_path: Path):
    path = tmp_path / "m.json"
    with pytest.raises(TypeError):
        _write_atomic(path, 123)  # type: ignore[arg-type]
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


# --- Fix 12: public dataclasses document their contract ------------------


def test_public_dataclasses_have_docstrings():
    assert RunResult.__doc__ and "total" in RunResult.__doc__
    assert "manifest_path" in RunResult.__doc__
    assert WavInfo.__doc__ and "duration_s" in WavInfo.__doc__


# --- Fix 13: MANIFEST_VERSION note is a comment, not a stray literal ------


def test_manifest_version_note_is_not_a_docstring():
    """The old stray string literal after the assignment looked like
    documentation but was attached to nothing introspectable."""
    import voice_generator.pipeline as pipeline_module

    assert pipeline_module.MANIFEST_VERSION == 1
    assert pipeline_module.__doc__ is not None
    # A module-level string literal following an assignment is not a
    # docstring; the note now lives as a comment in the source instead.
    source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
    assert "# Bump when the manifest schema changes incompatibly" in source


# --- Fix 14: write_pcm16_wav streams in blocks ---------------------------


def test_write_pcm16_wav_block_boundaries(tmp_path: Path):
    """Exact sample counts across the 65,536-sample block boundary."""
    for n in (1, 65_535, 65_536, 65_537, 131_072):
        path = tmp_path / f"n{n}.wav"
        write_pcm16_wav(path, [0.5] * n, 24000)
        assert wav_info(path).frames == n
        with wave.open(str(path), "rb") as w:
            data = w.readframes(n)
        assert struct.unpack(f"<{n}h", data) == (16383,) * n  # int(0.5 * 32767)


def test_write_pcm16_wav_large_input_bounded_memory(tmp_path: Path):
    """Two million samples complete with bounded peak memory.

    The old whole-render list + star-unpacked tuple held ~30x the output
    size in transient objects; the block-streaming writer peaks at one
    block plus file overhead.
    """
    path = tmp_path / "big.wav"
    n = 2_000_000
    samples = (0.5 * math.sin(i * 0.01) for i in range(n))
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        write_pcm16_wav(path, samples, 24000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert wav_info(path).frames == n
    assert peak < 20 * 1024 * 1024


def test_write_pcm16_wav_accepts_generators_and_clips(tmp_path: Path):
    """Generators (not just lists) work, and clipping is unchanged."""
    path = tmp_path / "clip.wav"
    write_pcm16_wav(path, (v for v in [2.0, -2.0, 0.5, 0.0]), 24000)
    with wave.open(str(path), "rb") as w:
        data = w.readframes(4)
    assert struct.unpack("<4h", data) == (32767, -32767, 16383, 0)


# --- Fix 16: trim_wav returns info without re-reading --------------------


def test_trim_wav_return_matches_header(tmp_path: Path):
    src = tmp_path / "src.wav"
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(12 * 16000)]
    write_pcm16_wav(src, samples, 16000)
    out = tmp_path / "trim.wav"
    info = trim_wav(src, out, start_s=2.0, end_s=5.0)
    assert info == wav_info(out)
    assert info.frames == 3 * 16000


# --- Fix 18: chunk_script docstring discloses its behaviors ---------------


def test_chunk_script_docstring_discloses_behaviors():
    assert "paragraph" in chunk_script.__doc__
    assert "closing quote" in chunk_script.__doc__


# --- Fix 19: module docstrings still state the current design -------------


def test_pipeline_docstring_describes_overrides_forwarding():
    assert "**overrides" in pipeline.__doc__


def test_synth_config_roundtrip_through_json(tmp_path: Path):
    """The manifest still records backend settings after the refactor."""
    ref = _make_wav(tmp_path / "ref.wav", seconds=10.0)
    out = tmp_path / "ep1.wav"
    result = synthesize_script(
        text="Hello world.", reference_wav=ref, out_wav=out, backend_name="mock"
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "mock"
    assert manifest["manifest_version"] == 1


# --- Fix 17 (CLI half): `serve` resolves default dirs to the project root --


def _serve_args(**over):
    args = {
        "refs_dir": Path("refs"),
        "output_dir": Path("output"),
        "host": "127.0.0.1",
        "port": 5000,
        "chatterbox_python": None,
        "qwen_python": None,
        "cosy_python": None,
        "voxcpm_python": None,
    }
    args.update(over)
    return SimpleNamespace(**args)


def test_serve_refuses_outside_a_project_root(tmp_path, monkeypatch):
    """No pyproject.toml above the CWD: refuse instead of creating an
    empty refs/ + output/ in the wrong directory."""
    monkeypatch.chdir(tmp_path)
    assert cli._serve(_serve_args()) == 1
    assert not (tmp_path / "refs").exists()
    assert not (tmp_path / "output").exists()


def test_serve_anchors_defaults_to_project_root(tmp_path, monkeypatch):
    """Run from a subdirectory of the project: the default dirs resolve
    to the root (where pyproject.toml lives), not the CWD."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "tools"
    sub.mkdir()
    monkeypatch.chdir(sub)
    seen = {}

    def fake_create_app(config):
        seen["config"] = config

        class _StubApp:
            def run(self, **kwargs):
                seen["ran"] = kwargs

        return _StubApp()

    monkeypatch.setattr("voice_generator.web.app.create_app", fake_create_app)
    assert cli._serve(_serve_args()) == 0
    assert seen["config"].refs_dir == tmp_path / "refs"
    assert seen["config"].output_dir == tmp_path / "output"
    assert (tmp_path / "refs").is_dir()
    assert (tmp_path / "output").is_dir()
    assert not (sub / "refs").exists()
    assert seen["ran"]["port"] == 5000
