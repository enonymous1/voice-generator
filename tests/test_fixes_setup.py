"""Tests for the 2026-11-25 review fixes in setup/ and tools/.

Covers: the checker false FAILs/crashes (24-bit clips, non-UTF-8
transcripts/subprocess output, the "version" false-PASS), the venv
Python version check, the mock-render manifest check, the CUDA
diagnosis split, the CosyVoice import/model/matcha checks, the py
launcher fallback, the installer --recreate / web-skip-core /
project-validation paths, the ab_compare guards (timeout, OSError,
manifest, upfront validation, temp-dir cleanup), the diag script's
import safety, and the packaging/workflow declarations.
"""

import json
import re
import subprocess
import sys
import tempfile
import wave as _wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import check  # noqa: E402
import diag_qwen_prompt  # noqa: E402
import install  # noqa: E402
from ab_compare import _run_backend  # noqa: E402
from ab_compare import main as ab_main
from test_setup import (  # noqa: E402
    _fake_run_factory,
    _install_setup,
    _make_project,
    _tone_wav,
)

ROOT = Path(__file__).resolve().parent.parent


# --- checker: reference clip ------------------------------------------------


def _wide_wav(path: Path, width: int, seconds: float = 12.0, rate: int = 16000) -> None:
    frames = int(seconds * rate)
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes((1000).to_bytes(width, "little", signed=True) * frames)


def test_ref_clip_24bit_and_32bit_pass_with_note(tmp_path):
    """A valid 24/32-bit clip used to FAIL; check_reference never
    checks width, and both read fine — now a PASS with a note."""
    project = tmp_path
    (project / "refs").mkdir()
    clip = project / "refs" / "me.wav"
    _wide_wav(clip, 3)
    result = check.check_ref_clip(project)
    assert result.status == "PASS" and "24-bit" in result.detail
    _wide_wav(clip, 4)
    result = check.check_ref_clip(project)
    assert result.status == "PASS" and "32-bit" in result.detail


def test_ref_transcript_non_utf8_fails_not_crash(tmp_path):
    """A transcript saved in Windows-1252 (a smart quote from Notepad)
    used to crash the whole run with UnicodeDecodeError."""
    project = tmp_path
    (project / "refs").mkdir()
    (project / "refs" / "me.txt").write_bytes(b"caf\x92s talk")
    result = check.check_ref_transcript(project)
    assert result.status == "FAIL"
    assert "UTF-8" in result.fix


# --- checker: run() / check_tool --------------------------------------------


def test_run_replaces_undecodable_subprocess_output():
    """A subprocess emitting non-UTF-8 bytes used to kill the run."""
    proc = check.run(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfe bad')"]
    )
    assert proc.returncode == 0
    assert "\ufffd" in (proc.stdout or "")


def test_tool_check_requires_version_shape(monkeypatch):
    """ffmpeg 8.1.2 exits 8 with 'ffmpeg version 8.1.2' — still a PASS.
    But an error that merely mentions 'version' must not pass."""

    def ffmpeg(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 8, "", "ffmpeg version 8.1.2\n")

    monkeypatch.setattr(check, "run", ffmpeg)
    assert check.check_tool("ffmpeg", fix="x").status == "PASS"

    def bad_flag(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 3, "", "bad version flag\n")

    monkeypatch.setattr(check, "run", bad_flag)
    assert check.check_tool("tool", fix="x").status == "FAIL"

    def unknown_flag(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 1, "", "error: unknown flag --version\n")

    monkeypatch.setattr(check, "run", unknown_flag)
    assert check.check_tool("tool", fix="x").status == "FAIL"


# --- checker: venv Python version -------------------------------------------


def test_venv_python_version_mismatch_fails(tmp_path, monkeypatch):
    """A .venv-qwen3 created on 3.11 used to pass every check while
    violating the documented layout."""
    project = _make_project(tmp_path)
    base_fake = _fake_run_factory(project)

    def fake(cmd, timeout=120, cwd=None):
        argv = [str(c) for c in cmd]
        if argv[1:] == ["--version"] and ".venv-qwen3" in argv[0]:
            return subprocess.CompletedProcess(cmd, 0, "Python 3.11.9\n", "")
        return base_fake(cmd, timeout=timeout, cwd=cwd)

    monkeypatch.setattr(check, "run", fake)
    results = check.collect(project)
    version = next(r for r in results if r.id == "venv-python-qwen3")
    assert version.status == "FAIL"
    assert "recreate the venv with py -3.12" in version.fix
    assert check.report(results) == 1


# --- checker: mock render manifest ------------------------------------------


def test_mock_render_missing_manifest_fails(tmp_path, monkeypatch):
    """A synth that writes the WAV but no manifest used to PASS."""
    project = tmp_path / "project"
    project.mkdir()
    base_fake = _fake_run_factory(project)

    def fake(cmd, timeout=120, cwd=None):
        argv = [str(c) for c in cmd]
        if argv[1:3] == ["-m", "voice_generator"] and argv[3] == "synth":
            out = argv[argv.index("--out") + 1]
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return base_fake(cmd, timeout=timeout, cwd=cwd)

    monkeypatch.setattr(check, "run", fake)
    result = check.check_mock_render(project, Path("/fake/python"))
    assert result.status == "FAIL" and "manifest" in result.detail


def test_mock_render_unreadable_manifest_fails(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    base_fake = _fake_run_factory(project)

    def fake(cmd, timeout=120, cwd=None):
        argv = [str(c) for c in cmd]
        if argv[1:3] == ["-m", "voice_generator"] and argv[3] == "synth":
            out = argv[argv.index("--out") + 1]
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"RIFF")
            Path(out).with_suffix(".manifest.json").write_text("{not json", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return base_fake(cmd, timeout=timeout, cwd=cwd)

    monkeypatch.setattr(check, "run", fake)
    result = check.check_mock_render(project, Path("/fake/python"))
    assert result.status == "FAIL" and "manifest" in result.detail


# --- checker: torch probe / CUDA diagnosis ----------------------------------


def test_probe_torch_parses_lines(monkeypatch):
    def fake(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(
            cmd,
            0,
            "torch: ok\ntorch_version: 2.5.1+cu121\ncuda: False\nimport: fail\n"
            "error: No module named 'voxcpm'\n",
            "",
        )

    monkeypatch.setattr(check, "run", fake)
    probe = check.probe_torch(Path("/fake/python"), "voxcpm")
    assert probe == {
        "torch": "ok",
        "torch_version": "2.5.1+cu121",
        "cuda": "False",
        "import": "fail",
        "error": "No module named 'voxcpm'",
    }
    cuda, imp = check.check_cuda_and_import(Path("/fake/python"), "voxcpm", "voxcpm")
    assert cuda.status == "FAIL" and "CPU-only" in cuda.detail
    assert imp.status == "FAIL" and "No module named 'voxcpm'" in imp.detail


def test_probe_torch_dead_probe_is_error_not_pass(monkeypatch):
    """A probe that dies (timeout, crash) must not read as a PASS."""

    def fake(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 1, "", "boom\n")

    monkeypatch.setattr(check, "run", fake)
    probe = check.probe_torch(Path("/fake/python"))
    assert "torch" not in probe and probe["error"] == "boom"
    assert check.check_cuda(Path("/fake/python"), "qwen3").status == "FAIL"
    assert check.cuda_diagnosis(Path("/fake/python"), "qwen3") == "error"


def test_cuda_diagnosis_splits_cases(monkeypatch):
    """Missing torch, CPU-only torch, and a broken install used to all
    get the same 'CPU-only' diagnosis and the cu121 fix."""

    def missing(cmd, timeout=120, cwd=None):
        # The real probe catches the import error and prints it, so the
        # missing-torch case arrives as stdout, not as a dead probe.
        return subprocess.CompletedProcess(
            cmd, 0, "torch: missing\nerror: No module named 'torch'\n", ""
        )

    monkeypatch.setattr(check, "run", missing)
    result = check.check_cuda(Path("/fake/python"), "qwen3")
    assert result.status == "FAIL" and "not installed" in result.detail
    assert "setup/install.py --backend qwen3" in result.fix
    assert check.cuda_diagnosis(Path("/fake/python"), "qwen3") == "no-torch"

    def cpu_only(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 0, "torch: ok\ncuda: False\nimport: ok\n", "")

    monkeypatch.setattr(check, "run", cpu_only)
    result = check.check_cuda(Path("/fake/python"), "qwen3")
    assert result.status == "FAIL" and "CPU-only" in result.detail
    assert "download.pytorch.org" in result.fix
    assert check.cuda_diagnosis(Path("/fake/python"), "qwen3") == "cpu-only"

    def broken(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(
            cmd,
            0,
            "torch: missing\nerror: ImportError: DLL load failed while importing torch\n",
            "",
        )

    monkeypatch.setattr(check, "run", broken)
    result = check.check_cuda(Path("/fake/python"), "qwen3")
    assert result.status == "FAIL" and "DLL load failed" in result.detail
    assert check.cuda_diagnosis(Path("/fake/python"), "qwen3") == "error"


def test_check_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown backend: nope"):
        check.check_backend(Path("/fake"), "nope", Path("/fake/python"))


# --- checker: py launcher fallback ------------------------------------------


def test_py_launcher_falls_back_to_python_exe(tmp_path, monkeypatch):
    """A Windows box with a Store-only Python (no py.exe) used to fail
    the launcher check even with a usable `python` on PATH."""
    monkeypatch.setattr(check.sys, "platform", "win32")
    project = _make_project(tmp_path)
    base_fake = _fake_run_factory(project)

    def fake(cmd, timeout=120, cwd=None):
        argv = [str(c) for c in cmd]
        if argv == ["py", "--version"]:
            return subprocess.CompletedProcess(cmd, 1, "", "py not found")
        if argv == ["python", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "Python 3.12.3\n", "")
        return base_fake(cmd, timeout=timeout, cwd=cwd)

    monkeypatch.setattr(check, "run", fake)
    results = check.collect(project)
    launcher = next(r for r in results if r.id == "py-launcher")
    assert launcher.status == "PASS"
    assert "python.exe" in launcher.detail


# --- checker: CosyVoice -----------------------------------------------------


def _cosy_fake(base_fake, *, import_rc=0):
    def fake(cmd, timeout=120, cwd=None):
        argv = [str(c) for c in cmd]
        if argv[:2] == ["conda", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "conda 24.1.0\n", "")
        if argv[:3] == ["conda", "env", "list"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                "# conda environments:\n#\nbase  *  /opt/conda\n"
                "cosyvoice  /opt/conda/envs/cosyvoice\n",
                "",
            )
        if argv[:4] == ["conda", "run", "-n", "cosyvoice"]:
            return subprocess.CompletedProcess(
                cmd,
                import_rc,
                "",
                "" if import_rc == 0 else "ModuleNotFoundError: No module named 'cosyvoice'\n",
            )
        return base_fake(cmd, timeout=timeout, cwd=cwd)

    return fake


def _cosy_clone(tmp_path):
    clone = tmp_path / "CosyVoice"
    (clone / "pretrained_models" / "Fun-CosyVoice3-0.5B").mkdir(parents=True)
    (clone / "third_party" / "Matcha-TTS").mkdir(parents=True)
    return clone


def test_cosyvoice_import_model_matcha_pass(tmp_path, monkeypatch):
    """A clone dir + env name used to count as 'verified' without the
    backend ever importing or the model snapshot existing."""
    monkeypatch.delenv("COSYVOICE_HOME", raising=False)
    monkeypatch.delenv("COSYVOICE_MODEL_DIR", raising=False)
    project = _make_project(tmp_path)
    _cosy_clone(tmp_path)
    monkeypatch.setattr(check, "run", _cosy_fake(_fake_run_factory(project)))
    by_id = {r.id: r for r in check.check_cosyvoice(project)}
    assert by_id["cosyvoice-import"].status == "PASS"
    assert by_id["cosyvoice-model"].status == "PASS"
    assert by_id["cosyvoice-matcha"].status == "PASS"


def test_cosyvoice_import_model_matcha_failures_are_optional(tmp_path, monkeypatch):
    monkeypatch.delenv("COSYVOICE_HOME", raising=False)
    monkeypatch.delenv("COSYVOICE_MODEL_DIR", raising=False)
    project = _make_project(tmp_path)
    (tmp_path / "CosyVoice").mkdir()  # clone dir, but no model, no matcha
    monkeypatch.setattr(
        check, "run", _cosy_fake(_fake_run_factory(project), import_rc=1)
    )
    results = check.check_cosyvoice(project)
    by_id = {r.id: r for r in results}
    assert by_id["cosyvoice-import"].status == "FAIL" and by_id["cosyvoice-import"].optional
    assert by_id["cosyvoice-model"].status == "FAIL" and by_id["cosyvoice-model"].optional
    assert by_id["cosyvoice-matcha"].status == "FAIL" and by_id["cosyvoice-matcha"].optional
    assert check.report(results) == 0  # optional failures never fail the run


def test_cosyvoice_relative_home_resolves_against_project_parent(tmp_path, monkeypatch):
    """A relative COSYVOICE_HOME used to resolve against the caller's
    working directory instead of the checkout's parent."""
    monkeypatch.setenv("COSYVOICE_HOME", "CosyVoice")
    monkeypatch.delenv("COSYVOICE_MODEL_DIR", raising=False)
    project = _make_project(tmp_path)
    _cosy_clone(tmp_path)
    monkeypatch.setattr(check, "run", _cosy_fake(_fake_run_factory(project)))
    by_id = {r.id: r for r in check.check_cosyvoice(project)}
    assert by_id["cosyvoice-clone"].status == "PASS"
    assert str(tmp_path / "CosyVoice") in by_id["cosyvoice-clone"].detail


# --- installer --------------------------------------------------------------


def test_install_recreate_deletes_venv(tmp_path, monkeypatch):
    """--recreate is the repair path for a venv whose pip is broken."""
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=True)
    venv_dir = project / ".venv-core"
    sentinel = venv_dir / "sentinel.txt"
    sentinel.write_text("old venv", encoding="utf-8")
    assert install.main(["--backend", "core", "--project", str(project), "--recreate"]) == 0
    assert not sentinel.exists()
    assert check.venv_python(venv_dir).is_file()


def test_install_web_skips_core_when_verified(tmp_path, monkeypatch):
    """--backend web used to re-install core unconditionally, even when
    the core venv already verified."""
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=True)
    assert install.main(["--backend", "web", "--project", str(project)]) == 0
    editable = [c for c in calls if "pip" in c and "-e" in c]
    assert [c[-1] for c in editable] == [".[web]"], calls


def test_install_rejects_non_project_dir(tmp_path, monkeypatch, capsys):
    """A typo'd --project used to create .venv-core in the wrong
    directory before failing at pip install."""
    project = tmp_path / "project"
    project.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(install, "run_stream", lambda cmd, cwd=None: calls.append(cmd) or 0)
    assert install.main(["--backend", "core", "--project", str(project)]) == 1
    assert calls == []
    assert "not a voice-generator checkout" in capsys.readouterr().err


def test_install_pip_failure_prints_recreate_guidance(tmp_path, monkeypatch, capsys):
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=True)

    def failing_stream(cmd, cwd=None):
        argv = [str(c) for c in cmd]
        calls.append(argv)
        if "pip" in argv and "-e" in argv:
            return 1
        return 0

    monkeypatch.setattr(install, "run_stream", failing_stream)
    assert install.main(["--backend", "core", "--project", str(project)]) == 1
    assert "use --recreate" in capsys.readouterr().err


def test_install_build_backend_installed_when_missing(tmp_path, monkeypatch):
    """The editable installs run with --no-build-isolation, so the venv
    must already have setuptools + wheel — installed when missing."""
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=False)
    base_fake = _fake_run_factory(project)

    def fake_run(cmd, timeout=120, cwd=None):
        if "import setuptools, wheel" in " ".join(str(c) for c in cmd):
            return subprocess.CompletedProcess(cmd, 1, "", "No module named 'setuptools'")
        return base_fake(cmd, timeout=timeout, cwd=cwd)

    monkeypatch.setattr(check, "run", fake_run)
    assert install.main(["--backend", "core", "--project", str(project)]) == 0
    setuptools_calls = [c for c in calls if c[-3:] == ["install", "setuptools", "wheel"]]
    editable = [c for c in calls if "pip" in c and "-e" in c]
    assert setuptools_calls, calls
    assert calls.index(setuptools_calls[0]) < calls.index(editable[0])


def test_install_cuda_fix_prints_torch_version(tmp_path, monkeypatch, capsys):
    """The cu121 install takes the latest torch on that index — the
    installed version is printed so drift is visible."""
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=True)
    python = check.venv_python(project / ".venv-chatterbox")
    python.parent.mkdir(parents=True)
    python.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(check, "cuda_diagnosis", lambda py, backend: "cpu-only")
    base_fake = _fake_run_factory(project)

    def fake_run(cmd, timeout=120, cwd=None):
        if "torch.cuda.is_available" in " ".join(str(c) for c in cmd):
            return subprocess.CompletedProcess(
                cmd,
                0,
                "torch: ok\ntorch_version: 2.5.1+cu121\ncuda: True\nimport: ok\n",
                "",
            )
        return base_fake(cmd, timeout=timeout, cwd=cwd)

    monkeypatch.setattr(check, "run", fake_run)
    assert install.main(["--backend", "chatterbox", "--project", str(project)]) == 0
    assert "torch 2.5.1+cu121" in capsys.readouterr().out


# --- ab_compare -------------------------------------------------------------


def _ab_inputs(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Hello.", encoding="utf-8")
    ref = tmp_path / "me.wav"
    _tone_wav(ref)
    return script, ref


def _good_manifest():
    return {"duration_s": 1.5, "sample_rate": 24000, "chunks": [{"a": 1}], "license": "MIT"}


def _ab_fake(*, rc=0, stderr="", stdout="", wav=True, manifest="good"):
    def fake(cmd, **kwargs):
        argv = [str(c) for c in cmd]
        out = Path(argv[argv.index("--out") + 1])
        if wav:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF....")
        if manifest == "good":
            out.with_suffix(".manifest.json").write_text(
                json.dumps(_good_manifest()), encoding="utf-8"
            )
        elif manifest == "bad":
            out.with_suffix(".manifest.json").write_text(
                json.dumps({"sample_rate": 24000}), encoding="utf-8"
            )
        elif manifest == "corrupt":
            out.with_suffix(".manifest.json").write_text("{not json", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, rc, stdout, stderr)

    return fake


def test_run_backend_missing_interpreter_records_failed_side(tmp_path, monkeypatch):
    """A backend python that cannot run used to crash the whole tool
    with FileNotFoundError instead of recording a failed side."""

    def boom(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file", cmd[0])

    monkeypatch.setattr(subprocess, "run", boom)
    script, ref = _ab_inputs(tmp_path)
    result = _run_backend(
        label="chatterbox:turbo",
        python=Path("/nonexistent/python.exe"),
        backend="chatterbox",
        script=script,
        ref=ref,
        out_wav=tmp_path / "out.wav",
        extra=[],
    )
    assert result["ok"] is False
    assert "/nonexistent/python.exe" in result["error"]


def test_run_backend_timeout(tmp_path, monkeypatch):
    """A hung backend used to hang the comparison forever (the Qwen3
    CPU-only stall is exactly this shape)."""

    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    script, ref = _ab_inputs(tmp_path)
    result = _run_backend(
        label="qwen3:1.7b",
        python=Path("/x/python"),
        backend="qwen3",
        script=script,
        ref=ref,
        out_wav=tmp_path / "out.wav",
        extra=[],
        timeout=1,
    )
    assert result["ok"] is False
    assert result["error"] == "timed out after 1s"


def test_run_backend_missing_manifest_records_failed_side(tmp_path, monkeypatch):
    """A synth that exits 0 without a manifest used to crash the tool
    with KeyError, losing every side's results."""
    monkeypatch.setattr(subprocess, "run", _ab_fake(manifest=None))
    script, ref = _ab_inputs(tmp_path)
    result = _run_backend(
        label="chatterbox:turbo",
        python=Path("/x/python"),
        backend="chatterbox",
        script=script,
        ref=ref,
        out_wav=tmp_path / "out.wav",
        extra=[],
    )
    assert result["ok"] is False and "manifest" in result["error"]


def test_run_backend_bad_and_corrupt_manifest(tmp_path, monkeypatch):
    script, ref = _ab_inputs(tmp_path)
    monkeypatch.setattr(subprocess, "run", _ab_fake(manifest="bad"))
    result = _run_backend(
        label="chatterbox:turbo",
        python=Path("/x/python"),
        backend="chatterbox",
        script=script,
        ref=ref,
        out_wav=tmp_path / "out.wav",
        extra=[],
    )
    assert result["ok"] is False and "manifest" in result["error"]
    monkeypatch.setattr(subprocess, "run", _ab_fake(manifest="corrupt"))
    result = _run_backend(
        label="chatterbox:turbo",
        python=Path("/x/python"),
        backend="chatterbox",
        script=script,
        ref=ref,
        out_wav=tmp_path / "out.wav",
        extra=[],
    )
    assert result["ok"] is False and "manifest" in result["error"]


def test_run_backend_empty_wav_records_failed_side(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _ab_fake(wav=True, manifest="good"))
    script, ref = _ab_inputs(tmp_path)
    out = tmp_path / "out.wav"
    # The fake writes a WAV; empty it afterwards via a second fake.
    monkeypatch.setattr(
        subprocess,
        "run",
        _ab_fake(rc=0, wav=False),
    )
    out.write_bytes(b"")
    result = _run_backend(
        label="chatterbox:turbo",
        python=Path("/x/python"),
        backend="chatterbox",
        script=script,
        ref=ref,
        out_wav=out,
        extra=[],
    )
    assert result["ok"] is False and "empty" in result["error"]


def test_run_backend_error_falls_back_to_stdout_and_exit_code(tmp_path, monkeypatch):
    """A side that fails with empty stderr used to record error: ''."""
    script, ref = _ab_inputs(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", _ab_fake(rc=3, stderr="", stdout="boom from stdout", wav=False)
    )
    result = _run_backend(
        label="chatterbox:turbo",
        python=Path("/x/python"),
        backend="chatterbox",
        script=script,
        ref=ref,
        out_wav=tmp_path / "out.wav",
        extra=[],
    )
    assert result["ok"] is False and result["error"] == "boom from stdout"
    monkeypatch.setattr(subprocess, "run", _ab_fake(rc=3, stderr="", stdout="", wav=False))
    result = _run_backend(
        label="chatterbox:turbo",
        python=Path("/x/python"),
        backend="chatterbox",
        script=script,
        ref=ref,
        out_wav=tmp_path / "out.wav",
        extra=[],
    )
    assert result["ok"] is False and result["error"] == "exit code 3"


def test_upfront_validation_runs_before_any_render(tmp_path, monkeypatch):
    """--voxcpm-mode ultimate without a transcript used to be rejected
    only inside the VoxCPM block, after the other sides had rendered."""
    script, ref = _ab_inputs(tmp_path)
    calls: list[list[str]] = []

    def spy(cmd, **kwargs):
        calls.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", spy)
    out_dir = tmp_path / "ab"
    base = [
        "--ref",
        str(ref),
        "--chatterbox-python",
        sys.executable,
        "--out-dir",
        str(out_dir),
    ]
    with pytest.raises(SystemExit):
        ab_main(
            ["--script", str(script), "--voxcpm-mode", "ultimate",
             "--voxcpm-python", sys.executable, *base]
        )
    with pytest.raises(SystemExit):
        ab_main(["--script", str(script), "--ref", str(tmp_path / "nope.wav"), *base[2:]])
    with pytest.raises(SystemExit):
        ab_main(["--script", str(tmp_path / "nope.txt"), *base])
    out_file = tmp_path / "outfile"
    out_file.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        ab_main(
            ["--script", str(script), "--ref", str(ref),
             "--chatterbox-python", sys.executable, "--out-dir", str(out_file)]
        )
    with pytest.raises(SystemExit):
        ab_main(
            ["--script", str(script), "--ref", str(ref),
             "--chatterbox-python", "/nonexistent/python.exe", "--out-dir", str(out_dir)]
        )
    assert calls == []


def test_demo_script_temp_dir_cleaned_up(tmp_path, monkeypatch):
    """The demo temp dir used to leak on every abort path (no context
    manager; p.error raises SystemExit before cleanup())."""
    ref = tmp_path / "me.wav"
    _tone_wav(ref)
    created = []
    real_tmp = tempfile.TemporaryDirectory

    class Spy(real_tmp):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)
            self.cleaned = False

        def cleanup(self):
            super().cleanup()
            self.cleaned = True

    monkeypatch.setattr(tempfile, "TemporaryDirectory", Spy)
    code = ab_main(
        [
            "--demo-script",
            "--ref",
            str(ref),
            "--chatterbox-python",
            sys.executable,
            "--out-dir",
            str(tmp_path / "ab"),
        ]
    )
    assert code == 1  # chatterbox is not installed here; the side fails
    assert len(created) == 1
    assert created[0].cleaned is True
    assert not Path(created[0].name).exists()


def test_demo_script_abort_before_render_creates_no_temp_dir(tmp_path, monkeypatch):
    ref = tmp_path / "me.wav"
    _tone_wav(ref)
    created = []
    real_tmp = tempfile.TemporaryDirectory

    class Spy(real_tmp):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", Spy)
    with pytest.raises(SystemExit):
        ab_main(["--demo-script", "--ref", str(ref), "--out-dir", str(tmp_path / "ab")])
    assert created == []


# --- diag script ------------------------------------------------------------


def test_diag_import_is_safe():
    """Importing the repro used to load a 1.7B model (no main guard)."""
    assert diag_qwen_prompt.REF == Path("refs") / "me_10s.wav"
    assert callable(diag_qwen_prompt.main)


def test_diag_log_timestamps(capsys):
    diag_qwen_prompt.log("hello")
    assert re.search(r"\[\d\d:\d\d:\d\d\] hello", capsys.readouterr().out)


def test_diag_main_help():
    with pytest.raises(SystemExit) as exc:
        diag_qwen_prompt.main(["--help"])
    assert exc.value.code == 0


# --- packaging / workflows --------------------------------------------------


def test_wheel_includes_web_templates():
    """The wheel used to ship no templates/static, so a non-editable
    install served nothing. The package-data declaration must match
    real files, and the installed package must resolve them."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text
    assert '"voice_generator.web" = ["templates/*.html", "static/*"]' in text
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "templates" in manifest and "static" in manifest
    from importlib.resources import files

    web = files("voice_generator.web")
    assert (web / "templates" / "base.html").is_file()
    assert (web / "static" / "style.css").is_file()
    assert len(list((web / "templates").iterdir())) >= 7


def test_workflows_agree_and_cache():
    """The pages ruff line used to miss `setup` (the third setup-lint
    drift); neither workflow cached pip."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pages = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "ruff check src tests tools setup" in ci
    assert "ruff check src tests tools setup" in pages
    assert 'cache: "pip"' in ci and 'cache: "pip"' in pages
    assert "fail-fast: false" in ci
    assert "timeout-minutes: 30" in ci
    assert "voice-gen freeze --out /tmp/frozen-check --seed-mock" in pages
    assert "diff -r --exclude=.gitkeep /tmp/frozen-check site" in pages


def test_dependency_pins():
    """Web/dev deps are pinned to the versions verified on this box
    (Flask 3.1.3, Markdown 3.11.3, pytest 9.2.0, ruff 0.17.3).
    Backend packages are NOT pinned from here — no invented versions."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.14"' in text
    for pinned in ("flask>=3.1,<4", "markdown>=3.11,<4", "pytest>=9,<10", "ruff>=0.17,<0.18"):
        assert pinned in text, pinned
    assert "qwen-tts>=" not in text and "voxcpm>=" not in text
    assert "chatterbox-tts>=" not in text
    assert "pip freeze" in text
