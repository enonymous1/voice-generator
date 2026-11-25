"""Tests for setup/check.py and setup/install.py.

Everything is mocked at the subprocess boundary (``check.run`` /
``install.run_stream``) or uses real files in tmp_path — no venv is
created, nothing is downloaded, no GPU is needed. The reference-clip
checks run against real WAVs written with the project's own writer.
"""

import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup"))

import check  # noqa: E402
import install  # noqa: E402

from voice_generator.audio import write_pcm16_wav  # noqa: E402


def _tone_wav(path: Path, seconds: float = 10.0, rate: int = 16000, channels: int = 1) -> None:
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
    if channels == 2:
        import wave as _wave

        with _wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(rate)
            frames = b"".join(
                int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True) * 2
                for s in samples
            )
            w.writeframes(frames)
        return
    write_pcm16_wav(path, samples, rate)


def _pass_result(check_id: str) -> check.Result:
    return check.Result(check_id, check_id, "PASS", "ok")


def _fake_run_factory(tmp_project: Path):
    """A fake ``check.run``: every probe succeeds, the mock render
    creates its output file and a manifest, ffmpeg answers with a
    non-zero exit (the real ffmpeg 8.1.2 behaviour — version text
    still counts). Answers the Python probes in both launcher forms:
    ``py -3.11`` (Windows) and ``python3.11`` (Linux/macOS) — the fake
    must speak whichever form the code under test uses, or the Windows
    CI run sees every Python as missing. The torch probe answers in
    the probe's own line format (``torch: ok`` / ``cuda: True`` /
    ``import: ok``), and a venv interpreter's ``--version`` answers
    per documented venv (3.11 for core/chatterbox, 3.12 for qwen3/
    voxcpm)."""

    def fake_run(cmd, timeout=120, cwd=None):
        argv = [str(c) for c in cmd]
        if argv == ["py", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "Python 3.12.3\n", "")
        if argv[1:] == ["--version"] and ("/" in argv[0] or "\\" in argv[0]):
            # A venv interpreter version probe (check_venv_python_version):
            # a bare `python3.12` launcher probe has no separator and
            # must fall through to the launcher branches below.
            if ".venv-qwen3" in argv[0] or ".venv-voxcpm" in argv[0]:
                return subprocess.CompletedProcess(cmd, 0, "Python 3.12.3\n", "")
            return subprocess.CompletedProcess(cmd, 0, "Python 3.11.9\n", "")
        if "-3.11" in argv or argv[:1] == ["python3.11"]:
            return subprocess.CompletedProcess(cmd, 0, "Python 3.11.9\n", "")
        if "-3.12" in argv or argv[:1] == ["python3.12"]:
            return subprocess.CompletedProcess(cmd, 0, "Python 3.12.3\n", "")
        if argv[:2] == ["git", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "git version 2.43.0\n", "")
        if argv[:2] == ["ffmpeg", "--version"]:
            return subprocess.CompletedProcess(
                cmd, 8, "", "ffmpeg version 8.1.2 Copyright (c) 2000-2026\n"
            )
        if argv[:2] == ["sox", "--version"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if argv[:2] == ["conda", "--version"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if "torch.cuda.is_available" in " ".join(argv):
            return subprocess.CompletedProcess(cmd, 0, "torch: ok\ncuda: True\nimport: ok\n", "")
        if argv[1:3] == ["-m", "voice_generator"] and argv[3] == "synth":
            out = argv[argv.index("--out") + 1]
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"RIFF")
            Path(out).with_suffix(".manifest.json").write_text(
                '{"manifest_version": 1, "duration_s": 1.0}', encoding="utf-8"
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


def _make_project(tmp_path: Path, *, with_venvs: bool = True) -> Path:
    project = tmp_path / "project"
    refs = project / "refs"
    refs.mkdir(parents=True)
    _tone_wav(refs / "me.wav", seconds=12.0)
    (refs / "me.txt").write_text("The test transcript.", encoding="utf-8")
    if with_venvs:
        for venv in (".venv-core", ".venv-chatterbox", ".venv-qwen3", ".venv-voxcpm"):
            python = check.venv_python(project / venv)
            python.parent.mkdir(parents=True)
            python.write_text("# stub\n", encoding="utf-8")
    return project


# --- platform helpers -------------------------------------------------------


def test_launcher_and_venv_paths(monkeypatch):
    monkeypatch.setattr(check.sys, "platform", "win32")
    assert check.launcher(3, 11) == ["py", "-3.11"]
    assert check.venv_python(Path("D:/Code/x/.venv-core")) == (
        Path("D:/Code/x/.venv-core/Scripts/python.exe")
    )
    monkeypatch.setattr(check.sys, "platform", "linux")
    assert check.launcher(3, 12) == ["python3.12"]
    assert check.venv_python(Path("/tmp/x/.venv-core")) == Path("/tmp/x/.venv-core/bin/python")


def test_python_check_pass_and_fail(monkeypatch):
    def ok(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 0, "Python 3.11.9\n", "")

    monkeypatch.setattr(check, "run", ok)
    result = check.check_python(3, 11)
    assert result.status == "PASS" and "3.11.9" in result.detail

    def missing(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 1, "", "[Errno 2] No such file")

    monkeypatch.setattr(check, "run", missing)
    result = check.check_python(3, 11)
    assert result.status == "FAIL" and "install Python 3.11" in result.fix


def test_tool_check_counts_version_text_despite_nonzero_exit(monkeypatch):
    """ffmpeg 8.1.2 prints its version to stderr and exits 8 — the
    checker must not call that 'not found'."""

    def ffmpeg_rc8(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 8, "", "ffmpeg version 8.1.2\n")

    monkeypatch.setattr(check, "run", ffmpeg_rc8)
    assert check.check_tool("ffmpeg", fix="x").status == "PASS"

    def absent(cmd, timeout=120, cwd=None):
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(check, "run", absent)
    result = check.check_tool("ffmpeg", fix="install ffmpeg")
    assert result.status == "FAIL" and result.fix == "install ffmpeg"


# --- reference clip ---------------------------------------------------------


def test_ref_clip_good_short_stereo_and_bad(tmp_path):
    project = tmp_path
    (project / "refs").mkdir()
    clip = project / "refs" / "me.wav"

    assert check.check_ref_clip(project).status == "FAIL"  # missing

    _tone_wav(clip, seconds=12.0)
    result = check.check_ref_clip(project)
    assert result.status == "PASS" and "12.0s" in result.detail

    _tone_wav(clip, seconds=2.0)
    result = check.check_ref_clip(project)
    assert result.status == "FAIL" and "2.0s" in result.detail

    _tone_wav(clip, seconds=5.0)
    result = check.check_ref_clip(project)
    assert result.status == "PASS" and "better clone" in result.detail

    _tone_wav(clip, seconds=12.0, channels=2)
    result = check.check_ref_clip(project)
    assert result.status == "PASS" and "stereo" in result.detail

    _tone_wav(clip, seconds=12.0, rate=8000)
    result = check.check_ref_clip(project)
    assert result.status == "PASS" and "16 kHz" in result.detail

    clip.write_bytes(b"not a wav at all")
    result = check.check_ref_clip(project)
    assert result.status == "FAIL" and "unreadable" in result.detail


def test_ref_transcript(tmp_path):
    project = tmp_path
    (project / "refs").mkdir()
    assert check.check_ref_transcript(project).status == "FAIL"
    (project / "refs" / "me.txt").write_text("   ", encoding="utf-8")
    assert check.check_ref_transcript(project).status == "FAIL"  # blank
    (project / "refs" / "me.txt").write_text("words", encoding="utf-8")
    assert check.check_ref_transcript(project).status == "PASS"


# --- full collect -----------------------------------------------------------


def test_collect_all_pass_and_report_zero(tmp_path, monkeypatch, capsys):
    project = _make_project(tmp_path)
    monkeypatch.setattr(check, "run", _fake_run_factory(project))
    results = check.collect(project)
    required_failed = [r for r in results if r.status == "FAIL" and not r.optional]
    assert required_failed == []
    assert check.report(results) == 0
    # The counts differ by platform — the py launcher is a real check
    # on Windows and a skip on Linux/macOS — so assert the summary
    # from the results instead of a hardcoded string.
    passed = sum(1 for r in results if r.status == "PASS")
    skipped = sum(
        1 for r in results if r.status == "SKIP" or (r.status == "FAIL" and r.optional)
    )
    assert f"{passed} passed, 0 failed, {skipped} skipped" in capsys.readouterr().out


def test_collect_all_pass_on_windows_forms(tmp_path, monkeypatch):
    """The Windows launcher forms (``py --version``, ``py -3.11``) run
    through the same collect and pass — this is the path the Windows
    CI run exercises, and the fake used to answer it with empty
    output, which read as 'not found'."""
    monkeypatch.setattr(check.sys, "platform", "win32")
    project = _make_project(tmp_path)
    monkeypatch.setattr(check, "run", _fake_run_factory(project))
    results = check.collect(project)
    assert [r for r in results if r.status == "FAIL" and not r.optional] == []
    assert check.report(results) == 0


def test_collect_required_failure_fails_report(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    monkeypatch.setattr(check, "run", _fake_run_factory(project))
    (project / "refs" / "me.wav").unlink()
    results = check.collect(project)
    assert any(r.id == "ref-clip" and r.status == "FAIL" for r in results)
    assert check.report(results) == 1


def test_collect_optional_failure_is_skip_not_fail(tmp_path, monkeypatch, capsys):
    """A missing optional piece (SoX) reports SKIP and the run still exits 0."""
    project = _make_project(tmp_path)
    monkeypatch.setattr(check, "run", _fake_run_factory(project))
    results = check.collect(project)
    sox = next(r for r in results if r.id == "sox")
    assert sox.status == "FAIL" and sox.optional
    assert check.report(results) == 0
    assert "[SKIP] SoX" in capsys.readouterr().out


def test_collect_single_venv_override(tmp_path, monkeypatch):
    """One shared venv: every backend check runs against the override
    python instead of the documented per-backend venvs."""
    project = _make_project(tmp_path, with_venvs=False)
    shared = project / ".venv" / "bin" / "python"
    shared.parent.mkdir(parents=True)
    shared.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(check, "run", _fake_run_factory(project))
    overrides = {name: shared for name in ("core", "chatterbox", "qwen3", "voxcpm")}
    results = check.collect(project, overrides)
    assert all(r.status != "FAIL" or r.optional for r in results)
    assert check.report(results) == 0


# --- installer --------------------------------------------------------------


def _is_venv_cmd(cmd: list[str]) -> bool:
    """A ``python -m venv <dir>`` command in either launcher form:
    ``python3.11 -m venv`` (Linux/macOS) or ``py -3.11 -m venv``
    (Windows) — the version flag shifts the indices, so match on the
    tail, not a fixed position."""
    return "-m" in cmd and len(cmd) >= 2 and cmd[-2] == "venv"


def _install_setup(monkeypatch, tmp_path, *, with_core_venv: bool):
    project = tmp_path / "project"
    project.mkdir()
    # The installer validates the project root before doing anything.
    (project / "pyproject.toml").write_text("[project]\nname = \"voice-generator\"\n")
    (project / "src" / "voice_generator").mkdir(parents=True)
    if with_core_venv:
        python = check.venv_python(project / ".venv-core")
        python.parent.mkdir(parents=True)
        python.write_text("# stub\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_stream(cmd, cwd=None):
        argv = [str(c) for c in cmd]
        calls.append(argv)
        if _is_venv_cmd(argv):
            # Simulate venv creation: the installer checks the
            # interpreter exists afterwards.
            python = check.venv_python(Path(argv[-1]))
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("# stub\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(install, "run_stream", fake_stream)
    monkeypatch.setattr(check, "run", _fake_run_factory(project))
    monkeypatch.setattr(
        check, "check_backend", lambda p, b, py: [_pass_result(f"{b}-ok")]
    )
    return project, calls


def test_install_core_reuses_existing_venv(tmp_path, monkeypatch):
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=True)
    assert install.main(["--backend", "core", "--project", str(project)]) == 0
    assert not any(_is_venv_cmd(c) for c in calls), calls
    assert any(
        "--no-build-isolation" in c and c[-2:] == ["-e", "."] for c in calls
    ), calls


def test_install_core_creates_missing_venv(tmp_path, monkeypatch):
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=False)
    assert install.main(["--backend", "core", "--project", str(project)]) == 0
    assert any(_is_venv_cmd(c) for c in calls), calls
    venv_index = next(i for i, c in enumerate(calls) if _is_venv_cmd(c))
    pip_index = next(i for i, c in enumerate(calls) if "pip" in c)
    assert venv_index < pip_index


def test_install_core_creates_missing_venv_windows_forms(tmp_path, monkeypatch):
    """The Windows form of the same path: ``py -3.11 -m venv`` puts
    ``-m venv`` at indices 2–3, which the fake (and the assertion)
    used to miss — the two remaining Windows CI failures."""
    monkeypatch.setattr(check.sys, "platform", "win32")
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=False)
    assert install.main(["--backend", "core", "--project", str(project)]) == 0
    assert any(c[:2] == ["py", "-3.11"] and _is_venv_cmd(c) for c in calls), calls
    venv_index = next(i for i, c in enumerate(calls) if _is_venv_cmd(c))
    pip_index = next(i for i, c in enumerate(calls) if "pip" in c)
    assert venv_index < pip_index


def test_install_web_ensures_core_first(tmp_path, monkeypatch):
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=False)
    assert install.main(["--backend", "web", "--project", str(project)]) == 0
    pip_calls = [c for c in calls if "pip" in c and "-e" in c]
    assert pip_calls[0][-1] == ".", pip_calls
    assert pip_calls[1][-1] == ".[web]", pip_calls


def test_install_chatterbox_cuda_fix(tmp_path, monkeypatch):
    project, calls = _install_setup(monkeypatch, tmp_path, with_core_venv=True)
    python = check.venv_python(project / ".venv-chatterbox")
    python.parent.mkdir(parents=True)
    python.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(check, "cuda_diagnosis", lambda py, backend: "cpu-only")
    assert install.main(["--backend", "chatterbox", "--project", str(project)]) == 0
    cu121 = [c for c in calls if "download.pytorch.org" in " ".join(c)]
    assert len(cu121) == 1, calls
    assert any(c[-1] == ".[chatterbox]" for c in calls), calls


def test_install_missing_python_aborts_before_any_work(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = \"voice-generator\"\n")
    (project / "src" / "voice_generator").mkdir(parents=True)
    calls: list[list[str]] = []
    monkeypatch.setattr(install, "run_stream", lambda cmd, cwd=None: calls.append(cmd) or 0)
    monkeypatch.setattr(
        check,
        "run",
        lambda cmd, timeout=120, cwd=None: subprocess.CompletedProcess(cmd, 1, "", ""),
    )
    assert install.main(["--backend", "core", "--project", str(project)]) == 1
    assert calls == []
