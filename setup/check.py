"""Setup checker: verify each SETUP.md step, print the fix for each gap.

Read-only: it never installs, downloads, or writes anything, except a
temporary file for the mock render (deleted afterwards). Run it from
anywhere:

  py setup/check.py [--project PATH]

Exit code 0 when every required check passes, 1 otherwise. Optional
checks (CosyVoice) never fail the run: a missing optional piece reports
SKIP, a present one PASS.

Per-backend venvs can be overridden, for example when several backends
share one venv:

  py setup/check.py --chatterbox-python .venv/Scripts/python.exe

Stdlib only, so it runs under any Python — including one without the
project installed. The checks that need a venv shell out to that venv's
own Python; the reference-clip check reads the WAV with the stdlib
``wave`` module and applies the same thresholds as
``voice_generator.audio.check_reference`` (3 s minimum, 10 s ideal,
16 kHz+, mono).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

# Backends the checker verifies: venv dir, Python version, import module.
BACKENDS: dict[str, dict[str, object]] = {
    "chatterbox": {"venv": ".venv-chatterbox", "python": (3, 11), "module": "chatterbox_tts"},
    "qwen3": {"venv": ".venv-qwen3", "python": (3, 12), "module": "qwen_tts"},
    "voxcpm": {"venv": ".venv-voxcpm", "python": (3, 12), "module": "voxcpm"},
}
CU121_URL = "https://download.pytorch.org/whl/cu121"
MIN_REF_S = 3.0
IDEAL_REF_S = 10.0


@dataclass
class Result:
    """One check's outcome."""

    id: str
    label: str
    status: str  # PASS, FAIL, or SKIP
    detail: str
    fix: str = ""
    optional: bool = False


def run(cmd: list[str], *, timeout: int = 120, cwd: Path | None = None):
    """Run ``cmd``; never raises — a missing binary or timeout is a failed run."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(cmd, 1, "", str(exc))


def launcher(major: int, minor: int) -> list[str]:
    """The command that runs a given Python: the ``py`` launcher on
    Windows, ``pythonX.Y`` elsewhere."""
    if sys.platform == "win32":
        return ["py", f"-{major}.{minor}"]
    return [f"python{major}.{minor}"]


def venv_python(venv_dir: Path) -> Path:
    """A venv's interpreter: ``Scripts\\python.exe`` on Windows, ``bin/python`` elsewhere."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ok(check_id: str, label: str, detail: str, *, optional: bool = False) -> Result:
    return Result(check_id, label, "PASS", detail, optional=optional)


def _fail(check_id: str, label: str, detail: str, fix: str, *, optional: bool = False) -> Result:
    return Result(check_id, label, "FAIL", detail, fix, optional=optional)


def _skip(check_id: str, label: str, detail: str) -> Result:
    return Result(check_id, label, "SKIP", detail, optional=True)


def check_python(major: int, minor: int) -> Result:
    """A given Python version is installed and runnable."""
    check_id = f"python-{major}.{minor}"
    proc = run([*launcher(major, minor), "--version"])
    version = (proc.stdout or proc.stderr).strip()
    if proc.returncode == 0 and version.startswith(f"Python {major}.{minor}"):
        return _ok(check_id, f"Python {major}.{minor}", version)
    return _fail(
        check_id,
        f"Python {major}.{minor}",
        "not found" if not version else version,
        f"install Python {major}.{minor} (py launcher on Windows, python{major}.{minor} elsewhere)",
    )


def check_tool(name: str, args: list[str] | None = None, *, fix: str) -> Result:
    """A command-line tool answers on PATH. Some tools (notably ffmpeg)
    print their version and exit non-zero, so the version text in the
    output counts as an answer too — a missing binary prints nothing."""
    proc = run([name, *(args or ["--version"])])
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode == 0 or "version" in output.lower():
        first = output.splitlines()
        return _ok(name, name, first[0] if first else "found")
    return _fail(name, name, "not found on PATH", fix)


def check_venv(python: Path, label: str, *, fix: str) -> Result:
    """A venv's interpreter exists (the documented venv, or the one an
    override points at)."""
    if python.is_file():
        return _ok(f"venv-{label}", f"venv {label}", str(python))
    return _fail(
        f"venv-{label}",
        f"venv {label}",
        f"{python} not found",
        fix,
    )


def check_doctor(python: Path) -> Result:
    """The core venv's CLI runs (``voice-gen doctor`` exits 0)."""
    proc = run([str(python), "-m", "voice_generator", "doctor"], timeout=180)
    if proc.returncode == 0:
        return _ok("core-doctor", "voice-gen doctor", "CLI runs in .venv-core")
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return _fail(
        "core-doctor",
        "voice-gen doctor",
        tail[-1] if tail else "doctor failed",
        "pip install -e . in .venv-core (or: py setup/install.py --backend core)",
    )


def check_mock_render(project: Path, python: Path) -> Result:
    """The mock backend renders end to end in the core venv — the
    pipeline (chunking, WAV write, manifest) works before any model."""
    with tempfile.TemporaryDirectory(prefix="vg-check-") as tmp:
        out = Path(tmp) / "smoke.wav"
        proc = run(
            [
                str(python),
                "-m",
                "voice_generator",
                "synth",
                "--backend",
                "mock",
                "--text",
                "Setup check.",
                "--out",
                str(out),
            ],
            timeout=180,
            cwd=project,
        )
        if proc.returncode == 0 and out.is_file():
            return _ok("core-mock", "mock render", "pipeline renders a WAV + manifest")
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return _fail(
        "core-mock",
        "mock render",
        tail[-1] if tail else "mock render failed",
        "py setup/install.py --backend core, then read the error above",
    )


def check_cuda(python: Path, backend: str) -> Result:
    """Torch in this venv sees CUDA — a CPU-only torch renders on CPU
    (Qwen3 looks hung for 10+ minutes) instead of the GPU."""
    proc = run(
        [str(python), "-c", "import torch; print(torch.cuda.is_available())"],
        timeout=180,
    )
    if proc.returncode == 0 and proc.stdout.strip() == "True":
        return _ok(
            f"cuda-{backend}", f"CUDA torch ({backend})", "torch.cuda.is_available() is True"
        )
    return _fail(
        f"cuda-{backend}",
        f"CUDA torch ({backend})",
        "torch is CPU-only or not installed",
        f"pip install torch torchvision torchaudio --index-url {CU121_URL} "
        f"in the {backend} venv (or: py setup/install.py --backend {backend})",
    )


def check_import(python: Path, backend: str, module: str) -> Result:
    """The backend's package imports in its venv."""
    proc = run([str(python), "-c", f"import {module}"], timeout=180)
    if proc.returncode == 0:
        return _ok(f"import-{backend}", f"import {module}", f"{module} imports in its venv")
    return _fail(
        f"import-{backend}",
        f"import {module}",
        "not installed in this venv",
        f"py setup/install.py --backend {backend}",
    )


def check_ref_clip(project: Path) -> Result:
    """refs/me.wav exists and is a usable reference clip. Mirrors
    check_reference: missing, unreadable, or under 3 s fails; stereo,
    low rate, short, or over-long clips pass with a note, exactly as
    check-ref warns instead of failing."""
    clip = project / "refs" / "me.wav"
    if not clip.is_file():
        return _fail(
            "ref-clip",
            "reference clip refs/me.wav",
            "not found",
            "record a clip (SETUP.md step 2) and save it as refs/me.wav",
        )
    try:
        with wave.open(str(clip), "rb") as w:
            channels = w.getnchannels()
            rate = w.getframerate()
            frames = w.getnframes()
            width = w.getsampwidth()
    except (wave.Error, EOFError, OSError) as exc:
        return _fail(
            "ref-clip",
            "reference clip refs/me.wav",
            f"unreadable WAV ({exc})",
            "ffmpeg -i refs/me.wav -c:a pcm_s16le refs/me_pcm16.wav "
            "(float WAVs, format tag 3, fail this way)",
        )
    if width != 2:
        return _fail(
            "ref-clip",
            "reference clip refs/me.wav",
            f"sample width is {width} bytes, not PCM-16",
            "ffmpeg -i refs/me.wav -c:a pcm_s16le refs/me_pcm16.wav",
        )
    duration = frames / rate if rate else 0.0
    notes: list[str] = []
    if duration < MIN_REF_S:
        return _fail(
            "ref-clip",
            "reference clip refs/me.wav",
            f"only {duration:.1f}s (< {MIN_REF_S:.0f}s minimum) — cloning will be poor",
            "re-record a longer clip (SETUP.md step 2)",
        )
    if duration < IDEAL_REF_S:
        notes.append(f"{duration:.1f}s — usable, but ~{IDEAL_REF_S:.0f}s gives a better clone")
    if duration > 60:
        notes.append(f"{duration:.0f}s — trim to the cleanest 10–30 s")
    if channels > 1:
        notes.append("stereo — most backends expect mono")
    if rate < 16000:
        notes.append(f"{rate} Hz — 16 kHz+ recommended")
    detail = f"{duration:.1f}s, {rate} Hz, {channels}ch"
    if notes:
        detail += " (" + "; ".join(notes) + ")"
    return _ok("ref-clip", "reference clip refs/me.wav", detail)


def check_ref_transcript(project: Path) -> Result:
    """refs/me.txt exists and is non-empty — the transcript Qwen3,
    CosyVoice, and VoxCPM ultimate need."""
    transcript = project / "refs" / "me.txt"
    if transcript.is_file() and transcript.read_text(encoding="utf-8").strip():
        return _ok("ref-transcript", "reference transcript refs/me.txt", "present")
    return _fail(
        "ref-transcript",
        "reference transcript refs/me.txt",
        "missing or empty",
        "copy what you actually read into refs/me.txt "
        "(the Setup page stores it for you when you record/upload there)",
    )


def check_web(python: Path) -> Result:
    """Flask is installed in the core venv, so ``voice-gen serve`` runs."""
    proc = run([str(python), "-c", "import flask"], timeout=120)
    if proc.returncode == 0:
        return _ok("web-ui", "web UI (flask)", "flask imports in .venv-core")
    return _fail(
        "web-ui",
        "web UI (flask)",
        "flask not installed in .venv-core",
        "py setup/install.py --backend web",
    )


def check_cosyvoice(project: Path) -> list[Result]:
    """CosyVoice checks — all optional: the backend is manual (Conda +
    a repo clone) and only earns its keep for multilingual work. Each
    check reports SKIP when its prerequisite is absent, so a machine
    without CosyVoice shows skips, never failures."""
    results: list[Result] = []
    sox = run(["sox", "--version"])
    if sox.returncode == 0:
        results.append(_ok("sox", "SoX", "sox found", optional=True))
    else:
        results.append(
            _fail(
                "sox",
                "SoX",
                "not found on PATH",
                "sox-14.4.2-win32.zip from SourceForge (SETUP.md step 6); "
                "`choco install sox` no longer works",
                optional=True,
            )
        )
    conda = run(["conda", "--version"])
    if conda.returncode != 0:
        results.append(_skip("conda", "Conda", "not found — CosyVoice checks skipped"))
        results.append(_skip("cosyvoice-clone", "CosyVoice clone", "needs Conda"))
        results.append(_skip("cosyvoice-env", "CosyVoice env", "needs Conda"))
        return results
    version = (conda.stdout or conda.stderr).strip().splitlines()
    results.append(
        _ok("conda", "Conda", version[0] if version else "found", optional=True)
    )
    import os

    home = os.environ.get("COSYVOICE_HOME")
    clone = Path(home) if home else project.parent / "CosyVoice"
    if clone.is_dir():
        results.append(
            _ok("cosyvoice-clone", "CosyVoice clone", str(clone), optional=True)
        )
    else:
        results.append(
            _fail(
                "cosyvoice-clone",
                "CosyVoice clone",
                f"{clone} not found",
                "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "
                "(SETUP.md step 6; or set COSYVOICE_HOME)",
                optional=True,
            )
        )
    envs = run(["conda", "env", "list"])
    if envs.returncode == 0 and re.search(r"^cosyvoice\s", envs.stdout or "", re.MULTILINE):
        results.append(
            _ok("cosyvoice-env", "CosyVoice env", "conda env cosyvoice exists", optional=True)
        )
    else:
        results.append(
            _fail(
                "cosyvoice-env",
                "CosyVoice env",
                "no conda env named cosyvoice",
                "conda create -n cosyvoice -y python=3.10 (SETUP.md step 6)",
                optional=True,
            )
        )
    return results


def collect(project: Path, overrides: dict[str, Path] | None = None) -> list[Result]:
    """Run every check, in SETUP.md order. ``overrides`` maps a backend
    name (core/chatterbox/qwen3/voxcpm) to that venv's Python."""
    overrides = overrides or {}
    results: list[Result] = []

    if sys.platform == "win32":
        proc = run(["py", "--version"])
        version = (proc.stdout or proc.stderr).strip()
        if proc.returncode == 0 and version.startswith("Python"):
            results.append(_ok("py-launcher", "py launcher", version))
        else:
            results.append(
                _fail(
                    "py-launcher",
                    "py launcher",
                    "not found",
                    "install Python from python.org (the launcher comes with it)",
                )
            )
    else:
        results.append(
            _skip(
                "py-launcher",
                "py launcher",
                "no py launcher on Linux/macOS — python3.11/3.12 below are the equivalent",
            )
        )
    results.append(check_python(3, 11))
    results.append(check_python(3, 12))
    results.append(
        check_tool("git", fix="install Git (https://git-scm.com/downloads)")
    )
    results.append(
        check_tool(
            "ffmpeg",
            fix="install ffmpeg and put it on PATH (https://ffmpeg.org/download.html)",
        )
    )

    core_python = overrides.get("core") or venv_python(project / ".venv-core")
    venv_result = check_venv(
        core_python, ".venv-core", fix="py setup/install.py --backend core"
    )
    results.append(venv_result)
    if venv_result.status == "PASS":
        results.append(check_doctor(core_python))
        results.append(check_mock_render(project, core_python))
        results.append(check_web(core_python))
    else:
        results.append(_skip("core-doctor", "voice-gen doctor", "needs .venv-core"))
        results.append(_skip("core-mock", "mock render", "needs .venv-core"))
        results.append(_skip("web-ui", "web UI (flask)", "needs .venv-core"))

    for backend, spec in BACKENDS.items():
        venv_name = str(spec["venv"])
        python = overrides.get(backend) or venv_python(project / venv_name)
        venv_result = check_venv(
            python, venv_name, fix=f"py setup/install.py --backend {backend}"
        )
        results.append(venv_result)
        if venv_result.status != "PASS":
            results.append(
                _skip(f"cuda-{backend}", f"CUDA torch ({backend})", f"needs {venv_name}")
            )
            results.append(
                _skip(f"import-{backend}", f"import {spec['module']}", f"needs {venv_name}")
            )
            continue
        results.append(check_cuda(python, backend))
        results.append(check_import(python, backend, str(spec["module"])))

    results.append(check_ref_clip(project))
    results.append(check_ref_transcript(project))
    results.extend(check_cosyvoice(project))
    return results


def check_backend(project: Path, backend: str, python: Path) -> list[Result]:
    """The checks for one backend (used by install.py after installing
    it): venv, CUDA torch, import — plus doctor/mock for core, flask
    for web."""
    if backend == "core":
        return [check_doctor(python), check_mock_render(project, python)]
    if backend == "web":
        return [check_web(python)]
    spec = BACKENDS[backend]
    return [
        check_cuda(python, backend),
        check_import(python, backend, str(spec["module"])),
    ]


def report(results: list[Result]) -> int:
    """Print the results; return 0 when every required check passes."""
    passed = failed = skipped = 0
    for r in results:
        status = r.status
        if r.optional and status == "FAIL":
            status = "SKIP"
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            skipped += 1
        print(f"[{status}] {r.label} — {r.detail}")
        if status == "FAIL" and r.fix:
            print(f"       fix: {r.fix}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (optional)")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify each SETUP.md step; print the fix for each gap."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="project root (default: the parent of setup/)",
    )
    for backend in ("core", "chatterbox", "qwen3", "voxcpm"):
        parser.add_argument(
            f"--{backend}-python",
            type=Path,
            default=None,
            help=f"python.exe of the {backend} venv (default: the documented venv)",
        )
    args = parser.parse_args(argv)
    project = args.project
    print(f"voice-generator setup check — {project}")
    overrides = {
        backend: getattr(args, f"{backend}_python")
        for backend in ("core", "chatterbox", "qwen3", "voxcpm")
        if getattr(args, f"{backend}_python") is not None
    }
    return report(collect(project, overrides))


if __name__ == "__main__":
    raise SystemExit(main())
