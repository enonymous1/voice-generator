"""Setup checker: verify each SETUP.md step, print the fix for each gap.

Writes nothing except a temp dir for the mock render (deleted
afterwards) — and the Python bytecode caches that running the project
creates, which .gitignore covers. It never installs or downloads
anything. Run it from anywhere:

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
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CU121_URL = "https://download.pytorch.org/whl/cu121"

# The one layout table: venv dir, Python version, import module, and
# the pip steps install.py runs for that backend. install.py reads
# this table (it used to keep a second copy, and the two could drift).
# "module" is None for core/web — they have no backend package to
# import; their checks are doctor/mock (core) and flask (web).
BACKENDS: dict[str, dict[str, object]] = {
    "core": {
        "venv": ".venv-core",
        "python": (3, 11),
        "module": None,
        "pip": [["-e", "."]],
        "cuda_fix": False,
    },
    "chatterbox": {
        "venv": ".venv-chatterbox",
        "python": (3, 11),
        "module": "chatterbox_tts",
        "pip": [["-e", ".[chatterbox]"]],
        "cuda_fix": True,
    },
    "qwen3": {
        "venv": ".venv-qwen3",
        "python": (3, 12),
        "module": "qwen_tts",
        "pip": [["-e", ".[qwen3]"]],
        "cuda_fix": True,
    },
    "voxcpm": {
        "venv": ".venv-voxcpm",
        "python": (3, 12),
        "module": "voxcpm",
        "pip": [
            ["torch", "torchvision", "torchaudio", "--index-url", CU121_URL],
            ["-e", ".[voxcpm]"],
        ],
        "cuda_fix": True,
    },
    "web": {
        "venv": ".venv-core",
        "python": (3, 11),
        "module": None,
        "pip": [["-e", ".[web]"]],
        "cuda_fix": False,
        "needs_core": True,
    },
}
MIN_REF_S = 3.0
IDEAL_REF_S = 10.0


@dataclass
class Result:
    """One check's outcome."""

    id: str
    label: str
    status: Literal["PASS", "FAIL", "SKIP"]
    detail: str
    fix: str = ""
    optional: bool = False


def run(cmd: list[str], *, timeout: int = 120, cwd: Path | None = None):
    """Run ``cmd``; never raises — a missing binary or timeout is a failed run."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout, cwd=cwd
        )
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
    print their version and exit non-zero, so a version-shaped first
    line counts as an answer too — a missing binary prints nothing.
    The match must look like a version ("ffmpeg version 8.1.2"), not
    just mention the word: a tool failing with "bad version flag"
    must not pass."""
    proc = run([name, *(args or ["--version"])])
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    first = output.splitlines()
    first_line = first[0] if first else ""
    if proc.returncode == 0 or re.search(r"version \d", first_line, re.I):
        return _ok(name, name, first_line or "found")
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


def check_venv_python_version(python: Path, backend: str, expected: tuple[int, int]) -> Result:
    """The venv's interpreter is the Python version the backend's layout
    requires — a .venv-qwen3 created on 3.11 would otherwise pass every
    check while violating the documented layout. Overrides are exempt:
    pointing a backend at a shared venv is an explicit choice, and one
    shared venv cannot be every backend's version at once."""
    major, minor = expected
    proc = run([str(python), "--version"])
    version = (proc.stdout or proc.stderr).strip()
    if proc.returncode == 0 and version.startswith(f"Python {major}.{minor}"):
        return _ok(f"venv-python-{backend}", f"venv Python ({backend})", version)
    return _fail(
        f"venv-python-{backend}",
        f"venv Python ({backend})",
        f"expected Python {major}.{minor}, got {version or 'no version output'}",
        f"recreate the venv with py -{major}.{minor} "
        f"(py setup/install.py --backend {backend} --recreate)",
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
    pipeline (chunking, WAV write, manifest) works before any model.
    Both outputs count: a render that writes the WAV but no readable
    manifest is a failure, not a pass."""
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
            manifest = out.with_suffix(".manifest.json")
            if not manifest.is_file():
                return _fail(
                    "core-mock",
                    "mock render",
                    "WAV written but its manifest is missing",
                    "py setup/install.py --backend core, then read the error above",
                )
            try:
                json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return _fail(
                    "core-mock",
                    "mock render",
                    f"manifest unreadable ({exc})",
                    "py setup/install.py --backend core, then read the error above",
                )
            return _ok("core-mock", "mock render", "pipeline renders a WAV + manifest")
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return _fail(
        "core-mock",
        "mock render",
        tail[-1] if tail else "mock render failed",
        "py setup/install.py --backend core, then read the error above",
    )


def probe_torch(python: Path, module: str | None = None) -> dict[str, str]:
    """One interpreter spawn answering the torch/CUDA and import questions.

    Runs a small ``-c`` script in ``python`` and parses its stdout lines
    into a dict: ``torch`` is "ok" or "missing", ``cuda`` is "True"/"False"
    (only when torch imports), ``torch_version`` is the torch version
    (same condition), and ``import`` is "ok"/"fail" (only when a
    ``module`` was given and torch imports). ``error`` holds the last
    line of whatever failed. A missing key means the probe itself died
    (timeout, crash) — callers treat that as an error, never a PASS.
    One spawn serves both checks, where two used to cost a torch import
    each (~2–10 s on Windows)."""
    lines = [
        "import sys",
        "try:",
        "    import torch",
        "except Exception as exc:",
        "    print('torch: missing')",
        "    print('error:', (str(exc).splitlines() or [type(exc).__name__])[-1])",
        "else:",
        "    print('torch: ok')",
        "    print('torch_version:', torch.__version__)",
        "    print('cuda:', torch.cuda.is_available())",
    ]
    if module:
        lines += [
            "    try:",
            f"        import {module}",
            "    except Exception as exc:",
            "        print('import: fail')",
            "        print('error:', (str(exc).splitlines() or [type(exc).__name__])[-1])",
            "    else:",
            "        print('import: ok')",
        ]
    proc = run([str(python), "-c", "\n".join(lines)], timeout=180)
    out: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in ("torch", "torch_version", "cuda", "import", "error"):
                out[key] = value
    if "torch" not in out:
        # The probe never printed: it died (timeout, crash) — keep the
        # last stderr line as the error so the caller can show it.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        out["error"] = tail[-1] if tail else "torch probe produced no output"
    return out


def _cuda_case(probe: dict[str, str], backend: str) -> tuple[str, str, str]:
    """Classify a parsed torch probe: (kind, detail, fix). kind is one
    of "ok", "no-torch" (torch does not import at all), "cpu-only"
    (torch imports but CUDA is unavailable), or "error" (anything else
    — a broken install, missing CUDA DLLs — where the cu121 reinstall
    would be the wrong prescription)."""
    if probe.get("torch") == "ok" and probe.get("cuda") == "True":
        return "ok", "torch.cuda.is_available() is True", ""
    if probe.get("torch") == "missing":
        error = probe.get("error", "")
        if "No module named 'torch'" in error or 'No module named "torch"' in error:
            return (
                "no-torch",
                "torch is not installed in this venv",
                f"py setup/install.py --backend {backend}",
            )
        return (
            "error",
            f"torch import failed ({error})",
            "the torch install looks broken — read the error above; try "
            f"recreating the venv (py setup/install.py --backend {backend} --recreate)",
        )
    if probe.get("torch") == "ok":
        if "cuda" not in probe:
            return (
                "error",
                f"CUDA probe failed ({probe.get('error', 'no output')})",
                "the torch install looks broken — read the error above; try "
                f"recreating the venv (py setup/install.py --backend {backend} --recreate)",
            )
        return (
            "cpu-only",
            "torch is CPU-only",
            f"pip install torch torchvision torchaudio --index-url {CU121_URL} "
            f"in the {backend} venv (or: py setup/install.py --backend {backend})",
        )
    return (
        "error",
        f"torch probe failed ({probe.get('error', 'no output')})",
        "the torch install looks broken — read the error above; try "
        f"recreating the venv (py setup/install.py --backend {backend} --recreate)",
    )


def _cuda_result(probe: dict[str, str], backend: str) -> Result:
    """Build the CUDA-torch Result from a parsed probe (see probe_torch)."""
    kind, detail, fix = _cuda_case(probe, backend)
    if kind == "ok":
        return _ok(f"cuda-{backend}", f"CUDA torch ({backend})", detail)
    return _fail(f"cuda-{backend}", f"CUDA torch ({backend})", detail, fix)


def _import_result(probe: dict[str, str], backend: str, module: str) -> Result:
    """Build the backend-import Result from a parsed probe (see probe_torch)."""
    if probe.get("import") == "ok":
        return _ok(f"import-{backend}", f"import {module}", f"{module} imports in its venv")
    if probe.get("torch") == "missing":
        return _fail(
            f"import-{backend}",
            f"import {module}",
            "not installed in this venv",
            f"py setup/install.py --backend {backend}",
        )
    return _fail(
        f"import-{backend}",
        f"import {module}",
        probe.get("error", "import failed") or "import failed",
        f"py setup/install.py --backend {backend}",
    )


def cuda_diagnosis(python: Path, backend: str) -> str:
    """Classify a venv's torch/CUDA state for the installer: "ok",
    "no-torch", "cpu-only", or "error" (see _cuda_case). One probe spawn.
    The installer only auto-reinstalls torch on "cpu-only" — a missing
    or broken torch needs a (re)install of the backend, not a CUDA torch
    overlay on top of it."""
    return _cuda_case(probe_torch(python), backend)[0]


def check_cuda(python: Path, backend: str) -> Result:
    """Torch in this venv sees CUDA — a CPU-only torch renders on CPU
    (Qwen3 looks hung for 10+ minutes) instead of the GPU. The failure
    is classified, not lumped: missing torch, CPU-only torch, and a
    broken install each get their own detail and fix."""
    return _cuda_result(probe_torch(python), backend)


def check_import(python: Path, backend: str, module: str) -> Result:
    """The backend's package imports in its venv."""
    return _import_result(probe_torch(python, module), backend, module)


def check_cuda_and_import(python: Path, backend: str, module: str) -> tuple[Result, Result]:
    """The CUDA-torch and backend-import checks from a single probe
    spawn — one torch import serves both."""
    probe = probe_torch(python, module)
    return _cuda_result(probe, backend), _import_result(probe, backend, module)


def check_ref_clip(project: Path) -> Result:
    """refs/me.wav exists and is a usable reference clip. The thresholds
    follow check_reference (missing, unreadable, or under 3 s fails;
    stereo, low rate, short, or over-long clips pass with a note) — but
    NOT its sample-width handling: check_reference never inspects width,
    and 24/32-bit integer PCM reads fine (stdlib wave, torchaudio), so
    a wide clip passes here with a note instead of failing."""
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
    if width != 2:
        notes.append(f"{width * 8}-bit — usable, PCM-16 is the tested format")
    detail = f"{duration:.1f}s, {rate} Hz, {channels}ch"
    if notes:
        detail += " (" + "; ".join(notes) + ")"
    return _ok("ref-clip", "reference clip refs/me.wav", detail)


def check_ref_transcript(project: Path) -> Result:
    """refs/me.txt exists and is non-empty — the transcript Qwen3,
    CosyVoice, and VoxCPM ultimate need. A transcript saved in another
    encoding (e.g. Windows-1252 from Notepad) fails with its fix
    instead of crashing the run: the backends read it as UTF-8 too."""
    transcript = project / "refs" / "me.txt"
    if transcript.is_file():
        try:
            text = transcript.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _fail(
                "ref-transcript",
                "reference transcript refs/me.txt",
                "not valid UTF-8",
                "re-save refs/me.txt as UTF-8 (Notepad: Save As, Encoding: UTF-8)",
            )
        except OSError as exc:
            return _fail(
                "ref-transcript",
                "reference transcript refs/me.txt",
                f"unreadable ({exc})",
                "check the file's permissions",
            )
        if text.strip():
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
    without CosyVoice shows skips, never failures. The import/model/
    Matcha checks only run when their prerequisite exists — a clone
    directory and an env name are not proof the backend works."""
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
        for check_id, label in (
            ("conda", "Conda"),
            ("cosyvoice-clone", "CosyVoice clone"),
            ("cosyvoice-env", "CosyVoice env"),
            ("cosyvoice-import", "CosyVoice import"),
            ("cosyvoice-model", "CosyVoice model"),
            ("cosyvoice-matcha", "CosyVoice Matcha-TTS"),
        ):
            results.append(_skip(check_id, label, "not found — CosyVoice checks skipped"))
        return results
    version = (conda.stdout or conda.stderr).strip().splitlines()
    results.append(_ok("conda", "Conda", version[0] if version else "found", optional=True))

    home = os.environ.get("COSYVOICE_HOME")
    if home:
        # A relative COSYVOICE_HOME is relative to the project checkout's
        # parent (where the clone normally sits next to it), not the
        # caller's working directory.
        home_path = Path(home)
        clone = home_path if home_path.is_absolute() else project.parent / home_path
    else:
        clone = project.parent / "CosyVoice"
    clone_ok = clone.is_dir()
    if clone_ok:
        results.append(_ok("cosyvoice-clone", "CosyVoice clone", str(clone), optional=True))
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
    env_ok = bool(
        envs.returncode == 0 and re.search(r"^cosyvoice\s", envs.stdout or "", re.MULTILINE)
    )
    if env_ok:
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
    if env_ok:
        import_proc = run(
            ["conda", "run", "-n", "cosyvoice", "python", "-c", "import cosyvoice"],
            timeout=180,
        )
        if import_proc.returncode == 0:
            results.append(
                _ok(
                    "cosyvoice-import",
                    "CosyVoice import",
                    "cosyvoice imports in the cosyvoice env",
                    optional=True,
                )
            )
        else:
            tail = (import_proc.stderr or import_proc.stdout).strip().splitlines()
            results.append(
                _fail(
                    "cosyvoice-import",
                    "CosyVoice import",
                    tail[-1] if tail else "cosyvoice does not import in the cosyvoice env",
                    "pip install -r requirements.txt in the CosyVoice clone (SETUP.md step 6)",
                    optional=True,
                )
            )
    else:
        results.append(_skip("cosyvoice-import", "CosyVoice import", "needs the cosyvoice env"))
    if clone_ok:
        model_env = os.environ.get("COSYVOICE_MODEL_DIR")
        if model_env:
            model_path = Path(model_env)
            candidates = [model_path if model_path.is_absolute() else clone / model_path]
        else:
            candidates = [
                clone / "pretrained_models" / "Fun-CosyVoice3-0.5B",
                clone / "pretrained_models" / "CosyVoice2-0.5B",
            ]
        found = next((c for c in candidates if c.is_dir()), None)
        if found is not None:
            results.append(
                _ok("cosyvoice-model", "CosyVoice model", str(found), optional=True)
            )
        else:
            results.append(
                _fail(
                    "cosyvoice-model",
                    "CosyVoice model",
                    f"{candidates[0]} not found",
                    "download the Fun-CosyVoice3-0.5B-2512 snapshot into the clone "
                    "(SETUP.md step 6)",
                    optional=True,
                )
            )
        matcha_dir = clone / "third_party" / "Matcha-TTS"
        if matcha_dir.is_dir():
            results.append(
                _ok("cosyvoice-matcha", "CosyVoice Matcha-TTS", str(matcha_dir), optional=True)
            )
        else:
            results.append(
                _fail(
                    "cosyvoice-matcha",
                    "CosyVoice Matcha-TTS",
                    f"{matcha_dir} not found",
                    "git submodule update --init --recursive in the CosyVoice clone "
                    "(SETUP.md step 6)",
                    optional=True,
                )
            )
    else:
        results.append(_skip("cosyvoice-model", "CosyVoice model", "needs the CosyVoice clone"))
        results.append(
            _skip("cosyvoice-matcha", "CosyVoice Matcha-TTS", "needs the CosyVoice clone")
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
            # No py launcher (e.g. a Store-only Python): a plain
            # `python` on PATH is the same interpreter, so probe that
            # before failing the check.
            fallback = run(["python", "--version"])
            fversion = (fallback.stdout or fallback.stderr).strip()
            if fallback.returncode == 0 and fversion.startswith("Python"):
                results.append(
                    _ok("py-launcher", "py launcher", f"{fversion} (python.exe — no py launcher)")
                )
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
    results.append(check_tool("git", fix="install Git (https://git-scm.com/downloads)"))
    results.append(
        check_tool(
            "ffmpeg",
            fix="install ffmpeg and put it on PATH (https://ffmpeg.org/download.html)",
        )
    )

    core_python = overrides.get("core") or venv_python(project / ".venv-core")
    venv_result = check_venv(core_python, ".venv-core", fix="py setup/install.py --backend core")
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
        module = spec.get("module")
        if module is None:
            continue  # core/web have no backend package to check here
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
                _skip(f"import-{backend}", f"import {module}", f"needs {venv_name}")
            )
            continue
        if backend not in overrides:
            version_result = check_venv_python_version(
                python, backend, spec["python"]  # type: ignore[arg-type]
            )
            results.append(version_result)
            if version_result.status != "PASS":
                results.append(
                    _skip(
                        f"cuda-{backend}",
                        f"CUDA torch ({backend})",
                        f"needs Python {spec['python']}",
                    )
                )
                results.append(
                    _skip(
                        f"import-{backend}",
                        f"import {module}",
                        f"needs Python {spec['python']}",
                    )
                )
                continue
        cuda_result, import_result = check_cuda_and_import(python, backend, str(module))
        results.append(cuda_result)
        results.append(import_result)

    results.append(check_ref_clip(project))
    results.append(check_ref_transcript(project))
    results.extend(check_cosyvoice(project))
    return results


def check_backend(project: Path, backend: str, python: Path) -> list[Result]:
    """The checks for one backend (used by install.py after installing
    it): doctor + mock render for core, flask import for web, CUDA torch
    + backend import for the rest. The venv itself is NOT checked here —
    the caller must have created it first (install.py does), and the
    venv's Python version is NOT checked either (collect() does that)."""
    if backend == "core":
        return [check_doctor(python), check_mock_render(project, python)]
    if backend == "web":
        return [check_web(python)]
    spec = BACKENDS.get(backend)
    if spec is None or spec.get("module") is None:
        raise ValueError(f"unknown backend: {backend}")
    return list(check_cuda_and_import(python, backend, str(spec["module"])))


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
