"""Setup installer: create the venv, install the backend, verify it.

  py setup/install.py --backend core|chatterbox|qwen3|voxcpm|web [--recreate]

Idempotent: an existing venv is reused, and pip re-runs are safe once
dependencies are pinned (pyproject.toml pins the known-good set;
unpinned, a re-run can upgrade them). --recreate deletes the venv
first, then rebuilds it — the repair path when pip inside the venv is
broken. CosyVoice is not installed by this script — its install is
Conda + a repo clone; see SETUP.md step 6 (``check.py`` verifies it
instead).

Pip output streams live, so a long download shows progress instead of
looking frozen; the verification at the end is the same code ``check.py``
runs. Exit 0 when the backend verifies afterwards, 1 otherwise.

Stdlib only. The venv layout follows SETUP.md: core and Chatterbox on
Python 3.11, Qwen3 and VoxCPM2 on 3.12, web into the core venv. On
Windows the ``py`` launcher picks the version; elsewhere
``pythonX.Y`` is used. The layout (venv, Python, pip steps) lives in
one table in check.py — this script reads it, it does not keep a copy.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check  # noqa: E402

# The shared layout table (see check.py). "pip" steps are the arguments
# after `pip install` — the runner adds --no-build-isolation, so the
# venv must already have setuptools + wheel (see _ensure_build_backend).
BACKENDS = check.BACKENDS

# A hung pip (stalled download, hung post-install hook) must not hang
# the installer forever: 30 minutes covers a multi-GB first download.
RUN_TIMEOUT_S = 1800


def run_stream(cmd: list[str], *, cwd: Path | None = None, timeout: int = RUN_TIMEOUT_S) -> int:
    """Run ``cmd`` with output inherited, so a long install shows
    progress. Returns the exit code; never raises."""
    print(f"  $ {shlex.join(str(c) for c in cmd)}")
    try:
        return subprocess.run(cmd, cwd=cwd, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(
            f"  timed out after {timeout}s: {shlex.join(str(c) for c in cmd)}",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"  could not run {shlex.join(str(c) for c in cmd)}: {exc}", file=sys.stderr)
        return 1


def ensure_python(major: int, minor: int) -> bool:
    """The Python this backend needs is installed."""
    result = check.check_python(major, minor)
    if result.status == "PASS":
        print(f"  Python {major}.{minor} found ({result.detail})")
        return True
    print(
        f"  Python {major}.{minor} not found — install it first "
        "(py launcher on Windows, python3.11/3.12 elsewhere)",
        file=sys.stderr,
    )
    return False


def ensure_venv(
    project: Path, venv_name: str, major: int, minor: int, *, recreate: bool = False
) -> Path | None:
    """Create the venv if it is missing; reuse it when it exists.
    With recreate=True, delete the existing venv first — the repair
    path when pip inside it is broken (a venv whose interpreter exists
    but whose pip is broken fails every re-run identically otherwise).
    Returns the venv's Python, or None when creation failed."""
    venv_dir = project / venv_name
    if recreate and venv_dir.exists():
        print(f"  deleting existing venv {venv_name} (--recreate)")
        shutil.rmtree(venv_dir)
    python = check.venv_python(venv_dir)
    if python.is_file():
        print(f"  reusing existing venv {venv_name}")
        return python
    print(f"  creating venv {venv_name} (Python {major}.{minor})")
    code = run_stream([*check.launcher(major, minor), "-m", "venv", str(venv_dir)])
    if code != 0 or not python.is_file():
        print(f"  venv creation failed (exit {code})", file=sys.stderr)
        return None
    return python


def _ensure_build_backend(python: Path, project: Path) -> bool:
    """setuptools + wheel exist in the venv: the editable installs below
    run with --no-build-isolation, so pip will not fetch a build backend
    for them — and a venv created by `python -m venv` on 3.12+ does not
    ship setuptools. Installs them when missing (a no-op when present)."""
    proc = check.run([str(python), "-c", "import setuptools, wheel"])
    if proc.returncode == 0:
        return True
    print("  installing setuptools + wheel (needed for --no-build-isolation)")
    code = run_stream(
        [str(python), "-m", "pip", "install", "setuptools", "wheel"], cwd=project
    )
    if code != 0:
        print(f"  setuptools/wheel install failed (exit {code})", file=sys.stderr)
        return False
    return True


def _print_torch_version(python: Path) -> None:
    """Print the torch version now installed, so a later drift is
    visible — the cu121 install takes the latest torch on that index,
    and the version is what a future re-run would have to match."""
    probe = check.probe_torch(python)
    version = probe.get("torch_version", "")
    if version:
        print(f"  CUDA torch installed (torch {version})")
    else:
        print("  CUDA torch installed (could not read the torch version)")


def install_backend(project: Path, backend: str, *, recreate: bool = False) -> int:
    """Install one backend and verify it. Returns a process exit code."""
    spec = BACKENDS[backend]
    major, minor = spec["python"]  # type: ignore[misc]
    venv_name = str(spec["venv"])

    if not ensure_python(major, minor):
        return 1
    if spec.get("needs_core") and backend != "core":
        # Web lives in the core venv: install core first — unless the
        # core venv already verifies, in which case re-running its whole
        # install (and re-resolving its deps) is wasted work.
        core_spec = BACKENDS["core"]
        core_python = check.venv_python(project / str(core_spec["venv"]))
        core_ok = False
        if core_python.is_file():
            core_results = check.check_backend(project, "core", core_python)
            core_ok = all(r.status == "PASS" for r in core_results)
        if core_ok:
            print("  core already installed and verified — skipping core install")
        else:
            print("  web lives in .venv-core — installing core first")
            if install_backend(project, "core", recreate=recreate) != 0:
                return 1
    python = ensure_venv(project, venv_name, major, minor, recreate=recreate)
    if python is None:
        return 1
    if not _ensure_build_backend(python, project):
        return 1
    for step in spec["pip"]:  # type: ignore[union-attr]
        code = run_stream(
            [str(python), "-m", "pip", "install", "--no-build-isolation", *step],
            cwd=project,
        )
        if code != 0:
            print(f"  install step failed (exit {code}) — see the output above", file=sys.stderr)
            print(
                f"  if this venv is broken, delete {project / venv_name} and re-run "
                "(or use --recreate)",
                file=sys.stderr,
            )
            return 1
    if spec.get("cuda_fix"):
        # A CPU-only torch build renders on CPU (Qwen3 looks hung) — the
        # documented fix, applied here instead of diagnosed later. Only
        # auto-reinstall when torch imports but CUDA is False: a missing
        # or broken torch needs a (re)install of the backend, not a CUDA
        # torch overlay on top of it — verification below is the arbiter
        # for those cases.
        diagnosis = check.cuda_diagnosis(python, backend)
        if diagnosis == "cpu-only":
            print("  torch is CPU-only — installing the CUDA build")
            code = run_stream(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    check.CU121_URL,
                ],
                cwd=project,
            )
            if code != 0:
                print(f"  CUDA torch install failed (exit {code})", file=sys.stderr)
                return 1
            _print_torch_version(python)
        elif diagnosis != "ok":
            print(
                f"  torch looks {diagnosis} in this venv — verification below will say more",
                file=sys.stderr,
            )
    print(f"  verifying {backend} …")
    results = check.check_backend(project, backend, python)
    return check.report(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the venv, install the backend, verify it."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=sorted(BACKENDS),
        help="which backend to install (web goes into the core venv)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="project root (default: the parent of setup/)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete the venv first, then recreate it from scratch "
        "(use when pip inside the venv is broken)",
    )
    args = parser.parse_args(argv)
    project = args.project
    if not (project / "pyproject.toml").is_file() or not (
        project / "src" / "voice_generator"
    ).is_dir():
        print(
            f"  {project} is not a voice-generator checkout "
            "(expected pyproject.toml and src/voice_generator/)",
            file=sys.stderr,
        )
        return 1
    print(f"voice-generator setup install — {args.backend} in {project}")
    return install_backend(project, args.backend, recreate=args.recreate)


if __name__ == "__main__":
    raise SystemExit(main())
