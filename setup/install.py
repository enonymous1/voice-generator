"""Setup installer: create the venv, install the backend, verify it.

  py setup/install.py --backend core|chatterbox|qwen3|voxcpm|web

Idempotent: an existing venv is reused, pip install re-runs harmlessly.
Never deletes anything. CosyVoice is not installed by this script — its
install is Conda + a repo clone; see SETUP.md step 6 (``check.py``
verifies it instead).

Pip output streams live, so a long download shows progress instead of
looking frozen; the verification at the end is the same code ``check.py``
runs. Exit 0 when the backend verifies afterwards, 1 otherwise.

Stdlib only. The venv layout follows SETUP.md: core and Chatterbox on
Python 3.11, Qwen3 and VoxCPM2 on 3.12, web into the core venv. On
Windows the ``py`` launcher picks the version; elsewhere
``pythonX.Y`` is used.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check  # noqa: E402

CU121_URL = "https://download.pytorch.org/whl/cu121"

# backend: Python version, venv dir, pip steps (each a venv-python argv),
# whether a CPU-only torch gets the cu121 reinstall, and whether the
# core venv must exist first (web lives in it).
BACKENDS: dict[str, dict[str, object]] = {
    "core": {
        "python": (3, 11),
        "venv": ".venv-core",
        "pip": [["-m", "pip", "install", "-e", "."]],
        "cuda_fix": False,
    },
    "chatterbox": {
        "python": (3, 11),
        "venv": ".venv-chatterbox",
        "pip": [["-m", "pip", "install", "-e", ".[chatterbox]"]],
        "cuda_fix": True,
    },
    "qwen3": {
        "python": (3, 12),
        "venv": ".venv-qwen3",
        "pip": [["-m", "pip", "install", "-e", ".[qwen3]"]],
        "cuda_fix": True,
    },
    "voxcpm": {
        "python": (3, 12),
        "venv": ".venv-voxcpm",
        "pip": [
            [
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                CU121_URL,
            ],
            ["-m", "pip", "install", "-e", ".[voxcpm]"],
        ],
        "cuda_fix": True,
    },
    "web": {
        "python": (3, 11),
        "venv": ".venv-core",
        "pip": [["-m", "pip", "install", "-e", ".[web]"]],
        "cuda_fix": False,
        "needs_core": True,
    },
}


def run_stream(cmd: list[str], *, cwd: Path | None = None) -> int:
    """Run ``cmd`` with output inherited, so a long install shows
    progress. Returns the exit code; never raises."""
    print(f"  $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, cwd=cwd).returncode
    except OSError as exc:
        print(f"  could not run {' '.join(cmd)}: {exc}", file=sys.stderr)
        return 1


def ensure_python(major: int, minor: int) -> bool:
    """The Python this backend needs is installed."""
    proc = check.run([*check.launcher(major, minor), "--version"])
    version = (proc.stdout or proc.stderr).strip()
    if proc.returncode == 0 and version.startswith(f"Python {major}.{minor}"):
        print(f"  Python {major}.{minor} found ({version})")
        return True
    print(
        f"  Python {major}.{minor} not found — install it first "
        "(py launcher on Windows, python3.11/3.12 elsewhere)",
        file=sys.stderr,
    )
    return False


def ensure_venv(project: Path, venv_name: str, major: int, minor: int) -> Path | None:
    """Create the venv if it is missing; reuse it when it exists.
    Returns the venv's Python, or None when creation failed."""
    python = check.venv_python(project / venv_name)
    if python.is_file():
        print(f"  reusing existing venv {venv_name}")
        return python
    print(f"  creating venv {venv_name} (Python {major}.{minor})")
    code = run_stream([*check.launcher(major, minor), "-m", "venv", str(project / venv_name)])
    if code != 0 or not python.is_file():
        print(f"  venv creation failed (exit {code})", file=sys.stderr)
        return None
    return python


def install_backend(project: Path, backend: str) -> int:
    """Install one backend and verify it. Returns a process exit code."""
    spec = BACKENDS[backend]
    major, minor = spec["python"]  # type: ignore[misc]
    venv_name = str(spec["venv"])

    if not ensure_python(major, minor):
        return 1
    if spec.get("needs_core") and backend != "core":
        # Web lives in the core venv: make sure the core is installed
        # there first, so `voice-gen serve` exists to run.
        print("  web lives in .venv-core — installing core first")
        if install_backend(project, "core") != 0:
            return 1
    python = ensure_venv(project, venv_name, major, minor)
    if python is None:
        return 1
    for step in spec["pip"]:  # type: ignore[union-attr]
        code = run_stream([str(python), *step], cwd=project)
        if code != 0:
            print(f"  install step failed (exit {code}) — see the output above", file=sys.stderr)
            return 1
    if spec.get("cuda_fix"):
        # A CPU-only torch build renders on CPU (Qwen3 looks hung) — the
        # documented fix, applied here instead of diagnosed later.
        cuda = check.check_cuda(python, backend)
        if cuda.status != "PASS":
            print("  torch is CPU-only — installing the CUDA build")
            code = run_stream(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    CU121_URL,
                ],
                cwd=project,
            )
            if code != 0:
                print(f"  CUDA torch install failed (exit {code})", file=sys.stderr)
                return 1
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
    args = parser.parse_args(argv)
    print(f"voice-generator setup install — {args.backend} in {args.project}")
    return install_backend(args.project, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
