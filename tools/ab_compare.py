"""A/B compare: render the SAME script + reference through up to four
backends and write a side-by-side comparison manifest.

Why a subprocess tool instead of a library call: the backends live in
separate venvs (their torch stacks conflict), so no single interpreter can
import all of them. This tool drives each venv's interpreter with
`python -m voice_generator synth ...` and then compares the results
(duration, sample rate, wall time) from the manifests each run produced.

Usage:
  python tools/ab_compare.py --script script.txt --ref refs/me.wav ^
      --ref-text-file refs/me.txt ^
      --chatterbox-python .venv-chatterbox/Scripts/python.exe ^
      --qwen-python .venv-qwen3/Scripts/python.exe ^
      --cosy-python <cosyvoice-conda-python> --cosyvoice-home C:/path/to/CosyVoice ^
      --voxcpm-python .venv-voxcpm/Scripts/python.exe ^
      --out-dir output/ab1

  --demo-script renders the built-in demo script (the emotional-range
  paragraph, ``voice_generator.demo_script.DEMO_SCRIPT``) instead of a
  file; it needs the project installed in the Python running this tool
  (``pip install -e .`` in the core venv).

Any backend side may be omitted (that side is skipped), which also makes
this usable for a single-backend timing run. The VoxCPM2 side runs in
its default controllable mode (no transcript needed); pass
--ref-text-file as well to add an ultimate-mode side on the same clip.
Stdlib only — run it with any Python.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# A hung backend must not hang the comparison forever: the Qwen3
# CPU-only stall (12+ minutes at one line) is the shape this guards
# against. Sides run sequentially, so one hung side would otherwise
# block every later side.
DEFAULT_TIMEOUT_S = 1800


def _run_backend(
    *,
    label: str,
    python: Path,
    backend: str,
    script: Path,
    ref: Path,
    out_wav: Path,
    extra: list[str],
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[str, object]:
    """Render one backend side and return its comparison entry.

    On success: {"backend", "ok": True, "wall_s", "wav", "duration_s",
    "sample_rate", "chunks", "license"}. On failure: {"backend",
    "ok": False, "wall_s", "error"} — a failed side never kills the
    comparison; the JSON still records it and the other sides still
    run. ``wall_s`` is wall-clock seconds for the whole side, including
    model load (and any first-run weight download), so wall times
    across runs with different cache states are not comparable.
    """
    cmd = [
        str(python),
        "-m",
        "voice_generator",
        "synth",
        "--backend",
        backend,
        "--script",
        str(script),
        "--ref",
        str(ref),
        "--out",
        str(out_wav),
        *extra,
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",  # same pin as setup/check.py run(): the
            errors="replace",  # locale encoding (cp1252 on Windows)
            check=False,  # would decode non-UTF-8 bytes cleanly instead
            timeout=timeout,  # of replacing them
        )
    except subprocess.TimeoutExpired:
        wall = round(time.time() - started, 2)
        return {
            "backend": label,
            "ok": False,
            "wall_s": wall,
            "error": f"timed out after {timeout}s",
        }
    except OSError as exc:
        wall = round(time.time() - started, 2)
        return {
            "backend": label,
            "ok": False,
            "wall_s": wall,
            "error": f"cannot run {python}: {exc}",
        }
    wall = round(time.time() - started, 2)
    if proc.returncode != 0:
        error = (
            proc.stderr.strip()[-2000:]
            or proc.stdout.strip()[-2000:]
            or f"exit code {proc.returncode}"
        )
        return {"backend": label, "ok": False, "wall_s": wall, "error": error}
    try:
        if not out_wav.is_file() or out_wav.stat().st_size == 0:
            raise OSError(f"{out_wav} missing or empty after a successful render")
        manifest = json.loads(out_wav.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        duration_s = manifest["duration_s"]
        sample_rate = manifest["sample_rate"]
        chunks = len(manifest["chunks"])
        license_ = manifest["license"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return {
            "backend": label,
            "ok": False,
            "wall_s": wall,
            "error": f"unreadable manifest for {out_wav}: {exc}",
        }
    return {
        "backend": label,
        "ok": True,
        "wall_s": wall,
        "wav": str(out_wav),
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "chunks": chunks,
        "license": license_,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse args, validate them up front (before any rendering — a bad
    combination must not waste whole backend renders), render each
    requested side sequentially, and write ab_comparison.json. Returns 0
    when every side succeeded, 1 otherwise. Sides run sequentially, not
    concurrently: four model loads at once would OOM a 12 GB card."""
    p = argparse.ArgumentParser(
        description="Render one script through up to four backends and compare"
    )
    p.add_argument("--script", type=Path, default=None, help="script file to render")
    p.add_argument(
        "--demo-script",
        action="store_true",
        help="render the built-in demo script (emotional-range paragraph) instead of a file",
    )
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument(
        "--ref-text-file", type=Path, default=None, help="reference transcript (Qwen3 side)"
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--chatterbox-python", type=Path, default=None, help="python.exe of the Chatterbox venv"
    )
    p.add_argument("--qwen-python", type=Path, default=None, help="python.exe of the Qwen3 venv")
    p.add_argument(
        "--cosy-python", type=Path, default=None, help="python of the CosyVoice Conda env"
    )
    p.add_argument(
        "--voxcpm-python", type=Path, default=None, help="python.exe of the VoxCPM2 venv"
    )
    p.add_argument(
        "--voxcpm-mode",
        default="controllable",
        choices=["controllable", "ultimate"],
        help="VoxCPM2 clone mode for its side (ultimate also needs --ref-text-file)",
    )
    p.add_argument("--cosyvoice-home", default=None, help="CosyVoice repo clone (CosyVoice side)")
    p.add_argument("--cosy-model-dir", default=None, help="CosyVoice model dir (CosyVoice side)")
    p.add_argument("--variant", default="turbo")
    p.add_argument("--language", default="en", help="language code passed to every backend side")
    p.add_argument("--model-size", default="1.7b", choices=["1.7b", "0.6b"])
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="seconds a backend side may run before it is recorded as timed out "
        f"(default: {DEFAULT_TIMEOUT_S})",
    )
    args = p.parse_args(argv)
    # All argument validation happens here, before the first side runs:
    # a bad combination used to be caught inside the VoxCPM block, after
    # the other sides had already rendered (minutes of GPU time, then
    # discarded with no record at all).
    if bool(args.script) == bool(args.demo_script):
        p.error("give exactly one of --script <file> or --demo-script")
    if args.voxcpm_mode == "ultimate" and not args.ref_text_file:
        p.error("--voxcpm-mode ultimate needs --ref-text-file")
    if (
        not args.chatterbox_python
        and not args.qwen_python
        and not args.cosy_python
        and not args.voxcpm_python
    ):
        p.error(
            "give at least one of --chatterbox-python / --qwen-python / "
            "--cosy-python / --voxcpm-python"
        )
    if args.script is not None and not args.script.is_file():
        p.error(f"--script not found: {args.script}")
    if not args.ref.is_file():
        p.error(f"--ref not found: {args.ref}")
    if args.ref_text_file is not None and not args.ref_text_file.is_file():
        p.error(f"--ref-text-file not found: {args.ref_text_file}")
    for flag, value in (
        ("--chatterbox-python", args.chatterbox_python),
        ("--qwen-python", args.qwen_python),
        ("--cosy-python", args.cosy_python),
        ("--voxcpm-python", args.voxcpm_python),
    ):
        if value is not None and not value.is_file():
            p.error(f"{flag} not found: {value}")
    if args.out_dir.exists() and not args.out_dir.is_dir():
        p.error(f"--out-dir is a file, not a directory: {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The demo script is a string in the package, not a file — but the
    # backend CLIs take a --script path, so write it to a temp file and
    # pass that path to every side. The comparison records the label,
    # not the temp path (which is meaningless after the run). The
    # TemporaryDirectory spans the render loop, so an abort (p.error,
    # Ctrl+C) cleans it up instead of leaking it into %TEMP%.
    with tempfile.TemporaryDirectory() as tmp_name:
        if args.demo_script:
            try:
                from voice_generator.demo_script import DEMO_SCRIPT
            except ImportError:
                p.error(
                    "--demo-script needs the project installed in this Python "
                    "(pip install -e . in the core venv); or pass --script with a file"
                )
            script_path = Path(tmp_name) / "demo-script.txt"
            script_path.write_text(DEMO_SCRIPT, encoding="utf-8")
            script_label = "demo-script (emotional-range paragraph)"
        else:
            script_path = args.script
            script_label = str(args.script)
        results: list[dict[str, object]] = []
        if args.chatterbox_python:
            results.append(
                _run_backend(
                    label=f"chatterbox:{args.variant}",
                    python=args.chatterbox_python,
                    backend="chatterbox",
                    script=script_path,
                    ref=args.ref,
                    out_wav=args.out_dir / "chatterbox.wav",
                    extra=["--variant", args.variant, "--language", args.language],
                    timeout=args.timeout,
                )
            )
        if args.qwen_python:
            extra = ["--model-size", args.model_size, "--language", args.language]
            if args.ref_text_file:
                extra += ["--ref-text-file", str(args.ref_text_file)]
            else:
                extra += ["--x-vector-only"]
            results.append(
                _run_backend(
                    label=f"qwen3:{args.model_size}",
                    python=args.qwen_python,
                    backend="qwen3",
                    script=script_path,
                    ref=args.ref,
                    out_wav=args.out_dir / "qwen3.wav",
                    extra=extra,
                    timeout=args.timeout,
                )
            )

        if args.cosy_python:
            extra = ["--language", args.language]
            if args.ref_text_file:
                extra += ["--ref-text-file", str(args.ref_text_file)]
            if args.cosyvoice_home:
                extra += ["--cosyvoice-home", args.cosyvoice_home]
            if args.cosy_model_dir:
                extra += ["--cosy-model-dir", args.cosy_model_dir]
            results.append(
                _run_backend(
                    label="cosyvoice:3",
                    python=args.cosy_python,
                    backend="cosyvoice",
                    script=script_path,
                    ref=args.ref,
                    out_wav=args.out_dir / "cosyvoice.wav",
                    extra=extra,
                    timeout=args.timeout,
                )
            )

        if args.voxcpm_python:
            extra = ["--language", args.language, "--voxcpm-mode", args.voxcpm_mode]
            if args.voxcpm_mode == "ultimate":
                extra += ["--ref-text-file", str(args.ref_text_file)]
            results.append(
                _run_backend(
                    label=f"voxcpm:{args.voxcpm_mode}",
                    python=args.voxcpm_python,
                    backend="voxcpm",
                    script=script_path,
                    ref=args.ref,
                    out_wav=args.out_dir / "voxcpm.wav",
                    extra=extra,
                    timeout=args.timeout,
                )
            )

        comparison = {
            "script": script_label,
            "reference": str(args.ref),
            "note": "Listen side by side; durations/wall times are diagnostics, "
            "not a quality score.",
            "results": results,
        }
        out_json = args.out_dir / "ab_comparison.json"
        out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        for r in results:
            if r["ok"]:
                print(
                    f"{r['backend']}: {r['duration_s']}s audio in {r['wall_s']}s wall "
                    f"-> {r['wav']}"
                )
            else:
                print(
                    f"{r['backend']}: FAILED ({r['wall_s']}s) — see {out_json}",
                    file=sys.stderr,
                )
        print(f"Comparison: {out_json}")
        return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
