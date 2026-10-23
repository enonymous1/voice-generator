"""A/B compare: render the SAME script + reference through two backends and
write a side-by-side comparison manifest.

Why a subprocess tool instead of a library call: the backends live in
separate venvs (their torch stacks conflict), so no single interpreter can
import both. This tool drives each venv's interpreter with
`python -m voice_generator synth ...` and then compares the results
(duration, sample rate, wall time) from the manifests each run produced.

Usage:
  python tools/ab_compare.py --script script.txt --ref refs/me.wav ^
      --ref-text-file refs/me.txt ^
      --chatterbox-python .venv-chatterbox/Scripts/python.exe ^
      --qwen-python .venv-qwen3/Scripts/python.exe ^
      --out-dir output/ab1

Either --chatterbox-python or --qwen-python may be omitted (that side is
skipped), which also makes this usable for a single-backend timing run.
Stdlib only — run it with any Python.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run_backend(
    *,
    label: str,
    python: Path,
    backend: str,
    script: Path,
    ref: Path,
    out_wav: Path,
    extra: list[str],
) -> dict[str, object]:
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
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall = round(time.time() - started, 2)
    if proc.returncode != 0:
        return {"backend": label, "ok": False, "wall_s": wall, "error": proc.stderr.strip()[-2000:]}
    manifest = json.loads(out_wav.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    return {
        "backend": label,
        "ok": True,
        "wall_s": wall,
        "wav": str(out_wav),
        "duration_s": manifest["duration_s"],
        "sample_rate": manifest["sample_rate"],
        "chunks": len(manifest["chunks"]),
        "license": manifest["license"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render one script through two backends and compare")
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--ref-text-file", type=Path, default=None, help="reference transcript (Qwen3 side)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chatterbox-python", type=Path, default=None, help="python.exe of the Chatterbox venv")
    p.add_argument("--qwen-python", type=Path, default=None, help="python.exe of the Qwen3 venv")
    p.add_argument("--cosy-python", type=Path, default=None, help="python of the CosyVoice Conda env")
    p.add_argument("--cosyvoice-home", default=None, help="CosyVoice repo clone (CosyVoice side)")
    p.add_argument("--cosy-model-dir", default=None, help="CosyVoice model dir (CosyVoice side)")
    p.add_argument("--variant", default="turbo")
    p.add_argument("--model-size", default="1.7b", choices=["1.7b", "0.6b"])
    args = p.parse_args(argv)
    if not args.chatterbox_python and not args.qwen_python and not args.cosy_python:
        p.error("give at least one of --chatterbox-python / --qwen-python / --cosy-python")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    if args.chatterbox_python:
        results.append(
            _run_backend(
                label=f"chatterbox:{args.variant}",
                python=args.chatterbox_python,
                backend="chatterbox",
                script=args.script,
                ref=args.ref,
                out_wav=args.out_dir / "chatterbox.wav",
                extra=["--variant", args.variant],
            )
        )
    if args.qwen_python:
        extra = ["--model-size", args.model_size]
        if args.ref_text_file:
            extra += ["--ref-text-file", str(args.ref_text_file)]
        else:
            extra += ["--x-vector-only"]
        results.append(
            _run_backend(
                label=f"qwen3:{args.model_size}",
                python=args.qwen_python,
                backend="qwen3",
                script=args.script,
                ref=args.ref,
                out_wav=args.out_dir / "qwen3.wav",
                extra=extra,
            )
        )

    if args.cosy_python:
        extra = []
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
                script=args.script,
                ref=args.ref,
                out_wav=args.out_dir / "cosyvoice.wav",
                extra=extra,
            )
        )

    comparison = {
        "script": str(args.script),
        "reference": str(args.ref),
        "note": "Listen side by side; durations/wall times are diagnostics, not a quality score.",
        "results": results,
    }
    out_json = args.out_dir / "ab_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    for r in results:
        if r["ok"]:
            print(f"{r['backend']}: {r['duration_s']}s audio in {r['wall_s']}s wall -> {r['wav']}")
        else:
            print(f"{r['backend']}: FAILED ({r['wall_s']}s) — see {out_json}", file=sys.stderr)
    print(f"Comparison: {out_json}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
