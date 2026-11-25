"""voice-gen CLI.

Examples (Windows, in the Chatterbox venv):
  voice-gen doctor
  voice-gen check-ref refs\\me_10s.wav
  voice-gen synth --script script.txt --ref refs\\me_10s.wav --out output\\ep1.wav
  voice-gen synth --text "Hello, this is my digital twin." \
      --ref refs\\me_10s.wav --out output\\hello.wav --backend mock
  voice-gen serve  (local web UI on http://127.0.0.1:5000/)
  voice-gen freeze --seed-mock  (static snapshot of the UI into site/)

Error contract: expected failures (missing files, bad input, a backend
not installed in this venv) print a one-line ``Error: ...`` to stderr and
exit 1, instead of dumping a traceback — the common cases are user
errors with a known fix, and the fix is printed with the error where
one exists. Unexpected exceptions still propagate with a traceback.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import wave
from pathlib import Path

from voice_generator.audio import check_reference, wav_info
from voice_generator.backends import BACKEND_IMPORTS, available_backends
from voice_generator.backends.base import BackendNotInstalledError
from voice_generator.pipeline import synthesize_script

# Errors the CLI converts to a clean message instead of a traceback.
# wave.Error covers malformed WAVs; the rest are the documented failure
# modes of the pipeline and backends. Bare RuntimeError is deliberately
# NOT here: backends raise it for genuine engine failures (CUDA OOM,
# shape mismatches, real bugs), and flattening those into a one-line
# "user error" would destroy exactly the information needed to diagnose
# them. Only the dedicated BackendNotInstalledError gets the clean message.
EXPECTED_ERRORS = (FileNotFoundError, ValueError, BackendNotInstalledError, wave.Error)


def _probe(module: str) -> bool:
    """True if ``module`` can be found without importing it.

    A probe that errors (missing parent package, bad ``__spec__``) counts as
    not found: ``doctor`` exists to diagnose a broken environment, so it
    must never crash on one.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _unit_float(value: str) -> float:
    """argparse type: a float in [0, 1], the range the engines expect.

    ``--exaggeration`` / ``--cfg`` used to accept any float and pass it
    straight to the backend, where an out-of-range value failed as a
    confusing engine error; argparse rejects it here with usage instead.
    """
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return number


def _doctor() -> int:
    """Print environment diagnostics: Python, torch/CUDA/VRAM, backends, ffmpeg."""
    print(f"python   {platform.python_version()}  ({sys.executable})")
    try:
        import torch

        print(f"torch    {torch.__version__}")
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            # A half-broken torch can raise here too; that is a "not
            # available" verdict, not a reason for doctor to traceback.
            cuda_ok = False
        if cuda_ok:
            print(f"cuda     {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"vram     {total:.1f} GB")
        else:
            print("cuda     not available — real backends need an Nvidia GPU (mock still works)")
    except (ImportError, OSError):
        # A torch install with missing CUDA DLLs raises OSError
        # ([WinError 126]) on Windows, not ImportError — and that is
        # precisely the broken environment doctor exists to diagnose.
        print("torch    not installed in this venv (expected unless this is a backend venv)")
    # One probe per backend instead of three hand-written try/except
    # blocks that drifted apart (the old code probed chatterbox/qwen-tts
    # by import and cosyvoice by import too, with different messages).
    hints = {
        "chatterbox": 'not installed — pip install -e ".[chatterbox]" in its venv',
        "qwen3": 'not installed — pip install -e ".[qwen3]" in its venv',
        "cosyvoice": "not importable here — it lives in its own Conda env + repo clone",
        "voxcpm": 'not installed — pip install -e ".[voxcpm]" in its venv',
        "mock": "built in — tone only, not speech (always available)",
    }
    for name in available_backends():
        module = BACKEND_IMPORTS[name]
        status = "installed" if _probe(module) else hints[name]
        print(f"{name:<11} {status}")
    home = os.environ.get("COSYVOICE_HOME")
    print(f"cosy-home  {home or 'unset (set COSYVOICE_HOME or pass --cosyvoice-home)'}")
    print(f"ffmpeg   {shutil.which('ffmpeg') or 'not found on PATH (needed later for Stage 2)'}")
    return 0


def _synth(args: argparse.Namespace) -> int:
    """Render one script/text to a WAV + manifest via ``synthesize_script``.

    ``--text`` and ``--script`` are mutually exclusive; an empty ``--text``
    is a real (empty) script, not a missing one, so it is tested with
    ``is not None`` — the old truthiness test fell through to
    ``args.script.read_text`` on ``None`` and crashed with AttributeError.
    """
    text = args.text if args.text is not None else args.script.read_text(encoding="utf-8")
    ref_text = args.ref_text
    if args.ref_text_file is not None:
        ref_text = args.ref_text_file.read_text(encoding="utf-8").strip()
    result = synthesize_script(
        text=text,
        reference_wav=args.ref,
        out_wav=args.out,
        backend_name=args.backend,
        variant=args.variant,
        language=args.language,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        max_chars=args.max_chars,
        ref_text=ref_text,
        x_vector_only=args.x_vector_only,
        model_size=args.model_size,
        attn=args.attn,
        cosyvoice_home=args.cosyvoice_home,
        cosy_model_dir=args.cosy_model_dir,
        cosy_mode=args.cosy_mode,
        cosy_version=args.cosy_version,
        voxcpm_model=args.voxcpm_model,
        voxcpm_mode=args.voxcpm_mode,
        voxcpm_cfg=args.voxcpm_cfg,
        voxcpm_timesteps=args.voxcpm_timesteps,
        voxcpm_optimize=args.voxcpm_optimize,
    )
    print(f"\nDone: {result.wav_path}  ({result.duration_s:.1f}s, {result.chunks} chunk(s))")
    print(f"Manifest: {result.manifest_path}")
    return 0


def _is_missing_web_dependency(exc: ImportError) -> bool:
    """True if ``exc`` is the web UI itself (or Flask/Markdown) missing.

    A bare ``except ImportError`` around the web import misattributes deep
    failures: a missing transitive dependency *inside* the web package
    would be reported as "install the web extra" for something already
    installed. Only the web package itself (or its two declared
    dependencies) gets the install hint; anything else propagates with
    its traceback, because it is a real bug.
    """
    name = exc.name
    if name is None:
        return False
    return (
        name == "voice_generator.web"
        or name.startswith("voice_generator.web.")
        or name in ("flask", "markdown")
    )


def _freeze(args: argparse.Namespace) -> int:
    """Render the web UI to a static site (default: the project's site/)."""
    try:
        from voice_generator.web.freeze import freeze_project
    except ImportError as exc:
        if not _is_missing_web_dependency(exc):
            raise
        print(
            'Error: freeze needs the web UI — pip install -e ".[web]" in this venv',
            file=sys.stderr,
        )
        return 1
    result = freeze_project(args.project, args.out, seed_mock=args.seed_mock)
    print(f"froze {len(result['files'])} files to {result['out']}")
    if result["seeded"]:
        print(f"  seeded mock demo data: {', '.join(result['seeded_files'])}")
    print("  commit + push — the Pages workflow publishes site/ as-is")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Start the local web UI (Flask, localhost only by default)."""
    try:
        from voice_generator.web.app import WebConfig, _find_project_root, create_app
    except ImportError as exc:
        if not _is_missing_web_dependency(exc):
            raise
        print(
            'Error: the web UI needs Flask — pip install -e ".[web]" in this venv',
            file=sys.stderr,
        )
        return 1
    backend_pythons = {
        name: python
        for name, python in (
            ("chatterbox", args.chatterbox_python),
            ("qwen3", args.qwen_python),
            ("cosyvoice", args.cosy_python),
            ("voxcpm", args.voxcpm_python),
        )
        if python
    }
    # Resolve the default dirs against the project root before creating
    # them: `voice-gen serve` run from anywhere but the project root used
    # to silently create an empty refs/ + output/ in the wrong directory.
    # Explicitly-passed dirs are the caller's choice and stay CWD-relative;
    # only the defaults are anchored (or refused when no project root is
    # found above the CWD at all).
    root = _find_project_root(Path.cwd())
    if root is not None:
        if args.refs_dir == Path("refs"):
            args.refs_dir = root / args.refs_dir
        if args.output_dir == Path("output"):
            args.output_dir = root / args.output_dir
    elif args.refs_dir == Path("refs") or args.output_dir == Path("output"):
        print(
            "Error: no voice-generator project found above "
            f"{Path.cwd()} — run `voice-gen serve` from the project root "
            "(or pass --refs-dir and --output-dir)",
            file=sys.stderr,
        )
        return 1
    config = WebConfig(
        refs_dir=args.refs_dir,
        output_dir=args.output_dir,
        backend_pythons=backend_pythons,
        host=args.host,
        port=args.port,
    )
    config.refs_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"Warning: listening on {args.host} — the UI is unauthenticated "
            "and can read your reference clips; prefer 127.0.0.1",
            file=sys.stderr,
        )
    app = create_app(config)
    print(f"voice-generator web UI at http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    print(f"  refs: {config.refs_dir}   output: {config.output_dir}")
    try:
        app.run(host=args.host, port=args.port, use_reloader=False, threaded=True)
    except OSError as exc:
        print(f"Error: could not start the server: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="voice-gen", description="Local FOSS voice cloning")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check python / torch / CUDA / VRAM / ffmpeg")
    sub.add_parser("backends", help="list backends available in this build")

    p_ref = sub.add_parser("check-ref", help="validate a reference voice clip")
    p_ref.add_argument("ref", type=Path)

    p = sub.add_parser("synth", help="synthesize a script (or one line of text) to a WAV")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="a single line / short passage")
    src.add_argument(
        "--script", type=Path, help="path to a .txt script file (chunked automatically)"
    )
    p.add_argument(
        "--ref", type=Path, default=None, help="reference voice clip (.wav, ~10 s for Chatterbox)"
    )
    p.add_argument("--out", type=Path, required=True, help="output .wav path")
    p.add_argument("--backend", default="chatterbox", choices=available_backends())
    p.add_argument("--variant", default="turbo", choices=["original", "turbo", "multilingual"])
    p.add_argument("--language", default="en")
    p.add_argument("--exaggeration", type=_unit_float, default=0.5)
    p.add_argument("--cfg", type=_unit_float, default=0.5, dest="cfg_weight")
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument(
        "--ref-text",
        default=None,
        help="exact transcript of the reference clip (Qwen3, CosyVoice, VoxCPM ultimate mode)",
    )
    p.add_argument(
        "--ref-text-file", type=Path, default=None, help="file containing that transcript"
    )
    p.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Qwen3: clone from speaker embedding only, no transcript needed (lower quality)",
    )
    p.add_argument("--model-size", default="1.7b", choices=["1.7b", "0.6b"], help="Qwen3 Base size")
    p.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Qwen3 attention implementation (sdpa is the Windows-friendly default)",
    )
    p.add_argument(
        "--cosyvoice-home", default=None, help="CosyVoice: path to the cloned CosyVoice repo"
    )
    p.add_argument("--cosy-model-dir", default=None, help="CosyVoice: model snapshot directory")
    p.add_argument(
        "--cosy-mode",
        default="zero_shot",
        choices=["zero_shot", "cross_lingual"],
        help="CosyVoice clone mode (cross_lingual needs no reference transcript)",
    )
    p.add_argument("--cosy-version", default="3", choices=["3", "2"], help="CosyVoice version")
    p.add_argument(
        "--voxcpm-model",
        default="openbmb/VoxCPM2",
        help="VoxCPM2 model id on Hugging Face",
    )
    p.add_argument(
        "--voxcpm-mode",
        default="controllable",
        choices=["controllable", "ultimate"],
        help="VoxCPM2 clone mode (ultimate needs the reference transcript)",
    )
    p.add_argument(
        "--voxcpm-cfg",
        type=float,
        default=2.0,
        help="VoxCPM2 guidance scale (upstream default 2.0, typical 1.0–3.0)",
    )
    p.add_argument(
        "--voxcpm-timesteps",
        type=int,
        default=10,
        help="VoxCPM2 diffusion steps (upstream default 10, recommended 4–30)",
    )
    p.add_argument(
        "--voxcpm-no-optimize",
        action="store_false",
        dest="voxcpm_optimize",
        help="VoxCPM2: skip torch.compile (use if the first render fails on Triton/compile errors)",
    )

    p_serve = sub.add_parser("serve", help="start the local web UI (localhost only by default)")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--refs-dir", type=Path, default=Path("refs"))
    p_serve.add_argument("--output-dir", type=Path, default=Path("output"))
    p_serve.add_argument(
        "--chatterbox-python", default=None, help="python.exe of the Chatterbox venv"
    )
    p_serve.add_argument("--qwen-python", default=None, help="python.exe of the Qwen3 venv")
    p_serve.add_argument(
        "--cosy-python", default=None, help="python of the CosyVoice Conda env"
    )
    p_serve.add_argument("--voxcpm-python", default=None, help="python.exe of the VoxCPM2 venv")

    p_freeze = sub.add_parser(
        "freeze", help="render the web UI to a static site for GitHub Pages"
    )
    p_freeze.add_argument(
        "--project", type=Path, default=Path("."), help="project root (default: .)"
    )
    p_freeze.add_argument(
        "--out", type=Path, default=Path("site"), help="output directory (default: site)"
    )
    p_freeze.add_argument(
        "--seed-mock",
        action="store_true",
        help="fill gaps with mock demo data (a tone reference, two mock renders) "
        "in a scratch copy — the real refs/ and output/ are never touched",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "backends":
            print("\n".join(available_backends()))
            return 0
        if args.command == "check-ref":
            info = wav_info(args.ref)
            print(f"{args.ref}: {info.duration_s:.1f}s, {info.sample_rate} Hz, {info.channels}ch")
            # The header was just read for the line above; pass it along
            # instead of opening the file a second time.
            warnings = check_reference(args.ref, info=info)
            for w in warnings:
                print(f"  ! {w}")
            if not warnings:
                print("  OK — good reference clip")
            return 0
        if args.command == "synth":
            return _synth(args)
        if args.command == "serve":
            return _serve(args)
        if args.command == "freeze":
            return _freeze(args)
    except EXPECTED_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
