"""Tests for tools/ab_compare.py — the --demo-script path only.

The full bench needs GPU backends in separate venvs, so this pins the
demo-script wiring: the flag resolves to the built-in demo text, the
comparison JSON records the demo label (not the temp path it was
written to), and a backend that cannot render fails cleanly with its
error recorded. --script and --demo-script are mutually exclusive.
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from ab_compare import main  # noqa: E402

from voice_generator.audio import write_pcm16_wav  # noqa: E402


def _tone_wav(path: Path, seconds: float = 10.0, rate: int = 16000) -> None:
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
    write_pcm16_wav(path, samples, rate)


def test_demo_script_flag(tmp_path):
    ref = tmp_path / "me.wav"
    _tone_wav(ref)
    out_dir = tmp_path / "ab"
    code = main(
        [
            "--demo-script",
            "--ref",
            str(ref),
            "--chatterbox-python",
            sys.executable,
            "--out-dir",
            str(out_dir),
        ]
    )
    # Chatterbox is not installed in this venv, so the side fails — but
    # the demo-script wiring ran first: the comparison exists and names
    # the demo text, not a temp path.
    assert code == 1
    comparison = json.loads((out_dir / "ab_comparison.json").read_text(encoding="utf-8"))
    assert comparison["script"] == "demo-script (emotional-range paragraph)"
    assert len(comparison["results"]) == 1
    result = comparison["results"][0]
    assert result["backend"] == "chatterbox:turbo"
    assert result["ok"] is False
    assert result["error"]


def test_script_and_demo_script_are_exclusive(tmp_path):
    ref = tmp_path / "me.wav"
    _tone_wav(ref)
    script = tmp_path / "script.txt"
    script.write_text("Hello.", encoding="utf-8")
    base = ["--ref", str(ref), "--chatterbox-python", sys.executable, "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        main(base)  # neither --script nor --demo-script
    with pytest.raises(SystemExit):
        main(["--script", str(script), "--demo-script", *base])  # both
