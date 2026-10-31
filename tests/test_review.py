"""Tests added by the 2026-10-31 staff review, pinning the fixes it made.

Each test names the defect it guards, so a future refactor that reopens
one of them fails loudly here instead of silently on a user's machine.
"""

import json
import struct
import wave
from pathlib import Path

import pytest

from voice_generator.audio import write_pcm16_wav
from voice_generator.backends.base import resolve_device
from voice_generator.backends.qwen_backend import Qwen3Backend
from voice_generator.cli import main
from voice_generator.pipeline import SynthConfig, synthesize_script
from voice_generator.text import chunk_script


def test_chunk_script_rejects_nonpositive_max_chars():
    # Old code: --max-chars 0 spun the hard-split loop forever.
    with pytest.raises(ValueError, match="max_chars"):
        chunk_script("Hello world.", max_chars=0)


def test_write_pcm16_clips_and_roundtrips(tmp_path: Path):
    out = tmp_path / "clip.wav"
    write_pcm16_wav(out, [2.0, -2.0, 0.5, 0.0], 24000)
    with wave.open(str(out), "rb") as w:
        values = struct.unpack("<4h", w.readframes(4))
    assert values == (32767, -32767, 16383, 0)


def test_manifest_records_backend_settings_and_version(tmp_path: Path):
    # Old code: the manifest named the backend but dropped its settings
    # (variant/model), so a render was not reproducible from the manifest.
    out = tmp_path / "ep.wav"
    result = synthesize_script(
        text="Hello world.",
        reference_wav=None,
        out_wav=out,
        backend_name="mock",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 1
    assert manifest["backend_settings"]["note"] == "tone only — not speech"


def test_synth_config_backend_kwargs():
    assert SynthConfig(text="x", reference_wav=None, out_wav=Path("o.wav")).backend_kwargs() == {
        "variant": "turbo"
    }
    qwen = SynthConfig(
        text="x", reference_wav=None, out_wav=Path("o.wav"), backend_name="qwen3", attn=None
    )
    assert qwen.backend_kwargs() == {"size": "1.7b"}
    cosy = SynthConfig(
        text="x",
        reference_wav=None,
        out_wav=Path("o.wav"),
        backend_name="cosyvoice",
        cosyvoice_home="/c/CosyVoice",
    )
    assert cosy.backend_kwargs() == {
        "version": "3",
        "mode": "zero_shot",
        "cosyvoice_home": "/c/CosyVoice",
    }


def test_qwen_rejects_bad_dtype_up_front():
    # Old code: KeyError inside _load, buried under a model-load traceback.
    with pytest.raises(ValueError, match="dtype"):
        Qwen3Backend(dtype="int8")


def test_resolve_device_auto_without_torch_is_cpu():
    # This venv has no torch installed; auto must not crash or claim CUDA.
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"


def test_cli_synth_mock_end_to_end(tmp_path: Path, capsys):
    code = main(
        [
            "synth",
            "--backend",
            "mock",
            "--text",
            "Hello world.",
            "--out",
            str(tmp_path / "hello.wav"),
        ]
    )
    assert code == 0
    assert (tmp_path / "hello.wav").exists()
    assert (tmp_path / "hello.manifest.json").exists()


def test_cli_missing_script_is_clean_error_not_traceback(tmp_path: Path, capsys):
    code = main(
        ["synth", "--backend", "mock", "--script", str(tmp_path / "nope.txt"), "--out", "x.wav"]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_check_ref_missing_is_clean_error(tmp_path: Path, capsys):
    assert main(["check-ref", str(tmp_path / "nope.wav")]) == 1


def test_cli_doctor_runs_without_gpu(tmp_path: Path, capsys):
    assert main(["doctor"]) == 0
    assert "python" in capsys.readouterr().out
