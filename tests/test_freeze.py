"""Tests for ``voice-gen freeze`` (static Pages snapshot of the web UI).

Uses Flask's test client via the freeze module — no server is started,
no GPU or model downloads are needed: seeded renders run through the
``mock`` backend, and the fixture's real render does too.
"""

import json
import math

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("markdown")

from voice_generator.audio import write_pcm16_wav  # noqa: E402
from voice_generator.cli import main  # noqa: E402
from voice_generator.pipeline import synthesize_script  # noqa: E402
from voice_generator.web.freeze import freeze_project  # noqa: E402

SCRIPT = "Hello, this is a test of my cloned voice."


def _tone_wav(path, seconds: float = 12.0, rate: int = 16000) -> None:
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
    write_pcm16_wav(path, samples, rate)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A throwaway project: refs, a real mock render, script, report, site/."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "refs").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "recording-script.txt").write_text(SCRIPT, encoding="utf-8")
    _tone_wav(tmp_path / "refs" / "me.wav")
    (tmp_path / "refs" / "me.txt").write_text(SCRIPT, encoding="utf-8")
    synthesize_script(
        text="Hello demo.",
        reference_wav=None,
        out_wav=tmp_path / "output" / "demo.wav",
        backend_name="mock",
    )
    (tmp_path / "docs" / "research").mkdir(parents=True)
    (tmp_path / "docs" / "research" / "report.md").write_text(
        "# Fast open voice-cloning models\n\n"
        "| Backend | License |\n|---|---|\n| VoxCPM2 | Apache-2.0 |\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "samples").mkdir(parents=True)
    (tmp_path / "site" / "listen.html").write_text(
        '<link rel="stylesheet" href="style.css">\n<div class="cards">listen</div>\n',
        encoding="utf-8",
    )
    (tmp_path / "site" / "samples.json").write_text(
        json.dumps({"expected": ["mock"], "samples": [], "script": SCRIPT}),
        encoding="utf-8",
    )
    return tmp_path


def test_freeze_writes_pages_and_rewrites_links(project, tmp_path):
    out = tmp_path / "frozen"
    result = freeze_project(project, out)
    assert result["seeded"] is False
    assert result["seeded_files"] == []
    for name in ("index.html", "synth.html", "setup.html", "history.html", "research.html"):
        assert (out / name).is_file(), name
    assert (out / "style.css").is_file()
    # The frozen banner is on every rendered page; the live UI never shows it.
    # (Collapse whitespace first — the banner text wraps across lines.)
    synth = " ".join((out / "synth.html").read_text(encoding="utf-8").split())
    assert "Static preview" in synth
    assert "mock demo data" not in synth
    # Absolute links are rewritten to relative file names.
    assert 'href="/synth"' not in synth
    assert 'href="synth.html"' in synth
    assert 'href="style.css"' in synth
    assert 'href="/setup"' not in synth
    # Forms are inert, not pointing at the live routes.
    setup = (out / "setup.html").read_text(encoding="utf-8")
    assert 'action="/setup/upload"' not in setup
    assert 'action="#"' in setup
    assert 'src="refs/me.wav"' in setup
    assert (out / "refs" / "me.wav").is_file()
    # History links its render + manifest, and both are copied.
    history = (out / "history.html").read_text(encoding="utf-8")
    assert "demo.wav" in history
    assert 'id="render-modal"' in history  # the modal ships in the frozen page
    assert 'target="_blank"' in history  # no-JS fallback: new tab, not navigation
    assert (out / "outputs" / "demo.wav").is_file()
    assert (out / "outputs" / "demo.manifest.json").is_file()
    # The research page renders the Markdown report from docs/research.
    research = (out / "research.html").read_text(encoding="utf-8")
    assert "<table" in research
    assert "VoxCPM2" in research
    # The landing page carries the site's copy.
    assert "Four engines behind one CLI" in (out / "index.html").read_text(encoding="utf-8")
    # The listen/compare page is copied verbatim.
    assert (out / "listen.html").is_file()
    assert (out / "samples.json").is_file()


def test_freeze_seed_mock_fills_gaps_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty"
    (empty / "refs").mkdir(parents=True)
    (empty / "output").mkdir()
    (empty / "recording-script.txt").write_text(SCRIPT, encoding="utf-8")
    out = tmp_path / "frozen"
    result = freeze_project(empty, out, seed_mock=True)
    assert result["seeded"] is True
    assert result["seeded_files"] == [
        "refs/demo.wav",
        "output/demo_hello.wav",
        "output/demo_range.wav",
    ]
    assert (out / "refs" / "demo.wav").is_file()
    assert (out / "outputs" / "demo_hello.wav").is_file()
    assert (out / "outputs" / "demo_range.wav").is_file()
    setup = " ".join((out / "setup.html").read_text(encoding="utf-8").split())
    assert "mock demo data" in setup
    # The real project was never written to.
    assert list((empty / "refs").glob("*.wav")) == []
    assert list((empty / "output").glob("*")) == []


def test_freeze_seed_mock_skips_when_data_exists(project, tmp_path):
    out = tmp_path / "frozen"
    result = freeze_project(project, out, seed_mock=True)
    assert result["seeded"] is False
    assert result["seeded_files"] == []
    assert not (out / "refs" / "demo.wav").exists()
    assert not (out / "outputs" / "demo_hello.wav").exists()


def test_freeze_without_site_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bare = tmp_path / "bare"
    (bare / "refs").mkdir(parents=True)
    (bare / "output").mkdir()
    (bare / "recording-script.txt").write_text(SCRIPT, encoding="utf-8")
    out = tmp_path / "frozen"
    result = freeze_project(bare, out)
    assert (out / "index.html").is_file()
    assert not (out / "listen.html").exists()
    assert "listen.html" not in result["files"]


def test_freeze_cli(project, tmp_path):
    out = tmp_path / "frozen"
    assert main(["freeze", "--project", str(project), "--out", str(out)]) == 0
    assert (out / "index.html").is_file()
