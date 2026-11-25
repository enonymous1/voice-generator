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
from voice_generator.web.scripts import RANGE_TEXT  # noqa: E402

SCRIPT = "Hello, this is a test of my cloned voice."


def _tone_wav(path, seconds: float = 12.0, rate: int = 16000) -> None:
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
    write_pcm16_wav(path, samples, rate)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A throwaway project: refs, a real mock render, report, site/."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "refs").mkdir()
    (tmp_path / "output").mkdir()
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
    # Seeded: the project holds a real reference clip, which must never
    # be published — the frozen site carries only the seeded tone clip.
    out = tmp_path / "frozen"
    result = freeze_project(project, out, seed_mock=True)
    assert result["seeded"] is True
    assert result["seeded_files"] == ["refs/demo.wav"]
    assert result["excluded_refs"] == ["me.wav"]
    for name in ("index.html", "synth.html", "setup.html", "history.html", "research.html"):
        assert (out / name).is_file(), name
    assert (out / "style.css").is_file()
    # The frozen banner is on every rendered page; the live UI never shows it.
    # (Collapse whitespace first — the banner text wraps across lines.)
    synth = " ".join((out / "synth.html").read_text(encoding="utf-8").split())
    assert "Static preview" in synth
    assert "mock demo data" in synth
    # Absolute links are rewritten to relative file names.
    assert 'href="/synth"' not in synth
    assert 'href="synth.html"' in synth
    assert 'href="style.css"' in synth
    assert 'href="/setup"' not in synth
    # Forms are inert, not pointing at the live routes.
    setup = (out / "setup.html").read_text(encoding="utf-8")
    assert 'action="/setup/upload"' not in setup
    assert 'action="#"' in setup
    # Only the seeded tone clip is published — the real clip is excluded.
    assert 'src="refs/demo.wav"' in setup
    assert (out / "refs" / "demo.wav").is_file()
    assert "me.wav" not in setup
    assert not (out / "refs" / "me.wav").exists()
    # History links its render + manifest, and both are copied.
    history = (out / "history.html").read_text(encoding="utf-8")
    assert "demo.wav" in history
    assert 'id="render-modal"' in history  # the modal ships in the frozen page
    assert 'target="_blank"' in history  # no-JS fallback: new tab, not navigation
    assert (out / "outputs" / "demo.wav").is_file()
    assert (out / "outputs" / "demo.manifest.json").is_file()
    # The modal reads its URLs from data attributes, so those are
    # rewritten to relative paths too — absolute ones 404d on Pages
    # the same way the script fetch did.
    assert 'data-wav="outputs/demo.wav"' in history
    assert 'data-manifest="outputs/demo.manifest.json"' in history
    assert 'data-wav="/outputs/' not in history
    assert 'data-manifest="/outputs/' not in history
    # The Synthesize button fetches the range script from a static
    # file now, not the live /script route (which 404s on Pages and
    # used to paste GitHub's 404 page into the textarea). This assertion
    # is load-bearing: the rewrite depends on one exact string literal
    # in the template, so a template edit that breaks it must fail here
    # instead of silently shipping a broken button.
    synth_raw = (out / "synth.html").read_text(encoding="utf-8")
    assert 'fetch("script-range.txt")' in synth_raw
    assert 'fetch("/script?id=range")' not in synth_raw
    assert (out / "script-range.txt").is_file()
    assert (out / "script-range.txt").read_text(encoding="utf-8") == RANGE_TEXT
    # The research page renders the Markdown report from docs/research.
    research = (out / "research.html").read_text(encoding="utf-8")
    assert "<table" in research
    assert "VoxCPM2" in research
    # The landing page carries the site's copy.
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Four engines behind one CLI" in index
    assert "Start here" in index
    # Frozen: the listen button still goes to the compare page, and the
    # Start-here links are rewritten to the frozen pages.
    assert 'href="listen.html"' in index
    assert "Listen to the samples" in index
    assert 'href="setup.html"' in index
    assert 'href="synth.html"' in index
    assert 'href="index.html"' in index  # the brand now links home
    assert '>Home</a>' in index  # explicit Home link in the nav
    assert 'target="_blank" rel="noopener">Get the code</a>' in index
    # Frozen pages never flash, so no confirmation dialog is rendered.
    assert "flash-dialog" not in index
    # Frozen pages can't delete (the forms would be inert), so the
    # History page carries no delete form — the JS handler still
    # references the class name, so assert on the form's attribute.
    history = (out / "history.html").read_text(encoding="utf-8")
    assert 'class="history-delete-form"' not in history
    # The listen/compare page is copied verbatim.
    assert (out / "listen.html").is_file()
    assert (out / "samples.json").is_file()


def test_freeze_unseeded_refuses_real_refs(project, tmp_path):
    # An unseeded freeze with a real reference clip refuses instead of
    # publishing the user's voice — the old behavior copied it silently.
    out = tmp_path / "frozen"
    with pytest.raises(ValueError, match="seed-mock"):
        freeze_project(project, out)
    # And the CLI turns the refusal into a clean error, not a traceback.
    assert main(["freeze", "--project", str(project), "--out", str(out)]) == 1
    assert not (out / "index.html").exists()


def test_freeze_seed_mock_fills_gaps_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty"
    (empty / "refs").mkdir(parents=True)
    (empty / "output").mkdir()
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


def test_freeze_seed_mock_excludes_real_refs_but_keeps_renders(project, tmp_path):
    # The project has a real clip and a real (mock-backend) render: the
    # seeded freeze excludes the clip — only the tone demo clip is
    # published — but keeps the render, and seeds no mock renders
    # because a manifest already exists.
    out = tmp_path / "frozen"
    result = freeze_project(project, out, seed_mock=True)
    assert result["seeded"] is True
    assert result["seeded_files"] == ["refs/demo.wav"]
    assert result["excluded_refs"] == ["me.wav"]
    assert (out / "refs" / "demo.wav").is_file()
    assert not (out / "refs" / "me.wav").exists()
    assert not (out / "outputs" / "demo_hello.wav").exists()
    assert (out / "outputs" / "demo.wav").is_file()
    # The real project was never written to.
    assert (project / "refs" / "me.wav").is_file()
    assert not (project / "refs" / "demo.wav").exists()


def test_freeze_without_site_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bare = tmp_path / "bare"
    (bare / "refs").mkdir(parents=True)
    (bare / "output").mkdir()
    out = tmp_path / "frozen"
    result = freeze_project(bare, out)
    assert (out / "index.html").is_file()
    assert not (out / "listen.html").exists()
    assert "listen.html" not in result["files"]


def test_freeze_cli(project, tmp_path):
    out = tmp_path / "frozen"
    assert main(["freeze", "--project", str(project), "--out", str(out), "--seed-mock"]) == 0
    assert (out / "index.html").is_file()
    assert (out / "refs" / "demo.wav").is_file()
    assert not (out / "refs" / "me.wav").exists()
