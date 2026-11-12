"""Freeze the local web UI to a static site for GitHub Pages.

``voice-gen freeze`` renders the Flask app's pages to plain HTML files in
an output directory (default ``site/``, which the Pages workflow already
publishes), so the public site is a snapshot of the living tool instead
of hand-maintained HTML that drifts from it.

What it does:
- Fetches /synth, /setup, /history, /research through the Flask test
  client and writes them as synth.html, setup.html, history.html,
  research.html; renders the landing template as index.html.
- Rewrites absolute links to relative file names (``/synth`` →
  ``synth.html``, ``/static/style.css`` → ``style.css``) and copies the
  media they point at (the Flask stylesheet, reference WAVs, render
  WAVs + manifests) into the output directory. Forms are made inert
  (their actions become ``#``).
- Copies the listen/compare page (listen.html + samples.json + samples/)
  and site/README.md verbatim when a site/ directory exists.
- Marks every frozen page with a static-preview banner (via the app's
  FROZEN config flag, read by base.html).

Seeding: with ``--seed-mock``, the freeze works on a scratch copy of the
project and fills gaps only — a tone reference when refs/ is empty, two
mock renders when output/ is empty — so the real refs/ and output/ are
never polluted. Seeded snapshots say so in their banner.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from pathlib import Path

from flask import render_template

from voice_generator.audio import write_pcm16_wav
from voice_generator.pipeline import synthesize_script
from voice_generator.web.app import WebConfig, create_app
from voice_generator.web.scripts import RANGE_TEXT

# Routes the freeze renders, and the file each becomes.
ROUTES = {
    "/synth": "synth.html",
    "/setup": "setup.html",
    "/history": "history.html",
    "/research": "research.html",
}

# The demo reference's transcript: the emotional-range paragraph's
# opening sentence — about 10 s spoken, matching the 10 s tone.
DEMO_REF_TEXT = RANGE_TEXT.split("!")[0] + "!"

_SEED_RENDERS = (
    ("demo_hello", "Hello, this is a demo render from the mock backend."),
    ("demo_range", RANGE_TEXT),
)

_ATTR_RE = re.compile(r'(href|src|action)="(/[^"]*)"')


def _rewrite_links(html: str, config: WebConfig) -> tuple[str, list[tuple[Path, str]]]:
    """Rewrite absolute links in fetched HTML to relative file names.

    Returns the rewritten HTML and the media files it points at, as
    (source path, relative destination) pairs for the caller to copy.
    """
    media: list[tuple[Path, str]] = []

    def replace(match: re.Match[str]) -> str:
        attr, path = match.group(1), match.group(2)
        if attr == "action":
            return f'{attr}="#"'  # frozen forms are inert
        if path in ROUTES:
            return f'{attr}="{ROUTES[path]}"'
        if path == "/":
            return f'{attr}="index.html"'
        if path.startswith("/static/"):
            return f'{attr}="{path[len("/static/"):]}"'
        if path.startswith("/refs/"):
            name = path[len("/refs/"):]
            media.append((config.refs_dir / name, f"refs/{name}"))
            return f'{attr}="refs/{name}"'
        if path.startswith("/outputs/"):
            name = path[len("/outputs/"):]
            media.append((config.output_dir / name, f"outputs/{name}"))
            return f'{attr}="outputs/{name}"'
        return match.group(0)  # anything else (e.g. /script in JS): leave alone

    return _ATTR_RE.sub(replace, html), media


def freeze_app(config: WebConfig, out_dir: str | Path, *, seeded: bool = False) -> dict:
    """Render the Flask app at ``config`` to static files in ``out_dir``.

    Returns ``{"out", "files", "seeded"}``. The app is built fresh here
    (never the running server), with the FROZEN banner flags set.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    app = create_app(config)
    app.config["FROZEN"] = True
    if seeded:
        app.config["FROZEN_SEEDED"] = True
    client = app.test_client()

    files: list[str] = []
    media: list[tuple[Path, str]] = []
    for route, filename in ROUTES.items():
        resp = client.get(route)
        if resp.status_code != 200:
            raise ValueError(f"freeze: {route} returned {resp.status_code}")
        html, found = _rewrite_links(resp.get_data(as_text=True), config)
        media.extend(found)
        (out / filename).write_text(html, encoding="utf-8")
        files.append(filename)

    # The landing page has no route: render its template directly.
    with app.test_request_context():
        html = render_template("landing.html")
    html, found = _rewrite_links(html, config)
    media.extend(found)
    (out / "index.html").write_text(html, encoding="utf-8")
    files.append("index.html")

    # One stylesheet for every frozen page, from the Flask static dir.
    shutil.copyfile(Path(app.static_folder) / "style.css", out / "style.css")
    files.append("style.css")

    for src, rel in media:
        if not src.is_file():
            continue  # a referenced file that is gone: leave the link broken, don't crash
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files.append(rel)

    return {"out": out, "files": sorted(set(files)), "seeded": seeded}


def seed_mock_data(work_dir: str | Path) -> list[str]:
    """Fill gaps in a scratch project copy with mock demo data.

    A tone reference when refs/ is empty, two mock renders when output/
    is empty. Returns the seeded paths (relative). Never touches the
    real project — callers pass a scratch copy.
    """
    work = Path(work_dir)
    seeded: list[str] = []
    refs_dir = work / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    if not list(refs_dir.glob("*.wav")):
        tone = [0.3 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(16000 * 10)]
        write_pcm16_wav(refs_dir / "demo.wav", tone, 16000)
        (refs_dir / "demo.txt").write_text(DEMO_REF_TEXT, encoding="utf-8")
        seeded.append("refs/demo.wav")
    output_dir = work / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not list(output_dir.glob("*.manifest.json")):
        for name, text in _SEED_RENDERS:
            synthesize_script(
                text=text,
                reference_wav=None,
                out_wav=output_dir / f"{name}.wav",
                backend_name="mock",
            )
            seeded.append(f"output/{name}.wav")
    return seeded


def _copy_site_files(project: Path, out: Path) -> list[str]:
    """Copy the listen/compare page and site README verbatim, if present."""
    site = project / "site"
    copied: list[str] = []
    for name in ("listen.html", "samples.json", "README.md"):
        src = site / name
        dst = out / name
        if src.is_file() and src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
            copied.append(name)
    samples = site / "samples"
    if samples.is_dir():
        dst = out / "samples"
        dst.mkdir(parents=True, exist_ok=True)
        for mp3 in sorted(samples.glob("*.mp3")):
            shutil.copyfile(mp3, dst / mp3.name)
            copied.append(f"samples/{mp3.name}")
    return copied


def freeze_project(
    project_dir: str | Path, out_dir: str | Path, *, seed_mock: bool = False
) -> dict:
    """Freeze the project at ``project_dir`` to a static site at ``out_dir``.

    With ``seed_mock``, the freeze works on a scratch copy and fills gaps
    with mock demo data; without it, the real project is read as-is and
    never written to. Returns ``{"out", "files", "seeded", "seeded_files"}``.
    """
    project = Path(project_dir)
    out = Path(out_dir)
    if seed_mock:
        tmp = Path(tempfile.mkdtemp(prefix="voice-gen-freeze-"))
        try:
            work = tmp / "project"
            work.mkdir()
            for name in ("refs", "output", "docs"):
                src = project / name
                if src.exists():
                    shutil.copytree(src, work / name)
            script = project / "recording-script.txt"
            if script.is_file():
                shutil.copyfile(script, work / "recording-script.txt")
            seeded_files = seed_mock_data(work)
            config = WebConfig(
                refs_dir=work / "refs",
                output_dir=work / "output",
                script_path=work / "recording-script.txt",
                research_path=work / "docs" / "research",
            )
            result = freeze_app(config, out, seeded=bool(seeded_files))
            result["seeded_files"] = seeded_files
            result["files"] = sorted(
                set(result["files"]) | set(_copy_site_files(project, out))
            )
            return result
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    config = WebConfig(
        refs_dir=project / "refs",
        output_dir=project / "output",
        script_path=project / "recording-script.txt",
        research_path=project / "docs" / "research",
    )
    result = freeze_app(config, out, seeded=False)
    result["seeded_files"] = []
    result["files"] = sorted(set(result["files"]) | set(_copy_site_files(project, out)))
    return result
