"""Guards for the GitHub Pages site in `site/`.

The listen page renders audio only from `samples.json`, so these tests
pin the contract that page depends on: the script text matches the
script the bench actually renders, every listed sample file exists and
is an MP3, and every relative link in the site resolves. A missing
sample must be a pending slot in the JSON, never a fake file.
"""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


class _LinkParser(HTMLParser):
    """Collects href/src attributes from one HTML file."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)


def _relative_links(html_path: Path) -> list[str]:
    parser = _LinkParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return [
        link
        for link in parser.links
        if not link.startswith(("http://", "https://", "#", "mailto:"))
    ]


def test_samples_json_valid_and_script_matches():
    data = json.loads((SITE / "samples.json").read_text(encoding="utf-8"))
    assert data["script_source"] == "recording-script.txt"
    recorded = (ROOT / "recording-script.txt").read_text(encoding="utf-8").strip()
    assert data["script"].strip() == recorded
    assert data["expected"] == ["chatterbox", "qwen3", "cosyvoice", "voxcpm"]
    assert isinstance(data["samples"], list)


def test_listed_samples_exist_and_are_mp3():
    data = json.loads((SITE / "samples.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    for sample in data["samples"]:
        backend = sample["backend"]
        assert backend in data["expected"], f"unknown backend in samples.json: {backend}"
        assert backend not in seen, f"duplicate sample for {backend}"
        seen.add(backend)
        assert sample["file"].endswith(".mp3"), (
            f"{backend}: samples must be MP3, got {sample['file']}"
        )
        assert (SITE / sample["file"]).is_file(), (
            f"{backend}: listed file missing: {sample['file']}"
        )
        assert sample["duration_s"] > 0
        assert sample["wall_s"] > 0


def test_site_links_resolve():
    for page in ("index.html", "research.html", "listen.html"):
        for link in _relative_links(SITE / page):
            target = (SITE / link).resolve()
            assert target.is_file(), f"{page} links to missing file: {link}"


def test_listen_page_loads_samples_from_json_not_hardcoded_audio():
    # The listen page builds its players from samples.json at runtime.
    # A hardcoded <audio> element in the HTML would bypass that contract
    # (and could point at a file that does not exist). The page may still
    # *mention* .mp3 filenames in its "how samples get here" instructions.
    for page in ("index.html", "research.html", "listen.html"):
        assert "<audio" not in (SITE / page).read_text(encoding="utf-8"), page
    listen = (SITE / "listen.html").read_text(encoding="utf-8")
    assert "samples.json" in listen
    assert re.search(r'fetch\("samples\.json"\)', listen)
