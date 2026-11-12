"""Tests for the local Flask web UI (``voice-gen serve``).

Uses Flask's test client — no server is started, no GPU or model
downloads are needed: renders run through the ``mock`` backend in a
subprocess (the same ``python -m voice_generator synth`` the UI drives
for real backends). The webm-conversion test needs ffmpeg on PATH and
skips without it.
"""

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from voice_generator.audio import wav_info, write_pcm16_wav  # noqa: E402
from voice_generator.pipeline import synthesize_script  # noqa: E402
from voice_generator.web.app import WebConfig, create_app  # noqa: E402

SCRIPT = "Hello, this is a test of my cloned voice."


def _tone_wav(path, seconds: float = 12.0, rate: int = 16000) -> None:
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
    write_pcm16_wav(path, samples, rate)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A throwaway project folder: refs/, output/, recording script."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "refs").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "recording-script.txt").write_text(SCRIPT, encoding="utf-8")
    _tone_wav(tmp_path / "refs" / "me.wav")
    return tmp_path


@pytest.fixture()
def client(project):
    config = WebConfig(
        refs_dir=project / "refs",
        output_dir=project / "output",
        script_path=project / "recording-script.txt",
    )
    return create_app(config).test_client()


def test_index_renders_landing(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Four engines behind one CLI" in html
    assert "Start here" in html
    assert 'href="/synth"' in html
    assert 'href="/setup"' in html
    assert 'href="/history"' in html
    assert 'href="/research"' in html
    # Live: the listen button goes to past renders, not the Pages page.
    assert "Listen to past renders" in html
    assert "listen.html" not in html
    assert "Static preview" not in html  # the frozen banner is freeze-only
    # The nav has an explicit Home link, and the external GitHub links
    # open in a new tab instead of navigating away.
    assert '>Home</a>' in html
    assert 'href="/"' in html
    assert 'target="_blank" rel="noopener">Get the code</a>' in html


def test_synth_page_lists_backends_and_refs(client):
    resp = client.get("/synth")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    for backend in ("mock", "chatterbox", "qwen3", "cosyvoice", "voxcpm"):
        assert backend in html
    assert "me — 12.0s, 16000 Hz" in html  # the fixture clip, no transcript yet
    assert "Static preview" not in html  # the frozen banner is freeze-only


def test_synth_page_offers_range_script_fill(client):
    resp = client.get("/synth")
    html = resp.get_data(as_text=True)
    # The emotional-range paragraph is the only script fill now —
    # "Use the recording script" is gone.
    assert 'id="load-range-script"' in html
    assert "Use the emotional-range script" in html
    assert "/script?id=range" in html
    assert 'id="load-script"' not in html
    assert "Use the recording script" not in html


def test_synth_page_explains_ref_and_mock(client):
    resp = client.get("/synth")
    # The template wraps the hints across lines, so collapse whitespace
    # before matching the sentences.
    text = " ".join(resp.get_data(as_text=True).split())
    # The reference-clip field explains its purpose and links to Setup.
    assert "The voice the backends clone from" in text
    assert 'href="/setup"' in text
    assert "Only the mock backend renders without one" in text
    # The mock note (a 220 Hz test tone, not speech) is on the page,
    # shown only when mock is the chosen backend.
    assert "Mock renders a 220 Hz test tone, not speech" in text
    assert "about 1 second per 14 characters" in text
    assert 'id="mock-hint"' in text


def test_synth_page_explains_max_chars(client):
    resp = client.get("/synth")
    # The template wraps the hint across lines, so collapse whitespace
    # before matching the sentences.
    text = " ".join(resp.get_data(as_text=True).split())
    # The chunk-size field carries its own explanation, under its label.
    assert "field-hint" in text
    assert "split into chunks of at most this many characters" in text
    assert "Smaller chunks use less memory" in text
    assert "300 is the default" in text


def test_setup_trim_saves_copy(client, project):
    resp = client.post(
        "/setup/trim",
        data={
            "source": "me",
            "name": "me_10s",
            "start": "0",
            "end": "10",
            "transcript": "Trimmed words here.",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "Saved refs/me_10s.wav" in resp.get_data(as_text=True)
    # The copy is 10s with the edited transcript; the original is untouched.
    assert (project / "refs" / "me_10s.wav").is_file()
    assert (project / "refs" / "me_10s.txt").read_text(encoding="utf-8") == "Trimmed words here."
    assert wav_info(project / "refs" / "me_10s.wav").duration_s == pytest.approx(10.0, abs=0.01)
    assert wav_info(project / "refs" / "me.wav").duration_s == pytest.approx(12.0, abs=0.01)


def test_setup_trim_refuses_duplicate_and_bad_range(client):
    client.post(
        "/setup/trim",
        data={"source": "me", "name": "me_10s", "start": "0", "end": "10", "transcript": "x"},
    )
    resp = client.post(
        "/setup/trim",
        data={"source": "me", "name": "me_10s", "start": "0", "end": "10", "transcript": "x"},
        follow_redirects=True,
    )
    assert "already exists" in resp.get_data(as_text=True)
    resp = client.post(
        "/setup/trim",
        data={"source": "me", "name": "me_2s", "start": "5", "end": "2", "transcript": "x"},
        follow_redirects=True,
    )
    assert "must be after start" in resp.get_data(as_text=True)


def test_setup_trim_empty_transcript_writes_no_txt(client, project):
    client.post(
        "/setup/trim",
        data={"source": "me", "name": "me_short", "start": "0", "end": "5", "transcript": "   "},
        follow_redirects=True,
    )
    assert (project / "refs" / "me_short.wav").is_file()
    assert not (project / "refs" / "me_short.txt").exists()


def test_setup_page_shows_trim_form(client):
    resp = client.get("/setup")
    html = resp.get_data(as_text=True)
    # Each readable reference offers trim-to-a-copy with an editable transcript.
    assert 'action="/setup/trim"' in html
    assert "Save trimmed copy" in html
    assert 'name="transcript"' in html
    assert "Trim a copy" in html


def test_ref_file_serves_wav_and_txt(client, project):
    (project / "refs" / "me.txt").write_text("The test transcript.", encoding="utf-8")
    resp = client.get("/refs/me.wav")
    assert resp.status_code == 200
    assert resp.data[:4] == b"RIFF"
    resp = client.get("/refs/me.txt")
    assert resp.status_code == 200
    assert "The test transcript." in resp.get_data(as_text=True)
    # Traversal, wrong suffix, and missing files are all 404s.
    assert client.get("/refs/../recording-script.txt").status_code == 404
    assert client.get("/refs/me.mp3").status_code == 404
    assert client.get("/refs/nope.wav").status_code == 404


def test_setup_page_ref_rows_expand(client, project):
    (project / "refs" / "me.txt").write_text("The test transcript.", encoding="utf-8")
    resp = client.get("/setup")
    html = resp.get_data(as_text=True)
    # The name is a toggle; its detail row holds a player for the clip
    # and the transcript, so the words can be checked against the audio.
    assert 'class="ref-toggle"' in html
    assert 'class="ref-detail"' in html
    assert 'src="/refs/me.wav"' in html
    assert "The test transcript." in html


def test_setup_page_add_reference_card(client):
    resp = client.get("/setup")
    html = resp.get_data(as_text=True)
    # Record and upload are one card now, side by side inside it
    # (record 1 part to upload's 3), under a single shared name field:
    # the upload form takes it via the form attribute, the record flow
    # reads the same input. The two old name fields are gone.
    assert 'class="addref-grid"' in html
    assert 'id="ref-name"' in html
    assert 'form="upload-form"' in html
    assert 'id="upload-form"' in html
    assert 'id="rec-name"' not in html
    assert 'id="name"' not in html
    assert 'getElementById("ref-name")' in html
    css = (
        Path(__file__).parent.parent
        / "src"
        / "voice_generator"
        / "web"
        / "static"
        / "style.css"
    ).read_text(encoding="utf-8")
    assert ".addref-grid" in css
    assert "grid-template-columns: 1fr 3fr" in css
    assert "@media (max-width: 700px)" in css
    # Both columns are the same height, and the record column's buttons
    # sit centred in its full height instead of floating at the top.
    assert "align-items: stretch" in css
    assert "#rec-ui" in css
    assert "justify-content: center" in css


def test_setup_page_shows_default_script(client):
    resp = client.get("/setup")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Emotional range is the default script; the project introduction
    # is no longer a setup choice.
    assert "6:45 AM" in html
    assert "Project introduction" not in html
    assert "Hi, I'm Justin" not in html


def test_setup_page_lists_script_options(client):
    resp = client.get("/setup")
    html = resp.get_data(as_text=True)
    assert "Emotional range" in html
    assert "Harvard sentences" in html
    assert "Project introduction" not in html
    # The range paragraph and its "why" notes are on the page.
    assert "87.5%" in html
    assert "upward inflection" in html
    # The fox line is labeled a pangram, not a Harvard sentence.
    assert "pangram" in html
    # The clip-length guidance the page now carries.
    assert "about 10 seconds" in html


def test_setup_page_script_card_has_equal_height_stack(client):
    resp = client.get("/setup")
    html = resp.get_data(as_text=True)
    # The picker and the script display sit in one card, and both
    # scripts render inside the stack — so the tallest defines the
    # card's height and swapping scripts never moves the page below.
    assert 'class="card"' in html
    assert 'class="script-stack"' in html
    assert "Emotional range" in html
    assert "Harvard sentences" in html
    css = (
        Path(__file__).parent.parent
        / "src"
        / "voice_generator"
        / "web"
        / "static"
        / "style.css"
    ).read_text(encoding="utf-8")
    assert ".script-stack" in css
    assert "grid-area: 1 / 1" in css
    # Hidden blocks stay in layout (visibility, not display).
    assert ".script-block[hidden]" in css
    assert "visibility: hidden" in css


def test_setup_page_record_and_upload_are_sections(client):
    resp = client.get("/setup")
    html = resp.get_data(as_text=True)
    # Record and upload are sections of the one Add-a-reference card.
    assert 'aria-label="Add a reference"' in html
    assert "Record in the browser" in html
    assert "Or upload a file" in html
    # A large mic button starts recording; the square stop button
    # starts disabled (greyed out) until recording begins.
    assert 'aria-label="Start recording"' in html
    assert 'aria-label="Stop recording"' in html
    # The stop button's own tag carries `disabled`; the mic button's does not.
    stop_tag = html.split('id="rec-stop"')[1].split(">", 1)[0]
    assert "disabled" in stop_tag
    start_tag = html.split('id="rec-start"')[1].split(">", 1)[0]
    assert "disabled" not in start_tag
    # The upload offers drag-and-drop as well as the file picker.
    assert 'id="dropzone"' in html
    assert "Drop an audio file here" in html
    assert "or click to choose a file" in html


def test_script_endpoint_with_id(client):
    resp = client.get("/script?id=range")
    assert resp.status_code == 200
    assert "6:45 AM" in resp.get_data(as_text=True)
    assert "V-12 project launched in 2023" in resp.get_data(as_text=True)
    # An unknown id falls back to the default script (emotional range).
    resp = client.get("/script?id=bogus")
    assert "6:45 AM" in resp.get_data(as_text=True)


def test_script_endpoint(client):
    resp = client.get("/script")
    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == SCRIPT


def _upload(client, name: str, wav_path, filename: str = "clip.wav", script: str | None = None):
    data = {"clip": (open(wav_path, "rb"), filename), "name": name}
    if script is not None:
        data["script"] = script
    return client.post("/setup/upload", data=data, content_type="multipart/form-data")


def test_setup_page_shows_script_check(client):
    resp = client.get("/setup")
    html = resp.get_data(as_text=True)
    # Under the name field, the check line names the script whose
    # transcript will be saved with the clip — so a wrong selection is
    # visible before saving. The radios carry the titles the JS reads
    # to update the line when the selection changes.
    assert 'id="script-check"' in html
    assert "Transcript saved with this clip" in html
    assert "Emotional range" in html
    assert "Honestly, I never expected" in html
    assert 'data-title="Emotional range"' in html
    assert 'data-title="Harvard sentences"' in html


def test_setup_flash_shows_as_dialog(client, project):
    # A save confirmation pops up as a dialog, not just a flash at the
    # top of the page (which goes unnoticed when scrolled down).
    wav = project / "upload.wav"
    _tone_wav(wav)
    resp = _upload(client, "me2", wav)
    assert resp.status_code == 302
    page = client.get("/setup")
    html = page.get_data(as_text=True)
    assert '<dialog id="flash-dialog"' in html
    assert "Saved refs/me2.wav" in html
    # The confirmation names the transcript that was stored, so a
    # wrong script selection is caught right after saving too.
    assert "Transcript: Emotional range" in html
    assert "showModal()" in html
    # An error flash gets the dialog too.
    resp = client.post(
        "/setup/upload", data={}, content_type="multipart/form-data"
    )
    assert resp.status_code == 302
    page = client.get("/setup")
    html = page.get_data(as_text=True)
    assert '<dialog id="flash-dialog"' in html
    assert "No file chosen" in html


def test_record_api_flash_shows_dialog_on_reload(client, project):
    # The record flow reloads after the save; the flashed confirmation
    # is what the dialog shows on the reloaded page.
    wav = project / "upload.wav"
    _tone_wav(wav)
    data = {"clip": (open(wav, "rb"), "recording.wav"), "name": "api"}
    resp = client.post("/api/record", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    page = client.get("/setup")
    html = page.get_data(as_text=True)
    assert '<dialog id="flash-dialog"' in html
    assert "Saved refs/api.wav" in html
    assert "Transcript: Emotional range" in html


def test_setup_delete_removes_ref_and_transcript(client, project):
    (project / "refs" / "me.txt").write_text("The test transcript.", encoding="utf-8")
    resp = client.post("/setup/delete", data={"name": "me"})
    assert resp.status_code == 302
    assert not (project / "refs" / "me.wav").exists()
    assert not (project / "refs" / "me.txt").exists()
    page = client.get("/setup")
    html = page.get_data(as_text=True)
    assert "Deleted refs/me.wav and its transcript." in html
    assert '<dialog id="flash-dialog"' in html
    # The row is gone too, and its delete form with it.
    assert ">me</button>" not in html
    assert 'class="ref-delete-form"' not in html


def test_setup_delete_unknown_or_bad_name_errors(client, project):
    resp = client.post("/setup/delete", data={"name": "nope"})
    assert resp.status_code == 302
    page = client.get("/setup")
    assert "no reference named" in page.get_data(as_text=True)
    resp = client.post("/setup/delete", data={"name": "../x"})
    assert resp.status_code == 302
    page = client.get("/setup")
    assert "bad reference name" in page.get_data(as_text=True)
    # Nothing was deleted by either attempt.
    assert (project / "refs" / "me.wav").is_file()


def test_setup_delete_form_in_detail_row(client, project):
    (project / "refs" / "me.txt").write_text("The test transcript.", encoding="utf-8")
    html = client.get("/setup").get_data(as_text=True)
    assert 'class="ref-delete-form"' in html
    assert "Delete this reference" in html
    assert 'action="/setup/delete"' in html


def test_upload_wav_saves_ref_and_transcript(client, project):
    wav = project / "upload.wav"
    _tone_wav(wav)
    resp = _upload(client, "me2", wav)
    assert resp.status_code == 302
    assert (project / "refs" / "me2.wav").is_file()
    # The transcript is the default script (emotional range) — no script
    # field was posted, so the upload falls back to it.
    txt = (project / "refs" / "me2.txt").read_text(encoding="utf-8")
    assert "V-12 project launched in 2023" in txt
    page = client.get("/setup")
    assert "me2" in page.get_data(as_text=True)


def test_upload_with_range_script_stores_range_transcript(client, project):
    wav = project / "upload.wav"
    _tone_wav(wav)
    resp = _upload(client, "range1", wav, script="range")
    assert resp.status_code == 302
    # The transcript is the paragraph the user read, not the demo script.
    txt = (project / "refs" / "range1.txt").read_text(encoding="utf-8")
    assert "V-12 project launched in 2023" in txt
    assert "87.5%" in txt


def test_upload_unknown_script_falls_back_to_range(client, project):
    wav = project / "upload.wav"
    _tone_wav(wav)
    resp = _upload(client, "fallback", wav, script="bogus")
    assert resp.status_code == 302
    txt = (project / "refs" / "fallback.txt").read_text(encoding="utf-8")
    assert "V-12 project launched in 2023" in txt


def test_upload_duplicate_name_refused(client, project):
    wav = project / "upload.wav"
    _tone_wav(wav)
    _upload(client, "me2", wav)
    resp = _upload(client, "me2", wav, filename="clip2.wav")
    assert resp.status_code == 302
    page = client.get("/setup", follow_redirects=True)
    assert "already exists" in page.get_data(as_text=True)


def test_upload_bad_name_refused(client, project):
    wav = project / "upload.wav"
    _tone_wav(wav)
    resp = _upload(client, "../evil", wav)
    assert resp.status_code == 302
    page = client.get("/setup", follow_redirects=True)
    assert "bad reference name" in page.get_data(as_text=True)
    assert not (project / "refs" / "evil.wav").exists()


def test_upload_webm_converts_with_ffmpeg(client, project):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    wav = project / "upload.wav"
    _tone_wav(wav)
    webm = project / "clip.webm"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-c:a", "libopus", str(webm)],
        check=True,
        capture_output=True,
    )
    resp = _upload(client, "phone", webm, filename="clip.webm")
    assert resp.status_code == 302
    assert (project / "refs" / "phone.wav").is_file()
    txt = (project / "refs" / "phone.txt").read_text(encoding="utf-8")
    assert "V-12 project launched in 2023" in txt


def test_record_api_accepts_wav(client, project):
    wav = project / "upload.wav"
    _tone_wav(wav)
    data = {"clip": (open(wav, "rb"), "recording.wav"), "name": "api"}
    resp = client.post("/api/record", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["duration_s"] == 12.0
    assert (project / "refs" / "api.wav").is_file()


def test_record_api_with_harvard_script(client, project):
    wav = project / "upload.wav"
    _tone_wav(wav)
    data = {
        "clip": (open(wav, "rb"), "recording.wav"),
        "name": "harvard1",
        "script": "harvard",
    }
    resp = client.post("/api/record", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    txt = (project / "refs" / "harvard1.txt").read_text(encoding="utf-8")
    assert "The quick brown fox jumps over the lazy dog." in txt
    assert "A king ruled the state in the early days." in txt


def _poll_job(client, job_id: str, timeout_s: float = 60.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").get_json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


def test_synth_mock_job_end_to_end(client, project):
    resp = client.post(
        "/synth",
        data={"text": "Hello world.", "backend": "mock", "ref": "", "max_chars": "300"},
    )
    assert resp.status_code == 302
    job_id = resp.headers["Location"].rsplit("/jobs/", 1)[1]
    body = _poll_job(client, job_id)
    assert body["status"] == "done", body
    assert body["wav"] and body["manifest"]
    wav_name = body["wav"].rsplit("/outputs/", 1)[1]
    assert (project / "output" / wav_name).is_file()
    manifest = json.loads((project / "output" / body["manifest"].rsplit("/", 1)[1]).read_text())
    assert manifest["backend"] == "mock"
    wav_resp = client.get(body["wav"])
    assert wav_resp.status_code == 200
    assert wav_resp.headers["Content-Type"].startswith("audio/")
    history = client.get("/history")
    assert wav_name in history.get_data(as_text=True)


def test_synth_unknown_backend_rejected(client):
    resp = client.post(
        "/synth", data={"text": "Hello.", "backend": "bogus", "ref": "", "max_chars": "300"}
    )
    assert resp.status_code == 400
    assert "Unknown backend" in resp.get_data(as_text=True)


def test_synth_real_backend_needs_ref(client):
    resp = client.post(
        "/synth",
        data={"text": "Hello.", "backend": "chatterbox", "ref": "", "max_chars": "300"},
    )
    assert resp.status_code == 400
    assert "need a reference clip" in resp.get_data(as_text=True)


def test_output_path_traversal_blocked(client):
    assert client.get("/outputs/../pyproject.toml").status_code == 404
    assert client.get("/outputs/%2e%2e/pyproject.toml").status_code == 404
    assert client.get("/outputs/evil.py").status_code == 404


def test_unknown_job_404(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_history_empty_state(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "No renders yet" in resp.get_data(as_text=True)


def test_research_page_renders_report(project, client):
    (project / "docs" / "research").mkdir(parents=True)
    (project / "docs" / "research" / "report.md").write_text(
        "# Fast open voice-cloning models\n\n"
        "| Backend | License |\n|---|---|\n| VoxCPM2 | Apache-2.0 |\n",
        encoding="utf-8",
    )
    resp = client.get("/research")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Fast open voice-cloning models" in html
    assert "<table" in html  # the Markdown table is rendered, not shown raw
    assert "VoxCPM2" in html
    assert 'href="/research"' in html  # Research is in the nav


def test_research_page_without_report(client):
    resp = client.get("/research")
    assert resp.status_code == 200
    assert "No research report found" in resp.get_data(as_text=True)


def test_research_page_without_markdown_warns(project, client, monkeypatch):
    (project / "docs" / "research").mkdir(parents=True)
    (project / "docs" / "research" / "report.md").write_text(
        "# Fast open voice-cloning models\n\nSome text.\n", encoding="utf-8"
    )
    monkeypatch.setitem(sys.modules, "markdown", None)
    resp = client.get("/research")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "isn't installed" in html
    assert 'pip install -e ".[web]"' in html
    assert "<pre>" in html  # raw text still shown, but the warning says why
    assert "Some text." in html


def test_history_render_opens_modal_or_new_tab(project, client):
    synthesize_script(
        text="Hello demo.",
        reference_wav=None,
        out_wav=project / "output" / "demo.wav",
        backend_name="mock",
    )
    resp = client.get("/history")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # The modal itself: player + manifest, filled by JS on open.
    assert 'id="render-modal"' in html
    assert 'id="modal-audio"' in html
    assert 'id="modal-manifest"' in html
    # The file link opens the modal with JS; without JS it opens in a
    # new tab so the History page stays put.
    assert 'target="_blank"' in html
    assert 'data-render' in html
    assert 'data-wav="/outputs/demo.wav"' in html
    assert 'data-manifest="/outputs/demo.manifest.json"' in html
    assert 'data-manifest-name="demo.manifest.json"' in html
    # The manifest link is the same deal, not a bare navigation.
    assert html.count('target="_blank"') >= 2


def test_history_empty_state_has_no_modal(client):
    resp = client.get("/history")
    assert "No renders yet" in resp.get_data(as_text=True)
    assert "render-modal" not in resp.get_data(as_text=True)


def test_setup_trim_form_has_preview(client):
    resp = client.get("/setup")
    assert resp.status_code == 200
    # Collapse whitespace first — the helper text wraps across lines.
    html = " ".join(resp.get_data(as_text=True).split())
    # The trim form lets you hear the slice before editing its transcript.
    assert 'class="btn secondary trim-preview"' in html
    assert "Preview trim" in html
    assert "preview the slice first" in html
