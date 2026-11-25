"""Tests for the 2026-11-25 web-UI review fixes (freeze guard, jobs, CSRF,
synth validation, refs atomicity, research sanitizer, templates, CSS).

Uses Flask's test client — no server is started, no GPU or model
downloads are needed: renders run through the ``mock`` backend in a
subprocess (the same ``python -m voice_generator synth`` the UI drives
for real backends).
"""

import json
import math
import sys
import time
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("markdown")

from voice_generator.audio import write_pcm16_wav  # noqa: E402
from voice_generator.web import jobs as jobs_mod  # noqa: E402
from voice_generator.web.app import WebConfig, create_app  # noqa: E402
from voice_generator.web.freeze import (  # noqa: E402
    _copy_output_dir,
    _rewrite_links,
    freeze_project,
)
from voice_generator.web.jobs import JobStore, start_job  # noqa: E402
from voice_generator.web.refs import save_reference, trim_reference  # noqa: E402


def _tone_wav(path, seconds: float = 12.0, rate: int = 16000) -> None:
    samples = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
    write_pcm16_wav(path, samples, rate)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A throwaway project folder: refs/ (one real clip), output/."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "refs").mkdir()
    (tmp_path / "output").mkdir()
    _tone_wav(tmp_path / "refs" / "me.wav")
    return tmp_path


@pytest.fixture()
def client(project):
    config = WebConfig(
        refs_dir=project / "refs",
        output_dir=project / "output",
    )
    return create_app(config).test_client()


def _wait_job(job, timeout_s: float = 30.0):
    deadline = time.time() + timeout_s
    while job.status not in ("done", "error") and time.time() < deadline:
        time.sleep(0.2)
    assert job.status in ("done", "error"), f"job stuck at {job.status}"
    return job


# --- freeze: unquoting, selective copy, missing media -----------------------


def test_rewrite_links_unquotes_encoded_names(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "my clip.wav").write_bytes(b"RIFF")
    config = WebConfig(refs_dir=refs, output_dir=tmp_path / "output")
    html, media = _rewrite_links('<audio src="/refs/my%20clip.wav">', config)
    # The filesystem path is the real name; the href keeps the encoding
    # (the static server decodes it back to the real name on request).
    assert media == [(refs / "my clip.wav", "refs/my clip.wav")]
    assert 'src="refs/my%20clip.wav"' in html


def test_copy_output_dir_is_selective(tmp_path):
    src = tmp_path / "output"
    (src / ".web-jobs").mkdir(parents=True)
    (src / ".demo.parts").mkdir()
    (src / "demo.wav").write_bytes(b"RIFF")
    (src / "demo.manifest.json").write_text("{}")
    (src / "notes.txt").write_text("x")
    (src / ".web-jobs" / "job.txt").write_text("x")
    (src / ".demo.parts" / "chunk_000.wav").write_bytes(b"RIFF")
    dst = tmp_path / "copy"
    _copy_output_dir(src, dst)
    assert (dst / "demo.wav").is_file()
    assert (dst / "demo.manifest.json").is_file()
    assert not (dst / "notes.txt").exists()
    assert not (dst / ".web-jobs").exists()
    assert not (dst / ".demo.parts").exists()


def test_freeze_reports_missing_media(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "project"
    (proj / "refs").mkdir(parents=True)
    (proj / "output").mkdir()
    # A manifest whose WAV is gone: the history page references it.
    (proj / "output" / "gone.manifest.json").write_text(
        json.dumps({"backend": "mock", "duration_s": 1.0, "sample_rate": 16000}),
        encoding="utf-8",
    )
    (proj / "docs" / "research").mkdir(parents=True)
    (proj / "docs" / "research" / "report.md").write_text("# R\n", encoding="utf-8")
    out = tmp_path / "frozen"
    result = freeze_project(proj, out)  # unseeded, no refs: allowed
    assert any("gone.wav" in m for m in result["missing"])
    assert (out / "history.html").is_file()


# --- jobs: pruning, timeout, WAV check, script cleanup, snapshot ------------


def test_job_store_prunes_old_jobs():
    store = JobStore()
    old = store.create("mock")
    old.created_s = time.time() - 25 * 3600  # older than a day
    assert len(store) == 1
    store.create("mock")
    assert len(store) == 1
    assert store.get(old.id) is None


def test_job_times_out(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_mod, "JOB_TIMEOUT_S", 1)
    store = JobStore()
    job = store.create("mock")
    start_job(job, [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "out.wav")
    _wait_job(job, timeout_s=10)
    assert job.status == "error"
    assert "timed out" in (job.error or "")


def test_job_requires_wav_on_success(tmp_path):
    # The command exits 0 and writes a valid manifest, but no WAV:
    # that used to report "done" with a player that 404d.
    script = tmp_path / "write_manifest.py"
    manifest_path = tmp_path / "out.manifest.json"
    script.write_text(
        f"import json; json.dump({{'backend': 'mock'}}, open({str(manifest_path)!r}, 'w'))\n",
        encoding="utf-8",
    )
    store = JobStore()
    job = store.create("mock")
    start_job(job, [sys.executable, str(script)], tmp_path / "out.wav")
    _wait_job(job)
    assert job.status == "error"
    assert "no WAV" in (job.error or "")


def test_job_deletes_script_file_when_done(tmp_path):
    script_file = tmp_path / "job.txt"
    script_file.write_text("Hello world.", encoding="utf-8")
    out_wav = tmp_path / "out.wav"
    store = JobStore()
    job = store.create("mock")
    start_job(
        job,
        [
            sys.executable,
            "-m",
            "voice_generator",
            "synth",
            "--backend",
            "mock",
            "--script",
            str(script_file),
            "--out",
            str(out_wav),
        ],
        out_wav,
        script_file=script_file,
    )
    _wait_job(job)
    assert job.status == "done", job.error
    assert out_wav.is_file()
    assert not script_file.exists()


def test_job_snapshot_is_a_copy():
    store = JobStore()
    job = store.create("mock")
    job.log.append("first line")
    snap = store.snapshot(job.id)
    assert snap is not None
    job.log.append("second line")
    assert snap["log"] == ["first line"]  # the snapshot doesn't move with the live log
    assert store.snapshot("nope") is None


# --- CSRF: Origin/Referer check, secret key ---------------------------------


def test_cross_origin_post_rejected(client):
    resp = client.post(
        "/history/delete",
        data={"name": "demo.wav"},
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 403


def test_cross_origin_referer_rejected(client):
    resp = client.post(
        "/history/delete",
        data={"name": "demo.wav"},
        headers={"Referer": "http://evil.example/page"},
    )
    assert resp.status_code == 403


def test_same_origin_post_allowed(client):
    # Same-origin posts (the browser's own forms and fetch calls) pass.
    resp = client.post(
        "/history/delete",
        data={"name": "nope.wav"},
        headers={"Origin": "http://localhost"},
    )
    assert resp.status_code == 302


def test_secret_key_random_per_start(project):
    app1 = create_app(WebConfig(refs_dir=project / "refs", output_dir=project / "output"))
    app2 = create_app(WebConfig(refs_dir=project / "refs", output_dir=project / "output"))
    assert app1.secret_key != "voice-generator-local-only"
    assert app1.secret_key != app2.secret_key


def test_secret_key_from_env(project, monkeypatch):
    monkeypatch.setenv("VOICE_GEN_SECRET", "test-secret")
    app = create_app(WebConfig(refs_dir=project / "refs", output_dir=project / "output"))
    assert app.secret_key == "test-secret"


# --- synth route: server-side validation ------------------------------------


def test_synth_rejects_bad_ref_name(client):
    resp = client.post(
        "/synth",
        data={"text": "Hello.", "backend": "mock", "ref": "../x", "max_chars": "300"},
    )
    assert resp.status_code == 400
    assert "Bad reference name" in resp.get_data(as_text=True)


def test_synth_rejects_max_chars_out_of_range(client):
    for bad in ("0", "49", "2001", "5000"):
        resp = client.post(
            "/synth",
            data={"text": "Hello.", "backend": "mock", "ref": "", "max_chars": bad},
        )
        assert resp.status_code == 400, bad
        assert "between 50 and 2000" in resp.get_data(as_text=True)
    resp = client.post(
        "/synth",
        data={"text": "Hello.", "backend": "mock", "ref": "", "max_chars": "2000"},
    )
    assert resp.status_code == 302


def test_synth_rejects_bad_voxcpm_numerics(client):
    base = {"text": "Hello.", "backend": "voxcpm", "ref": "me"}
    for field, bad in (
        ("voxcpm_cfg", "0"),
        ("voxcpm_cfg", "-1"),
        ("voxcpm_cfg", "abc"),
        ("voxcpm_timesteps", "0"),
        ("voxcpm_timesteps", "abc"),
    ):
        data = dict(base, voxcpm_cfg="2.0", voxcpm_timesteps="10")
        data[field] = bad
        resp = client.post("/synth", data=data)
        assert resp.status_code == 400, (field, bad)
    resp = client.post("/synth", data=dict(base, voxcpm_cfg="2.0", voxcpm_timesteps="10"))
    assert resp.status_code == 302


def test_synth_job_not_created_when_script_write_fails(project):
    # output/ is a file, so the .web-jobs dir can't be created: the
    # render must fail before any job exists, not leave one "queued".
    (project / "output-file").write_text("not a dir", encoding="utf-8")
    config = WebConfig(refs_dir=project / "refs", output_dir=project / "output-file")
    app = create_app(config)
    resp = app.test_client().post(
        "/synth",
        data={"text": "Hello.", "backend": "mock", "ref": "", "max_chars": "300"},
    )
    assert resp.status_code == 500
    assert len(app.vg_jobs) == 0


# --- WebConfig: project-root resolution -------------------------------------


def test_resolve_defaults_anchors_to_project_root(tmp_path, monkeypatch):
    proj = tmp_path / "project"
    (proj / "refs").mkdir(parents=True)
    (proj / "output").mkdir()
    (proj / "pyproject.toml").write_text('[project]\nname = "voice-generator"\n')
    (proj / "sub").mkdir()
    monkeypatch.chdir(proj / "sub")
    app = create_app()
    assert app.vg_config.refs_dir == proj / "refs"
    assert app.vg_config.output_dir == proj / "output"


def test_resolve_defaults_refuses_when_no_project_found(tmp_path, monkeypatch):
    # No pyproject.toml anywhere up the tree, and no refs/ here: fail
    # loudly instead of serving an empty project.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="project root"):
        create_app()


# --- refs: truncated clips, atomic publish ----------------------------------


def test_save_reference_truncated_wav_is_clean_error(project):
    truncated = project / "truncated.wav"
    _tone_wav(truncated, seconds=2.0)
    data = truncated.read_bytes()
    truncated.write_bytes(data[:20])  # header cut mid-chunk: EOFError from wave
    with pytest.raises(ValueError, match="unreadable|convert"):
        save_reference(
            upload_path=truncated,
            upload_suffix=".wav",
            name="trunc",
            refs_dir=project / "refs",
            transcript="Hello.",
        )
    assert not (project / "refs" / "trunc.wav").exists()
    assert not (project / "refs" / "trunc.txt").exists()


def test_trim_reference_truncated_source_is_clean_error(project):
    truncated = project / "refs" / "trunc.wav"
    _tone_wav(truncated, seconds=2.0)
    data = truncated.read_bytes()
    truncated.write_bytes(data[:20])
    with pytest.raises(ValueError, match="unreadable"):
        trim_reference(
            source=truncated,
            name="trunc_trim",
            refs_dir=project / "refs",
            start_s=0,
            end_s=1,
            transcript="Hello.",
        )
    assert not (project / "refs" / "trunc_trim.wav").exists()


# --- research: sanitizer -----------------------------------------------------


def test_research_page_strips_raw_html(project, client):
    (project / "docs" / "research").mkdir(parents=True)
    (project / "docs" / "research" / "report.md").write_text(
        "# Report\n\n"
        "<script>alert(1)</script>\n\n"
        "<img src=x onerror=alert(1)>\n\n"
        "[click](javascript:alert(1))\n\n"
        "| Backend | License |\n|---|---|\n| VoxCPM2 | Apache-2.0 |\n",
        encoding="utf-8",
    )
    resp = client.get("/research")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "<script" not in html
    assert "onerror" not in html
    assert "javascript:" not in html
    assert "<img" not in html
    # Legitimate Markdown still renders: the table survives sanitizing.
    assert "<table" in html
    assert "VoxCPM2" in html


# --- templates: dialog guard, job poller, blob revoke, CSS -------------------


def test_flash_dialog_has_feature_guard(client):
    resp = client.post("/setup/upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 302
    html = client.get("/setup").get_data(as_text=True)
    assert '<dialog id="flash-dialog"' in html
    assert "if (!d || !d.showModal) return;" in html


def test_job_page_poller_handles_unknown_jobs(client):
    job = client.application.vg_jobs.create("mock")
    html = client.get(f"/jobs/{job.id}").get_data(as_text=True)
    # The poller fetches the API route by url_for (not a hardcoded path),
    # checks the response before touching the payload, and reports an
    # orphaned job instead of "Status: undefined" on a loop.
    assert f'fetch("/api/jobs/{job.id}")' in html
    assert "if (r.status === 404)" in html
    assert "This render is no longer known" in html
    assert ".catch(function (err)" in html
    assert "failures >= 3" in html


def test_job_page_api_path_uses_script_root(client):
    # Mounted under a subpath, the poller must still hit the API route:
    # url_for builds the path, a hardcoded "/api/jobs/..." would not.
    job = client.application.vg_jobs.create("mock")
    html = client.get(
        f"/jobs/{job.id}", environ_overrides={"SCRIPT_NAME": "/vg"}
    ).get_data(as_text=True)
    assert f'fetch("/vg/api/jobs/{job.id}")' in html


def test_setup_page_revokes_blob_urls(client):
    html = client.get("/setup").get_data(as_text=True)
    assert "URL.revokeObjectURL(playbackUrl)" in html


def test_btn_uses_btn_bg():
    css = (
        Path(__file__).parent.parent
        / "src"
        / "voice_generator"
        / "web"
        / "static"
        / "style.css"
    ).read_text(encoding="utf-8")
    btn = css.split(".btn {", 1)[1].split("}", 1)[0]
    assert "var(--btn-bg)" in btn  # dark mode: white on #2b6cb0, not #7db3f5
    assert "var(--accent)" not in btn
    # The stale pre-v0.5.17 comment above .addref-grid is gone.
    assert "Setup page: record (1 part) beside upload (3 parts)" not in css


# --- docstrings (PEP 257) ----------------------------------------------------


def test_web_docstrings_present(client):
    app = client.application
    assert app.view_functions["synth"].__doc__ is not None
    assert "max_chars" in app.view_functions["synth"].__doc__
    assert app.view_functions["job_api"].__doc__ is not None
    from voice_generator.web.jobs import JobStore as _JobStore
    from voice_generator.web.refs import valid_name as _valid_name
    from voice_generator.web.scripts import Script as _Script

    assert _JobStore.create.__doc__ is not None
    assert _JobStore.get.__doc__ is not None
    assert _valid_name.__doc__ is not None
    assert _Script.__doc__ is not None
