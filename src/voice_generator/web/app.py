"""Local web UI for voice-generator (Flask + Jinja2).

Started by ``voice-gen serve``; binds localhost only by default.
Three pages:

  /synth   — paste text, pick a backend + reference clip, render, listen.
  /setup   — the recording script, record/upload a reference clip.
  /history — past renders (WAV + manifest) from the output directory.

Synthesis runs in a background thread that drives the CLI as a
subprocess (see jobs.py), so the server never imports torch and the web
UI can never drift from the CLI's render path. The reference transcript
is stored next to each clip (``refs/<name>.txt``) by the setup page, so
the backends that need one (Qwen3, CosyVoice, VoxCPM ultimate) get it
automatically; Qwen3 falls back to ``--x-vector-only`` and CosyVoice to
cross-lingual mode when a clip has no transcript.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from voice_generator.audio import wav_info
from voice_generator.backends import available_backends
from voice_generator.web.jobs import JobStore, start_job
from voice_generator.web.refs import MAX_UPLOAD_BYTES, save_reference
from voice_generator.web.scripts import SCRIPTS, get_script, resolve_text

# Which backends take which extra CLI flags from the synth form.
_BACKEND_FLAGS: dict[str, list[str]] = {
    "chatterbox": ["variant"],
    "qwen3": ["model_size"],
    "voxcpm": ["voxcpm_mode", "voxcpm_cfg", "voxcpm_timesteps"],
}


@dataclass
class WebConfig:
    """Everything the UI needs to know about the project folders."""

    refs_dir: Path = Path("refs")
    output_dir: Path = Path("output")
    script_path: Path = Path("recording-script.txt")
    backend_pythons: dict[str, str] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = 5000

    def python_for(self, backend: str) -> str:
        """Interpreter that runs ``backend`` (default: this server's own)."""
        return self.backend_pythons.get(backend, sys.executable)

    def script_text(self) -> str:
        if self.script_path.is_file():
            return self.script_path.read_text(encoding="utf-8")
        return ""

    def list_refs(self) -> list[dict[str, object]]:
        """Reference clips in ``refs_dir``, with validation info."""
        refs: list[dict[str, object]] = []
        if not self.refs_dir.is_dir():
            return refs
        for wav in sorted(self.refs_dir.glob("*.wav")):
            entry: dict[str, object] = {"name": wav.stem, "wav": wav.name}
            try:
                info = wav_info(wav)
                entry.update(
                    {
                        "duration_s": round(info.duration_s, 1),
                        "sample_rate": info.sample_rate,
                        "channels": info.channels,
                    }
                )
            except Exception as exc:  # unreadable clip: show it, don't crash the page
                entry["error"] = str(exc)
            txt = self.refs_dir / f"{wav.stem}.txt"
            entry["has_transcript"] = txt.is_file()
            refs.append(entry)
        return refs

    def list_history(self) -> list[dict[str, object]]:
        """Past renders, newest first, from the manifests in ``output_dir``."""
        rows: list[dict[str, object]] = []
        if not self.output_dir.is_dir():
            return rows
        for manifest_path in self.output_dir.glob("*.manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            wav_name = manifest_path.name[: -len(".manifest.json")] + ".wav"
            rows.append(
                {
                    "name": wav_name,
                    "manifest": manifest_path.name,
                    "backend": manifest.get("backend", "?"),
                    "duration_s": manifest.get("duration_s"),
                    "sample_rate": manifest.get("sample_rate"),
                    "mtime": manifest_path.stat().st_mtime,
                }
            )
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return rows


def create_app(config: WebConfig | None = None) -> Flask:
    """App factory (the project's Flask convention)."""
    cfg = config or WebConfig()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "voice-generator-local-only"  # localhost UI; flash messages only
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.vg_config = cfg  # type: ignore[attr-defined]
    app.vg_jobs = JobStore()  # type: ignore[attr-defined]

    @app.get("/")
    def index():
        return redirect(url_for("synth"))

    @app.get("/script")
    def script():
        # ?id=range / ?id=harvard picks a script; default is the intro.
        text = resolve_text(get_script(request.args.get("id")), cfg.script_text())
        return text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.route("/synth", methods=["GET", "POST"])
    def synth():
        backends = available_backends()
        refs = cfg.list_refs()
        if request.method == "GET":
            return render_template(
                "synth.html", backends=backends, refs=refs, default_backend="chatterbox"
            )
        text = (request.form.get("text") or "").strip()
        backend = request.form.get("backend") or "chatterbox"
        ref_name = request.form.get("ref") or ""
        if not text:
            flash("Nothing to render — paste some text first.", "error")
            return render_template("synth.html", backends=backends, refs=refs), 400
        if backend not in backends:
            flash(f"Unknown backend {backend!r}.", "error")
            return render_template("synth.html", backends=backends, refs=refs), 400
        ref_path = (cfg.refs_dir / f"{ref_name}.wav") if ref_name else None
        if ref_name and not (ref_path and ref_path.is_file()):
            flash(f"Reference clip {ref_name!r} not found — record one on the Setup page.", "error")
            return render_template("synth.html", backends=backends, refs=refs), 400
        if backend != "mock" and ref_path is None:
            flash(
                "Real backends need a reference clip — pick one, or record one on Setup.",
                "error",
            )
            return render_template("synth.html", backends=backends, refs=refs), 400

        job = app.vg_jobs.create(backend)
        jobs_dir = cfg.output_dir / ".web-jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        script_file = jobs_dir / f"{job.id}.txt"
        script_file.write_text(text, encoding="utf-8")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_wav = cfg.output_dir / f"{backend}-{stamp}-{job.id[:6]}.wav"

        cmd = [
            cfg.python_for(backend),
            "-m",
            "voice_generator",
            "synth",
            "--backend",
            backend,
            "--script",
            str(script_file),
            "--out",
            str(out_wav),
        ]
        if ref_path is not None:
            cmd += ["--ref", str(ref_path)]
        # Backend-specific flags from the form; the transcript comes from
        # refs/<name>.txt when the setup page stored one.
        transcript = (cfg.refs_dir / f"{ref_name}.txt") if ref_name else None
        has_transcript = bool(transcript and transcript.is_file())
        form = request.form
        if backend == "chatterbox":
            cmd += ["--variant", form.get("variant") or "turbo"]
        if backend == "qwen3":
            cmd += ["--model-size", form.get("model_size") or "1.7b"]
            if has_transcript:
                cmd += ["--ref-text-file", str(transcript)]
            else:
                cmd += ["--x-vector-only"]
        if backend == "cosyvoice":
            if has_transcript:
                cmd += ["--ref-text-file", str(transcript)]
            else:
                cmd += ["--cosy-mode", "cross_lingual"]
        if backend == "voxcpm":
            mode = form.get("voxcpm_mode") or "controllable"
            cmd += ["--voxcpm-mode", mode]
            if mode == "ultimate" and has_transcript:
                cmd += ["--ref-text-file", str(transcript)]
            cmd += ["--voxcpm-cfg", form.get("voxcpm_cfg") or "2.0"]
            cmd += ["--voxcpm-timesteps", form.get("voxcpm_timesteps") or "10"]
        try:
            max_chars = int(form.get("max_chars") or 300)
        except ValueError:
            max_chars = 300
        cmd += ["--max-chars", str(max_chars)]

        start_job(app.vg_jobs, job, cmd, out_wav)
        return redirect(url_for("job_page", job_id=job.id))

    @app.get("/jobs/<job_id>")
    def job_page(job_id: str):
        job = app.vg_jobs.get(job_id)
        if job is None:
            abort(404)
        return render_template("job.html", job=job)

    @app.get("/api/jobs/<job_id>")
    def job_api(job_id: str):
        job = app.vg_jobs.get(job_id)
        if job is None:
            return jsonify({"error": f"unknown job {job_id}"}), 404
        return jsonify(
            {
                "id": job.id,
                "backend": job.backend,
                "status": job.status,
                "log": job.log,
                "wav": url_for("output_file", filename=job.wav_name) if job.wav_name else None,
                "manifest": (
                    url_for("output_file", filename=job.manifest_name)
                    if job.manifest_name
                    else None
                ),
                "error": job.error,
            }
        )

    @app.get("/setup")
    def setup():
        scripts = [
            {
                "id": s.id,
                "title": s.title,
                "blurb": s.blurb,
                "why": s.why,
                "text": resolve_text(s, cfg.script_text()),
            }
            for s in SCRIPTS
        ]
        return render_template(
            "setup.html", scripts=scripts, refs=cfg.list_refs()
        )

    def _store_upload(upload, name: str, script_id: str | None = None) -> dict[str, object]:
        """Shared intake for the form upload and the JS recorder."""
        suffix = Path(upload.filename or "").suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            upload.save(tmp.name)
            tmp_path = Path(tmp.name)
        try:
            return save_reference(
                upload_path=tmp_path,
                upload_suffix=suffix,
                name=name.strip(),
                refs_dir=cfg.refs_dir,
                # The transcript is the script the user actually read.
                transcript=resolve_text(get_script(script_id), cfg.script_text()),
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post("/setup/upload")
    def setup_upload():
        upload = request.files.get("clip")
        name = request.form.get("name") or "me"
        if upload is None or not upload.filename:
            flash("No file chosen — pick a WAV (or record one in the browser).", "error")
            return redirect(url_for("setup"))
        try:
            result = _store_upload(upload, name, request.form.get("script"))
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("setup"))
        info = result["info"]
        flash(
            f"Saved refs/{name}.wav — {info.duration_s:.1f}s, "
            f"{info.sample_rate} Hz, {info.channels}ch. "
            + (" ".join(str(w) for w in result["warnings"]) or "No warnings."),
            "ok",
        )
        return redirect(url_for("setup"))

    @app.post("/api/record")
    def record_api():
        upload = request.files.get("clip")
        name = request.form.get("name") or "me"
        if upload is None or not upload.filename:
            return jsonify({"error": "no recording received"}), 400
        try:
            result = _store_upload(upload, name, request.form.get("script"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        info = result["info"]
        return jsonify(
            {
                "ok": True,
                "name": name,
                "duration_s": round(info.duration_s, 1),
                "sample_rate": info.sample_rate,
                "warnings": result["warnings"],
            }
        )

    @app.get("/history")
    def history():
        return render_template("history.html", rows=cfg.list_history())

    @app.get("/outputs/<path:filename>")
    def output_file(filename: str):
        """Serve a render WAV or manifest from the output directory only."""
        if Path(filename).suffix.lower() not in (".wav", ".json"):
            abort(404)
        base = cfg.output_dir.resolve()
        target = (base / filename).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            abort(404)
        return send_file(target)

    return app


__all__ = ["WebConfig", "create_app"]
