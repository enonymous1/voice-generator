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
import os
import secrets
import sys
import tempfile
import time
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

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
from voice_generator.web.refs import (
    MAX_UPLOAD_BYTES,
    delete_reference,
    save_reference,
    trim_reference,
    valid_name,
)
from voice_generator.web.scripts import SCRIPTS, get_script

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
    research_path: Path = Path("docs/research")
    backend_pythons: dict[str, str] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = 5000

    def research_file(self) -> Path | None:
        """The research report to show: a file directly, or the newest
        ``*.md`` in a directory (reports are dated, so the newest name
        wins). None when there is nothing to show."""
        if self.research_path.is_file():
            return self.research_path
        if self.research_path.is_dir():
            reports = sorted(self.research_path.glob("*.md"))
            return reports[-1] if reports else None
        return None

    def research_markdown(self) -> str | None:
        """The research report's Markdown, or None when there is none."""
        report = self.research_file()
        if report is None:
            return None
        try:
            return report.read_text(encoding="utf-8")
        except OSError:
            return None

    def python_for(self, backend: str) -> str:
        """Interpreter that runs ``backend`` (default: this server's own)."""
        return self.backend_pythons.get(backend, sys.executable)

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
            if txt.is_file():
                try:
                    entry["transcript"] = txt.read_text(encoding="utf-8")
                except OSError:
                    pass  # transcript unreadable: show the row without it
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


# Tags python-markdown itself generates for the research report (headings,
# tables, code, lists, links, hard line breaks). Anything else in the
# HTML is dropped by _sanitize_html below.
_SAFE_TAGS = {
    "a", "blockquote", "br", "code", "em", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "li", "ol", "p", "pre", "strong", "table", "tbody", "td", "th",
    "thead", "tr", "ul",
}
_SAFE_ATTRS = {"a": {"href"}}
_VOID_TAGS = {"br", "hr"}


def _safe_href(value: str) -> str | None:
    """An href kept only when it is not a script: URL. Relative links and http(s)/mailto pass."""
    cleaned = "".join(ch for ch in value.strip() if ord(ch) > 0x20)
    scheme = urlparse(cleaned).scheme.lower()
    return value if scheme in ("", "http", "https", "mailto") else None


class _HTMLSanitizer(HTMLParser):
    """Allowlist sanitizer for markdown-rendered HTML.

    python-markdown passes raw HTML straight through (its safe_mode was
    removed in 3.0), and the report file is user-droppable — any ``*.md``
    in docs/research renders on the Research page. So the HTML is rebuilt
    keeping only the tags markdown itself generates, plus safe hrefs;
    everything else (script, img, event handlers, javascript: links) is
    dropped, text included. No bleach dependency — the web extra is flask
    + markdown only, and this page only needs the tags above. ``<img>``
    is deliberately not allowlisted: a report image would need a src
    attribute, and src is the classic onerror vector.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip: str | None = None  # tag whose content is dropped too

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip is not None:
            return
        if tag in ("script", "style"):
            self._skip = tag
            return
        if tag not in _SAFE_TAGS:
            return
        kept = ""
        for name, value in attrs:
            if name in _SAFE_ATTRS.get(tag, set()) and value is not None:
                if name == "href":
                    value = _safe_href(value)
                    if value is None:
                        continue
                kept += f' {name}="{escape(value, quote=True)}"'
        self._out.append(f"<{tag}{kept}>")

    def handle_endtag(self, tag: str) -> None:
        if self._skip is not None:
            if tag == self._skip:
                self._skip = None
            return
        if tag in _SAFE_TAGS and tag not in _VOID_TAGS:
            self._out.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            self.handle_starttag(tag, attrs)
        # Every other self-closing tag is dropped (img included).

    def handle_data(self, data: str) -> None:
        if self._skip is None:
            self._out.append(escape(data))

    def handle_comment(self, data: str) -> None:
        pass  # comments are dropped, not echoed

    def render(self) -> str:
        """The sanitized HTML built so far."""
        return "".join(self._out)


def _sanitize_html(html: str) -> str:
    """Rebuild ``html`` keeping only markdown-generated tags (see _HTMLSanitizer)."""
    parser = _HTMLSanitizer()
    parser.feed(html)
    parser.close()
    return parser.render()


def _find_project_root(start: Path) -> Path | None:
    """Walk up from ``start`` to the directory holding ``pyproject.toml``."""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _resolve_defaults(cfg: WebConfig) -> None:
    """Anchor relative project dirs to the project root, in place.

    ``voice-gen serve`` defaults the dirs to ``refs`` / ``output`` /
    ``docs/research`` relative to the CWD, so launching from anywhere
    but the project root silently served an empty refs/ and output/ in
    the wrong directory. Walk up to pyproject.toml and resolve the
    relative dirs against that root instead. Explicitly-passed absolute
    dirs are left alone — their caller owns them (the freeze command
    and the tests pass absolute dirs). When a defaulted refs dir still
    doesn't exist after resolution, fail loudly rather than serve an
    empty project — ``voice-gen serve`` creates the dirs first, so a
    missing one here means the launch directory is wrong.
    """
    root = _find_project_root(Path.cwd())
    defaulted_refs = not cfg.refs_dir.is_absolute()
    for attr in ("refs_dir", "output_dir", "research_path"):
        value = getattr(cfg, attr)
        if value.is_absolute():
            continue
        setattr(cfg, attr, root / value if root is not None else value)
    if defaulted_refs and not cfg.refs_dir.is_dir():
        raise ValueError(
            f"no refs/ directory at {cfg.refs_dir} — run `voice-gen serve` "
            "from the project root (or pass --refs-dir)"
        )


def create_app(config: WebConfig | None = None) -> Flask:
    """App factory (the project's Flask convention)."""
    cfg = config or WebConfig()
    _resolve_defaults(cfg)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    # Random per start (or VOICE_GEN_SECRET when set): the old constant
    # was committed to a public repo, so anyone could forge session
    # content. Sessions only carry flash messages, which are set and
    # read within one request cycle, so a per-start key costs nothing.
    app.secret_key = os.environ.get("VOICE_GEN_SECRET") or secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.vg_config = cfg  # type: ignore[attr-defined]
    app.vg_jobs = JobStore()  # type: ignore[attr-defined]

    @app.before_request
    def _check_origin():
        """Reject cross-site POSTs (localhost CSRF).

        The UI binds localhost with no auth, so a malicious page visited
        while the server runs could otherwise auto-submit a form POST to
        /setup/delete or /history/delete. Same-origin requests carry an
        Origin (or Referer) naming this host; a present-but-foreign one
        is rejected. Requests with neither header (curl, some clients)
        are allowed — there is nothing to compare — so this is a
        same-origin check, not authentication.
        """
        if request.method != "POST":
            return None
        origin = request.headers.get("Origin") or request.headers.get("Referer")
        if not origin:
            return None
        if urlparse(origin).netloc != request.host:
            abort(403)
        return None

    @app.get("/")
    def index():
        """Landing page: what the tool is, and links into it."""
        return render_template("landing.html")

    @app.get("/script")
    def script():
        # Serves a setup script's text by id; no id (or an unknown one)
        # falls back to the range paragraph — the default script and the
        # project's demo text.
        return (
            get_script(request.args.get("id")).text,
            200,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    @app.route("/synth", methods=["GET", "POST"])
    def synth():
        """Render a script with a chosen backend (GET shows the form).

        POST validates the form, writes the script to
        ``output/.web-jobs/<job-id>.txt`` *before* the job exists — the
        write happens in a temp file first, so a failed write can't
        leave a job stuck at "queued" — then starts a background job
        driving the CLI. Backend flags come from the form; the
        transcript comes from ``refs/<name>.txt`` when the setup page
        stored one — Qwen3 falls back to ``--x-vector-only`` and
        CosyVoice to cross-lingual mode without one, VoxCPM only uses
        the transcript in ultimate mode. Numeric fields are validated
        here, not just in the browser: ``max_chars`` must be 50–2000,
        VoxCPM cfg > 0, timesteps >= 1.
        """
        backends = available_backends()
        refs = cfg.list_refs()
        if request.method == "GET":
            return render_template(
                "synth.html", backends=backends, refs=refs, default_backend="chatterbox"
            )
        form = request.form
        text = (form.get("text") or "").strip()
        backend = form.get("backend") or "chatterbox"
        ref_name = form.get("ref") or ""
        if not text:
            flash("Nothing to render — paste some text first.", "error")
            return render_template("synth.html", backends=backends, refs=refs), 400
        if backend not in backends:
            flash(f"Unknown backend {backend!r}.", "error")
            return render_template("synth.html", backends=backends, refs=refs), 400
        if ref_name and not valid_name(ref_name):
            flash(f"Bad reference name {ref_name!r}.", "error")
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
        try:
            max_chars = int(form.get("max_chars") or 300)
        except ValueError:
            max_chars = 300
        if not 50 <= max_chars <= 2000:
            flash("Max chars per chunk must be between 50 and 2000.", "error")
            return render_template("synth.html", backends=backends, refs=refs), 400
        voxcpm_cfg: float | None = None
        voxcpm_timesteps: int | None = None
        if backend == "voxcpm":
            try:
                voxcpm_cfg = float(form.get("voxcpm_cfg") or "2.0")
            except ValueError:
                flash("VoxCPM guidance scale (cfg) must be a number.", "error")
                return render_template("synth.html", backends=backends, refs=refs), 400
            if voxcpm_cfg <= 0:
                flash("VoxCPM guidance scale (cfg) must be greater than 0.", "error")
                return render_template("synth.html", backends=backends, refs=refs), 400
            try:
                voxcpm_timesteps = int(form.get("voxcpm_timesteps") or "10")
            except ValueError:
                flash("VoxCPM diffusion steps must be a whole number.", "error")
                return render_template("synth.html", backends=backends, refs=refs), 400
            if voxcpm_timesteps < 1:
                flash("VoxCPM diffusion steps must be at least 1.", "error")
                return render_template("synth.html", backends=backends, refs=refs), 400

        jobs_dir = cfg.output_dir / ".web-jobs"
        try:
            jobs_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", dir=jobs_dir, suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(text)
                tmp_script = Path(tmp.name)
        except OSError as exc:
            flash(f"Could not start the render — cannot write the script file: {exc}", "error")
            return render_template("synth.html", backends=backends, refs=refs), 500
        job = app.vg_jobs.create(backend)
        script_file = jobs_dir / f"{job.id}.txt"
        try:
            os.replace(tmp_script, script_file)
        except OSError as exc:
            tmp_script.unlink(missing_ok=True)
            app.vg_jobs.remove(job.id)
            flash(f"Could not start the render — cannot write the script file: {exc}", "error")
            return render_template("synth.html", backends=backends, refs=refs), 500
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
            cmd += ["--voxcpm-cfg", str(voxcpm_cfg), "--voxcpm-timesteps", str(voxcpm_timesteps)]
        cmd += ["--max-chars", str(max_chars)]

        start_job(job, cmd, out_wav, script_file=script_file)
        return redirect(url_for("job_page", job_id=job.id))

    @app.get("/jobs/<job_id>")
    def job_page(job_id: str):
        """A render's status page: live log, then the player when it finishes."""
        job = app.vg_jobs.get(job_id)
        if job is None:
            abort(404)
        return render_template("job.html", job=job)

    @app.get("/api/jobs/<job_id>")
    def job_api(job_id: str):
        """Live status + log for one render, as a point-in-time snapshot (see JobStore.snapshot)."""
        snap = app.vg_jobs.snapshot(job_id)
        if snap is None:
            return jsonify({"error": f"unknown job {job_id}"}), 404
        return jsonify(
            {
                "id": snap["id"],
                "backend": snap["backend"],
                "status": snap["status"],
                "log": snap["log"],
                "wav": (
                    url_for("output_file", filename=snap["wav_name"])
                    if snap["wav_name"]
                    else None
                ),
                "manifest": (
                    url_for("output_file", filename=snap["manifest_name"])
                    if snap["manifest_name"]
                    else None
                ),
                "error": snap["error"],
            }
        )

    @app.get("/setup")
    def setup():
        """Setup page: recording scripts, record/upload, saved references."""
        scripts = [
            {
                "id": s.id,
                "title": s.title,
                "blurb": s.blurb,
                "why": s.why,
                "text": s.text,
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
        script = get_script(script_id)
        try:
            result = save_reference(
                upload_path=tmp_path,
                upload_suffix=suffix,
                name=name.strip(),
                refs_dir=cfg.refs_dir,
                # The transcript is the script the user actually read.
                transcript=script.text,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        result["script_title"] = script.title
        return result

    @app.post("/setup/upload")
    def setup_upload():
        """Store an uploaded recording as a reference (validates it first)."""
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
            f"Transcript: {result['script_title']}. "
            + (" ".join(str(w) for w in result["warnings"]) or "No warnings."),
            "ok",
        )
        return redirect(url_for("setup"))

    @app.post("/setup/delete")
    def setup_delete():
        """Delete a saved reference clip and its transcript."""
        name = request.form.get("name") or ""
        try:
            result = delete_reference(name=name, refs_dir=cfg.refs_dir)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("setup"))
        flash(
            f"Deleted refs/{name}.wav"
            + (" and its transcript." if result["had_transcript"] else "."),
            "ok",
        )
        return redirect(url_for("setup"))

    @app.post("/setup/trim")
    def setup_trim():
        """Trim a saved reference to [start, end) and save it as a new one."""
        source = (request.form.get("source") or "").strip()
        name = (request.form.get("name") or "").strip()
        if not valid_name(source):
            flash("Unknown source reference.", "error")
            return redirect(url_for("setup"))
        try:
            start_s = float(request.form.get("start") or 0)
            end_s = float(request.form.get("end") or 0)
        except ValueError:
            flash("Start and end must be numbers of seconds.", "error")
            return redirect(url_for("setup"))
        try:
            result = trim_reference(
                source=cfg.refs_dir / f"{source}.wav",
                name=name,
                refs_dir=cfg.refs_dir,
                start_s=start_s,
                end_s=end_s,
                transcript=request.form.get("transcript") or "",
            )
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("setup"))
        info = result["info"]
        flash(
            f"Saved refs/{name}.wav — trimmed {source} to {info.duration_s:.1f}s, "
            f"{info.sample_rate} Hz, {info.channels}ch. "
            + (" ".join(str(w) for w in result["warnings"]) or "No warnings."),
            "ok",
        )
        return redirect(url_for("setup"))

    @app.post("/api/record")
    def record_api():
        """Store a browser recording as a reference (JSON in, flash out — the page reloads)."""
        upload = request.files.get("clip")
        name = request.form.get("name") or "me"
        if upload is None or not upload.filename:
            return jsonify({"error": "no recording received"}), 400
        try:
            result = _store_upload(upload, name, request.form.get("script"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        info = result["info"]
        # Flashed, not just returned: the record page reloads after the
        # save, and the flash is what the confirmation dialog shows.
        flash(
            f"Saved refs/{name}.wav — {info.duration_s:.1f}s, "
            f"{info.sample_rate} Hz, {info.channels}ch. "
            f"Transcript: {result['script_title']}. "
            + (" ".join(str(w) for w in result["warnings"]) or "No warnings."),
            "ok",
        )
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
        """Past renders, newest first, from the manifests in the output dir."""
        return render_template("history.html", rows=cfg.list_history())

    def _delete_render(name: str) -> bool:
        """Delete a render's WAV and its manifest from the output dir.

        Returns whether a manifest was deleted too. Raises ValueError
        when the name is not a plain ``.wav`` filename in the output
        dir or no such render exists.
        """
        if not name.endswith(".wav") or "/" in name or "\\" in name:
            raise ValueError(f"bad render name {name!r}")
        wav_path = (cfg.output_dir / name).resolve()
        if not wav_path.is_relative_to(cfg.output_dir.resolve()):
            raise ValueError(f"bad render name {name!r}")
        if not wav_path.is_file():
            raise ValueError(f"no render named {name!r} to delete")
        manifest_path = wav_path.with_name(wav_path.stem + ".manifest.json")
        wav_path.unlink()
        had_manifest = manifest_path.is_file()
        if had_manifest:
            manifest_path.unlink()
        return had_manifest

    @app.post("/history/delete")
    def history_delete():
        """Delete a render's WAV and its manifest."""
        name = request.form.get("name") or ""
        try:
            deleted_manifest = _delete_render(name)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("history"))
        flash(
            f"Deleted output/{name}"
            + (" and its manifest." if deleted_manifest else "."),
            "ok",
        )
        return redirect(url_for("history"))

    @app.get("/research")
    def research():
        """Render the research report as HTML (sanitized — see _sanitize_html)."""
        md = cfg.research_markdown()
        html = None
        if md is not None:
            try:
                import markdown

                html = _sanitize_html(
                    markdown.markdown(md, extensions=["tables", "fenced_code"])
                )
            except ImportError:
                html = None  # no markdown lib: the template shows the raw text
        return render_template("research.html", content=html, raw=md)

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

    @app.get("/refs/<path:filename>")
    def ref_file(filename: str):
        """Serve a reference WAV or transcript from the refs directory only."""
        if Path(filename).suffix.lower() not in (".wav", ".txt"):
            abort(404)
        base = cfg.refs_dir.resolve()
        target = (base / filename).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            abort(404)
        return send_file(target)

    return app


__all__ = ["WebConfig", "create_app"]
