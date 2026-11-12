"""Background synthesis jobs for the web UI.

A render can take a minute on GPU, so the request that starts one must
not block on it: the job runs in a thread and the browser polls
``/api/jobs/<id>`` for status and the live log.

A job runs the CLI in a subprocess — ``python -m voice_generator
synth ...`` — the same code path as the terminal, so the web UI can
never drift from the CLI's render path, and the server process itself
never imports torch. Which interpreter runs the backend comes from the
web config (default: this server's own Python, which is the single-venv
setup on the target box; separate backend venvs pass their own
``--<backend>-python`` to ``voice-gen serve``).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

MAX_LOG_LINES = 500


@dataclass
class Job:
    """One render: its status, streamed CLI log, and result files."""

    id: str
    backend: str
    status: str = "queued"  # queued | running | done | error
    log: list[str] = field(default_factory=list)
    wav_name: str | None = None
    manifest_name: str | None = None
    error: str | None = None
    created_s: float = field(default_factory=time.time)


class JobStore:
    """Thread-safe registry of jobs, keyed by id."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, backend: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], backend=backend)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


def start_job(store: JobStore, job: Job, cmd: list[str], out_wav: Path) -> None:
    """Run ``cmd`` in a daemon thread; stream its output into ``job.log``.

    On exit 0 the CLI has written the WAV + manifest, so the job records
    their names; otherwise the tail of the log becomes the error.
    """

    def _run() -> None:
        job.status = "running"
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except OSError as exc:
            job.error = f"could not start the synth process: {exc}"
            job.status = "error"
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                job.log.append(line)
                if len(job.log) > MAX_LOG_LINES:
                    del job.log[:100]
        proc.wait()
        if proc.returncode == 0:
            manifest_path = out_wav.with_suffix(".manifest.json")
            try:
                json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                job.error = f"render finished but the manifest is unreadable: {exc}"
                job.status = "error"
                return
            job.wav_name = out_wav.name
            job.manifest_name = manifest_path.name
            job.status = "done"
        else:
            job.error = "\n".join(job.log[-20:]) or (
                f"synth exited with code {proc.returncode}"
            )
            job.status = "error"

    threading.Thread(target=_run, daemon=True).start()
