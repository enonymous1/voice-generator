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
MAX_JOB_AGE_S = 24 * 3600  # jobs older than a day are pruned on the next create
JOB_TIMEOUT_S = 30 * 60  # a render running longer is killed, not left "running"


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
        """Register a new job, dropping jobs older than a day first.

        The store would otherwise grow forever — every render adds a Job
        holding up to 500 log lines, and nothing ever removed one.
        ``created_s`` exists for exactly this pruning.
        """
        job = Job(id=uuid.uuid4().hex[:12], backend=backend)
        now = time.time()
        with self._lock:
            self._jobs = {
                job_id: old
                for job_id, old in self._jobs.items()
                if now - old.created_s < MAX_JOB_AGE_S
            }
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """The job with ``job_id``, or None when it was never created (or was pruned)."""
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, object] | None:
        """A point-in-time copy of a job's fields for the API, or None.

        The worker thread keeps appending to the live log while a render
        runs; serializing that live list would hand jsonify a list that
        mutates mid-serialization. Copying it under the lock gives the
        API a consistent snapshot instead. (The copy is safe to take
        without the writer holding the lock — ``list()`` of a list is
        atomic in CPython — and terminal status is only set after the
        reader thread is joined, so a "done" snapshot always carries
        the final log.)
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "id": job.id,
                "backend": job.backend,
                "status": job.status,
                "log": list(job.log),
                "wav_name": job.wav_name,
                "manifest_name": job.manifest_name,
                "error": job.error,
            }

    def remove(self, job_id: str) -> None:
        """Drop a job from the registry (its setup failed before it ran)."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def __len__(self) -> int:
        """Jobs currently tracked (including finished ones not yet pruned)."""
        with self._lock:
            return len(self._jobs)


def start_job(job: Job, cmd: list[str], out_wav: Path, script_file: Path | None = None) -> None:
    """Run ``cmd`` in a daemon thread; stream its output into ``job.log``.

    On exit 0 the CLI has written the WAV + manifest, so the job records
    their names — but only after checking both files actually exist; a
    backend that exits 0 without writing audio used to report "done"
    with a player that 404d. Otherwise the tail of the log becomes the
    error. A render that runs longer than ``JOB_TIMEOUT_S`` is killed
    and marked ``error`` — a hung backend (the Qwen3 CPU-only stall this
    project already hit) used to stick at "running" forever, with no way
    out short of killing the server.

    ``script_file`` is the job's input script in ``output/.web-jobs``;
    it is deleted once the job reaches a terminal state, so render
    inputs don't accumulate forever next to the renders.
    """

    def _execute() -> None:
        job.status = "running"
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except OSError as exc:
            job.error = f"could not start the synth process: {exc}"
            job.status = "error"
            return
        if proc.stdout is None:
            job.error = "could not capture the synth process output"
            job.status = "error"
            return

        def _drain() -> None:
            # Own thread: reading stdout to EOF blocks forever on a hung
            # process that prints nothing, so the wait below (which
            # enforces the timeout) can't share this thread.
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    job.log.append(line)
                    if len(job.log) > MAX_LOG_LINES:
                        del job.log[:100]

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=JOB_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=5)
            job.error = f"render timed out after {JOB_TIMEOUT_S // 60} minutes — killed"
            job.status = "error"
            return
        reader.join()
        if proc.returncode == 0:
            manifest_path = out_wav.with_suffix(".manifest.json")
            if not out_wav.is_file():
                job.error = "render finished but no WAV was written"
                job.status = "error"
                return
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

    def _run() -> None:
        _execute()
        if script_file is not None:
            script_file.unlink(missing_ok=True)

    threading.Thread(target=_run, daemon=True).start()
