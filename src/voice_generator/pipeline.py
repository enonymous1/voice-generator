"""Script -> chunked synthesis -> one WAV + a provenance manifest.

The manifest is the contract with Stage 2: it records exactly which backend,
reference clip and settings made the audio, chunk by chunk, with durations —
so a lip-sync render can always be traced back to its voice settings.

Design note: ``synthesize_script`` is the stable public entry point; it
takes the three required options explicitly and forwards the rest through
``**overrides`` to ``SynthConfig`` (unknown keys are rejected, not silently
dropped), so a new option is added to the dataclass only — the old design
listed all 22 fields three times (dataclass, signature, forwarding), and a
field added in one place but missed in the forwarding was silently dropped
by the public API. ``SynthConfig`` is the single object the run actually
passes around, and ``backend_kwargs()`` is the one place that translates
generic options into per-backend constructor arguments — adding a fourth
backend means extending that method (and the CLI), not touching the render
loop.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

from voice_generator.audio import WavInfo, check_reference, concat_wavs, wav_info
from voice_generator.backends import get_backend
from voice_generator.backends.base import SynthesisRequest
from voice_generator.text import chunk_script

MANIFEST_VERSION = 1
# Bump when the manifest schema changes incompatibly; Stage 2 must accept
# missing ``manifest_version`` as version 0 (pre-review manifests).


@dataclass
class SynthConfig:
    """All options for one render, backend-agnostic plus backend-specific.

    Backend-specific fields are ignored by backends that do not use them;
    the mapping to constructor arguments lives in ``backend_kwargs()`` so
    the render loop never branches on backend names.
    """

    text: str
    reference_wav: Path | None
    out_wav: Path
    backend_name: str = "chatterbox"
    variant: str = "turbo"
    language: str = "en"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    max_chars: int = 300
    work_dir: Path | None = None
    ref_text: str | None = None
    x_vector_only: bool = False
    model_size: str = "1.7b"
    attn: str | None = None
    cosyvoice_home: str | None = None
    cosy_model_dir: str | None = None
    cosy_mode: str = "zero_shot"
    cosy_version: str = "3"
    voxcpm_model: str = "openbmb/VoxCPM2"
    voxcpm_mode: str = "controllable"
    voxcpm_cfg: float = 2.0
    voxcpm_timesteps: int = 10
    voxcpm_optimize: bool = True

    def backend_kwargs(self) -> dict[str, object]:
        """Translate generic options into this backend's constructor kwargs."""
        if self.backend_name == "chatterbox":
            return {"variant": self.variant}
        if self.backend_name == "qwen3":
            kwargs: dict[str, object] = {"size": self.model_size}
            if self.attn is not None:
                kwargs["attn_implementation"] = self.attn
            return kwargs
        if self.backend_name == "cosyvoice":
            kwargs = {"version": self.cosy_version, "mode": self.cosy_mode}
            if self.cosyvoice_home is not None:
                kwargs["cosyvoice_home"] = self.cosyvoice_home
            if self.cosy_model_dir is not None:
                kwargs["model_dir"] = self.cosy_model_dir
            return kwargs
        if self.backend_name == "voxcpm":
            return {
                "model_id": self.voxcpm_model,
                "mode": self.voxcpm_mode,
                "cfg_value": self.voxcpm_cfg,
                "inference_timesteps": self.voxcpm_timesteps,
                "optimize": self.voxcpm_optimize,
            }
        return {}


@dataclass
class RunResult:
    """What one ``synthesize_script`` run produced.

    ``duration_s`` is the total length of the finished WAV, not a per-chunk
    figure; ``manifest_path`` is the provenance file Stage 2 reads to
    reproduce (and lip-sync) the render.
    """

    wav_path: Path
    manifest_path: Path
    chunks: int
    duration_s: float


def synthesize_script(
    *,
    text: str,
    reference_wav: Path | None,
    out_wav: Path,
    **overrides: object,
) -> RunResult:
    """Render ``text`` through ``backend_name`` to ``out_wav`` + manifest.

    The three required options are explicit; every other ``SynthConfig``
    field is passed as a keyword in ``**overrides`` and forwarded verbatim
    (unknown keys raise ``TypeError`` — a typo must fail loudly, not vanish
    into a default). Raises ``ValueError`` for an empty script or a
    backend's missing inputs (reference clip, transcript),
    ``FileNotFoundError`` for a missing reference clip, and
    ``BackendNotInstalledError`` when a backend package is not installed
    in the current environment. Chunk WAVs are kept in ``work_dir``
    (default ``.<stem>.parts`` beside the output) so a failed chunk can be
    inspected without re-rendering the whole script; on success the
    auto-derived directory is removed again (an explicitly passed
    ``work_dir`` is always left alone).
    """
    known = {f.name for f in fields(SynthConfig)} - {"text", "reference_wav", "out_wav"}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise TypeError(
            "synthesize_script() got unexpected keyword argument(s): " + ", ".join(unknown)
        )
    return _run(SynthConfig(text=text, reference_wav=reference_wav, out_wav=out_wav, **overrides))


def _run(config: SynthConfig) -> RunResult:
    """Do one render: chunk, synthesize each chunk, join, write the manifest.

    Kept separate from ``synthesize_script`` so the public wrapper only
    builds the config and this function owns the render loop. The chunk
    work directory is removed on success when it was auto-derived (a
    successful render must not leave its intermediates beside the output
    forever); on failure it is left in place, exactly so the failed chunk
    can be inspected, and an explicitly passed ``work_dir`` is never
    removed — the caller asked for it.
    """
    if config.reference_wav is not None:
        for warning in check_reference(config.reference_wav):
            print(f"  ! {warning}")
    backend = get_backend(config.backend_name, **config.backend_kwargs())

    chunks = chunk_script(config.text, max_chars=config.max_chars)
    if not chunks:
        raise ValueError("script is empty")
    work = config.work_dir or config.out_wav.parent / f".{config.out_wav.stem}.parts"
    work.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    manifest_chunks: list[dict[str, object]] = []
    backend_settings: dict[str, object] = {}
    for i, chunk in enumerate(chunks, start=1):
        part = work / f"chunk_{i:03d}.wav"
        result = backend.synthesize(
            SynthesisRequest(
                text=chunk,
                reference_wav=config.reference_wav,
                language=config.language,
                exaggeration=config.exaggeration,
                cfg_weight=config.cfg_weight,
                ref_text=config.ref_text,
                x_vector_only=config.x_vector_only,
            ),
            part,
        )
        parts.append(part)
        # Backend settings are constant across chunks in a run; keep the
        # first chunk's so the manifest can reproduce the render. (Before
        # this fix the manifest dropped them entirely — it named the
        # backend but not its variant/model, so it was not reproducible.)
        if i == 1:
            backend_settings = dict(result.settings)
        manifest_chunks.append(
            {"index": i, "text": chunk, "wav": part.name, "duration_s": round(result.duration_s, 3)}
        )
        print(f"  [{i}/{len(chunks)}] {result.duration_s:.1f}s  {chunk[:60]}...")

    if len(parts) > 1:
        final = concat_wavs(parts, config.out_wav)
    else:
        final = _copy_single(parts[0], config.out_wav)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "backend": config.backend_name,
        "license": backend.license,
        "backend_settings": backend_settings,
        "reference_wav": str(config.reference_wav) if config.reference_wav else None,
        "reference_text": config.ref_text,
        "x_vector_only": config.x_vector_only,
        "language": config.language,
        "output": config.out_wav.name,
        "duration_s": round(final.duration_s, 3),
        "sample_rate": final.sample_rate,
        "chunks": manifest_chunks,
    }
    manifest_path = config.out_wav.with_suffix(".manifest.json")
    _write_atomic(manifest_path, json.dumps(manifest, indent=2))
    if config.work_dir is None:
        # Success, and the work dir was auto-derived: the intermediates
        # have served their purpose (the joined WAV + manifest are the
        # render). A shorter re-render used to leave stale chunk files
        # behind here, silently, on every run.
        shutil.rmtree(work, ignore_errors=True)
    return RunResult(
        wav_path=config.out_wav,
        manifest_path=manifest_path,
        chunks=len(chunks),
        duration_s=final.duration_s,
    )


def _write_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a temp file + rename.

    A crash mid-write must never leave a half-written manifest: Stage 2
    reads this file as the render's provenance, and a corrupt manifest
    looks like a corrupt render. The temp file is unique per call — a fixed
    ``.tmp`` name let two renders of the same output clobber each other —
    and fsynced before the rename, so the manifest survives a crash, not
    just a clean exit. A failed write removes its temp file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
    )
    try:
        with tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    os.replace(tmp.name, path)


def _copy_single(part: Path, out_wav: Path) -> WavInfo:
    """Copy a one-chunk render to its output path (a no-op when identical)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if part.resolve() != out_wav.resolve():
        shutil.copyfile(part, out_wav)
    return wav_info(out_wav)


__all__ = ["MANIFEST_VERSION", "RunResult", "SynthConfig", "synthesize_script"]
