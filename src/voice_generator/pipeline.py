"""Script -> chunked synthesis -> one WAV + a provenance manifest.

The manifest is the contract with Stage 2: it records exactly which backend,
reference clip and settings made the audio, chunk by chunk, with durations —
so a lip-sync render can always be traced back to its voice settings.

Design note: ``synthesize_script`` is the stable public entry point and
keeps its keyword arguments for callers and tests. ``SynthConfig`` is the
single object the run actually passes around; new options are added to
the dataclass, not to a growing parameter list, and ``backend_kwargs()``
is the one place that translates generic options into per-backend
constructor arguments — adding a fourth backend means extending that
method (and the CLI), not touching the render loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from voice_generator.audio import WavInfo, check_reference, concat_wavs, wav_info
from voice_generator.backends import get_backend
from voice_generator.backends.base import SynthesisRequest
from voice_generator.text import chunk_script

MANIFEST_VERSION = 1
"""Bump when the manifest schema changes incompatibly; Stage 2 must accept
missing ``manifest_version`` as version 0 (pre-review manifests)."""


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
        return {}


@dataclass
class RunResult:
    wav_path: Path
    manifest_path: Path
    chunks: int
    duration_s: float


def synthesize_script(
    *,
    text: str,
    reference_wav: Path | None,
    out_wav: Path,
    backend_name: str = "chatterbox",
    variant: str = "turbo",
    language: str = "en",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    max_chars: int = 300,
    work_dir: Path | None = None,
    ref_text: str | None = None,
    x_vector_only: bool = False,
    model_size: str = "1.7b",
    attn: str | None = None,
    cosyvoice_home: str | None = None,
    cosy_model_dir: str | None = None,
    cosy_mode: str = "zero_shot",
    cosy_version: str = "3",
) -> RunResult:
    """Render ``text`` through ``backend_name`` to ``out_wav`` + manifest.

    Raises ``ValueError`` for an empty script or a backend's missing
    inputs (reference clip, transcript), ``FileNotFoundError`` for a
    missing reference clip, and ``RuntimeError`` when a backend package
    is not installed in the current environment. Chunk WAVs are kept in
    ``work_dir`` (default ``.<stem>.parts`` beside the output) so a failed
    chunk can be inspected without re-rendering the whole script.
    """
    return _run(
        SynthConfig(
            text=text,
            reference_wav=reference_wav,
            out_wav=out_wav,
            backend_name=backend_name,
            variant=variant,
            language=language,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            max_chars=max_chars,
            work_dir=work_dir,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
            model_size=model_size,
            attn=attn,
            cosyvoice_home=cosyvoice_home,
            cosy_model_dir=cosy_model_dir,
            cosy_mode=cosy_mode,
            cosy_version=cosy_version,
        )
    )


def _run(config: SynthConfig) -> RunResult:
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
    looks like a corrupt render.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _copy_single(part: Path, out_wav: Path) -> WavInfo:
    import shutil

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if part.resolve() != out_wav.resolve():
        shutil.copyfile(part, out_wav)
    return wav_info(out_wav)


__all__ = ["MANIFEST_VERSION", "RunResult", "SynthConfig", "synthesize_script"]
