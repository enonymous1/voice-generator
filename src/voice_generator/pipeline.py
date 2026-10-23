"""Script -> chunked synthesis -> one WAV + a provenance manifest.

The manifest is the contract with Stage 2: it records exactly which backend,
reference clip and settings made the audio, chunk by chunk, with durations —
so a lip-sync render can always be traced back to its voice settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voice_generator.audio import check_reference, concat_wavs, wav_info
from voice_generator.backends import get_backend
from voice_generator.backends.base import SynthesisRequest
from voice_generator.text import chunk_script


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
) -> RunResult:
    if reference_wav is not None:
        for warning in check_reference(reference_wav):
            print(f"  ! {warning}")
    backend_kwargs: dict[str, object] = {}
    if backend_name == "chatterbox":
        backend_kwargs = {"variant": variant}
    elif backend_name == "qwen3":
        backend_kwargs = {"size": model_size}
        if attn is not None:
            backend_kwargs["attn_implementation"] = attn
    backend = get_backend(backend_name, **backend_kwargs)

    chunks = chunk_script(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("script is empty")
    work = work_dir or out_wav.parent / f".{out_wav.stem}.parts"
    work.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    manifest_chunks: list[dict[str, object]] = []
    for i, chunk in enumerate(chunks, start=1):
        part = work / f"chunk_{i:03d}.wav"
        result = backend.synthesize(
            SynthesisRequest(
                text=chunk,
                reference_wav=reference_wav,
                language=language,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                ref_text=ref_text,
                x_vector_only=x_vector_only,
            ),
            part,
        )
        parts.append(part)
        manifest_chunks.append(
            {"index": i, "text": chunk, "wav": part.name, "duration_s": round(result.duration_s, 3)}
        )
        print(f"  [{i}/{len(chunks)}] {result.duration_s:.1f}s  {chunk[:60]}...")

    final = concat_wavs(parts, out_wav) if len(parts) > 1 else _copy_single(parts[0], out_wav)
    manifest = {
        "backend": backend_name,
        "license": backend.license,
        "reference_wav": str(reference_wav) if reference_wav else None,
        "reference_text": ref_text,
        "x_vector_only": x_vector_only,
        "language": language,
        "output": out_wav.name,
        "duration_s": round(final.duration_s, 3),
        "sample_rate": final.sample_rate,
        "chunks": manifest_chunks,
    }
    manifest_path = out_wav.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RunResult(
        wav_path=out_wav,
        manifest_path=manifest_path,
        chunks=len(chunks),
        duration_s=final.duration_s,
    )


def _copy_single(part: Path, out_wav: Path) -> object:
    import shutil

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if part.resolve() != out_wav.resolve():
        shutil.copyfile(part, out_wav)
    return wav_info(out_wav)


__all__ = ["RunResult", "synthesize_script"]
