"""Reference-clip intake for the setup page.

The user records in the browser (MediaRecorder → webm) or uploads a
file, and this module turns it into a validated ``refs/<name>.wav`` plus
``refs/<name>.txt`` — the transcript is the recording script the setup
page displayed, so Qwen3 / CosyVoice / VoxCPM-ultimate get their exact
transcript without the user retyping it.

Rules: the name is restricted to ``[A-Za-z0-9_-]`` (it becomes a file
name); an existing name is refused, never silently overwritten. A
non-WAV upload is converted with ffmpeg to 16 kHz mono PCM-16; a WAV
that the stdlib reader rejects (e.g. 32-bit float, which some recorders
produce) gets the same ffmpeg conversion as a fallback.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from voice_generator.audio import check_reference, wav_info

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB — a 60 s clip is a few MB


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))


def _ffmpeg_convert(src: Path, dst: Path) -> None:
    """Convert ``src`` to 16 kHz mono PCM-16 WAV at ``dst``.

    Raises ValueError with ffmpeg's own error tail when it fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError(
            "this recording is not a WAV — ffmpeg is needed to convert it, "
            "and ffmpeg was not found on PATH (run `voice-gen doctor` to check)"
        )
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000", str(dst)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg could not convert this recording: {proc.stderr.strip()[-500:]}")


def save_reference(
    *,
    upload_path: Path,
    upload_suffix: str,
    name: str,
    refs_dir: Path,
    transcript: str,
) -> dict[str, object]:
    """Validate an uploaded recording and store it as ``refs/<name>.wav``.

    ``upload_path`` is a temp file holding the upload; it is not modified
    and is left for the caller to delete. Returns the wav path, its info,
    and any validation warnings. Raises ValueError with a human-readable
    message on any failure, and leaves no partial files behind.
    """
    if not valid_name(name):
        raise ValueError(
            f"bad reference name {name!r} — use letters, digits, '-' and '_' only"
        )
    refs_dir.mkdir(parents=True, exist_ok=True)
    wav_path = refs_dir / f"{name}.wav"
    if wav_path.exists():
        raise ValueError(
            f"a reference named {name!r} already exists — pick another name, "
            "or delete the old one first"
        )

    with tempfile.TemporaryDirectory(prefix="voice-gen-ref-") as tmp:
        tmp = Path(tmp)
        candidate = tmp / f"upload{upload_suffix.lower()}"
        shutil.copyfile(upload_path, candidate)
        if candidate.suffix != ".wav":
            converted = tmp / "converted.wav"
            _ffmpeg_convert(candidate, converted)
            candidate = converted
        try:
            info = wav_info(candidate)
        except wave.Error:
            # A WAV the stdlib reader rejects (32-bit float from some
            # recorders): convert it instead of failing on a format tag.
            converted = tmp / "converted.wav"
            _ffmpeg_convert(candidate, converted)
            candidate = converted
            info = wav_info(candidate)  # raises wave.Error if still unreadable
        warnings = check_reference(candidate)
        shutil.copyfile(candidate, wav_path)
    (refs_dir / f"{name}.txt").write_text(transcript, encoding="utf-8")
    return {"wav": wav_path, "info": info, "warnings": warnings}
