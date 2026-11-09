# Changelog

All notable changes to voice-generator, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Chatterbox Turbo failed on first render with `TypeError: 'NoneType'
  object is not callable` at `PerthImplicitWatermarker` (seen
  2026-11-09 on the target Windows box, after a successful 4 GB model
  download): `resemble-perth` imports `pkg_resources`, removed in
  setuptools 81+, and perth swallows the resulting ImportError. The
  `chatterbox` extra now pins `setuptools<81`; existing venvs need
  `pip install "setuptools<81"` once. Documented in SETUP.md and the
  README troubleshooting table.

### Added
- `SETUP.md` — ordered Windows quickstart (prove the pipeline, record
  the reference, first Chatterbox render, Qwen3 A/B) for first-time
  voice testing, with the README kept as the full reference.
- `recording-script.txt` — ready-made text to read aloud for the
  reference clip (~50 s; trim to the cleanest 10–30 s). It doubles as
  the transcript (`refs\me.txt`) for Qwen3-TTS / CosyVoice.

### Changed
- Repository renamed from `ss-voiceGenerator` to `voice-generator`
  (approved 2026-10-31). Display names only — directory, README title,
  distribution name, and copyright line. The `voice_generator` package,
  `voice-gen` CLI, and all behaviour are unchanged.

## [0.3.1] — 2026-10-31 (staff review)

### Fixed
- Manifest now records `backend_settings` (variant, model, mode) and a
  `manifest_version` (1) — previously it named the backend but dropped
  the settings, so a render was not reproducible from its manifest.
- `--max-chars 0` no longer hangs the chunker (validated, raises
  `ValueError`).
- Chatterbox/Qwen3 device default is now `auto` (CUDA when available,
  else CPU) instead of hardcoded `cuda`/`cuda:0`, which failed
  cryptically on CPU-only machines.
- Qwen3 rejects an invalid `dtype` in the constructor instead of a bare
  `KeyError` inside model load.
- CLI converts expected failures (missing files, bad input, backend not
  installed, malformed WAV) to a one-line `Error:` + exit 1 instead of a
  traceback.
- Manifest is written atomically (temp file + rename) so a crash cannot
  leave a half-written provenance file.

### Changed
- Pipeline refactored around a `SynthConfig` dataclass; per-backend
  constructor arguments now come from one `backend_kwargs()` method
  instead of an if/elif chain in the render loop.
- `write_pcm16_wav` packs samples in bulk instead of one `struct.pack`
  per sample (long renders were needlessly slow); the mock backend now
  reuses the shared writer instead of duplicating WAV code.
- Version is now single-sourced from `voice_generator.__version__`
  (pyproject `dynamic`); ruff lint rules expanded (E/F/W/I/UP/B) and all
  long lines fixed.
- CosyVoice backend initialises `_model` in `__init__` (was created
  implicitly via `getattr` in `_load`); `assert`-based user-input checks
  in Qwen3/CosyVoice replaced with explicit `ValueError`s.
- `tools/ab_compare.py` docstring corrected (three backends, not two)
  and gained a `--language` passthrough.

### Added
- `LICENSE` (MIT), `CHANGELOG.md`, `REVIEW.md`, CI workflow
  (`.github/workflows/ci.yml`: pytest + ruff on Python 3.10–3.12).
- 10 new tests in `tests/test_review.py` (27 total), each pinning a fix
  above.

## [0.3.0] — 2026-10-22
- CosyVoice 3 backend (local repo clone in its Conda env), A/B tool
  extended to all three backends, shared PCM-16 writer.

## [0.2.0] — 2026-10-22
- Qwen3-TTS backend in its own venv, `--ref-text` support, A/B compare
  tool.

## [0.1.0] — 2026-10-22
- Core + Chatterbox + mock backends, CLI, chunking, manifest, tests.
