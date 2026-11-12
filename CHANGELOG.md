# Changelog

All notable changes to voice-generator, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.2] — 2026-11-12 (Setup script card, equal height)

### Changed
- The "Choose a script" section is a card now, and the script display
  is one equal height for all three scripts: the blocks share a single
  grid cell, so the card is as tall as the tallest script and swapping
  scripts never moves the rest of the page. Hidden blocks stay in
  layout (`visibility`, not `display`), so the height re-measures
  itself on resize or font change with no JavaScript measuring.

### Fixed
- Two tests lost their identities in the v0.5.1 edits (single-line
  edit anchors folded `test_script_endpoint_with_id` into the card
  test and `test_upload_duplicate_name_refused` into the fallback
  test — their assertions still ran, but under the wrong names, so the
  suite reported 65 instead of 67). Both are restored as their own
  tests; the suite is 67.

## [0.5.1] — 2026-11-12 (recording-script options)

### Added
- The Setup page offers three recording scripts instead of one
  (`src/voice_generator/web/scripts.py`): the project introduction
  (default), an emotional-range paragraph (questions, exclamation,
  pauses, alphanumeric data, lower register), and Harvard sentences
  (IEEE List 5 — verified against the Columbia/McGill copies; the
  quick-brown-fox line is labeled a pangram, not a Harvard sentence).
  Each script shows its "why this works" notes, and the clip-length
  guidance: backends clone from ~10 s (Chatterbox/Qwen3/CosyVoice) or
  up to 30 s (VoxCPM2), so a longer recording is material to trim from,
  and trimming a clip means updating its matching `.txt` transcript.
  The chosen script's text is stored as the reference transcript;
  `/script?id=range` (or `harvard`) serves any script's text.

## [0.5.0] — 2026-11-12 (local web UI)

### Added
- Local Flask web UI (`voice-gen serve`, `src/voice_generator/web/`):
  three pages on localhost only — Synthesize (paste text, pick a
  backend + reference, render in the background with a live log, listen
  and download), Setup (the recording script, record in the browser or
  upload a clip; ffmpeg converts non-WAV audio, the clip is validated
  like `check-ref`, and the script text is stored as the transcript),
  and History (past renders with their manifests). Renders drive the
  CLI as a subprocess, so the UI can never drift from the CLI's render
  path; the server never imports torch. New `[web]` extra (`flask`);
  CI installs `.[dev,web]`. The UI follows the system light/dark theme.
- GitHub Pages site (`site/` + `.github/workflows/pages.yml`): landing
  page, the fast-open-models research brief, and a listen/compare page
  whose audio slots fill in from `samples.json` after the A/B bench.
  The full research report is in the repo at
  `docs/research/fast-open-voice-cloning-tts-2026-11-10.md`. One-time
  setup after the first push: Settings → Pages → Source: GitHub
  Actions; the Pages deploy itself is unrun until then.
- The Pages site follows the system light/dark theme
  (`prefers-color-scheme`, with
  `<meta name="color-scheme" content="light dark">`).

## [0.4.0] — 2026-11-10 (VoxCPM2 backend)

### Added
- VoxCPM2 backend (`--backend voxcpm`, OpenBMB, Apache-2.0 code +
  weights): `pip install -e ".[voxcpm]"` in its own Python 3.12 venv
  (upstream: Python 3.10–<3.13, PyTorch >= 2.5, CUDA >= 12). Clones
  from a 5–30 s reference clip with no transcript in the default
  controllable mode; `--voxcpm-mode ultimate` passes the clip plus its
  exact transcript for maximum similarity. Output is 48 kHz PCM-16
  via the shared writer. Knobs: `--voxcpm-model`
  (default `openbmb/VoxCPM2`), `--voxcpm-cfg` (default 2.0),
  `--voxcpm-timesteps` (default 10).
- VoxCPM2 torch.compile fallback: the backend loads with
  `optimize=True` (upstream default; the vendor's RTF ~0.30 on a 4090
  was measured with it) and retries once with `optimize=False` when a
  load or a first generate fails with a Triton/compile-shaped error —
  the Windows failure mode the 2026-11-10 model survey flagged.
  `--voxcpm-no-optimize` skips the attempt outright. The effective
  value is recorded in the manifest.
- `tools/ab_compare.py` gains a fourth side (`--voxcpm-python`,
  `--voxcpm-mode`) so the RTX 3060 bench can run all four backends on
  the same script and clip.

### Changed
- Roadmap: v0.4.0 is the VoxCPM2 backend; the Stage 2 handoff moves to
  v0.5.0.

### Not yet verified
- No RTX 3060 speed figure exists for VoxCPM2 anywhere public — the
  vendor's RTF ~0.30 is on a 4090. The 3060 bench (same script, same
  clip, `tools/ab_compare.py`) is the gate before VoxCPM2 becomes the
  recommended fast path. The denoiser is not loaded
  (`load_denoiser=False`); there is no denoise knob in v0.4.0.

## [0.3.2] — 2026-11-09 (first real GPU runs)

### Fixed
- Chatterbox rendered audio and then failed with `Error: unknown format:
  3` (seen 2026-11-09 on the first real GPU render, after the Perth
  fix): the backend wrote its float32 output with `torchaudio.save`,
  which produces a 32-bit float WAV, and this project's stdlib WAV
  reader only reads integer PCM. The backend now writes PCM-16 via the
  shared `write_pcm16_wav` like Qwen3/CosyVoice, and `wav_info` now
  reports a float WAV with the conversion command instead of the bare
  format tag. (Also fixed: the SETUP.md advice to re-render flat Turbo
  output with `--exaggeration` — Turbo ignores that knob; only
  `original` honours it.)
- Chatterbox Turbo failed on first render with `TypeError: 'NoneType'
  object is not callable` at `PerthImplicitWatermarker` (seen
  2026-11-09 on the target Windows box, after a successful 4 GB model
  download): `resemble-perth` imports `pkg_resources`, removed in
  setuptools 81+, and perth swallows the resulting ImportError. The
  `chatterbox` extra now pins `setuptools<81`; existing venvs need
  `pip install "setuptools<81"` once. Documented in SETUP.md and the
  README troubleshooting table.
- A Qwen3 render sat with no progress for 12+ minutes after model load
  (seen 2026-11-09, twice, then again with `--x-vector-only` and a
  10-second clip): the venv's torch was a CPU-only build, so prompt
  creation ground on CPU — and nothing said the GPU was not in use.
  `resolve_device("auto")` now warns on stderr when torch is present
  but CUDA is not, naming the cause and the CUDA-torch install command.
  The README/SETUP.md "hangs for minutes" rows now distinguish a
  download (progress bars moving) from this stall (no progress after
  model load). `voice-gen doctor` already reported CUDA status; the
  synth path was the silent one.
- `.gitignore` did not cover `refs/` (voice clips are personal data)
  or per-backend venvs (`.venv-*/`) — only the single `.venv/` was
  ignored. Both added, so a reference clip or a second venv can never
  be committed by accident.
- SoX install docs were unsourced: `choco install sox` no longer works
  (the package is gone from the Chocolatey community feed, verified
  2026-11-09). README and SETUP.md now point at the SourceForge
  `sox-14.4.2-win32.zip`, the PATH step, and the VS Code restart
  needed for its terminals to see a PATH change.

### Added
- `tools/diag_qwen_prompt.py` — minimal upstream repro (direct
  qwen-tts calls with timestamps, no wrapper) that isolated the
  2026-11-09 stall to a CPU-only torch build. Run it in the Qwen3
  venv from the project root.
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
