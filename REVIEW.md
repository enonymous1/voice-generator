# Staff review — voice-generator (2026-10-31)

Reviewer posture: staff engineer + open-source maintainer, reviewing the
v0.3.0 tree as it stood before this pass. Scope: all of `src/`,
`tests/`, `tools/`, `pyproject.toml`, and the README. The real backends
could not be executed here (no GPU, no model weights), so backend API
correctness is reviewed against the upstream contracts the code itself
documents, not by running renders — the same caveat the README carries.

## Verdict

**Ship it, after this pass.** The architecture is sound — the
torch-free core, lazy backend registry, and WAV+manifest contract are
the right shape for this problem, and the fake-module tests are genuinely
good. What v0.3.0 lacked was not design but engineering hygiene: the
manifest dropped the settings it claimed to record, the CLI had no error
contract, a handful of real bugs hid in edge cases, and the repo was
missing the files (LICENSE, CHANGELOG, CI) that make a project
maintainable by anyone else. All of that is fixed in v0.3.1; the
remaining items are documented below as deferred work with reasons,
not as open defects.

**Headline numbers:** 17 → 27 tests, ruff clean under expanded rules
(E/F/W/I/UP/B, previously default rules only), wheel builds as 0.3.1
from a single version source, README rewritten into the required
Description / Prerequisites / Installation / Local Development /
Deployment structure.

## 1. Best practices & code quality — findings

| # | Severity | Finding | Fix in v0.3.1 |
|---|---|---|---|
| F1 | High | **Manifest was not reproducible.** `pipeline.py` recorded the backend name but discarded `SynthesisResult.settings` (variant, model size, mode) — the manifest claimed to be provenance while omitting the knobs that change the output. | Manifest now carries `backend_settings` (first chunk's settings, constant across a run) + `manifest_version: 1`. Test: `test_manifest_records_backend_settings_and_version`. |
| F2 | Medium | **CLI had no error contract.** Any expected failure (missing script, missing reference, backend not installed, malformed WAV) dumped a full traceback. | `cli.py` catches `EXPECTED_ERRORS` (FileNotFoundError, ValueError, RuntimeError, `wave.Error`) → one-line `Error:` + exit 1. Unexpected exceptions still propagate. Tests: CLI clean-error tests. |
| F3 | Medium | **`--max-chars 0` hung forever.** The hard-split loop in `chunk_script` sets `cut = max_chars = 0` and never shrinks `rest`. A CLI user could hit this by typo. | `chunk_script` validates `max_chars >= 1` (ValueError). Test pinned. |
| F4 | Medium | **Hardcoded CUDA devices.** Chatterbox defaulted to `device="cuda"`, Qwen3 to `"cuda:0"` — a CPU-only machine died inside `from_pretrained` with a confusing error. | Both default to `"auto"`, resolved at load time via `base.resolve_device()` (CUDA if torch reports it, else CPU). Explicit values still honoured. |
| F5 | Medium | **Bare `KeyError` on bad Qwen3 dtype.** `dtype` was validated only by dict lookup inside `_load`. | Constructor validates against the three allowed dtypes with a clear `ValueError`. Test pinned. |
| F6 | Medium | **Non-atomic manifest writes.** A crash mid-write left a half-written manifest, which Stage 2 would read as provenance. | `_write_atomic()` (temp file + `os.replace`). |
| F7 | Low | **`assert` used for user-input validation** (`qwen_backend._prompt`, `cosy_backend._prompt_text`). Asserts vanish under `python -O`. | Replaced with explicit `ValueError`s (via `require_reference` where applicable). |
| F8 | Low | **CosyVoice `_model` created implicitly** via `getattr` in `_load` — hidden state, confusing readers, masking real `AttributeError`s. | Initialised to `None` in `__init__`; `_load` is now straightforward. |
| F9 | Low | **Per-sample `struct.pack` loop** in `write_pcm16_wav` — the slowest code in the pipeline for long renders (millions of samples). | Bulk pack: one `struct.pack` over a clipped list. Same output bytes, ~order-of-magnitude faster. Clipping test pinned. |
| F10 | Low | **Mock backend duplicated WAV-writing code** instead of using the shared writer it was supposed to exercise. | Mock now builds samples and calls `write_pcm16_wav` — the mock path now tests the production joining path. |
| F11 | Low | **`_doctor` had mid-function imports and three drifting try/except probes** for the same question ("is this engine importable here?"). | Top-level imports; one `_probe()` helper driven by the new `BACKEND_IMPORTS` map; `COSYVOICE_HOME` always displayed. |
| F12 | Low | **Version lived in two places** (`pyproject.toml` + `__init__.__version__`), in sync only by hand. | pyproject `dynamic = ["version"]` from `voice_generator.__version__`; wheel verified as 0.3.1. |
| F13 | Low | **Ruff ran default rules only** — E501, import order, pyupgrade, bugbear findings were invisible, and the tree had 16 long lines. | `[tool.ruff.lint] select = ["E","F","W","I","UP","B"]`; all lines fixed. `ruff format` deliberately *not* enforced (would reflow every file in one noisy diff — a maintainer call, revisit after v0.4). |
| F14 | Low | **No LICENSE file** despite `license = "MIT"` in pyproject and a README claim of MIT. | `LICENSE` (MIT) added. |
| F15 | Low | **`ab_compare.py` docstring said "two backends"**; it drives three, and had no `--language` passthrough (all sides rendered in the CLI default regardless of the script's language). | Docstring corrected; `--language` added and passed to every side. |

**What was already good (do not "fix"):** the lazy registry (core
imports no torch), the fake-module test strategy, the per-backend venv
isolation, the stdlib-only WAV path, and the module-level "why"
docstrings on the backends (Qwen3 prompt caching, CosyVoice v3 prompt
format) — those docstrings are the reason this review could verify
upstream contracts without running the models.

**Anti-patterns checked and cleared:** no global mutable state shared
across backends (Qwen3's prompt cache is per-instance); no shell
invocation (the A/B tool uses `subprocess` with a list, no `shell=True`);
no secrets in the tree (checked: no tokens, no hardcoded paths outside
upstream defaults); no `print` debugging left in library code — the
pipeline's progress prints are CLI-facing output, acceptable for a CLI
tool, but see §3 for the logging proposal.

## 2. Commenting & in-code documentation

Policy applied: **docstrings explain the why and the contract, not the
what.** A docstring that restates the code ("splits text into chunks")
was rewritten or extended with the reason, the edge case, or the
invariant a future reader needs.

- `pipeline.synthesize_script` had **no docstring at all** despite being
  the public entry point — added, including its raised-exception
  contract.
- `base.py`: `SynthesisRequest` / `SynthesisResult` / `BaseBackend`
  documented, including why `settings` exists (manifest reproducibility
  — the F1 lesson, written down so it is not re-broken).
- `base.require_reference` / `base.resolve_device`: new helpers, each
  documented with the failure mode they prevent (model download before
  a user error; cryptic CUDA failure on CPU-only machines).
- `text.chunk_script`: documents the `max_chars` validation and the
  known abbreviation limitation (see §3) instead of silently claiming
  sentence-perfect splitting.
- `audio.write_pcm16_wav`: documents clipping rationale (overshoot must
  not wrap into clicks) and the bulk-pack change.
- `pipeline._write_atomic`: documents why atomicity matters here
  (Stage 2 trusts this file).
- `cosy_backend._load`: documents the `sys.path` mutation as a known
  limitation for library embedding (see §3).
- `backends/__init__.BACKEND_IMPORTS`: documents why registry keys and
  import names differ (`qwen3` vs `qwen_tts`).
- CLI module docstring now states the error contract explicitly, so a
  contributor adding a new failure mode knows which bucket it belongs in.

Deliberately **not** added: inline comments restating obvious code,
docstrings on trivial private helpers, or type-comment noise where the
annotations already say it.

## 3. Future-proofing & architecture

**Implemented in v0.3.1 (modularisation, low risk):**

- `SynthConfig` dataclass + `backend_kwargs()`: the 19-parameter
  `synthesize_script` signature is preserved for callers/tests, but the
  run now passes one config object and the per-backend constructor
  mapping lives in exactly one method. Adding backend #4 = extend that
  method + the CLI + registry, not the render loop.
- `BACKEND_IMPORTS` probe map: `doctor` derives its probes from data,
  so a new backend cannot be registered but forgotten by `doctor`.
- Manifest schema versioning (`manifest_version`, with missing = v0):
  Stage 2 can branch on schema instead of sniffing keys.
- CI (`.github/workflows/ci.yml`): pytest + ruff on Ubuntu + Windows,
  Python 3.10–3.12. It intentionally does **not** gate on real renders —
  those need the target GPU box and belong to the deployment gate in the
  README, not to CI.

**Recommended, deferred with reasons (not implemented — each is a
design decision, not a drive-by refactor):**

1. **Backend metadata registry.** Today the registry maps names to
   classes, and license / min-reference / transcript-requirement /
   install-hint live in each backend class and in scattered CLI/help
   text. The next step is a `BackendInfo` dataclass on the registry
   (license, `needs_transcript`, `min_reference_s`, install hint) that
   `doctor`, `check-ref`, and the CLI derive from — so `check-ref` can
   warn per-backend ("Chatterbox wants ~10 s, you gave 3 s") instead of
   one generic ideal. Deferred: it changes the backend constructor
   surface and deserves its own PR with tests.
2. **Plugin entry points.** Out-of-tree backends currently require
   editing `backends/__init__.py`. A `voice_generator.backends`
   entry-point group would let a third-party backend package register
   itself. Deferred: no third-party backend exists yet; adding the
   mechanism before the first consumer is speculative generality.
3. **Logging instead of prints.** The pipeline prints progress and
   warnings to stdout, which is fine for a CLI but unusable as a
   library (Stage 2 embedding). The migration is: `logging` with a
   CLI handler that preserves today's output format exactly. Deferred:
   output-format compatibility with the A/B tool's parsing (it reads
   manifests, not stdout, so this is safe — but it is a behaviour
   change that should ship with its own test).
4. **Model manager abstraction.** Downloads, cache verification, and
   version pinning are currently implicit in each engine's
   `from_pretrained` call. A thin `ModelStore` (resolve → download →
   verify → path) would make offline-first operation explicit and
   testable. Deferred: each engine's download mechanism differs enough
   that a premature abstraction would leak.
5. **CosyVoice `sys.path` mutation.** The backend inserts the clone
   root at `sys.path[0]` for the process lifetime. Fine for a CLI run;
   a library embedding two CosyVoice versions (or a name collision with
   another package) would break. The fix is importlib-based loading
   from the clone path. Deferred: no embedding use case yet.
6. **Sentence tokenization.** `chunk_script` splits after abbreviations
   ("Dr. Smith") and decimals. A proper tokenizer (e.g. a small
   abbreviation list, or an optional `nltk`/`spaCy` extra) would fix
   it, but adds a dependency to a stdlib-only core. Deferred:
   document the limitation (done, in `text.py` + README-adjacent
   troubleshooting) until a real script hits it.
7. **Streaming / chunked output and resume.** Today a failed chunk
   means re-running the script (chunks are kept, but nothing resumes
   from them). A `--resume` flag keyed on the `.parts` folder + manifest
   is the natural v0.5 feature. Deferred: needs a manifest state
   machine, not a flag.

**Tightly coupled components, honestly assessed:** the pipeline and
the manifest schema are coupled by design (the manifest *is* the
pipeline's output contract) — that coupling is the point, and schema
versioning is the mitigation. The CLI and the pipeline are coupled
through `SynthConfig` now, which is the right seam: the CLI owns
argument parsing, the pipeline owns rendering, neither imports the
other's internals.

## 4. README modernization — audit

The v0.3.0 README was already instructive (571 lines) but structured as
a narrative, not a maintainer README: no Description / Prerequisites /
Installation / Local Development / Deployment skeleton, no LICENSE /
CHANGELOG / CI to point at, and several claims that had drifted from
the code (CLI `--attn` default described as `sdpa` while the parser
default was `None` — behaviourally identical via the backend default,
but the README and the parser disagreed on paper).

The rewritten README (v0.3.1) keeps everything that was correct —
per-backend installs, reference-clip guide, troubleshooting, license
exclusions — and restructures it into: **Description** (with the
architecture diagram and backend table), **Prerequisites** (hardware
table + per-backend Python split), **Installation** (core smoke test,
then one chapter per backend), **Local Development** (tests, how the
fake-module strategy works, backend-authoring guide, layout, honest
status), **Deployment** (the honest version for a local CLI: target-box
gate, packaging, what CI does and does not cover), plus **Visual
assets** (see §5). Every command in it was re-checked against the
v0.3.1 CLI during this review; the `--attn` discrepancy is resolved
(parser default is now literally `sdpa`).

## 5. Visual asset strategy

Full specification (filenames, content, capture method, exact placement)
lives in the README's **Visual assets** section — summary here:

1. **Architecture diagram — shipped.** `docs/assets/architecture.svg`,
   embedded at the top of Description. SVG, not a screenshot, because
   the pipeline is stable while renders are not.
2. **Terminal quickstart GIF — to capture.** Real `doctor` + mock
   synth run; placed after the core smoke-test block in Installation.
   Proves the 5-minute claim in ~20 seconds.
3. **A/B waveform figure — to capture after first GPU renders.**
   Three aligned waveform strips for the same sentence + the comparison
   table, generated from real A/B output WAVs; placed at the end of the
   A/B compare section.
4. **Reference-clip figure — to capture, lowest priority.** Good vs
   bad waveform strips; placed at the end of the reference-clip guide.

Standing rule, enforced in the README: **no faked visuals.** A missing
image is honest; a fabricated render in a voice-cloning repo is not.

## 6. What was not changed, and why

- **No backend API changes** beyond device/dtype defaults: the upstream
  call shapes (Chatterbox `generate`, Qwen3 prompt caching,
  CosyVoice segment flattening) were verified against upstream docs in
  v0.1–v0.3 and cannot be re-verified without a GPU here; changing
  them on speculation would be worse than leaving them.
- **No new runtime dependencies.** The core stays stdlib-only; every
  fix above is stdlib or test-only.
- **No `ruff format` enforcement, no mypy gate.** Format would reflow
  the whole tree in one diff; mypy would need a `py.typed` + stub
  strategy for the untyped upstream packages. Both are reasonable next
  steps, neither belongs in a review pass that also changes behaviour.
- **No repository URL in pyproject.** The repo has not been created
  yet; inventing a URL would be worse than omitting one. Add
  `[project.urls]` when the remote exists.

## 7. Verification

```
ruff check src tests tools   # clean (expanded rule set)
pytest -q                    # 27 passed (was 17)
python -m build              # wheel ss_voice_generator-0.3.1, Version: 0.3.1
```

All verification on a GPU-less Linux box with fake engine modules;
real-backend renders remain the manual gate on the RTX 3060, per the
README's Deployment section.

## 8. Follow-up, 2026-11-09 (first real GPU runs)

Two real renders on the RTX 3060, and the review's blind spot was the
same in both: the synth path trusted the venv instead of verifying it.

- **Perth / setuptools pin** and the **Chatterbox float-WAV fix** are
  in the Changelog's Unreleased section (commits `b67393c`,
  `8d5f1e7`); both were found by running, not by reading.
- **CPU-only torch** (the expensive one): Qwen3 sat with no progress
  for 12+ minutes after model load, twice, then again with
  `--x-vector-only` and a 10-second clip. The clip length and the
  missing SoX were both chased first and both ruled out. A timestamped
  upstream repro (`tools/diag_qwen_prompt.py`, added here) showed
  `cuda available: False` — the venv's torch was a CPU-only build, so
  everything had been running on CPU all along. Fix: `resolve_device`
  now warns on stderr when torch is present but CUDA is not, with the
  CUDA-torch install command; two tests pin it (fake torch module,
  warning present / absent). Docs now carry the explicit
  `torch.cuda.is_available()` check after each backend venv install,
  and the "hangs for minutes" rows distinguish a download from this
  stall.
- **SoX**: `choco install sox` no longer works (package gone from the
  community feed); docs now point at the SourceForge zip, and at the
  VS Code restart needed for its terminals to inherit a PATH change.
- **`.gitignore` gaps**: `refs/` and `.venv-*/` were not ignored.
  Both added.
- **Not changed**: no version bump — these ride with the 0.3.1 push
  (the Unreleased section was already holding the Perth and PCM
  fixes); no backend call shapes touched; CosyVoice trial paused by
  the user (Conda/pynini friction) and not re-litigated here.
