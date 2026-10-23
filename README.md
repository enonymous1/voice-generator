# ss-voiceGenerator

Local, fully FOSS **voice cloning from a text script** — Stage 1 of the AI Digital
Twin pipeline. Give it a short recording of a voice and a text script; it renders
the script in that voice to a WAV file, entirely on your own machine.

**Zero APIs, zero cloud.** No ElevenLabs, no HeyGen, no OpenAI. The target
hardware is a Windows box with an **RTX 3060 12 GB / 64 GB RAM**; everything here
was selected and sized for that machine.

Stage 2 (the audio-driven talking head that will turn this WAV into a
lip-synced video) is a separate, later project. This project's output contract
to Stage 2 is: **one WAV file + one `.manifest.json`** describing exactly how
that WAV was made.

---

## Contents

- [What it does (and doesn't)](#what-it-does-and-doesnt)
- [System requirements](#system-requirements)
- [Design decisions](#design-decisions)
- [The backends](#the-backends)
- [Quick start: 5-minute smoke test (no GPU, no models)](#quick-start-5-minute-smoke-test-no-gpu-no-models)
- [Install: Chatterbox (recommended first)](#install-chatterbox-recommended-first)
- [Install: Qwen3-TTS](#install-qwen3-tts)
- [Install: CosyVoice 3 (advanced)](#install-cosyvoice-3-advanced)
- [Recording a good reference clip](#recording-a-good-reference-clip)
- [Usage](#usage)
- [The manifest (output contract)](#the-manifest-output-contract)
- [A/B compare: hear the backends side by side](#ab-compare-hear-the-backends-side-by-side)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [CLI reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Performance and VRAM notes](#performance-and-vram-notes)
- [Known limitations / honest status](#known-limitations--honest-status)
- [Licenses](#licenses)
- [Development](#development)
- [Roadmap](#roadmap)

---

## What it does (and doesn't)

**Does**

- Clones a voice from a short reference recording (zero-shot — no training,
  no fine-tuning) and reads any text script in that voice.
- Runs three interchangeable, fully local TTS engines behind one CLI:
  **Chatterbox** (Resemble AI), **Qwen3-TTS** (Alibaba Qwen), and
  **CosyVoice 3** (FunAudioLLM), plus a `mock` engine for testing.
- Splits long scripts into sentence-sized chunks automatically, renders them
  one at a time, and joins them into a single seamless WAV.
- Writes a provenance manifest next to every render, so any output can be
  reproduced or traced back to its exact engine, reference clip, and settings.
- Validates reference clips before you waste a render on a bad one.

**Doesn't (yet)**

- No real-time / streaming synthesis — this is offline script rendering.
- No voice *training* or fine-tuning — zero-shot cloning only.
- No video — that's Stage 2.
- No GUI — command line only (deliberately: it's scriptable and testable).

---

## System requirements

| | Minimum | Recommended (target) |
|---|---|---|
| OS | Windows 10/11 64-bit (Linux/macOS work for the core + mock) | Windows 11 |
| GPU | Nvidia with 8 GB VRAM (small models only) | **RTX 3060 12 GB** or better |
| System RAM | 16 GB | 64 GB |
| Disk | ~20 GB free (envs + model weights) | NVMe SSD, 40 GB+ free |
| Nvidia driver | Recent enough for CUDA 12.x PyTorch wheels | Latest Game Ready / Studio |

Software prerequisites:

- **Git** (to clone this project, and the CosyVoice repo if you use it).
- **Python** — and note that *each backend wants its own*: Chatterbox uses
  3.11, Qwen3-TTS upstream recommends 3.12, CosyVoice uses Conda with 3.10.
  The `py` launcher on Windows lets all three coexist; that's by design.
- **Conda / Miniconda** — only for the CosyVoice backend.
- **ffmpeg** on your PATH — not needed by Stage 1 itself, but Stage 2 will
  require it; `voice-gen doctor` checks for it now so you're ready.
- **sox** — only for CosyVoice (upstream requirement; on Windows install the
  sox binary and put it on PATH). This is the main reason CosyVoice is the
  "advanced" install.

You do **not** need a CUDA toolkit install: the PyTorch wheels each backend
pulls in bring their own CUDA runtime.

---

## Design decisions

These came out of the 2026-10-22 FOSS digital-twin pipeline research, and
they're the reason the project is shaped the way it is:

- **One interface, interchangeable engines.** The CLI, the script chunking,
  the WAV joining, and the manifest know *nothing* about any specific model.
  A backend is one small class implementing `synthesize()`. The TTS field
  moves monthly — when a better model appears, adding it is an afternoon,
  not a rewrite.
- **Backends are isolated in separate environments.** Chatterbox, Qwen3-TTS,
  and CosyVoice pin conflicting torch / CUDA stacks. They are *never*
  installed into the same environment. The core package itself has **no
  torch dependency** and imports every backend lazily, so the core (and
  the whole test suite) runs on a machine with no GPU at all.
- **Sequential, not concurrent.** On 12 GB of VRAM you load one model,
  render, unload. This project only ever holds the TTS model in memory —
  Stage 2 runs afterwards, in its own process, with the full card free.
- **Licenses are a feature, not an afterthought.** Every engine here is
  MIT or Apache-2.0, including its weights. Models with non-commercial
  weights (Fish Speech, F5-TTS, Spark-TTS), a custom license with revenue
  thresholds (IndexTTS-2), or no FOSS release at all ("Breeze TTS 2") were
  researched and deliberately **excluded** — see [Licenses](#licenses).

---

## The backends

| Backend | Status | License | Clone reference | VRAM (community figures) | Character |
|---|---|---|---|---|---|
| `mock` | ✅ working | — (tone, not speech) | — | 0 | Plumbing/timing tests |
| `chatterbox` | ✅ wired v0.1 | MIT — outputs carry a PerTh watermark | ~10 s, no transcript | ~1.5–6 GB | Easiest install, most Windows-friendly |
| `qwen3` | ✅ wired v0.2 | Apache-2.0 | ~3 s **+ exact transcript** | ~4–8 GB | Small, fast, 10 languages, 97 ms-class streaming design |
| `cosyvoice` | ✅ wired v0.3 | Apache-2.0 | ~3 s **+ exact transcript** | ~8–10 GB | Best multilingual/dialect range, hardest install |

**Which should you use?**

- **Start with Chatterbox.** `pip install`, no transcript bookkeeping, works.
- **Try Qwen3-TTS next** if you want a smaller/faster Apache-2.0 model, or
  the A/B comparison sounds fun (it is — same script, your voice, two engines).
- **CosyVoice 3** is worth it for multilingual work, Chinese dialects, or
  instruct-style control — but budget real setup time on Windows.

Chatterbox variants (`--variant`): `turbo` (default — English, fast, supports
paralinguistic tags like `[laugh]`, `[sigh]`), `original` (English, strongest
CFG / exaggeration control), `multilingual` (23+ languages; pass
`--language es`, `fr`, etc.).

---

## Quick start: 5-minute smoke test (no GPU, no models)

This verifies the whole flow — CLI, chunking, WAV output, manifest — before
you install a single model. It uses the `mock` backend, which writes a tone
WAV of realistic duration instead of speech.

```bat
cd ss-voiceGenerator
py -3.11 -m venv .venv-core
.venv-core\Scripts\activate
pip install -e .
voice-gen doctor
voice-gen backends
voice-gen synth --backend mock --text "Hello world. This is a plumbing test." --out output\smoke.wav
```

You should see `output\smoke.wav` and `output\smoke.manifest.json`. If those
appear, the pipeline works and only the model installs remain.

> Linux/macOS: same commands, with `python3 -m venv` and
> `.venv-core/bin/activate`.

---

## Install: Chatterbox (recommended first)

```bat
cd ss-voiceGenerator
py -3.11 -m venv .venv-chatterbox
.venv-chatterbox\Scripts\activate
pip install -e ".[chatterbox]"
voice-gen doctor
```

`doctor` should report torch with CUDA available and your GPU's VRAM.
First synthesis downloads the Chatterbox weights from Hugging Face
(one-time, a few GB) — after that everything is offline.

Pinokio also has a community "ChatterBox-Multilingual" one-click script
(search-index evidence only; the manual venv above is the verified path —
if you use Pinokio's Chatterbox, you don't need this CLI's Chatterbox venv
for that engine, but the CLI still gives you chunking + manifests).

---

## Install: Qwen3-TTS

Upstream recommends a fresh **Python 3.12** environment for `qwen-tts`.

```bat
py -3.12 -m venv .venv-qwen3
.venv-qwen3\Scripts\activate
pip install -e ".[qwen3]"
voice-gen doctor
```

Qwen3 cloning facts (verified against the upstream README, 2026-10-22):

- Cloning lives in the **Base** checkpoints: `--model-size 1.7b` (default)
  or `0.6b`. The CustomVoice/VoiceDesign checkpoints do other jobs and are
  not used here. First run downloads the weights from Hugging Face.
- **Full cloning needs the reference clip's exact transcript** —
  `--ref-text "..."` or `--ref-text-file refs\me.txt`. Word-for-word,
  including fillers; if the transcript doesn't match the audio, clone
  quality drops. Without a transcript, `--x-vector-only` clones from the
  speaker embedding only (works, noticeably worse).
- The clone prompt is built **once per run** and reused for every chunk.
- Attention defaults to `sdpa` (Windows-friendly). Upstream recommends
  FlashAttention 2; opt in with `--attn flash_attention_2` *only* if you
  have it installed — on Windows most people don't, and `sdpa` is fine.
- Languages are given to the model as full names; the CLI maps its ISO
  codes for you (`en` → `English`, `es` → `Spanish`, …). Supported:
  en, zh, ja, ko, de, fr, ru, pt, es, it.

---

## Install: CosyVoice 3 (advanced)

CosyVoice is **not a pip package**. Upstream's path (verified 2026-10-22):

```bat
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
cd CosyVoice
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"
pip install -e <path-to-ss-voiceGenerator>
```

Notes:

- The `--recursive` clone matters: the backend loads the repo's
  `third_party/Matcha-TTS` submodule onto `sys.path`, exactly like
  upstream's `example.py`. If imports fail with `No module named matcha`,
  run `git submodule update --init --recursive` inside the clone.
- sox must be installed and on PATH.
- Point the CLI at the clone with `--cosyvoice-home C:\path\to\CosyVoice`
  (or set the `COSYVOICE_HOME` environment variable), and at a non-default
  model location with `--cosy-model-dir` / `COSYVOICE_MODEL_DIR`.
- Zero-shot mode needs the reference transcript (same rule as Qwen3).
  CosyVoice has no transcript-free clone mode; `--cosy-mode cross_lingual`
  is the alternative — it clones across languages without a transcript.
- For CosyVoice 3 the backend automatically formats the zero-shot prompt
  the way upstream's official example does:
  `You are a helpful assistant.<|endofprompt|>` + your transcript.
  With `--cosy-version 2` the bare transcript is used. Don't add the
  prefix yourself — if your transcript file already contains
  `<|endofprompt|>`, the backend leaves it untouched.
- On Windows this is the fiddliest install of the three. If it fights you,
  that's expected — Chatterbox and Qwen3 already cover most use cases,
  and WSL2 is a legitimate escape hatch for this backend only.

---

## Recording a good reference clip

Clone quality is mostly decided here, before any model runs.

- **Length:** ~10 s for Chatterbox; ~3–10 s for Qwen3 / CosyVoice.
  Longer is *not* better — 10–30 s of clean speech beats a noisy minute.
- **Content:** natural, expressive, normal speaking pace. Read a paragraph
  you'd actually say. Avoid whispering, shouting, or a "radio voice".
- **Environment:** quiet room, no music, no TV, no echoey bathroom acoustics,
  microphone 15–30 cm away, no clipping (don't peak the meter).
- **One speaker only.** Any second voice, cough track, or ad jingle in the
  clip becomes part of the "voice".
- **Format:** WAV, mono preferred, 16 kHz or higher (24/44.1/48 kHz fine).
  Other formats? Convert first: `ffmpeg -i ref.mp3 -ac 1 -ar 24000 ref.wav`.
- **Transcript (Qwen3 / CosyVoice):** type *exactly* what is said,
  fillers and all, and save it next to the clip (`refs\me.txt` beside
  `refs\me.wav` is the convention this project uses).
- **Check it:** `voice-gen check-ref refs\me.wav` flags clips that are too
  short, stereo, or low sample-rate before you burn a render on them.

---

## Usage

Every command is `voice-gen <subcommand>`. Run these inside the venv of
the backend you're using.

```bat
:: 1. Sanity-check the environment
voice-gen doctor
voice-gen backends

:: 2. Validate your reference clip
voice-gen check-ref refs\me_10s.wav

:: 3. Chatterbox — one line
voice-gen synth --text "Hello, this is my digital twin." --ref refs\me_10s.wav --out output\hello.wav

:: 4. Chatterbox — a full script file (chunked automatically)
voice-gen synth --script script.txt --ref refs\me_10s.wav --out output\ep1.wav

:: 5. Chatterbox — expressive / multilingual
voice-gen synth --text "I can't believe it [laugh] we actually did it." --ref refs\me_10s.wav --out output\laugh.wav --variant turbo --exaggeration 0.7
voice-gen synth --text "Hola, esta es mi gemela digital." --ref refs\me_10s.wav --out output\hola.wav --variant multilingual --language es

:: 6. Qwen3-TTS (activate .venv-qwen3 first)
voice-gen synth --backend qwen3 --script script.txt --ref refs\me_3s.wav --ref-text-file refs\me.txt --out output\ep1_qwen.wav
voice-gen synth --backend qwen3 --text "Smaller and faster." --ref refs\me_3s.wav --ref-text-file refs\me.txt --model-size 0.6b --out output\small.wav
voice-gen synth --backend qwen3 --text "No transcript handy." --ref refs\me_3s.wav --x-vector-only --out output\xvec.wav

:: 7. CosyVoice 3 (activate the cosyvoice Conda env first)
voice-gen synth --backend cosyvoice --script script.txt --ref refs\me.wav --ref-text-file refs\me.txt --cosyvoice-home C:\path\to\CosyVoice --out output\ep1_cosy.wav
voice-gen synth --backend cosyvoice --text "Cross lingual, no transcript." --ref refs\me.wav --cosy-mode cross_lingual --cosyvoice-home C:\path\to\CosyVoice --out output\cross.wav
```

Long scripts are split into sentence-sized chunks (default ≤ 300 chars;
tune with `--max-chars`). Smaller chunks = more consistent voice, more
per-chunk overhead; larger chunks = faster, but models can drift on long
inputs. 250–350 is the sweet spot for all three engines.

Output per render: the WAV, a `.manifest.json` beside it, and a hidden
`.parts` folder next to the output holding the per-chunk WAVs (kept so you
can re-join or inspect a bad chunk without re-rendering everything).

---

## The manifest (output contract)

Every render writes `<output-name>.manifest.json`. Real example (mock run):

```json
{
  "backend": "mock",
  "license": "MIT (project code; generates a tone, not speech)",
  "reference_wav": "refs/me.wav",
  "reference_text": null,
  "x_vector_only": false,
  "language": "en",
  "output": "ep1.wav",
  "duration_s": 5.643,
  "sample_rate": 24000,
  "chunks": [
    {"index": 1, "text": "Hello world…", "wav": "chunk_001.wav", "duration_s": 5.643}
  ]
}
```

This file is the Stage 2 contract: a talking-head render will read the WAV
path, duration, and provenance from here instead of guessing. If a render
ever sounds wrong, the manifest tells you which engine, reference, and
settings produced it — reproduce first, tweak second.

---

## A/B compare: hear the backends side by side

The backends live in separate environments, so no single interpreter can
import two of them. `tools\ab_compare.py` (stdlib-only — run it with any
Python) drives each environment's interpreter as a subprocess with the
*same* script and reference, then compares the manifests. Any subset of
the three sides may be given.

```bat
python tools\ab_compare.py --script script.txt --ref refs\me.wav --ref-text-file refs\me.txt --chatterbox-python .venv-chatterbox\Scripts\python.exe --qwen-python .venv-qwen3\Scripts\python.exe --cosy-python C:\Users\you\miniconda3\envs\cosyvoice\python.exe --cosyvoice-home C:\path\to\CosyVoice --out-dir output\ab1
```

Produces `chatterbox.wav`, `qwen3.wav`, `cosyvoice.wav`, and
`ab_comparison.json` (per-backend duration, sample rate, and wall-clock
time). A backend that isn't installed doesn't crash the run — its failure
and the install hint are recorded in the JSON. The comparison itself is
your ears; the JSON is diagnostics, not a quality score.

---

## How it works

```
script.txt ──► chunk_script() ──► [chunk 1..N] ──► backend.synthesize() each
                                                          │  (one model loaded
                                                          │   for the whole run)
                                                          ▼
output.wav ◄── concat_wavs() ◄── chunk_001.wav … chunk_N.wav
   │
   └──► output.manifest.json  (backend, license, reference, per-chunk durations)
```

1. **Chunk** — split on sentence boundaries up to `--max-chars`; a single
   over-long sentence is hard-split on commas/spaces.
2. **Validate** — the reference clip is checked (length, channels, rate)
   and warnings are printed before synthesis starts.
3. **Synthesize** — the chosen backend loads its model *once*, renders each
   chunk, and writes PCM-16 WAVs (all backends write the same format via a
   shared stdlib writer, which is what makes joining trivial and exact).
   Backend-specific prep is done once per run too — e.g. Qwen3's voice-clone
   prompt is built once and reused for every chunk.
4. **Join + manifest** — chunks are concatenated with the stdlib `wave`
   module (sample-accurate; formats are verified to match first), and the
   manifest is written.

---

## Project layout

```
ss-voiceGenerator/
├── pyproject.toml            # package metadata; per-backend extras
├── README.md                 # this file
├── src/voice_generator/
│   ├── cli.py                # voice-gen: doctor / backends / check-ref / synth
│   ├── pipeline.py           # script → chunks → backend → WAV + manifest
│   ├── text.py               # sentence chunking
│   ├── audio.py              # WAV info, reference checks, PCM-16 write, concat
│   ├── __main__.py           # python -m voice_generator (used by the A/B tool)
│   └── backends/
│       ├── base.py           # the SynthesisRequest/Result contract
│       ├── __init__.py       # lazy registry (no heavy imports at startup)
│       ├── mock_backend.py   # tone WAV, realistic duration, zero deps
│       ├── chatterbox_backend.py
│       ├── qwen_backend.py
│       └── cosy_backend.py
├── tools/
│   └── ab_compare.py         # cross-environment A/B rendering
└── tests/                    # pytest; fake engine modules, no GPU/downloads
    ├── test_core.py
    ├── test_qwen.py
    └── test_cosy.py
```

---

## CLI reference

`voice-gen doctor` — Python version, torch/CUDA/GPU/VRAM if present, which
backends are importable in this env, ffmpeg on PATH.
`voice-gen backends` — list the backends in this build.
`voice-gen check-ref <ref.wav>` — duration/rate/channels + warnings.

`voice-gen synth` options:

| Option | Default | Applies to | Meaning |
|---|---|---|---|
| `--text` / `--script` | (one required) | all | inline text, or a `.txt` script file |
| `--ref` | — | all real | reference voice clip (.wav) |
| `--out` | (required) | all | output `.wav` path |
| `--backend` | `chatterbox` | all | `mock` / `chatterbox` / `qwen3` / `cosyvoice` |
| `--language` | `en` | all | ISO code; Qwen maps to full names |
| `--max-chars` | `300` | all | chunk size ceiling |
| `--variant` | `turbo` | chatterbox | `original` / `turbo` / `multilingual` |
| `--exaggeration` | `0.5` | chatterbox | emotion intensity 0–1+ |
| `--cfg` | `0.5` | chatterbox | CFG weight (adherence vs freedom) |
| `--ref-text` / `--ref-text-file` | — | qwen3, cosyvoice | exact transcript of `--ref` |
| `--x-vector-only` | off | qwen3 | clone from speaker embedding only |
| `--model-size` | `1.7b` | qwen3 | `1.7b` / `0.6b` Base checkpoint |
| `--attn` | `sdpa` | qwen3 | `sdpa` / `eager` / `flash_attention_2` |
| `--cosyvoice-home` | `$COSYVOICE_HOME` | cosyvoice | path to the CosyVoice clone |
| `--cosy-model-dir` | clone default | cosyvoice | model snapshot directory |
| `--cosy-mode` | `zero_shot` | cosyvoice | `zero_shot` / `cross_lingual` |
| `--cosy-version` | `3` | cosyvoice | prompt format / default model dir |

---

## Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `torch … CUDA not available` in `doctor` | CPU-only torch wheel got installed, or driver too old → reinstall the backend extra in a *fresh* venv; update the Nvidia driver. |
| Backend "not installed in this environment" | You're in the wrong venv → activate that backend's env (each backend has its own; this is by design). |
| First synthesis hangs for minutes | It's downloading model weights from Hugging Face (one-time, several GB). Check your network; later runs are offline. |
| `OutOfMemoryError` / CUDA OOM on 12 GB | Something else is holding VRAM (a game, another model, Stage 2) → close it; for Qwen3 try `--model-size 0.6b`; render stages sequentially, never in parallel. |
| Clone sounds like a different person | Reference clip is noisy/multi-speaker/too short, or the transcript doesn't match the audio word-for-word (Qwen3/CosyVoice) → re-record clean, fix the transcript. |
| Clone sounds right but "flat" | Reference delivery was flat (the model copies it), or Chatterbox `--exaggeration` too low → raise to 0.6–0.8; re-record with more expression. |
| Robotic gaps between sentences | Chunk boundaries — raise `--max-chars` a bit so sentences group more naturally, or accept it (Stage 2 hides a lot). |
| Qwen3: transcript error on synth | Full cloning needs `--ref-text`/`--ref-text-file`, or pass `--x-vector-only`. |
| Qwen3: `flash_attention_2` import error | You opted in without it installed → drop `--attn` (default `sdpa`). |
| CosyVoice: `No module named cosyvoice` | Not in the Conda env, or `--cosyvoice-home` wrong → the backend's error message prints the full setup recipe. |
| CosyVoice: `No module named matcha` | Submodules missing → `git submodule update --init --recursive` in the clone. |
| CosyVoice: sox errors / silent failures | sox not installed or not on PATH → install sox, restart the terminal, verify with `sox --version`. |
| Garbled / wrong-language output | `--language` mismatch with the script's language, or Chatterbox `turbo`/`original` (English-only) used for non-English → use `multilingual` or the right `--language`. |
| WAV plays but is silent / a tone | You rendered with `--backend mock`. It writes a tone *on purpose* — switch to a real backend. |

---

## Performance and VRAM notes

- All figures are per-stage, with nothing else on the GPU. VRAM numbers in
  the backend table are community-reported ranges, not vendor specs —
  treat the first render on your machine as the real measurement
  (`voice-gen doctor` shows total VRAM; watch usage in Task Manager /
  `nvidia-smi` during a render).
- Chunk count is the main speed lever: a 10-minute script at 300 chars is
  roughly 40–60 chunks. Model load time is paid once per run, so batch
  your scripts into one run instead of many small ones.
- Qwen3 `0.6b` is the fast/small option; `1.7b` is the quality option.
  Chatterbox `turbo` is faster than `original`. CosyVoice times vary most
  with install quality (TensorRT paths exist upstream but are out of scope
  here).

---

## Known limitations / honest status

- **v0.3.0.** Verified on a GPU-less Linux box: 17/17 tests (the Qwen3 and
  CosyVoice backends are tested against fake upstream modules that record
  their calls; the mock pipeline is tested end-to-end for real), ruff clean.
- **Not yet verified:** a real render on the RTX 3060. The backend APIs were
  verified against the upstream READMEs/examples on 2026-10-22, but the
  first GPU run is yours — if a backend breaks there, the error messages
  and this README's troubleshooting table are the first stop, and the
  chunk/manifest plumbing around it is already proven.
- Japanese in CosyVoice cross-lingual mode needs katakana-converted text
  (upstream note).
- Chatterbox outputs carry Resemble's imperceptible PerTh watermark —
  disclosed by upstream, not removable, and not a license restriction.

---

## Licenses

- This project: MIT.
- Chatterbox: MIT (code + weights). Outputs watermarked (PerTh).
- Qwen3-TTS: Apache-2.0 (tokenizers + models, per the technical report).
- CosyVoice (2 and 3): Apache-2.0.
- Deliberately excluded after research: Fish Speech (weights
  non-commercial), IndexTTS-2/2.5 (custom bilibili license — commercial
  use only below 100M MAU / RMB 1B revenue; outputs can't be used to
  improve other models), F5-TTS and Spark-TTS (weights CC-BY-NC),
  Higgs Audio (research/non-commercial weights), "Breeze TTS 2"
  (no FOSS release found — only a proprietary hosted API).

Cloning a voice you don't have the rights to is your legal problem, not
the models'. Use your own voice or get permission.

---

## Development

```bat
pip install -e ".[dev]"
pytest        :: 17 tests, no GPU, no downloads, no torch needed
ruff check src tests tools
```

How the tests work: the core is tested for real (chunking, WAV handling,
full mock pipeline). The GPU backends are tested against *fake* upstream
modules injected into `sys.modules` — the fakes record their arguments, so
tests pin the exact API contract (e.g. Qwen3's prompt is built once and
reused; CosyVoice 3's prompt gets the `<|endofprompt|>` format, v2's
doesn't). If upstream changes its API, these tests are where it shows.

**Adding a new backend** (the extension point, ~1 hour):

1. Create `src/voice_generator/backends/<name>_backend.py` implementing
   `BaseBackend.synthesize(request, out_wav) -> SynthesisResult`.
   Import heavy deps *inside* methods, never at module top.
2. Register it in `backends/__init__.py` (one line in `BACKENDS`).
3. Always write output as PCM-16 WAV via `audio.write_pcm16_wav` —
   that's what chunk concatenation depends on.
4. Add its constructor knobs to `pipeline.synthesize_script` and the CLI.
5. Add fake-module tests in the style of `test_qwen.py` / `test_cosy.py`.

---

## Roadmap

1. **v0.1** — core + Chatterbox + mock, CLI, manifest, tests.
2. **v0.2** — Qwen3-TTS backend in its own venv, `--ref-text` support,
   A/B compare tool.
3. **v0.3 (this)** — CosyVoice 3 backend (local repo clone in its Conda
   env), A/B tool extended to all three backends, detailed README.
4. v0.4 — the handoff contract with Stage 2 (EchoMimicV3 Flash /
   MuseTalk): WAV + manifest in, lip-synced MP4 out.
