# ss-voiceGenerator

Local, fully FOSS voice cloning from a text script — **Stage 1** of the AI Digital
Twin pipeline (Stage 2, the talking head, is a separate project that will consume
the WAV + manifest this one produces).

Zero APIs. Everything runs on your own machine: an **RTX 3060 12 GB / 64 GB RAM**
Windows box is the target hardware.

## Design decisions (from the 2026-10-22 pipeline research)

- **One interface, interchangeable engines.** The CLI, script chunking, WAV
  concatenation and the manifest know nothing about any specific model.
  A backend is ~one small class. That matters because the TTS field is moving
  monthly, and because the candidate models have wildly different licenses.
- **Backends are isolated in separate venvs.** Chatterbox, CosyVoice and
  Qwen3-TTS pin conflicting torch / CUDA stacks. They are never installed into
  the same environment; the core package itself has **no torch dependency**
  and imports every backend lazily.
- **Sequential, not concurrent.** On 12 GB you load one model, render, unload.
  This project only ever holds the TTS model — Stage 2 runs afterwards.

## Backends

| Backend | Status | License | Clone ref | VRAM (community) |
|---|---|---|---|---|
| `mock` | ✅ working | — (tone, not speech) | — | 0 — plumbing / timing tests |
| `chatterbox` | ✅ wired (v0.1) | MIT — outputs carry a PerTh watermark | ~10 s | ~1.5–6 GB |
| `cosyvoice` | planned | Apache-2.0 (CosyVoice 3) | ~3 s | ~8–10 GB |
| `qwen3` | planned | Apache-2.0 (Qwen3-TTS) | ~3 s | ~4–8 GB |

Explicitly **not** planned, per the research: Fish Speech (non-commercial
weights), IndexTTS-2 (custom bilibili license — best duration control in the
field, revisit only if those terms are acceptable), F5-TTS / Spark-TTS
(non-commercial weights), and "Breeze TTS 2" (no FOSS release found).

Chatterbox variants: `turbo` (default — English, fast, paralinguistic tags like
`[laugh]`), `original` (English, CFG/exaggeration control), `multilingual`
(23+ languages, pass `--language`).

## Install (Windows, Chatterbox venv)

```bat
cd ss-voiceGenerator
py -3.11 -m venv .venv-chatterbox
.venv-chatterbox\Scripts\activate
pip install -e ".[chatterbox]"
voice-gen doctor
```

Pinokio also has a community "ChatterBox-Multilingual" one-click script
(search-index evidence only — the manual venv above is the verified path).

## Use

```bat
voice-gen check-ref refs\me_10s.wav
voice-gen synth --script script.txt --ref refs\me_10s.wav --out output\ep1.wav
voice-gen synth --text "Hello, this is my digital twin." --ref refs\me_10s.wav --out output\hello.wav --variant multilingual --language es
```

Long scripts are split into sentence-sized chunks (default ≤300 chars),
synthesized one at a time, and concatenated into a single WAV. Alongside it a
`<name>.manifest.json` is written: backend, license, reference clip, settings,
and per-chunk durations — the provenance Stage 2 will read.

No GPU / no install? The whole flow still runs with `--backend mock`, which
writes a tone WAV of realistic duration — useful for testing Stage 2 timing.

## Reference clip tips

- 10 s of clean, single-speaker speech for Chatterbox (3 s is enough for the
  planned CosyVoice / Qwen backends). No music, no room echo, no other voices.
- `voice-gen check-ref` flags clips that are too short, stereo, or low-rate.

## Tests

```bat
pip install -e ".[dev]"
pytest
```

Tests cover chunking, reference validation, WAV concatenation, and a full
mock-backend pipeline run. They need no GPU and no model downloads.

## Roadmap

1. **v0.1 (this)** — core + Chatterbox + mock, CLI, manifest, tests.
2. v0.2 — second real backend (CosyVoice 3 *or* Qwen3-TTS) in its own venv,
   plus an A/B script that renders the same script through both backends.
3. v0.3 — handoff contract with Stage 2 (EchoMimicV3 Flash / MuseTalk):
   WAV + manifest in, lip-synced MP4 out.
