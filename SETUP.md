# Setup — voice-generator

The short, ordered path from an empty folder to hearing your own cloned
voice. This is the quickstart; the README is the full reference
(install details per backend, CLI reference, troubleshooting table).

Target machine: Windows 11, Nvidia RTX 3060 12 GB, 64 GB RAM. The core
and the mock backend also run on Linux/macOS — the model installs below
are written for Windows PowerShell.

## 0. Before you start

You need, in this order:

- **Git** (to clone, or you already have the extracted project folder).
- **Python** — the `py` launcher, with 3.11 *and* 3.12 installed. Each
  backend wants its own version (Chatterbox 3.11, Qwen3-TTS 3.12), so
  having both now saves a reinstall later. Check with `py --list`.
- **ffmpeg** on PATH — for converting reference clips. Check with
  `ffmpeg -version`. (Not needed by Stage 1 itself, but Stage 2 will
  require it.)
- A quiet room and a microphone for step 2. No studio gear needed.

No CUDA toolkit install is needed — the PyTorch wheels bring their own
CUDA runtime. No training, no dataset, no cloud account: every engine
here clones zero-shot from one short recording.

## 1. Get the project and prove the pipeline works (5 minutes, no GPU)

From the project folder, in PowerShell:

```powershell
py -3.11 -m venv .venv-core
.venv-core\Scripts\activate
pip install -e .
voice-gen doctor
voice-gen synth --backend mock --text "Hello world. This is a plumbing test." --out output\smoke.wav
```

You should see `output\smoke.wav` and `output\smoke.manifest.json`.
The mock backend writes a tone, not speech — its job is only to prove
the CLI, chunking, WAV output, and manifest all work before you install
a model. If this step fails, stop here and fix it first: a model
install will not fix a broken pipeline.

## 2. Record your reference clip (the step that decides quality)

This is the most important step in the whole setup. Clone quality is
mostly decided here, before any model runs.

1. Read [`recording-script.txt`](recording-script.txt) out loud, in a
   quiet room, microphone 15–30 cm away, in your normal speaking voice
   — not too fast, not too slow, not a "radio voice". No music, no TV,
   no other voices, no clipping.
2. Record with whatever is at hand: your phone's Voice Recorder, or
   Windows Sound Recorder. Aim for the whole script (~50 seconds); you
   will trim it down, and a longer take gives you the cleanest 10–30
   seconds to keep.
3. Save it as `refs\me.wav`. If it comes out as MP3/M4A, convert it:

   ```powershell
   ffmpeg -i refs\me.m4a -ac 1 -ar 24000 refs\me.wav
   ```

4. **Save the transcript.** `recording-script.txt` *is* the transcript
   — but only of what you actually said. If you read only part of it,
   copy just that part into `refs\me.txt`. Chatterbox does not need a
   transcript, but Qwen3-TTS and CosyVoice do, and the transcript must
   match the audio word for word, fillers and all.
5. Validate the clip before you burn a render on it:

   ```powershell
   voice-gen check-ref refs\me.wav
   ```

   It flags clips that are too short, stereo, or low sample-rate.
   Target: ~10 seconds for Chatterbox (3–10 s is enough for Qwen3 /
   CosyVoice), mono, 16 kHz or higher, integer PCM, one speaker only.
   A 32-bit float WAV (format tag 3, what some recorders write) fails
   here with `unknown format: 3` — convert it first:
   `ffmpeg -i refs\me.wav -c:a pcm_s16le refs\me_pcm16.wav`.

## 3. Install Chatterbox and hear your voice (start here)

Chatterbox is the first real backend to try: easiest install, no
transcript needed.

```powershell
py -3.11 -m venv .venv-chatterbox
.venv-chatterbox\Scripts\activate
pip install -e ".[chatterbox]"
voice-gen doctor
```

`doctor` should report torch with CUDA available and your GPU's VRAM.
Double-check with the explicit test — it must print `True`:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, your torch is a CPU-only build and every render
will run on CPU (Chatterbox survives that; Qwen3 looks hung for 10+
minutes — see "If something goes wrong" below). Fix it in this venv
before rendering anything:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then one short test line — judge this before rendering anything long:

> If the first render fails with `TypeError: 'NoneType' object is not
> callable` at `PerthImplicitWatermarker`, run
> `pip install "setuptools<81"` in this same venv and retry. That is an
> upstream incompatibility, not your setup: Chatterbox Turbo's
> watermarking library (`resemble-perth`) imports `pkg_resources`,
> which setuptools 81+ removed, and perth silently swallows its own
> import error — so the failure only surfaces at model load, after the
> 4 GB download. New installs get the pin automatically through the
> `chatterbox` extra; existing venvs need the one command above.

```powershell
voice-gen synth --text "Hello, this is a test of my cloned voice." --ref refs\me.wav --out output\hello.wav
```

The first render takes a while: it is downloading the model weights
from Hugging Face (one-time, a few GB). Listen to `output\hello.wav`.

**How to judge it:**

- Sounds like someone else, or muddy? Re-record the clip (step 2).
  A bad reference is the cause 90% of the time — do not try to fix it
  with settings.
- Sounds like you but flat? Turbo ignores `--exaggeration` — re-record
  with more expression, or try `--variant original --exaggeration 0.7`
  (the original variant honours those knobs; Turbo does not).
- Sounds good? Move on. Don't chase perfection on one sentence.

Then render a real script to check chunk joins and pacing:

```powershell
voice-gen synth --script script.txt --ref refs\me.wav --out output\ep1.wav
```

A `<name>.manifest.json` is written beside every render — it records
the backend, settings, and reference, so you can reproduce that exact
render later.

## 4. Try Qwen3-TTS and compare (optional, recommended)

A second engine on the same script and voice, so you can pick a winner
by ear. Qwen3 needs the transcript from step 2.

```powershell
py -3.12 -m venv .venv-qwen3
.venv-qwen3\Scripts\activate
pip install -e ".[qwen3]"
python -c "import torch; print(torch.cuda.is_available())"
voice-gen synth --backend qwen3 --script script.txt --ref refs\me.wav --ref-text-file refs\me.txt --out output\ep1_qwen.wav
```

The `torch.cuda.is_available()` check must print `True` — same
CPU-only-torch trap as step 3, same fix (`pip install torch
torchvision torchaudio --index-url
https://download.pytorch.org/whl/cu121` in this venv).

Or run both sides at once with the A/B tool (stdlib-only, run it with
any Python):

```powershell
python tools\ab_compare.py --script script.txt --ref refs\me.wav --ref-text-file refs\me.txt --chatterbox-python .venv-chatterbox\Scripts\python.exe --qwen-python .venv-qwen3\Scripts\python.exe --out-dir output\ab1
```

It writes one WAV per backend plus `ab_comparison.json` (durations /
wall times). The comparison is your ears — the JSON is diagnostics,
not a quality score.

## 5. CosyVoice 3 (optional, advanced)

Only worth it for multilingual work or Chinese dialects, after the
first two engines sound good. It is not a pip package: recursive git
clone + Conda Python 3.10 env + `requirements.txt` + sox, and it is the
fiddliest Windows install of the three. Full instructions are in the
README under "Install: CosyVoice 3 (advanced)". If it fights you, that
is expected — Chatterbox and Qwen3 already cover most use cases.

SoX note (verified 2026-11-09): `choco install sox` no longer works —
the package is gone from the Chocolatey community feed. Download
`sox-14.4.2-win32.zip` from SourceForge instead, unzip it, add the
folder to your user PATH, then close and reopen the terminal (or
restart VS Code — its terminals inherit the environment from launch)
and verify with `sox --version`.

## If something goes wrong

- `voice-gen doctor` first — it tells you which backend is actually
  installed in the venv you are in. Most "backend not installed" errors
  mean you are in the wrong venv: each backend has its own.
- First synthesis "hangs" for minutes: if progress bars are moving,
  it is downloading weights. Later runs are offline. But if it stalls
  *after* model load with no progress at all (seen 2026-11-09: Qwen3
  sat 12+ minutes on the line after `pad_token_id`), that is not a
  download — it is a CPU-only torch build grinding on CPU. Kill it,
  run `python -c "import torch; print(torch.cuda.is_available())"` in
  that venv (must be `True`), and install a CUDA torch build with
  `pip install torch torchvision torchaudio --index-url
  https://download.pytorch.org/whl/cu121` if it is not.
- CUDA out of memory on 12 GB: close whatever else holds VRAM, and for
  Qwen3 try `--model-size 0.6b`.
- The README's troubleshooting table covers the rest
  (transcript errors, sox/matcha errors, wrong-language output, …).
- If a render errors out, keep the exact error text — that is what a
  diagnosis needs.
