# voice-generator website

The public site for this project, deployed by GitHub Pages from this
folder (`site/`) via `.github/workflows/pages.yml`. **This folder is
generated output — do not hand-edit it.** Run `voice-gen freeze` (from
the project root) to regenerate it from the living web UI; with
`--seed-mock` it fills gaps with mock demo data when the project has
no real clips or renders yet. Static HTML + CSS + a little vanilla JS;
no build step, no external assets, no framework. Every link is
relative, so the site works at the project Pages URL
(`enonymous1.github.io/voice-generator/`) and in a local preview. The
stylesheet is system-aware: `prefers-color-scheme: dark` switches the
site to a dark theme, and every page declares
`<meta name="color-scheme" content="light dark">`.

## Pages

- `index.html` — landing: what the tool is, how it works, backends table,
  quickstart. Generated from the Flask landing template
  (`src/voice_generator/web/templates/landing.html`); edit that, then
  re-freeze. Content mirrors the repo README; keep the two in sync when
  a backend ships or a version changes.
- `research.html` — the fast-open-voice-cloning-models brief, generated
  from the full report in `docs/research/` (newest dated file). The
  report is the single source: edit it, then re-freeze.
- `synth.html`, `setup.html`, `history.html` — frozen snapshots of the
  web UI's pages, with a static-preview banner; their forms don't work
  here, the live tool is `voice-gen serve`.
- `listen.html` — same script, four backends, side by side. It fetches
  `samples.json` and renders one card per backend in `expected`: a player
  when a sample exists, a pending slot when it does not. **Never put a
  placeholder or fake recording on this page** — a missing sample is a
  pending slot, not a stand-in.

## Samples (`samples.json` + `samples/`)

`samples.json` is the listen page's data file:

```json
{
  "script": "<exact text of recording-script.txt — keep in sync>",
  "script_source": "recording-script.txt",
  "expected": ["chatterbox", "qwen3", "cosyvoice", "voxcpm"],
  "samples": [
    {"backend": "voxcpm", "file": "samples/voxcpm.mp3",
     "duration_s": 52.3, "wall_s": 18.4, "sample_rate": 48000}
  ]
}
```

- `script` must match `recording-script.txt` (there is a test for this);
  the bench must render that script, or the page shows text no backend
  actually read.
- `samples` starts empty. After the A/B bench (`tools/ab_compare.py`),
  convert each backend's WAV to MP3 and add one entry per backend.
  `duration_s` and `wall_s` come from the bench's `ab_comparison.json`.
  MP3 (`libmp3lame -q:a 4`) keeps the page light; the WAVs stay the
  source of truth in `output/` and are never committed *for the listen
  page*. Note the freeze also copies render WAVs + manifests into
  `site/outputs/` for the frozen History page — those *are* committed
  as part of the snapshot, so only freeze renders you're happy to
  publish.
- The raw reference clip (`refs/me.wav`) is never published — personal
  data, gitignored. Only rendered clones go on this page.

## Preview locally

```bat
cd site
python -m http.server
```

then open `http://localhost:8000/`. Opening the HTML files directly
(`file://`) does **not** work for the listen page: `fetch` of
`samples.json` needs HTTP.

## Deploy

Push to `main` and the `pages.yml` workflow runs the test suite, then
deploys this folder. One-time setup (his step, after the first push):
Settings → Pages → Source: GitHub Actions. Until that is set, pushes do
not publish anything.
