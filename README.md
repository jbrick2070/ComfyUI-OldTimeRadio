# ComfyUI-OldTimeRadio (SIGNAL LOST)

Turn **real news, public-domain stories, Shakespeare, or fully original LLM fiction** into a
finished **radio-drama video** — script, voices, music, and CRT-style video — fully automated
inside ComfyUI. Drop it in, queue one workflow, walk away, and a complete episode lands in
your output folder.

**Pipeline:** story source → LLM script → character voices (IndexTTS2) + announcer (Kokoro) +
themes (Stable Audio) → 48 kHz master mix → model-agnostic video (HuMo / LTX / Wan / CRT
visualizer) → final MP4.

100% local by default. No API keys required. Optional hosted LLM and all-cloud
routes exist; they stay off unless you turn them on.

> **Branch note:** active development lives on the **`v2.0-alpha`** branch (the Open Video
> Model Platform below). Check out `v2.0-alpha` to get the current pipeline.

---

## New to ComfyUI? Start here

You need four things: ComfyUI, a GPU, the models, and this node pack. ~20 minutes start to finish.

### 1. Install ComfyUI

Use the official [ComfyUI Desktop installer](https://www.comfy.org/download) (easiest), or a
manual/portable ComfyUI install. Launch it once to confirm it opens in your browser.

### 2. Install this node pack

In **ComfyUI Manager → Install via Git URL**, paste:

```
https://github.com/jbrick2070/ComfyUI-OldTimeRadio
```

Then **check out the `v2.0-alpha` branch** (the active line). Restart ComfyUI so it loads the nodes.

*(Manual alternative: `git clone` into `ComfyUI/custom_nodes/`, `git checkout v2.0-alpha`, restart.)*

### 3. Install the models

OTR downloads most assets on first run, but the heavy diffusion checkpoints (HuMo, LTX, Wan,
Flux) and the TTS/music weights must be present in your ComfyUI models tree (`diffusion_models/`,
`vae/`, etc.). If a model is missing, the engine fails **loudly** and the pipeline falls back to
a guaranteed CRT floor — it never silently produces garbage. Watch the console on the first run;
it names any missing weight and where it expects it.

### 4. Run it

1. Drag **`workflows/otr_canonical.json`** into the ComfyUI canvas.
2. Hit **Queue Prompt**.
3. Walk away. Script, voices, music, mastering, and video all run automatically.
4. Find the finished episode in **`output/otr/obs/`**.

---

## Requirements

- **GPU:** an NVIDIA card is recommended for the local video engines. The shipped
  canonical workflow is the quick 30-word smoke canvas; heavier/local/cloud
  routing is handled by explicit profile overrides.
- **OS:** Windows or Linux. Tested heavily on Windows + RTX (Blackwell/sm_120).
- **Other setups:** per-platform workflow variants + recipes ship in-repo (16 GB NVIDIA
  canonical, cloud-lane variant, Mac ~10 GB ceiling, AMD). The Mac/AMD variants are
  drafts — not yet verified on real hardware.
- **Disk:** the model set is large (tens of GB). Episodes are a few dozen MB each.

---

## How it works

**Audio is the source of truth.** The writer produces a script, the voice/music engines render
it, and everything is assembled into a single **frozen 48 kHz master mix**. That master defines
the episode timeline and the per-beat clip budget. Video is rendered to fit the audio and is
**muxed in last, byte-identical** — the audio is never re-encoded or altered by the video stage.

```
News → LedgerScriptWriter (LLM) → FreezeCascade → CastLock
     → IndexTTS2 (characters) + Kokoro (announcer) + Stable Audio (themes)
     → SceneSequencer → AudioEnhance → EpisodeAssembler  ==> 48 kHz MASTER (frozen)
     → VideoDirector / ShotLock (per-role engine + per-beat prompts)
     → VideoRenderBatch (render each beat through its engine)
     → SilentComposite → CaptionBurn → MasterAudioMux  ==> final MP4 in otr/obs
```

---

## Story sources (source banks)

The writer's `source_bank` dropdown selects where each episode's story comes from.
Default: `scifi_fable2`. Every lane is an INDEPENDENT bank (its own story pack +
story_rules); a `_v3` suffix on some ids is a bake-off naming artifact, not a
dependency on any other lane.

| Bank | What it does |
|------|--------------|
| `scifi_fable2` | LLM-first multipass sci-fi writer (the shipped default) |
| `scifi_codex_v4` | proof-pressure sci-fi radio, a distinct multipass architecture |
| `media_archive` | media RSS / archive items → restoration-adventure episodes |
| `public_domain_story_v3` | faithful radio adaptation of a public-domain source |
| `shakespeare_v3` | Folger scene adaptation |
| `original_radio` | no-source original fiction seeded from an entropy spark draw |

A typed `custom_premise` rides along as an operator hint on the original lanes and as a
source override on the article lanes. Every lane is fail-closed: a bad source, context
overflow, or contract violation stops loudly instead of shipping a degraded story, and
the LLM writes all story text — Python validates, it never rewrites prose.

---

## v2.0-alpha — the Open Video Model Platform

The video layer is **model-agnostic**: a registry of pluggable engine adapters, chosen
**per role**, with no single model treated as "primary." You pick the engine for each kind of
beat; every chain ends at a guaranteed CRT "radio-floor" clip, so a missing or OOMing engine
**degrades loudly** and never aborts the episode or touches the frozen audio.

**Roles** (each selectable in `OTR_VideoDirector`):

| Role | What it is | Canonical default |
|------|------------|-------------------------------|
| `announcer_visual` | the announcer bookends | procedural visualizer |
| `music_visual` | opening/closing theme bookends | procedural visualizer |
| `character_video` | character dialogue beats | procedural visualizer / still floor |

(The former `sfx` speaker role and `scene_broll` / `background_abstract` video roles were
removed in the 2026-07-01 cleanbreak — old ledgers using them fail loud by design.)

**Engines available:** HuMo (audio-driven face, 14B + 1.7B tiers), LTX (text/image→video and
audio-in), Wan (TI2V / I2V), and the cheap CPU floors (CRT **visualizer**, Ken-Burns, flat
still, station card). Audio-driven engines are offered only where audio exists; engines load
one at a time with explicit VRAM reclaim between stages, and renders are request-hash
deterministic. The old VRAM tier system is gone — per-platform workflow variants are the
sizing mechanism now.

### Headless canonical path

Agents and API tests use exactly one workflow file: `workflows/otr_canonical.json`.
By default, the headless wrapper applies no profile and leaves the saved dropdowns
alone. Explicit profiles are still available for deliberate route testing, such as
`cloud_all` for the hosted Partner API path.

Headless/API smoke runs must use the canonical workflow wrapper:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Words 30
```

For a no-queue validation of the exact API prompt shape:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_canonical_api_run.py --offline-schemas --dry-run --words 30
```

Cloud route example:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile cloud_all -Words 30
```

That path always loads `workflows/otr_canonical.json`; engine dropdowns move only
through explicit profiles, and ad-hoc `--set` patches are limited to creative/story
widgets.

### Optional: hosted LLM via OpenRouter (off by default)

The writer runs locally (Mistral-Nemo) out of the box. You can optionally route the creative
and/or technical slot to a hosted frontier model via OpenRouter — it only activates when you
set both `OPENROUTER_API_KEY` and `OTR_ENABLE_OPENROUTER=1`, is cost-guarded, and fails closed.
Full walkthrough: [`docs/openrouter-setup.md`](docs/openrouter-setup.md).
The docs index is [`docs/README.md`](docs/README.md).

---

## Output layout

Everything for an episode lands under your ComfyUI `output/otr/` tree:

- `output/otr/episodes/<episode>/` — working assets (audio, frames, intermediate clips).
- `output/otr/obs/` — the **finished, playable episodes** (what you watch / publish).

Point OBS (or any player) at `otr/obs/` for a continuous broadcast — new finished episodes
appear there as they render.

---

## Troubleshooting

- **"SERVER DID NOT COME UP" on headless boot** — set `PYTHONUTF8=1` and
  `PYTHONIOENCODING=utf-8`; a non-UTF-8 console crashes on the first emoji log line.
  (The shipped launcher `scripts/_otr_soak_server_launch.cmd` already sets both, and a
  regression guard in the sibling survival-guide repo keeps it that way.)
- **An engine "fails loudly" mid-render** — that's by design; check the log for the missing
  model/dependency. The beat falls back to a CRT floor so the episode still completes.
- **Out of VRAM on a local video tier** — use the canonical procedural path or an
  explicit lighter/cloud profile.
- **No audio under the end credits** — known limitation: the credits scroll can outlast the
  master mix's closing theme. Tracked for a fix.
- **Nodes don't appear after install** — restart ComfyUI; confirm you're on the `v2.0-alpha`
  branch.

---

## Quality discipline

Development runs under a sibling QA harness — the
[ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide):
a 197-entry machine-readable Bug Bible distilled from this project's live production
incidents, plus a static regression suite that runs against this pack after every change.
Production bugs are staged in [`docs/PROD_BUG_LOG.md`](docs/PROD_BUG_LOG.md) and promoted
to the Bible in verified batches. Only bugs that actually failed in a live run qualify —
review findings never create entries on their own.

---

## The LEMMY easter egg

Every so often a character named **Lemmy** makes a cameo — a small tribute carried across the
project's generations. Born of the machine, still raising hell on the airwaves. 🤘

---

## Changelog

The current line is **v2.0-alpha** (Open Video Model Platform; per-role engines; ten story
source banks including four independent multipass sci-fi writer lanes and an original
fair-play mystery lane; per-platform workflow variants; frozen byte-identical audio master).
Full per-version history is in the git log and the GitHub Releases page.

## License & Credits

See [`LICENSE`](LICENSE). Built on ComfyUI and the open-source HuMo / LTX / Wan / Flux /
IndexTTS2 / Kokoro / Stable Audio ecosystems — thanks to all of their authors.
