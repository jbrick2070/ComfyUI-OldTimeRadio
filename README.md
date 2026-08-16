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
Default: `scifi_news_pro` (the local, offline-first sci-fi lane). Every lane is an
INDEPENDENT bank (its own story pack + story_rules) with no dependency on any
other lane.

| Bank | What it does |
|------|--------------|
| `scifi_news_pro` | sci-fi radio drawn from a live science feed; the local default, an LLM-first multipass writer using the configured model slots |
| `media_archive` | media RSS / archive items → restoration-adventure episodes |
| `public_domain` | faithful radio adaptation of a public-domain source |
| `shakespeare` | Folger scene adaptation |
| `original` | no-source original fiction seeded from an entropy spark draw |

A typed `custom_premise` rides along as an operator hint on the original lanes and as a
source override on the article lanes. Every lane is fail-closed: a bad source, context
overflow, or contract violation stops loudly instead of shipping a degraded story, and
the LLM writes all story text — Python validates, it never rewrites prose.

**Add your own source bank:** every bank is independent, and you can author a sixth
peer to the shipped five — your own feed, archive, or source strategy — running through
the same trusted writer. The requirements contract (above all: the episode ledger must
be COMPLETE for every downstream consumer) lives in
[`docs/EXTENDING_OTR.md`](docs/EXTENDING_OTR.md); read it before authoring.

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

### The video model reference — read these two before adding or changing an engine

| doc | what it holds | kept true by |
|---|---|---|
| [`docs/ENGINE_MATRIX.md`](docs/ENGINE_MATRIX.md) | **every per-model number** — clip window, frame ladder, continuity, join mode, segment counts, effective canvas | **generated + drift-gated.** `python tools/engine_matrix.py --check` is a suite test, so it cannot disagree with the adapters |
| [`docs/2026-08-02-FINAL-all-engine-maths-and-stills.md`](docs/2026-08-02-FINAL-all-engine-maths-and-stills.md) | the things a generator cannot derive — still logic and the local/cloud re-mint split, the fix list with per-item status, the open decisions, the padding rule | by hand, with a dated verification stamp |

**The rule between them: a hand-maintained doc must never re-type a number the
generated one already owns.** That is not style. On 2026-08-06 the hand-written
tables were found asserting 3 and 10 segments for HuMo where the live registry
said 5 — a ceiling that had moved four days earlier — while the drift-gated
matrix had been right the whole time. Cite the generated matrix; do not copy it.

Multi-clip coverage itself (how a long beat is partitioned into chained or
jump-cut segments) is settled in `nodes/_otr_video_engines/coverage_plan.py`, and
the arithmetic totals exactly on every engine, local and cloud.

**Padding rule (operator, 2026-08-06):** no mirror and no ping-pong anywhere —
every second of audio gets ORIGINAL video, and a short render fails loud rather
than filling. The single sanctioned exception is the closing-theme backdrop that
holds the last drama clip under the closing theme; the credits roll itself
freezes a frame and never loops.

### Headless canonical path

Agents and API tests use exactly one workflow file: `workflows/otr_canonical.json`.
By default, the headless wrapper applies no profile and leaves the saved dropdowns
alone. Explicit profiles are still available for deliberate route testing, such as
`cloud_all` for the hosted Partner API path.

Headless/API smoke runs must use the canonical workflow wrapper:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Acts 3
```

For a no-queue validation of the exact API prompt shape:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_canonical_api_run.py --offline-schemas --dry-run --act-count 3
```

Cloud route example:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile cloud_all -Acts 3
```

`--words` no longer exists anywhere on this path. It went with the `target_words`
widget: act count is the only episode-shape knob, and length is an observation.

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

## Known limitation: character drift

**Some episodes will hand a line to the wrong character**, and this pack does not
fix that. You may hear a character claim a job that belongs to someone else, say
they don't know something they plainly do, or — rarest and most obvious — address
themselves by name. It is uncommon, it does not break a render, and the episode
still plays. But it is real and you should know about it before you run this.

Everything else in the acceptance test **is** handled. A post-story clean stage
reads every spoken row with a model and rewrites anything that is not speech —
stage directions, sound cues, narration that would otherwise be read aloud by the
voice actor. That works, and it is measured (see below).

**Why character drift is not fixed, honestly.** It was built and tested, and it
did not pass. On a planted test set where the same line appears twice — once in
the wrong character's mouth and once in the right one — the detector found 3 of 6
planted defects on one run and 1 of 6 on the next, with an identical detector and
identical inputs. It reliably caught only the blatant case (a character naming
themselves) and never caught the subtler ones. A detector that unstable cannot be
trusted to rewrite dialogue, so it ships **disabled** rather than quietly making
episodes worse.

**The constraint is hardware, not design.** This project is deliberately 100%
local and offline — the reference machine is a 16 GB laptop GPU, and the largest
model that comfortably fits is in the 12B class. Judging whether a line belongs to
a particular character means holding the whole cast, and who knows what, in mind
while reading a single sentence. That is a harder ask than spotting a stage
direction, and a 12B is not reliable at it.

**If you want to chase it:** the pass is written, tested and ready — set
`JUDGE_ATTRIBUTION = True` in `nodes/_otr_ledger_clean.py`, and use a **frontier
model well above what a 16 GB card can run**. The measurement rig is included so
you can check whether your model actually does better rather than taking anyone's
word for it:

```bash
python scripts/otr_clean_stage_lab.py --f2 --model <your-model>
```

It scores recall (planted defects caught) *and* false alarms on clean lines, which
is the half a normal render cannot show you. Do not judge a change on recall
alone — it is easy to catch everything by suspecting everything, and that rewrites
good dialogue.

The same rig, without `--f2`, measures the stage-direction cleanup that **is**
shipped and on by default.

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
