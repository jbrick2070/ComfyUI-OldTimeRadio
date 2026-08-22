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
> Model Platform below). Check out `v2.0-alpha` to get the current pipeline — or skip the
> branch juggling entirely and install the packaged alpha from the
> [ComfyUI Registry](https://registry.comfy.org/publishers/fluxus/nodes/comfyui-old-time-radio).

---

## New to ComfyUI? Start here

You need four things: ComfyUI, a GPU, the models, and this node pack. ~20 minutes start to finish.

### 1. Install ComfyUI

Use the official [ComfyUI Desktop installer](https://www.comfy.org/download) (easiest), or a
manual/portable ComfyUI install. Launch it once to confirm it opens in your browser.

### 2. Install this node pack

**Easiest — ComfyUI Registry:** in **ComfyUI Manager**, search for **ComfyUI-OldTimeRadio**
and click Install. Or from a terminal:

```
comfy node registry-install comfyui-old-time-radio
```

Registry installs are packaged, versioned snapshots of the current alpha
([registry page](https://registry.comfy.org/publishers/fluxus/nodes/comfyui-old-time-radio)) —
no branch checkout needed. Restart ComfyUI so it loads the nodes.

**From git (bleeding edge):** clone into your `ComfyUI/custom_nodes/` folder and check out
the active branch:

```
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
cd ComfyUI-OldTimeRadio
git checkout v2.0-alpha
```

Then restart ComfyUI so it loads the nodes.

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

## Which video models fit your card

**This table is the profile.** Pick your card, read the column, choose that name
in the `OTR_VideoDirector` dropdown. Names below are exactly the dropdown
entries.

Every figure is MEASURED, not estimated, and each says where it came from. A
blank verdict means nobody has measured it -- that is recorded as unknown rather
than guessed, because a guessed VRAM number is the one thing a user cannot
recover from.

**Read the 8 GB column as CANDIDATES, not promises**, and here is exactly how
far the evidence goes.

The strongest 8 GB evidence is a **CLAMPED SIMULATION**, not a run on an 8 GB
card. `MiniMax H3 MIME I2V` was rendered on the 16 GB card under a
`--reserve-vram 12` clamp, which forces ComfyUI to operate inside roughly an
8 GB budget:

| | measured |
|---|---|
| peak VRAM | **7.28 GiB** (baseline 2.52) |
| peak host RAM | **27.56 GiB** (baseline 17.35) |
| boot lane | `sage-free, no-pinned, reserve-12gb` |
| duration | 178.9 s |
| verdict | `PASS (cold)`, output visually approved |

**The host RAM number is the one people miss.** 27.56 GiB peak against an
8 GB laptop's typical 32 GiB of system memory is a tighter margin than the VRAM
is. A machine with 16 GiB of RAM will struggle with this pipeline regardless of
its GPU, and no VRAM table will warn you.

**Two limits, stated so nobody reads more into this than it carries.** The lab's
own rule is that a recipe is only `PASS` when a **second consecutive warm run**
records it, and both H3 MIME receipts are `run_number 1`, cold -- so this is a
cold pass, not a completed gate. And the physical 8 GB laptop has had its
hardware inventoried (**8,188 MiB VRAM, 31.7 GiB host RAM**) and has rendered
**nothing**: its own report says `HARDWARE_OBSERVED_NOT_ENROLLED`.

For contrast, `ltx25_high_video` measured **14.48 GiB** on the 16 GB card --
roughly 6.5 GiB past an 8 GB card's entire capacity, which is why its label says
5080-only rather than "high VRAM".

### Local video models

| Dropdown name | Measured VRAM | 8 GB | 12 GB | 16 GB |
|---|---|:--:|:--:|:--:|
| `ltx098_low_video (16:9)` | 6.8 GiB @ 512x288x161 | maybe | yes | yes |
| `h3_low_audio_in (16:9)` | 6.9-7.2 GiB @ 864x480 | **likely** | yes | yes |
| `h3_low_video (16:9)` | **7.28 GiB under an 8 GB clamp** | **likely** | yes | yes |
| `ltx23_low_audio_in (16:9)` | 7.36 GiB @ 1024x576x193 | maybe | yes | yes |
| `animatediff15_video (16:9)` | not measured (3.9 GB of weights) | ? | ? | yes |
| `wan22_high_video (16:9)` | 12.1 GiB @ 832x480x193 | no | maybe | yes |
| `humo17_high_audio_in_portrait (portrait)` | 12.84 GiB @ 480x832x129 | no | maybe | yes |
| `humo14_high_audio_in_wide (16:9)` | 13.06 GiB @ 832x480x97 | no | no | yes |
| `humo14_high_audio_in_portrait (portrait)` | 13.22 GiB @ 480x832x97 | no | no | yes |
| `ltx23_high_video (16:9)` | 13.3 GiB @ 1024x576x169 | no | no | yes |
| `wan22_high_fast (16:9)` | 12.8 GiB measured 2026-08-22 | no | maybe | yes |
| `wan22_high_i2v (16:9)` | 13.9 GiB @ f33 | no | no | yes |
| `ltx25_high_video (16:9)` | **14.48 GiB measured** | no | no | **5080-only** |
| `humo17_high_audio_in_wide (16:9)` | not measured at this aspect | ? | ? | yes |
| `mesh_stage (16:9)` | not measured | ? | ? | yes |

### Procedural and still lanes -- these run anywhere

`still_flat (16:9)`, `still_motion (16:9)`, `still_pan (16:9)`,
`still_word (16:9)`, `word_razzle (16:9)`, `viz_camera (16:9)`,
`viz_green (16:9)`, `viz_mxc_cpu (16:9)`, `viz_mxc_mandala (16:9)`.

The four `viz_*` lanes are pure numpy/PIL/ffmpeg with no model at all and no GPU
requirement. The `still_*` and `word_razzle` lanes cost whatever your chosen
IMAGE model costs, since the video side is a pan or a hold over a still.

### Cloud lanes -- no local VRAM, but they are paid services

`cloud_kling_avatar (16:9)`, `cloud_seedance_2 (16:9)`,
`cloud_vidu_q2_pro_fast_720p (16:9)`, `cloud_wan_i2v (16:9)`,
`cloud_wan_i2v_audio (16:9)`, `google_omni_video (16:9)`,
`google_veo_video (16:9)`. All are OFF by default; this project is
offline-first and nothing here is required to make an episode.

### Two lanes need their own boot

`h3_low_video` and `h3_low_audio_in` require a sage-free boot with pinned memory
disabled and VRAM reserved. They will fail preflight on a standard boot -- and
that is the guard working, not the lane being broken. Both are also the slowest
local lanes by a wide margin.

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

### Preflight guides -- the checklists that gate a change

Each subsystem has a preflight document: a gate-by-gate checklist run whenever
that subsystem is added to or materially changed, each backed by an enforcement
suite so the doc cannot drift from the code.

| subsystem | guide | enforced by |
|---|---|---|
| video models | [`docs/VIDEO_LANE_PREFLIGHT.md`](docs/VIDEO_LANE_PREFLIGHT.md) | `tests/test_lane_preflight_matrix.py` |
| image models | [`docs/IMAGE_GEN_PREFLIGHT.md`](docs/IMAGE_GEN_PREFLIGHT.md) | `tests/test_image_gen_preflight_matrix.py` |
| TTS voices | [`docs/TTS_VOICE_PREFLIGHT.md`](docs/TTS_VOICE_PREFLIGHT.md) | `tests/test_tts_voice_preflight_matrix.py` |
| source banks / story | [`docs/SOURCE_BANK_PREFLIGHT.md`](docs/SOURCE_BANK_PREFLIGHT.md) | the roster/bijection suites it names |

The rule that binds them (operator, 2026-08-21): **every video lane obeys the
per-role image-model dropdowns** -- the picture a `still_*` or motion lane holds
is minted by whichever image engine the operator selected for that role. The
only exemption is the `viz_*` visualizer family, which is procedural and mints
no still -- and each of those lanes declares that exemption out loud
(`accepts_still = False`); staying silent is a test failure. Adding your own
engine? Start at `docs/EXTENDING_OTR.md`, then run the matching preflight.

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
