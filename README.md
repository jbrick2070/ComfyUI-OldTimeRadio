# ComfyUI-OldTimeRadio (SIGNAL LOST)

Turn **real news, public-domain stories, Shakespeare, or fully original LLM fiction** into a
finished **radio-drama video** — script, voices, music, and CRT-style video — fully automated
inside ComfyUI. Drop it in, queue one workflow, walk away, and a complete episode lands in
your output folder.

**Pipeline:** story source → LLM script → character voices + announcer + music themes (a
swappable 7-voice / 5-music engine roster; IndexTTS2 + Kokoro + Stable Audio 3 ship as the
defaults) → 48 kHz master mix → model-agnostic video (procedural CRT floor by default, or
HuMo / LTX / Wan / AnimateDiff / MiniMax H3 once you dial a heavier lane in) → final MP4.

100% local by default. No API keys required. Optional hosted LLM and all-cloud
routes exist; they stay off unless you turn them on.

> **Already installed it? Load the show:** **Workflow → Browse Templates →
> EXTENSIONS → comfyui-old-time-radio → otr_canonical**, then **Queue Prompt**.
> The 34 `OTR_` nodes are the parts; that workflow is the thing you actually run.

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

The shipped canonical workflow is lighter than you'd expect: its video-role dropdowns default
to the **procedural still/CRT floor** (no GPU video checkpoint required at all), and its image
role defaults to **Z-Image-Turbo** (Apache-2.0, no license friction). So a first run needs only:
the local writer LLM (Gemma or your own choice), Z-Image-Turbo, and the default voice/music
weights (IndexTTS2, Kokoro, Stable Audio 3). The heavier local video checkpoints — HuMo, LTX,
Wan, AnimateDiff, MiniMax H3 — are **optional upgrades** you dial in later via the
`OTR_VideoDirector` dropdowns; see [Which video models fit your card](#which-video-models-fit-your-card)
before downloading any of them. If a model is missing, the engine fails **loudly** and stops —
it never silently substitutes another model or quietly produces garbage. There is no automatic
fallback: the procedural CRT path is a route you **select** (and the canonical workflow ships
with it selected), not a net that catches a failed engine. Watch the console on the first run;
it names any missing weight and where it expects it.

### 4. Run it

1. Open **Workflow → Browse Templates**, scroll the left sidebar to **EXTENSIONS**,
   click **comfyui-old-time-radio**, and pick **otr_canonical**. (The console prints
   this same path on every start, right under the `[OldTimeRadio]` load banner.)
   *Prefer files? `workflows/otr_canonical.json` inside the installed pack is the
   same graph — drag it onto the canvas.*
2. Hit **Queue Prompt**.
3. Walk away. Script, voices, music, mastering, and video all run automatically. The shipped
   graph rolls a random story bank each run and renders through the procedural still/CRT floor
   — the fast, guaranteed-to-complete path. Swap dropdowns once you're ready for a specific bank
   or a GPU video engine.
4. Find the finished episode in **`output/otr/obs/`**.

---

## Requirements

- **GPU:** an NVIDIA card is recommended for the local video engines. The shipped canonical
  workflow renders through the procedural still/CRT floor by default (no GPU video model
  required); heavier local/cloud routing is opt-in via the `OTR_VideoDirector` dropdowns or
  explicit profile overrides. Episode length is set by act count, not a word target.
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
| `animatediff15_h3_video (16:9)` | same weights as `animatediff15_video`, hold-3 cadence | ? | ? | yes |
| `animatediff15_h5_video (16:9)` | same weights as `animatediff15_video`, hold-5 cadence | ? | ? | yes |
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

`animatediff15_h3_video` and `animatediff15_h5_video` are peers of the golden
`animatediff15_video` lane, differing only in how many delivered frames each
generated frame fills (hold-3 / hold-5 vs. the golden hold-2) — same weights,
same VRAM class. Three other AnimateDiff lanes (`animatediff15_v2_video`,
`animatediff15_v3_video`, `animatediff15_v3_haunted_video`) exist in the
registry but have not yet rendered a proving episode; they are not listed here
until they ship.

**Licensing note on this table:** most engines here are open weights, but two
are not. `h3_low_video` / `h3_low_audio_in` (MiniMax H3) run under a personal,
non-transferable authorization the maintainer obtained directly from MiniMax —
it does not transfer to your install; treat H3 as off unless you have your own
agreement with MiniMax. `animatediff15_video` and its two cadence peers load a
motion module with **no published license grant** (`commercial_clean = False`
in the adapter) — fine for personal/hobby use, not cleared for commercial
redistribution. See [License & Credits](#license--credits) for the full list.

### Procedural and still lanes -- these run anywhere

`still_flat (16:9)`, `still_motion (16:9)`, `still_pan (16:9)`,
`still_word (16:9)`, `viz_camera (16:9)`, `viz_green (16:9)`,
`viz_mxc_cpu (16:9)`, `viz_mxc_mandala (16:9)`.

The four `viz_*` lanes are pure numpy/PIL/ffmpeg with no model at all and no GPU
requirement. The `still_*` lanes cost whatever your chosen IMAGE model costs,
since the video side is a pan or a hold over a still. `still_flat` is the
canonical workflow's shipped default for every video role.

### Cloud lanes -- no local VRAM, but they are paid services

`cloud_kling_avatar (16:9)`, `cloud_seedance_2 (16:9)`,
`cloud_vidu_q2_pro_fast_720p (16:9)`, `cloud_wan_i2v (16:9)`,
`cloud_wan_i2v_audio (16:9)`, `google_omni_video (16:9)`,
`google_veo_video (16:9)`, `word_razzle (16:9)` (a Comfy Cloud partner lane
despite the name — it renders provider-side via Pixverse). All are OFF by
default; this project is offline-first and nothing here is required to make
an episode.

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
**muxed in last, byte-identical, in the archival copy** written to `otr/episodes/` — the audio
there is never re-encoded or altered by the video stage. The published copy in `otr/obs/` (the
one you actually watch) re-encodes that same audio to AAC 320 kbps for player compatibility;
the PCM content is unchanged, only the container codec differs.

```
Story bank → LedgerScriptWriter (LLM) → FreezeCascade → CastLock
     → character voices + announcer + music themes (per-role engine roster)
     → SceneSequencer → AudioEnhance → EpisodeAssembler  ==> 48 kHz MASTER (frozen)
     → VideoDirector / ShotLock (per-role engine + per-beat prompts)
     → VideoRenderBatch (render each beat through its engine)
     → SilentComposite → CaptionBurn → CreditsRoll → MasterAudioMux  ==> final MP4 in otr/obs
```

Only two of the five story banks (`scifi_news_pro`, `media_archive`) actually pull from a news
or archive feed — `public_domain` and `shakespeare` adapt a fixed source text, and `original` is
entropy-seeded with no external input at all. "Story bank" above covers all five; see
[Story sources](#story-sources-source-banks) for what each one actually consumes.

---

## Story sources (source banks)

The writer's `source_bank` dropdown selects where each episode's story comes from. The
shipped canonical workflow rolls randomly across every eligible bank each run
(`scifi_news_pro` is only the code-level fallback for a freshly-dropped, unconfigured node —
pin the dropdown to one bank if you want a fixed lane). Every lane is an INDEPENDENT bank (its
own story pack + story_rules) with no dependency on any other lane.

| Bank | What it does |
|------|--------------|
| `scifi_news_pro` | sci-fi radio drawn from a live science feed; an LLM-first multipass writer using the configured model slots |
| `media_archive` | media RSS / archive items → restoration-adventure episodes |
| `public_domain` | faithful radio adaptation of a public-domain source |
| `shakespeare` | Folger scene adaptation. **The Folger Digital Texts are CC BY-NC 3.0 (noncommercial)** — episodes from this bank inherit that restriction on the source text. |
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
beat, and that pick is honoured exactly: a missing or OOMing engine **fails loudly** and stops
the render rather than swapping in a substitute you did not choose. The frozen audio is never
touched either way. If you want the zero-GPU procedural CRT path, select it — it is the
canonical workflow's shipped default, not a rescue lane.

**Roles** (each selectable in `OTR_VideoDirector`):

| Role | What it is | Canonical default |
|------|------------|-------------------------------|
| `announcer_visual` | the announcer bookends | `still_flat` (image-model still, no video model) |
| `music_visual` | opening/closing theme bookends | `still_flat` (image-model still, no video model) |
| `character_video` | character dialogue beats | `still_flat` (image-model still, no video model) |

(The former `sfx` speaker role and `scene_broll` / `background_abstract` video roles were
removed in the 2026-07-01 cleanbreak — old ledgers using them fail loud by design.)

**Engines available:** HuMo (audio-driven face, 14B + 1.7B tiers), LTX (text/image→video and
audio-in), Wan (TI2V / I2V), AnimateDiff (Ghost Signal, three shipped cadence peers), MiniMax
H3 (personal license only, see the licensing note above), `mesh_stage`, and the cheap CPU
floors (CRT **visualizer**, Ken-Burns, flat still). Audio-driven engines are offered only where
audio exists; engines load one at a time with explicit VRAM reclaim between stages, and
renders are request-hash deterministic. The old VRAM tier system is gone — per-platform
workflow variants are the sizing mechanism now.

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

### Image models

The picture behind every still/motion lane and video role comes from a per-role image-model
dropdown (`announcer_image_model`, `music_image_model`, `character_image_model`), independent
of which video engine you picked. **`z_image_turbo` (Apache-2.0) is the shipped canonical
default for all three roles.** Same open-set "registry IS the menu" story as video: drop in an
adapter, no other edits, and it's selectable everywhere.

| Engine | License | Notes |
|---|---|---|
| `z_image_turbo` | Apache-2.0 | shipped default, all three roles |
| `lumina_image` | Apache-2.0 | ~7-12 GB measured |
| `flux2_klein` | Apache-2.0 | FLUX.2 Klein 4B |
| `flux_gen1` (Flux.1-dev) | **BFL non-commercial** | coded as the in-stack "gen-1" default; the canonical workflow does not select it |
| `ideogram4_local` | **non-commercial weights, opt-in** | heaviest local image engine, 16 GB-class only; typography-first for the `still_word` title card |

Six more engines run through the Comfy Cloud partner bridge (`cloud_flux_pro`,
`cloud_nano_banana_2`, `cloud_seedream_2`, `cloud_krea_2_turbo`,
`cloud_luma_photon_flash`, `ideo`) plus a direct Google Gemini/Nano-Banana adapter
(`google_image`, BYO API key). All are OFF by default — same "the dropdown pick is the
enable" pattern as the cloud video lanes, no local VRAM, budget-estimated per call, fail
loud without credentials. Full contract: [`docs/IMAGE_GEN_PREFLIGHT.md`](docs/IMAGE_GEN_PREFLIGHT.md).

### Headless canonical path

Agents and API tests use exactly one workflow file: `workflows/otr_canonical.json`.
By default, the headless wrapper applies no profile and leaves the saved dropdowns
alone. Explicit profiles are still available for deliberate route testing, such as
`otr_cloud_lanes` for the hosted Partner API path.

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
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile otr_cloud_lanes -Acts 3
```

`--words` no longer exists anywhere on this path. It went with the `target_words`
widget: act count is the only episode-shape knob, and length is an observation.

That path always loads `workflows/otr_canonical.json`; engine dropdowns move only
through explicit profiles, and ad-hoc `--set` patches are limited to creative/story
widgets.

### Optional: hosted LLM via OpenRouter (off by default)

The writer runs locally out of the box — the shipped canonical workflow pins Gemma-4-12B for
both the creative and technical slots (Mistral-Nemo is only the code-level fallback for a
freshly-dropped, unconfigured node). You can optionally route either slot to a hosted frontier
model via OpenRouter — it activates as soon as `OPENROUTER_API_KEY` is set (no separate opt-in
flag anymore), is cost-guarded, and fails closed. Full walkthrough:
[`docs/openrouter-setup.md`](docs/openrouter-setup.md).
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
  model/dependency. The render stops there rather than substituting a different engine, so fix
  the named dependency (or select the procedural CRT path) and run it again.
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
a 306-entry machine-readable Bug Bible distilled from this project's live production
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

The current line is **v2.0-alpha** (Open Video Model Platform; per-role video AND image
engines; five independent story source banks; per-platform workflow variants; frozen 48 kHz
audio master, byte-identical in the archival copy). Full per-version history is in the git log
and the GitHub Releases page.

## License & Credits

See [`LICENSE`](LICENSE). Built on ComfyUI and the open-source HuMo / LTX / Wan / AnimateDiff /
Z-Image-Turbo / Lumina / FLUX.2 Klein / IndexTTS2 / Kokoro / Stable Audio ecosystems, plus
several optional engines (Chatterbox, Dia, Bark, MusicGen, and others) — thanks to all of
their authors.

**A few optional, off-by-default pieces carry restricted terms, not open licenses:**

- `flux_gen1` (Flux.1-dev) — BFL non-commercial license.
- `ideogram4_local` — non-commercial model agreement; code ships, weights don't.
- `h3_low_video` / `h3_low_audio_in` (MiniMax H3) — a personal, non-transferable
  authorization the maintainer obtained directly from MiniMax; it does not carry over to
  your install.
- `animatediff15_video` and its cadence peers — the shipped motion module publishes no
  license grant at all (`commercial_clean = False`); fine for personal use, not cleared for
  commercial redistribution.
- The `shakespeare` story bank adapts Folger Digital Texts, which are CC BY-NC 3.0
  (noncommercial).

None of these are required for a first run — the canonical workflow's shipped defaults
(Gemma writer, Z-Image-Turbo, IndexTTS2/Kokoro/Stable Audio, procedural video floor) are all
open and commercial-friendly. Check each engine's own license before commercial use of the
others.
