# Which music model ships -- the bench that settles the fork

Driver: Claude Fable 5.1 (Cowork, 5080), sole judge. Written 2026-09-12 for the
first ARC row of `docs/GO_FORWARD_PLAN.md`. Operator, this morning: *"the whole
machine is yours"* -- so every arm below was rendered, including the one that
needed a download, and nothing here is guessed.

**This document qualifies nothing about the episode pipeline.** Every render is
a stock-node graph against the resident headless server (PID 56996, booted
06:20, code through `17238f4f`), not the canonical workflow. It settles a
design fork only. Testing is closed; no canonical leg was run.

## 1. The premise the row was built on is wrong, and it changes nothing

The row says *"Small-Music is a LOOP model by its publisher's own
description"*. It is not. Read from the live model cards on 2026-09-12:

* `stabilityai/stable-audio-3-small-music` -- *"a family of fast latent
  diffusion models ... for variable length audio generation and editing. Since
  our models can generate several minutes of audio, variable-length generations
  are key"*. The card's own example asks for a 120-second synthpop
  instrumental. The words "loop" and "one-shot" do not appear on it.
* The loop wording belongs to **Stable Audio Open 1.0**, a different model,
  and reached this repo through a comment in `nodes/_otr_music_prompt.py`
  ("Stable Audio Open is built to make loops and one-shots") that was written
  about that model and read as if it described this one.

What IS true, and is the real reason the fork exists: the operator heard a
loop (*"a loop tape deck"*, 2026-09-12), and the loudness envelope of that cue
repeated on a ~0.25 s period. That is a measurement about a render, not a
description of a model. The question is therefore not "is Small a loop model"
but "does a bigger checkpoint render the show's cues with less repeat and
more structure, at a cost the pack can carry". That is what the bench asks.

## 2. The candidates, grounded

| candidate | file | params | on disk | licence |
|---|---|---|---|---|
| Small-Music base | `stable_audio_3_small_music_base.safetensors` (2,270,384,940 B) | 0.6B, F32 | yes | Stability Community |
| Medium base | `stable_audio_3_medium_base.safetensors` (9,222,116,660 B) | 2B, F32 | yes (fetched 02:24) | Stability Community |
| ACE-Step 1.5 | `acestep_v1.5_turbo` 4.79 GB + `qwen_0.6b_ace15` 1.19 GB + `qwen_4b_ace15` 8.38 GB + `ace_1.5_vae` 0.34 GB | DiT + LM planner | fetched this session | Apache-2.0 (Comfy-Org repack), MIT (upstream) |

Each BASE card carries the same note, pointing at its OWN post-trained
sibling: *"This is the base (pre-trained) model intended for fine-tuning. If
you are looking to generate audio directly, please use Stable Audio 3 Small
Music instead"* on the small-base card, and *"... Stable Audio 3 Medium
instead"* on the medium-base card (the first draft of this document
misquoted the small card as pointing at Medium; cursor r1 caught it). The
pack prefers the base file anyway, and PBUG-20260912-03 is the measured
reason: only the
base file answers cfg and the negative prompt at all, and the post-trained one
driven at any guidance above 1.0 renders the broadband burst the operator
hears as a tape scratch. That finding stands; this bench does not reopen it.

ACE-Step 1.5 is a different architecture: a Qwen language model plans a song
(metadata, structure, audio codes), then a DiT renders it. Its native ComfyUI
graph (`blueprints/Text to Audio (ACE-Step 1.5).json`) is
`UNETLoader(turbo) -> ModelSamplingAuraFlow(3.0) -> KSampler(8 steps, cfg 1,
euler/simple)` with `DualCLIPLoader(qwen_0.6b, qwen_4b, type=ace)` feeding
`TextEncodeAceStepAudio1.5` (tags, lyrics, bpm, duration, key, language,
audio-code generation on, LM cfg 2.0). The empty-latent node accepts 1 s
upward, so a 4-12 s cue is a legal request. Shipping it would mean a NEW
engine module and registration (five fixtures and two rosters), a ~14.7 GB
download in place of 3.5 GB, and an LM stage in front of every cue.

## 3. The recipe held constant

The two SA3 arms run exactly what `eng_stable_audio_3.py` ships for a base
checkpoint -- cfg 4.0, 100 steps, `dpmpp_3m_sde_gpu` / `exponential`, a
conditioning window 3x the cue -- so the only variable between them is the
checkpoint. The separate "Which SA3 recipe ships" row is not touched here.
ACE-Step runs its blueprint recipe verbatim: KSampler cfg 1.0 and the LM
planner's cfg 2.0, both held at the blueprint's values.

Six prompt families, three seeds each, per arm:

| family | what it tests | cue length | negative |
|---|---|---|---|
| organ_cathedral, organ_toccata | the burst control from `scripts/otr_organ_bench.py` | 20 s | the organ bench's production negative |
| shakespeare_opening, shakespeare_closing | the sustained lane -- the only bank still carrying the anti-loop negative | 12 s, 8 s | `NEGATIVE_PROMPT_DEFAULT` |
| house_opening, techno_opening | the two genre lanes that came back as pads on the 2026-09-12 canonical legs | 12 s | `NEGATIVE_PROMPT_RHYTHMIC` |

The four show families are composed at run time by `nodes/_otr_music_prompt`
(`compose_music_prompt` -> `compose_engine_prompt`) from a story meta per
bank, so the text is byte-for-byte what an episode would send. The bench
script, the summary script and the per-render JSON sit beside this document
(`music_model_bench.py`, `bench_summary.py`, the raw `bench_sa3.json` and
`bench_ace.json`, and `bench_all_with_repeat.json` with the post-hoc repeat
metric added to all 54 rows), so every number below is re-measurable from
the FLACs under `output/organ_bench/mmb/`. One protocol slip, for the
record: the two smoke renders (`small_base__house_opening__seed0` and
`ace15__house_opening__seed0`) were written with the same label as the
arm's first seed, so those two labels have an `_00001` (smoke) and an
`_00002` (arm) file; the JSON points at the arm's. The SA3 pair is
identical audio; the ACE pair is NOT (correlation -0.01 between two renders
of the same seed and prompt), which is recorded under ACE below.

## 4. What is measured, and what each number can and cannot say

* **bursts** -- the organ bench's own detector (50 ms blocks within 12 dB of
  peak, flatness > 0.20, > 45 % of energy above 4 kHz). On sustained
  material it is the tape-scratch defect. **On a rhythmic lane it also fires
  on a legitimate hi-hat or clap**, which has exactly that signature, so a
  burst count on house or techno is read beside the clip, not alone.
* **loopiness** -- the organ bench's envelope autocorrelation max over
  0.25-6 s. On sustained material the max sits AT the 0.25 s floor, which
  is envelope smoothness, not a repeat. Carried for comparison with the
  65-render campaign; not used to decide.
* **repeat @ lag** -- the strongest LOCAL peak of the same autocorrelation
  between 1 s and 6 s: a bar or phrase that actually comes back, and where.
  This is the honest loop number for the sustained lane.
* **pulse, tempo** -- onset-envelope autocorrelation peak in 60-180 BPM
  (clarity of a beat) and the tempo librosa commits to. Meaningful on the
  genre lanes; on organ or consort the tempo drifts to librosa's 120 BPM
  prior and pulse is low, which is the correct reading of no beat.
* **wall seconds, VRAM peak** -- nvidia-smi total GPU use sampled at 2 Hz
  during the render. The server held ~5.8 GB resident before the bench.

## 5. Results

### 5.1 Stable Audio 3: small base against medium base, 18 renders each

Means over three seeds; bursts and clipped samples are totals. `loopiness` is
the organ bench's 0.25-6 s max (carried for comparison with the 65-render
campaign), `repeat` the 1-6 s local peak, `pulse` beat clarity, `tempo` what
librosa commits to. VRAM is total GPU use with the server's ~5.8 GB already
resident.

| family | arm | bursts | clipped* | loopiness | repeat | pulse | tempo | peak dBFS | wall s | VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| organ_cathedral | small | 0 | 0 | 0.78 | 0.47 | 0.50 | 108 | -3.3 | 7.1 | 5824 |
| organ_cathedral | medium | 0 | 0 | 0.39 | 0.26 | 0.05 | 124 | -5.3 | 10.1 | 9288 |
| organ_toccata | small | 0 | 0 | 0.60 | 0.18 | 0.31 | 108 | -2.8 | 7.6 | 5824 |
| organ_toccata | medium | 0 | 0 | 0.28 | 0.15 | 0.07 | 123 | -6.5 | 9.1 | 9288 |
| shakespeare_opening | small | 0 | 0 | 0.80 | 0.23 | 0.44 | 115 | -1.9 | 7.6 | 5756 |
| shakespeare_opening | medium | 0 | 0 | 0.68 | 0.21 | 0.14 | 114 | -3.8 | 9.1 | 9180 |
| shakespeare_closing | small | 0 | 0 | 0.59 | 0.17 | 0.32 | 115 | -3.8 | 6.1 | 5756 |
| shakespeare_closing | medium | 0 | 0 | 0.46 | 0.22 | 0.08 | 119 | -2.9 | 9.1 | 9148 |
| house_opening | small | 2 | 2 | 0.52 | 0.44 | 0.78 | 121 | -1.1 | 6.7 | 5756 |
| house_opening | medium | 0 | 0 | 0.74 | 0.63 | 0.53 | 123 | -2.6 | 9.1 | 9116 |
| techno_opening | small | 1 | 142 | 0.77 | 0.51 | 0.63 | 129 | 0.0 | 6.1 | 5756 |
| techno_opening | medium | 0 | 0 | 0.73 | 0.65 | 0.81 | 129 | -3.9 | 9.2 | 9180 |

\* `clipped` counts samples at full scale in the bench FLAC, which is the
save node's clamp on the raw decode. **Production never ships that**: every
cue goes through `_ceiling_the_cue` in `nodes/stable_audio_theme.py:333`, a
limiter on the float waveform at -1 dBFS (Opus r1). So the column says a
render came back HOT, not that an episode would clip.

Per arm: small 3 bursts, three techno seeds at or above full scale before
any ceiling, mean 6.9 s, 5.8 GB; medium 0 bursts, never above -1.8 dBFS,
mean 9.3 s, 9.3 GB.

**What the numbers say, read the way section 4 says to read them.**

* **On the sustained lane the arms separate on every render.** Twelve of
  twelve small renders (two organ pieces and two consort cues, three seeds
  each) carry beat clarity of 0.31-0.50 in the 60-180 BPM band on material
  that has no beat, and librosa commits to a tempo near 108-115 for them;
  twelve of twelve medium renders sit at 0.05-0.17, a held chord reading as a
  held chord. Loopiness and repeat lean the same way on the organ pieces
  (0.78 to 0.39 and 0.47 to 0.26 on the cathedral) and are level on the
  consort cues. **This is a proxy, not the complaint itself:** the operator's
  cue measured a ~0.25 s envelope period, which is 240 BPM and outside the
  pulse band; what the pulse metric sees is the same periodic onset structure
  at its half-rate. It is systematic (24 renders, no overlap), and whether it
  is the texture he hears is his ear's call, not this table's.
* **On the genre lanes both commit to the asked tempo** (house asked 122,
  got 120-123; techno asked 128, got 129) -- twelve of twelve base renders,
  so at cfg 4.0 on a base file the "pads" of the n=1 canonical legs do not
  reproduce on either arm. **The genre result is mixed:** medium's techno
  pulse is stronger (0.81 vs 0.63) and its house pulse weaker and more
  variable (0.32-0.69 vs a flat 0.78). A bar repeating every ~1.9 s at
  123-129 BPM is what a groove IS, so the higher `repeat` on the genre cues
  is the requested behaviour, not the defect.
* **Small's three bursts are all on the rhythmic lanes**, where a hi-hat
  has the burst signature, so the count alone does not convict. Its techno
  renders came back at full scale on all three seeds; the production
  ceiling would take that down, so it is a level observation, not a defect.
* **Cost.** Medium adds about 2.4 s per cue (three cues an episode: under
  ten seconds on a ~24 min run) and about 3.4 GB of GPU memory over small.
  On a 16 GB card that is nothing. On an 8 GB card it is a 2B-parameter
  model at bf16 (~4 GB) plus the 1.2 GB text encoder plus activations for a
  36 s window: plausible under ComfyUI's low-VRAM path and NOT proven here.
  The 4060 owns that answer.

### 5.2 ACE-Step 1.5, 18 renders, blueprint recipe

| family | bursts | clipped* | repeat | pulse | tempo (asked) | peak dBFS | RMS dBFS | wall s | VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|
| organ_cathedral | 0 | 0 | 0.44 | 0.45 | 125 | -14.4 | -32.1 | 7.6 | 15375 |
| organ_toccata | 0 | 0 | 0.45 | 0.26 | 127 | -16.6 | -37.0 | 7.6 | 15483 |
| shakespeare_opening | 0 | 0 | 0.34 | 0.32 | 133 | -6.7 | -29.4 | 4.5 | 15433 |
| shakespeare_closing | 0 | 0 | 0.03 | 0.37 | 126 | -10.5 | -30.3 | 4.5 | 15388 |
| house_opening | 0 | 1009 | 0.37 | 0.27 | 112 (122) | -3.6 | -25.1 | 4.7 | 15408 |
| techno_opening | 1 | 87 | 0.43 | 0.67 | 128 (128) | 0.0 | -16.2 | 4.5 | 15495 |

**Cut, on the bar's own terms -- durability, not taste.**

* **Its level is not a property of the prompt.** The same house prompt
  rendered at 0.0 dBFS peak / -15 dBFS RMS on one seed and -5 dBFS / -31
  dBFS on the next: a 16 dB swing in loudness between seeds, and the organ
  and consort families sit at -29 to -40 dBFS RMS, which is near silence
  for a music bed. The pipeline's ceiling never gains a cue UP
  (`_ceiling_the_cue` is a limiter, and `scene_sequencer` levels dialogue
  only), so a cue like that ships nearly inaudible. That is the "silently
  wrong render" the bar names.
* **It is not reproducible from its seed.** The smoke render and the arm's
  render of `house_opening` seed 0 -- same graph, same seed passed to both
  the sampler and the LM planner -- came back as different music
  (correlation -0.01; the SA3 pair rendered the same way is identical
  audio). The pack's replay contract makes the seed the determinism
  carrier; a music engine that ignores it cannot be A/B'd or replayed.
* **It missed the asked tempo on house** (125, 128 and 83 BPM for a 122 BPM
  request; pulse 0.27) while techno landed. n=3 is thin, but the SA3 arms hit
  their tempo on 12 of 12.
* **Memory.** Total GPU use read 15.4-15.5 GB during every ACE render,
  against the same ~5.8 GB resident baseline the SA3 arms started from, so
  ACE's own footprint is roughly 9.7 GB (turbo DiT plus a 4B planner): the
  largest of the three by a wide margin, a non-starter on 8 GB, and a
  forced-offload proposition on 16 GB. This is a stock-node number, not a
  pipeline claim.
* **Its sustained cues carry a pulse** (0.26-0.45), the texture small has
  and medium does not.
* It IS the fastest arm (5.6 s mean against medium's 9.3 s), and that is
  the one measured gain; it does not buy back a bed that is inaudible on
  one seed and clipping on the next. The price of finding out more is a new
  engine module, five fixtures, two rosters and a 14.7 GB download in place
  of 3.5 GB.

The four ACE-Step files stay on disk under `C:\ComfyUI-Models` (14.7 GB) for
the operator to keep or delete; nothing in the pack references them.

## 6. Decision

**The fetch fix ships now on the small BASE file. Medium goes to the
operator's ear with the paired clips. ACE-Step is cut. The engine's
preference ladder does not change today.**

1. **What ships in this change** is the CODE row: a fresh install fetches
   `stable_audio_3_small_music_base.safetensors` -- the file the engine
   already prefers and the only one whose cfg and negative prompt are live.
   That is the silent-wrong-render fix and it is independent of which
   checkpoint the operator ends up preferring.
2. **Medium is measured better on the sustained lane, mixed on the genre
   lanes, never hot, and costlier**: 9.22 GB on disk, +3.4 GB GPU, +2.4 s a
   cue, unproven on 8 GB. That is a recipe change on the shipping surface,
   and the rulings this repo runs under are explicit about who decides one:
   *"judge it as radio drama"*, *"recipes are hard-won"*, and CLAUDE.md
   section 0A, *"a measurement is not a proof, and only the canonical path
   ships"*. Flipping the ladder on a stock-node bench would change every
   5080 episode before he has heard one paired clip. So the fork goes to
   section 3 of the plan as a one-word ruling, with the eighteen paired
   clips per arm in `output/otr/obs/music_model_bench/` (same seed and
   prompt across arms). The first draft of this document flipped the ladder;
   both contrarians refuted that, and they were right.
3. **If the word is "medium"**, the change is small and named: the head of
   `_CKPT_PREFERENCE` in `eng_stable_audio_3.py`; a `stable_audio_3_medium`
   lane and `LANE_INFO` row in `scripts/otr_fetch_lane_weights.py` (which
   the asset index will then advertise as a public one-command path, so the
   row must say "16 GB, unproven on 8 GB"); the provisioner's music lane
   forked on `low_vram` exactly as `otr_provision.py:1749` already forks the
   image lane; and a COVERAGE_OWED line for the 4060 to prove or refuse
   medium on 8 GB. The universal fetch default stays the small base file
   either way. **If the word is "small"**, nothing further happens.
4. **ACE-Step is cut** for the reasons in 5.2 -- level instability, seed
   non-determinism, a tempo miss, the largest footprint, and a new engine
   for no measured gain. Not a taste call.

## 7. The CODE row this settles, and what it actually changed

The plan row said one word at three sites. Traced through the real code it
is more than that, and the three-word version would have turned the silent
wrong render into a dead one:

* **The mechanism.** `resolve_ckpt()` fell through to `_CKPT_PREFERENCE[-1]`
  -- the POST-TRAINED file -- with nothing on disk (`eng_stable_audio_3.py:257`
  on HEAD). The preflight asks the engine which file it will load
  (`_otr_visual_assets.py:435`) and downloads THAT name if the manifest
  allows it. So on a fresh install the fall-through chose the download.
  Repointing the manifest alone would have made the preflight ask for a name
  the manifest no longer allowed, and `native_requests` refuses that with
  `VisualAssetError` at `:377`.
* **The fix.** A named `_FETCH_DEFAULT` (the small base file) is what the
  fall-through returns, with `is_base=True`, when nothing is on disk or
  ComfyUI is absent. The manifest gains the base row and KEEPS the
  post-trained row, so an explicit `OTR_SA3_CKPT` pin to it -- the A/B
  harness's control arm -- still downloads on a box that lacks it (Opus r1
  item 5, cursor r1 item 10). Existing installs are untouched: a box holding
  only the post-trained file resolves it and downloads nothing, which is a
  deliberate prior ruling (`test_an_install_with_only_the_post_trained_file_still_renders`).
  The one command that upgrades such a box is now
  `python scripts/otr_fetch_lane_weights.py stable_audio_3`, and the 4060
  is such a box (`docs/4060_DRILL_LOG.md:4832`); its window owns that.
* **The wiring assertion that was missing.**
  `test_a_fresh_install_is_sent_to_a_file_the_manifest_can_fetch` mocks an
  empty disk, asserts the fall-through is the named base constant, that it
  is in the engine's own ladder, and that it is a key in `MANIFEST` -- and
  that the post-trained name stays allowlisted but is not what an empty
  disk is sent to fetch. `test_resolution_never_raises_when_comfy_is_absent`
  now expects the fetch default. The planner test's `_SA3` stub predates
  `resolve_ckpt` and was already failing on HEAD; it now mirrors the real
  adapter surface (`resolve_ckpt()` plus `_TENC`, with `_CKPT` empty).
* **The other sites.** `scripts/otr_fetch_lane_weights.py`'s `stable_audio_3`
  lane and `config/profiles/otr_runpod_starter.json`'s required model, then
  `build_variants.py --all` (93 emitted, `--check` clean) for the launch
  recipe. **A consequence, stated:** that profile's `required_models` is an
  enforced presence gate (`test_preflight_required_models_are_gateable.py`),
  so a persistent pod volume holding only the post-trained file gets a
  loud, named `PREFLIGHT FAIL` until the provisioner's lane -- repointed in
  the same change -- fetches the base file; a fresh pod fetches it at start.
  `scripts/otr_organ_bench.py` defaults move to the base file at cfg 4.0
  (the old pair was PBUG-20260912-03 itself). Stale comments fixed at the
  engine's recipe block, its receipt (which quoted a publisher "loops" line
  that no card carries), and the preflight's `_CKPT` note.
* **The asset index.** `docs/MODEL_ASSET_INDEX.md` was already drifted on
  HEAD, and the reason was a generator wart: its weight regex harvested the
  literal `"_base.safetensors"` from `endswith(...)` and listed a file that
  does not exist. `scripts/otr_asset_index.py` now skips suffix fragments;
  the regenerated index lists the three real files.
* **Blast radius.** The 5080 renders exactly what it rendered yesterday
  (the base file is on disk; the ladder is unchanged). The 4060, the Mac and
  a pod fetch the small BASE file on a fresh install and otherwise change
  nothing; an existing install keeps its file. `nodes/` is 5080-owned; the
  fetcher and the provisioner profile are the fresh-install path, which the
  4060 owns for proof -- recorded in the handoff so its next drill runs the
  new fetch.

## 8. Reviewers, and what survived

Roster, stated exactly:

* **codex** -- DEAD this session: `codex exec` failed at 08:41 with
  `token_revoked` / "refresh token was revoked". The operator has to sign in
  again; no codex round ran.
* **cursor r1** on the design (this document, first draft): VERDICT
  REFUTED, 14 items, 19 minutes.
* **Opus 5 subagent r1** on the same draft, seated in parallel when cursor
  had been silent ten minutes: VERDICT REFUTED, 11 items, 7 minutes.
* **Finished-diff review**: cursor, briefed to refute -- see the handoff
  receipt for its verdict and what was folded.

Dispositions, grounded against the files (the driver remains the judge):

| item | verdict | disposition |
|---|---|---|
| "Medium ships" was only true on this box (cursor 1, 9; Opus 3) | grounded | Accepted: the ladder no longer flips; medium is a section-3 ruling. |
| Existing installs keep the post-trained file, the 4060 included (cursor 2) | grounded | Accepted as a scope boundary: a deliberate prior ruling, one command to upgrade, recorded for the 4060. |
| Small-base card says "Small Music", not "Medium" (cursor 3) | grounded | Fixed in section 2. |
| The provisioner already forks image lanes on `low_vram` (cursor 4) | grounded, `otr_provision.py:1749` | Named in section 6.3 as the mechanism if medium is chosen. |
| A manifest assertion against a live `resolve_ckpt` would see medium (cursor 5) | grounded | The new test mocks an empty disk and asserts the named constant. |
| The index harvests every filename in the engine (cursor 6, Opus 7) | grounded | Ladder unchanged, no medium row; the harvester's suffix wart fixed. |
| ACE clipped total inflated by an extra house file (cursor 7) | partly | The `_00002` file is the arm's; the smoke is `_00001`. Totals are the arm's n=3. The pair disagreeing became the non-determinism finding. |
| The record is one Markdown file (cursor 8, Opus 11) | grounded | Scripts and JSON copied beside this document. |
| Patch list understated (cursor 10; Opus 5) | grounded | Post-trained row kept; stub rewritten; `_CKPT` comment fixed; bench cfg default fixed. |
| ACE VRAM wording, cfg, wall gain (cursor 11) | grounded | Reworded in 5.2. |
| Receipt comment carries the false "loops" quote (cursor 12) | grounded | Comment rewritten. |
| The held recipe is OTR's, a confound the recipe row owns (cursor 13) | grounded | Stated; the recipe row carries a clause. |
| Runpod presence gate; cross-box (cursor 14; Opus 4) | grounded | Lane and profile repointed together; the persistent-volume consequence and ownership recorded. |
| Bench clipping is the save clamp; production limits the float (Opus 1) | grounded, `stable_audio_theme.py:333`, `:480` | Accepted; the column is footnoted and the argument dropped. |
| Pulse cannot see a 0.25 s period; loopiness absent from the tables (Opus 2) | grounded | Loopiness column added; the sustained claim restated as a proxy; twelve of twelve, not nine. |
| A medium-only box could not self-heal (Opus 6) | **refuted**, `_otr_visual_assets.py:374-378` | `add()` consults the manifest only when the file is absent; a present file needs no spec. Moot anyway. |
| HEAD is red with two unregistered failures (Opus 9) | grounded | Both fixed here: the stub, and the index regenerated from a corrected harvester. |
