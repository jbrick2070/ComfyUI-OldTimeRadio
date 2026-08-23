# Ghost Prompt v2 -- build and A/B receipt

**Date:** 2026-08-22
**Branch:** `v2.0-alpha`
**Start HEAD:** `e126520874be0e24fce2deb81c9c51a4c45c9e38` (another window landed
`388bfaaa` on CLAUDE.md mid-session; this work sits on top of it)
**Code commit:** `a8fad82cf14b542ab9e59ca039e251262cbb17ce`
**Plan:** `docs/2026-08-22-GHOST-PROMPT-V2-CONTROLLED-ABSTRACTION-PLAN.md`
**Baton:** `docs/2026-08-22-GHOST-PROMPT-V2-OPUS-HANDOFF.md`
**Model rung:** 5 (Opus, coder window). Design was settled; no panel re-run.

---

## 1. What the A arm proved, in its own bytes

The formal baseline ran the current v1 composer through the real
`workflows/otr_canonical.json` on the official AnimateDiff v3 engine, and
published.

* Episode `signal_lost_disc_of_destiny_20260822_163533`
* `RESULT SUCCESS`, `Prompt executed in 00:22:22`, `obs_publish OK`
* Final media in the LIVE obs tree
* Evidence + hashes: `<episode>\evidence\ghost_prompt_v1_a\SHA256SUMS.txt`

**Every one of its eight prompts was reconstructed and HASH-MATCHED** -- the
composer was loaded from the A-arm commit (`git show e1265208:...`) and
recomposed from the archived ledger, 8/8 by `prompt_sha8`. So these are the
bytes that rendered, not a retelling:

| beat | v1 prompt (excerpt) | defect |
|---|---|---|
| `b002` | `... moves with mali vance demands dr sterling hand, Tense mood, scene, ...` | two cast NAMES in the picture; ends mid-clause; `scene` shipped as a word |
| `b005` | `... moves with gulliver reeves forcefully seizes the shredded, ...` | cast NAME; ends mid-clause |
| `b001` / `b006` | byte-identical (`prompt_sha8=f82c66e0`) | both announcer beats ask for the same picture |
| `music_opening_001` / `music_closing_001` | byte-identical (`prompt_sha8=1e48c63c`) | both music beats ask for the same picture |

Character prompts measured 289--317 characters against the 320 ceiling.

---

## 2. What shipped

One new pure module and seven edited files. `workflows/otr_canonical.json` has
**zero diff** and its git blob is still `c27dff3690030e78d88c3a2607a9ac54fd3935d9`
-- the authored object is internal to the durable `ShotRow`, so no node, socket,
widget or link changed. `pyproject.toml` is untouched (it is a registry-publish
trigger).

* `nodes/_otr_video_engines/ghost_signal_author.py` (new) -- safe projection,
  opaque `g000...` ids, deterministic mode scheduling, the compact recurrence
  motif, strict batch parsing, leaf validation, request/output hashes, the
  deterministic fallback pools, the installed-SD1 token measurer, and the shared
  `finalize_ghost_prompt_v2(...)`.
* `ghost_signal_prompt.py` -- `distill_sigil_components` (shared, byte-stable),
  `compose_ghost_prompt_v2`, `GHOST_MODE_LAWS_V2`, `GHOST_PROMPT_VERSION_V2`,
  and the deletion of the `moves with <six words>` branch.
* `otr_shot_lock.py` -- `_resolve_writer_llm_binding` (the raw message-based
  writer seam), the one Ghost authoring transaction, exact cast-time coverage,
  the durable stamp, and the writer unload + assertion.
* `schemas.py` -- `ghost_prompt: Optional[dict] = None`.
* `render_driver.py` -- the v2 branch ahead of the legacy sigil guard, the
  finalizer's banana receipt, the suppressed second banana pass, the trace
  allowlist.
* `_otr_motion_clause.py` -- `skip_shot` + a LAZY `generate_fn_factory`.
* `otr_video_render_batch.py` -- the capability-based Ghost skip, the uninvoked
  factory, `_assert_writer_released()` before `run_real_episode`.
* `eng_ghost_signal.py` -- `motion_source = "ledger_ghost_drawable_beat"`.

### Decisions worth keeping

* **The leaf is the only model-owned field.** The model never receives
  dialogue, the title, the M4 wall, raw cast prose or a name -- by
  construction, because none of them are parameters of the request.
* **Thirteen hashed keys, template digest included.** Change the instruction,
  its temperature or its output budget and every stored leaf reauthors rather
  than replaying text written under different orders.
* **The window is MEASURED, never asserted.** ComfyUI's SD1 encoder CHUNKS an
  overlength prompt rather than dropping it, so 77 is a salience choice.
  Measured on the installed encoder: 75 payload tokens is 77 total in ONE
  window; 76 spills to two. Padded row LENGTH is never counted. Production
  fails closed if that tokenizer cannot be reached.
* **Bucket greediness is real and is worked around, not hidden.** The sigil
  distiller assigns one source phrase to the FIRST matching bucket, so a row
  reading *"a broad steady man in a charcoal overcoat"* lands whole in the
  silhouette bucket and the costume bucket falls to its pool. The motif
  therefore scans costume THEN silhouette for its colour. The LANDMARK bucket
  is never scanned -- that is where the jaw, brow, scar and hair live, and
  face-adjacent recurrence is what v2 exists to replace.
* **`sci_fi_radio` authors no style cue.** Its `pack_cue` component is
  legitimately empty and must stay empty rather than acquire one. Pinned across
  all nine shipped packs.

---

## 3. Verification

| gate | result |
|---|---|
| Full Windows suite | **12225 passed / 134 skipped / 1 xfailed**, exit 0 (379.6 s) |
| Bug Bible (separate repo) | **22 passed / 26 skipped / 3 xfailed** -- baseline |
| Canonical workflow validator | OK, **23 nodes / 57 links** |
| `build_variants.py --check` | **54 variants / 0 failures** |
| Canonical workflow git blob | `c27dff36...` unchanged |
| Forbidden-symbol sweep | 0 runtime hits |
| BOM / 0-byte / AST | clean on all 12 committed files |
| Focused Ghost + author + wiring tests | 522 passed / 1 skipped |

**New executable coverage:** `tests/test_ghost_signal_author.py` (110 cases)
and `tests/test_ghost_prompt_v2_lane.py` (32 cases).

### The independent review found a real blocker, and it was fixed

One post-coding QA pass on the frozen diff (Sonnet, read-only, grounded against
the Windows files). Nine of its ten areas came back clean. The tenth did not:

> **BLOCKER.** `_assert_writer_released()` sits in a `finally:` INSIDE node 92's
> `except Exception -- never break the render`, which logs the RuntimeError as
> `"motion_clause skipped"` and falls straight through to `run_real_episode`.

Confirmed and fixed: the release now runs OUTSIDE that catch-all, gated on
whether the pass was enabled. **A guard inside the thing that swallows guards
is not a guard**, and the suite was green through it -- so the replacement test
is structural (`test_the_writer_release_is_not_inside_a_catch_all` walks the
AST and refuses the call inside any broad-`except` try).

Two smaller findings, both confirmed and fixed:
* `_ghost_unload_writer` returned early when the unload itself raised, skipping
  the one line that proves anything -- and that is the case most likely to have
  left weights resident.
* A duplicated lazy-load block in `_otr_motion_clause.py`: a patch whose OLD
  text was a SUBSTRING of its own NEW text, so the "already applied?" guard
  matched and applied it twice. Inert, but deleted and pinned.

---

## 4. The three live arms

All three published to the LIVE `otr/obs/` tree. Every leg `RESULT SUCCESS` +
`obs_publish OK`, evidence archived with SHA256SUMS under each episode's
`evidence/` directory.

| arm | episode | prompts | technical model | ghost source |
|---|---|---|---|---|
| **A** | `signal_lost_disc_of_destiny_20260822_163533` | v1 | Mistral-Nemo | -- |
| **B1** | `signal_lost_disc_of_destiny_20260822_171254` | v2 | Mistral-Nemo | `deterministic_fallback` x8 |
| **B2** | `signal_lost_turntables_lament_the_last_spin_20260822_174415` | v2 | gemma-4-12b | **`writer_llm` x8** |

### A vs B1 IS the same-seed A/B, and it is exact

Measured, not assumed (`tmp/ghost_v2_arm_comparison.txt`):

    SAME : voiced_text_sha, episode_seed, style, roles, frames,
           video_seeds, cast, creative_model, negative_sha8
    DIFF : -- nothing --

Same script (`9c1e40b90ff0d5b8`), same eight video seeds, same 250/253/180/196/
110/98/514/200 frame contract, same negative. **The prompt is the only variable**,
which is exactly what the plan asked for. Prompt length fell from 208--317
characters to 164--198, every one measured at 32--43 installed SD1 tokens in a
single window.

### A vs B2 is NOT same-script, and this receipt says so rather than implying it

    SAME : episode_seed, style, roles, video_seeds, cast, creative_model, negative_sha8
    DIFF : voiced_text_sha, frames

The technical slot participates in the writer's structured passes, so pinning it
to gemma changed the script. B2 is therefore the proof of the **LLM treatment**
-- eight rows `source=writer_llm`, empty fallback reason, no replay -- not a
same-script pixel comparison. **A same-script LLM arm is not obtainable without
reverting code**: the v1 content route no longer exists, so there is no way to
re-render a v1 baseline on gemma's script. The A arm was captured before the
change precisely because this was foreseeable.

### The model choice was measured, not assumed

The first live leg exposed something no unit test would have: Mistral-Nemo
answered in a perfectly valid envelope but wrote four-word abstractions
(`signal oscillates, broadcast begins`, `silhouette shreds papers, tension
builds`). The validator rejected both attempts and the batch fell to the
deterministic pools -- machinery correct, instruction too thin.

Two fixes followed. The batch template gained worked examples AT the target
length plus named counter-examples (a word COUNT is a number a model does not
feel). Then both candidates were put to the real batch prompt directly --
two minutes instead of two renders:

| model | leaves accepted by the production validator |
|---|---|
| `google/gemma-4-12b-it` | **8 / 8** |
| `mistralai/Mistral-Nemo-Instruct-2407` | 4 / 8 (3x a hand in `object` mode, 1x dangling tail) |

gemma's leaves, as rendered in B2:

    signal  a jagged waveform pulses and expands across a dark screen
    object  the radio dial rotates and a small emblem glows bright
    object  a charcoal lantern flickers and casts a long jagged shadow
    signal  a cream chart line spikes and vibrates against a grid
    figure  an angular silhouette reaches for a lantern in a dark room
    object  a cream chart emblem spins rapidly on a heavy turntable
    signal  a single radio dial signal fades into a thin horizontal line
    object  the broadcast console emblem pulses once and then goes dark

**OPEN, and it is the operator's call:** `config/profiles/otr_ghost_signal_v3.json`
still pins `technical_model` to Mistral-Nemo. B2 pinned gemma per-leg via
`--set` rather than editing the shipped profile, because the technical slot also
drives the writer's structured passes -- so promoting it changes SCRIPTS on this
lane, and story output is a closed subject. The measurement above is the case
for promoting it; the decision is not a coder's to make silently.

---

## 5. Admissions

* **No new PBUG and no new Bug Bible entry.** The v1 fragment is reproducible
  from the checked-in composer and is now proven against a published render,
  but it belongs to the already-admitted dangling-preservation class
  (`BUG_BIBLE.yaml` 12.127 territory); this sprint added executable coverage
  rather than a new portable rule.
* **NO VRAM CLAIM.** No ceiling was measured or enforced on these legs.
* **The boot-time `basline-models` warning is pre-existing and benign** -- ComfyUI
  warns about every sibling directory under `custom_nodes` that is not a node
  pack, and the VRAM workbench is one. It carries a traceback, which cost one
  false leg-fault alarm during this session; it is not an OTR fault.
