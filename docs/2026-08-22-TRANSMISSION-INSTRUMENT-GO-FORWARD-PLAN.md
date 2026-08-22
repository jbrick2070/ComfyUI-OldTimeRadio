# Go-forward plan: the Transmission Instrument

**Date:** 2026-08-22
**Branch:** `v2.0-alpha`
**Status:** proposal for review. Nothing is built. No graph change is authorised
until the Stage 2 review gate below has been passed.
**Scope:** Ghost Signal video lane only. `otr_canonical.json` is untouched by
every stage in this document.

---

## 1. The proposal

AnimateDiff v3 ships an optional companion to its motion module,
`v3_sd15_adapter.ckpt` -- a LoRA applied to the **image model** (SD1.5), not to
the motion module. Its authors describe it as a domain adapter that exists to
absorb the training set's own visual defects, removable at inference or
integrated with an adjustable scalar.

We propose to use it as a **third creative authority** in the Ghost Signal lane,
alongside the two that already exist.

| Authority | Controls | Where it lives today |
|---|---|---|
| Visual-style pack | What the world is made of | `nodes/_otr_visual_styles.py`, episode-level |
| Ledger | What happens in the beat | `ledger["lines"]`, per beat |
| Domain adapter | How intact, or haunted, the transmission feels | proposed, per beat |

Ghost Signal's fiction is a haunted broadcast. The adapter is a dial from "none
of the training set's grime" to "all of it". Driven off the ledger's existing
narrative-arc field, that dial gives an episode a visual arc that tracks its
story arc -- clean when the signal is strong, degraded as narrative pressure
rises -- with no new model, no new measurement system, and no mandatory LLM pass.

The intended outcome is a second show personality, not a defect repair. The
existing lane remains the clean reference and is not modified.

### What makes this cheap

The derivation pattern already ships in production, twice:

* `render_driver.py:1524` `_ARC_CLAUSES` is a per-beat table keyed on
  `arc_phase` returning a value. It returns a string; we need a float.
* `nodes/_otr_delivery_vector.py` derives an 8-dimensional numeric bundle per
  line in pure Python from keyword cues, stamps it on the ledger after freeze
  with a table version guard, and feeds it to an engine. No LLM, no RNG.

Every ledger field the derivation needs already exists and is already populated
on every beat.

---

## 2. Ledger fields this plan consumes

All read-only. No field is added to `ledger["beats"]`, `ledger["lines"]`, or the
outline `Beat` model.

| Field | Row | Type | Guarantee |
|---|---|---|---|
| `arc_phase` | `ledger["lines"]` | str | Always present; defaults to `"setup"`; validated against `EpisodeBudget.arc_phases` |
| `speaker_role` | `ledger["lines"]` | str | Enum-pinned in `nodes/_otr_ledger_freeze.py:100-107`: `character`, `announcer`, `music_open`, `music_close`, `music_inter` |
| `role` | video shot row | str | Mapped from `speaker_role` by `otr_shot_lock.py:57-64` into `Role`: `character_video`, `announcer_visual`, `music_visual` |
| `traits` | `ledger["lines"]` | str/None | Free-text mood prose. Optional input, Stage 3 only |
| `beat_intent` | `ledger["lines"]` | str | Free-text intent prose. Optional input, Stage 3 only |

Two constraints the implementation must respect:

* **There is no per-beat framing or shot-type field.** Framing is decided by
  `role` via `GHOST_FRAMING` in
  `nodes/_otr_video_engines/ghost_signal_prompt.py:100-104`, and the lane
  deliberately strips camera and framing words from its prompt source
  (`ghost_signal_prompt.py:243-244`). Face protection is therefore expressed as a
  clamp on the `character_video` role, never as a close-up test.
* **`scene_tension` and `tension` are phantom fields.** They are read in
  `_otr_delivery_vector.py:194` and `_otr_voice_node_common.py:1052`, but no
  producer in the repo writes either key, so both always resolve to `0.0`.
  Nothing in this plan may key on them.

`traits` and `beat_intent` are prose, not enums -- a live example already
documented at `render_driver.py:1543-1550` is the intent string
`"open the episode and orient the listener."`. Any mapping over them must have an
unmapped-key fallback, as `_INTENT_CLAUSES` does.

---

## 3. Stage 1 -- Prove the mechanism

**Purpose:** establish that the adapter produces a visible, monotonic, motion-safe
effect before any repo change is proposed. This is a measurement. Nothing ships
from it.

**Review gate:** none required. There is one verifiable right answer.

### Work

1. Fetch `v3_sd15_adapter.ckpt` into `models/loras`.
2. In a throwaway copy of the Ghost Signal internal graph, insert a
   `LoraLoaderModelOnly` between the checkpoint's `MODEL` output and the
   AnimateDiff loader. CLIP and VAE keep their existing direct paths, untouched.
3. Render a fixed seed and a fixed card-heavy prompt (a beat with a sign, a dial
   or a poster) at `strength_model` values `0.0, 0.25, 0.5, 0.75, 1.0`.
4. Repeat across three seeds so seed noise is separable from adapter effect.
5. Render into a **new** output directory. Never re-render in place -- the
   resulting stills are cited by sha256 in the receipt below.

### Acceptance

All three must hold:

* **Visible.** `0.0` and `1.0` are distinguishable at a glance on the same seed.
* **Monotonic.** `0.5` reads as between its neighbours on all three seeds. If it
  behaves randomly, this is seed noise, not a dial.
* **Motion-safe.** Motion quality at `1.0` is not visibly worse than at `0.0`.
  The adapter is on the image model; if motion degrades, the mechanism is not
  what the upstream documentation describes.

Any one failing ends the whole initiative. Record the verdict, leave the repo
unchanged.

### Deliverable

A short results note under `docs/`, listing each still by path and sha256, the
strength that produced it, and the three verdicts above. This note pins the
frozen strength value used by Stage 2.

---

## 4. Stage 2 -- The haunted sibling lane

**Purpose:** ship a working second lane at a single frozen strength. No ledger
derivation yet.

**Review gate -- MANDATORY BEFORE ANY CODE.** This stage amends a test-pinned
graph contract and adds a lane, which is a design item with more than one
defensible answer. It gets a full four-round `kibitz-plugin:kibitz` arc
(r1 arc, r2 coding, r3 wiring, r4 convergence) with a code-grounded
`driver_anchor.md` written first. The open design questions the panel must
settle are listed in section 4.4.

### 4.1 The lane

| | Value |
|---|---|
| New engine id | `animatediff15_v3_haunted_video` |
| Untouched reference lane | `animatediff15_v3_video` |
| Untouched golden lane | `mm-p_0.5` -- has no matching adapter; operator condition stands |
| Artifact | `v3_sd15_adapter.ckpt` |
| Artifact location | `models/loras` |

### 4.2 The internal graph

```
Checkpoint --MODEL--> LoraLoaderModelOnly --MODEL--> AnimateDiff loader --> KSampler
Checkpoint --CLIP---------------------------------> (existing path, unchanged)
Checkpoint --VAE----------------------------------> (existing path, unchanged)
```

* Loader class: `LoraLoaderModelOnly`
* Loader inputs: `model`, `lora_name`, `strength_model`
* Loader output: `MODEL` only
* Resulting contract: **nine nodes, eleven links** (the reference lane's eight
  and ten are unchanged)

The model lifecycle now has three states -- base, adapter-patched, and
AnimateDiff-patched. Whatever tracks model identity for cache and receipt
purposes must distinguish all three.

### 4.3 Behaviour when the adapter file is absent

**Fail closed.** If `v3_sd15_adapter.ckpt` is not present in `models/loras`, the
haunted lane raises a clear error naming the missing file and the expected
directory. It must never silently fall back to the clean lane -- a receipt that
says "haunted" over clean output is a hole in the record, and the operator reads
receipts to know what he is looking at.

### 4.4 Design questions for the review panel

The implementer should not settle these alone:

1. Is the sibling a distinct engine module, or a parameterised variant of the
   existing v3 engine? Both are defensible; the duplication cost and the
   divergence risk pull in opposite directions.
2. Does the existing eight-node/ten-link test contract get amended to cover both
   lanes, or does the sibling get its own separate contract?
3. Does the frozen strength live in the engine recipe (the `eng_fastwan_8gb.py`
   pattern, `lora_strength` with an `OTR_*` environment override) or somewhere
   the ledger can later reach without a second refactor? Stage 3 must not require
   undoing this choice.
4. What is the receipt's exact shape, given Stage 3 will add a per-beat value to
   it later?

### 4.5 Receipt

Every haunted-lane render records, at minimum:

* adapter filename
* `strength_model` actually applied
* lane / engine id
* the existing render request hash

### 4.6 Tests

* Graph contract test for the sibling lane: nine nodes, eleven links, and the
  loader sits between checkpoint and AnimateDiff on the MODEL path only.
* CLIP and VAE paths assert unchanged.
* Reference lane contract test still passes untouched.
* Missing-adapter case raises, with the filename in the message.
* Existing Ghost Signal peer tests
  (`tests/test_ghost_signal_peers.py`) stay green.

### 4.7 Acceptance -- live proof required

A green unit suite does not close this stage. Required:

* A full leg through `workflows/otr_canonical.json` on the haunted lane.
* `RESULT SUCCESS` and `obs_publish OK` in the leg log.
* The published episode present in `otr/obs/`.
* The asset confirmed on disk at its canonical path under `otr/episodes/<ep>/`.

---

## 5. Stage 3 -- Ledger-driven `transmission_state`

**Purpose:** replace Stage 2's frozen constant with a per-beat value derived from
the ledger. This is the smallest stage. It adds no schema field.

**Review gate:** one clean finished-diff review by an independent reviewer,
grounded against the real files by the driver. A full arc is not required --
Stage 2 settled the design; this is a table and a clamp.

### 5.1 The presets

Named presets, each resolving to a frozen `strength_model` float. The continuous
value remains available underneath for tuning; the names are what the ledger
schedules.

| Preset | Meaning |
|---|---|
| `clear_channel` | Adapter bypassed. Pristine v3. |
| `tape_ghost` | Faint contamination, aged-video texture. |
| `signal_bleed` | Stronger degradation as narrative pressure rises. |
| `broadcast_possession` | Full strength. The climax. |

The exact float behind each name is chosen from the Stage 1 results note and
frozen, in the same way every other recipe value in this lane is frozen.

### 5.2 The derivation

Pure Python. No LLM. No RNG. Deterministic for a given ledger.

1. **Base lookup on `arc_phase`**, keyed exactly as `_ARC_CLAUSES` at
   `render_driver.py:1524` already is. Suggested starting map, to be confirmed by
   eye against real output:

   | `arc_phase` | preset |
   |---|---|
   | `setup` | `clear_channel` |
   | `rising` | `tape_ghost` |
   | `climax` | `broadcast_possession` |
   | `falling` | `signal_bleed` |
   | `resolution` | `tape_ghost` |

   Any unmapped or missing value falls back to `clear_channel`.

2. **Role clamp, applied after the lookup.** This is the face-continuity
   protection and it is not optional:

   | `role` | rule |
   |---|---|
   | `character_video` | clamped -- never exceeds `tape_ghost` |
   | `announcer_visual` | clamped -- never exceeds `tape_ghost`; restrained archival patina |
   | `music_visual` | unclamped -- music and transition beats may take the full range |

3. **Result** is one preset name plus its float, per beat.

### 5.3 Where it is computed

In `render_driver.py`, at the point where the driver **already** re-joins the
shot to its frozen line row -- `render_driver.py:2803-2808`, which reads
`line["beat_intent"]`, `line["traits"]` and `line["arc_phase"]` for
`compose_ghost_prompt`. Every field the derivation needs is in scope there.

This is deliberate and is the reason the stage is cheap:

* **No `ShotRow` change.** `ShotRow` is `extra="forbid"`
  (`nodes/_otr_video_engines/schemas.py:346-366`), so any new key would have to be
  declared. Deriving at the join point avoids that entirely.
* **No change to `extract_beats`** (`otr_shot_lock.py:598-606`), which drops
  `arc_phase` and `traits` on the way to the video side. The existing re-join is
  the sanctioned workaround and we follow it rather than widening the beat.
* The render path stays read-only and deterministic.

### 5.4 Versioning

The preset table carries a version constant, following
`DELIVERY_TABLE_VERSION` in `_otr_delivery_vector.py:28`. The receipt records the
version alongside the value, so a re-render under a changed table is detectable
rather than mysterious.

### 5.5 Receipt

Stage 2's receipt gains, per shot:

* `transmission_state` preset name
* the resolved `strength_model` float
* the preset table version

### 5.6 Tests

* Every `arc_phase` in `EpisodeBudget.arc_phases` maps to a known preset.
* An unknown or missing `arc_phase` falls back to `clear_channel` and does not
  raise.
* `character_video` and `announcer_visual` are clamped even when `arc_phase` is
  `climax`.
* `music_visual` at `climax` reaches `broadcast_possession`.
* The derivation is deterministic: same ledger in, same values out, twice.
* The clean reference lane is unaffected by the presence of the table.

### 5.7 Acceptance

A live leg, published to `otr/obs/`, whose receipt shows more than one distinct
`transmission_state` across the episode's beats. A single value throughout means
the derivation is not actually reading the ledger.

---

## 6. Stage 4 -- Optional prose nudge

**Purpose:** let a beat's authored mood push its contamination above or below what
`arc_phase` alone would give it. This is the only place an LLM appears anywhere
in this plan.

**Ships disabled.** Environment-gated off by default. The Stage 3 deterministic
table is the floor and remains fully functional with this stage absent, disabled,
or failing.

### The pattern to copy

`nodes/_otr_motion_clause.py` is the precedent and should be followed closely:

* a separate batch pass, not an inline call on the render path
* `FLAG_ENV`-gated, off by default (`_otr_motion_clause.py:33`)
* `compute_source_hash` invalidation (`_otr_motion_clause.py:56`)
* a static, non-LLM fallback when generation is unavailable
  (`GENERATED_MODEL_UNSET = "static-role-map"`, `_otr_motion_clause.py:43`)
* the render path stays read-only and deterministic

### Behaviour

The pass reads `traits` and `beat_intent` for each beat and returns a bounded
nudge -- one step up or down the preset ladder, never more. The role clamp from
Stage 3 is applied **after** the nudge and always wins, so a prose cue can never
un-protect a character or announcer beat.

The receipt records whether the nudge was applied, and which model produced it.

---

## 7. Dependencies and stopping conditions

* **Stages 2, 3 and 4 depend on v3 remaining in contention in the video bakeoff.**
  If the bakeoff retires v3 on looks, the haunted sibling retires with its parent
  and this plan closes.
* Stage 1 failing on any of its three acceptance criteria ends the initiative.
* Stage 3 may not begin until Stage 2 has a live leg published to `otr/obs/`.
* Stage 4 may not begin until Stage 3 has a live leg published to `otr/obs/`.
* The golden lane is never modified by any stage.
* `workflows/otr_canonical.json` is never modified by any stage. If any stage
  discovers it must be, stop and re-plan.

---

## 8. Standing rules that apply to every stage

* One branch: `v2.0-alpha`. Commit and push together, same session, every green
  chunk.
* Regression suite plus the Bug Bible run after every code change.
* UTF-8, no BOM. AST-parse touched Python before declaring done.
* Every render leg ends in `otr/obs/`. A leg that does not reach it did not pass.
* Evidence stills are cited by sha256 and never re-rendered in place.

---

## 9. Attribution

Upstream supports treating `v3_sd15_adapter.ckpt` as an optional, scalable
UNet-domain adapter. The storytelling interpretation in this document is
intentionally ours.

* AnimateDiff technical explanation:
  https://github.com/guoyww/AnimateDiff#technical-explanation
* Official adapter:
  https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_adapter.ckpt
