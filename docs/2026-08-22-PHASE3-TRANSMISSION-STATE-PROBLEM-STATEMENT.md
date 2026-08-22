# Problem statement for scoping: phase 3, ledger-driven transmission state

**Date:** 2026-08-22
**For:** an external scoping pass (Codex), then a driver-grounded judgment
**Status:** nothing is built. No schema change is authorised by this document.
**Branch:** `v2.0-alpha` at `930080d5`

---

## 0. The one-line question

Ghost Signal's domain-adapter lane applies a **constant** contamination strength
to every beat. Phase 3 would make that strength **track the story's arc** — clean
when the signal is strong, degraded as narrative pressure rises. Is that worth a
schema change, and what is the right shape for it?

---

## 1. Where this stands after a full day of live evidence

**The adapter lane exists, is live-proven, and the operator prefers it.**
`animatediff15_v3_haunted_video` = the clean v3 lane plus AnimateDiff v3's
optional domain adapter (`v3_sd15_adapter.ckpt`, 97 MB) on the MODEL path via
stock `LoraLoaderModelOnly`. Two independent viewers — the operator and Gemini —
preferred its output over the clean arm. It is Apache-2.0 end to end, which
makes it the first Ghost configuration that could be published.

**But the cross-episode difference is small.** The operator's verdict on the
adapter arm versus the clean arm was *"they are all good and similar."* Four
attempted comparisons today were confounded because each leg writes a different
script; that is now fixable (`OTR_WRITER_SEED`, shipped `930080d5`) but was not
fixed in time for these judgments.

**And that is exactly what makes phase 3 interesting rather than moot.** A
cross-episode comparison asks the viewer to hold a difference in memory between
two videos. Phase 3 is a change that happens *inside a single episode, in front
of the viewer*. A subtle effect that is hard to recall across two files can be
obvious when you watch it move. **Nobody has tested that.**

---

## 2. What is verified in the code (checked, not assumed)

| Fact | Where |
|---|---|
| `arc_phase` is on every frozen ledger line, defaults `"setup"` | validated against `EpisodeBudget.arc_phases` |
| `_ARC_CLAUSES` is already a per-beat table keyed on `arc_phase` | `render_driver.py:1524` |
| Ghost's prompt composer already receives `arc_phase` as `story_accent` | `render_driver.py:2807` |
| ...but it enters as a **raw 48-char string** and is the **FIRST thing trimmed** under budget pressure | `ghost_signal_prompt.py:642`, `GHOST_TRIM_ORDER` at `:574` |
| `speaker_role` is enum-pinned to 5 values | `_otr_ledger_freeze.py:100-107` |
| `extract_beats` **drops** `arc_phase` and `traits` on the way to the video side | `otr_shot_lock.py:598-606` |
| The driver **re-joins** the shot to its frozen line and reads `arc_phase` there | `render_driver.py:2803-2808` |
| `scene_tension` / `tension` are **phantom fields** — read in two places, written by nobody | `_otr_delivery_vector.py:194`, `_otr_voice_node_common.py:1052` |
| There is **no per-beat framing or shot-type field**; framing is decided by role | `GHOST_FRAMING` at `ghost_signal_prompt.py:100` |

**Note the fourth row.** Ghost is *already* arc-conditioned — but weakly, through
a raw word in a text prompt that gets thrown overboard first whenever the beat
runs long. On real episodes (12s beats) that budget pressure is routine. So the
existing arc channel is both weak and unreliable, which is a genuine argument
that a weights-level channel would do something the prompt cannot.

---

## 3. The blocker, stated precisely

`VideoRequest` is `extra="forbid"` (`schemas.py:138`). The request dict must pass
`VideoRequest.model_validate`, so **a per-beat value cannot simply ride along**.

Its one open dict is `observability`, and that field is documented in the schema
as *"Trace-only observability stamps … NEVER conditioning, never hashed into
request identity."* Using it to carry a conditioning value would be a quiet
contract violation, not a shortcut. **It is not a back door and should not be
proposed as one.**

So arc-driven strength requires **one declared field** on a `extra="forbid"`
model. That is a schema change, which is why this is being scoped rather than
built.

---

## 4. THE CHEAP PROBE — and this is the most important section

**A beat-position ramp needs NO schema change.**

`render_clip` already receives `shot_id` on every request (`eng_ghost_signal.py:711`).
An engine can therefore vary its adapter strength across an episode by beat
position — a ramp, a pulse, anything keyed on where the beat sits — **today, with
no `VideoRequest` field and no `ShotRow` change.**

That probe answers the actual open question:

> **Is within-episode variation of the adapter visible at all?**

* If **no** — phase 3 dies with evidence rather than as an opinion, and the
  operator stops wondering. Cost: one leg.
* If **yes** — the schema change is justified by a measured effect instead of a
  hypothesis, and the arc mapping becomes a refinement of something already
  known to work.

**A scoping pass should say whether this probe is the right first step, and if
so, what shape the ramp should take** (monotonic ramp? centre-weighted pulse?
role-aware?) so that a positive result transfers cleanly to the arc-driven
version rather than having to be redone.

---

## 4A. THE OPERATOR'S SIGNAL -- and it supersedes what this document opened with

The operator asked, unprompted: *"how about how emotional the character voice
is?"* That is a better answer than either option in section 5, and the machinery
for it is ALREADY IN THE REPO.

`nodes/_otr_delivery_vector.py` computes an 8-dimension emotion bundle per line
and groups four of them as AROUSAL:

    _AROUSAL = ("angry", "afraid", "surprised", "happy")
    arousal = sum(scores[e] for e in _AROUSAL)

Measured on real dialogue at commit `930080d5`:

| line | arousal |
|---|---|
| "The reels are catalogued and shelved, as they always were." | 0.00 |
| "I think there is something on the tape we were not meant to hear." | 0.00 |
| "It is here. God help us, it is already here." | 0.33 |
| "Get away from the projector! Now!" | 0.40 |
| "Rest now. The broadcast is over." | 0.00 |

**Why this beats `arc_phase` and beats a new LLM-extracted field:**

1. `deterministic_delivery_vector(text)` is a PURE FUNCTION OF THE LINE TEXT. The
   driver can call it at `render_driver.py:2803`, the same place it already reads
   `line["traits"]` and `line["arc_phase"]`. **No new ledger field. No LLM. No
   RNG.** The render path stays read-only and deterministic, which is the
   constraint every other proposal here has to work around.
2. It is PER-LINE and DRAMATIC. `arc_phase` is per-act and structural: a quiet
   line inside a climax scores the same as a scream. Arousal does not.
3. It IS the house pattern rather than a new one. `_otr_delivery_vector` exists
   to derive a numeric bundle per line and hand it to an engine, already carrying
   `DELIVERY_TABLE_VERSION` for exactly the "detect a changed table" reason
   section 5 asks about.
4. **The voice and the picture would be driven by the SAME NUMBER.** The image
   destabilises precisely when the performance does, by construction rather than
   by tuning. On a show whose fiction is a haunted broadcast, that is the whole
   idea in one wire.

**What a scoping pass should settle about it:**

* **The mapping curve.** Observed arousal on ordinary dialogue tops out near
  0.40, so a raw pass-through would never approach full strength. Linear
  rescale, gamma, or a small preset ladder? This is tuning, not architecture,
  but it should be decided once and frozen like every other recipe value.
* **Smoothing across beats.** Arousal is per-line and can swing hard between
  adjacent lines. Does the picture follow it beat-to-beat, or through a filter
  so contamination drifts rather than flickers? A 12-second beat next to a
  0.40 spike is a very visible cut.
* **Whether `scene_tension` should be revived.** `deterministic_delivery_vector`
  already accepts a `scene_tension` argument that no producer writes -- it always
  resolves to 0.0 (see the phantom-field row in section 2). Feeding it would
  change arousal itself, which affects the VOICE too. That is a bigger blast
  radius than phase 3 and should probably stay out of scope.
* **It still needs the same one field on `VideoRequest`.** Nothing about this
  removes section 3's blocker: the derivation gets cheaper and better, but the
  driver-to-engine hop is unchanged.

Section 5 below is retained as the fallback if arousal is rejected.

---

## 5. The design fork the panel should settle

**If the probe succeeds and a schema field is warranted — which field?**

**(a) Raw `arc_phase: Optional[str]` on `VideoRequest`.** The engine owns the
mapping from phase to strength.
- Other engines could use the arc later for their own purposes.
- But engines start knowing about story structure, which they currently do not.

**(b) Derived `transmission_state` (name + float) on `VideoRequest`.** The driver
derives it at the existing re-join point and the engine just consumes a number.
- Matches the house precedent: `_otr_delivery_vector` derives an 8-dimensional
  numeric bundle per line in pure Python from keyword cues, stamps it after
  freeze with a table version guard (`DELIVERY_TABLE_VERSION` at `:28`), and
  hands it to an engine. No LLM, no RNG.
- Keeps the engine dumb, which is how every other engine here works.

The driver's current lean is **(b)**, on precedent. The panel should challenge
that rather than ratify it.

**Sub-questions worth settling in the same pass:**
1. Does the value carry a **version constant** like `DELIVERY_TABLE_VERSION`, so
   a re-render under a changed table is detectable rather than mysterious?
2. What does the **receipt** carry — preset name, resolved float, table version?
   The clip receipt already carries `domain_adapter` and
   `domain_adapter_strength` as of `92317a7d`.
3. Is there a case for keying on something **other than `arc_phase`**? Note the
   phantom-field trap: `scene_tension` looks like the natural key and is written
   by nobody.

---

## 6. The face-continuity clamp is not optional

The operator has ruled that a character's face changing between beats is a
**correctness defect**, explicitly carved out as still-open even under the
"story quality is done" directive. Heavy adapter strength changes the image
domain, which will move faces.

Any proposal must clamp `character_video` and `announcer_visual` to a light
setting regardless of arc, leaving `music_visual` free to take the full range.
Role is already a real field on `VideoRequest`, so **the clamp costs nothing and
needs no schema change** — it is available even in the cheap probe.

---

## 7. Constraints any answer must respect

1. **The canvas does not move.** 512x288 render, 1920x1080 delivery. Operator
   ruling, direct: *"i don't want to mess with canvas."*
2. **The golden lane is never modified.** It renders the dailies.
3. **Additive as a peer**, with per-lane constants as CLASS attributes a sibling
   can override — preflight G1.3. This rule was violated three separate times in
   one day (a byte floor, a module name, the hold-2 cadence functions), so it is
   the single most likely way a new lane goes wrong here.
4. **`workflows/otr_canonical.json` is not modified.** If a stage discovers it
   must be, stop and re-plan.
5. **Every leg publishes to `otr/obs/`.** A leg that does not reach it did not
   pass.
6. **No LLM on the render path.** If a prose-driven nudge is proposed at all, it
   follows `_otr_motion_clause.py`: a separate batch pass, env-gated off by
   default, with a static non-LLM fallback, leaving the render path read-only and
   deterministic.

---

## 8. What would make phase 3 NOT worth doing

Stated up front so the answer is not motivated:

* The cheap probe shows within-episode variation is invisible.
* The role clamp turns out to bind so tightly that only music beats can move,
  making the arc a curiosity rather than a look.
* The effect is real but reads as inconsistency rather than intent — the risk
  with any per-beat visual change on a show whose beats average 12 seconds.

---

## 9. What is explicitly NOT being asked

Not asked: whether the adapter should exist (settled — it ships, at 1.0), whether
the canvas should change (no), or whether prose should drive anything (deferred).
The question is narrowly: **is arc-driven strength worth a declared schema field,
what shape should it take, and is the no-schema beat-position probe the right way
to find out first.**
