# r3 judgment -- Prompt v3, wiring round

Round: r3 (wiring). Reviewer seat: **Cursor** (`cursor-grok-4.6-high`, ask mode).
Driver: Claude (Cowork, 5080). Verdict returned: *yes-with-fixes*.

**Every must-fix is accepted, and the first one is the important one: it found a
hole in Half A that would have made the whole experiment read as a null result.**

---

## 1. MF1 -- the coat is in the LEAF, not only in the motif. CONFIRMED, and it
## changes what Half A composes

Cursor's claim: Half A drops `motif_cue` but keeps `drawable_beat`, and the
stored leaves themselves carry the costume, so the picture would not change.

Checked against the stored objects of "The Faded Ledger":

| stored `motif_cue` | stored `drawable_beat` |
|---|---|
| a lean figure in a **charcoal coat**, carrying a **satchel** | a figure in a **charcoal coat** holds a **satchel** out |
| a tall figure in a **black shawl**, carrying a ledger | a figure in a **shawl** holds a ledger toward a desk |

**CONFIRMED.** The leaf is the motif with a verb attached, which is exactly what
the author's own rule 8 asks it not to do and exactly what it does anyway,
because the motif is the only content the author was given. Dropping the motif
alone leaves the coat in the frame.

**The operator's rewrites settle this, and they settle it harder than either
reviewer did.** Across all ten rewrites in `operator_rewrites.md` he kept **none**
of the three composed slots:

* the **costume motif** -- gone from all ten (his rule 9: *"definitely not some
  coat or figure that is not even mentioned in the dialogue"*);
* the **authored leaf** -- gone from all ten, replaced by the world's own motion
  (his rule 3);
* the **mode law** -- gone from all ten. Not one rewrite carries *"mid-shot or
  wider, whole figure legible, one clear action, unbroken shot"* (his rule 5),
  and that sentence measures 17-19 tokens, 45% of every prompt.

What he wrote instead, verbatim from the form, is one shape:

    a broadcast radio console with flashing lights, in a storybook engraving style
    a vast cold water reservoir, animated, in a storybook engraving style
    storybook engraving. a mass of organic matter moves with the drift in a large water reservoir
    storybook engraving of GPS global telemetry sensors for the water reservoir and land
    storybook engraving. a bakelite radio set, with a background of British Columbia's
      Williston Reservoir with floating driftwood

**Style, subject, the subject's own motion, the place. Four things, and three of
them come from the ledger's `key_objects` and `story_brief_terms`.**

**So Half A composes `cue + kernel + world-motion + setting`, and it reads
neither `motif_cue` nor `drawable_beat` nor `GHOST_MODE_LAWS_V2`.** Cursor
reached that recommendation from the wiring; the operator had already written it
by hand. Two independent routes, same answer, so it is taken.

The stored fields stay on the row, unread and unrenamed (contract V3). Nothing
about the authored object changes, so replay stays byte-stable and Half B still
has its inputs.

## 2. MF2 -- a budget refusal at plan time would kill the episode. CONFIRMED

`build_request_from_shot` is not only the render path: ShotLock calls it as the
cast-time preflight (`otr_shot_lock.py:1708`), inside a `try` that catches
**only** `DeferredImageGapError`. A `GhostBudgetError` raised there propagates
and plan build dies -- after the writer LLM has already run.

`assert_shell_fits` also proves the v2 shell with a deliberately worst-case
costume (`"costume": "charcoal overcoat", "prop": "briefcase"`,
`ghost_signal_author.py:1417-1425`) through `candidate_fits` ->
`finalize_ghost_prompt_v2`.

**Accepted in full.** `finalize_ghost_prompt_v2` is not edited in place;
`finalize_ghost_prompt_v3` is a new function. v2 keeps its never-trim contract
and its admission role for the stored leaf. v3 **drops to fit and never raises
on a frozen row** -- the valve is the drop order, not a refusal.

## 3. MF3, MF6 -- banana and the receipts. CONFIRMED, both accepted

Banana runs inside the finalizer and the driver then skips the common funnel via
`_ghost_v2_finalized` (`render_driver.py:2877`, `:2949`). v3 applies the route
exactly once, re-measures after it (the transform can grow a token), and the
flag generalises to a version-neutral name.

Thresholds, as Cursor puts them and as the constants confirm: drop toward
`GHOST_AUTHOR_TOKEN_TARGET` (69) **before** banana, refuse at
`GHOST_CLIP_WINDOW_TOKENS` (77) **after**.

Receipts: a new `GHOST_PROMPT_VERSION_V3` constant beside the v2 one;
`GHOST_PROMPT_PROFILE` is NOT bumped (the registry rows key on it);
`prompt_slots` stays a list of names; `prompt_slot_tokens` and `prompt_dropped`
are added to the stamp **and** to the trace allowlist in the same change.

## 4. MF4 -- the kernel must never refuse. CONFIRMED

Brief failure stamps `key_objects: []` and `setting: []`
(`_otr_story_brief.py:419-454`). `resolve_crux_kernel(meta)` therefore lives in
the author module, not the pure composer, and is total: bounded first key object
plus the first setting term; else a bounded `story_brief` slice; else **omit the
kernel slot** and compose cue plus pooled motion. It never raises on a frozen
row, and it never writes into `story_brief_terms` (that would move `brief_hash`
and every seed with it).

## 5. MF5 -- the verifier. CONFIRMED, and it matches the driver's own read

Written independently in anchor section 11 before this review arrived, and
Cursor reached the same place: `actual_request_sha` moves, `seed` and
`render_request_hash` hold, and the A/A check would report FAIL on a correct
A/B. The V6 command is **one published source and one v3 replay**, plus a
sibling `--ab` check asserting equal seeds and unequal prompt shas. The two arms
are never passed as the A/A pair.

Cursor flagged one assumption -- that the Faded Ledger carries
`meta.render_trace`. **It does: eight rows, verified.**

## 6. MF7 -- the plate. ACCEPTED, and DEFERRED

The plate's protected head is already the full `positive_tail` measured against
the 69-token target, and `GHOST_PROMPT_MAX_CHARS` (320) is shared by the video
prompt, the plate and `GhostSignalEngine.prompt_budget_chars`. Cursor's warning
is the decisive one: landing a video change and a plate change in the same GPU
A/B confounds item 2's plate isolation.

**So Half A does not touch the plate and does not raise the shared char
ceiling.** Only the still-in lab peer consumes `plate_prompt`, and the shipping
lane sets `wants_plate_prompt` False, so nothing published is affected. The
plate kernel is revisited after V6.

## 7. Should-fixes

* **SF2 accepted and it matters.** Stored `mode` stays `figure|object|signal`,
  and Half A maps those three onto a VANTAGE clause (wide / object-large /
  hand-or-back) rather than calling `GHOST_MODE_LAWS_V2`. Without that map the
  three modes collapse into one picture.
* **SF3 accepted.** `OTRVideoRenderBatch` has **no `IS_CHANGED`** -- confirmed,
  the grep is empty. A resident GUI session could serve cached v2 clips under v3
  code. The A/B runs on a fresh server per arm (section 4 reset does this
  anyway), and the version constant goes into `IS_CHANGED` so the trap cannot
  fire again.
* **SF1 accepted.** The v2 composer's purity comment says it exists so author
  and render cannot disagree; Half A makes them disagree deliberately, and the
  v3 finalizer's docstring must say so, or the next editor "fixes" it by routing
  v3 back through v2.
* **SF5 accepted, and it corrects the driver.** Cursor is right that
  `include_appearance=False` on the foley and mime lanes exists because the
  joint latent SPEAKS the prompt, not because I2V lanes have a general identity
  rule. `other_lanes_audit.md` section 5 leans harder on redundancy than the
  evidence supports. **Item 3b is amended: add a crux clause beside `appearance`
  on the silent I2V lanes; do not drop `appearance` there on this reasoning
  alone.** The `wan_ti2v` 100-word cap finding stands on its own and is the one
  place where something has to give.
* **SF4 noted** -- moot while the plate is deferred.

## 8. Cuts

All six accepted. Of note: **do not delete `motif_for_character`,
`MOTIF_FALLBACK_POOLS` or `GHOST_MODE_LAWS_V2` in this change** -- ShotLock
authoring and `assert_shell_fits` still call them
(`otr_shot_lock.py:2593`, `:2612-2615`). They are dead to the PROMPT and alive
to the AUTHOR until Half B. R3 of the original contract said delete them; it is
overruled.

## 9. The contract after r3 (supersedes V1-V6 where they differ)

* **W1** Half A composes **`pack cue + crux kernel + world motion + setting`**,
  with a vantage clause mapped from the stored mode. It reads no `motif_cue`, no
  `drawable_beat` and no mode law.
* **W2** `finalize_ghost_prompt_v3` is a NEW function. v2 is untouched and keeps
  its author-time admission role.
* **W3** v3 drops whole optional units to fit and NEVER raises on a frozen row.
  Drop toward 69 before banana; refuse at 77 after; the kernel, the subject and
  the setting are never word-sliced.
* **W4** `resolve_crux_kernel` is total and lives in the author module. Empty
  `key_objects` omits the slot; it never refuses.
* **W5** New version constant, `prompt_slot_tokens` and `prompt_dropped`
  stamped AND allowlisted, `prompt_slots` still a list of names,
  `GHOST_PROMPT_PROFILE` unchanged.
* **W6** No plate change, no `GHOST_PROMPT_MAX_CHARS` change, no other-lane
  change, no deletion of the v2 author constants.
* **W7** The A/B is one published source plus one v3 replay, on a fresh server
  per arm, verified with an `--ab` rule: seeds equal, prompt shas different.
