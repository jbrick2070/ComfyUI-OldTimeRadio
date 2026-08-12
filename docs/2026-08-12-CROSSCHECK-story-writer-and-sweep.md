# CROSS-CHECK BRIEF -- the story-writer sprint and the 21-lane sweep, 2026-08-12

**For an independent reviewer (Antigravity).** This is not a summary to agree
with. Every section below is a CLAIM with a file, a line and a commit, and the
job is to confirm or refute it. Where I already know I am uncertain, I say so.

**Ground rules for the cross-check**
- Use the REAL Windows files at
  `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.
  Do NOT use a Linux sandbox mount -- it lags and shows stale copies, which has
  produced phantom "corruption" findings on this repo before.
- Python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`,
  `$env:PYTHONUTF8=1`.
- **A GPU campaign is running.** Review only; do not edit, and do not run the
  full suite (it is CPU-heavy and the box is busy).
- Label every finding CONFIRMED / REFUTED / UNVERIFIABLE with the line you read.

---

## 1. WHAT SHIPPED TODAY, and what each fix CLAIMS

All on `v2.0-alpha`. Suite at the end: 10273 passed / 110 skipped / 1 xfailed;
`build_variants --check` 50 variants, 0 failures.

| commit | claim |
|--------|-------|
| `3a5cf77f` | PBUG-03: the stage-direction repair note could not fire on its own defect class -- it matched only `BAD_LINE_SHAPE` opening `(`/`[`, while the live defect was `UNKNOWN_SPEAKER: *SFX` |
| `9d03cba9` | PBUG-02 root fix: `CastShape.register` collided with a metaclass attribute, so pydantic adopted it as the field DEFAULT -- the field went optional, **the schema handed to the writer stopped requiring it**, and an omitted value reached the prompt as a bound-method repr |
| `bf1d02a1` | a cross-bank writer gate that PINS the bank instead of rolling it |
| `39b29d0f` | PBUG-04: a live `VisualStyleCardModel` rode `meta.update()` into the ledger; plus `describe_execution_error`, replacing a `str(messages)[:500]` truncation that cut mid-traceback |
| `98fb258f` | `num_characters` is a REQUEST, not a cap (operator directive, all banks) |
| `2572b493` | **the repair turn now carries the rejected draft.** It never did |
| `45d1d3f8` | the one-shot `format_example` path was DEAD CODE; now a validated gardening-programme example |
| `1eba7ab3`, `8a7a4d62` | the deal receipt could silently not persist; fix B silently ate part of fix A's budget |
| `f9b51675` | the sweep can pin its source bank |
| `61ae356c` | required ledger saves REFUSE instead of continuing silently |

### The single most important claim to check

**`2572b493` -- the repair ladder never sent the rejected draft back.** The
retry turn was `base_user` + *"Repair only the malformed FORMAT defects below...
Keep the same story, cast, events, and wording wherever the format is already
valid"* + the defect list. The draft appeared nowhere. So every attempt after
the first was a COLD REGENERATION against complaints about an invisible text --
which is why four attempts produced four DIFFERENT malformed shapes instead of
converging.

Supporting evidence I consider strong: `_MARKUP_LADDER_TEMPS` decays to 0.30 and
its own docstring calls that rung *"repeats 0.30 WITH THE DEFECT QUOTE"*. The
temperature was tuned toward determinism FOR a repair context the code never
supplied.

**Verify:** read `_run_markup_ladder` in `nodes/_otr_scifi_fable2.py` at the
commit BEFORE `2572b493` and confirm the draft is absent. Then confirm it is
present now, on the correct turn, in both the `format_example is None` and
non-None message branches.

---

## 2. CLAIMS I AM LEAST SURE OF -- attack these first

### 2.1 "Fix A alone was not enough" (moderate confidence)

The sweep's server carries fix A and the deal receipt but NOT fix B (server
booted 11:41; fix B committed 11:57). Under that configuration `ltx_8gb` rolled
`scifi_news_pro` and the ladder still exhausted:

    UNKNOWN_SPEAKER: DR. MENA PATEL (line 5)

**I claim** this is evidence that carrying the draft alone does not stop an
exhaustion. **I do not claim** anything about A+B together, because B is not in
that server. **Verify:** is my boot-time reasoning right? Is there any way that
server DID have fix B?

### 2.2 The cast-coverage seam (LOW confidence -- Codex already broke my version)

`mesh_stage` died because a cast member (MARIA) had no line; the freeze cascade
hard-fails that, globally, for every bank. I proposed a surjectivity check at
assembled-outline level.

**Codex refuted the retry premise:** `generate_outline()` completes every Stage-2
and Stage-3 call, assembles at `_otr_outline.py:1953`, and its post-combine
checks only RAISE `OutlineFailedError` at `:1964-1985`. There is no retry there,
so "the same retry path a cast-membership miss already takes" was WRONG. Codex
places the check after the Stage-2 loop at `:1862`, before any Stage-3 call.

**Verify Codex, not me.** Specifically:
- Is `:1862` really before all Stage-3 calls?
- Codex says a deterministic sorted round-robin fallback already picks story
  speakers at `:1303-1335` and `:1839-1861`, which would CONTRADICT the
  project's "no deterministic Python deciding story" rule and could satisfy any
  coverage check without a model ever choosing. **Is that real?**
- Codex says outline coverage is necessary but INSUFFICIENT, because
  `compose_line_draft` (`_otr_line_composer.py:977-1046`) accepts any nonempty
  string while `clean_spoken_text` (`_otr_ledger_cleanup.py:251-267`) can later
  empty it into `skip=True`, and the freeze then rejects the character
  (`_otr_ledger_freeze.py:575-593`). Confirm that chain end to end.

### 2.3 The `fastwan_8gb` failure (LOW confidence -- barely diagnosed)

    RenderError: still-spine handoff missing materialized scene still
    for shot shot_music_closing_001 beat music_closing_001 engine fastwan_8gb

`music_closing_001` is a SYNTHETIC closing-beat still target reserved
optimistically at `nodes/otr_meta_brief_image_prompt.py:1234`, whose own comment
says *"an unused still is cheap; a missing still reaches video dispatch too late
to repair safely"* -- so this exact failure was anticipated and the guard did
not hold. The error says missing **materialized**, so I suspect the row exists
and the file does not.

**8 of 13 heavy engines require a scene still** (measured via
`render_driver._still_spine_requires_scene`): `fastwan_8gb`, `ltx_8gb`,
`ltx_audio_in`, `wan_i2v`, `wan_ti2v`, `mesh_stage`, `minimax_h3_video`,
`minimax_h3_audio_in`. If this is systemic, it takes most of the remaining
sweep.

**Verify:** is the target planned-but-unmaterialized, or never planned? What
decides `include_synthetic_closing`? Why did the 21 per-lane smoke receipts pass
if this is systemic -- different beat topology, or a real regression?

---

## 3. CROSS-BANK FINDINGS -- confirm the ones that matter

Every story fix today landed in `_otr_scifi_fable2.py`, and **`scifi_news_pro`
is the ONLY bank routed to that writer** (`banks.json`:
`scifi_news_pro_multipass`; `grep -rl "TITLE:" nodes/story_packs/` returns
exactly one file). So none of them touched the writers the other banks use.

Codex's cross-bank sweep found these; **each is a claim to verify**:

1. **The legacy line composer has fix A's defect class.** Every attempt in
   `compose_line_draft` reuses unchanged `messages`
   (`_otr_line_composer.py:1011-1046`) and never sends the rejected line or the
   reason back. **If true this is the same cold-regeneration bug in a writer
   five banks use.** Highest-value item in this document.
2. **`_otr_scifi_codex.py` has NO `led.save()` at all** (I confirmed: `grep -c`
   returns 0). Receipts accumulate in memory and the ledger is not assembled
   until `:3215`, so a P3/P5 death loses every accepted-stage receipt.
3. **`scifi_news` ALREADY closes the coverage asymmetry** -- per-beat membership
   at `_otr_scifi_codex.py:881-885`, the converse raising
   `RadioScoreDraftCompileError(code="cast_coverage")` at `:945-976`, inside a
   fresh-candidate loop at `:2312-2328`. If so, that writer is the model the
   legacy one should follow, and must NOT be modified.
4. **My "examples are empty" claim was WRONG.**
   `media_restoration_adventure.json:19`, `faithful_radio_adaptation.json:19`
   and `folger_scene_adaptation.json:19` are POPULATED; only
   `original_radio_drama.json:21` is empty. All are inert because
   `_otr_story_pack.py:155-159` type-checks them so. **Confirm they are inert**,
   so nobody "fixes" deliberately dead data.
5. **`custom_source_bank` is `"runnable": false`** -- not a shipping writer.
6. Unchecked `led.save()` remain at `OTR_LedgerScriptWriter.py:4774, 5759, 5896,
   5945, 5990`. I fixed only the two in the fable2 module (`61ae356c`).

---

## 4. WHAT I DELIBERATELY DID NOT DO, and why

- **The candidate-retirement ladder (lesson 35).** Judged WAIT. All three dead
  legs predate fixes A and B, and a judging pass found that the *alternate
  producer slot* every lane assumed -- Fable proposed it, I accepted it, Codex
  and Antigravity both wrote MUST-FIX items for it -- **does not exist**:
  `_ALLOWED_SLOTS = ("creative", "technical")` at
  `OTR_LedgerScriptWriter.py:558`, and `repair_slot_fn` is never passed anywhere
  in the fable2 module. **Verify that.** If it is true, four reviewers built on a
  resource none of them checked for.
- **Adding the cast-coverage rule to four pack prompts.** The natural seam,
  `outline_phase_system`, plans ONE phase and structurally cannot enforce an
  episode-level invariant. Codex independently agreed and added that the packs'
  `examples` are inert anyway.
- **Upstream address-shaped roster names** (Fable's r1 proposal). Both mechanical
  lanes rejected it: unproven causal theory, and `CastShape.name` is the ledger
  join key into casting, credits, portraits and voices.

---

## 5. THE HARD CONSTRAINT every proposal must pass

**Operator rule: the writer MUST fill the ledger completely for downstream
consumers. They read FIELDS, not intentions. A hole in the ledger is a broken
render.**

Grounded: the speaker string is the JOIN KEY across `lines[].speaker`,
`beats[].speaker`/`char_id`, and `cast[].name`/`char_id`/`voice_preset`/
`voice_ref_id`/`voice_engine`. A shortened cue reaching `lines[].speaker` while
`cast[].name` keeps the full name breaks the join: that line gets NO VOICE, and
captions, credits, per-beat slicing and shot direction all read a character that
does not exist -- silently.

**This is why prefix/fuzzy matching of abbreviated speaker cues was rejected
three times.** If you propose anything that lets a non-canonical name survive
into a ledger field, say explicitly how the join stays intact.

---

## 6. LIVE SWEEP STATE (for context, not for review)

21 legs: 19 local lanes on one boot, then the two H3 lanes on a second boot with
`--reserve-vram 12`. **8 pass, 3 fail, 10 to run.**

- PASS: `still_flat`, `still_pan`, `still_motion`, `still_word`, `viz_camera`,
  `viz_green`, `viz_mxc_cpu`, `viz_mxc_mandala`
- FAIL: `mesh_stage` (cast coverage), `ltx_8gb` (writer, pre-fix-B),
  `fastwan_8gb` (the still-spine gap in section 2.3)

`still_flat` and `viz_green` are the two legs that died this morning on PBUG-02
and PBUG-03; both now PASS, which is the live proof those two fixes work.

---

## 7. THE FOUR QUESTIONS I MOST WANT ANSWERED

1. **Does `compose_line_draft` really never send the rejected line back?** If
   yes, it is fix A's bug in the writer five banks share, and it is the next
   thing I build.
2. **Is the deterministic round-robin speaker fallback real**, and does it
   already violate "no deterministic Python deciding story"?
3. **Is the `fastwan_8gb` still-spine gap systemic** across the 8 scene-still
   engines, or specific to that lane?
4. **Is the alternate producer slot genuinely absent?** Four reviewers assumed
   it exists.

Anything you find that none of the four lanes said is worth more than
confirming what we already believe.
