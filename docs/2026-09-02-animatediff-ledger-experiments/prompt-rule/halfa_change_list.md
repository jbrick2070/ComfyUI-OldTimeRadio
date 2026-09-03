# Prompt v3 Half A -- the file-by-file change list

Written after r3, before r4 converges, so implementation starts the moment it
does. The governing contract is **W1-W7** (`../../..//kibitz-runs/2026-09-02-prompt-v3-crux/r3/judgment.md`
section 9). Nothing here may contradict it.

Branch: `prompt-v3`, worktree `C:\Users\jeffr\Documents\ComfyUI\_worktrees\otr-promptv3`.

---

## What the prompt becomes

**v2, as rendered tonight (mean 40 of 77 SD1 tokens):**

    storybook engraving. a lean figure in a charcoal coat, carrying a satchel,
    a figure in a charcoal coat holds a satchel out,
    mid-shot or wider, whole figure legible, one clear action, unbroken shot

**v3 Half A, same beat, same seed:**

    storybook engraving. film canisters in a high-security archive,
    harsh fluorescent overheads, <world motion>, <vantage>

Style, subject, place, light, motion, vantage. Six short units, every one of
them either the pack's or the episode's own words. No costume, no authored leaf,
no framing law.

## Where the per-beat variety comes from (the answer to r4's Q2)

**Not from the motion pool -- from cycling the ledger's own term lists.** This
episode carries 5 `key_objects`, 4 `setting` terms and 4 `lighting` terms. Cycled
deterministically by beat ordinal, that is far more distinct combinations than a
29-beat episode needs, and every one of them is the story's own vocabulary.

The motion clause can therefore stay a SMALL pool (the v2 fallback pools
exhausted at six because they were the only varying part; here they are the
least varying part). Keyed by `_hash_int(episode_seed, beat_id, ...)` with the
same collision-probing walk `deterministic_leaf` already uses, so no episode
repeats a clause.

**What Half A cannot do, stated plainly:** subject-appropriate motion. The
operator's own rewrites say *"a mass of organic matter moves with the drift"* and
*"sensors coming down from the sky, hitting the water"* -- motion that belongs to
that subject. A deterministic pool cannot produce that, and deriving it from the
subject noun would be exactly the unbounded vocabulary classifier r2 cut. **That
is Half B's job**, and Half B is where the beat's own dialogue reaches the
author. Half A buys the story's object, place and light on every beat; Half B
buys the story's motion.

## The changes

### 1. `nodes/_otr_video_engines/ghost_signal_prompt.py`
* **ADD** `GHOST_PROMPT_VERSION_V3 = "ghost_signal_v3"` beside the v2 constant
  (line 63). **Do NOT touch `GHOST_PROMPT_PROFILE`** (line 51) -- the registry
  rows key on it.
* **ADD** `GHOST_V3_SLOTS = ("pack_cue", "kernel", "light", "motion", "vantage")`.
* **ADD** `GHOST_VANTAGE_V3` mapping the three STORED modes onto a vantage
  clause (r3 SF2): `figure -> "wide, the people small in the space"`,
  `object -> "the object large in the frame"`,
  `signal -> "lit against the dark, the light moving"`. A separate table; it
  never calls `GHOST_MODE_LAWS_V2`.
* **ADD** `compose_ghost_prompt_v3(role, style, mode, kernel, light, motion)` --
  pure and scalar like its v2 sibling, joins the present units, front-anchors the
  pack cue through the shared `prefix_style_cue`, returns
  `{"positive", "negative", "components", "slots"}`. It takes **no** `motif_cue`
  and **no** `drawable_beat` (W1).
* **UNCHANGED:** `compose_ghost_prompt_v2`, `GHOST_MODE_LAWS_V2`,
  `GHOST_V2_SLOTS`, `GHOST_PROMPT_MAX_CHARS`, `compose_ghost_negative`.

### 2. `nodes/_otr_video_engines/ghost_signal_author.py`
* **ADD** `resolve_crux_kernel(meta)` -- TOTAL, never raises (W4). Bounded first
  `key_objects` entry plus the first `_read_setting` term; else a bounded
  `story_brief` slice; else `("", "omitted")`. Returns the text and a
  `kernel_source` reason (`key_object | brief | omitted`) for the receipt.
* **ADD** `resolve_world_motion(meta, *, beat_id, episode_seed, used)` -- the
  small pool with the existing collision-probing walk.
* **ADD** `finalize_ghost_prompt_v3(...)` -- a NEW function (W2). Composes, drops
  whole optional units to fit, applies banana exactly ONCE, re-measures after it,
  and **never raises on a frozen row** (W3). Drop order:
  `light -> motion -> vantage -> setting half of the kernel`. The subject noun
  and the pack cue are never dropped and never word-sliced.
  Target 69 before banana, refuse only above 77 after -- and even that refusal
  cannot fire, because the drop order runs first.
* **UNCHANGED:** `finalize_ghost_prompt_v2`, `candidate_fits`,
  `assert_shell_fits`, `motif_for_character`, `MOTIF_FALLBACK_POOLS`,
  `GHOST_AUTHOR_VERSION`, `GHOST_REQUEST_HASH_KEYS`, `validate_drawable_beat`,
  `deterministic_leaf` (W6 -- ShotLock authoring still calls all of them).

### 3. `nodes/_otr_video_engines/render_driver.py`
* In the Ghost branch (2870-2960): call `finalize_ghost_prompt_v3` instead of
  the v2 finalizer, passing `ledger_meta`.
* Stamp `prompt_version = GHOST_PROMPT_VERSION_V3`, keep `prompt_slots` a list
  of names, add `prompt_slot_tokens` (map) and `prompt_dropped` (list) and
  `kernel_source` (W5).
* Rename `_ghost_v2_finalized` to `_ghost_prompt_finalized` (version-neutral,
  r2 MF10 / r3 MF3).
* **ADD** the three new keys to the `/history` trace allowlist at 4981-5006 in
  the SAME change, or they never reach the report.
* The `ghost_*` object receipts (`ghost_motif_cue`, `ghost_drawable_beat`, ...)
  keep being stamped from the stored object. It is still the row's authored
  record even though the prompt no longer uses it.

### 4. `nodes/otr_video_render_batch.py`
* **ADD** `IS_CHANGED` returning the prompt version alongside the ledger hash
  (r3 SF3) -- the node has none today, so a resident session can serve cached v2
  clips under v3 code.

### 5. `scripts/otr_verify_replay.py`
* **ADD** an `--ab` mode: seeds and `render_request_hash` equal, `text_prompt`
  and `actual_request_sha` different. The existing A/A path is untouched (W7).

### 6. Tests
* **REWRITE, never delete**, `tests/test_ghost_prompt_v2_lane.py` where it
  asserts `prompt_version == GHOST_PROMPT_VERSION_V2` and `motif_cue in
  text_prompt` -- these become a v2-object / v3-prompt split.
* **NEW** `tests/test_ghost_prompt_v3.py`: the kernel resolver is total over
  empty `key_objects`, empty `setting`, missing brief and a failed brief; the
  composer emits no motif, no leaf and no law; every one of the nine packs fits
  inside 69 tokens on the longest real kernel; the drop order sheds whole units
  and never word-slices; banana applies once; the vantage map covers all three
  stored modes; and **every frozen `ghost_prompt` on both proof episodes
  finalizes without raising** (the r3 MF2 pin).
* Run the full suite, then `tests/bug_bible_regression.py` from the Bible repo.

## Explicitly NOT in this change (W6)

The plate (`ghost_plate_prompt.py`), `GHOST_PROMPT_MAX_CHARS`, any other video
lane, any `INPUT_TYPES` / widget / link, the canonical workflow JSON, the story
brief schema, `crux_subject`, and any deletion of the v2 author constants.

## Proof, in order

1. Full suite green, Bible green, AST parse, no BOM.
2. Sonnet QA on the finished diff, scoped to the named functions.
3. Arm 0: freeze "The Faded Ledger", replay UNCHANGED. Instrument proof plus
   control.
4. Arm 1: replay the same bundle on v3. `otr_verify_replay --ab`.
5. Publish both to `otr/obs/`. His eye decides.
