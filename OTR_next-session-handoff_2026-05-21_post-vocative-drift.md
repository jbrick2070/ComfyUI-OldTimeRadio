# OTR -- Next Session Handoff (post-vocative-drift)

- **Date:** 2026-05-21 (continued session)
- **Branch:** v2.0-alpha
- **Last commit:** `6f2abcd` -- BUG-LOCAL-233 vocative-drift pass
- **HEAD:** local == origin == `6f2abcd`, working tree clean

---

## Session result

Four commits landed on `v2.0-alpha` on top of `a0bdaa1`. Each had its
own full `tests/` walk, green, with zero regressions. No workflow JSON
changes were needed -- no commit touched a node surface (class names,
INPUT_TYPES widgets, output sockets all unchanged).

### 1. `8fab1a3` -- KNOWN-FAIL-007/008: re-audit Gemma 4 license as Apache 2.0

The triage handoff framed KNOWN-FAIL-007/008 as a "product/licensing
decision" (Gemma-4 vs Mistral-Nemo for the shipped default). It was not
a decision -- it was stale audit metadata. The two
`google/gemma-4-E{2,4}B-it` catalog rows + audit files were tagged
`gated_terms`/`pending` on 2026-05-16, assuming Gemma 4 inherited the
old restricted Gemma license. It doesn't: Gemma 4 ships under
**Apache 2.0** (confirmed on the official Google HuggingFace model
cards for both E2B-it and E4B-it). Corrected the catalog rows +
`docs/model-license-google--gemma-4-e{2,4}b-it.md` to
`apache_2_0`/`mit_equivalent`; the Sprint D / D3 creative-binding gate
now passes. KNOWN-FAIL-007/008 promoted; `conftest.py::EXPECTED_FAILED_NODEIDS`
is now empty. No workflow or product change -- the shipped default
still binds gemma-4-E4B-it.

### 2. `90a5700` -- BUG-LOCAL-240 [FIXED]: relax freeze-cascade style-slug validator

`_otr_ledger_freeze.py::_check_meta_invariants` rejected any
`meta.gen_params_initial.style` outside the 10-slug seed palette as
"writer drift" -- but the style picker's "let the story decide"
sentinel invents new snake_case slugs by design, and the membership
check also guarded a dead consumer (musicgen_theme stopped using the
slug as a palette key at Path F, 2026-05-18). Replaced the
`KNOWN_STYLE_SLUGS` membership check with `_is_well_formed_style_slug`
(snake_case regex + 64-char cap); `test_style_palette_drift.py` updated.
**BUG-LOCAL-236 was investigated and found already fixed** -- the
Sprint-E "K.5.7" title-clobber block was deleted 2026-05-20 and
`test_writer_stamps_episode_title.py` already guards it. BUG_LOG marked
it [FIXED]; no code change.

### 3. `12e9719` -- video_engine.py: refresh stale title-chain comment

Audited the episode-title flow end to end (LLM regen -> ledger stamp
-> procgen video). The chain is correct and consistent: the writer's
J.5 pass LLM-regenerates the title, stamps `meta.episode_title`, and
`video_engine.py` reads that as title-chain slot 1 -- the same title
flows to the CRT title bar, the telemetry HUD, the `.mp4` filename,
and the treatment `.txt`. The only issue was a stale comment +
log-warning in `video_engine.py` still claiming slot 1 was "not
stamped by OTR_LedgerScriptWriter" (pre-J.5 wording). Comment /
log-string text only; no behavior change.

### 4. `6f2abcd` -- BUG-LOCAL-233: vocative-drift pass

Characters were speaking the production label "ANNOUNCER" aloud --
e.g. in `docs/2026-05-19-first-stable-run-ledger.json`, character
c02's lines b002/b004 ("It wasn't just geology, ANNOUNCER.") reached
`text_for_tts` with empty `compose_flags`. The "responding to
ANNOUNCER" prompt induction was already gone (Tier 1 fix #4,
2026-05-11); the real foothold was the `[ANNOUNCER]:` speaker label
in the LAST SPOKEN window block, one step above the WRITE LINE slot.
Three-part fix in `nodes/_otr_line_composer.py`:

- **A2** -- `_format_last_lines` renders the announcer's window
  entries as `[narration]:` instead of `[ANNOUNCER]:`. "narration"
  is 9 chars, same as "ANNOUNCER" -- zero prompt-size cost.
- **B** -- `compose_line` stamps `compose_flags="vocative_drift:ANNOUNCER"`
  on any non-announcer line addressing the label, so the drift is
  visible in `compose_flag_summary` instead of silent (the phantom
  gate could never catch it -- "ANNOUNCER" is always roster-whitelisted).
- **C** -- `strip_announcer_vocative` deterministically removes the
  comma/boundary-anchored address (trailing / mid / leading). Plain
  noun references ("the announcer") and real cast-name address
  ("Alice, run!") are left untouched.

cc553b7's prompt strengthening is kept as belt-and-braces. D (reroll)
remains the documented, unbuilt fallback. 17 new tests in
`tests/test_vocative_drift.py`.

### Verification

- Full `tests/` walk after the final commit: **2458 passed / 22
  skipped / 0 failed** (2480 collected), zero regressions,
  `KNOWN-FAIL-GUARD` green (exit 0). Each commit had its own green walk.
- All changed `.py` files AST-parse clean.
- `local HEAD == origin HEAD == 6f2abcd`; working tree clean.
- **Bug Bible regression NOT run** -- the
  `comfyui-custom-node-survival-guide` sister repo is not present in
  this Cowork environment. The changes are metadata + a validator
  regex + line-composer text logic; none overlap the Bug Bible's
  VRAM / ffmpeg / parse-fatal coverage, but a run on your machine
  would close it formally.

### Environment note

The Linux sandbox was used only for directory listings; all file
edits used the Windows file tools and all git ran through Desktop
Commander `cmd`. The forbidden-sweep test regenerates
`docs/2026-05-13-S28-new-forbidden-hits.txt` + `docs/s28_diff_tmp.txt`
on every walk -- these were reverted post-commit and are NOT in any
of the four commits (still worth gitignoring).

---

## YOUR SMOKE -- BUG-LOCAL-233 verification

The vocative fix is fully unit-tested, but the real proof is a writer
run on the live LLM. You do NOT need a full episode -- the fix lives
entirely in the script writer, so the **writer + freeze phase is
enough** (you can interrupt before the long HuMo render).

**Run:**

1. ComfyUI Desktop at localhost:8000; load `workflows/otr_scifi_16gb_full.json`.
2. Queue one episode -- a short one (low target_words, 2 characters)
   is plenty.
3. Let it run through the writer + freeze cascade. Once the ledger is
   saved you can interrupt -- the audio/video render is not needed for
   this check.

**Pass criterion:** zero character lines speak "ANNOUNCER".

**What to check (any one is enough -- pasting me the console is easiest):**

- **Console:** `[OTR_LineComposer] vocative drift on <speaker> line:
  stripped N 'ANNOUNCER' address(es)` -- if these appear, the LLM
  still drifted but C stripped it (output is still clean). Zero of
  these means A2's relabel held on its own.
- **Console:** the `[OTR_LedgerScriptWriter] phase 0
  compose_flag_summary:` line -- a `vocative_drift` count > 0 means
  drift happened and was caught; that is fine, the flag is the audit
  trail.
- **The ledger** (`...ledger.json` in the episode dir): every
  `lines[]` row with `"speaker_role": "character"` -- its `text` must
  NOT contain "ANNOUNCER". This is the real pass/fail.

**Easiest path:** paste me the console output (from "Exec" start
through the freeze cascade), OR just tell me the run finished and next
session I can pull the newest ledger via `/otr/latest_ledger` and grep
every character line for "ANNOUNCER" myself.

**If it fails** -- a character line still contains "ANNOUNCER" AND
there is no `vocative_drift` flag for it: the strip missed an address
shape; send me the exact line text and I will widen
`strip_announcer_vocative`. If it fails WITH a flag present, that is a
different (rarer) edge -- still send the line.

---

## CARRY-OVER -- priority order

### 1. BUG-LOCAL-233 soak verification (your smoke, above)

The only open item on the vocative fix. Until a live writer run
confirms zero "ANNOUNCER" in character text, BUG_LOG keeps BUG-233 at
`[FIX LANDED -- pending soak verification]`.

### 2. OPTIONAL -- round-robin review of BUG-249/250 meta.brief work

Still pending from the 2026-05-20 handoff -- the `_build_portrait_prompt`
brief-leading-order question. The round-robin runs on your Windows
machine (HKCU keys) -- not from Cowork.

### 3. LOW PRIORITY -- redundant partial folder_paths stubs

The partial stubs in
`test_story_brief_{humo_c5f,ltx_c5e,portraits_c5d}.py` are now
harmless no-ops (conftest installs the complete stub). Deletable in a
cleanup pass; left as a defensive fallback. Not urgent.

### 4. NOTED -- stale docstring in `_otr_style_palette.py`

While fixing BUG-240 I noticed `_otr_style_palette.py`'s module
docstring still says musicgen_theme consumes `STYLE_PALETTE` and
"unknown slug is a hard fail" -- both stale since Path F (2026-05-18).
Cosmetic; a Sprint G cleanup item, not logged as a bug.

### Bigger picture

BUG-LOCAL-231 (FLUX sampler ~10-15x slow) remains THE blocker on
Sprint A (downstream render verification), the Priority-1
ledger-durability sprint, and BUG-234/235 verification. It is
round-robin-gated and needs telemetry smokes on your machine.
BUG-LOCAL-237 / 238 / 239 / 241 / 242 stay filed-and-deferred until
the pipeline closes. BUG-LOCAL-232 (cast generator stamping a
malformed ANNOUNCER cast row) is still PENDING and is upstream of --
and distinct from -- the BUG-233 vocative drift fixed this session.

---

## Paste-ready prompt for the next conversation

```
OTR (ComfyUI-OldTimeRadio / SIGNAL LOST), branch v2.0-alpha.
First moves: read CLAUDE.md, BUG_LOG.md header, then
git log --oneline -6 on v2.0-alpha.

SHIPPED last session (4 commits, a0bdaa1 -> 6f2abcd, all pushed +
verified, every walk green):
- 8fab1a3 KNOWN-FAIL-007/008: gemma-4 re-audited as Apache 2.0
  (was stale audit metadata, not a product decision); catalog +
  audit files corrected; quarantine set now empty.
- 90a5700 BUG-LOCAL-240 [FIXED]: freeze-cascade style-slug
  validator relaxed to a snake_case shape check. BUG-LOCAL-236
  found already fixed (K.5.7 deleted 2026-05-20).
- 12e9719 video_engine.py stale title-chain comment refreshed
  (title flow audited end-to-end -- correct).
- 6f2abcd BUG-LOCAL-233 vocative-drift pass: A2 [narration]
  relabel + B vocative_drift flag + C strip_announcer_vocative,
  in _otr_line_composer.py; 17 tests in test_vocative_drift.py.
Full walk 2458 passed / 22 skipped / 0 failed.

CARRY-OVER, priority order:
1. BUG-LOCAL-233 soak verification -- I ran a writer smoke; here
   is the console output / ledger: <paste>. Confirm zero
   character lines speak "ANNOUNCER".
2. OPTIONAL round-robin review of BUG-249/250 meta.brief work.
3. LOW PRI: delete redundant partial folder_paths stubs.
4. NOTED: stale _otr_style_palette.py docstring (Sprint G cosmetic).

Then: <your new tasks here>

Constraints: see CLAUDE.md. Bug Bible regression could not run
last session (survival-guide repo absent) -- run it if you can.
```

---

*Drop your next tasks in where it says `<your new tasks here>`.*
