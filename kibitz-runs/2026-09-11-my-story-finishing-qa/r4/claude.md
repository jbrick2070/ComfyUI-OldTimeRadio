VERDICT: yes-with-fixes. Every specific factual claim in F2/B/C/Mac/O2 checks out
against the real files (details below); the only concrete problems found are a
repo-hygiene hazard (stray unrelated diff dumps in the working tree) and two
open-status items the driver already disclosed as pending, not new defects.

GROUNDING SUMMARY (spot-checked against real files, not the input.md prose):
- F2 retry cap: `nodes/otr_meta_brief_image_prompt.py:1740`
  `max_attempts=min(2, max(0, int(max_reseed)) + 1)`, and
  `nodes/_otr_story_source.py:25` `SOURCE_REWRITE_ATTEMPTS = 2` /
  `:97` `attempt_limit = min(SOURCE_REWRITE_ATTEMPTS, max_attempts)`. Matches
  "at most two calls ... capped by max_reseed + 1" exactly.
- F2 unresolved-correction fallback: `otr_meta_brief_image_prompt.py:1763-1766`
  falls back to `initial` and appends a warning when `corrected is None`;
  `:1753-1762` catches `BaseException`, logs, then bare `raise` (real
  provider/OOM/cancel propagates, nothing falsely claimed persisted). Matches.
- F2 test coverage: `tests/test_my_story_visual_source.py` has 17 tests
  including budget-cap (`:156`), malformed-repair-once (`:166`),
  OOM/cancel-escape (`:177`), log-before-rethrow (`:189`), jump-merge cache
  identity (`:365`), stale-evidence disclosure (`:386`).
- B request/response identity: `nodes/OTR_LedgerScriptWriter.py:605-625`
  `_record_successful_model_call` — non-openrouter stays
  `basis="request_identity"` (`:613`); openrouter reads
  `base._otr_response_model_receipt` and sets `basis="response_model"` or
  `"unreported"` (`:614-619`) when `reported_model_id` is absent. Called at
  `:764` right after the generation call returns (matches "returned
  generation, not accepted authorship" as an accepted, disclosed property).
- B publish-after-validation/runaway: `nodes/_otr_openrouter_backend.py:1696`
  resets `generate_fn._otr_response_model_receipt = None` at call start;
  `:1721-1727` runs `assert_no_verbatim_cycle` (the runaway guard) BEFORE
  `:1728` stamps the receipt. A raised runaway/validation error leaves the
  receipt `None` and the exception propagates before the writer's record call
  is ever reached. Matches "published only after validation/runaway checks".
- C hero wrap: `nodes/otr_credits_roll.py:738-766` `_hero_lines` wraps using
  `draw.textbbox` (not advance-width), breaks on grapheme clusters via
  `_hero_clusters` (`:718-735`, base+mark, flag pairs, ZWJ) with no ellipsis;
  `:1000-1005` re-tightens `hero_pt` against the real ink bbox (negative
  bearing/overhang) after the coarse `_autoshrink_pt` advance-width pass
  (`:711-715`); `:1007-1011` compensates `top_ink`/`bottom` per drawn line.
  Test coverage: `tests/test_credits_roll_spec.py:175,216` parametrize
  `(832,480),(1280,720),(1920,1080),(3840,2160)`; `:217-219` uses the real
  production title "THE LANTERN BURNS BRIGHT WHILE SLANDER HIDES" against the
  footer floor. Matches every specific claim in C.
- Mac snapshots: exactly the 7 call sites named, with the exact labels named
  in the prompt — `OTR_LedgerScriptWriter.py:1098` (`writer_generation_returned`),
  `_otr_constrained_generate.py:347` (`constrained_generation_returned`),
  `_otr_model_loader.py:2468`/`:2661` (`base_generation_returned`,
  `polish_generation_returned`), `_otr_model_loader.py:1580/1598/1654`
  (`llm_retirement_before/_after_cpu_move/_after_allocator_flush`).
  `nodes/_vram_log.py:110-144` `memory_snapshot`: RSS and MPS reads are each in
  their own `try/except Exception` (`:122-126`, `:127-137`), the whole
  function never raises (`:138-143` swallows the logging failure too), and
  nothing here calls into allocator/free/empty_cache. Matches "lazy,
  independent, non-raising" and "no new allocation policy".
- O2 frozen roles: `nodes/_otr_video_engines/render_driver.py:5622-5633`
  `frozen_route_from_ledger` reads `ledger["video"]["roles_effective"]` and
  returns a **copy** (`dict(frozen)`), never re-derives from live row state;
  `:6708` the manifest builder stamps `"roles_effective": frozen_route_from_ledger(led)`
  once at build time. `:6042-6116` `check_ltx_open_health` — `healthy`/
  `not_requested`/`unknown`/`sanctioned`/`degraded` are five distinct states
  (`:6082,6087,6092,6095,6097`), only `degraded` rows land in the returned
  `bad` list, and `status` (`:6105-6107`) is computed from the full
  `observations` set, not from `bad` — so "empty offenders is not health
  proof" is actually true of this implementation, not just an aspiration.
  `report_out` is only mutated when the caller passes a dict (`:6108-6110`);
  a bare call reads only. `_LTX_OPEN_ENGINES` (`:6029-6036`) has exactly six
  members: `ltx_video, ltx_8gb, ltx25_video, ltx_audio_in, ltx25_foley_plus,
  ltx25_mime`. Matches every specific claim in O2.

MUST-FIX BEFORE BUILD:
1. [repo hygiene, not in the F2/B/C/Mac/O2 diff itself] Two untracked files,
   `diff.txt` and `diff_utf8.txt`, sit in the repository root (confirmed via
   the git status snapshot: both `??`). Their content is NOT this campaign —
   it is the already-shipped, already-logged PBUG-20260909-04 font
   measure/draw fix (`video_engine.py` `_load_font`, `docs/SHIPPING_JSON_RECIPES.md`,
   `scripts/otr_canonical_api_run.py`), which `docs/PROD_BUG_LOG.md:13579`
   already records as fixed 2026-09-09 and which is NOT among this session's
   modified files (`video_engine.py` does not appear in the git status M list
   at all). Concrete fix: delete both files (or move them out of the repo)
   before the next commit/push. Why this is MUST-FIX and not cosmetic: this
   review was pointed at "the finished F2/B/C/Mac/O2 diff" with no explicit
   path, and `diff_utf8.txt` is the most diff-shaped artifact sitting in the
   tree — a reviewer or tool that opens it first (as this review initially
   did, before cross-checking git status) would produce a review of the wrong
   change entirely and could fold stale, unrelated findings back into this
   campaign's record. The project's own git-push protocol (root CLAUDE.md)
   requires temp/scratch artifacts be cleaned before any commit for exactly
   this reason.

SHOULD-FIX:
1. [process, not code] `docs/PROD_BUG_LOG.md` has no entry for F2's two
   review-found defects ("setting omission and lost jump-source identity").
   Checked this against the admission rule in
   `custom_nodes/ComfyUI-OldTimeRadio/CLAUDE.md` ("only a bug verified by a
   live production artifact ... may enter PROD_BUG_LOG.md ... a review
   observation ... may verify a known production bug, but never creates a new
   PBUG on its own") — since input.md states these were "Independent review
   found," not live-run found, the CURRENT absence is actually rule-compliant,
   not a gap. Flagging only so the next live canonical run (see VERIFY-AT-BUILD)
   is the trigger to add the entry once it reproduces/confirms live, not
   before.

OPTIONAL / NICE-TO-HAVE:
1. [C, cosmetic] `otr_credits_roll.py:996-1005` always shrinks the hero title
   to the size that would fit as ONE line before `_hero_lines` ever wraps it
   (confirmed: `_autoshrink_pt` and the bbox-tightening loop both measure the
   whole unwrapped string). A title that would wrap comfortably at a larger
   size two lines will still render at the font floor. Input.md frames this
   as intentional ("hero-only ... wrapping after existing autoshrink reaches
   its font floor," "existing footer/abridgment policy stays"), so this is a
   disclosed design property, not a bug — noted only as a possible future
   quality improvement, explicitly NOT something to chase per the "story
   quality is done" / visual-recipe-stability rulings already on file for
   this project.

CUT THESE:
None — nothing in the reviewed scope reads as speculative or unused; every
code path I traced (F2 retry ladder, B identity basis, C wrap/bbox, Mac
snapshot sites, O2 health states) is exercised by the new tests cited above.

VERIFY-AT-BUILD checklist (items the input.md itself marks pending, or that I
was explicitly told not to independently attest):
1. "Bible check still pending" (input.md line 55) — confirm it runs and
   passes before calling this build-ready; no Bible entry currently exists
   for this campaign's fixes (see SHOULD-FIX #1) so this is also where a new
   entry would be added if a live run confirms the F2 defects reproduced.
2. Full-suite baseline "14,283 passed / 51 existing failures / 183 skipped /
   1 xfailed against F1 baseline" — confirm the 51 failures are the SAME 51
   as the pre-campaign baseline (a diff of failing-test names against the
   prior baseline run), not a coincidentally-equal count masking a new
   failure offset by a newly-fixed one. [ASSUMPTION: I have not run the
   suite or the F1 baseline myself; this is exactly what the input.md asked
   me not to independently attest.]
3. Canonical validator "23 nodes, 63 links, no interfaces or widgets changed"
   — confirm by running `OTR_WorkflowValidator` fresh against
   `workflows/otr_canonical.json` per section 0 of this repo's CLAUDE.md; I
   did not recompute node/link counts myself.
4. `docs/PROD_BUG_LOG.md:14170` (`PBUG-20260829-14` addendum, the B/original-
   cleanup fix) is explicitly still `status: code corrected; fresh canonical
   live qualification remains OPEN and on hold until all campaign coding is
   complete.` Confirm that live qualification leg actually runs and
   publishes to `otr/obs/` before treating B as production-proven, per this
   repo's standing obs-publication rule.
5. Hardware qualification (Mac/4060) is explicitly out of scope for this
   round per input.md's own closing line — confirm it is scheduled, not
   silently dropped, once code QA closes.
6. `_wrap`/`_hero_lines` and the retirement-snapshot call sites were read
   statically; no test run was executed by this review (per the read-only
   contract). Confirm `tests/test_credits_roll_spec.py`,
   `tests/test_my_story_visual_source.py`, `tests/test_original_model_credit.py`
   and `tests/test_memory_observation.py` actually pass in this session's
   venv, not just that they exist and assert the right things.

[ASSUMPTION] I did not execute any test, workflow validator, or model call —
per the tool-health/read-only contract in this task, every claim above is
either a direct file citation or explicitly marked as unverifiable by me and
deferred to VERIFY-AT-BUILD.
