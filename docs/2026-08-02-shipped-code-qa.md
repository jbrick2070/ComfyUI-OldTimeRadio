# QA the SHIPPED code: three commits, 624 lines, already pushed

This is an adversarial QA pass on code that is **already committed and running**
-- not a design review. The design reviews happened first and are in
`kibitz-runs/2026-08-02-cast-member-voice/`. Your job is to find what is WRONG
with the implementation, not to relitigate the approach.

Commits under review (`030b9b67`, `1ab714a8`, `19401dd4`):

| file | what it does |
|---|---|
| `nodes/_otr_cast_voice_coverage.py` (NEW, 149) | the sayable-text voice gate |
| `nodes/_otr_content_authorship.py` (+8) | calls the gate at `stamp_receipt` |
| `nodes/_otr_ledger_freeze.py` (+46) | escalates the per-cast check to an ERROR |
| `scripts/run_video_arm_bakeoff.py` (+193) | the `--estimator-fit` mode |
| 3 test files (+263) | coverage for all of the above |

Suite: 8271 passed, 131 skipped, 1 xfailed.

## WHAT THE CODE CLAIMS

**Claim 1 -- every cast member gets a voice, on all seven banks.** A "voice"
means at least one non-skipped line whose text SURVIVES `clean_spoken_text`
(the same stripper TTS uses), because the defect was a character whose five
lines were pure stage direction -- `(static crackles)`, `[low hum]` -- which
looked voiced to every raw-text check, then got emptied by the writer-tail
cleanup, then surfaced as a proof-coverage mismatch five stages later.

Two layers, deliberately:
* `require_voice_coverage` at `_otr_content_authorship.stamp_receipt` -- the one
  call site both content-owned lanes share, BEFORE line proofs are minted;
* the escalated per-cast check in `_otr_ledger_freeze._check_per_cast_invariants`
  -- the cascade every bank converges on, since five of seven banks never mint
  a receipt at all.

The claim that the two layers agree BY CONSTRUCTION: by the time the cascade
runs, the cleanup has already turned "nothing sayable" into `skip=True`, so
"non-skipped" there means what "sayable" means at the earlier gate.

**Claim 2 -- the estimator-fit mode can legally produce a cost row.** The
campaign's own contract forbids refitting `FRAME_COST_MODEL` from clamped
cells, so `--estimator-fit` runs UNCLAMPED (reserve 0), arm A only, full 4n+1
ladder, 3 fresh-boot repeats, NVML at 0.1 s, and emits a receipt -- never a
code change. Unit: demand above the cell's own quiescent baseline
(`peak_delta_mib`). Row: upper envelope, slope clamped >= 0, rounding only ever
UP.

## WHAT I ALREADY KNOW IS OPEN (do not spend your review here)

* `_otr_voice_node_common.py:109-127` still hands a dangling `char_id` a real
  randomly-seeded fallback voice instead of raising. Known, deliberately a
  follow-up chunk, already recorded.
* `wan_ti2v`'s mirror, cost row and topology are UNCODED and under a separate
  live panel. Out of scope here.

## WHAT TO ATTACK

1. **Break the voice gate.** Find a ledger shape that has a mime in the cast and
   still passes BOTH layers. Consider: a cast row with a blank/whitespace
   `char_id`; duplicate `char_id`s; a character whose only line is on a
   different bank's sentinel; `skip` vs `skip_tts` (the ledger uses BOTH -- does
   the gate read the right one, and is `_voiced_rows` using the same field?);
   a line whose `char_id` matches a cast row only after case/whitespace
   normalisation; `announcer_only_fallback` episodes; music-only episodes;
   an empty `lines` list; `cast` present but not a list.
2. **Break the announcer credit.** The new code matches the announcer cast row
   by `name.upper() == "ANNOUNCER"` or role, and credits it via the
   `"announcer"` line sentinel. What if an episode legitimately casts a
   CHARACTER named "Announcer"? What if two rows claim announcer? What if the
   announcer's name is localised or empty?
3. **Is the two-layer agreement claim TRUE?** Verify the cleanup really runs
   before the cascade on every lane -- including the five banks that never mint
   a receipt. If any lane reaches the freeze WITHOUT the cleanup having run,
   "non-skipped" and "sayable" diverge and the backstop has a hole. Cite the
   call order.
4. **Did escalating a warning to an error break a legitimate episode shape?**
   The announcer mismatch was one such landmine and is fixed. Find the others:
   any shipped profile, bank or fixture that legitimately carries a cast member
   with no lines. `announcer_only_fallback` is handled -- what else? Search the
   test fixtures and `config/profiles/` for shapes that will now refuse.
5. **Break the fit math.** `fit_cost_row` in `run_video_arm_bakeoff.py`. All
   points identical; a single outlier 10x high; lengths not ascending; a
   negative demand (baseline sampled above peak); float precision at the
   ceil boundary. Does the envelope ALWAYS cover every input point?
6. **Break the fit MODE.** `--estimator-fit` mutates three module globals
   (`ACTIVE_CLAMP_GIB`, `ACTIVE_SAMPLER_INTERVAL`, `BENCH_DIR`). Does anything
   read them before `main()` sets them? Can a campaign run and a fit run collide
   in the same process or on disk? Is `BENCH_DIR` reroute safe when combined
   with `--regrade` / `--dry-validate` / `--offline-only`? Does the repeats loop
   interact correctly with the "stop this arm's ladder on failure" break?
7. **Test quality.** Are the new tests actually adversarial, or do they only
   prove the happy path plus one failure? Name any assertion that would still
   pass if the implementation were deleted.

## GROUND EVERYTHING

Read the real files under
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`. Cite
`file:line`. Mark anything you cannot verify UNVERIFIED rather than asserting
it. **A GPU measurement is running right now -- do not boot a server, do not
launch renders, do not edit files.** Read-only.
