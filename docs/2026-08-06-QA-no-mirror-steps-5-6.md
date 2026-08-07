# QA REVIEW REQUEST -- no-mirror steps 5 and 6 (SHIPPED, review after the fact)

You are doing an adversarial QA pass on a change that is ALREADY COMMITTED. Be
blunt. If it should be reverted or amended, say so.

**The change:** commit `57f92f74` on `v2.0-alpha`. Inspect it with:

    git show 57f92f74 --stat
    git diff e75cc321..57f92f74

Suite was green at 9018 passed / 131 skipped / 1 xfailed, Bug Bible 17,
`tools/engine_matrix.py --check` OK, `workflows/otr_canonical.json` unchanged.
**A green suite is not the question. The question is what a green suite cannot
see.**

## WHAT IT DOES

Step 5 -- ATOMIC DELETION of the LTX boomerang from
`nodes/_otr_video_engines/eng_ltx_video.py`: `_boomerang_frames`,
`_ltx_loop_source_length`, `_LTX_LOOP_MIN_DECODE_FRAMES_DEFAULT`,
`_LOOP_VIA_REVERSE_DEFAULT`, `_loop_via_reverse`, `_loop_fill_allowed`, both
render branches (`render_clip` and `_render_clip_hq`), and both
`ltx_loop_via_reverse` raw fields. The mirror rendered HALF a beat and doubled it
back forward-then-reverse. It was already unreachable (the gate returned False
unconditionally since 2026-08-02) but had previously been RE-ARMED by a default
flip, which is why it was deleted rather than left disarmed.

Consequence: the step-1 honesty receipts in both raw returns collapsed from
`"ping_pong" if mirrored else "none"` to an unconditional `"none"`.

`tests/test_ltx_boomerang.py` was CONVERTED into a tripwire (it previously proved
the mirror WORKED). `tests/test_video_motion_forward.py` lost two assertions on
the retired `ltx_loop_via_reverse` field.

Step 6 -- a fossil sweep of stale prose across `frame_contract.py`,
`eng_wan_ti2v.py`, `eng_ltx_8gb.py`, `beat_session.py`, `wrapper_bridge.py`,
`scripts/otr_w45_campaign.py`, and one LIVE INFO LOG in `nodes/scene_sequencer.py`
that was telling the operator mid-render that the boomerang "doubles the rendered
half-clip back to full audio duration".

## WHAT TO HUNT -- ground every claim in a file:line you actually read

1. **IS THE DELETION COMPLETE AND CONSISTENT?** Any surviving reference,
   import, attribute access, `__all__` entry, or call to a deleted symbol --
   anywhere in `nodes/`, `scripts/`, `tools/`, `tests/`. A dangling reference to
   a deleted module-level function is an ImportError or AttributeError at run
   time, and this adapter is a production render lane.

2. **DID THE DELETION TAKE SOMETHING LIVE WITH IT?** `render_clip` and
   `_render_clip_hq` had lines removed from the middle of a working render path.
   Read both methods END TO END in the new version and confirm the surviving
   code is coherent: is `length` still resolved before the graph build, are all
   variables still defined where used, is `prepared` still consumed correctly,
   does the HQ path still stage its init image? A removed block can leave a
   variable unbound on a branch the suite never exercises (no GPU in CI).

3. **THE RECEIPT COLLAPSE.** Both raw returns now stamp
   `native_frame_count: n` and `extension_mode: "none"` unconditionally. Verify
   `n` is the EMITTED count from the encoder at that point in both methods, and
   that nothing between the decode and the encode can still change the frame
   count on either path. If anything can, `"none"` is now a lie.

4. **IS THE TRIPWIRE REAL OR DECORATIVE?** Read
   `tests/test_ltx_boomerang.py`. It parses the two render methods with `ast`
   and asserts on string literals and identifiers rather than scanning raw
   source (an earlier draft failed on its own tombstone comments). Would it
   actually FAIL if someone reintroduced the mirror? Try to imagine a
   reintroduction it would MISS -- e.g. a differently-named helper, a mirror in
   a different module, a numpy slice inline in the render method. Say what it
   does not cover.

5. **COVERAGE LOSS.** The suite went 9033 -> 9018. Confirm the 15 lost tests are
   only the ones that proved the deleted machinery worked, and that no assertion
   about SURVIVING behaviour was deleted with them. Check
   `tests/test_video_motion_forward.py` especially -- two assertions were
   rewritten, not deleted, and the rewrite must be at least as strong.

6. **THE FOSSIL SWEEP'S HONESTY.** Several comments were CORRECTED in place
   rather than deleted, with the old claim quoted and marked expired. Verify each
   correction is factually right at HEAD -- particularly the
   `frame_contract.py` claim that `wan_ti2v` was added to `PLANNING_CAP_ENGINES`
   BECAUSE the mirror was removed, and the `eng_wan_ti2v.py` claim that the
   adapter now REFUSES a beat it cannot render in one pass. If either is wrong
   the comment is now more misleading than what it replaced.

7. Anything else you would revert or amend.

OUTPUT: numbered findings with file:line and the concrete failure each causes.
End with SHIP-AS-IS, AMEND (naming the smallest fix), or REVERT.
