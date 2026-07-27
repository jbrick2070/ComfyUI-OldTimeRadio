# LANE 2 QA findings -- the prequalification receipt names WHICH cell

CODER window, 2026-07-27 (remote Cowork), immediately after LANE 1. Panels: a
3-lens pre-push Sonnet fan-out (correctness/blast-radius, decorative-test,
ledger-integrity) plus three mutation rounds. $0 external spend -- no Codex, no
agy, no cloud roundtable; two-strikes never invoked.

Every finding was GROUNDED against the real Windows files through Desktop
Commander before it was acted on.

## What LANE 2 shipped

`71e231ec` (ltx_8gb + the shared format) and `8424f369` (both WAN adapters).

A measurement clip's receipt now names the knobs that cell departed from:
`ltx098_distilled_2b_i2v_single_pass_v2+prequalification[tiled_vae=off]`.
Before this, all four cells of the 2026-07-27 sweep stamped one generic
`+prequalification`, so a winning artifact could not prove which knob values
produced it -- the winner was selected from a table kept OUTSIDE the ledger
that outlives the run.

Rules the implementation holds:

- **RESOLVED values, never env presence.** Exporting a knob at the value it
  already has changes nothing, and a receipt claiming a departure there would
  describe a cell that does not exist -- worse than the generic mark, because a
  specific false claim is more credible than a vague true one.
- **Only knobs that actually bound the render.** The tile geometry is reported
  only when tiled decode ran. Reading it otherwise would also newly refuse a
  sweep whose stale tile value has no effect on the cell it is measuring.
- **A production receipt is byte-identical to what B6 shipped**, and the
  production path never reaches the resolver at all.
- **Per-entry bounds, never a truncated list.** Prose becomes a `#<8 hex>`
  digest; the departure list itself is never capped, because a silent cap would
  destroy exactly the distinguishability this chunk exists for.
- **One FORMAT for three adapters across two lanes that must not import each
  other.** `recipe_departures.py` is that one implementation and nothing else --
  pure functions, no engine imports, no environment reads. One consumer chain
  reads this string; two implementations would grow two dialects in the ledger.

## The defect the fan-out found: a latent lie in the receipt

**`_build_graph` lets a per-shot `negative_prompt` win** (`eng_ltx_8gb.py:1117`),
which is correct on a production leg and is why B6 called the negative a
demotion rather than a removal. But **the receipt is SESSION-scoped**: it is
element [1] of `session_identity`, which is read before the weights land and
again before every segment, so it may only describe request-independent things.
It can therefore only ever report what the RECIPE resolved.

So a sweep varying `OTR_LTX_8GB_NEGATIVE` against a shot carrying its own
negative would have rendered one conditioning and stamped a receipt naming
another. Not live today -- `render_driver` never populates `negative_prompt`
for a video shot (grep: zero occurrences) -- but nothing in the code prevented
it, and LANE 2 is what turns it from harmless into a false claim.

**Fixed structurally rather than by documentation:** under the consent act a
per-shot negative that would displace the measured one is now TERMINAL, by
name. Production keeps the director's channel exactly as B6 designed it. The
alternative -- making the receipt request-aware -- was rejected because it would
have made the receipt differ between the two stamp sites (`render_clip` has a
request, `resolve_session_config` does not), which is a `session_identity` drift
refusal on every multi-segment sweep beat.

Also fixed from the same fan-out: `render_clip` resolved the render config three
times per clip, so a sweep leg logged its "honouring X from the environment"
notice three times per clip. A sweep's log is the evidence the sweep exists to
produce; the resolved knobs are now threaded to the stamp.

## Test defects the panels and the mutation rounds found

The pre-push lenses found four; the mutation rounds then found four more that
three lenses had just cleared.

| # | found by | the defect |
|---|---|---|
| 1 | mutation | `pytest.raises(KeyError)` passed with the named drift guard DELETED -- the comprehension one line below raises the same type incidentally. Now `match=`es the message AND the offending key. |
| 2 | mutation | The production-path guard could be deleted and the suite stayed green, because every accessor returns the frozen value anyway. That made the guarantee depend on nine accessors staying correct forever. Now proven by DETONATING the resolver. |
| 3 | lens B | `test_a_long_value_becomes_a_DIGEST_not_a_truncation` was satisfied by `"#" + text[:8]` -- a truncation wearing a costume, passing the test named for refusing one. Now pins `#[0-9a-f]{8}`, plus a case where two values share a prefix. |
| 4 | lens B | The graph-vs-receipt tile cross-check covered 1 of 4 keys, while `_decode_inputs` hand-lists its four calls -- so a second implementation could creep back into the other three. Now parameterised over all four, on both adapters. |
| 5 | lens B | The one free-form knob (`negative`) was never driven end to end through a real adapter, only through synthetic strings in the pure tests. |
| 6 | lens B | `MAX_INLINE_VALUE` boundary untested -- a `<=` to `<` slip misfires only there. |
| 7 | lens B | A malformed tile value was proven fail-closed through `_decode_inputs` but only INFERRED through the departure path. |
| 8 | mutation | Dropping `negative` from **wan_ti2v**'s departure report stayed green, because only the wan_i2v twin of that test existed. Now parameterised over both. |

**And one of my own CONTROLs went red**, which is worth recording because the
lesson is not the one it looks like: a control that fails tells you nothing
about the harness and everything about the control. It renamed a dict at its
assignment and left three readers pointing at the old name -- a broken mutant
wearing a control's label. Replaced with a genuine no-op (a loop respelled as a
comprehension).

## Mutation results

Named CONTROL mutants throughout, to prove the harness discriminates rather
than reporting red on everything.

- Round 1 (format + ltx): 16/16 after two survivors were fixed; 2 CONTROLs survived.
- Round 2 (the fan-out fixes): **21/21**, 3 CONTROLs survived.
- Round 3 (WAN): 11/11 after one survivor and one bad control were fixed; 2 CONTROLs survived.

## Panel claims recorded but NOT acted on

- **`by_engine.setdefault` collapses same-engine clips (lens C #7).** CONFIRMED
  at `nodes/otr_video_render_batch.py:87`: the roll-up keeps the FIRST clip's
  receipt per engine id. Pre-LANE-2 that lost nothing (all sweep clips stamped
  the same string); now it would discard genuinely distinguishing data. NOT
  fixed here, and not a hole in the ledger: `per_clip` (line 85) keeps every
  clip's receipt in full, and the 2026-07-27 sweep ran one EPISODE per cell, so
  each ledger carries one recipe anyway. It is also outside the adapter lane and
  pre-existing. Recorded as an OPEN BUG with that reasoning.
- **The credits card would overflow if the display gap is ever closed (lens C
  #8).** `_draw_models` still never reads `video_suffix`, so nothing renders the
  string today; `_row()` has no clamp, so whoever wires it must add one. Folded
  into the existing credits-card OPEN BUG rather than opened as a new one.
- **The sampler cannot get an opposing-override test on `wan_ti2v`.** Its
  whitelist has exactly one member, so no legal value opposes the frozen one.
  Inherent, recorded in the test rather than faked.

## Gate

Full Windows suite **7346 passed / 27 skipped / 1 xfailed** (7291 before LANE 2),
Bug Bible **17 passed / 24 skipped / 3 xfailed**, AST/BOM/zero-byte/UTF-8/ASCII
clean on all eight touched files, canonical workflow
`9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
byte-identical -- LANE 2 adds no node, widget or link; it enriches a string on
the manifest row.
