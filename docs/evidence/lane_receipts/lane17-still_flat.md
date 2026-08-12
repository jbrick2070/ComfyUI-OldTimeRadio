# VIDEO_LANE_PREFLIGHT receipt -- lane 17, `still_flat`

`VIDEO_LANE_PREFLIGHT receipt: still_flat | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane17_still_flat/ | verdict PASS -- 7/7`

The third still lane, and the one that finishes the shelf: **all four still
families now refuse a missing still**, and the synthesised dark floor has no
registered occupant left.

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 | PASS | PASS | no weights |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT |
| G3 | PASS | PASS | already green -- lane 10's shared-base fix |
| G4 / G6 | n/a | n/a | exempt -- CPU/ffmpeg lane |
| G5 audio law | PASS | PASS | already probes its own emitted mp4 |
| G7 | PASS | PASS | matrix unchanged (nothing declared) |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures**.

## G2 -- INERT, premise re-checked on a THIRD builder

This lane sets `_still_motion = False`, so it renders through
`wrapper_bridge.ffmpeg_still_static_cmd` -- scale to FIT plus PAD, nothing
cropped -- **not** the pan builder lanes 15-16 verified. L19 says check the
premise per engine, and a different builder is exactly the case that rule
exists for.

Same answer: it takes the caller's width/height and letterboxes into them, so
there is no native canvas. The `even_dim()` snap is the same yuv420p mod-2 codec
requirement. Declaring would overrule `OTR_VIDEO_LANDSCAPE_CANVAS` for this lane
alone. Five profiles set `render.canvas_w/h` here; the channel is INERT.

**The smoke corroborates the builder split, which is the nice part.** Lane 16
showed `still_motion` and `still_pan` producing BYTE-IDENTICAL output
(`3692f155...`) because they share the pan builder. This lane's leg A, same
still, same canvas, same frame count, hashes **`56d48f21...`** -- different,
because fit+pad is not cover+crop. Same-builder lanes match; different-builder
lanes do not. The claim is now falsifiable in both directions.

## S8b-12(b) -- the last family to rule

`_require_still = True`. The case is strongest here: this engine's entire
contract is "hold the chosen image". A missing still is not a degraded version
of that, it is the absence of the thing.

Same evidence as lanes 15-16, re-run: `default_roles` empty, nothing routes here
automatically (the chain was ripped 2026-07-02; the only surviving "floor"
references are comments and a campaign-harness DETECTOR that checks which
engines DELIVERED, never one that routes), `accepts_still` means the dispatcher
mints this lane's still, so absent means MINTING FAILED.

## The dark floor now has no occupant -- and is KEPT, which REVISES lane 16

Lane 16's ledger entry said: when the last family rules, the `else:` branch is
dead code and should be DELETED. **On reaching it, that was revised**, and the
revision is the interesting part of this lane:

* The branch is reached by a family with `uses_still = False`, which is a
  **documented capability of this shelf** -- the attribute's own comment says
  "False families always synthesize a procedural floor". There is no such family
  registered today.
* So this is a **control with no occupant, not dead code**, and lane 4 already
  ruled on that shape: *a control whose last occupant leaves gets REWRITTEN with
  the reason, never deleted, because the invariant outlives every occupant.*
* Deleting it would also strand `_lavfi_source` and `ffmpeg_lavfi_floor_cmd`.

Both halves of "kept deliberately" are now asserted, because a kept branch with
no test is indistinguishable from one left behind:

* `test_the_synthesised_floor_is_UNREACHABLE_from_every_registered_engine`
  walks the registry and fails if any engine can reach it -- so if a future
  family DOES, its author is told to record the ruling rather than letting a
  black beat quietly become reachable again.
* `test_the_synthesised_floor_STILL_WORKS_for_a_uses_still_False_family`
  renders through it via a minimal stand-in family, so the capability is proved
  functional rather than assumed from the fact that it compiles.

## A second scope guard fired, from a different sprint

Lane 16's fired lane 15's guard. This lane fired **Sprint B's**:
`test_still_word.py::test_still_flat_sibling_unchanged_no_require_still`, which
had asserted since 2026-07-03 that `still_flat` still had the always-renders
floor -- the scoping guard for `still_word` being the FIRST family to refuse.

It is rewritten rather than deleted. Sprint B's reasoning was not wrong, it was
narrow: *"a silent black floor would swallow a mint failure exactly where it
matters"* turned out to matter everywhere, not only on a word card. The test now
records that `still_word` was first, that all four have since ruled, and that
the base default is still opt-in so a new cheap family inherits nothing.

## The solo smoke -- LIVE PASS, two legs

Stock `default` boot; box reset per CLAUDE.md section 4 first.

| | LEG A -- real still | LEG B -- the REFUSAL |
|---|---|---|
| Harness | `--engine still_flat --frames 100 --portrait <png>` | `--engine still_flat --frames 100 --expect-fail "requires a base still"` |
| Prompt id | `94588160-24fe-4cc2-9153-27cee1d18449` | fail-closed, NAMED |
| Canvas PROBED | **832x480** | n/a |
| Frames PROBED | **100** exactly | n/a |
| Rate / codec | 25/1, h264, yuv420p | n/a |
| Audio | **zero audio streams** | n/a |
| sha256 | `56d48f215d58868c924b9346c175b90b03ecfd3fa260f63392bff1b638393921` | n/a |

## The still shelf, finished

| family | refuses a missing still? | ruled by |
|---|---|---|
| `still_word` | yes | Sprint B, 2026-07-03 (first) |
| `still_motion` | yes | lane 15 |
| `still_pan` | yes | lane 16 |
| `still_flat` | yes | **lane 17** |
| the BASE default | no | so a new family opts IN, never inherits |

## What the QA pass found -- including a FALSE-PASS in this lane's key test

Nine findings, all fixed. Two matter beyond bookkeeping:

**1. The reachability test could have produced a FALSE PASS.** Its first draft
skipped any subclass whose `render_clip` override did not literally mention
`_require_still` in its source. A future family that overrides `render_clip` but
**delegates with `super().render_clip(...)`** satisfies that: its short body
mentions nothing, its method object differs from the base -- and it would have
been silently skipped WHILE ACTUALLY RUNNING the base's floor logic. That is a
false pass in the sole evidence for "the floor is unreachable", which is worse
than having no test at all.

Rewritten FAIL-CLOSED: an override must be named in a `DECLARED_OVERRIDES`
table with the reason it cannot reach the branch, and an undeclared one fails
the assertion instead of being skipped. `mesh_stage` is the one entry.

**2. A misleading RUNTIME LOG, not just a comment.** `render_driver` warned, on
every missing scene still, that *"a cheap family (still_pan/still_flat)
synthesizes its dark floor; a still-REQUIRED engine (still_word/ltx_audio_in)
fails LOUD"*. Wrong since lane 15 and emitted to the operator at exactly the
moment they are debugging a missing still -- the reader most likely to be
misled, and against the "clean logs" rule. It now says all four still families
fail loud.

The remaining seven were the same stale "always renders / fallback terminus"
claim in `StillFlatFamily`'s own docstring (in the class this lane edited),
`StillPanFamily`'s narration of lane 17 as pending, `_CheapFamilyBase`'s
`uses_still` and `_require_still` comments, `ffmpeg_lavfi_floor_cmd`'s docstring
(which predicted this lane's ruling and needed the outcome), `slot_matrix`'s
`DEFAULT_VIDEO_BASELINE` comment, and a self-contradiction inside the ledger
test's own docstring -- it still said `still_flat False -- LANE 17'S CALL, not
yet made` seven lines above `"still_flat": True`.

**The pattern across lanes 15-17 is worth naming:** a behaviour change on a
shared shelf leaves stale prose in roughly a dozen places, and they surface a
few at a time as each lane lands. The QA pass has caught them every time; the
grep to run is the CLAIM ("always renders", "terminus", "floor"), not the file.

## Deliberately NOT done here

**No VRAM number, no cost row** -- CPU/ffmpeg lane, G4 exempt.
**`still_word` untouched** -- lane 18 verifies it (both its halves are already
done: it always had the refusal, and lane 15's base fix gave it the ffmpeg gate).
**No profile, variant, workflow or `ENGINE_MATRIX.md` change** -- nothing declared.
