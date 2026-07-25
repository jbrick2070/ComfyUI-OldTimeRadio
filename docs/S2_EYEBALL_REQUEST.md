# S2 -- OPERATOR EYEBALL REQUEST (still-plans build)

**Written 2026-07-24 23:19 local by the CODER WINDOW A autonomous run.**
HEAD when this file lands: `a98b1d5d` on `v2.0-alpha` (S1 done).
Spec of record: `docs/2026-07-25-still-plans-locked-build-spec.md` @
`84328aa1` (section 11 chunk table).
Prior work in this arc, all green:

  - S0a  @ `33c4d8cf` -- characterization fixture (31 engines x 8
    configurations locked at HEAD).
  - S0a-b @ `e60185a0` -- isolation-property amendment (mutate one
    engine's plan proxies -> prove OTHER engines' rows byte-identical;
    mixed-policy episode -> prove per-role render_decisions equal each
    role's single-engine baseline).
  - S0b  BLOCKED-on-kibitz @ `c8db4c92` -- `docs/S0b_KIBITZ_NEEDED.md`
    filed instead of half-landing the routing freeze (cross-module
    atomic refactor beyond a single autonomous window's budget without
    silently breaking the routing-is-frozen-first invariant).
  - S1   @ `a98b1d5d` -- `nodes/_otr_shared/still_plan_helpers.py`,
    31 per-engine `still_plan` class attributes, and
    `tests/test_still_plan_audit.py` (6 tests). Nothing reads the plan
    yet; S2 wires it in.

## What S2 IS (per spec section 11)

Atomic cutover of the seven still-plan consumers + all six
`_SCENE_INIT_FAMILIES` call sites to READ the plan (via the S1 helpers
+ resolve_row_aspect). The HuMo expectations flip during S2:

  **OPERATOR EYEBALL: HuMo announcer/music stills go 832x1216 -> 832x480.**

That is the one visible pixel-shape change S2 introduces. Every other
cell in the S0a fixture (authored / materialized / render_decisions
across the 8 configurations x 31 engines) MUST diff to zero -- the spec
is explicit: "S0b does not change any still dimension" and S2's spec-named
delta is the four HuMo rows.

## The eyeball request

Two questions the operator sees BEFORE S2 begins production edits:

1. **The HuMo announcer/music cutover.** Today the still-plan proxies
   (`OTR_VideoDirector._role_aspects`) route the `humo` and `humo_1.7B`
   engines to a PORTRAIT still (832x1216) for the announcer/music bookend
   roles when they are ever picked into those slots. The spec's S2
   corrects this to a WIDE still (832x480) via the `_169` siblings'
   render_aspect -- because a HuMo announcer/music beat renders LANDSCAPE
   16:9 in production (the operator watches the finished episode on a
   landscape display), and a portrait still fed into a landscape render
   pillarboxes the face. The S1 plan already encodes the intent:
   `_HUMO_STILL_PLAN`'s portrait row is `aspect="inherit_engine"`, so
   `resolve_row_aspect` reads the engine's shipped `render_aspect` --
   which is `"wide"` on `humo_1.7B_169` / `humo_14B_169` today. The
   S2 cutover WIRES that reading in.
   - **Question:** OK to flip? The change is intentional per spec, but
     it does change a headline shape a live render will produce.
     Confirmation before the S2 coder begins.
2. **The routing-freeze prerequisite.** S2 assumes S0b's frozen
   `routing_state` is available (spec section 11: S2 "atomic cutover of
   all seven sites"; the sites read `routing_state.effective_video_models`
   per section 9). S0b did NOT land in this window (see
   `docs/S0b_KIBITZ_NEEDED.md`). Two forward paths:
   - **Path A -- unblock S0b first**, then execute S2 against the frozen
     `routing_state`. Cleanest match to the spec, but re-opens the S0b
     scope reality the prior autonomous window's handoff captured.
   - **Path B -- execute S2 against the current live-env resolvers**,
     then re-wire when S0b lands. S1's `resolve_row_aspect` accepts
     both key shapes (`render_aspect` / `engine_render_aspect`) precisely
     so the S1/S2 boundary is not sensitive to S0b's exact facts-dict
     key names; the trade-off is that S2's `required="when_ltx_i2v_enabled"`
     evaluator has to read `os.environ["OTR_ENABLE_LTX_I2V"]` directly
     in the interim, which is exactly the "indirect live read" S0b was
     meant to close.
   - **Operator preference?** Path A honors the spec's ordering; Path B
     unblocks S2 immediately but retains the ordering defect S0b exists
     to close.

## What this window is NOT doing

- Not touching S2 code (halt gate; needs the operator eyeball).
- Not touching S3 (shim + stale-prose deletion; downstream of S2).
- Not touching S4 (LIVE requires two fresh-boot render cycles).
- Not reverting anything.

## Suggested next-window kickoff line

Depending on operator answers above:

> resume the OTR build -- you are CODER WINDOW A per GO_FORWARD
> "Window packing"; execute S0b atomically per `docs/S0b_HANDOFF.md`
> (kibitz r3+r4 at CURRENT HEAD `a98b1d5d` first); then S2 with the
> HuMo 832x1216 -> 832x480 cutover confirmed; then S3; then S4 live.

Or, if the operator prefers to skip S0b for now:

> resume the OTR build -- you are CODER WINDOW A per GO_FORWARD
> "Window packing"; execute S2 (Path B: read live env for
> when_ltx_i2v_enabled in the interim, HuMo 832x1216 -> 832x480 CONFIRMED
> by operator); then S3; then S4 live; leave S0b's routing freeze on
> the KIBITZ_NEEDED docket.

## Two-strikes tally

Zero solo attempts consumed on S2. This is a gate stop, not a fix
attempt. Any S2 attempts start fresh in the next window.
