# LANE 1 QA findings -- freeze the WAN recipes in CODE

CODER window, 2026-07-27 (remote Cowork). Panels: a 3-lens pre-push Sonnet
fan-out (correctness/blast-radius, decorative-test, ledger-completeness) plus
two mutation rounds. $0 external spend -- no Codex, no agy, no cloud roundtable;
the two-strikes law was never invoked (no fix needed a third attempt).

Every finding was GROUNDED against the real Windows files through Desktop
Commander before it was acted on. Claims that did not survive grounding, or that
were correct but out of scope, are recorded at the end.

## What LANE 1 shipped

`wan_recipe.py` is the MECHANISM -- consent-act detection, presence-only
demotion, receipt suffixing, and one range-checked fail-closed parser. Each
adapter owns its own DATA: `WAN_TI2V_RECIPE` / `RECIPE_WAN_TI2V` /
`OTR_WAN_TI2V_PREQUALIFICATION` (`71753cb4`) and `WAN_I2V_RECIPE` /
`RECIPE_WAN_I2V` / `OTR_WAN_I2V_PREQUALIFICATION` (`3acc7fed`).

Honest framing, restated because it matters: both v1 dicts are **today's
shipped defaults, not a measured selection.** No WAN sweep has run. Freezing
them is behaviour-preserving on any box that set nothing, and reversible --
a prequalification run measures and produces v2.

**PER-ADAPTER CONSENT VARS, not one shared switch.** A single switch would open
both tiers at once and stamp `+prequalification` on a clip that had rendered
with its frozen recipe. A receipt that lies in the safer direction still lies.

**WHAT IS DELIBERATELY NOT FROZEN, and why WAN differs from ltx here:**

- `OTR_WAN_TI2V_MAX_FRAMES` is a render-length CEILING, and unlike the ltx
  tier's it is a LIVE SHIPPED CHANNEL -- `config/profiles/otr_8gb_wan.json` sets
  both `launch.env.OTR_WAN_TI2V_MAX_FRAMES` and `video.max_render_frames`.
  Folding it into the recipe would have silently retired the 8 GB tier's launch
  contract, which is `PBUG-20260723-02` itself.
- The weight NAMES and their loader-class selectors stay live TOGETHER. The
  loader class is inferred from the resolved basename, so freezing the class
  while its filename still moved would give one fact two owners.
- `wan_i2v`'s sampler stays `uni_pc`, NOT the portable-floor `euler` that
  `wan_ti2v` freezes. wan_ti2v is the 8GB/Mac/AMD floor and carries a
  portability whitelist; the 14B lane never had one. The freeze preserves
  behaviour; it does not add policy.
- The six `wan_i2v` knobs keep their UN-NAMESPACED `OTR_WAN_*` names. Renaming
  an operator-facing knob is an operator's call, and the freeze already removes
  the power that made the missing namespace dangerous. Flagged, not changed.

## Defects the fan-out found in already-green code

### 1. `eng_wan_i2v` measured a VRAM peak, logged it, and threw it away (lens C)

`render_clip` started a `VramPeakProbe`, read `render_peak`, logged it, and
returned `{"out_path", "frame_count"}`. So `_clip_from_raw`'s
`raw.get("vram_peak_mb")` was `None` on every wan_i2v clip forever, and
`render_shot` silently fell back to an INSTANTANEOUS post-render read -- which
can under-report the true peak, and that weaker number rolls up into the
episode figure and the credits card. NEWBUG-1 fixed exactly this for `wan_ti2v`
on 2026-07-20 and never reached the sibling. One owner, no other claimant, one
line. Fixed in `3acc7fed`.

### 2. Four test gaps in the new suite (lens A + lens B)

- **No key-set parity test.** The ltx reference has one; the adapter's own
  comment names the exact risk ("a return shape that varied by mode would hand
  the next reader a KeyError that only reproduces under the consent act") and
  nothing pinned it. Added for both adapters, pinned against a LITERAL as well
  as against the other branch.
- **No opposing `scheduler` override.** The resolver-against-itself trap
  surviving on one field: a hard-coded `"simple"` in `_build_graph` would have
  compared EQUAL to the frozen value. `beta` is legal and differs, so the test
  exists now. The SAMPLER cannot get the same test -- `_PORTABLE_SAMPLERS` has
  exactly one member, so no legal value opposes the frozen one. That is
  inherent, recorded in the test, and gets its twin if the whitelist grows.
- **`config_text`'s empty-string fallback was untested.** `config_flag` got a
  dedicated test for the same `or`-based contract; `config_text` is a different
  function and had none. A regression to `get(env, frozen)` would push an EMPTY
  negative prompt into the graph under a measurement run.
- **Two of the four tile knobs were never independently exercised.** All four
  share one `_i(key)` closure, and a swapped `_RECIPE_ENV_KEYS` entry between
  `vae_overlap` and `vae_temporal_overlap` would have been invisible to the
  whole suite -- a set-vs-set check cannot see a permutation. Now parameterised
  over all four, each asserting its own graph input moved AND that every other
  one stayed frozen.

### 3. Two of my own tests were decorative (lens B, both confirmed)

- `test_the_consent_switch_is_namespaced_to_THIS_adapter` compared the imported
  constant to the literal on its own source line. No implementation change
  could break it. Replaced by a test that SETS the var and observes the
  behaviour change.
- `test_the_version_lives_INSIDE_the_receipt_string` asserted a literal ends in
  `_v1`. Replaced by the drift that can actually happen: the number of
  `WAN_*_RECIPE_V*` dicts must equal the version in the receipt string, and the
  active binding must BE the newest dict -- so landing a v2 and forgetting to
  bump the receipt fails.

## The mutation round found what three lenses did not

Two rounds, both with named CONTROL mutants to prove the harness discriminates
rather than reporting red on everything.

- Round 1a (ti2v + the mechanism): **16/16 caught**, 2 CONTROLs survived.
- Round 1b (i2v): **4 of 10 REAL mutants SURVIVED.** That is the headline.
- After the fixes: **20/20 and 10/10 caught**, all 4 CONTROLs survived.

The four survivors, each a distinct blind spot no lens had named:

1. **A renamed consent constant was undetectable.** Every test set
   `PREQUALIFICATION_ENV` -- the imported constant -- so renaming it renamed
   what the tests set too, and an adapter reading a var no operator will ever
   set stayed green. Tests now set the DOCUMENTED LITERAL an operator types.
   The same hole existed on `wan_ti2v` and was fixed there too.
2. **`recipe` dropped from `render_clip`'s return survived**, because the
   receipt was only ever checked on a HAND-BUILT raw. The test constructed the
   thing it was verifying -- the chunk-6 shape where a test's own builder agreed
   with the bug.
3. **`vram_peak_mb` dropped survived** for the same reason.
4. **`shift` escaping back to an inline `os.environ` read survived**, because
   the production-leg test set steps/cfg/sampler/negative and not shift, while
   the consent-act test AGREED with the mutant.

Fixes 2 and 3 needed a test that drives the real `render_clip`; it stubs the
encode/probe tail so it needs no ffmpeg and no GPU, and runs for both adapters.

## Panel claims recorded but NOT acted on

- **"`eng_wan_i2v` still has the whole defect" (lens A #1).** Correct when
  written -- it was chunk 1b, which landed at `3acc7fed`. Worth recording that
  lens A also verified the blast radius honestly: no shipped profile routes the
  in-process `wan_i2v` today (`otr_cloud_lanes.json` uses the distinct
  `cloud_wan_i2v_audio` adapter), so it was a live landmine rather than a live
  fire.
- **"The sampler needs an opposing-override test" (implied by lens B #1).**
  Structurally impossible while `_PORTABLE_SAMPLERS` has one member. Recorded in
  the test rather than faked.
- **Redundant-but-safe pairs (lens B #6, #7).** The receipt tests and the
  env-map set checks are each tautological in isolation and complementary in
  pairs. Left as they are; the mutants that matter are caught elsewhere.

## Gate

Full Windows suite **7291 passed / 27 skipped / 1 xfailed** (7226 before LANE 1),
Bug Bible **17 passed / 24 skipped / 3 xfailed**, AST/BOM/zero-byte/UTF-8/ASCII
clean on all six touched files, canonical workflow
`9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
byte-identical -- LANE 1 adds no node, widget or link; it CLOSES an env channel,
and `workflows/otr_canonical.json` contains no `wan_i2v` or `wan_ti2v` recipe
reference at all, so CLAUDE.md section 0 is satisfied without an edit.
