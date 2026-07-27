# B6 QA findings -- freeze the ltx_8gb recipe in CODE

Window A, 2026-07-27. Panels: pre-push (3 Sonnet lenses: correctness /
decorative-test / ledger-completeness), post-fix (2 Sonnet lenses: fix
verification / blast radius). $0 external spend -- no Codex, no cloud
roundtable, two-strikes never invoked.

Every finding below was GROUNDED against the real Windows files before it was
acted on. Panel claims that did not survive grounding are recorded at the end
with the reason, per the standing rule that the panel proposes and Claude
disposes.

## What B6 shipped

`LTX8_RECIPE_V1` in `nodes/_otr_video_engines/eng_ltx_8gb.py` is the frozen
recipe. Its env vars bind ONLY under `OTR_LTX_8GB_PREQUALIFICATION`; outside
that they are NAMED in a warning and ignored **without being parsed**.
`OTR_LTX_8GB_MAX_FRAMES` is deliberately excluded -- it is a render-length
CEILING, not a recipe value, and B4's pre-render refusal reads it.

Honest framing, restated because it matters: v1 is **today's shipped
defaults, not a measured selection.** The judgment orders "mechanics first,
MEASURE second, freeze third" and no measurement has happened. Freezing now is
behaviour-preserving on any box that set nothing, and it is reversible --
prequalification measures and produces v2.

## Defects the panel found in already-green, mutation-proven code

### 1. `OTR_LTX_8GB_NEGATIVE` was never demoted (lens A + lens C, independently)

`_build_graph` read it straight from `os.environ` on every leg. It is a RENDER
INPUT: two boxes with different stale values produce visibly different clips
and both stamp the same recipe receipt -- the exact defect B6 exists to close,
left open for the one knob nobody listed. Folded into the recipe behind a
`_negative_prompt()` accessor. The per-shot `negative_prompt` still wins; only
the server-boot channel is closed.

Worth recording: `render_driver.build_request_from_shot` never populates
`negative_prompt` for video shots, so before this fix the boot environment was
the **sole** author of the negative conditioning.

### 2. Four tiled-decode geometry knobs were undemoted (lens A + lens C)

`OTR_LTX_8GB_VAE_TILE` / `_VAE_OVERLAP` / `_VAE_TEMPORAL` /
`_VAE_TEMPORAL_OVERLAP` in `_decode_inputs`. Latent -- reachable only while
tiled decode is on, which the frozen recipe keeps off. Frozen anyway: the day a
measured v2 flips tiling on is the day four boot-environment values start
deciding what a published clip looks like, with nothing naming them.

### 3. A prequalification sweep stamped the PRODUCTION receipt (lens C)

The ledger-integrity defect B6 itself created. Under the consent act the knobs
genuinely bind, so a sweep's clip may share none of v1's values -- while
`recipe` rides `_clip_from_raw` -> the manifest row -> the render-batch receipt
-> `stamp_durable(meta.render_engines)`, a DURABLE ledger a published episode
carries. Nothing downstream could tell a sweep artifact from a production one.
Fixed with `recipe_receipt()`, which appends `+prequalification`; both stamp
sites go through it, and a source-level guard pins that they always will.

### 4. Two range-check implementations with OPPOSITE failure modes (lens D)

Introduced by fix 2. `_resolve_render_config._num` raised MALFORMED_CONFIG on a
bad value; the geometry's `_i()` swallowed it and substituted the default. So
the tile knobs were the single knob on this adapter that failed OPEN -- a sweep
could mistype the value it was measuring, render at something else, and stamp a
receipt saying it had measured it. Hoisted to one `_config_number` method that
both use, plus `_VAE_TILE_BOUNDS` from the live `/object_info` capture so a
value under the NODE'S OWN floor is refused by name instead of dying inside
ComfyUI.

### 5. Accessor defaults were literals, not the recipe (lens A)

`_t5_device` / `_tiled_vae` defaulted to `"cpu"` / `"0"` under prequalification
instead of reading the frozen dict. Dormant only because v1's literals coincide.
A sweep that opens the knobs and re-exports some of them would then measure a
third configuration. Both now derive from `LTX8_RECIPE_V1`, and the tests pin it
by `monkeypatch.setitem`-ing the recipe to a different value -- otherwise the
fix would be untestable until v2.

### 6. `_tiled_vae` empty-string semantics diverged (lens D)

`get(name, dflt)` returns `""` for an exported-empty var -- not truthy -- so the
knob would read OFF against a frozen default of ON. Every other accessor treats
empty as unset. Now `or dflt`.

## Decorative tests the panel caught (lens B, all confirmed)

- **Neither warning's DIRECTION was pinned.** Both bodies name the knob, both
  interpolate the recipe, and both contain the substring `PREQUALIFICATION` --
  it is inside the env var's own name. Swapping the two bodies stayed green.
  Now pinned on `FROZEN` and `measurement run`, each asserted present in one
  branch and absent in the other.
- **`test_the_RESOLVED_RECIPE_VALUES_reach_the_nodes_that_consume_them`** was
  comparing the resolver against itself: post-freeze a clean env makes it return
  the frozen constants, so a hard-coded `8` / `1.0` / `"euler"` in `_build_graph`
  would compare EQUAL. Its own docstring claimed to catch exactly that. Now runs
  under the consent act with values that differ from every frozen one, and
  asserts they differ before comparing.
- **`assert "FROZEN" not in caplog.text`** was vacuous -- nothing can log with
  no knob set, so it passed with the freeze deleted. Now asserts silence,
  scoped to this adapter's logger so an unrelated warning cannot make it red.
- **The key-set test** compared two sets to each other and never to a literal;
  deleting a whole branch kept it green. Now also pinned against the literal set.
- **The "captured at resolution time" test** was guaranteed by namedtuple
  immutability, and its comment claimed something false (`session_identity` is
  explicitly NOT cached, so a fresh resolution *does* move). Now asserts both
  halves: the held snapshot does not move, a fresh resolve does.
- **Three `_ENVS` scrub lists** were incomplete while their own comments claimed
  completeness -- the T-6 leak class. Each now carries a test asserting
  `set(_RECIPE_ENV_KEYS.values()) <= set(_ENVS)`, so the list cannot fall behind
  the recipe again.

## Mutation results

Two rounds, both with named CONTROL (semantically equivalent) mutants to prove
the harness discriminates rather than reporting red on everything.

- Round 1 (the freeze): **13/13 real mutants caught**, both CONTROLs survived.
- Round 2 (the panel-driven fixes): **10/10 real mutants caught**, both CONTROLs
  survived.

Mutants included: consent always/never granted; presence-counts-as-consent;
nothing reported as ignored; empty var counts as an override; production leaks
the accessor-owned keys; each accessor ignoring the freeze; the ceiling frozen
too; production parsing before ignoring; both receipt directions inverted; both
stamp sites reverting to the bare constant; the demotion notice losing a knob.

## Panel claims DISCARDED after grounding

- **"The render boundary should call `profile_max_render_frames` like WAN
  does" (lens A #1, lens C #3).** The split is real but it is NOT B6's, and the
  live risk today is zero: the env is unset in production (default 161) and the
  profile is unpinned (161), so the two owners agree. Pinning the profile to 97
  would change how a 237-frame beat partitions -- a production planning
  decision on the eve of 7d, not a cleanup. Recorded as an OPEN BUG instead,
  with the shape spelled out in the preset's own `_ceiling_note`.
- **"Make `_draw_models` draw the recipe on the credits card" (lens C #2).**
  The claim grounded: `otr_credits_roll` builds `models["video_suffix"]` from
  the recipe and nothing reads it. But the DURABLE LEDGER does carry the recipe,
  so this is a display gap, not a hole in the ledger. Fixing the credits
  renderer is a different chunk. The adapter's docstring was narrowed to say
  exactly what is true, and the gap is an OPEN BUG.
- **"The arc judgment says MEASURED and the code freezes unmeasured defaults"
  (lens E #3).** True, and deliberately not "fixed": a judgment document is a
  RECORD of what was decided, not a living doc. Rewriting it to match what
  shipped would destroy the evidence that the ordering was departed from. The
  departure is stated in the code and in GO_FORWARD instead.
- **"`assert caplog.text == ""` is brittle" (lens E #5)** vs **"it must not be a
  substring check" (lens D #4).** Both right. Resolved by scoping to
  `logger="OTR.video.ltx_8gb"` and filtering records by name -- silence proven,
  without coupling to global logging.

## Gate

Full suite **7213 passed / 27 skipped / 1 xfailed** (7158 before B6), Bug Bible
**17 passed**,
canonical workflow untouched (B6 adds no node, widget or link -- it CLOSES an
env channel; `workflows/otr_canonical.json` contains no `ltx_8gb` reference at
all, so CLAUDE.md section 0 is satisfied without an edit).
