# Story-Grammar Build -- Chunks 4-6 (the WIRING) -- RESULTS

**Date:** 2026-06-24
**Branch / HEAD:** `v2.0-alpha` @ `4c9793b2` (== origin)
**Lever:** `OTR_ENABLE_STYLE_GRAMMAR` (env-only, **default OFF**, bundled with
`OTR_STORY_QUALITY_L12`). No workflow-JSON change.

## What shipped (3 commits)

- **C4 `762b20d7`** -- `_otr_line_composer.LineRequest.ending_template` field +
  the `Ending:` render block (only when populated); `build_sq_data(...,
  climax_role=...)` param threaded into `assign_beat_roles`; `_otr_config.
  style_grammar_enabled()`. All inert/dark -- nothing wires it yet.
- **C5 `e86adb59`** -- the WIRING goes live (still default-OFF):
  - `_otr_outline._assemble_outline`: announcer close gated via a DIRECT
    `os.environ.get("OTR_ENABLE_STYLE_GRAMMAR")` read. OFF = the exact
    pre-grammar string; ON = a non-outcome close (kept <= Beat.intent's 200-char
    cap). The close is **never removed** -- still an announcer beat, so budget
    validator #7 (announcer count) stays satisfied.
  - `OTR_LedgerScriptWriter` F2: when the lever is on, after `generate_outline`,
    `slug = select_style(outline.premise, meta, cast_seed)` -> the style's
    `ending_tag` becomes the climax `climax_role` threaded into `build_sq_data`
    (bundled with L12 so the role flows) -> the climax-class beat id + the
    style's `ending_template_for(slug)` are passed to the line composer's
    LineRequest for THAT beat only -> `meta.story_quality {style_slug,
    ending_tag, final_beat_crisis_nouns}` stamped. Fails soft (never breaks
    audio); on any error the climax stays `irreversible_choice` and the ending
    template is dropped.
- **C6 `4c9793b2`** -- `tests/test_story_grammar_wiring.py` (26 tests):
  default-OFF byte-identity golden, flag-ON ending-template render, announcer
  non-outcome + close-not-removed, `climax_role` threading, selector
  determinism.

## Gates (all green)

- **Targeted subset** (story_quality_l12 / style_catalog / line_composer /
  outline / story_quality_scan + the new file): 162 + 26 pass.
- **Full suite:** green **except the 5 pre-existing `267a53e` workflow-pin
  fails** (16gb-profile / workflow-structure / audio-wiring pins). Verified
  pre-existing by stash + rerun on the clean baseline -- NOT this sprint (it
  touches zero workflow JSON). `test_audio_byte_identical` GREEN (default-OFF).
- **Bug Bible:** 16 passed / 7 skipped / 3 xfailed (baseline).
- Touched .py files: UTF-8 no BOM, `py_compile` clean (AST parse), HEAD==origin.

## Deterministic A/B -- the ending-tag SPREAD (no GPU; pure `select_style`)

12 realistic OTR premises x 250 cast_seeds = 3000 sampled episodes.

| metric | baseline (lever OFF) | lever ON |
|---|---|---|
| climax = `irreversible_choice` | **100.0%** (forced console standoff) | **2.1%** (62/3000) |
| NON-doomsday share | 0% | **97.9%** (target >= 80%) |
| distinct ending classes | 1 | **9 of 9** |
| distinct styles chosen | n/a | 98 of 100 |

Lever-ON ending-class distribution: revelation 19.3%, reversal 14.5%,
unresolved_final_sound 13.1%, reconciliation 12.7%, bittersweet_parting 11.9%,
quiet_acceptance 11.3%, confession 8.5%, ironic_twist 6.7%, irreversible_choice
2.1%. **The forced "blow everything up" climax is demoted from 100% -> ~2%; the
climax SHAPE is now varied and >=80% non-doomsday.** select_style is
sha256(cast_seed)-keyed -> deterministic per episode (C7-safe), varied across
episodes.

## OPEN -- operator-gated LIVE LLM behavioral A/B soak

The remaining half of C6 is a LIVE render soak measuring whether the local
writer *obeys* the ending template at the final beat (the behavioral lift, vs
the structural spread proven above) + the shipped-text final-beat crisis-noun
density (`meta.story_quality.final_beat_crisis_nouns` + the scrub's
`ungrounded_crisis`). It is **operator-gated** per repo pattern (every
story-quality handoff: "operator flips the flag + eyeballs an N=3 re-soak"),
and the box was **not free**: `:8000` is the operator's **interactive Comfy
Desktop** (not a headless leftover), which CLAUDE.md notes cannot be relaunched
from the DC shell -- so I did not reset it or contend for the active GPU. Ollama
is up on :11434 (the local writer lane is ready).

**To run it (operator):** reset the box per CLAUDE.md S4 (selective CIM kill of
the ComfyUI `main.py` + any soak pythons -- never a blanket python kill), boot a
fresh headless server, then run a small N=3 baseline vs N=3 lever-on soak on the
canonical `workflows/otr_scifi_16gb_full.json`:

- baseline leg: env clean.
- lever leg: `OTR_ENABLE_STYLE_GRAMMAR=1` `OTR_STORY_QUALITY_L12=1`.
- measure per shipped ledger: `meta.story_quality.{style_slug, ending_tag,
  final_beat_crisis_nouns}` and scan the FINAL character line for the generic
  crisis-noun vocabulary (target ~0) + confirm the ending tags spread off
  doomsday. Default-OFF ships byte-identical; prod/main + tags GATED.
