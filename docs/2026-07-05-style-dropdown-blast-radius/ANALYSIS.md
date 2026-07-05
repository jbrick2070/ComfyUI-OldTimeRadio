# Style dropdown blast-radius analysis (2026-07-05)

**ANALYSIS ONLY -- no code touched.** Every claim below is grounded in the current
file state (Read/Grep on the real repo files), not memory.

## TL;DR

The little 10-item dropdown in the screenshot never went anywhere and was never
replaced. It is a *different, older* system than the 100-style list you're
remembering. Both are real, both are live in the code today, and they have
never been merged -- they run in parallel, disconnected, feeding the same
episode from two different directions. The dropdown is not a regression. It's
the original narrow system that a newer, richer system was quietly built
*next to* on 2026-06-24, without ever touching the widget.

## The four style surfaces (all real, all live, not one list)

| # | File | What it is | Size | Selection method |
|---|------|-----------|------|-------------------|
| 1 | `nodes/OTR_LedgerScriptWriter.py` `_STYLE_CHOICES` (line 267) | the literal COMBO widget you see in the node UI | sentinel + 10 hand-picks | user click, or sentinel |
| 2 | `nodes/OTR_LedgerScriptWriter.py` `_STYLE_PICKER_SEED_POOL` (line 297) | "inspiration" flavors fed to the 2-pass LLM inventor (`_otr_style_picker.py`) when the sentinel is picked | same 10 slugs, snake_case | LLM invents a NEW free-text descriptor; the 10 are just seed flavor, not the answer |
| 3 | `nodes/_otr_style_palette.py` `STYLE_PALETTE` | MusicGen's opening/closing/interstitial cue bank | pinned 1:1 to the same 10 slugs (test-enforced) | looked up by whatever `meta.style` string ends up being |
| 4 | `nodes/_otr_style_catalog.py` `STYLE_CATALOG` | **the 100-style set you remember** -- `"Operator-authored 100-style set (2026-06-24)"` | 100 entries, each with sound_world / story_engine / ending_mode | **deterministic sha256(cast_seed) hash draw. No LLM. No UI. Runs on every episode by default.** |

Surface #4's own file comment (`_otr_style_catalog.py` lines 30-31) literally
says its first 10 entries are *"the original ten, upgraded"* -- i.e. whoever
built the 100-list on 2026-06-24 took the exact same 10 slugs from surfaces
#1-#3, expanded and enriched each one, then added 90 more net-new entries
(11-100). It was built as an upgrade in spirit, but it was wired to a
completely different consumer than the dropdown.

## Why the small list "reappeared" -- it never left

Nobody ripped the 100-list out. It is running right now, on every episode,
by default (`_otr_config.STYLE_GRAMMAR_DEFAULT = True`, kill-switch env var
`OTR_ENABLE_STYLE_GRAMMAR=0`). What you're seeing in the screenshot is simply
a widget that the 2026-06-24 change never touched, because the 100-list
was built to solve a *different* problem:

- **The dropdown (#1) exists to pick a short tone label** for the outline
  prompt (`resolved["style"]`, e.g. "noir interrogation" or an LLM-invented
  phrase). It's the user-facing, pick-one-or-let-the-LLM-invent-one axis.
- **The 100-catalog (#4) exists to pick the CLIMAX SHAPE and SOUND WORLD**
  deterministically from cast_seed (`_OTRSTYLE.build_story_contract(...)`,
  `OTR_LedgerScriptWriter.py` line 3340). It was purpose-built (commit-tagged
  "KILL 2, 2026-06-24") to stop every episode's ending from collapsing into
  the same "console standoff / kill-switch" shape -- a narrower, specific
  fix, not a UI feature. Nothing in that work item asked for a dropdown
  replacement.

So: two forks happened off the same 10-slug root on the same day, for two
different purposes. One became a rich internal data table consumed by pure
Python. The other stayed a small combo box. Nobody was tasked with
reconciling them, so they didn't get reconciled.

## The part that's actually a problem, not just a stale UI

Every render today feeds the outline prompt **two independent, uncoordinated
style signals at once** (`OTR_LedgerScriptWriter.py` lines 3364-3377):

- `style=resolved["style"]` -- whatever the user picked from the 10-item
  combo, or whatever the 2-pass LLM invented.
- `style_grammar=contract.grammar` / `story_engine=contract.story_engine` --
  whatever the 100-catalog's sha256(cast_seed) hash landed on.

These two picks have **no relationship to each other.** The user (or the
LLM) can pick "mission control procedural" in the dropdown while the
catalog's deterministic draw silently hands the climax shape and sound
world of "cursed-object inventory" or "hotel switchboard thriller" to the
same episode. You cannot see which of the 100 was used until after the
render, by reading `meta.story_contract.slug` inside the ledger JSON --
there's no user control over it and no UI surface that shows it.

This exact failure pattern -- two parallel style-slug lists silently
drifting apart -- already caused a real production bug once before:
**BUG-LOCAL-216** (MusicGen halting mid-pipeline because the writer's seed
pool and the music cue palette had drifted). The fix was to hoist the
palette to one file (`_otr_style_palette.py`) and pin it against the seed
pool with a regression test (`tests/test_style_palette_drift.py`). That
test protects surfaces #1-#3 against drifting from each other. **It does
not know surface #4 exists.** The 100-catalog was added a month after that
fix and was never folded into the same drift guard. Confirmed by grep:
`_otr_style_palette.py` (MusicGen's cue lookup) has zero references to
`story_contract` or `contract.slug` -- MusicGen cues are chosen purely off
the small 10-slug axis and can never see or match the rich style the
catalog silently picked.

## What is NOT broken

- The 100-catalog itself is fine, self-validating (`validate_catalog()`),
  and every entry has a real climax-class ending template.
- The deterministic hash draw is intentional and documented as such
  (non-LLM, cast_seed-keyed, so a given episode seed always reproduces the
  same pick -- this is the project's C7 byte-identity requirement).
- The workflow JSON's frozen default (`widgets_values[8] ==
  "let the story decide"`) and the dropdown's position are both
  test-pinned (`tests/test_workflow_json_guardrails.py`) and match the
  litegraph positional-widget rule in this repo's CLAUDE.md -- so nothing
  here is silent widget drift in the BUG-LOCAL-097 sense.

## Options (analysis only -- pick one before anyone touches code)

**A. Leave it alone.** The two axes technically don't conflict at the code
level (one sets tone label, the other sets climax shape + sound world) --
they were designed as separate levers, just never surfaced together. Zero
risk, zero payoff on the confusion.

**B. Retire the small dropdown, make the 100-catalog the single style
source of truth.** Delete `_STYLE_CHOICES` / `_STYLE_PICKER_SEED_POOL` /
the 2-pass LLM inventor; wire the widget to expose (or randomly draw from,
with an override) `_otr_style_catalog.STYLE_CATALOG` directly, and route
`resolved["style"]` through `contract.label` instead of a separate draw.
This is the cleanbreak per this repo's own no-legacy-shim rule, but it
touches: the widget's `INPUT_TYPES` combo (positional widget --
`otr_scifi_16gb_full.json` widget-value reindex risk per this repo's
litegraph gotcha), `test_workflow_json_guardrails.py`, `test_style_
palette_drift.py`, `test_otr_api_companions.py`, the MusicGen cue palette
(`_otr_style_palette.py` would need 100 entries or a fallback), and the
2-pass LLM picker module (`_otr_style_picker.py`) would become dead code
to delete outright, not shim.

**C. Keep both axes, but surface the catalog's pick to the user.** Leave
the dropdown as the tone-label control; add a read-only stamp (ledger meta
already has `story_contract.slug`/`.label`) somewhere visible pre-render or
in the HUD/overlay, so the deterministic pick isn't invisible anymore. Low
risk, addresses the "I can't see what it actually picked" problem without
touching the widget or its test pins.

**D. Reconcile MusicGen's blind spot regardless of A/B/C.** Independent of
what happens to the dropdown, `test_style_palette_drift.py` should arguably
know about surface #4 so a future MusicGen consumer of `contract.slug`
doesn't repeat BUG-LOCAL-216. This is a guard-rail addition, not a UI
change, and stands on its own.

No code changes were made. This document only maps what exists and why,
for you to pick a direction.
