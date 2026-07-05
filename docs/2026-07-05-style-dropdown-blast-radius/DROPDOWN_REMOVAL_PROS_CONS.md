# Style dropdown removal — pros/cons (3-way Sonnet fan-out, 2026-07-05)

Question: should the `style` COMBO widget be deleted entirely (zero manual
override — every episode's style comes purely from the deterministic
`select_style()`/`build_story_contract()` engine keyed on `cast_seed`), or
kept and repopulated with all 100 catalog entries + one sentinel (so a
user can still hard-pick a specific style)?

Three independent Sonnet subagents each took one angle, grounded against
the real repo, read-only (no code touched, no conflict with any other
active coder window). All findings verified by direct citation.

## Finding that changes the whole question: the combo is ALREADY disconnected

The deterministic catalog engine (`build_story_contract()`,
`nodes/OTR_LedgerScriptWriter.py:3340`) is called with signature
`(cast_seed, script_brief, news_seed, meta)` — it has NEVER read the
combo's value. The "forced_slug" idea from the earlier kibitz rounds was
closing a gap that doesn't exist in a vacuum — today, picking a combo
value does nothing to which of the 100 catalog styles actually gets used.
On top of that, the combo's OWN current 10 choices ("closed room
suspense", etc.) aren't even snake_case catalog slugs — feeding one into
`get_style()` as-is wouldn't match anyway.

## Architecture / positional-widget risk (agent 1)

`style` is widget index 8, `style_custom` is index 9 — adjacent,
immediately followed by `creativity` (10) through `visual_style` (26).
Cutting `style_custom` ALONE already forces re-indexing 17 downstream
widgets and two hardcoded test assertions (`wv[8]`, `wv[24]`) in the same
edit — that cost is unavoidable regardless of what happens to `style`.
Extending that same one-time reindex to also drop `style` (index 8) is
marginal additional work, not a second separate risk — it's bundled into
work already required this sprint. Repopulating `style`'s choice list
instead (keep the widget, swap its allowed values) is metadata-only and
zero positional risk on its own, but that framing undersells the real
comparison once you weigh what deleting saves elsewhere (below).

## Creative / UX impact (agent 2)

The 100 catalog entries ARE meaningfully distinct writing (differentiated
sound-world/story-engine/ending per entry) — hand-picking has real
creative value in principle. But: the combo's existing tooltip already
describes the OLD two-pass LLM inventor, a promise the architecture
doesn't keep even today; "let the story decide" is the sole value ever
shipped in the frozen workflow default, meaning real usage essentially
never exercises manual selection; and a 100-item flat alphabetical combo
is a worse creative tool than either full trust in the engine or a
proper future browse-by-mood/tag UI. Verdict: delete now, build a real
hand-pick UI later if wanted, rather than reviving a disconnected list.

## Wiring / dead-code implications (agent 3)

If the combo is deleted, `forced_slug`/`label_override` never ship at
all — `build_story_contract()` stays exactly as it is today, zero new
parameters, zero new branch, zero new reconciliation between the combo's
non-slug strings and the catalog's real slugs. Confirmed: nothing outside
`OTR_LedgerScriptWriter.py` reads this widget's value. Keeping the combo
requires building real slug validation (today's 10 strings aren't slugs),
threading `forced_slug`, and reconciling it against the SEPARATE existing
`resolved["style"]` three-way resolver so the two don't silently diverge
again — the exact class of bug this whole rip-out was started to kill.

## Recommendation

**Delete the `style` combo widget entirely**, in the same edit that
removes `style_custom` (one combined reindex pass, not two). This is the
simpler, lower-total-risk, and more honest option: it ships zero new
speculative API surface (`forced_slug`/`label_override` never need to
exist), it removes a control that's already disconnected from what it
claims to do, and it matches how the workflow is actually used today. If
hand-picking a specific style is wanted later, build it as a real,
separate feature against `_otr_style_catalog.get_style()` directly —
not by reviving this list.

This changes plan section 1 for the next kibitz round: no `forced_slug`/
`label_override` parameters needed at all; `build_story_contract()` ships
unchanged; `_resolve_inputs`'s three-way `style`/`style_custom`/`llm_auto`
resolver collapses to nothing (both `style` and `style_custom` widgets are
gone); the workflow JSON removes TWO adjacent widget slots (8 and 9) in
one pass, with slots 10-26 shifting down by two and both hardcoded test
indices (`wv[8]`, `wv[24]`) updated in the same commit.
