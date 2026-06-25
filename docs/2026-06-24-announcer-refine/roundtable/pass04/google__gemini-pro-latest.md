<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan is highly converged and the scope cuts successfully removed the riskiest wiring. However, there is a direct code contradiction in Step E that violates the input starvation guarantee, and a potential `AttributeError` in Step G.

MUST-FIX BEFORE BUILD:
1. **[STEP E] Contradiction in Safe Open call site**: The text says "DO NOT pass `script_brief` content", but the provided code snippet explicitly passes `script_brief=script_brief`, which feeds the spoiler directly into the composer.
   *Fix*: Change the snippet to pass `script_brief=("" if _style_grammar_on else script_brief)` to guarantee input starvation at the Python level.
2. **[STEP G] `AttributeError` risk on `ending_template`**: Step G assigns `_ending_template = contract.ending_template`. Step B's definition of `StoryContract` does not list `ending_template` as a field (only slug, label, ending_tag, grammar, story_engine).
   *Fix*: Either explicitly add `ending_template: str` to the `StoryContract` dataclass in Step B, OR change Step G to fetch it safely: `_ending_template = _OTRSTYLE.get_style(contract.slug).get("ending_template", "")`.
3. **[STEP D] Invalid Python syntax**: The snippet uses `<meta/period or "">` which will cause a `SyntaxError`.
   *Fix*: Change to standard dictionary access: `meta.get("period", "")`.

SHOULD-FIX:
1. **[STEP B] Scope of `resolved`**: The snippet uses `resolved.get("news_seed","")`. [ASSUMPTION] `resolved` is a local dict in scope at this exact line. If the class uses `self.resolved_options` or a `kwargs` dict, this will fail.
   *Fix*: Use `self.resolved_options.get("news_seed", "")` or the exact local variable name available at `run()` top.
2. **[STEP H] Ambiguous truncation formula**: The plan says "Truncation clamp (the `max(0,...)` slice fix + reserve formula)" but doesn't provide the code, leaving it up to the builder to reinvent.
   *Fix*: Provide the exact replacement line: `new_intent = (new_intent[:max(0, _INTENT_MAX - len(fc) - 1)].strip() + " " + fc)`.

OPTIONAL / NICE-TO-HAVE:
- **[STEP F]**: When `close` is empty and `story_scaffold` is ON, explicitly log a warning (e.g., `logger.warning("News coda triggered but news_close_brief is empty")`) right before emitting `news_coda_fallback` to telemetry. This makes debugging input starvation much easier.

CUT THESE:
- None — plan converged. The cuts made in R3 (dropping the compact register tag and deferring the open spoiler belt) perfectly stripped the remaining over-engineering.

VERIFY-AT-BUILD checklist:
- Confirm `meta.get("period")` actually contains the era string for `SafeOpenBrief` (if empty, ensure the prompt still behaves gracefully).
- Confirm ledger lines have `beat_id` populated on the climax row at outro time (doc'd ledger:96).
- Confirm `news_close_brief` is distinct from `ending_change` and never empty at runtime.
- Update `OutlineRequest` asdict/repr snapshot fixtures to account for the new `style_grammar` default `""` key.
- Confirm `build_reroll_line_request` remains the ONLY LineRequest reroll-rebuild site (to ensure no future line-level metadata is lost on reroll).