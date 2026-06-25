<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — core design is sound but several wiring details are underspecified or unstable; must resolve before build.

MUST-FIX BEFORE BUILD:
1. [VALIDATOR] `validate_news_coda_bridge` signature lacks the `news_close_brief` parameter required for the n‑gram copy guard. The spec says it checks “no >=5-token verbatim run copied from the cleaned brief” but the function is declared as `validate_news_coda_bridge(text) -> (ok, cleaned)`. Fix: change signature to `validate_news_coda_bridge(text, news_close_brief_cleaned)` and pass the cleaned brief from the assembly step.
2. [FALLBACK FLOOR] Fallback pool selection uses `hash(str(cast_seed))` which is not deterministic across Python processes (hash randomization). This violates the “deterministic fallback” requirement. Fix: replace with a stable hash, e.g., `int(hashlib.md5(str(cast_seed).encode()).hexdigest(), 16) % len(POOL)`.
3. [ASSEMBLY] Normalization of the bridge’s ending punctuation is underspecified: “bridge ends with ':' or em dash (normalize if not)”. If the bridge ends with a period or other punctuation the resulting `coda` may be malformed (e.g., “bridge.: fact”). Fix: define normalization as: if bridge does not end with ':' or '—', strip any trailing punctuation (., !, ?) and append a colon and space before the fact. Alternatively, always ensure the bridge ends with a colon by appending one if missing, but handle the dash case explicitly.
4. [SEQUENCING] The validator’s n‑gram copy guard needs the cleaned brief, but the spec does not order the cleaning before validation. The assembly step cleans the brief with `clean_one_line(news_close_brief, max_chars=200)`. Fix: explicitly state that the brief is cleaned first, and the cleaned string is passed to the validator.
5. [CALL-SITE] The plan says to stamp `news_coda_no_brief` on the `LineResult` from `compose_announcer_outro` when `_style_grammar_on` is True and brief is empty, but does not specify how to add a flag to an existing `LineResult`. The `compose_flags` tuple is likely immutable. Fix: either modify `compose_announcer_outro` to accept an optional `extra_flags` parameter, or after the call create a new `LineResult` with `compose_flags = outro_res.compose_flags + ("news_coda_no_brief",)`. Specify the chosen method.

SHOULD-FIX:
1. [ASSEMBLY] The total coda length cap of ~300 chars is only a comment; no enforcement. Add a max‑length check in the final sanitizer (e.g., 320 chars) and truncate or fallback if exceeded.
2. [VALIDATOR] The optional outcome‑verb blocklist is “best‑effort”. If it triggers false positives it could reject valid bridges. Consider making it a hard reject with a curated list, or remove it to avoid unpredictable failures.
3. [REROLL] The reroll prompt alteration is minimal. Adding a small variation to the system prompt (e.g., “Use a different pivot phrase”) could improve diversity, but not required.

OPTIONAL / NICE-TO-HAVE:
- Make the bridge length cap (80 chars) a named constant.
- Add a check that the bridge does not introduce proper names absent from the premise context (low risk).

CUT THESE (over-engineering):
- None beyond the already‑dropped “≥1 news content token” and semantic outcome validators.

[ASSUMPTION] The plan relies on `script_brief` containing only the premise/setup and not the outcome. If `script_brief` sometimes includes the ending, the bridge could inadvertently reference the outcome. Verify that `script_brief` is indeed premise‑only.