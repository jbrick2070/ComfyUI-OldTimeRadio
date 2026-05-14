# S28 Phase 3 — Producer audit b4 (`_otr_line_composer.py` caller-shape)

Per S28 plan §Phase 3 §Step 3.1. Audits each `_otr_line_composer.py`
consumer-side back-compat fallback to determine whether the producer
side already populates the field, and (per Rule D / Rule E) whether
a producer-side fix is needed before the consumer-side fallback can
be deleted.

| Site | Field read | Producer | Producer always populates? | Rule | Action |
|------|------------|----------|----------------------------|------|--------|
| `_otr_line_composer.py:468` docstring | `allowed_roster` field default = `frozenset()` | `OTR_LedgerScriptWriter` builds the roster via `build_allowed_roster(...)` after cast-lock + news_interpreter and passes it on every real call. | YES — empty default exists only to keep early-stage tests importable (dataclass ordering forbids non-defaulted fields after defaulted ones). | A | Update docstring to drop the back-compat framing; default stays as a dataclass-ordering artifact, not a tolerated legacy shape. No code change beyond the docstring. |
| `_otr_line_composer.py:856` docstring | `allowed_people` / `allowed_things` | `OTR_LedgerScriptWriter` populates both on every per-line LineRequest after the cast-lock + news-interpreter pass (`build_allowed_people_things_lookup`). | YES — pre-v2 callers that only set `allowed_roster` are extinct (every D.5+ writer site uses the new shape). | A | Drop the "back-compat callers that only set the legacy `allowed_roster`" tolerance branch + the comment that documents it. The NAMED ENTITIES block always renders from `allowed_people` / `allowed_things`. |
| `_otr_line_composer.py:1215` docstring + :1265 active_fn | `polish_generate_fn=None` fallback to `generate_fn` | `OTR_LedgerScriptWriter` builds `polish_generate_fn` via `_OTRML.make_polish_generate_fn(cache_entry)` inside a `try/except` that falls back to `polish_generate_fn = None` on factory failure. **Producer leak.** | NO — producer's best-effort `except` branch sets `None` when the factory fails. | A + D | Apply Rule D: producer fix is a one-line edit — drop the producer-side `try/except` so the factory always succeeds (or raises loudly). Then drop the consumer-side `polish_generate_fn if polish_generate_fn is not None else generate_fn` fallback. The `make_polish_generate_fn` factory is required infrastructure in v2.0; "older builds" without it no longer exist. |
| `_otr_line_composer.py:1492` comment | (same polish_generate_fn shape — refers to the same flow) | Same producer site as above. | Same. | A + D | Update the comment to drop the back-compat framing. The polish call always uses a populated `polish_generate_fn`. |

## Producer fix scope (committed before Step 3.2)

`OTR_LedgerScriptWriter.py:1495-1506` — drop the `try/except` that
silently sets `polish_generate_fn = None`. After the fix, the factory
call is required; any factory failure surfaces as a hard failure of
the script-writer node (the correct behaviour — polish is not optional
under the v2.0 contract; awkward sampling on a polish path is a
silent quality regression).

## Step 3.2 deletion targets (consumer-side)

After the producer fix lands, the four consumer-side fallbacks at
`:468` (docstring), `:856` (docstring + tolerance code), `:1215`
(docstring), `:1492` (comment) all become unreachable. Delete the
runtime branch at `:1265` (`active_fn = polish_generate_fn if ...
else generate_fn`) and replace the four annotations with forensic
comments.

## Audio-byte-identical risk

LOW. None of these sites touches the audio waveform. The
`polish_generate_fn` change affects LLM output text, not audio. The
audio-byte-identical regression runs after Step 3.2 anyway per the
plan's per-step regression battery (see §Step 3.2 regression).
