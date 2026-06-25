# DYNAMIC NEWS-CODA SEGUE -- BUILD-SPEC (coda-segue pass02, post-R2)

R2 (implementability) converged on a tighter, safer split and killed two
unimplementable validators. Self-contained build spec below.

## THE RESOLVED SHAPE (R2)
- The bridge LLM sees the fiction's PREMISE/SETUP only (specific to tonight's tale)
  -- NOT the outcome (`ending_change` / `final_character_line`) and NOT the news
  facts. So it can be specific WITHOUT restating the fiction OR bleeding the news.
- The appended `news_close_brief` carries 100% of the news specificity.
- => DROP the ">=1 news content token" bridge requirement (unimplementable w/o NLP
  AND unnecessary -- the payload is the news; all 3 panel converged) and DROP the
  semantic "asserts no outcome" validator (unimplementable; use a tight length cap +
  the prompt + a small outcome-verb blocklist).

## NEW FUNCTION (cleanest: a dedicated coda composer -- keeps `compose_announcer_outro` UNTOUCHED)
```python
# _otr_line_composer.py
def compose_news_coda(*, creative_fn, news_close_brief, premise_context,
                      intro_text="", cast_seed=0, creative_repo_id=None) -> LineResult:
```
- `premise_context` = the fiction SETUP, NOT the outcome: pass `script_brief` (the
  premise/arc) and optionally `intro_text` (tone echo). NEVER `ending_change` /
  `final_character_line` / `news_close_brief` facts.
- LLM writes ONLY the bridge clause. Append the cleaned brief deterministically.

## `_NEWS_CODA_SYSTEM` (exact, flag-gated -- panel all 3 asked for the literal text)
```
You are the radio announcer for SIGNAL LOST, an old-time radio drama.
Write ONE short bridge clause that turns from tonight's fictional tale to the real
world. The real news report is added AFTER your clause by the producer -- you do
NOT write it.

OUTPUT - strict:
- Only the words the announcer says out loud. One line, no line breaks.
- No speaker name, no colon-label, no quotation marks, no brackets, no sound cues.
- A SHORT pivot clause, at most ~16 words, ending with a colon or an em dash.

VOICE:
- A period radio host: warm, measured.
- Reference tonight's tale by its SUBJECT or SETTING (not how it ended).
- Do NOT state any fact, number, date, or the story's outcome.
- Do NOT open with a stock phrase ("And now", "But in the real world", "In reality",
  "The real story", "Meanwhile"). Make the turn specific to tonight's tale.
```
User prompt: `f"Tonight's tale (setup only):\n{premise_context}\n\nThe announcer's
opening line was:\n{intro_text}\n\nWrite the bridge clause now."` (intro_text omitted
if empty).

## VALIDATOR -- bridge only (`validate_news_coda_bridge(text) -> (ok, cleaned)`)
- one line; no leading speaker label / bracket; length <= ~80 chars (Gemini cap --
  physically prevents rambling into facts);
- NOT a `BRIDGE_GENERIC_OPENERS` opener (normalized lowercase startswith over
  `("and now", "but in the real world", "in reality", "the real story",
  "meanwhile", "tonight we", "what really happened")`);
- no >=5-token verbatim run copied from the cleaned brief (cheap n-gram copy guard --
  near-zero risk since the bridge never sees the brief, but cheap insurance);
- (optional, best-effort) reject a tiny outcome-verb blocklist (`"proved","revealed",
  "confirmed","was killed","was found","ended with","resulted in"`).
DROPPED: the `ending_change`-overlap gate (R1) and the semantic no-outcome check (R2).

## ASSEMBLY (deterministic)
```python
bridge = validated bridge  # ends with ':' or em dash (normalize if not)
fact   = clean_one_line(news_close_brief, max_chars=200)  # one line, capped
fact   = fact[:1].upper() + fact[1:] if fact else fact     # capitalize after a turn
coda   = f"{bridge} {fact}".strip()
# final sanitizer: one line, no bracket/label, total cap (~300 chars), non-empty.
```

## REROLL (no seed API -- GPT grounded)
`_announcer_generate(creative_fn, messages)` takes NO seed. Reroll = re-call with an
ALTERED PROMPT (append "Attempt 2 -- use different wording; be more specific to the
tale."). One retry on invalid bridge.

## FALLBACK FLOOR (bridge fails twice) -- deterministic rotating pool
```python
POOL = ("The real story:", "The true account:", "From tonight's headlines:")
prefix = POOL[abs(hash(str(cast_seed))) % len(POOL)]   # or a stable hashlib mod
return LineResult(text=f"{prefix} {fact}", compose_flags=("news_coda_fallback","news_coda_bridge_invalid"))
```
`BRIDGE_GENERIC_OPENERS` is BRIDGE-ONLY -- the fallback prefixes are the deterministic
floor and are NOT validated as bridges (GPT#11).

## EMPTY news_close_brief -> NOT a coda (caller decides)
If `news_close_brief` is empty, the caller does NOT enter coda mode: it runs the
EXISTING `compose_announcer_outro` (the normal fictional outro), flagged
`news_coda_no_brief`. Never fabricate a "real story" (Gemini#6/GPT#9).

## compose_flags TAXONOMY
`("news_coda_bridge",)` happy path; `("news_coda_bridge_reroll",)` reroll succeeded;
`("news_coda_fallback","news_coda_bridge_invalid")` pool floor;
`("news_coda_no_brief",)` set by the caller when it skips to the fictional outro.

## CALL-SITE WIRING (writer :4615-4634) -- EARLY BRANCH (GPT#7)
```python
if _style_grammar_on and (nc_brief or "").strip():
    outro_res = _OTRLC.compose_news_coda(
        creative_fn=creative_generate_fn,
        news_close_brief=nc_brief,
        premise_context=script_brief,          # SETUP, not outcome
        intro_text=intro_text, cast_seed=cast_seed,
        creative_repo_id=resolved["creative_writing_model"])
else:
    outro_res = _OTRLC.compose_announcer_outro( ... existing args unchanged ... )
    # if _style_grammar_on and not nc_brief: also stamp news_coda_no_brief
```
`compose_announcer_outro` is UNTOUCHED (no new params) -> the OFF path + the no-brief
path are byte-identical. `cast_seed` is in scope (:2878).

## SUPERSEDES the main campaign (pass04_plan.md STEP F)
This REPLACES STEP F's fixed `NEWS_CODA_LEAD_IN` + "validate body has no lead-in /
prepend lead-in" mechanics with `compose_news_coda` (dynamic bridge + appended fact +
rotating-pool fallback). The STEP F goals (deliver the real fact, never restate the
fiction, byte-identical off, decouple the outro from "last line") still hold; the
climax-line decoupling (STEP F) is independent and stays. Remove the
`NEWS_CODA_LEAD_IN` constant + `validate_news_coda_line` from STEP F; do not leave
dead code (DeepSeek CUT).

## NON-NEGOTIABLE (carried)
Behind `_style_grammar_on`, byte-identical off (compose_announcer_outro untouched);
the coda always delivers the real `news_close_brief`; the bridge never sees the
fictional outcome; 100% local (`creative_fn` is the configured local creative slot);
deterministic fallback; UTF-8 no BOM; SFW.

## VERIFY-AT-BUILD
- `key_terms`/news fields: NOT needed for the bridge after this redesign (the bridge
  uses premise_context, not news). `news_close_brief` read from `meta["news"]` at
  outro time (writer :3949) -- confirm non-empty handling.
- `_announcer_generate` truly has no seed arg (reroll is prompt-based).
- `clean_one_line` cap behavior on the brief (it already supports max_chars).
- the n-gram copy guard never fires on legit bridges (it shouldn't -- bridge never
  sees the brief).
