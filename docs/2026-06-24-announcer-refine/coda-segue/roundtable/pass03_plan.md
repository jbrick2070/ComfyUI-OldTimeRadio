# DYNAMIC NEWS-CODA SEGUE -- FINAL CONVERGED BUILD TICKET (coda-segue pass03)

Converged R1 (split architecture) -> R2 (mechanics) -> R3 (wiring; precision only).
R3 returned NO architecture change -> CONVERGED; no R4 needed (stop-at-convergence).
Panel each round: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro; Claude grounded judge.
Total coda-segue spend ~$0.30. Self-contained; supersedes the main-campaign STEP F
lead-in mechanics.

## THE DESIGN (one line)
The LLM writes ONLY a short dynamic BRIDGE clause (specific to tonight's tale,
never the outcome, never the news fact); the real `news_close_brief` is APPENDED
deterministically. Reliability is structural; the operator gets a crafted,
news-following segue that varies every episode.

## NEW FUNCTION (keeps `compose_announcer_outro` UNTOUCHED -> off-path byte-identical)
```python
# _otr_line_composer.py
import hashlib, dataclasses

NEWS_CODA_POOL = ("The real story:", "The true account:", "From tonight's headlines:")
BRIDGE_GENERIC_OPENERS = ("and now", "but in the real world", "in reality",
                          "the real story", "meanwhile", "tonight we", "what really happened")
_BRIDGE_MAX_CHARS = 80
_CODA_FACT_MAX = 200
_CODA_TOTAL_MAX = 320

def compose_news_coda(*, creative_fn, news_close_brief, premise, intro_text="",
                      cast_seed=0, creative_repo_id=None) -> LineResult:
    # 1) clean the FACT first (defined before generate/validate/fallback -- GPT#4)
    fact = clean_one_line(news_close_brief or "", max_chars=_CODA_FACT_MAX)
    if not fact:
        return LineResult(text="", compose_flags=("news_coda_no_brief",))  # caller handles
    fact = (fact[0].upper() + fact[1:]) if fact[0].isalpha() else fact

    def _assemble(bridge: str) -> str:
        b = clean_one_line(bridge, max_chars=_BRIDGE_MAX_CHARS).rstrip(".!?,;: ")
        b = b + ":"                                   # normalize the turn (no "x.:" -- Gemini#5)
        return clean_one_line(f"{b} {fact}", max_chars=_CODA_TOTAL_MAX)

    # 2) dynamic bridge: setup-only inputs (NO ending_change / final_char / news fact)
    def _msgs(retry: bool):
        u = f"Tonight's tale (setup only):\n{premise}"
        if intro_text:
            u += f"\n\nThe announcer's opening line was:\n{intro_text}"
        if retry:
            u += "\n\nAttempt 2 -- different wording; be more specific to the tale."
        return [{"role":"system","content":_NEWS_CODA_SYSTEM},
                {"role":"user","content":u}]               # fresh 2-msg array, no role-stutter (Gemini#4)

    for attempt, flag in ((False,"news_coda_bridge"), (True,"news_coda_bridge_reroll")):
        raw = _announcer_generate(creative_fn, _msgs(attempt))   # no seed arg (grounded)
        ok, bridge = validate_news_coda_bridge(strip_line_formatting(raw or ""))
        if ok:
            return LineResult(text=_assemble(bridge), compose_flags=(flag,))

    # 3) deterministic fallback floor -- stable hash (NOT builtin hash(); mirror select_style)
    h = int(hashlib.sha256(f"news-coda:{cast_seed}".encode("utf-8")).hexdigest(), 16)
    prefix = NEWS_CODA_POOL[h % len(NEWS_CODA_POOL)]
    return LineResult(text=clean_one_line(f"{prefix} {fact}", max_chars=_CODA_TOTAL_MAX),
                      compose_flags=("news_coda_fallback","news_coda_bridge_invalid"))
```

## `premise` SOURCE (R3 bleed fix -- GPT#2 / DeepSeek ASSUMPTION)
Pass `outline.premise` (the macro dramatic premise -- "extrapolates dramatically
from the story", setup-framed), NOT `script_brief` (news_interpreter's distillation
can hint the resolution). `intro_text` is the SAFE no-spoiler open (built from
SafeOpenBrief when the flag is on) -> safe to pass for tone. NEVER pass
`ending_change`, `final_character_line`, or the news facts.

## `_NEWS_CODA_SYSTEM` (exact, flag-gated)
```
You are the radio announcer for SIGNAL LOST, an old-time radio drama.
Write ONE short bridge clause that turns from tonight's fictional tale to the real
world. The real news report is added AFTER your clause by the producer -- you do
NOT write it.

OUTPUT - strict:
- Only the words the announcer says out loud. One line, no line breaks.
- No speaker name, no quotation marks, no brackets, no sound cues.
- A SHORT pivot clause, at most ~16 words, ending with a colon.

VOICE:
- A period radio host: warm, measured.
- Reference tonight's tale by its SUBJECT or SETTING (not how it ended).
- Do NOT state any fact, number, date, or the story's outcome.
- Do NOT open with a stock phrase ("And now", "But in the real world", "In reality",
  "The real story", "Meanwhile"). Make the turn specific to tonight's tale.
```

## VALIDATOR -- bridge only (coda-specific; do NOT reuse `validate_announcer_line`)
```python
def validate_news_coda_bridge(text) -> tuple[bool, str]:
    cleaned = clean_one_line(text or "", max_chars=0)
    if not cleaned: return False, ""
    if "\n" in (text or "").strip(): return False, ""
    if any(ch in cleaned for ch in "[]{}"): return False, ""
    up = cleaned.upper()
    if any(up.startswith(p) for p in _ANNOUNCER_BAD_PREFIXES): return False, ""  # leading ANNOUNCER:
    low = cleaned.lower()
    if any(low.startswith(g) for g in BRIDGE_GENERIC_OPENERS): return False, ""
    if len(cleaned) > _BRIDGE_MAX_CHARS: return False, ""
    return True, cleaned
```
A TRAILING colon is allowed (the turn); only a LEADING speaker label is rejected
(GPT#6 -- that is why we do NOT reuse `validate_announcer_line`).
DROPPED (R3): the n-gram copy guard (bridge never sees the brief -> guard is
low-value + an interface mismatch -- GPT CUT#1); the outcome-verb blocklist
(false-positive risk, low value -- GPT CUT#2 / DeepSeek).

## CALL-SITE WIRING (writer ~:4615-4634) -- early branch + else-scoped outro vars (GPT#1)
```python
nc_brief = (meta.get("news") or {}).get("news_close_brief") or ""   # already read ~:3949
if _style_grammar_on and nc_brief.strip():
    outro_res = _OTRLC.compose_news_coda(
        creative_fn=creative_generate_fn, news_close_brief=nc_brief,
        premise=str(getattr(outline, "premise", "") or ""), intro_text=intro_text,
        cast_seed=cast_seed, creative_repo_id=resolved["creative_writing_model"])
else:
    # build the EXISTING outro inputs INSIDE the else (only the fictional path needs them)
    _outro_ending_change = str((meta.get("dramatic_state") or {}).get("ending_change") or "")
    _outro_final_char_line = ""   # the existing reversed() last-character scan (:4619-4623)
    for _ln in reversed(led.data.get("lines") or []):
        if str(_ln.get("speaker_role") or "").strip() == "character":
            _t = str(_ln.get("text") or "").strip()
            if _t: _outro_final_char_line = _t; break
    outro_res = _OTRLC.compose_announcer_outro( ...existing args UNCHANGED... )
    if _style_grammar_on:   # on-flag but no brief -> mark it, text unchanged (frozen LineResult)
        outro_res = dataclasses.replace(
            outro_res, compose_flags=outro_res.compose_flags + ("news_coda_no_brief",))
```
`cast_seed` is in scope (:2878). `compose_announcer_outro` gets NO new params (off
path byte-identical). `LineResult` is frozen -> `dataclasses.replace` (Gemini#3/all 3).

## SUPERSEDES the main campaign (pass04_plan.md STEP F + CODE_MAP.md C3)
- REMOVE STEP F's `NEWS_CODA_LEAD_IN` constant + `validate_news_coda_line` +
  "validate body has no lead-in / prepend lead-in" -- replaced by `compose_news_coda`.
- DROP the STEP F climax-line DECOUPLING (`climax_character_line`): the ON-flag coda
  is a premise->news pivot that never touches the fictional climax, so Job 2
  ("protect the character climax") holds BY CONSTRUCTION; the OFF-flag path is
  unchanged. Keep `_climax_beat_id` only for KILL 3's own later build. (R3
  simplification -- one fewer edit + verify item.)
- Keep STEP F's: separate `_NEWS_CODA_SYSTEM`, suppress the resolved-fiction branch
  (now moot on the ON path since `compose_news_coda` never calls it), exclude
  ending_change. These are subsumed by `compose_news_coda`.

## compose_flags TAXONOMY
`("news_coda_bridge",)` / `("news_coda_bridge_reroll",)` /
`("news_coda_fallback","news_coda_bridge_invalid")` / `("news_coda_no_brief",)`.

## BYTE-IDENTITY
OFF (`_style_grammar_on=False`): the else branch runs `compose_announcer_outro` with
its CURRENT args, no replace, no new function -> byte-identical. `compose_news_coda`
+ the constants are dead code when off.

## NON-NEGOTIABLE (carried)
Behind `_style_grammar_on`; the coda always delivers the real `news_close_brief`;
the bridge never sees the fictional outcome (premise + safe intro only); 100% local
(`creative_fn` = configured local creative slot); DETERMINISTIC fallback (sha256, not
builtin hash); UTF-8 no BOM; SFW.

## VERIFY-AT-BUILD
- `outline.premise` is reliably setup-framed (it is "extrapolates dramatically from
  the story" -- a premise, not a resolution); spot-check on a soak that it does not
  carry the ending.
- `_announcer_generate` has no seed arg (reroll is prompt-based) -- confirmed by the
  grounding; the retry is a fresh 2-message array (no role-stutter).
- `LineResult` is a frozen dataclass (`dataclasses.replace` works); `_ANNOUNCER_BAD_PREFIXES`
  exists in the composer (used by `validate_announcer_line`).
- the freeze cascade re-runs the WHOLE writer (deterministic via forced cast_seed) ->
  the coda recomposes through the same branch; it does NOT separately reroll the
  announcer outro line in-loop (`build_reroll_line_request` is body-only). Confirm.
