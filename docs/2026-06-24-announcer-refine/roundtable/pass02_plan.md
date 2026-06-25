# ANNOUNCER REDESIGN + NEWS CODA + KILL-2 -- BUILD-SPEC (pass02, post-R2)

R2 (implementability) converged on concrete data contracts + three real coding
bugs. This pass pins exact signatures so a coder cannot hit an ambiguous step.
All grounding below is verified against the real files this session.

OPERATOR THESIS (unchanged): the show TEACHES; drama delivers, the NEWS is the
payload, explicit at the very end. All behind `story_scaffold`; byte-identical off.

---

## 0. DATA CONTRACTS (exact -- R2 MUST-FIX, all 3)

```python
# nodes/_otr_style_catalog.py
@dataclass(frozen=True)
class StoryContract:
    slug: str
    label: str
    sound_world: str
    story_engine: str
    ending_mode: str          # human ending descriptor (catalog field)
    ending_tag: str           # the climax-class role (CLIMAX_CLASS_ROLES member)
    ending_template: str      # ending_template_for(slug)
    grammar: str              # == render_style_grammar(slug)

def build_story_contract(cast_seed, script_brief: str, news_seed: str,
                         meta) -> StoryContract:
    text = (script_brief or news_seed or "")
    slug = select_style(text, meta, cast_seed)   # reuse: only the emergency-pool
                                                 # input shifts from premise->text;
                                                 # the pick stays sha256(cast_seed)-keyed
    s = get_style(slug) or {}
    return StoryContract(slug, s.get("label",""), s.get("sound_world",""),
        s.get("story_engine",""), s.get("ending_mode",""),
        s.get("ending_tag",""), ending_template_for(slug), render_style_grammar(slug))
```
GROUNDED: `select_style` has EXACTLY ONE caller (writer :3224) -> safe to move.
`select_style(premise,meta,cast_seed)` uses `premise` only via
`premise_wants_emergency`; the hash is cast_seed-keyed -> feeding `text` is a
contained, flag-gated change to the emergency-pool decision only.

```python
# the open's outcome-free inputs (frozen)
@dataclass(frozen=True)
class SafeOpenBrief:
    setting: str
    time_of_day: str
    opening_status_quo: str
    cast: tuple[tuple[str, str], ...]   # (name, role)
    era: str = ""                       # verify-at-build source (meta/period); "" ok
    tone: str = ""                      # contract register, optional
```

meta storage (JSON-safe -- do NOT store the dataclass):
`meta["story_contract"] = {"slug": c.slug, "label": c.label, "ending_tag": c.ending_tag}`.
Do NOT touch `resolved["style"]` / `meta.style` / `visual_plan.style`.

`OutlineRequest` (frozen, `_otr_outline.py:283`) gets TWO new defaulted fields,
DISTINCT from the existing user `style`:
`story_engine: str = ""`, `ending_mode: str = ""`. Rendered in
`_build_macro_user_prompt` (`:1133`) ONLY when `story_scaffold` + non-empty.
VERIFY-AT-BUILD: `_build_phase_user_prompt` (:1187) / `_build_beat_user_prompt`
(:1236) take `macro`, not `OutlineRequest` -> thread the contract fields through
the macro/combiner or add a param; confirm the handoff before wiring.

---

## 1. KILL 2 -- StoryContract (build pre-outline; inject by layer)
- Build `build_story_contract(cast_seed, script_brief, news_seed, meta)` in run()
  AFTER cast-lock (`cast_seed` @ :2878) + briefs (`script_brief` @ :2785), BEFORE
  `OutlineRequest` (@ :3032). GROUNDED feasible (both inputs precede :3032).
- DELETE the late `select_style(outline.premise,...)` @ :3224 (single caller);
  reuse the contract's slug/ending_tag/ending_template downstream.
- OUTLINE injection: render `story_engine` + `ending_mode` in the macro prompt
  (and phase/beat per the verify-above). This is the structural steer.
- LINE level: pass ONLY a compact register tag + the existing `conflict_object`
  (do NOT thread `sound_world`/`story_engine` into LineRequest -- R1 leak risk).
- `sound_world` -> render/mood routing is DEFERRED (call sites unknown; R2 cut).
- meta: add `meta["story_contract"]` dict; telemetry `story_contract_slug`.
ACCEPTANCE (objective): build_story_contract CALLED; macro prompt contains
story_engine/ending_mode under flag; meta.story_contract recorded; delete-it
reverts; re-soak shows two slugs -> different conflict objects/structure (eval,
not a unit assert). DEFERRED: premise-specific conflict objects.

## 2. ANNOUNCER -- JOB 1 OPEN (input starvation + cheap belt)
`compose_announcer_intro` -- backward-compatible signature:
```python
def compose_announcer_intro(*, creative_fn, script_brief, creative_repo_id=None,
        story_scaffold=False, safe_open_brief=None, forbidden_tokens=frozenset()):
```
- flag OFF or no brief: EXACT current path (byte-identical).
- flag ON + safe_open_brief: build the user prompt from the SAFE fields ONLY
  (NEVER `script_brief`); use a flag-gated rewritten `_ANNOUNCER_INTRO_SYSTEM`
  (cold-open: S1 orients era/time/place/cast/status-quo, S2 intrigue, no outcome
  terms). Fallback = a NEW `fallback_safe_open(safe_open_brief)` that NEVER reads
  `script_brief`.
- `opening_status_quo` (deterministic, NO new LLM field): = the FIRST character
  beat's `intent` (the setup beat), `clean_one_line`-sanitized. Outcome-free by
  construction (setup precedes the climax). [REJECTS adding an LLM macro field --
  weak-model risk + the field could itself carry the outcome.]
- BELT (input-starvation is the guarantee; the belt is the deferrable extra):
  reuse the KILL-1 token helpers (`_TOKEN_RE` / `_content_tokens` /
  `_strip_possessive` in `_otr_story_quality_l12`). `forbidden_tokens` =
  content tokens of `ending_change` + `news_close_brief`, minus stopwords, cast
  names, setting/time terms, len>=4. Reject if the open shares >=2; reroll ONCE;
  else `fallback_safe_open`. Telemetry `open_spoiler_reroll` (one), `open_gate_failed`.
- Call site (writer :4465): under flag pass `safe_open_brief` + `forbidden_tokens`
  + `story_scaffold=True`; do NOT pass `script_brief` content into the open prompt.
ACCEPTANCE: open names setting+era+characters + the opening situation, no
outcome/twist token, produced WITHOUT script_brief (assert the call args).

## 3. ANNOUNCER -- JOB 2/3 CLOSE = the NEWS CODA (repurpose the outro line)
GROUNDED: ONE trailing announcer beat (`last_announcer_id`); the news coda IS the
repurposed outro -- NO new beat. `compose_announcer_outro` -- extended signature:
```python
def compose_announcer_outro(*, creative_fn, script_brief, news_close_brief,
        intro_text, creative_repo_id=None, ending_change="", final_character_line="",
        story_scaffold=False, climax_character_line="", coda_lead_in=""):
```
- flag OFF: EXACT current path incl. the resolved-fiction "State this outcome
  plainly" branch (:2854) (byte-identical).
- flag ON: NEWS-CODA path:
  - flag-gated rewritten `_ANNOUNCER_OUTRO_SYSTEM`: deliver the REAL fact from
    `news_close_brief` plainly; "Start immediately with the facts; do NOT write an
    introductory phrase" (prevents the double-lead-in stutter); the concrete-image
    rule is relaxed for the coda.
  - SUPPRESS the resolved-fiction branch; pass `ending_change` only as forbidden
    "do NOT restate this fictional outcome" content.
  - DETERMINISTIC lead-in is a POST-GENERATION PREFIX: the LLM writes only the
    fact body; the composer returns `f"{coda_lead_in} {body}"`. The validator
    asserts the body does NOT already contain a lead-in variant.
  - Fallback = NEW `fallback_news_coda_outro(coda_lead_in, news_close_brief)` that
    NEVER reads `script_brief`/`ending_change` and starts with the lead-in.
- NEW `validate_news_coda_line(text, *, lead_in, news_close_brief, ending_change,
  min_words=18, max_words=45) -> (ok, cleaned)`: word band; no leading bracket;
  body has NO lead-in variant; >=1 key term from `news_close_brief` present
  (reuse the news key_terms if available); NO strong content-token overlap with
  `ending_change`.
- CLIMAX decoupling (byte-identical now): at the outro call site (:4619), find the
  ledger line where `_ln.get("beat_id") == _climax_beat_id` (GROUNDED:
  `_otr_ledger.py:96` -> lines carry `beat_id`; `_climax_beat_id` in scope @ :3271)
  and pass it as `climax_character_line`; fall back to the existing last-character
  scan if not found. Today climax==last -> identical.
- `coda_lead_in`: FIRST build = ONE fixed lead-in (reduce test surface). Operator
  leans "The real story:"; RECOMMEND a period-in-voice lead-in to protect the OTR
  fiction ("From tonight's headlines:", "The true account:", "What the record
  shows:"). A small CLOSED seed-keyed set (3-5) is a later step. Final wording =
  operator's creative call.
ACCEPTANCE: coda states the real fact after the character climax; announcer never
restates the fictional outcome under flag; lead-in present deterministically;
byte-identical off.

## 4. KILL 4 -- un-starve the body (exact, with the slice-bug fix)
- Role-keyed enrichment for real constants (l12:55-72): `BEAT_ROLE_SETUP`,
  `BEAT_ROLE_PRESSURE`, `BEAT_ROLE_PERSONAL_STAKE` + every `CLIMAX_CLASS_ROLES`
  member (class-specific text). DO NOT delete `BEAT_ROLE_CONSEQUENCE` enrichment
  -- DEFER (KILL 3 makes it reachable; R1 catch).
- Truncation ORDER + the negative-slice bug (Gemini): replace
  `new_intent[:_INTENT_MAX]` with:
  ```python
  sep = " "
  reserve = _INTENT_MAX - len(sep) - len(enrichment)
  if reserve <= 0:
      final = enrichment[:_INTENT_MAX]
  else:
      final = (original_intent[:max(0, reserve)].strip() + sep + enrichment.strip())[:_INTENT_MAX]
  ```
  Always clamp the slice with `max(0, ...)` -- `s[:-5]` does NOT clamp.
  Test: `len(original)+len(enrichment) > 200`.

## 5. KILL 3 -- DEFERRED (principle settled). Only the outro climax-decoupling
(§3) is pulled forward, and it is byte-identical today.

## 6. BUILD CHUNKS (separate commits for bisectability -- R2 GPT#7)
- C1: StoryContract + build_story_contract + move select_style pre-outline +
  OutlineRequest fields + macro-prompt render + meta dict + telemetry.
- C2: announcer OPEN (SafeOpenBrief + intro signature + system rewrite +
  fallback_safe_open + opening_status_quo derivation + belt).
- C3: announcer CODA (outro signature + system rewrite + lead-in prefix +
  validate_news_coda_line + fallback_news_coda_outro + climax decoupling).
- C4: KILL 4 (role map + truncation clamp).
Each: full suite + Bug Bible green vs the 5 pre-existing 267a53e fails;
byte-identical OFF (golden tests on open line, outro line, ledger meta);
commit AND push to v2.0-alpha per green chunk.

## 7. BYTE-IDENTITY (explicit boundary)
OFF => no contract build, no select_style move effect (off path unchanged), no
new prompt text, no meta.story_contract, no changed fallback/outro/intro text, no
LineRequest payload, no climax-line override. New OutlineRequest fields default ""
and render only under flag. VERIFY-AT-BUILD (R2 GPT#8): if any existing test
compares `OutlineRequest` `asdict`/repr, the two new default-"" keys may shift a
snapshot -> update those fixtures or gate serialization.

## 8. TELEMETRY (under flag only; 3-test "baked in")
`meta.story_quality.{story_contract_slug, open_spoiler_reroll, open_gate_failed,
open_safe_fallback, news_coda_emitted, news_coda_fallback}`.

## 9. PIPELINE ORDER (verified)
cast-lock (:2878) -> briefs (:2785) -> build_story_contract -> OutlineRequest
(:3032, carries story_engine/ending_mode) -> generate_outline -> [line loop:
OPEN composed from SafeOpenBrief (setup-beat intent = opening_status_quo); body
lines carry register+conflict_object] -> post-loop outro = news coda
(news_close_brief + lead-in + climax_character_line).

## INVARIANTS (a fix that breaks one is rejected)
Behind `story_scaffold` -> byte-identical off; audio spine FROZEN
(`test_audio_byte_identical`, mux-LAST); suite + Bug Bible per chunk; commit+push
per green chunk to v2.0-alpha; 100% local; determinism (seed-keyed); LOUD
fallbacks; UTF-8 no BOM; SFW; prod/main + tags GATED.

## VERIFY-AT-BUILD (carried)
- phase/beat prompt threading of contract fields (they take `macro`).
- `era` source for SafeOpenBrief (meta/period; "" acceptable).
- ledger line `beat_id` present on the climax row at outro time (doc'd at
  ledger:96 -- confirm at runtime).
- news_close_brief never empty + distinct from ending_change.
- OutlineRequest asdict/repr snapshot fixtures (byte-identity).
