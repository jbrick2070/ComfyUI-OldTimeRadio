# Story-Quality R2 C1 + C2 -- SPRINT-READY (3-pass roundtable converged)

Campaign: pass01 arch + pass02 coding + pass03 wiring; each = my grounded critique
+ GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro; Claude judge. Total ~$0.25. CONVERGED.

## DECISIONS (unanimous)
- C1 = INJECTION-ONLY (no reroll gate); CUT the outline nudge; CUT the LLM opt-in
  (pure deterministic); anchors come SOLELY from the curated `key_terms` (no news
  regex). Zero-ripple injection (append existing strings; no signature/schema change).

## PURE MODULE `nodes/_otr_specificity.py` (stdlib only)
Shared casefold tokenizer + defensive (never-raise; coerce None/str/non-iterable):
- `_cast_tokens(cast) -> set[str]`: for each row name (dict/obj), add the full
  casefolded name AND each sub-token whose `len > 2` (Gemini: avoid "a"/"j"/"will"
  polluting the exclusion set).
- `derive_specificity_anchors(key_terms, cast) -> list[str]`: candidates =
  key_terms (coerce->str, trim, collapse ws); EXCLUDE empty / <2 chars / casefold
  == a cast token / token-intersection with cast tokens; dedupe case-insensitively
  (keep first casing); cap `MAX_ANCHORS = 5`. Returns list[str].
- `derive_central_object(key_terms, cast) -> str` (conservative): first candidate
  that survives ALL: not empty; not cast-token-matched; not ALL-CAPS; not a single
  word with uppercase first char (`len(split)==1 and [0].isupper()`); NOT Title-Case
  (reject when EVERY alphabetic token is capitalized -- Gemini); not in
  `_ENTITY_SUFFIX` {agency,administration,university,institute,corp,corporation,inc,
  company,department,island,city,county,state,observatory,laboratory,lab,center,
  centre}; not purely numeric; AND has >=1 entirely-lowercase alphabetic token
  (GPT positive rule). Else "".
- `sanitize_for_prompt(s) -> str`: collapse whitespace + strip control/newlines
  ONLY (no length cap -- key_terms are short; no vague delimiter stripping).
- `inject_anchors_into_header(canon_header, anchors) -> str` (pure, testable):
  return header + a delimited block built from sanitized non-empty anchors, joined
  "\n- "; return header UNCHANGED if no safe anchors.
- `inject_central_object_into_brief(brief, obj) -> str` (pure): append
  " Central object, if useful: <sanitized obj>" (+ a period only if not already
  sentence-terminated); UNCHANGED when obj is "".

## WIRING (OTR_LedgerScriptWriter) -- idempotent via META FLAGS (DeepSeek)
- C1, after canon_header is built (~2746) and BEFORE `meta["canon_header"]=...`
  (@3255) + the line loop:
  `if "specificity_anchors" not in meta: meta["specificity_anchors"] =
   derive_specificity_anchors((meta.get("news") or {}).get("key_terms") or
   key_terms_tuple, cast)`
  `if not meta.get("_specificity_anchors_injected") and meta.get(
   "specificity_anchors"): canon_header = inject_anchors_into_header(canon_header,
   meta["specificity_anchors"]); meta["_specificity_anchors_injected"] = True`
- C2, before the `compose_announcer_outro` call (~3275), on a LOCAL brief string:
  `if "central_object" not in meta: meta["central_object"] =
   derive_central_object((meta.get("news") or {}).get("key_terms") or
   key_terms_tuple, cast)`
  `news_close_brief = inject_central_object_into_brief(str(news_close_brief or ""),
   meta.get("central_object") or "")`
- Idempotency: derive only when the DATA key is absent; inject once via the META
  FLAG (no header substring scan -- avoids false-suppression). meta["news"][
  "key_terms"] is the canonical source for both.

## INVARIANTS
meta free-form (additive keys; if a writer-metadata SNAPSHOT test asserts exact
meta keys, update it same-commit). Deterministic + idempotent. Model-agnostic. NO
LineRequest / compose_announcer_outro signature change (verified: outro brief has
`clean_one_line(..., max_chars=0)` -> no input length limit). NO workflow-JSON
change. Audio = spine/no-JSON. UTF-8 no BOM.

## TESTS
- _cast_tokens (len>2 sub-tokens; full name); anchors (filter/dedupe/cap/empty;
  partial-cast-name not over-excluded "a new car"); central_object (reject
  NASA/Swift/James Webb Telescope/cast/suffix/numeric; accept "three robotic arms",
  "a $500M telescope"; "" when none; idempotent).
- inject helpers (block present iff anchors; unchanged when empty; central object
  present iff set; no double period; idempotent on second call via the writer flag).
- sanitize (newlines/control stripped; words not sliced).

## FINAL QA (next)
Extend `scripts/story_quality_scan.py` with the shipped levers' metrics
(flag_thesis_close / flag_cliche / flag_stage_business / flag_on_the_nose /
detect_leading_stage_business / wants_are_default + anchors/central_object presence)
over frozen ledgers; a short re-soak read.
