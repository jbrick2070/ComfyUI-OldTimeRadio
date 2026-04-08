# SIGNAL LOST v1.2 — Bug Fix Guide (AI-consumption format)

**Date:** 2026-04-07
**Branch:** `v1.2-narrative-beta` @ `e2f210a`
**Source:** boot test runs 1 + 2, treatment `signal_lost_the_last_frequency_20260407_213635_treatment.txt`

---

## Bug Index

| ID | Severity | Area | Status |
|---|---|---|---|
| B-001 | CRITICAL | Pattern 3/5/6 metadata emission | OPEN — awaiting peer review |
| B-002 | MEDIUM | JSONDecodeError at pipeline start | OPEN — source unknown |
| B-003 | MEDIUM | Baked character names in Gemma output | OPEN — prompt drift |
| B-004 | LOW | Procedural rename not reflected in script text | OPEN — by design, revisit |
| B-005 | INFO | Pattern 4 untestable in test.json | EXPECTED — use scifi_lite |

---

## B-001 — Pattern 3/5/6 metadata blocks silently fail to emit

- **Severity:** CRITICAL (blocks v1.3 Python post-processing dependencies)
- **Workflow:** `old_time_radio_test.json` (1 min, 3 chars, act_breaks=False)
- **Model:** google/gemma-4-E4B-it
- **Expected:** `<vocal_blueprints>` before SCENE 1; `<locked_decisions>` between SCENE 2/3; `<epilogue_candidates>` before EPILOGUE
- **Actual:** zero metadata blocks emitted. Script opens directly at `=== SCENE 1 ===`
- **Evidence:** treatment file line 32 — first scene marker with no preceding XML block
- **Root cause hypothesis:**
  1. Sections 6–9 buried at end of 30,480-char `SCRIPT_SYSTEM_PROMPT`
  2. Phrased as descriptive spec ("emit a single block") not imperative ("THE FIRST THING YOU WRITE MUST BE")
  3. Gemma 4 E4B-it recency-weights behavioral rules (Section 5) over one-shot emission directives (Sections 6–9)
- **Fix strategy (pending reviewer input):**
  1. Reorder Sections 6–9 to front of prompt, after `SCAFFOLDING_PREAMBLE`
  2. Rephrase with imperative emission template + filled worked example
  3. If still failing: escalate Pattern 3 to v1.3 multi-call, keep Patterns 5/6 as single-pass with hardened directives
- **v1.2 disposition:** HOLD. Do not hack prompt until peer review returns.
- **v1.3 fallback:** multi-call architecture per ROADMAP N-series

---

## B-002 — JSONDecodeError at pipeline start

- **Severity:** MEDIUM (non-fatal, pipeline completes)
- **Trace:**
  ```
  json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
    json/__init__.py:346 in loads
    json/decoder.py:356 in raw_decode
  ```
- **Location:** top of boot log, before "got prompt"
- **Hypothesis:** stale/empty JSON read on startup — possibly cached workflow state, empty Director response buffer, or prior-run residue
- **Fix strategy:**
  1. Wrap suspect `json.loads` calls in try/except with source-file logging
  2. Check Director response path for empty-string handoff
  3. Audit ComfyUI cache dir for 0-byte json files
- **v1.2 disposition:** INVESTIGATE. Add logging, identify source, defer fix if non-blocking.

---

## B-003 — Baked character names in Gemma output

- **Severity:** MEDIUM (violates "zero baked names except LEMMY" rule)
- **Evidence:** treatment cast block shows `JIAN`, `REX` hard-coded in dialogue despite procedural Director map (`REX→NIRAN KENDALL`, `JIAN→ZURI STONE`)
- **Root cause:** Gemma drifts to trained-common names for 3-character sci-fi casts. `gemma4_orchestrator.py` code is CLEAN — no baked strings in source. Drift is at generation time.
- **Fix strategy:**
  1. Add explicit "DO NOT use the names REX, JIAN, SARAH, JOHN, ALEX, MAYA" anti-list to Section 5 forbidden constructs
  2. OR post-process: apply `_randomize_character_names` rename AFTER parse, rewriting dialogue speaker tags
- **v1.2 disposition:** Option 2 (post-process rename) is safer — no prompt bloat, deterministic.

---

## B-004 — Procedural rename not reflected in script text

- **Severity:** LOW
- **Evidence:** Director assigns `REX→v2/en_speaker_7/NIRAN KENDALL` but treatment shows `REX  [v2/en_speaker_7]`. Rename map exists in plan, never applied to script body.
- **Root cause:** `_randomize_character_names` returns mapping but no downstream writer substitutes script text.
- **Fix strategy:** apply rename at `_parse_script` output stage, before Bark handoff. Replace `speaker` field in dialogue entries using gender_map.
- **v1.2 disposition:** bundle with B-003 fix.

---

## B-005 — Pattern 4 (Yes-But/No-And) untestable

- **Severity:** INFO
- **Cause:** `test.json` has `include_act_breaks=False`. No act breaks = no escalation surface.
- **Action:** run `old_time_radio_scifi_lite.json` (5 min, act_breaks=True) for Pattern 4 verification. Documented in QA_GUIDE section "Pattern coverage per workflow".

---

## Green Path (working, do not touch)

| Component | Status | Evidence |
|---|---|---|
| Gemma 4 load / inference / VRAM handoff | OK | Run 1: 01:33:28 total; Run 2: 00:22:09 |
| News fetch + content filter + Lemmy RNG | OK | Ars Technica octopus seed consumed |
| Pattern 1 AISM Filter | OK | Concrete SFX ("high-pitched electrical WHINE"), sensory-first, no em-dash, no Rule of Three, burstiness alternation |
| Pattern 2 Scaffolding Preamble | OK | Dramaturg tone consistent |
| Content filter `[BLEEP]` | OK | "damn" → `[BLEEP]` at REX line (treatment line 63) |
| Director JSON extraction + voice assignment | OK | 3 speakers mapped cleanly |
| `_parse_script` canonical tokens | OK | 17 dialogue + 5 SFX parsed |
| BatchBark hallucination guard | OK | `[clears throat]` + temp floor 0.6 activated first-line per preset |
| EpisodeAssembler crossfades + normalize | OK | 229.9s output |
| h264_nvenc video render | OK | 241.3 MB mp4, 1920x1080@24fps |
| Treatment file writer | OK | Generated alongside mp4 |

---

## Fix Order (proposed)

1. B-002 logging (diagnostic only, no behavior change)
2. B-003 + B-004 post-process rename (deterministic, no prompt risk)
3. B-001 — ONLY after peer review returns with recommended approach
4. Re-run `test.json` + `scifi_lite.json` end-to-end
5. If B-001 still failing after prompt hardening → escalate to v1.3 multi-call, tag v1.2-narrative without Patterns 3/5/6 block guarantees

---

## Regression Guards (must stay green after any fix)

- AST parse clean
- `SCRIPT_SYSTEM_PROMPT.format()` smoke test passes (len ~30480)
- Zero baked names except LEMMY in source
- `INPUT_TYPES` untouched
- Workflow JSONs untouched
- Widget index map stable (16 entries)
- `_LEMMY_RNG = SystemRandom()` untouched
- v1.1 tag `75249ad` untouched
- main `4b33563` untouched

---

*End of v1.2 bug fix guide. AI-consumption format. Next action: peer review return on B-001.*
