# Driver anchor -- 2.4-source: HTML block joins and source-digest scope

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round. The panel
proposes; the driver disposes and verifies every claim against the real Windows files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding.** A no-brainer gets no arc -- this
row is here because grounding found a genuine fork with more than one defensible
answer.

**THE OPERATOR'S BAR:** *"as long as it doesn't crash when it's not supposed to."*
Exactness is NOT the goal -- this is a fun experimental app, and aesthetic drift is
explicitly acceptable. This row is classified **CORRECTNESS**.

---

## 1. What is TRUE TODAY, grounded against current code

Of the row's four clauses, three are closed or refer to dead code (see stale_claims); only the capacity-VALUE half of clause 3 is genuinely live. Current mechanism: nodes/_otr_scifi_news_pro.py:759,3110,3200,4678 already run scifi_news_pro on `output_budget_mode="provider_capacity"` (the whole-artifact retry contract from commit 314dd481, 2026-07-24), replacing the old fixed requested_output=2800-vs-hardcoded-512 framing. The CRASH half of the remaining item was fixed and pushed today: commit d4b55523 (2026-09-11 10:28:34, 'Stop a repair-turn prompt overflow from killing the episode') wraps the repair-turn creative_fn call (nodes/_otr_scifi_news_pro.py:3038-3090) in try/except PromptContextOverflowError, checks `exc.phase == CAPACITY_PHASE_PROMPT_NO_ROOM` and that a draft was actually carried, drops the draft, retries the same rung once at the same temperature, and re-raises anything else (including the GenerationDegeneracyError subclass) -- proven by tests/test_scifi_news_pro_repair_overflow_does_not_crash.py (4 tests, including the two negative cases where the guard must not fire). What remains open: the PREDICTION that decides whether to attempt carrying the draft in the first place, `_draft_fits_repair_turn` (nodes/_otr_scifi_news_pro.py:2861-2915), reads `cap = int(_otr_model_catalog.HARD_VRAM_CONTEXT_LIMIT) or 8192` -- one flat, env-overridable VRAM-shaped constant -- rather than the real context window of whichever transport is actually serving the call (local transformers, GGUF-native/llama-cpp, or OpenRouter cloud each have genuinely different real windows). A too-generous estimate now degrades gracefully (thanks to d4b55523) instead of crashing, but the prediction itself is still wrong by construction for at least two of the three transports.

**Key files:** `nodes/story_orchestrator.py`, `nodes/_otr_scifi_news_pro.py`, `nodes/_otr_scifi_p0_contract.py`, `nodes/_otr_gguf_backend.py`, `nodes/OTR_LedgerScriptWriter.py`, `nodes/story_packs/banks.json`, `docs/GO_FORWARD_PLAN.md`, `docs/GO_FORWARD_ARCHIVE.md`, `docs/HANDOFF_LOG.md`, `tests/test_feed_fetch_seam.py`, `tests/test_scifi_news_pro_repair_overflow_does_not_crash.py`

**Blast radius:** Confined to nodes/_otr_scifi_news_pro.py's own repair-turn draft-carry logic (`_draft_fits_repair_turn` + `_run_markup_ladder`), exercised only by the `scifi_news_pro_multipass` lane. It runs identically on every machine that can pick that bank, and its behavior differs by WHICH writer transport the operator selects for that run (local transformers vs GGUF-native vs OpenRouter) rather than by which physical machine runs it. It does not touch media_archive/original/public_domain/shakespeare/my_story banks, which have their own separate runner code, nor the retired scifi_codex/scifi_news paths (already deleted).

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- "HTML block joins pending an operator digest ruling" is FALSE today. It was ruled and landed 2026-07-30 at commit 331f46ea ("Preserve RSS block boundaries during feed extraction"): nodes/story_orchestrator.py:994-1046 defines `_extract_rss_fragment_text` / `_RSS_FRAGMENT_BLOCK_OR_BREAK_RE` / `_RSS_FRAGMENT_ANY_TAG_RE`, wired live at :1041 (`_select_rss_content`) and reached via :1161/:1306 (`_resolve_body`/`_select_news_body`) in the same NewsFetcher path OTR_LedgerScriptWriter.py:1398-1440 uses to build scifi_news_pro's payload. docs/HANDOFF_LOG.md ~line 11909-11938 records full suite 7898 passed, canonical workflow hash unchanged, a mutation receipt (reverting the production call makes the wiring test red), the exact 4-string fixture from the row's own examples in tests/test_feed_fetch_seam.py (test_production_block_boundaries_are_not_fused etc.), and states plainly 'Item 5 is closed.' It was deliberately scoped future-only ('summary, derived seed_text, URL scraping, normalization, frozen artifacts, and the workflow are untouched') so it does NOT change any already-accepted source digest -- satisfying the row's own stated constraint. docs/GO_FORWARD_PLAN.md section 6 still lists '2.4 source digest ruling (HTML block joins)' as blocked-on-operator; that line is stale and should be removed.
- "scifi_news P0 literal-span convergence" names a bank that no longer exists. `scifi_news` (distinct from `scifi_news_pro`) was retired together with the whole codex lane at commit dae1fb3c (2026-08-16, 'rip: retire the scifi_news bank and the codex lane (full-family teardown)'). nodes/story_packs/banks.json source_bank_id list today (lines 5,37,71,91,129,167,192) is media_archive/original/scifi_news_pro/public_domain/shakespeare/my_story/custom_source_bank -- no plain scifi_news. Whatever P0 literal-span work remains lives entirely inside scifi_news_pro's own contract (nodes/_otr_scifi_p0_contract.py, imported only by nodes/_otr_scifi_news_pro.py per grep), so this clause is not a separate item from clause 3/4, it is dead naming.
- "P9/GGUF follow-ups" -- 'P9' is dead terminology. P0-P9 was the retired `scifi_codex` pipeline's stage numbering (docs/GO_FORWARD_ARCHIVE.md:5345 traces the '<P9 8K structured-capacity follow-up>' note to a 2026-07-30 section literally titled 'The P0 / source-span cluster', about `scifi_codex:P0`); nodes/_otr_scifi_codex.py was deleted at dae1fb3c (2026-08-16). Grep of nodes/*.py today finds zero 'P9' pass-id references. The GGUF-structured-enforcement half traces to docs/2026-07-20-OTR-video-tiers/NEWBUG-gemma4-gguf-structured-enforcement.md, whose only live reproduction was also against the now-deleted scifi_codex:P0; its own 'Status' line says it was never promoted to a PBUG/Bible rule.
- The row's trailing note "No deterministic source-prune rung" is accurate and is itself evidence the row is stale: docs/GO_FORWARD_ARCHIVE.md:8503 (dated 2026-09-04) records this as TOMBSTONED -- `repair_literal_source_metadata`, `_validate_fact_index` and `a0_payload` are absent (0 grep hits) from nodes/ today; only `allowed_source_fields` survives in _otr_scifi_p0_contract.py.

## 3. The fork -- this is what the round must pressure-test

Who/what should `_draft_fits_repair_turn` read the real capacity window from, given local-transformers, GGUF-native and OpenRouter cloud each expose a genuinely different real context/output ceiling? Candidates with no single obviously-correct answer: (a) keep one flat catalog constant but make it env/profile-aware per transport, (b) have each transport module (the local loader, the GGUF backend, the OpenRouter backend) expose its own real window and have the ladder read whichever is active, (c) drop the prediction step entirely and always let the transport's own refusal be the signal (now safe to do post-d4b55523, but spends a guaranteed-wasted call every time a draft would not have fit). GO_FORWARD_PLAN.md section 3 and section 5 (Tier 1) both explicitly name this as needing an arc rather than a constant swap, precisely because it is a three-transport design choice, not a grep-and-fix.

## 4. Questions for the reviewer -- answer these; do not restate section 1

1. **Is the mechanism in section 1 complete and correct?** Read the cited files
   yourself. Is there a step the driver missed, or something that already partially
   mitigates this?
2. **Which side of the fork survives contact with the real code?** Name the files and
   functions each choice would actually touch, and say which you would not write.
3. **What is the smallest change that resolves it?** Smallest that is CORRECT -- not
   smallest that compiles.
4. **Blast radius, measured not asserted.** This project runs on a 16 GB 5080 and an
   8 GB 4060 (CLAUDE.md section 0B). If the change touches shared code, what
   measurement proves the machine you are NOT fixing is unchanged?
5. **Is any part render-inert enough to ship before a four-machine test wave, or must
   all of it wait?** Say plainly.
6. **What would make this row WRONG to do at all?** Argue the other side once.

## 5. Hard constraints -- a proposal violating any of these is rejected on sight

- **No new content gates**, story-rejection gates, word/duration/cast-size limits,
  standalone checkers, chunkers, or recursive retry loops.
- **No prompt rewrites.** Prompts are hand-crafted per model and CHARACTER-BUDGETED
  (`motion_registers` 240 chars enforced at load, BUG-LOCAL-112; `_fit_motion_slot`
  truncates to 60). Adding conditional nuance spends budget that does not exist.
- **Story/prose QUALITY work is DONE** by operator directive. Correctness defects are
  still open; better prose is not.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful. The goal is to stop CAUSING avoidable ones, never to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more than
  it saves is a worse fix.
- **One canonical graph.** No second workflow JSON, no new output-path owner, no
  reviving deleted code.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling.
- Style: no curse words, never the name "dummy".
