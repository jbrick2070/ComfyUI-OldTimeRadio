# OTR Handoff Log

Append-only session log, newest at top. What each session actually did;
GO_FORWARD_PLAN.md stays lean and forward-only.

## 2026-07-24 07:35 -- HEAD ed8d5a6d (v2.0-alpha) -- WINDOW PLANNER (Fable)

Did: leaned GO_FORWARD_PLAN.md to open work + bugs only (665->398 lines; done
strata retired to git history + this log; stale refs re-grounded: retired
banks pruned, phase-C gating -> "no code mid-sweep"); added the MODEL & CREDIT
BUDGET section + per-window model rungs; authored + delivered otr-handoff
SKILL v2 (commit-AND-push policy, tracker/audio-freeze staleness removed).
Current step: land the dirty-tree six-bank no-prose-gate chunk (active coder
window); PLANNER next = sec-16 ratification + r5 extensibility kibitz.
Next: fresh Opus PLANNER window takes this baton; run the sec-16/r5 kibitz
(codex gpt-5.6-sol high + agy Gemini 3.5 Flash (High)) while both pools fresh.
Models: Claude Fable, docs-only session; no panels or roundtable spent.
Commits: ed8d5a6d + this handoff commit.


## 2026-07-22 early -- v2.0-alpha [CODER: live candidates stay fresh]

PBUG-20260721-18's episode-liveness root fix is pushed at `67996907`; the
live qualification follow-up is pushed at `81ee21df`. The deterministic
in-band ledger remains the only delivery judge. Four consecutive no-progress
calls retire only the current producer candidate. Row repair escalates to the
alternate LLM and then to another complete producer-owned candidate without an
outer model-output ceiling.

ROOT FOLLOW-UP:
- Canonical prompt `32b374e2-7c89-4d4a-bb8c-42e180571ecc` stayed alive for
  more than two hours and retired more than a dozen candidates, proving that
  no LLM miss or observer exit killed the episode. It also exposed a real
  convergence defect: both logical slots resolved to the same seeded Gemma
  backend, so two fixed P5 prompt shapes replayed the same drafts.
- Every complete reroll after Candidate 0 now carries a model-visible,
  monotonically unique candidate nonce and explicit fresh-candidate
  instruction. The compact typed-repair context preserves that identity.
  Corrected prompt `3fdf7349-7b2e-46f5-8182-982f72e5e261` has already
  produced visibly distinct Phase One/Phase Three P5 candidates and continued
  through P6/P8 without a terminal episode failure.
- `poll_history(timeout_s=0)` is now explicit wait-until-terminal operator
  mode. Default callers retain the 5,400-second timeout; only the overnight
  qualification harness opts into no observation wall clock.

VALIDATION: whole Windows suite **8,349 passed / 33 skipped / 1 expected
xfail** in 205.28 seconds. Bug Bible 12.70 passed **17 / 23 skipped / 3
expected xfails**. AST, UTF-8/no-BOM, nonzero-file, JSON round-trip, link/input,
live widget-vector, and OTR workflow validator coverage are green. The
canonical workflow stayed byte-identical at
`f9d9c2c3a101ec607c9658456f6e191a164d8214be7b6d560bc68975d0511e9a`
(23 nodes / 58 links). `HEAD == origin/v2.0-alpha == 81ee21df`.

LIVE QUALIFICATION: run tag `qual320_nonce_20260722` is active. A hidden
pass-gated chain adopted the corrected `scifi_news` canary and will launch
`scifi_news_pro`, `original`, `media_archive`, `public_domain`, and
`shakespeare` sequentially only after each prior leg records RESULT SUCCESS
and passes the strict ledger, exact word receipt, caption, credits, asset, mux,
and OBS publication audit. Any real leg or audit failure stops the chain.


## 2026-07-20 late -- v2.0-alpha [CODER: spoken hygiene ships with a stamped repair]

Closed PBUG-20260720-03: a CRAFT/quality rejection on one voiced row can no
longer terminal-skip an otherwise renderable episode. The contract now applies
to all six runnable banks (`media_archive`, `original`, `public_domain`,
`shakespeare`, `scifi_news`, and `scifi_news_pro`).

ROOT FIX:
- Added a total per-line ladder: the existing same-slot repair, a sharpened
  gate-specific CRITICAL repair at lower temperature, the other writer slot,
  then an idempotent deterministic SFW floor. Every accepted repair is
  rescored and stamps `hygiene_repaired_after_reroll:<gate>:<rung>`.
- Extended the floor across the full spoken contract. Existing cliche and
  stage-business scrubbers are now terminal rungs; whole-line action/cue text
  becomes a short speakable utterance; one-breath, anchor-stuffing,
  objective-literal, on-the-nose, and thesis findings receive bounded
  sentence-preserving repairs. Speaker-aware detection catches a character
  narrating their own action by name. Non-dialogue material is not moved into
  SFX yet; that ledger layer remains future work.
- Moved whole-script Codex P5/P7/P9 craft failures inside the typed-repair
  factory, after graph/roster preflight, so a local wording defect never spends
  or truncates a whole `ScriptArtifactV4` retry. Content-owned lanes repair the
  exact TTS projection before rebuilding raw/parsed/proof/hash seals. Shared
  lanes receive a final ledger scour plus a post-readiness guard.
- Removed quality exhaustion from terminal freeze semantics. Empty output from
  the mechanical floor is isolated to that row; genuinely invalid graph state
  remains structural. The deterministic G9 SFW/content-safety ship-stop was not
  softened and still sanitizes or fails closed.

VALIDATION: focused cascade/Codex coverage **119 passed**; expanded six-bank
surface **395 passed**; workflow/freeze surface **268 passed / 3 skipped**;
whole Windows suite **8161 passed / 33 skipped / 1 expected xfail**. The clean
survival-guide worktree passed **17 / 19 skipped / 3 expected xfails** and
BUG-11.56's OTR executable regression passed; portable rule update
`ef7e327ded9cf80b9f050a690b4e09cc33d8e8d7` is pushed to the guide's `main`.
`workflows/otr_canonical.json` needed no node/input/widget/link change and stayed
byte-identical (`222D19478A308C91171DFCBDCCBEC01C55DD639283E2550EBB59EB9842D0882D`);
validator, JSON round-trip, 23-node/57-link audit, live input names, references,
and widget-vector drift (`0`) are green.

LIVE PROOF: an initial canonical episode, `signal_lost_the_price_of_wakefulness_20260720_210832`,
published successfully through the late floor and exposed the remaining
whole-artifact boundary and raw-token trim; both were root-fixed. Final
canonical prompt `f3770246-2d6a-4302-90af-153120edddf2` then hit real defects at
P5 (`one_breath`, four rows) and P7 (`spoken_format` / `stage_direction`). Each
immediately logged `craft-only rejection resolved by the line-local A/B/C/floor
cascade; whole-artifact repair bypassed`, and the ledger carries
`shared_artifact_repair_bypassed=true` plus gate/rung stamps. The episode froze
`frozen_with_warns` (only stale word-count telemetry), rendered all four clean
lines / 45 words, completed TTS/video/captions/credits, and published the
22,892,541-byte OBS asset:
`output/otr/obs/signal_lost_the_weight_of_height_20260720_221418_silent_procgen_blended_captioned_with_credits_final.mp4`.
Targeting 30 words already produced a clean 45-word episode, so no minimum-word
widget change was needed; increasing the minimum would not fix the separate P9
8K structured-artifact capacity limit.

## 2026-07-20 -- v2.0-alpha [CODER: Gemma 4 12B Transformers/HF writer restored]

Restored `google/gemma-4-12b-it` as the saved creative + technical writer on
the fully local, in-process Transformers/HF lane. OTR uses no Ollama,
llama.cpp, model sidecar, or model-serving port for this path. The official HF
weights remain under `C:\ComfyUI-Models\huggingface\hub`; both canonical slots
select `cuda` / `sdpa` / `bnb_nf4`, OTR context 8192. No LoRA, adapter, or
auxiliary tensor artifact is required.

ROOT FIX:
- Upgraded the runtime contract to native Gemma4Unified support
  (`transformers>=5.10.4`), restored the curated HF row, removed its hard
  reject, and made cache resolution pair the newest materialized-weight
  snapshot with the newer local chat-template metadata revision.
- Kept tokenizer/config/model loading fully offline with
  `local_files_only=True`; there is no hidden Hub fallback in `load_llm`.
- Bound each exact P0-P9 result schema into the real local scheduler calls,
  including the narrower P3 authored-text patch, so lm-format-enforcer removes
  invalid JSON continuations at token selection.
- The first live leg found one grammar-compiler incompatibility in P5:
  `list[dict[str, Any]]` emitted `additionalProperties:true`, which LMFE 0.11.3
  treated as a schema object and crashed on. `ScriptSceneV4` now expresses the
  actual closed `scene_id` / `env` / `description` contract. A complete P5
  artifact is exercised character-by-character through the installed parser.
- Updated the real `workflows/otr_canonical.json` and revalidated all 23 nodes,
  57 links, positional widget vectors, live input names, references, and JSON
  round-trip.
- Added `scripts/otr_gemma4_doctor.py` for the official offline NF4 + coherent
  prose + constrained-JSON contract. Bark/MusicGen compatibility tests stay
  green under Transformers 5.10.4. The separately installed legacy
  `parler-tts 0.2.2` pin is incompatible with Transformers 5 and must remain
  isolated; Parler is not an OTR dependency.

MEASURED: official `Gemma4UnifiedForConditionalGeneration`, 331 Linear4bit
layers, `is_loaded_in_4bit=True`; 7.152 GiB allocated / 7.286 GiB doctor peak,
and a 7.15 GiB live model-load delta. Canonical structured generation peaked
around 13.9 GiB total GPU use including the desktop baseline and KV state,
inside the 16 GB board.

VALIDATION: exhaustive fresh-process inventory **8123 passed / 33 skipped / 1
expected xfail across all 488 test files**. Focused post-fix compatibility
surface: 291 passed / 2 skipped. Survival-guide suite: 30 passed; BUG-02.16 and
BUG-11.55 OTR regressions: 2 passed. The Bible loader reports 205 entries and
only its 12 pre-existing xref-tag format findings.

LIVE PROOF: canonical prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f` loaded the
official checkpoint fully offline for both slots. P0 began with `{`, decoded,
and reached semantic source-span validation; deterministic coordinate repair
accepted it without another model call. P1-P4 and P3 rewrite cleared. P5 then
produced a complete schema-valid JSON artifact, proving the LMFE crash fixed.
The leg did **not** publish media: Gemma repeated an existing spoken-hygiene
defect after its bounded P5 model repair, so the lane failed closed as designed.
This is a runtime/grammar qualification through P5, not a full-episode or
comparative quality-bakeoff verdict.

LM STUDIO CONVENIENCE: imported the existing Q4_K_M GGUF as an NTFS hard link
at `C:\ComfyUI-Models\LMStudio\unsloth\gemma-4-12b-it-GGUF\gemma-4-12b-it-Q4_K_M.gguf`.
It consumes no second weight copy and is separate from OTR's HF runtime. LM
Studio and its service/server were left stopped.

STILL OPEN: the GGUF lane's structured-enforcement gap remains separate and
the optional GGUF row was not presented as the canonical writer. The local
Gemma-vs-Mistral quality matrix also remains open.

## 2026-07-18 evening -- HEAD `ed7b37de` (v2.0-alpha) [RENDER->CODER: short-episode structural COUNT gates -> advisory (Gate 3)]

Started as the RENDER window for the local Mistral-Nemo bake-off (codex_v4 vs
fable2 vs base codex). Precondition confirmed (HEAD `c507acff`, exact 8-id roster).
The Step-1 wiring smokes surfaced a blocker that turned into the session's real work.

DIAGNOSIS (docs/2026-07-18-render-step1-blocker.md):
- 30w AND 120w canonical smokes hard-fail in the WRITER on deterministic STRUCTURAL
  COUNT gates: codex P3 exact-beat-count (`beat count 6 must equal advisory 12`;
  root: `_otr_scifi_codex.py:3297` derived beats from `cast*3`, word-blind), and
  fable2 WORD_BUDGET/SCENE_COUNT bands. NOT a rip regression (c507acff never touched
  the video path or fable2 lane); the gates are v4-bake-off-era regressions (git:
  `c22eef0a`/`c942b2ae`/`95582643`) -- the pre-source-bank lanes ran any length.
- One EARLY false lead: the first codex smoke booted in leaked `OTR_TEST_MODE=1`
  (Start-Process inherits parent env) -> in-memory stubs -> empty video manifest.
  Fixed the harness (leg runner strips test env) and re-ran clean.
- Governing contract = `docs/SOURCE_BANK_PREFLIGHT.md` Gate 3: "no model-produced or
  unused count field can gate production"; `target_words` advisory, never a fatal
  quota gate. The gates were non-compliant.

FIX (committed `ed7b37de`; operator-approved "fix the gates", kibitz r3 hardened):
- codex: beat count scales to the word budget; a beat-count mismatch is RECONCILED
  (advisory rebuilt to the draft's actual count) and propagated into P3/P4/P3_rewrite;
  cast_coverage is advisory; an out-of-range cue anchor is deterministically CLAMPED.
  Dangling-reference gates (shot_index/cast_id/fact_id/cue_id/unused_shot/graph) stay
  fatal -- `_validate_radio_score_graph` still closes.
- fable2: word/scene COUNT defects drive bounded rerolls only; on exhaustion the
  cleanly-parsed draft is ACCEPTED and residuals recorded advisory in the ledger
  (`f2.parse`/`parse_p5`). PARSE defects still fail closed.
- kibitz r3 (Codex + Antigravity/Gemini 3.5 Flash High) caught 3 real wiring gaps I
  folded: advisory-recording, the P3_rewrite reconcile propagation, and the
  cast_coverage accidental-fatal-successor (both fired correctly on the live leg).

VALIDATION: full suite 8082 passed / 32 skipped / 1 xfailed; Bug Bible 17 passed;
AST + no-BOM + HEAD==origin verified. LIVE PROOF: `scifi_fable2` 120w Mistral-Nemo
leg RESULT SUCCESS + obs_publish OK + asset on disk ("The Caretaker's Dilemma",
108.0 MB) -- previously a hard WORD_BUDGET fail.

STILL OPEN (a SEPARATE facet, NOT count gates): codex_v4 short legs still fail
stochastically on P2 cast-name Title-Case (e.g. `Maxwell 'Max' Hart`) and P5
self-vocative -- a codex-writer robustness follow-up under the same Gate-3
"mechanical normalization" principle (P2 could be as small as stripping quote
tokens from names). The local Mistral bake-off itself is NOT yet run (blocked on
these codex facets); run it at 420/720w once codex short legs are clean.

## 2026-07-18 midday -- baseline HEAD 178e935a (v2.0-alpha) [CODER: Sonnet-bake-off rip -- 4 banks retired]

Executed `docs/2026-07-18-rip-4-banks-plan.md` in one green chunk.

Did:
- RETIRED `scifi_sonnet_v3` (FULL sonnet lane): bank row + pack + story_rules +
  `sonnet_archive_multipass_v3` pipeline (both registries) + the
  `_run_scifi_sonnet_lane` runner + deleted `nodes/_otr_scifi_sonnet.py` +
  `tests/test_scifi_sonnet_lane.py`. RETIRED `media_archive_v3` / `scifi_codex_v3`
  / `scifi_fable2_v3` (v3-only): row + pack + story_rules + each dedicated pipeline
  in BOTH `_RUNNER_BY_PIPELINE` and `pipelines.json`. KEPT the `scifi_codex` /
  `scifi_fable2` / `media_archive` bases, `scifi_codex_v4`, and `legacy_many_pass_v3`.
  Roster: 12->8 visible, 11->7 runnable.
- MUST-KEEP fence honored: deleted ONLY `_make_v3_runner`; KEPT `run_v3_advisory`
  / `_v3_focus_metric` / `_v3_max_run` (public_domain_story_v3 + shakespeare_v3 call
  them every render). KEPT the now-unreachable `base=="scifi_sonnet"` focus branch
  and the `_otr_scifi_p0_contract.py` P0-contract comment -- the only 2 surviving
  bare-`scifi_sonnet` hits, both in shipped code. Dropped `fable2_multipass_v3` from
  the writer target-word gate; refreshed the stale `_RUNNER_BY_PIPELINE` comment.
- CLEAN RIP tests (positive only): migrated the surviving-machinery advisory tests
  to `public_domain_story_v3` / `shakespeare_v3`; scrubbed `_otr_scifi_sonnet`
  imports + sonnet-only cases from schema-parity / rss-admission / source-repair;
  regenerated the roster/bijection pins and the v4-guard `_CURRENT_BANKS` lists.
  Operator eyeball on the v4-guard gate-off contrast: KEEP base `scifi_codex` (guard
  genuinely OFF), NO `_v4` substitute.
- NEWBUG->PBUG: appended `PBUG-20260718-01` to PROD_BUG_LOG FIRST, then marked
  `docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md` CLOSED-BY-RIP (retained, never deleted).
- Docs: README roster table, GO_FORWARD current-roster + NEWBUG note refreshed.

Gate (all green): import-smoke 0 skips; `_ensure_loaded()` carries no retired
pipeline id (atomic delete validated by the crossref sweep); `otr_canonical.json`
byte-unchanged; source-only retired-id scan over nodes/tests/workflows = ZERO;
bare-sonnet scan = EXACTLY 2 (both kept); no surviving `meta["scifi_sonnet"]` reader;
runtime-advisory proof via the migrated `public_domain_story_v3` unit test (plan's
"targeted unit test OR 30w live smoke" -- unit-test path taken; live smoke not run);
**full Windows suite 8081 passed / 32 skipped / 1 xfailed** (was ~8144 pre-rip -- drop
is the retired banks' own tests); **Bug Bible 17 passed / 16 skipped / 3 xfailed**;
no-BOM/UTF-8 + AST-parse on every touched file. Counts recorded, not pinned.

## 2026-07-18 morning -- HEAD 60c73618 (v2.0-alpha) [RENDER: Sonnet-4.5 cross-bank bake-off COMPLETE]

Did (render window, autonomous overnight):
- Ran the creative=`claude-sonnet-4.5` (OpenRouter remote) / technical=`Mistral-Nemo` (local) bake-off
  across all 11 runnable banks x 420/720 = 22 story-only legs (18 SUCCESS / 4 FAIL). Built the harness
  (tmp/_sonnet_bakeoff_sweep.ps1); fixed 2 wiring bugs live: the concrete-4.5 dropdown pin (the picker
  prunes concrete slugs for ~latest aliases -> surface it via OTR_OPENROUTER_SLOT_A_DEFAULT) and the
  -Banks [string[]] array-binding trap via Start-Process/-File (-> single comma-string the script splits).
- Fable BLIND grade of the 10 720-SUCCESS transcripts. NEW WINNER under Sonnet = scifi_codex_v4 (24/25,
  "The Halicin Gamble"); runner-up scifi_fable2 (24/25, monologue-capped at 720); the codex circuit swept
  #1/#3/#4; weakest scifi_sonnet_v3 (12/25, essayistic). The crown SHIFTS from the aion baseline's fable2.
- Cost ~3.07M Sonnet tokens ~= $15-20 (creative slot only; technical local/free; 0 creative VRAM).
- FAILs diagnosed: original_radio 420 (deterministic news_source_framing gate; PASSED at 720),
  scifi_codex_v4 420 (codex P5 all-caps-word gate; PASSED at 720), scifi_fable2_v3 BOTH tiers = NEWBUG
  (fable2 revision_contract hardcodes rules_id=='scifi_fable2', model-independent) ->
  docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md.
- Scoreboard: docs/2026-07-17-model-bakeoff-scoreboard.md. Full-media confirmation: the winner
  scifi_codex_v4 @ 720w canonical FAILED fast (codex 240-char string_too_long on a fresh source -> the
  winner is production-fragile with Sonnet, different gate than its 420 all-caps fail). Re-ran on the
  robust runner-up scifi_fable2 @ 720w -> RESULT SUCCESS + obs_publish OK ("The Stone Frequency", 406 MB,
  34:12). Shippable Sonnet pairing = scifi_fable2; codex_v4 = best script, least reliable producer.
Current step: bake-off item 3 (Sonnet arm) DONE; Mistral-Nemo stays the free local default, cloud opt-in.
Next: (coder) fix the scifi_fable2_v3 rules_id NEWBUG; (render, optional) the local mistral/gemma writer matrix.
Commits: docs only (scoreboard + NEWBUG + GO_FORWARD + HANDOFF); NO code changes (NEWBUG deferred to coder).

## 2026-07-17 night6 -- HEAD 9730e2dc (v2.0-alpha) [v4 P2 bank #1 scifi_codex_v4 GREEN + LIVE-PROVEN]

Did (coder window, autonomous + operator cross-check):
- Resumed after the operator cross-check verdict: BUG A (P0 literal-span fail) = NEW upstream root in the
  S5 family (-> PBUG-20260717-01); BUG B (P3 premise string_too_long) = re-occurrence of PBUG-20260713-04,
  and my base-seam 144 re-add was the -04 anti-pattern (exposing the rejection edge).
- P0 fix @ 26ba8e1d: normalize the 4 span-bearing source fields to single-spaced text in
  validate_payload_envelope -- at admission, UPSTREAM of the digest/projection/validator (BUG-11.37
  offset-shift constraint); point the P0 validator at env.payload. Codex-scoped (shared
  validate_source_payload stays byte-identical for science). +1 test; reverted the anti-pattern caps.
- Live legs: 6883758f (P3 premise), ac027c36 (P0), 90f22b15 (cleared P0 -> BUG A proven, then P3
  string_too_long on premise+description: the -04 recipe is insufficient for the verbose v4 lane).
- Operator "allow longer text": P3 fix @ 9730e2dc = RAISE the non-spoken metadata caps (premise 144->240,
  scene/shot description 72->144) across draft+final models + _p3_text_patch_cap + replacement_text schema
  + receipt. Caps are LOAD-BEARING (P3 draft fits the 8192 context+output budget) -> resized the reservation
  1647->1829 + updated every exact-token guard (max-width helper draft 1418->1576; envelope re-verified
  prompt+output=5935<=8192). Full suite 8144 / Bible 17 at each chunk; canonical unchanged.
- LIVE PROOF: leg c1f3891f RESULT SUCCESS + obs_publish (signal_lost_the_whisker_effect..._final.mp4,
  56.6 MB; obs + episode dirs Test-Path OK). Bank #1 DONE.
- PBUGs: PBUG-20260717-01 (P0) LIVE-VERIFIED; BUG B recorded as re-occurrence of -04 (not a new PBUG);
  PBUG-20260710-07 = retire candidate via the green codex leg (announcer rows clean, freeze passed).
Current step: bank #1 scifi_codex_v4 GREEN + live-proven. NEXT = bank #2 shakespeare_v4.
Next: build shakespeare_v4 (own idiom; inline legacy_many_pass_v4; genre+outro gates safe there;
  pre-emptively raise tight non-spoken caps + resize the budget/guards when raising).
Commits: 3b74b7e3 (contract-visibility fold), 48f2a278 (caps re-add, later reverted), cc76dcc5 (pause docs),
  26ba8e1d (P0 fix + anti-pattern revert), 9730e2dc (P3 caps raised). All pushed, HEAD==origin.

## 2026-07-17 night5 -- HEAD 48f2a278 (v2.0-alpha) [v4 P2 bank #1: two-strikes kibitz + P3 fold; LIVE LEG BLOCKED at P0+P3 -> PAUSED for cross-check]

Did (coder window, autonomous):
- Two-strikes gate on the codex P3 contract: ran /kibitz r2 (local $0; Codex gpt-5.6-sol + Antigravity
  Gemini 3.1 Pro, both grounded + Claude anchor/judge). Panel BROKE the framing: the seam cap-restatement
  was (argued) redundant with the surface instruction's tighter ceilings, and FOUR deterministic P3 compiler
  gates were model-invisible (unused_shot/cast_coverage/cue_id/cue_anchor), plus a 12-beat distribution trap.
  Folded grounded survivors @ 3b74b7e3: reverted the cap list, exposed the 4 gates + 12-beat clause in the
  shared surface/topology instruction, enriched the beat_count receipt (observed-vs-expected), +4 tests, doc
  fixes (PBUG cites -> -02/-06; cast beats 6/9/12; P5 does not cap prose). Suite 8143 / Bible 17.
- LIVE 30w Mistral-both leg 1 (6883758f) FAILED P3 string_too_long on `premise`. Grounded: the text-patch
  deliberately never clips prose (_otr_scifi_codex.py:1748), so model-visible caps are the ONLY lever -> the
  live evidence OVERTURNED the panel's "redundant" call (reverting the caps regressed it). RE-ADDED the caps
  + a premise-brevity nudge @ 48f2a278 (suite 8143 / Bible 17).
- LIVE leg 2 (ac027c36) then FAILED EARLIER at P0 PostValidationError -- FactIndex literal-span vs
  whitespace-polluted RSS source (full_text leading \n+8 tabs; offset slices land mid-word; model
  paraphrases; exact-literal contract rejects). PRE-EXISTING + SHARED across all codex banks; NOT v4-caused.
  So the P3 caps fix is UNPROVEN (leg 2 never reached P3).
- Per operator's new directive, wrote BOTH bugs as problem statements (docs/2026-07-17-v4-campaign/NEWBUG-*.md)
  for a cross-check window. Operator chose PAUSE for cross-check. Reset the box (killed the resident server).
  NO further codex code until the operator returns the fix approach.
Current step: v4 P2 bank #1 live leg BLOCKED at P0+P3 -> PAUSED for operator cross-check vs past PBUGs.
Next: operator cross-checks BUG A (P0 span/whitespace) + BUG B (P3 premise) vs PROD_BUG_LOG/BUG_BIBLE/
  BUG_SYMPTOM_INDEX; then kibitz the offset-sensitive P0 fix + confirm the P3 caps on a leg that clears P0.
Commits: 3b74b7e3 (contract-visibility fold), 48f2a278 (caps re-add). Both pushed, HEAD==origin.

## 2026-07-17 night4 -- HEAD 1fd7743d (v2.0-alpha) [v4 P2: bank #1 scifi_codex_v4 CODE SHIPPED]

Did (coder window, autonomous):
- Built scifi_codex_v4 as a fully INDEPENDENT bank: banks.json row (before custom) + pack
  nodes/story_packs/scifi_codex_v4/scifi_codex_v4.json (11 codex seams + the proof-pressure
  delta: want / gating proof / mandatory cost beat / one reversal) + story_rules/scifi_codex_v4.json
  (exact id) + pipeline scifi_codex_circuit_v4 mapped DIRECTLY to _run_scifi_codex_lane (NOT the
  v3 advisory wrapper) + roster/bijection tests (test_bank_variants 11->12 visible/10->11 runnable
  + TestScifiCodexV4; test_fable2_registry tail/order). Gates ON: require_science_floor +
  placeholder_guard(G13) + scene_coherence_check(G15). Gates DEFERRED: genre_guard_spoken(G10) +
  require_outro_cast_complete(G12) -- the dedicated codex runner does NOT cross the inline I.7/I.8
  authored-repair boundary, so they would be no-repair hard gates (vetoable). Full suite 8139 /
  Bible 17 / AST+JSON+BOM clean / canonical hash unchanged / HEAD==origin. Commit 1fd7743d pushed.
- Live 30w leg via scripts/otr_headless_canonical.ps1: attempt1 (Mistral-Nemo both) AND attempt2
  (gemma-4-E4B creative) BOTH failed at codex P3 RadioScoreV4 string_too_long -> proven
  MODEL-INDEPENDENT = the unstated-cap class (PBUG-20260713-11/12). ROOT FIX (operator-steered):
  restate the exact RadioScoreV4 caps in the codex_radio_score_system seam -- NOT a model swap.
  Re-proving with Mistral-both + the restated caps (the strict model-agnostic test).
- Wrote docs/BANK_PLAN_scifi_codex_v4.md (tracked; wiring + gate rationale + the PBUGs/lessons +
  the go-forward recipe for the remaining 4 banks).
Current step: scifi_codex_v4 code shipped @ 1fd7743d; live leg re-proving with the P3 cap fix.
Next: confirm RESULT SUCCESS + obs_publish + asset; if green, commit the cap fix + bank plan +
  doc refresh and retire PBUG-20260710-07, then bank #2 shakespeare_v4 (inline lane -> genre+outro
  gates ARE safe there). If the fix leg fails P3 again -> /kibitz (two-strikes) before a 3rd fix.
Commits: 1fd7743d (code). Cap-restatement + bank plan + doc refresh pending the live proof.

## 2026-07-17 night3 -- HEAD d29ba920 (v2.0-alpha) [v4 campaign: PHASE 1 COMPLETE (ii-viii)]

Did (coder window, autonomous -- continued):
- P1(v) @ 0066f5ab: outro cast-completeness. New nodes/_otr_outro_guard.py (final
  cast = character char_ids with a non-skipped spoken line -> name; outro = LAST
  announcer line BY POSITION; missing = name absent full-or-significant-token,
  casefold word-bounded, titles ignored). Authored keep-if-complete repair (creative
  slot; Python never appends prose; restores original on exhaustion). Deterministic
  G12 terminal, opt-in via defaults.require_outro_cast_complete. Root fix caught by
  my own tests: outro is positional, not last-non-empty (that was the intro).
- P1(vii) @ e7bfb1fe: literal placeholder-token guard. New _otr_placeholder_guard.py
  (whole-value, token-boundary, quote/punct/case-tolerant over NAMED fields; X/Y/TBD/
  ...; 'X marks the spot' NOT flagged; music out of scope). G13, opt-in
  defaults.placeholder_guard. No repair (placeholder = generation bug the pack fixes).
- P1(viii) @ 4f8bd7aa: source-provenance normalizer. New _otr_provenance.py
  (public_domain license_status + shakespeare license_label/commercial_use_allowed +
  synthetic -> one record; spoken_coda + printed_credit templates). Writer stamps
  meta.provenance + fills credits_source_line when the bank default did not.
  Deterministic G14 blocks publish on research_only (operator decision).
- P1(vi) @ d29ba920: header<->scene STRUCTURAL coherence. New _otr_scene_guard.py
  (unique scene_ids + no non-music line referencing an undeclared scene). Semantic
  scene-vs-beat match is an unlawful LLM gate -> structural only; exact
  scene.line_count matching omitted (unit-ambiguity risk). G15, opt-in
  defaults.scene_coherence_check. INTERPRETATION FLAGGED structural (vetoable at the
  Phase-2 consuming chunk). Done LAST after vii/viii per its under-specification.
- Pattern for all 7: each shared fix is a SELF-CONTAINED module + a deterministic
  terminal in _otr_ledger_freeze.run_gap_audit (G10 genre, G11 beat-floor, G12 outro,
  G13 placeholder, G14 provenance, G15 scene) -- the ONE path every execution family
  crosses (codex phase_10 finalizer, inline run_freeze_cascade, fable2 finalizer),
  mirroring G9. Every gate is OPT-IN via a validated scalar bank default (_parse_bank
  bool loop) -> INERT for all 10 current banks, so the full suite stayed green while
  the machinery is ready for Phase-2 v4 banks to flip on. THE LAW honored throughout
  (deterministic terminal ends; authored repairs only improve).
- Gates each chunk: full suite (8018->8031->8061->8084->8110->8134) + Bible 17 +
  AST/BOM/zero-byte + commit AND push + HEAD==origin. No canonical JSON change (no
  graph edit in any Phase-1 chunk).
Current step: v4 campaign PHASE 1 DONE. NEXT = Phase 2 -- build the 5 v4 banks,
  serialized, each an atomic per-bank chunk gated on a LIVE GPU leg (RESULT SUCCESS +
  obs_publish + asset). Order: scifi_codex_v4, shakespeare_v4, public_domain_story_v4,
  media_archive_v4, original_radio_v4.
Next: scifi_codex_v4 -- bank row + pack + story_rules(exact id) + pipeline
  scifi_codex_circuit_v4 (executable:true, runner-map) + roster/bijection tests; flip
  the opt-in gates it wants; runnable:true LAST; then the live leg (per-lane
  announcer-sentinel mint retires PBUG-20260710-07).
Commits: 0066f5ab, e7bfb1fe, 4f8bd7aa, d29ba920 (+ f5acd44a docs checkpoint)

## 2026-07-17 night2 -- HEAD 90ed495e (v2.0-alpha) [v4 campaign: P1(ii)+(iii)+(iv) pushed]

Did (coder window, autonomous):
- P1(ii) @ f859036c: named regression pinning PBUG-20260710-07 (the cast-keyed
  mutation class) -- INVARIANT A (every coercion stamps a role_coerce reason
  breadcrumb + meta.role_coercions audit; no silent flip) + INVARIANT B
  (announcer-sentinel / name-excluded lines never coerced). Test-only; NO coerce
  code added (root fix shipped pre-campaign; adding more = shim). PBUG stays
  ROOT-OPEN until a live v4 leg. tests/test_pbug_20260710_07_cast_keyed_mutation.py.
- P1(iii) @ e7ba2627: bank-aware GENRE/spoken-text guard. New nodes/_otr_genre_guard.py
  (casefolded/Unicode boundary matcher: gun !~ begun, +s/es plural, phrase ws-flex;
  + writer-boundary authored repair via creative slot, keep-if-clean, never raises,
  breadcrumb). Deterministic terminal = G10 in run_gap_audit -> Phase-10
  FreezeAssertionError (one path every family crosses; mirrors G9). OPT-IN via
  validated scalar default defaults.genre_guard_spoken (default False -> INERT for
  all 10 current banks; v4 banks flip in Phase 2). Fixed 2 static-audit collisions
  at root (LLM slot tag on the creative_fn call; label=pre in the collect-test).
- P1(iv) @ 90ed495e: beat_bounds structural contract in _otr_episode_budget
  (WORDS_PER_BEAT=40 SOFT/recorded; STRUCTURAL_MIN_BEATS=3; family caps codex 12 /
  inline 40; target_beat_count round-half-up; classify) + deterministic G11 floor
  terminal in run_gap_audit (opt-in via meta.beat_bounds; counts distinct spoken
  beat_ids; raises below floor). Writer stamps meta.beat_bounds. Operator: length
  recorded-not-gated -> only the structural floor gates; MAX + word->beat derivation
  deferred to Phase-2 live. 8031 suite green with the writer stamping every real
  episode = empirical proof the floor never false-fails a shipping lane.
- Each chunk: full suite (7980 -> 8018 -> 8031) + Bible 17 + AST/BOM/zero-byte +
  commit AND push + HEAD==origin verified. No canonical JSON change (no graph edit).
Current step: v4 campaign Phase 1 -- P1(v) outro completeness validator (next), then
  (vi) header<->scene, (vii) placeholder token, (viii) provenance normalizer; then
  Phase 2 (5 v4 banks, each a live GPU leg).
Next: P1(v) bounded authored per-line outro patch (Python only canonicalizes an
  already-present unambiguous alias; never appends prose; seed from episode seed).
Commits: f859036c, e7ba2627, 90ed495e

## 2026-07-17 night -- HEAD c3a9d420 (v2.0-alpha) [v4 campaign: Phase 0 done + P1(i) pushed]

Did:
- Phase 0: root-caused PBUG-20260710-07 STATICALLY -- the D3 pre-freeze coerce
  sweep (_otr_freeze_cascade.py:1367 -> production_ledger.coerce_speaker_role_for_char_id)
  resolves the announcer<->char_id ambiguity via cast_ids (announcer-named slots
  excluded; the "Chandra c02" mis-stamp is a real character, correctly coerced).
  Already closed by sentinel char_id mint + name exclusion + the role_coerce
  compose_flags breadcrumb; pinned by tests/test_d3_role_coercion.py (14/14). NO
  coerce code change -- adding one is a shim (operator directive). Durable v4
  protection = per-lane "announcer lines carry the sentinel char_id" minting
  invariant, enforced in Phase 2; a live v4 leg formally retires the PBUG (kept
  ROOT-OPEN in PROD_BUG_LOG until then). Exact-id/sidecar audit + nine-defect
  disposition done. Defect #2 (name-splice) stays OPEN per the timebox.
- P1(i) @ c3a9d420: validated scalar bank defaults (style_pool_class,
  require_science_floor, propagate_adaptation_cast) added to _parse_bank; deleted
  the strict_v4_banks set + the (shakespeare_v3,public_domain_story_v3) tuple + the
  media/adaptation literal branches in select_style. Writer stamps
  meta.style_pool_class from bank.defaults; select_style reads meta (hash keys
  UNCHANGED -> byte-identical slugs, C7); science-floor + adaptation-cast consumers
  read bank.defaults directly. Migrated all 10 runnable banks.json rows.
  tests/test_bank_scalar_defaults.py (new, 27) + updated test_style_catalog.py.
  Full suite 7974 passed / 32 skipped / 1 xfailed; Bug Bible 17; AST/JSON/BOM PASS.
  Visual-STYLE pool axis is separate from the source FEED (science_rss vs
  media_archive_rss); scifi_fable2 keeps the science_rss feed but no science floor
  (matches prior). base_source_bank_id retained (bakeoff logic) -- only its use in
  the 3 consumers removed.
Current step: v4 campaign Phase 1 -- P1(ii) breadcrumb regression + reason stamp.
Next: P1(ii) -> P1(iii) genre/spoken-text -> (iv) beat_bounds -> (v) outro -> (vi)
  header<->scene -> (vii) placeholder -> (viii) provenance (each its own green pushed
  chunk); then Phase 2 (5 v4 banks, each a live GPU leg). Operator decisions defaulted
  (vetoable at the consuming chunk): WORDS_PER_BEAT=40 (soft; length recorded-not-gated),
  media_archive_v4 OWN drama_seeds, public_domain research_only BLOCKS publish.
Commits: c3a9d420

## 2026-07-17 evening -- HEAD 659ce5b2 (v2.0-alpha) [v4 campaign: full kibitz arc r1-r4 CONVERGED; final.md plan of record; NO code yet]

Did:
- Ran the LESSONS GATE (PRODUCTION_SPRINT_LESSONS incl. lesson 24 + PROD_BUG_LOG + Bug Bible)
  and mapped the live seams for the 5 lanes -> docs/2026-07-17-v4-campaign/LESSONS_GATE_BRIEF.md.
- Ran the FULL kibitz arc r1-r4 (operator routing: Codex @ gpt-5.6-sol + agy @ Gemini 3.1 Pro
  (High); Claude anchor+judge; $0 local). agy model corrected to "Gemini 3.1 Pro (High)" (3.5 Pro
  is not an installed slug). Every folded panel claim grounded CONFIRMED against real Windows files
  (5 grounding subagents). Artifacts: docs/2026-07-17-v4-campaign/{pass00,r1_plan,r2_plan,r3_plan,
  final}.md + r{1..4}_judgment.md + roundtable/r{1..4}_claude_anchor.md + kibitz-runs/2026-07-17-v4-campaign/.
- Converged design of record = final.md. Key grounded corrections vs the naive plan: a `_v4` id
  silently drops out of style pool / science floor / adaptation-cast (:4286) / sidecars -> each v4
  re-owns via validated scalar bank defaults (style_pool_class, require_science_floor,
  propagate_adaptation_cast); wiring mirrors v3 (shared legacy_many_pass_v4 for the 3 inline lanes,
  original_multi_pass_v4 + scifi_codex_circuit_v4 executable:true); genre banned_phrases does NOT
  gate spoken text today -> new boundary-aware spoken-text validator (writer-boundary repair +
  Phase-10 FreezeAssertionError scan); beat_bounds terminal = raise (no STORY_META output); outro
  missing name = bounded authored patch (no forced coordinate); text_for_tts already FIXED (dropped);
  weapons_smoking is an EXISTING lexicon-corroborated hard class (retain+author to pass, no new filter);
  A/B "strictly better" = POST-BUILD qualification (may be cloud), ship gate = green+live.
- Plan is Phase 0 (audit + PBUG-20260710-07 breadcrumb root-fix + verifies) -> Phase 1 (8 shared
  fixes, each green pushed chunk, canary per execution family) -> Phase 2 (5 v4 banks serialized,
  atomic per-bank chunk). 11-item VERIFY-AT-BUILD checklist in final.md.
Current step: v4 campaign -- ARC DONE; awaiting operator GO to start Phase 0 (first code).
Next: Phase 0 audit + breadcrumb root-hunt; then Phase 1 shared fixes; then the 5 v4 banks.
  Open operator decisions surfaced in final.md: WORDS_PER_BEAT constant, media_archive_v4 sidecar
  own-vs-share, whether public_domain research_only blocks publish.
Commits: none (docs only; campaign docs under gitignored docs/2026-07-17-v4-campaign/ + kibitz-runs/).

## 2026-07-17 afternoon -- HEAD 499386aa (v2.0-alpha) [roster trim -> 10 INDEPENDENT lanes + science_news family retired; ONE combined commit]

Did:
- Executed the operator roster trim as ONE combined commit @ 499386aa. Ripped
  the whole science_news family (v1/v2/v3), ALL _v2 lanes, orphan bases
  (public_domain_story/shakespeare/scifi_sonnet v1) + original_radio_v3 -> 10
  runnable lanes + custom. banks.json + pipelines.json + 14 pack dirs +
  story_rules + both canonical workflows (widget[23] -> scifi_fable2), all same
  commit. Roster now: media_archive(+_v3), original_radio, scifi_fable2(+_v3),
  scifi_codex(+_v3), public_domain_story_v3, shakespeare_v3, scifi_sonnet_v3.
- Independence (operator "real future-proof, no family dependency"): each kept
  lane resolves its OWN story_rules by EXACT id -- severed base_source_bank_id
  family-map in _otr_story_rules (resolve + coverage), the strict_v4 set, and
  the adaptation-cast classifier. Added 6 _v3 rules packs; renamed 3 orphan
  bases -> _v3; DEFAULT_RULES_ID -> scifi_fable2. Default repoint SPLIT:
  lane-selecting sites -> scifi_fable2; legacy-seam resolvers -> media_archive
  (kibitz r3 build-breaker catch: scifi_fable2 declares no legacy seams).
- Retired dead pipelines sonnet_archive_multipass (base) + original_multi_pass_v3
  and their runner-map / inline-set entries (bijection restored; _run_scifi_sonnet_lane
  kept -- the _v3 wrapper uses it).
- Method: /kibitz r3 (codex, grounded) on the rip PLAN first; ~150 stale
  roster/science-baseline tests repointed via 4 parallel subagents (disjoint
  file groups) + verified centrally. Obsolete science-lane / base-map /
  byte-identity tests removed (intent preserved by repointing to
  media_archive/original_radio where possible).
- Gates: full suite 7947 passed / 32 skipped / 1 xfailed; Bug Bible 17 passed;
  canonical 23 nodes / 57 links (widget value only); no BOM / no 0-byte;
  AST+JSON parse clean; HEAD == origin @ 499386aa.
Current step: v4 improvement campaign (post-rip) -- NOT started.
Next: roundtable R1-R2 (frontier panel + the new Kimi 3) then /kibitz R3-R4 to
  produce v4 for scifi_codex (improve on v1), shakespeare, public_domain,
  media_archive, original_radio; author the v4 lanes as INDEPENDENT banks.
  Parked (task 7): canonical root-fixes (scifi_codex P3 unstated-contract,
  scifi_fable2 SCENE_WORD_GROSS scene-gate, original_radio weapons/X-Y-placeholder/
  phantom-outro) + the shared pipeline-bug class the scoreboard flagged
  (speaker-attribution collapse, name-token splice, contract-vocab bleed,
  720-length knob).
Commits: 499386aa.

## 2026-07-17 morning -- HEAD f265c044 (v2.0-alpha) [variant scoreboard delivered; roster-trim decision -> rip in a fresh window]

Did:
- Ran the full story-only variant sweep (v2/v3 x {420,720}) on the harness. aion
  (OpenRouter) had a ~3-4am HTTP-502 outage that killed ~11 of the 720 legs;
  classified aion-drops vs content-fails and re-ran ONLY the aion drops (hardened
  tmp/_rerun_failed_720.ps1 to never blind-retry a content fail). Final: 420 rung
  COMPLETE; 720 rung 12/16 clean + 4 DISQUALIFIED content-fails (original_radio_v2
  weapons gate, scifi_codex_v2/v3 P3 contract, scifi_fable2_v3 SCENE_WORD_GROSS).
- Grading pipeline: tmp/_extract_for_grading.py + tmp/_assemble_matrix.py ->
  tmp/grading/matrix/*.txt (42/48 cells). ONE Fable pass -> the scoreboard at
  **docs/2026-07-17-variant-scoreboard.md**. fable2 v1 = flagship; order fable2 >>
  public_domain > original_radio > codex > shakespeare > media_archive > sonnet >
  science_news. BIG finding: most defects are PIPELINE bugs, not bank problems --
  speaker-attribution collapse (5/7 cases are _v2 cells), speaker-name splice into
  dialogue, phantom outro characters, contract-vocab bleed, and the 720-length knob
  barely steering. Code fixes that lift every bank.
- OPERATOR ROSTER-TRIM DECISION (task 8): KEEP 11 lanes -- fable2 v1+v3,
  public_domain v3, original_radio v1, shakespeare v3, science_news v3,
  scifi_sonnet v3, media_archive v1+v3, scifi_codex v1+v3. RIP 13 -- all 8 _v2 +
  public_domain v1 + original_radio v3 + shakespeare v1 + science_news v1 +
  scifi_sonnet v1. To be done as a CLEAN rip in a FRESH window (kibitz the plan
  first; canonical source_bank roster in the same commit; suite+Bible+push;
  precedent = codex56sol+gemini rip @ 3312aec7). Sonnet-on-v1 model-check killed
  (deck cleared); re-run it on the 11 kept lanes AFTER the rip.
- Earlier this session: sonnet decoration root-fix (2794e8a2) + story-only scoring
  harness (f265c044), both pushed, suite 7984 + Bible 17 green.
Current step: roster trim (task 8) in a fresh window.
Next: clean 13-lane rip -> Sonnet check on kept lanes -> parked canonical root-fixes (task 7).
Commits: 2794e8a2, f265c044 (pushed). Scoreboard doc uncommitted.

## 2026-07-16 evening -- HEAD f265c044 (v2.0-alpha) [sonnet decoration root-fix + story-only scoring harness; 32-leg variant sweep RUNNING]

Did:
- Root-fixed the scifi_sonnet 320w bake-off FAIL ("ORUM: spoken text contains
  decoration '('"): the spoken-purity contract (`_spoken_error`) was enforced
  ONLY at the terminal `validate_spoken_text_and_lock` raise, so a stray
  parenthetical killed the episode with no bounded repair. Wired it into the
  P2a/P2b (CitedLineV4) + P5 (RewriteResultV4) typed-repair ladder so the model
  fixes its own line (LLM-first); terminal gate stays the deterministic last
  word. Live: scifi_sonnet 320w RESULT SUCCESS + obs asset (recovery_session,
  508w/13 lines). Commit 2794e8a2. Applies to all 3 sonnet versions (shared runner).
- Built the story-only scoring harness (operator: "splice the canonical, use the
  latest"): `OTR_LedgerFreezeCascade.OUTPUT_NODE=True` + `otr_canonical_api_run.py`
  opt-in `--workflow` (default = canonical WITH its path assertion) + wrapper
  `-Workflow` passthrough + `scripts/build_story_only.py` ->
  `workflows/otr_story_only.json` (validator->writer->freeze, 3 nodes / 6 links).
  Skips the ~30 min TTS/video tail; each leg ~12-20 min, produces the frozen
  ledger/transcript we grade from (video carries no cross-bank grading signal).
  Live 30w leg RESULT SUCCESS in 10:37, freeze terminal executes. Commit f265c044.
- Suite 7984 passed / 32 skipped / 1 xfailed + Bible 17 passed after BOTH commits.
- LAUNCHED the 32-leg story-only variant sweep (16 `_v2`/`_v3` lanes x {420,720},
  aion-3.0-mini + Mistral-Nemo) for the v1/v2/v3 comparison. Receipts
  `tmp/_storysweep_receipts.csv`; ~9-12h; hourly scheduled check-in task
  "otr-story-sweep-checkin". Base v1 420/720 transcripts reused from existing
  ledgers (no re-render). 4 full-render `_v2` @420 legs already banked
  (media_archive/original_radio/public_domain_story/science_news).
Current step: 32-leg story-only variant sweep RUNNING (render window).
Next: as legs land, root-fix any failing variant lane per THE LAW (sonnet
  decoration already fixed; watch P3 AuditVerdictV4 / P6 attestation / codex
  premise-cap), then build the 8x3x3 scoring report (v1/v2/v3 per bank at
  420+720) + whittle to the top-8 keepers (best version per bank).
Commits: 2794e8a2 (sonnet fix), f265c044 (story-only harness).

## 2026-07-16 -- HEAD f58ed6e6 (v2.0-alpha) [Qwen3-8B GGUF writer row PROMOTED -- orthogonal model-roster task]

Did (GGUF-row bake-off per `docs/2026-07-16-gguf-row-registry.md`; NOT a forward-order step):
- 3-leg live Qwen3-8B-Q4_K_M bake-off, both writer slots Qwen, ctx=8192 on CUDA:
  3x RESULT SUCCESS + obs asset; peak ~11.8 GB (<14.5); KV 5.60 GB @ 8192 =
  0.70/1k; no silent fallback. Row PROMOTED UNKNOWN->PASS (pinned
  size=5027784512 / sha256=120307ba... / kv=0.70). First GGUF build roster is
  now gemma-4-12b + Qwen3-8B (14B deferred).
- Leg 1 root-fixed 7 Mistral-era assumptions that break a reasoning model:
  `_fetch_science_news` signature; `/no_think` on every gguf call (non-structured
  truncation + json_object `{}`); announcer stop-hygiene + robust dangling-`<think>`
  strip; freeze/shot `load_config` threading (live: a VRAM-eviction cache-miss
  reloaded Qwen NOT gemma); shot-lock re-raise (no silent template);
  `PreAuditReport` null->default (a clean audit's null reason was forcing a
  spurious needs_full_rerun). `/kibitz` (codex) on the `<think>` class per the
  two-strikes law -- it converged + flagged the load_config gap before it cost a leg.
- Full suite **7967** + Bug Bible green. Fail-loud rip honored throughout (operator
  "no local-LM fallbacks"). Docs (gitignored): `docs/2026-07-16-qwen-thinking/`.
Current step: UNCHANGED forward order -- Source-bank bake-off (render window).
Next (operator directive 2026-07-16): complete the **8-bank x 3-leg** bake-off --
  run the remaining legs, ROOT-FIX any failing lane (THE LAW / no-fallback /
  LLM-first: model/prompt/budget-contract fix or explicit lane disqualification,
  NEVER a canned line or blind retry bump), then produce the final 8x3 per-bank
  verdicts + World Cup scoreboard (GO_FORWARD "Then, in order" item 1).
Commits: ee0b2318 (7 fixes), f58ed6e6 (row pinned).

## 2026-07-15 late night -- HEAD 4cd36761 (v2.0-alpha) [plan-stack baseline: every go-forward doc re-grounded]

Did (docs-only session -- no code, no suite run needed; phase C render untouched):
- Read-only fan-out audit (3 grounded agents) of the full plan stack vs HEAD
  4cd36761. Status headers folded into 10 docs -- verdicts: dynamic_story
  CURRENT (rev-5 stands; wiring snapshot still matches live canonical);
  lean-mean-rip NEEDS a bounded re-verify before execution (kill lists + W5
  positional obligation re-verified LIVE and intact; SW-1/SW-3 re-surveys, W6
  keep-list adds, W7 tombstone re-triage, R-7 re-grep -- see its header);
  randomizer-r2 STALE (lane-specs authority absorbed by user-source-lanes;
  24-lane roster; factory-wrapped _v3 runners); vibe-coder-r2 + codex56sol
  telemetry + fable2-s2-QA-r2 + source-banks-v2 SUPERSEDED; llm-first STALE
  with a LIVE remainder (`repair_cliche_span` still rewrites spoken lines +
  `cliche_replacements` in all 8 story_rules JSONs -- X1-X4 queued as a
  quick-win); announcer-framing defect fully OPEN (fix surface untouched in
  code; original_radio_v2 seam is prior art); CLOUD_ENGINE_COVERAGE PARKED
  (babysit harness gone at HEAD; node-83 wiring changed @ 6899d940).
- GO_FORWARD_PLAN lower half REWRITTEN (2026-07-12 sprint table retired):
  telemetry + PBUG-17 items retired (target lane ripped @ 3312aec7), item 8
  re-pointed to user-source-lanes-architecture (~21-31 d, gated on sec-16
  ratification + r5), old item-10 bakeoff removed (superseded by the real
  campaigns), verdict IMPROVE passes + cliche excision + announcer contract +
  ENGINE_MATRIX folded into a quick-wins block, lean-mean added as big block 1
  (order vs extensibility = operator call, recommendation lean-mean first).
  Campaign block + THE LAW + current step preserved as written.
- PROD_BUG_LOG hygiene: duplicate id PBUG-20260713-10 resolved (the
  P1-overlong-question entry renumbered to -21; -10 stays with the P9-audit
  entry). BUG_BIBLE.yaml carries two `legacy_id: -10` rows (~:4357/:4379) --
  reconcile at next fan-out. PBUG-20260712-17 marked SUPERSEDED (its lane was
  ripped; diagnostic-gap class carried by the context/cap quick-win).
- Committed the stranded untracked docs (720 verdict, the 07-13 rip-gates set,
  codex handoff, bakeoff observations, cue-ledger prompt) -- never-lose-work.
- Operator mid-session directives, executed: (1) "nuke it" -> the
  otr-build-tracker artifact is RETIRED (tombstone page pointing at
  HANDOFF_LOG + GO_FORWARD; it had been stale since 06-29). (2) GO_FORWARD
  leaned to TRULY forward-only: campaign shipped-lists, THE LAW done-narrative
  + live-proof table, and the per-lane ladder section stripped (this log +
  PROD_BUG_LOG own them); the "lost anchor" doctrine moved to
  PRODUCTION_SPRINT_LESSONS.md as lesson 24. (3) kibitz r4 confirm pass run on
  the baseline GFP -- panel = codex gpt-5.6-sol (verified via
  codex_model_selected.txt) + agy "Gemini 3.5 Flash (High)", Claude anchor +
  judge; anchor caught 2 must-fixes itself (quick-win-1 reverify vehicle
  overstated: phase C runs only _v2/_v3 lanes, so the base scifi_codex 120w
  reverify needs its own leg or explicit operator acceptance; quick-wins range
  arithmetic understated: ~6-13 d, combined ~33-55 d) -- both folded; panel
  survivors folded per kibitz-runs/2026-07-15-gfp-baseline/r4/final.md.
- ROADMAP swept for the parallel lane; GO_FORWARD gains a Window-packing
  section (RENDER + CODER A-G + PLANNER, one-line otr-handoff kickoffs, credit
  rules) and the lean-mean/extensibility order DISSOLVES on ROADMAP's ratified
  edges: front waves (W0..C1-C5) before extensibility, SW tail (SW1-SW3, C6,
  C7, W8) after extensibility/randomizer/dynamic_story. Combined range now
  ~45-71 coder-days through the tail. Live dashboard artifact rebuilt
  (otr-plan-dashboard: GFP queue + HANDOFF current step + phase-C receipts via
  Desktop Commander), replacing the retired tracker.
- Live observation from receipts (23:19): `scifi_codex_v2` 30w local FAILED at
  P3 -- `RadioScoreDraftV4` ValidationError after 2 attempts -- the exact
  PBUG-20260712-22..25 transport seam awaiting reverify. Campaign window owns
  triage; quick-win 1's reverify just got more interesting.
Current step: UNCHANGED -- phase C 30w smoke sweep (the render window owns it;
  monitor tmp/_phaseC_receipts.csv).
Next: campaign window RE-READS GO_FORWARD before its wrap-up edit (rewritten,
  then leaned, 2026-07-15 late night). Coder queue order per the re-grounded
  queue. NO code lands while phase C is mid-sweep (uniform-code confound).
Commits: b94f0c70 (baseline), 0ed44a3b (lean + kibitz fold), + the
  packing/parallel-lane commit (docs only).

## 2026-07-15 evening -- HEAD b57be02b (v2.0-alpha) [three-phase bake-off campaign: A PASSED, B F2 PROVEN, C smokes LAUNCHED]

Did:
- Confirmed live tip = b57be02b (HEAD==origin), tracked tree clean (only tmp/ +
  docs scratch dirty). Fixed the doc-lag: GO_FORWARD + prior top log entry said
  c28af5f4; live tip is the b57be02b docs-handoff commit atop it.
- PHASE A (Fable final gate on the 8 _v3 promotions + source-snapshot B7/B8):
  PASSED, no build-breakers, nothing folded, tree stays clean. general-purpose
  grounded review = NO build-breakers (all 5 checks file:line grounded); my anchor
  independently confirmed the KeyError class (5 _v3 pipelines defined at
  pipelines.json 566/665/715/824/966 + wired in _RUNNER_BY_PIPELINE/_INLINE_V3_
  PIPELINES; fable2 gate catches _v3; base_source_bank_id maps variants; snapshot
  strict-by-default). Fable UNAVAILABLE (out of usage credits -- failed loud);
  codex CLI unhealthy today (17-min hang + stalled relaunch, killed after ~50min).
  Substitute gate = the two grounded reviews + the live renders themselves.
- PHASE B (F2 live-replay proof): DONE. Captured a real source snapshot for
  original_radio (local spark draw, seeded OTR_ORIGINAL_SEED, sha ed1c941f8e99) ->
  tmp/_phaseB_snapshot_manifest.json; strict loader self-verified for base/_v2/_v3.
  Ran the triplet at 30w local under OTR_C7=1 + manifest. Acceptance met on all 3:
  server log shows source-snapshot REPLAY sha=ed1c941f8e99 + ledger meta
  cast_seed_source == "OTR_CAST_SEED override". RESULT: base GREEN (52.9MB obs
  asset); _v2 AND _v3 both content-FAILED IDENTICALLY on the deterministic
  weapons_smoking gate ("cocking his revolver") -- a clean F2 demonstration that the
  PACK is the only causal variable (same frozen source+seeds, base seam -> clean
  story, v2/v3 seam -> identical weapon content). Lawful under THE LAW (deterministic
  gate). Finding: original_radio _v2/_v3 seam steers to weapons content vs base.
- PHASE C (160-leg bake-off = 16 _v2/_v3 lanes x 5 tiers x 2 profiles): 30w smoke
  sweep (32 legs) LAUNCHED in production mode (no C7/manifest -- verified first leg
  science_news_v2 sources live). Runner tmp/_phaseC_sweep.ps1 (tier-param), receipts
  tmp/_phaseC_receipts.csv, progress tmp/_phaseC_progress.txt, per-leg .done markers.
  ~9 min/30w leg -> smokes ~5h; full 160 legs is a multi-day autonomous run.
- Harness note (follow-up): the launcher's [launch] C7/manifest echoes go to the
  hidden Start-Process console + python's `> %1` truncates, so they do NOT reach the
  server log; the writer's own REPLAY line + cast_seed_source are the ground-truth
  proofs. A one-line launcher/wrapper fix (append, echo the two vars into %1) would
  satisfy the literal-echo acceptance.
Current step: Phase C 30w smoke sweep running (autonomous). After smokes gate:
  120 -> 320 -> 420 -> 720, both profiles; then durable report + World Cup scoreboard.
Next: monitor tmp/_phaseC_receipts.csv; when smokes complete, launch
  `tmp\_phaseC_sweep.ps1 -Tiers 120,320,420,720 -Label full`; content-FAILs
  (weapons/profanity) are RECORDED with reason, never re-rolled to force green.
Commits: docs only (no code fold in Phase A). tmp/ sweep scripts are scratch.

## 2026-07-15 night -- HEAD c28af5f4 (v2.0-alpha) [bank-bakeoff: kibitz r4 CONVERGED + hardened]

Did:
- Ran kibitz r4 convergence on the as-built bake-off (chunks 1/2/4 + B7/B8).
  Panel = Codex @ gpt-5.5 high (rc=0) + Claude anchor; Antigravity FAILED (agy
  rc=1, the known Cowork flake). The skills-cache kibitz.py ignored
  KIBITZ_CODEX_MODEL=gpt-5.6-sol and ran gpt-5.5 (documented drift) -- fine for r4.
- Grounded Codex's review. CONFIRMED one real footgun (MUST-FIX 1): the snapshot
  loader returned None when a manifest was configured but the selected base was
  absent -> silent live sourcing, invalidating the F2 control. FOLDED: source-
  snapshot is now STRICT by default (configured-manifest miss RAISES; opt-in
  "allow_partial": true restores freeze-some/source-rest-live). REJECTED Codex's
  "unconditional raise" (breaks the normal triplet run). Codex MUST-FIX 2 (C7
  proof) -> a LOUD C7-replay warning in code + render-window acceptance criteria
  in GO_FORWARD. Codex OPTIONAL (advisory-key wording) -> doc-only, no code.
- Gates: full suite 7907 passed / 31 skipped / 1 xfailed (+3 r4 tests: strict
  raise, allow_partial, C7 warn/quiet); Bug Bible 17 passed; no BOM; canonical
  delta = none; HEAD==origin. Artifacts under kibitz-runs/2026-07-15-bank-
  bakeoff-r4/r4/ (claude_anchor, codex, final) + docs/.../kibitz/r4-convergence-plan.md.
Current step: Fable final gate (HELD for operator go) + the live replay triplet
  proof (render window).
Next: operator decides on the Fable gate; then the F2 live replay proof under C7.
Commits: 031851ce (B7/B8), 57393879 (docs), c28af5f4 (r4 strict fold)

## 2026-07-15 night -- HEAD 031851ce (v2.0-alpha) [bank-bakeoff: source-snapshot B7/B8 SHIPPED]

Did:
- Built the bake-off frozen-source replay layer (r3 rulings B7/B8). New stdlib
  leaf `nodes/_otr_source_snapshot.py`: a process-wide manifest (env
  `OTR_SOURCE_SNAPSHOT_MANIFEST`) keyed by BASE bank, so one frozen source serves
  the base/_v2/_v3 triplet. `load_snapshot_for_bank` validates the envelope
  (base match via `base_source_bank_id`, seven-key payload presence, non-empty
  seed_source, optional payload_sha256 receipt) and REJECTS base-mismatch /
  malformed / altered-payload loud; returns None when no manifest is configured.
- Wired it into `OTR_LedgerScriptWriter._resolve_inputs` as the FIRST source
  branch, immediately after bank resolution and BEFORE entropy/custom/fetch, so a
  replay bypasses RSS/random; the replayed source_meta carries spark_atoms
  (original) / cast_hints (adaptation) so no downstream owner is starved.
- B8 seed control in `scripts/_otr_soak_server_launch.cmd`: pin
  `OTR_FABLE2_SEED=42` alongside CAST/STYLE under C7 (cleared otherwise) + an
  auditable manifest echo. Dropped an mtime-keyed cache (Windows coarse-mtime
  stale-read hazard) -- the manifest is re-read per episode.
- Gates: full suite 7904 passed / 31 skipped / 1 xfailed (+20 new); Bug Bible 17
  passed; no BOM; py_compile clean; canonical delta = none; dry registry-load 24
  runnable / 25 visible + round-trip 23 nodes/57 links. Pushed; HEAD==origin.
Current step: kibitz r4 convergence + Fable final gate on the v3 promotions + the
  source-snapshot layer (see GO_FORWARD NEXT).
Next: run kibitz r4 (local Codex+Antigravity) then the Fable final gate; then the
  live replay triplet proof in the render window.
Commits: 031851ce

## 2026-07-15 late -- HEAD c32d4c04 (v2.0-alpha) [bank-bakeoff build: chunk 4 SHIPPED + kibitz r2]

Did:
- Ran kibitz r2 on the chunk-4 per-lane matrix (Codex gpt-5.5 high OK; agy lane
  failed -- the known Cowork flake; codex + Claude anchor was the reliable panel).
  Codex DISSOLVED the main risk: I had MISREAD the assemble timing -- codex/sonnet
  DO assemble the ledger IN-runner (led.set_* inside _assemble_ledger), so a v3
  wrapper reads led.data["lines"] uniformly. It also caught the fable2 early
  word-budget gate hard-matching only "fable2_multipass" (a _v3 id would bypass it),
  the runner-map bijection test, and simplified 3 runner files -> ONE wrapper
  factory. Artifacts: docs/.../kibitz/r2-anchor.md + kibitz-runs/2026-07-15-chunk4-
  v3-lanes/r2/{codex.md,final.md}.
- CHUNK 4 SHIPPED @ c32d4c04: pipelines.json +5 clone pipelines; banks.json +8 _v3
  rows (before custom; change default_story_model + default_story_pipeline); 8 v3
  packs (copy v2 + header triple). Writer: run_v3_advisory (deterministic,
  advisory-only, reads assembled ledger, stamps meta["<bank>_v3_advisory"],
  try/except -> never raises, never mutates rows); _make_v3_runner wrapper factory +
  3 sci-fi v3 registrations; _INLINE_V3_PIPELINES + the 2 inline v3 ids in
  _LEGACY_INLINE_PIPELINES; one post-Phase-0 (after :6470 led.save) inline advisory
  hook; fable2 early-gate now family-matches ("fable2_multipass" or "..._v3");
  tooltip de-staled. TestChunk4V3Rows + 2 advisory regressions; pinned tuples
  updated; bijection test validates the wiring.
- Gates: suite 7884 passed / 31 skipped / 1 xfailed; Bug Bible 17 passed; canonical
  delta = none (git diff --exit-code otr_canonical.json clean); no BOM; py_compile.
Current step: source-snapshot injection (B7/B8) -- see GO_FORWARD NEXT.
Next: build the snapshot-envelope load in _resolve_inputs + OTR_C7/OTR_FABLE2_SEED
  controls; then kibitz r4 + Fable final gate + final registry/canonical verify.
Commits: c32d4c04

## 2026-07-15 evening -- HEAD 19872aa6 (v2.0-alpha) [bank-bakeoff build: chunk 2 SHIPPED]

Did:
- CHUNK 2 SHIPPED @ 19872aa6 (pushed, HEAD==origin, no BOM, py_compile clean).
  8 `<bank>_v2` rows inserted before custom_source_bank (mirror base, only
  default_story_model changed; byte-identical banks.json round-trip) + 8 v2 packs
  (base prompt_stages copied, Sec-D target seams edited per pass01 Sec D with
  Section-19 L-1/L-2/L-5/L-6/L-8; header triple = path coords, base pipeline kept).
- B1 owner_bank threading: scifi codex/sonnet/fable2 stamp owner_bank=
  source_bank_row.source_bank_id (never base-mapped); `_assemble` gained an
  owner_bank param. Confirmed the writer stamps meta.source_bank to the SELECTED id
  (:3758) BEFORE runner dispatch (:3853), so scifi_*_v2 pass the authorship gate.
- B5 pinned tuples updated (test_fable2_registry tail + full-order); new
  TestChunk2V2Rows (16 runnable / 17 visible + per-v2 own-pack/base-pipeline).
  test_fable2_assembly direct _assemble calls pass owner_bank.
- F8 resolved on first pass: "EDNA FROST've" is model output, NOT the shared
  _otr_ledger_scrub._normalize_whitespace_and_quotes (which only normalizes
  quotes/whitespace) -> the ALL-CAPS-no-contraction rule lives in media_archive_v2's
  line_composer/exchange seams, not a baseline fix.
- Gates: full suite 7873 passed / 31 skipped / 1 xfailed; Bug Bible 17 passed.
- Grounded CHUNK 4 fully (dispatch/_LEGACY_INLINE_PIPELINES/_resolve_lane_runner/
  telemetry/inline body/authorship) and wrote the per-lane v3 matrix into
  GO_FORWARD_PLAN CURRENT STEP.
Current step: CHUNK 4 (8 v3 lanes: sci-fi own-runner + adaptation/original inline).
Next: build chunk 4 per the GO_FORWARD per-lane matrix; two-strikes -> /kibitz.
Commits: 19872aa6

## 2026-07-15 13:15 PDT -- HEAD 9e0fdf9e (v2.0-alpha) [bank-bakeoff build: chunk 1 + r3]

Did:
- Started the Bank-Improvement Bake-off BUILD (24 rows = 8 base + 8 _v2 + 8 _v3 in
  the one existing source_bank dropdown; zero canonical-JSON diff). Grounded the
  wiring against live HEAD (the tail refactor WriterTailContext/_run_writer_tail/
  TailFinalizer has landed since the r2 anchor -- r2-wiring-anchor.md is stale).
- Ran kibitz r3 (Codex @ gpt-5.6-sol + Antigravity/Gemini-3.5-Flash-High). Judgment:
  docs/2026-07-15-bank-improvement-bakeoff/kibitz/r3-final.md (that folder is
  gitignored -- read from disk). It caught 3 build-breakers.
- CHUNK 1 SHIPPED @ 9e0fdf9e: nodes/_otr_bank_variants.py (base_source_bank_id) +
  5 family-behaviour sites + tests/test_bank_variants.py (32). Suite 7864 green;
  Bug Bible 17 green. Pushed; HEAD==origin.
Key r3 rulings: B1 owner_bank uses the ACTUAL variant id (never base-mapped);
  B2 adaptation v3 stays INLINE not own-runner + D.2 extraction CUT; B5 variant rows
  insert BEFORE custom_source_bank and update the pinned tuples same chunk.
Current step: bakeoff chunk 2 (8 v2 rows + packs + owner_bank fix + pinned-test updates).
Next: build chunk 2 per r3-final.md Sec C.2.
Commits: 9e0fdf9e

## 2026-07-11 -- HEAD 6899d940 (v2.0-alpha) [720-bakeoff C3 coder window]

Did:
- C3 SHIPPED @ 6899d940 (atomic code + canonical JSON + tests): music cue
  manifest + third-bus wiring, per FINAL_HARDENED_PLAN.md. NEW
  nodes/_otr_cue_manifest.py (manifest_version 1; shared parse/fail-loud
  validate; keyed cue_id+batch_index; contiguous-batch + dup + placement gates).
  Node 83 (StableAudioTheme) now emits ONE padded cue batch + manifest (4-tuple
  cue_audio_clips/cue_manifest_json/render_log/done): renders each
  ledger.music[] row (fable2) OR synthesizes opening/closing/interstitial
  (legacy, byte-parity slot seeds); writes each cue wav to the episode audio
  dir; placement mapping so inter_NN never KeyErrors compose_music_prompt.
  SceneSequencer + EpisodeAssembler take music_cue_audio/manifest as a THIRD
  bus (own index, never C2's two-bus check); opening/closing sliced from the
  batch by sample_count (direct slice, no silence-trim) + resampled;
  interstitials inserted inline by anchor_line_id (fable2 only; legacy stays
  unconsumed = pre-C3 parity); MF-H scene_audio->master_mix shift extended to
  music rows.
- Canonical JSON same commit: links 241/242/243 out, 280-283 in (node 83 ->
  nodes 3/7 fanout by name); node-7 opening/closing + node-12 closing_audio kept
  DECLARED/unlinked (BUG-LOCAL-097 slot-drift guard); last_link_id 279 -> 283.
  OTR_WorkflowValidator OK, widget_vector_drift=0, JSON round-trip + link-ref +
  input-name + widgets_values-count audits clean.
- Tests: NEW tests/test_cue_manifest.py (schema/dup/slice/byte-parity); rewrote
  test_stable_audio_theme (4-tuple + fable2 lane) + test_full_workflow_v2_audio_
  wiring (new fanout, 241/242/243 gone); fixed 2 constant-pin regressions caught
  by the known-fail guard (test_audio_determinism_wrap 4-tuple,
  test_google_video_sfx_workflow last_link_id 283).
- Suite 7510/31/1 + Bug Bible 17/7/3 green. HEAD==origin, no BOM/0-byte, AST OK.
- LIVE PROOF (LTX lane, headless :8000): 30w = SUCCESS (frozen_circuitry 62.9
  MB, audio_byte_identical OK, 7 beats covered no gaps); 720w all-visual =
  SUCCESS (ticking_lockdown 123.7 MB, audio_byte_identical OK, 18 beats incl. 2
  music_inter covered, budget OK no gaps, 18:50 render). Byte-parity held on
  both.
Current step: C3 done + live-proofed. NEXT = C4a/C4b (S2 full loop) in an Opus
window. Post-C3 follow-up queued: richer per-cue music-still prompting (separate
chunk, image/video director prompt derivation).
Next: C4a/C4b in an Opus window (do NOT start here).
Commits: 6899d940 (code+JSON+tests). Docs refresh = this commit's follow-up.

## 2026-07-11 -- HEAD 2f335c28 (v2.0-alpha) [720-bakeoff C1/C2 coder window]

Did:
- C1 SHIPPED @ 9949bb6e: durable-field identity in production_ledger --
  _row_identity gates the disk merge so durable render fields (wav/timing)
  copy forward ONLY on unchanged content identity (lines=sha of text,
  music=cue_spec_sha256, clips=render-spec); empty-source -> no gate (skip/
  clear preserves durable per the ownership contract). set_music now carries
  anchor_line_id/placement/target_duration_s + stamps cue_spec_sha256. 5 new
  tests; golden fable2 fixture regenerated. Suite 7468/31/1 + Bible green.
- C2 SHIPPED @ 2f335c28: text_for_tts delivery routing. _otr_readiness
  stamps text_for_tts + source sha + receipt on fable2 voiced lines (canonical
  untouched -- restores the pronunciation the P0 fold switched off). NEW
  _otr_text_delivery resolver (LEGACY passthrough = byte-identical spine;
  CONTENT_OWNED = verified stamp, absent/stale = terminal before gen). Voice
  node routes prep/vector/hash through it. scene_sequencer two-bus surplus+
  shortfall terminal check. 26 new tests incl. science_news byte-parity fixture.
  Suite 7494/31/1 + Bible green.
- C3 wiring kibitz'd (r3, Codex + Claude Code grounded; Antigravity timed out).
  HARDENED spec = docs/2026-07-11-c3-cue-manifest-wiring/FINAL_HARDENED_PLAN.md.
  Surfaced real build-breakers before touching the canonical JSON: legacy
  ledger.music[] is empty (node 83 must synthesize legacy cues; inter_NN
  KeyErrors compose_music_prompt), sentinel lines have no cue_id (use C1's
  anchor_line_id), node-7 input deletion = widget-slot drift (keep declared),
  music must be a 3rd bus, slice by sample_count (no silence-trim) + resample.
Current step: 720-bakeoff C3 (cue manifest + canonical workflow wiring) --
CODE-READY per the hardened spec; canonical-JSON rewire, one atomic commit.
Next: build C3 in a fresh window from FINAL_HARDENED_PLAN.md (re-derive live
literals per the VERIFY-AT-BUILD list); STOP after C3 green+pushed.
Commits: 9949bb6e (C1), 2f335c28 (C2) -- both pushed. C3 docs this commit.

## 2026-07-10 ~14:20 -- HEAD af378aad (v2.0-alpha) [scifi_fable2 S1b coder window, QA fold]

Did:
- External QA analysis (docs/2026-07-10-fable2-s1b-QA-ANALYSIS.md) folded: it
  OVERTURNED the 5C-mutator theory -- real chain = doctor 'skip' clears text ->
  Ledger.save() stale-disk merge resurrects old text -> Phase 10 gap. P0 fixes
  shipped @ af378aad: ownership-aware merge (_MERGE_OWNED_ROW_FIELDS), doctor
  skip stamps tts_skip_reason, 5B/5C lane capability gate
  (_legacy_line_compose_applicable; fable2 pack has no line_composer_system).
  QA regression file tests/test_ledger_merge_ownership.py. Suite 7451/31/1.
- LTX MEDIA PATH GREEN: "The Butterfly's Gambit" published to obs (1787s,
  41.8 MB) -- character lane ltx_audio_in + stills; capability gate fired live;
  freeze passed; canonical no-diff.
Current step: fable2 S2 (full loop, 350w) with the QA runway items folded in:
proof-provenance (doctor/Phase-7 rewrite after proof seal -> text_for_tts),
inter-scene music wiring, caption/credits sentinel alias, HuMo stale guard,
per-scene band allocation (all pinned w/ file:line in the QA analysis doc).
Next: S2 in a fresh coder window; operator eyeball on both fable2 episodes.
Commits: af378aad (+ this docs commit) -- pushed.

## 2026-07-10 ~13:15 -- HEAD 8e3d9228 (v2.0-alpha) [scifi_fable2 S1b coder window]

Did:
- S1b SHIPPED: runner + dispatch + registry flips + 80+ tests @ a24b75c4;
  25-roll live-smoke hardening (kibitz r2/r3/r4 + sonnet/opus fan-out per the
  new kibitz-every-failure directive) @ ff4c226d + 8e3d9228. FIRST GREEN
  EPISODE: "Einstein's Echo" in obs (570s); canonical no-diff + validator OK.
- ROOT-CAUSE fix: reviewer role_mismatch flipped sentinel announcer rows to
  character breadcrumb-lessly (sonnet+opus converged on reviewer.py role
  branch); symmetric guard + breadcrumb + regression tests shipped.
- OPEN BLOCKER: cascade 5C-reroll failure path stamps skip=True on target
  rows when fable2's pack (correctly) lacks line_composer_system -> Phase 10
  needs_full_rerun. LTX media roll (stills+ltx_audio_in via _tmp probe,
  16gb_full + character_visual override) got 25 min deep; blocked on this.
- External-QA brief written per operator: docs/2026-07-10-fable2-s1b-QA-
  PROBLEM-STATEMENT.md (big problems + full downstream landmine audit ask).
Current step: resolve the skip-mutator blocker (QA brief) -> green LTX-lane
fable2 roll -> then fable2 S2 (full loop, 350w).
Next: operator runs the QA brief through the external analyst; fold findings.
Commits: a24b75c4, ff4c226d, 8e3d9228 (+ this docs commit) -- all pushed.

## 2026-07-10 ~08:00 -- HEAD c932880f (v2.0-alpha) [scifi_fable2 coder window]

Did:
- scifi_fable2 S1a SHIPPED: writer tail (J.5 -> M save) extracted into
  `_run_writer_tail(ctx)` + 17-field WriterTailContext (doc s11 pins);
  moved body verified character-identical vs pre-extraction modulo the 2
  pinned gates (title override precedence + run_story_spine gate, s14/8);
  late _OTRC/_PL imports followed the tail. 11 new tests
  (test_fable2_tail_context.py: ctx contract, no-closure, delegation,
  same-run byte identity, spine gate both ways, title precedence x3,
  refine stash x2). 3 AST pin modules updated to follow the move
  (story_brief_c5a2 fixture, announcer title-regen pin, title scratchpad).
  ROOT-CAUSE find: my byte-identity test leaked production_ledger._CURRENT
  (singleton) -> broke lfc C4 tests downstream; autouse save/restore
  fixture added. Commit `948c5a0a`.
- ONE legacy science_news 30w live smoke on the extracted tail: RESULT
  SUCCESS 555s (baseline band), "Etna's Secret" published to obs (60.7 MB,
  Test-Path confirmed); J.5 regen fired live (title_source=
  llm_post_composition). Ledger scrubbed (paths anonymized, article text
  truncated, all keys/rows kept) -> tests/fixtures/fable2/
  legacy_reference_ledger.json + README. Commit `c932880f`.
- Gates: suite 7332/31/1 + Bug Bible 17/7/3 green at 948c5a0a (+ post-
  fixture full-suite re-run green); BOM/AST/0-byte/HEAD==origin verified.
  Also committed a leftover ENGINE_MATRIX docs hunk from the prior
  session (`5f5820a7`).
Current step: scifi_fable2 S1b -- spine, live (runner P0/P1-one-pitch/
P2b/P3/P6/P7 + P8 audit-only; flip runnable+executable SAME change; doc
s13 S1b test set; 30w live smoke; validator no-diff record).
Next: S1b in a coder window (doc sections 5/8/11/13; re-pin splice lines
in the S1b commit).
Commits: 5f5820a7, 948c5a0a, c932880f (+ this docs commit) -- all pushed.

## 2026-07-10 ~06:45 -- HEAD d7379920 (v2.0-alpha) [scifi_fable2 coder window]

Did:
- scifi_fable2 S0 SHIPPED (all inert, doc = 2026-07-10-scifi-fable2-architecture.md):
  banks.json row before custom_source_bank + fable2_multipass pipeline row
  (registry-legal slots); 9-seam pack scifi_fable2_v1.json (FORMAT block
  byte-identical script/revision); frame_deck.json 14 cards + 6 stances +
  sidecar registration; detection-only story_rules (empty replacements);
  _otr_fable2_markup.py parser (full defect enum, collected defects, split
  word counters, per-constituent lines); 66 new tests incl. rss-not-spark,
  slot-enum rejection, deck lint, science_news pinned row. Doc s14 pins
  1/5/10 resolved in-doc. science_news untouched; NO workflow diff.
- COMMIT NOTE: my staged S0 files were swept into the freeze-cascade
  window's commit d7379920 mid-session (one bundled commit, pushed). Content
  verified file-by-file; full suite re-certified at that HEAD.
- Gates at HEAD: suite 7321 passed/31 skipped/1 xfailed; Bug Bible 17/7/3;
  BOM/AST/JSON verify clean; HEAD == origin.
Current step: scifi_fable2 S1a -- tail extraction ALONE (writer
_run_writer_tail(ctx) + WriterTailContext, byte-identity pin
test_fable2_tail_context.py, ONE legacy science live smoke, then scrub the
ledger into tests/fixtures/fable2/legacy_reference_ledger.json). Nothing
fable2-visible ships in S1a.
Next: fresh coder window claims the slot, reads doc sections 11+13+14, does
S1a only, then S1b (spine + runnable flip same change).
Commits: none under my own SHA (work rode d7379920); this docs commit.

## 2026-07-10 ~02:45 -- HEAD 636d78cf (v2.0-alpha) [original_radio window]

Did (operator overnight directive: "run two more 420w, analyze, optimize
the original path, prompts not py"):
- 420w night batch, 4 rolls total. PUBLISHED: "Ashes of the Pawn"
  (otr\obs\signal_lost_ashes_of_the_pawn_20260710_014548_..._final.mp4,
  18 min e2e). Roll A died at QA: the confirm judge "proved"
  news_source_framing by quoting the CLEAN intro verbatim -- fixed at
  root (3d32b265: news_source_framing + machine_attribution join
  weapons as lexicon-only kill classes; suite 7153 green then). Roll C
  died HONESTLY: writer armed a climax ("holding his revolver") --
  correct lexicon kill. Roll D died at concept: empty cast name x2
  (archetype "The Stenographer").
- ANALYSIS (leg 1): 239/420 words (thin brief -> thin outline);
  key_terms landed 1/5 (story diverged from concept); intro
  ventriloquized a character quote; ZERO quote-wrapped lines and ZERO
  stage directions at 420w (30w observations did not recur); no audible
  name drift (visual portrait prompt invented "Ferrywoman Edith" --
  eyeball item); outro button landed well.
- OPTIMIZED (prompt/data only, 636d78cf, pack JSON): concept demands
  non-empty CAPS personal names w/ example; script_brief demands
  episode-shape (opening/two turns/closing image) + key_term weaving +
  no-arms menace rule; both intro seams forbid quoting characters.
- NOT re-verified live: the portability coder window claimed the repo
  mid-session (S1 in flight, 9 py files dirty + llm_policy.py
  untracked); full suite red from ITS tree, my lane tests 42/42 green.
  NEXT lane action = one 420w verification roll AFTER the portability
  window settles, then eyeball all published episodes.
Current step: original_radio pre-ship -- operator eyeball (now 2
episodes in obs: page_in_the_tempest 30w, ashes_of_the_pawn 420w) +
one post-tune 420w verification roll.
Next: eyeball; verification roll; source-bank e2e sweep.
Commits: 3d32b265, 636d78cf (+ this docs commit) -- pushed. Suite was
7153 green pre-portability-dirt; Bug Bible 17/7/3.

## 2026-07-10 ~01:30 -- HEAD 1c735c2d + docs (v2.0-alpha)

Did:
- LIVE 30w original_radio OBS smoke: GREEN on roll 6 -- "Page in the
  Tempest" published (otr\obs\...20260710_010652...final.mp4, 48 MB,
  RESULT SUCCESS, 548s). Five real production bugs found+fixed at root
  across the failed rolls, each with tests, suite+bible green, pushed:
  7f459e21 (A2 verbatim grounding: ws-normalized match + typed repair +
  deterministic key_term prune -- the prune FIRED live on a later roll),
  75173fc4 (original_qa evidence bar: hard kills need lexicon
  corroboration or a confirm-pass verbatim quote; discards stamped LOUD),
  a61ab2ed (kill authority per class: weapons/anachronism lexicon-only
  -- a grounded quote proves the line, not the class), 6fdf3f6e (ladder
  logs raw-output head on every failure -- exposed gemma truncation),
  d526c8b7 (creative slot -> nemo in canonical: gemma-4 Q8 cannot hold
  n_ctx 4096 on 16GB, the silent 2048 downgrade truncated concept JSON;
  enforces the standing bake-off rejection), 1c735c2d (epilogue_missing
  deterministically refuted when the outro row exists + slot pins
  retargeted).
- Bug Bible +BUG-11.26 (verbatim-grounding gates) + static tripwire +
  kebab fix, pushed (survival guide @ 1a01037).
- Validator record: OTR_WorkflowValidator OK in the green run (23/55,
  drift=0); the lane itself = NO workflow diff.
Current step: original_radio pre-ship -- smoke + validator gates GREEN;
OPERATOR EYEBALL is the only remaining gate (content notes in
GO_FORWARD section 0: name drift, stage-direction leak, quote-wrapped
lines, sci-fi premise tension).
Next: operator eyeballs the published mp4; then source-bank e2e sweep.
Commits: 7f459e21, 75173fc4, a61ab2ed, 6fdf3f6e, d526c8b7, 1c735c2d
(+ this docs commit) -- all pushed. Operator's own windows added
b288d8b6, bff86af9 (portability docs, benign).

## 2026-07-09 ~night -- HEAD 604ccdd3 (v2.0-alpha)

Did:
- /kibitz r2 (coding plan) on ARCHITECTURE_V4 + INTRO_REWRITE_SPEC:
  anchor-first, Codex auto green, agy auto timed out -> operator pasted
  the manual prompt, its review judged. 3-way convergence; shape A
  locked; synthesis = R2_CODING_PLAN.md. Operator left ("do r3-r4 and
  start coding") -> full autonomy.
- /kibitz r3 (wiring): 5 codex must-fixes verified+folded (seam-accessor
  wall, briefs return shape, dual source_meta restamp, title-regen
  staleness root-cause, QA-before-aggregates order) = R3_WIRING_DELTAS.md.
  /kibitz r4: converged, pins P1-P8 (agy auto dead 3x; codex + anchor).
- BUILT + PUSHED CHUNK A `181506e8` (intro rewrite all banks + title fix;
  c5a2 pin retargeted to the script_text L-opener per its own docstring).
- BUILT + PUSHED CHUNK B `604ccdd3` (the whole original_radio
  SAME-COMMIT set, runnable:true). Mid-build catches fixed at root:
  spark deck needed the routing pack-SIDECAR registration; the
  bank-shape dispatch needed the runnable conjunct (custom keeps its
  pinned LOUD SourceContractMissingError path).
- Suite 7136/31/1 + Bug Bible 16/7/3 green after each chunk; AST/BOM/
  0-byte verify clean; HEAD == origin. No workflow JSON diff.
- Note: `3060fd3a` (portability brief) is the operator's own docs commit
  from his other window -- audited, benign.
Current step: original_radio campaign -- BUILD SHIPPED; remaining gates =
live 30w original_radio smoke + OTR_WorkflowValidator no-diff record +
OPERATOR EYEBALL (queued).
Next: run the live 30w smoke (selective reset first), then eyeball, then
the source-bank end-to-end sweep.
Commits: 181506e8, 604ccdd3 (+ this docs commit) -- all pushed.

## 2026-07-09 ~evening -- HEAD 5a09984c (v2.0-alpha)

Did:
- 5-agent Sonnet QA fan-out on all 4 source-bank routes + ledger contract
  (operator skipped further live smokes). Synthesis:
  docs/2026-07-09-source-route-qa/QA_SYNTHESIS.md (local; dated dirs are
  gitignored).
- FIXED+PUSHED closing-seam bank routing (QA F1) -- coda/announcer
  seams pack-route; PD+Shakespeare coda re-authored to bridge contract;
  title_form_label wired; 30 tests. SHA CORRECTION (codex fan-out catch):
  the CODE+TESTS live in `40535ddc` (the operator's Codex loop committed
  the in-flight tree bundled with its dia hardening); `321bcc9c` on top
  carries only docs (dated doc dirs gitignored). Cite 40535ddc for the
  closing-seam code.
- FIXED+PUSHED 5a09984c: produced-story meta split -- K.5.6 summary pass
  stamps meta["produced_story"]; credits/HUD/treatment/music repointed.
- Seated tencent/hy3:free on the roundtable panel until 2026-07-21
  (62962121) + CLAUDE.md section 8 arc routing (R1 cloud, r2-r4 kibitz).
- original_radio R1 COMPLETE: ARCHITECTURE_V1 + anchor review -> live
  4-model roundtable (GPT-5.6-sol / Gemini-3.1-pro / DeepSeek-v4-pro /
  hy3:free; ~$0.13) -> pass01_judgment.md -> ARCHITECTURE_V2.md. Key
  redesigns: creative front (concept/select/brief) runs INSIDE
  build_original_briefs at D.2 BEFORE structure; v2-plan naming adopted
  (original_multi_pass + original_*_system seams); whole-script
  original_qa gate; disclosure must EXPLICITLY say machine-generated;
  cast pass collapsed; num_characters widget feeds the concept pass.

- R1 pass02 run on ARCHITECTURE_V3 (operator overrides: Hitchcock ironic
  epilogue instead of spoken disclosure; NO era frame / raw timeless
  story; RUNNABLE ON BUILD, no staged flips, no fallbacks, HARD FAILS
  ACCEPTED; north star = max story complexity / max code elegance).
  Panel 4x"no" -> judged -> **ARCHITECTURE_V4.md = BUILD SPINE**. Key:
  the epilogue is the ANNOUNCER OUTRO line (empty news_close_brief
  routes there; outro already knows the produced ending) -- zero new
  passes; disclosure lives in the printed layer (news_used + bank-aware
  HUD label replacing hardcoded "NEWS SEED" + unconditional credits
  line); anachronism defense is prompt-side + lexicon only.

- Local read-only fan-out QA (operator request) on the two shipped chunks:
  Antigravity returned NO blockers/majors; 2 verified MINORs FIXED same
  session (stopword bypass in produced-story cast grounding; off-by-one
  dropping the closing excerpt window at exact cap boundary -- also fixed
  in the older reflection builder it was copied from). Codex CLI not on
  system PATH from this session; operator pasting the brief into Codex
  manually -- its report landed at docs/2026-07-09-source-route-qa/
  local_fanout/codex_review_manual.md and was judged SAME SESSION: one
  real BLOCKER-class bookkeeping catch (the 321bcc9c/40535ddc SHA mixup,
  corrected in these docs); all its code checks CLEARED the current tree.
  Fan-out verdict overall: architecture sound, 3 real minors total, all
  fixed and pushed.

- NEW OPERATOR FEATURE (late): post-composition INTRO REWRITE -- once the
  story is done, rewrite the announcer intro from the PRODUCED first
  scene + cast, spoiler-safe by input starvation (scene-1 rows only).
  Spec: docs/2026-07-09-original-radio/INTRO_REWRITE_SPEC.md (shape A =
  derive ProducedOpenBrief -> existing safe-open composer, anchor lean;
  shape B = new rewrite seam). Runs BEFORE outro compose so the
  tone-echo reads the final intro. Joins kibitz r2 scope.

Current step: original_radio campaign -- R1 CONVERGED (2 passes,
~$0.26 total). Next: /kibitz r2 (coding plan) on
docs/2026-07-09-original-radio/ARCHITECTURE_V4.md + INTRO_REWRITE_SPEC.md,
then r3 wiring, r4 convergence, then build: tests first, SAME-COMMIT
registry set SHIPPING runnable:true, pre-ship gates = suite + Bug Bible
+ mocked pipeline + live 30w smoke + operator eyeball.
Commits: 62962121, (40535ddc co-authored), 321bcc9c, 5a09984c -- all pushed.

## 2026-07-11 -- original_codex56sol constrained implementation claim

Operator authorized non-GPU Chunks A-C/E to begin while the current Sci-Fi Codex
live run remains active. Base and origin were both
`26952a7ea64d61a2178485ac2708e350b52f9b48` on `v2.0-alpha`. Prior-owner dirty
files (`nodes/_otr_scifi_codex.py`, `scripts/otr_run_watcher.ps1`,
`tests/test_scifi_lane_schema_parity.py`, and the cue-ledger prompt) and all live
processes are excluded. Overlapping changes and Chunk D remain gated on operator
release. First action: force-publish the locked fingerprint, comparison, and
wording-corrected coding plan, then implement non-overlapping Chunk A surfaces.
