# Production Bug Log (staging pre-Bible)

**Contract (operator, 2026-07-10; AMENDED 2026-08-07):** Claude appends entries
here AUTONOMOUSLY, but ONLY for bugs that actually failed in a live/prod run
(live render, headless lane, soak, published episode). Dev/audit/review catches
get fixed, never logged. Promoted entries get a `- promotion: BUG-...` mapping;
rejected ones get marked `REJECTED` and stay for the record. Append-only,
newest last.

**AMENDMENT 2026-08-07 -- a window MAY now promote a single genuinely-uncovered
entry directly.** The original rule said "NO entry here touches the Bug Bible
directly -- at ship time the operator triggers a BUG FAN-OUT". That was written
when checking coverage meant re-scraping the whole bug history, which was far
too expensive to do per session. **That constraint is gone:**
`otr_coverage_index.yaml` in the survival-guide repo now maps all 369 OTR bug
records through 2026-08-07 to Bible ids, so a window can check ONE new bug
against the index in seconds instead of paying for a full scrape.

So the rule now splits by SIZE, not by authority:

* **A single new entry -- the window promotes it**, at wrap-up, if and only if:
  it clears the admission rule above (verified by a live artifact, not a review
  finding); it is checked against `otr_coverage_index.yaml` AND `BUG_BIBLE.yaml`
  and found genuinely uncovered; and it lands under the **Three-File Contract**
  in ONE commit in the survival-guide repo -- YAML entry + README count +
  executable coverage -- with its `otr_coverage_index.yaml` row appended in the
  same change. Then stamp `- promotion: BUG-...` here.
* **The BULK FAN-OUT over the backlog stays the OPERATOR'S**, unchanged. Batch
  promotion, re-litigating older entries, and any judgment call about whether a
  historical incident deserves a rule are not a window's to make.
* **When in doubt, record the candidate and leave it.** A window that cannot
  cleanly establish "genuinely uncovered" writes the candidate into
  `docs/GO_FORWARD_PLAN.md` for the fan-out rather than guessing. Never
  re-scrape indexed history; only the delta past the index date is ever scraped.

This amendment resolves a real contradiction, not a hypothetical one: the
`otr-handoff` v2 skill instructs a wrapping session to promote a genuinely
uncovered bug, and the pre-amendment contract forbade it, so the 2026-08-07
session recorded its candidate instead of promoting. Under this rule that
session would have promoted.

`promotion` and `status` are deliberately separate axes. `promotion` means a
real production incident has supplied a reusable Bible rule; `status` continues
to describe the implementation's own fix/requalification state. A fixed rule
may therefore be promoted while its same-seed 120-word confirmation remains in
progress. Conversely, an unproven review finding never receives either marker.
Running tests/bug_bible_regression.py after every code change stays automatic
and is unrelated to this log.

Entry format:

```
## PBUG-YYYYMMDD-NN -- short title
- surfaced: <which live run: smoke/soak/episode + date>
- symptom: <one line, what the operator/log saw>
- root cause: <one line>
- fix: <commit sha + one line>
- verify idea: <candidate machine check for the future bible test>
- bible-worthy: <yes/no guess + why -- operator decides at fan-out>
- promotion: <BUG-id after approved fan-out; omit while pending>
- status: OPEN | PROMOTED <id> | REJECTED
```

---

Backfill note: entries PBUG-20260612-01 .. PBUG-20260711-02 were mined 2026-07-10/11
by a two-agent backsweep (git history + handoff/smoke docs), prod-only bar applied,
cross-checked against BUG_BIBLE.yaml (BUG-11.26 family, 12.47, 07.16 excluded as
already promoted). Confidence tags preserved from the sweep.

## PBUG-20260612-01 -- headless boot dies on cp1252 emoji print
- surfaced: detached headless soak/API boot, 2026-06-12
- symptom: boot dies ~13s, exit 1, "SERVER DID NOT COME UP"
- root cause: detached cmd inherits cp1252; prestartup_script.py printed U+2705/U+2713 -> UnicodeEncodeError
- fix: scripts/_otr_soak_server_launch.cmd sets PYTHONUTF8=1 + PYTHONIOENCODING=utf-8; rule codified in CLAUDE.md section 5
- verify idea: launcher-path boot succeeds; any new boot path asserts UTF-8 env
- bible-worthy: yes -- Windows console-codec boot killer, hits any custom node that prints unicode at import
- confidence: MED (sourced from operating-rules doc, no dated incident log)
- status: PROMOTED BUG-02.15

## PBUG-20260616-01 -- LTX-AV soak VRAM peak 15.8GB over the 14.5GB cap
- surfaced: LTX full-episode soak, 2026-06-16 (976ab329)
- symptom: soak measured 15.8GB peak on a 14.5GB gate, both device modes
- root cause: Gemma text encoder stayed GPU-resident through the LTX pass
- fix: b0925c37 moved encoder to cpu; 1e5d66f4 REVERTED it after soak re-measure proved the offload ineffective -- record documents a fix attempt that live evidence disproved
- verify idea: full-episode soak VRAM peak check; assert S9 offload state matches the reverted decision
- bible-worthy: yes -- "the obvious offload fix measurably did nothing" is worth pinning so it isn't retried blind
- confidence: HIGH
- status: PROMOTED BUG-07.17

## PBUG-20260618-01 -- remote creative slot crashed episode with KeyError
- surfaced: live run with creative_model='openrouter:slot-a', 2026-06-18
- symptom: episode aborted at line-compose with KeyError
- root cause: resolve_creative_system_prompt did rows[repo_id] against a CURATED_LLM_MODELS-only dict; remote handles aren't in it
- fix: 1f196ac3 -- rows.get(repo_id) with MODERN-prompt default
- verify idea: full episode with a remote slot handle completes, modern prompt used
- bible-worthy: yes -- exact-match lookup vs non-curated id, recurring trap
- confidence: HIGH
- status: PROMOTED BUG-11.27

## PBUG-20260618-02 -- visualizer soak found 4-bug integration cluster
- surfaced: Task 2 visualizer soak, 2026-06-18 (4a92ed66, 21 clips)
- symptom: crashes/misbehavior on 0-frame beats, silent beats, missing master-audio slice, over-gated audio_ref
- root cause: four missing guards -- no 0-frame floor, no idle-scope handling, audio_ref wrongly gated in assert_usable, b000 master slice never fed
- fix: afab1a3 + c5c14c90 + d4607974 + bad1bba3
- verify idea: visualizer soak forcing silent/0-frame beats, status=success
- bible-worthy: yes -- soak-found cluster, four distinct root causes
- confidence: HIGH
- status: PROMOTED BUG-07.18

## PBUG-20260620-01 -- published episode bars overlay read the silent source
- surfaced: obs-final render pipeline, 2026-06-20 (8d7e6604 verification)
- symptom: bottom bars overlay baked flat/green instead of audio-reactive in a PUBLISHED episode
- root cause: bars overlay read the silent blend source instead of the master WAV
- fix: f6788882 -- bars read the master WAV
- verify idea: obs final render, assert bars track master audio amplitude
- bible-worthy: yes -- defect shipped to a published artifact
- confidence: HIGH
- status: PROMOTED BUG-08.07

## PBUG-20260622-01 -- UnboundLocalError crashed every episode at flag-stamp
- surfaced: night-soak window, 2026-06-22 (096ef64e)
- symptom: every episode crashed with UnboundLocalError at execution
- root cause: local `import os` inside run() made os function-local; the L2/L7 meta-stamp referenced os.environ before the local import line executed
- fix: 096ef64e -- local import at the stamp site; suite never exercised the heavy node so it slipped through
- verify idea: end-to-end test exercising the L2/L7 stamp; lint for mid-function shadowed imports
- bible-worthy: yes -- Python scoping trap invisible to unit tests
- confidence: HIGH
- status: PROMOTED BUG-05.10

## PBUG-20260622-02 -- announcer coerced to character role, voice engine crash
- surfaced: live-smoke, 2026-06-22 (ffe23245, "(live-smoke)" tag)
- symptom: pre-freeze sweep re-roled the announcer intro to character -> bark engine -> EngineUnusable
- root cause: cast_ids_from_ledger didn't exempt a cast row NAMED ANNOUNCER from role coercion
- fix: ffe23245 -- exclude ANNOUNCER-named rows from coercion
- verify idea: episode with announcer keyed as ordinary cast id renders clean
- bible-worthy: yes -- naming-convention trap in role coercion
- confidence: HIGH
- status: PROMOTED BUG-07.19

## PBUG-20260622-03 -- stage-direction-only character line crashed voice render
- surfaced: live-smoked fix set, 2026-06-22 (f8a8645e)
- symptom: a line with zero spoken content reached the voice engine and crashed the render
- root cause: no handling for a dialogue row that was pure stage direction
- fix: e62081f9 recompose to real dialogue (root); 9a4f0a71 silence backstop (NOTE: backstop is a fail-soft -- flag against current no-fallback law at fan-out)
- verify idea: force a stage-direction-only line through; assert recompose path, no crash
- bible-worthy: yes -- degenerate-content class
- confidence: MED
- status: PROMOTED BUG-07.20

## PBUG-20260623-01 -- refine-loop save failures racing the freeze cascade
- surfaced: live-smoke, 2026-06-23 (9f29f644)
- symptom: intermittent save failures during the refine loop
- root cause: loser-directory cleanup raced the freeze cascade
- fix: 9f29f644 -- ship the LAST revision, drop the racing cleanup
- verify idea: repeated refine-loop runs, zero save failures, freeze lands
- bible-worthy: yes -- race class, easy to reintroduce with future cleanup code
- confidence: HIGH
- status: PROMOTED BUG-12.48

## PBUG-20260702-01 -- night-queue proof9c: VRAM ceiling breach, zero clips
- surfaced: overnight night-queue run, 2026-07-02 (4dd79dbe verdict)
- symptom: leg produced zero clips; VRAM ceiling ops breach mid-run
- root cause: never fully isolated; retried at 832x448 per the verdict
- fix: none identified (diagnostic verdict only)
- verify idea: n/a until root-caused
- bible-worthy: no -- unresolved diagnostic, keep for the record
- confidence: LOW
- status: OPEN

## PBUG-20260703-01 -- overnight soak died: Ollama daemon down
- surfaced: overnight model-matrix soak, 2026-07-03 (c36dfe3e)
- symptom: soak died mid-run, local-LLM legs had nothing to call
- root cause: daemon down; no preflight health check in the soak launcher
- fix: c36dfe3e -- daemon started, soak relaunched (env fix, not code)
- verify idea: soak launcher preflights daemon health before queuing legs
- bible-worthy: maybe -- precondition-check class, though root cause was environmental
- confidence: HIGH
- status: OPEN

## PBUG-20260704-01 -- Sonilo cloud music rejected 422 provider_rejected
- surfaced: live cloud-audio proving run, 2026-07-04 (8f146394 "FIXED+PROVEN live")
- symptom: music calls rejected HTTP 422
- root cause: requested duration under provider minimum, no floor applied
- fix: 8f146394 -- min-duration floor + trim
- verify idea: short-duration Sonilo request completes
- bible-worthy: yes -- cloud-API contract violation class
- confidence: HIGH
- status: PROMOTED BUG-09.05

## PBUG-20260704-02 -- nano_banana_2 TypeError: string indices must be integers
- surfaced: live cloud-image coverage sweep, 2026-07-04 (606dc7f1)
- symptom: cloud_nano_banana_2 requests crashed with TypeError
- root cause: GeminiNanoBanana2V2 expects model as DYNAMICCOMBO_V3 dict; node sent a bare slug string (seedream's different node takes the bare string -- contract varies per node)
- fix: 606dc7f1 -- send the dict shape
- verify idea: live nano_banana_2 render completes
- bible-worthy: yes -- dict-vs-string contract mismatch across V3 cloud nodes
- confidence: MED
- status: PROMOTED BUG-09.06

## PBUG-20260709-01 -- distinct Chatterbox voice ids shared one WAV
- surfaced: all-Chatterbox 30w OBS live smoke, 2026-07-09
- symptom: two logically distinct voice ids resolved to the same underlying WAV
- root cause: no same-asset/provider collision check when allow_voice_reuse=False
- fix: same-day fix blocks asset/provider collisions under no-reuse (see GO_FORWARD 2026-07-09)
- verify idea: resolve N ids under allow_voice_reuse=False, assert distinct WAV hashes
- bible-worthy: yes -- no-reuse-gate class for any engine with shared assets
- confidence: HIGH
- status: PROMOTED BUG-07.21

## PBUG-20260710-01 -- gemma-4 Q8 silent n_ctx downgrade truncated concept JSON
- surfaced: original_radio live 30w smoke, 2026-07-10
- symptom: creative-slot output truncated -> schema failures downstream
- root cause: gemma-4 Q8 can't hold n_ctx 4096 on 16GB; silent 2048 downgrade
- fix: d526c8b7 creative slot -> Mistral-Nemo in canonical; portability S1 later made ALL silent n_ctx downgrades raise
- verify idea: request n_ctx over capacity, assert raise not downgrade (S1 test should already pin)
- bible-worthy: yes -- silent-downgrade class, though S1 now kills it globally
- confidence: HIGH
- status: PROMOTED BUG-11.28

## PBUG-20260710-02 -- epilogue_missing false-positive killed a roll with outro present
- surfaced: original_radio live smoke hardening, 2026-07-10
- symptom: roll killed for "epilogue_missing" while the outro row existed
- root cause: detection check + slot pins mistargeted
- fix: 1c735c2d -- deterministic refutation when the outro row exists, pins retargeted
- verify idea: fixture with outro row at retargeted slot, assert no false kill
- bible-worthy: check overlap with BUG-11.26 family at fan-out (this commit was NOT in the four folded into 11.26)
- confidence: MED
- status: PROMOTED (folded into BUG-11.26 law d, no new entry)

## PBUG-20260710-03 -- QA judge "proved" a violation by quoting clean text
- surfaced: original_radio 420w night batch Roll A, 2026-07-10
- symptom: confirm judge killed a roll for news_source_framing citing the CLEAN intro verbatim
- root cause: judge kill lacked lexicon-only corroboration for closed-vocabulary classes
- fix: 3d32b265 -- news_source_framing + machine_attribution became lexicon-only kill classes
- status: PROMOTED (folded into Bible BUG-11.26 follow-on law c, survival-guide commit 2833863)

## PBUG-20260710-04 -- fable2 P3 reroll: jinja TemplateError on consecutive user messages
- surfaced: scifi_fable2 30w live smoke roll 2, 2026-07-10
- symptom: TemplateError mid-render on the P3 reroll path
- root cause: reroll emitted two consecutive user-role messages; chat template requires alternation
- fix: fold reroll into ONE user message (docs/2026-07-10-fable2-s1b-smoke-hardening.md)
- verify idea: construct a P3 reroll, assert strict role alternation
- bible-worthy: yes -- chat-template alternation, easy to reintroduce in any lane
- confidence: HIGH
- status: PROMOTED BUG-11.29

## PBUG-20260710-05 -- fable2 casting JSON truncated at 1000-token budget
- surfaced: scifi_fable2 30w live smoke roll 18, 2026-07-10
- symptom: casting JSON truncated at ceiling; salvage pulled a partial object that failed schema
- root cause: 1000-token budget too small for the structured payload
- fix: budget 1400 + wrapper-tolerant before-validator (same doc)
- verify idea: near-ceiling casting payload completes without the salvage path firing
- bible-worthy: yes -- token-ceiling truncation-then-salvage class, already recurred cross-lane
- confidence: HIGH
- status: PROMOTED BUG-11.30

## PBUG-20260710-06 -- fable2 word-band exhaustion: proportional band too narrow at small targets
- surfaced: scifi_fable2 30w live smoke roll 17, 2026-07-10
- symptom: roll died on WORD_BUDGET exhaustion (54 words vs 24-36 band)
- root cause: +/-20% proportional band is only 12 words wide at target=30
- fix: absolute slack floor +/-25 words; proportional governs >=125w (same doc)
- verify idea: unit test _word_band at target=30, absolute floor governs
- bible-worthy: yes -- same defect class flagged UNFIXED in original_radio P1-1; not yet generalized
- confidence: HIGH
- status: PROMOTED BUG-11.31

## PBUG-20260710-07 -- fable2 announcer row silently mutated to character+skip, reason null
- surfaced: scifi_fable2 30w live smoke roll 22, 2026-07-10
- symptom: postamble row arrived speaker_role=character, skip=True, tts_skip_reason=null after a green 8-pass spine -- no compose-flag breadcrumb
- root cause: UNKNOWN -- an unsanctioned cast-keyed mutator downstream; ROOT MUTATOR STILL UNIDENTIFIED
- fix: partial -- announcer sentinel char_id exempts rows from cast-keyed paths; mutator not found
- verify idea: trace/assert every cast-keyed mutation path; no path may flip announcer without stamping a reason
- bible-worthy: yes, HIGH PRIORITY -- silent data corruption with unresolved root cause
- confidence: MED
- status: PROMOTED BUG-11.32 (ROOT CAUSE OPEN)

## PBUG-20260710-08 -- fable2 injected fictional character into the real-news read
- surfaced: scifi_fable2 30w live smoke roll 9, 2026-07-10
- symptom: model placed its fictional heroine ("Lia") in the read-only real-news pass
- root cause: no gate against invented cast names leaking into the source-read pass
- fix: cast-name-in-read gate with teaching error (same doc)
- verify idea: fixture with fictional name in read output, assert gate rejects with repair prompt
- bible-worthy: yes -- fiction/fact bleed class, distinct from verbatim grounding
- confidence: HIGH
- status: PROMOTED BUG-11.33

## PBUG-20260710-09 -- fable2 CODA terminal punctuation killed a clean draft
- surfaced: scifi_fable2 30w live smoke roll 15, 2026-07-10
- symptom: otherwise-passing draft killed solely for CODA ending '.' instead of ':'
- root cause: colon is structurally load-bearing to a parser; treated as stylistic by the model, no normalization before the check
- fix: pivot colon normalized in shared pre-lex (flagged); inner sentence break remains the true defect (same doc)
- verify idea: CODA ending '.' normalizes before parse, no false kill
- bible-worthy: yes -- structural-punctuation-as-parser-key class; original_radio P2-2 flags same risk
- confidence: HIGH
- status: PROMOTED BUG-11.34

## PBUG-20260710-10 -- scifi bake-off canonical smoke halted at Codex P0: source-span mismatch
- surfaced: first scifi_codex canonical 30w live smoke (roll 2a), 2026-07-10
- symptom: technical model returned a fact whose source_spans quote != the payload slice; validator correctly halted before any dialogue/media spend
- root cause: repair prompt not explicit about field/start:end slice contract; typed repair reproduced the mismatch
- fix: `40a765ac` hardened originating-slot repair prompt showing required payload[field][start:end] identity + slice-mismatch diagnostics, applied to ALL THREE lanes (cross-lane audit found the same contract shape in Gemini/Sonnet P0)
- verify idea: offset-span fixture converges within the repair ladder budget
- bible-worthy: yes -- evidence-span contract class, cross-lane by construction
- confidence: HIGH
- status: PROMOTED BUG-11.35

## PBUG-20260711-01 -- scifi bake-off Codex P0: evidence-ID shape F0/F1 vs required F01/F02
- surfaced: scifi bake-off canonical 30w smoke roll 2b, 2026-07-10/11
- symptom: local model returned evidence IDs F0/F1/F2 where the v4 contract requires zero-padded F01/F02/F03; P0 validator halted the run
- root cause: typed-repair contract didn't give the model explicit lexical ID mappings; ID-shape expectation implicit
- fix: `731d49f7` repair contract tightened at the shared lane boundary across Codex/Gemini/Sonnet -- explicit lexical ID mappings + recompute-quotes-from-payload-slice instruction (dialogue untouched, metadata repair deterministic); roll 3 rerun pending
- verify idea: fixture returning unpadded IDs, assert repair converges to padded shape within budget; pin pad width in schema tests
- bible-worthy: yes -- ID-shape contract drift, second member of the P0-contract class with PBUG-20260710-10
- confidence: HIGH
- status: PROMOTED BUG-11.36 (roll 3 exposed the NEXT defect rather than hiding it -- see PBUG-20260711-02)

## PBUG-20260711-02 -- scifi bake-off Codex P0: correct ID, wrong quote offsets (span-integrity)
- surfaced: scifi bake-off canonical 30w smoke roll 3, 2026-07-11
- symptom: after the ID repair converged (F0 -> F01 correct), the model repeated a quote with WRONG offsets -- a separate P0 span-integrity failure; validator halted honestly
- root cause: repair contract fixed ID shape but did not force offsets to be recomputed against the payload slice
- fix: `731d49f7` fail-closed METADATA-ONLY repair module (nodes/_otr_scifi_source_repair.py + test): may reindex an EXACT quote already present in the source and normalize IDs; may NOT invent or rewrite dialogue. Dialogue rewrites remain the province of a later context-aware structured creative pass (premise + beats + cast lock + audit feedback in hand) -- operator ruling: never a blind Python hack or context-free LLM retry that breaks the story arc
- verify idea: offset-shifted exact-quote fixture reindexes deterministically; ID normalizer pins F0 -> F01 (NOT F00 -- an actual test defect caught during this fix); dialogue field asserted byte-identical through repair
- bible-worthy: yes -- completes the P0 evidence-contract trilogy (span fidelity / ID shape / offset integrity); strong class entry at fan-out
- confidence: HIGH
- status: PROMOTED BUG-11.37

## PBUG-20260711-03 -- Codex creative score pass returned legacy Markdown shape
- surfaced: scifi bake-off canonical 30w smoke, Codex P3, 2026-07-11
- symptom: base and structural attempts returned Markdown prose; typed repair returned a legacy score object with missing `RadioScoreV4` keys and extra advisory-plan keys; the strict ladder halted before dialogue/media spend
- root cause: the pack seam said JSON-only but did not state the exact current schema's top-level keys, so the local model copied input structure instead of the requested typed artifact
- fix: `0d94c437` appends exact required top-level keys to every typed Codex/Gemini/Sonnet pass and repair seam; it preserves the full story context and forbids Markdown/prose
- verify idea: capture a typed pass prompt for each lane and assert the schema's required top-level keys are named; live smoke must reach the next pass without legacy-key drift
- bible-worthy: yes -- live structured-output contract failure, with cross-lane prevention
- confidence: HIGH
- status: PROMOTED BUG-11.38

## PBUG-20260711-04 -- Codex P0 used a full quote with truncated or wrong source field metadata
- promotion: BUG-11.46
- surfaced: scifi bake-off canonical 30w smoke, Codex P0, 2026-07-11
- symptom: a full headline quote was returned with `headline[0:55]`, so the validator saw only a truncated payload slice and halted the lane
- root cause: the model supplied a stale end offset and, in some artifacts, source-field labels did not identify the field containing the exact quote
- fix: `55f3cf17` rehomes an exact quote only when exactly one allowed payload field contains it, then recomputes start/end; absent or ambiguous evidence still fails closed
- verify idea: fixture with wrong field and offset rehomes to the unique literal field; fixture with absent or duplicate quote returns no repair
- bible-worthy: yes -- live source-evidence metadata failure, cross-lane helper
- confidence: HIGH
- status: OPEN

## PBUG-20260711-05 -- JSON parser salvaged a nested fact from a broken outer artifact
- promotion: BUG-11.47
- surfaced: scifi bake-off canonical 30w smoke, Codex P0, 2026-07-11
- symptom: malformed outer fact JSON was scanned past its first brace; the parser returned the first nested fact object, producing misleading missing-top-level-key errors and preventing the intended repair path
- root cause: shared fallback scanning treated a nested child as a valid top-level object when the response began with an invalid outer object
- fix: `5489baa8` fails closed when a response begins with malformed outer JSON instead of salvaging nested children; all source packs use the shared parser
- verify idea: malformed outer-with-valid-child fixture raises a top-level parse error; valid leading prose plus a valid object still parses normally
- bible-worthy: yes -- shared structured-call integrity defect across source packs
- confidence: HIGH
- status: OPEN

## PBUG-20260711-06 -- Codex P3 omitted required nested scene graph fields
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke, Codex P3, 2026-07-11
- symptom: score JSON had the correct top-level artifact but omitted required nested `scene_id`, `shot_id`, and `visual_prompt` fields; strict validation halted before script/dialogue/media work
- root cause: the prompt named top-level keys but hand-described no complete nested required-field tree, so the local model repeated an incomplete graph
- fix: `b9cfc508` generates a compact required-path inventory from each Pydantic model's `model_json_schema()` and injects it into all three lane prompt builders
- verify idea: assert `scenes[*].shots[*].scene_id` and equivalent nested paths appear in generated prompts; live smoke must pass P3 graph validation
- bible-worthy: yes -- live nested-schema contract failure, same family as PBUG-20260711-03
- confidence: HIGH
- status: OPEN

## PBUG-20260711-07 -- Codex P0 overclaimed beyond the supplied RSS payload
- promotion: BUG-11.46
- surfaced: scifi bake-off canonical 30w smoke roll 6, Codex P0, 2026-07-11
- symptom: the model returned a quote longer than the literal `full_text` payload; typed repair repeated it and the evidence validator halted before downstream work
- root cause: the model treated a claim-like sentence as source evidence even though the supplied payload did not contain that exact span
- fix: `6e6ff57b` drops unsupported facts/entities/numbers during metadata-only repair and retains only literal evidence; if no supported fact remains, the pass still fails closed
- verify idea: mixed fixture keeps literal facts and drops paraphrased facts; all-paraphrase fixture remains invalid
- bible-worthy: yes -- live grounding overclaim, same evidence-contract family as PBUG-20260711-01/02/04
- confidence: HIGH
- status: OPEN

## PBUG-20260711-08 -- Codex P3 generic repair repeated an incomplete graph
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke roll 7, Codex P3, 2026-07-11
- symptom: base and generic typed repair both omitted required nested scene graph fields despite a valid top-level score object
- root cause: non-P0 passes used the generic repair factory, which did not present the failed artifact and validation error with lane-specific graph-preservation instructions
- fix: `a27206df` routes typed repair for every Codex/Gemini/Sonnet pass through a schema-aware failed-artifact/error prompt while preserving premise, beats, cast, and authored content
- verify idea: force a nested graph validation failure and assert the repair prompt includes the failed artifact, exact validation error, schema paths, and context-preservation rule
- bible-worthy: yes -- live repair-contract failure, cross-lane by construction
- confidence: HIGH
- status: OPEN

## PBUG-20260711-09 -- Codex P3 repair omitted cast-locked speaker fields
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke roll 8, Codex P3, 2026-07-11
- symptom: schema-aware repair reduced the failure to two missing `speaker` fields on beats; the lane halted before script/media work
- root cause: nested graph repair did not explicitly bind each beat's speaker to its cast row by `char_id`
- fix: `fca99a5a` adds the cast-lock mapping rule to typed repair prompts for all three lanes
- verify idea: force missing beat speakers and assert the repair prompt requires cast-row lookup by `char_id`; live Codex P3 must clear
- bible-worthy: yes -- live cast/graph integrity contract failure, cross-lane prevention
- confidence: HIGH
- status: OPEN

## PBUG-20260711-10 -- Codex P5 repair omitted ScriptLine boundary metadata
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke roll 9, Codex P5, 2026-07-11
- symptom: full script artifact was otherwise shaped, but all eight lines omitted required `boundary` values; strict validation halted before audio/media work
- root cause: the repair contract named nested fields but did not define the boundary derivation from shot/beat order
- fix: `94331eb2` adds the structural rule: first line in shot = `shot_start`, first line in beat = `beat_start`, otherwise `continue`
- verify idea: force missing boundaries and assert the repair instruction contains the three-way derivation rule; live P5 must clear
- bible-worthy: yes -- live script graph metadata failure
- confidence: HIGH
- status: OPEN

## PBUG-20260711-11 -- Canonical RSS selector delivered a thin science payload
- promotion: BUG-11.49
- surfaced: scifi bake-off canonical 30w smoke roll 10, 2026-07-12
- symptom: run halted before P0 with `RSS payload is below the 80/12 thinness floor`; Gemini and Sonnet remained not-started because the serialized smoke gate stopped at Codex
- root cause: common science RSS selection returned a thin article to a lane whose source contract requires a substantial RSS body
- fix: `d01cf8bc` makes the shared RSS selector inspect up to ten candidates for sci-fi v4, require the same >=400-char/80-word/12-unique-token source floor before selection, and fail at selection if none qualify; legacy `science_news` keeps its existing richest-body fallback
- verify idea: canonical RSS fetch should either return a payload meeting the 80/12 floor or fail before queueing the sci-fi lane with a clear source-selection reason
- bible-worthy: yes -- live shared source-precondition failure
- confidence: HIGH
- status: OPEN

## FAN-OUT RECORD -- 2026-07-11 (operator-triggered)
23 entries promoted to the Bible (156 -> 179) @ survival-guide commit d50d773;
1 folded into BUG-11.26 law d (epilogue false-kill class); suite 17 passed /
7 skipped / 3 xfailed green; all 23 as non-testable notes (runtime-only
verifies), per the existing note pattern. Held OPEN: PBUG-20260702-01 (no
root cause), PBUG-20260703-01 (environmental). Mapping stamped per entry above.

## PBUG-20260711-12 -- Codex P5 output reservation truncated its own schema contract
- promotion: BUG-11.50
- surfaced: scifi bake-off canonical 30w smoke roll 11, Codex P5, 2026-07-11
- symptom: both P5 attempts returned prose or a score-shaped object instead of `ScriptArtifactV4`; the prompt guard reported `Truncated ... -> 1692 tokens` before each call
- root cause: P5 reserved a fixed 6500 output tokens inside an 8192-token context even for a 30-word script, leaving too little input budget for the failed artifact, graph, schema paths, and repair instructions
- fix: `fdc413ed` scales Codex whole-script P5/P7/P9 output reservation from the requested word steer (30w = 2200 instead of 6500), keeps every generated required path, removes the duplicate full schema from typed repair, and records token-budget/raw-size receipts; eight Kibitz reviews converged on the exact call-site wiring
- verify idea: 30w P5 prompt is not truncated, required ScriptArtifactV4 paths remain in the effective prompt, and canonical Codex reaches publish
- bible-worthy: yes -- live context-budget/structured-output contract failure
- confidence: HIGH
- status: OPEN

## PBUG-20260711-13 -- Codex P5 typed repair retained two forbidden legacy metadata values
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w Codex reroll after `fdc413ed`, 2026-07-11
- symptom: full-contract P5 base output failed eight fields; typed repair corrected six but retained `schema_version=scifi_codex.script_artifact.v1` and one `boundary=beat_end`, so strict ScriptArtifactV4 validation halted before publish
- root cause: the repair prompt exposed the exact literal and boundary enum contract but the local model copied two legacy values from its own failed artifact; there is no deterministic metadata-only normalization for ScriptArtifactV4 yet
- fix: `e679b754` adds `repair_script_artifact_metadata` -- a deterministic, metadata-only ScriptArtifactV4 repair that derives every mechanical field from the already accepted score graph: it sets the exact v4 schema literal, drops forbidden strict-model extras (e.g. `speaker`), maps each line's `shot_id` from the accepted graph, and derives `boundary` from accepted line/shot/beat order. It never touches dialogue, premise, beats, character intent, or any other story content, and fails closed when a graph or raw-line mapping is missing or ambiguous. The typed-repair factory short-circuits the LLM repair call whenever the deterministic result also satisfies the pass content validators
- verify idea: a metadata-only repair may set the schema literal, remove forbidden extra keys, map line shot IDs from the accepted score, and derive boundary from accepted shot/beat order without changing any dialogue; canonical Codex must then publish before Gemini/Sonnet or 720 starts
- verified: live canonical 30w Codex roll 12 (2026-07-11 08:18) reproduced the exact defect (`boundary=beat_end`) and the deterministic repair resolved it with NO LLM repair call; the lane cleared P5 and continued into the media tail
- bible-worthy: yes -- live legacy-enum persistence in typed repair
- confidence: HIGH
- status: FIXED (awaiting fan-out)

## PBUG-20260711-14 -- content-owned lanes never stamped the TTS delivery text
- promotion: BUG-12.51
- surfaced: scifi bake-off canonical 30w smoke, Codex voice gate, 2026-07-11 (first roll to survive P5)
- symptom: the lane cleared every structured pass, then halted at the voice handoff because its ledger lines carried no pronunciation-safe delivery string
- root cause: content-owned lanes seal canonical `text` in their own runner and bypass the legacy producer that stamps `text_for_tts`; the shared writer tail never stamped it for them
- fix: `e679b754` stamps delivery text in the one shared producer boundary every content-owned bank passes through -- after the last writer-side text mutation and before the lane finalizer's Phase-10 freeze; legacy lanes keep their byte-identical canonical-text delivery path
- verify idea: content-owned tail test asserts delivery stamps exist before the finalizer runs; legacy tail test asserts no stamps are introduced
- bible-worthy: yes -- shared producer-boundary gap that hits every content-owned source bank
- confidence: HIGH
- status: FIXED (awaiting fan-out)

## PBUG-20260711-15 -- content-owned lanes reached CreditsRoll with no cast-seed receipt
- surfaced: scifi bake-off canonical 30w smoke, Codex credits node, 2026-07-11 (first roll to survive the voice gate)
- symptom: the run rendered audio and media, then failed at the final credits node -- the ledger lacked the durable cast/episode seed the no-fallback credits provenance contract requires
- root cause: content-owned lane runners construct their own cast and so bypass the legacy cast-lock producer that stamps the seed receipt; nothing else establishes an authoritative episode seed
- fix: `e679b754` establishes one authoritative cast/episode seed in the shared writer tail (upstream of CastLock, freeze, and CreditsRoll) when the lane has not already supplied one; the seed also drives deterministic downstream voice assignment
- verify idea: content-owned tail test asserts the seed receipt is present before the finalizer runs; credits provenance passes with no fallback
- bible-worthy: yes -- same producer-boundary class as PBUG-20260711-14
- confidence: HIGH
- status: SUPERSEDED by PBUG-20260711-16 -- the receipt was right, the KEY was wrong (see below)

## PBUG-20260711-16 -- a "seed receipt" told CastLock to replay a cast nobody rolled
- promotion: BUG-12.51
- surfaced: scifi bake-off canonical 30w smoke roll 12, Codex CastLock, 2026-07-11 (first roll to survive P5 + the voice gate)
- symptom: the lane cleared every structured pass, stamped 13 delivery lines, rendered, and then died ~14 minutes in with `ValueError: num_characters must be 1-6, got 0` (cast_lock.py:189 -> _assign_bark_voices -> _otr_casting.replay_voice_assignment -> assemble_pre_locked_rows:1211)
- root cause: `meta.cast_contract.cast_seed` is not a generic episode seed -- it is a claim that the WRITER's seeded cast picker produced this cast and can be REPLAYED from it. Content-owned lanes build their own cast rows and stamp their own voice presets in the lane runner, so the picker never ran and the contract carries no `num_characters_request` -> `int(None or 0)` -> 0 -> ValueError. The PBUG-20260711-15 credits fix stamped `cast_seed` as a generic receipt and thereby CLOSED the `cast_seed is None` escape hatch these lanes had always relied on. A fix for one producer gap opened another.
- fix: the shared writer tail stamps `meta.episode_seed` ONLY (otr_credits_roll.py:279-284 already accepts it as the seed receipt, so credits provenance holds without asserting a replayable cast); and cast_lock._assign_bark_voices VERIFIES instead of REPLAYING for a content-owned lane -- it preserves the lane's `voice_preset` values and still runs the Gate 1 invariants, so such a lane can never ship duplicate or non-`v2/` bark voices. The legacy replay path is untouched (test_cast_voice_replay_parity pins it byte-for-byte).
- verify idea: a content-owned meta carrying a cast_seed must NOT enter the replay; a content-owned cast with two identical bark voices must still raise; the fable2 tail test asserts episode_seed is present AND cast_contract.cast_seed is absent
- bible-worthy: yes -- a receipt key that silently doubles as a behavior switch; the "my fix opened the next gap" class
- confidence: HIGH
- status: FIXED (awaiting fan-out)

## PBUG-20260711-17 -- P7 echoed the request envelope and truncated against its own output cap
- promotion: BUG-11.50
- surfaced: scifi bake-off canonical 30w smoke roll 12, Codex P7, 2026-07-11
- symptom: `OUTPUT_CAP: prompt_tokens=4543 generated_tokens=2800 max_new_tokens=2800` then `no decodable top-level JSON object found`; the raw head shows the model emitting `{ "artifact_inputs": { "accepted_line_count": 13, ...` -- the INPUT envelope -- instead of the artifact root. The structural retry happened to recover, so the run survived on luck.
- root cause: (1) the whole-script root contract forbade returning a score/scene/beat/patch but never forbade echoing the request envelope keys (`pass_id`, `artifact_inputs`, `result_json_schema`); (2) `_script_output_token_budget` scaled the reservation from the WORD STEER alone, but a ScriptArtifactV4 serializes strict per-line metadata for every accepted line -- the accepted LINE COUNT drives its size as much as the dialogue does, so a wide graph under-reserves and truncates
- fix: the root contract now names the forbidden envelope keys and requires the response to begin at the v4 schema literal; `_script_output_token_budget(requested_words, accepted_line_count)` scales on both drivers, is computed after the score is final (P3/P3_rewrite), and records a token-budget receipt
- verify idea: budget rises with line count at a fixed word steer; the AST test still pins `script_token_budget` on P5/P7/P9; a 720w run must not truncate
- bible-worthy: yes -- structured-output sizing driven by the wrong dimension; sibling of PBUG-20260711-12
- confidence: HIGH
- status: FIXED (30w); the 720w context-cap ceiling below is still OPEN

## PBUG-20260711-18 -- 720w whole-script passes cannot fit the 8192 context cap (OPEN)
- surfaced: analysis during roll 12, 2026-07-11 -- NOT yet hit live (30w fits)
- symptom (predicted): at 720 words the P7/P9 prompt (full previous script + line graph + review) and the output (the whole script re-emitted) both grow; local `context_cap` defaults to 8192 and the generate_fn LEFT-TRUNCATES silently, eating the system/schema prefix -- the PBUG-20260711-12 failure class, but silent
- root cause: `_build_truncating_generate_fn` uses `int(cache_entry.get("context_cap") or 8192)`; the local transformers path sets no context_cap, so 8192 is an arbitrary default, not a model limit (Mistral-Nemo supports 128k). P5/P7/P9 do not set `prompt_must_fit=True`, so they truncate instead of failing loudly
- fix: NOT APPLIED -- open fork: (a) derive context_cap from the model config with a VRAM-aware ceiling, (b) make P7/P9 a line-level PATCH pass so output stays flat as word count grows, (c) other. Out for a grounded local-panel opinion before the 720w bake-off
- verify idea: measure the real P7 prompt+output cost at 720w; whichever option lands, P5/P7/P9 should fail loud rather than silently truncate
- bible-worthy: yes -- silent context truncation of a provenance-bearing prompt
- confidence: HIGH (arithmetic), UNPROVEN (not yet observed live)
- status: OPEN -- gates the 720w bake-off

## PBUG-20260712-01 -- Gemma packed three owned items into suffixed fields
- promotion: BUG-11.45
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `0c1bb246-fae0-41c6-8f12-4cd8cccd27f3`, 2026-07-12
- symptom: P3 emitted `lost_object_2`, `lost_object_3`, and `resolution_links_2`; typed repair renamed them to `lost_object_secondary` / `lost_object_tertiary` instead of removing the schema violations, so the run failed closed after 459 seconds
- root cause: the P3 prompt named the collections but never stated that every selected lost object owns one separate `caller_threads` row with one singular `lost_object`, nor that every thread owns exactly one resolution row; Python also did not validate exact cross-artifact lost-object coverage
- fix: `5fd661ab` makes the base and repair contracts explicit, forbids numbered/suffixed pseudo-fields, validates the selected-object multiset, requires clue coverage per thread, and requires exactly one resolution per thread
- verify idea: validate a three-object selected possibility against a truth map with exactly three caller rows, at least one clue per thread, and exactly one resolution per thread; reject packed/suffixed fields, missing objects, duplicate resolutions, and repair-only renames; run the same canonical 30-word bank through Mistral and Gemma families
- bible-worthy: yes -- cross-model structured-output ownership ambiguity is reusable beyond OTR and survived a typed repair by changing only the illegal field names
- confidence: HIGH
- status: FIXED (the next E4B run used one row per object with no suffixed fields; it exposed the distinct nesting bug below; awaiting fan-out)

## PBUG-20260712-02 -- Gemma nested top-level truth collections inside caller rows
- promotion: BUG-11.45
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `fc362a77-ec2f-4bf0-a4fc-ac9017eeec53`, 2026-07-12
- symptom: P3 returned a schema-complete top-level truth map but also put a `causal_steps` array inside each of three `caller_threads` rows; typed repair repeated the forbidden nesting unchanged, and the run failed closed after 461.82 seconds
- root cause: the P3 seam and typed-repair rules described collection contents but did not state the exact top-level collection placement or exact caller-row field set; the repair ladder had no safe deterministic relocation for declared collections placed at the wrong depth
- fix: `8f5b3d4d` -- the P3 seam and repair rules name exact nesting, and a P3-only deterministic repair treats an existing top-level collection as authoritative or lifts nested rows verbatim only when top-level is absent/empty; strict schema plus full truth-graph validation must pass or the normal typed LLM repair runs
- verify idea: test authoritative top-level plus nested extras, absent top-level plus verbatim nested rows, non-list nested values, unknown fields, duplicate graph IDs, and a full mocked ladder proving the deterministic repair spends no additional LLM call; repeat Gemma/Mistral canonical smoke
- bible-worthy: yes -- strict item schemas do not prevent a model from placing a valid declared collection at the wrong depth, and typed repair may reproduce the nesting unchanged
- confidence: HIGH
- status: FIXED (next E4B run cleared P3/P4 and exposed the distinct P5 nesting bug below; awaiting fan-out)

## PBUG-20260712-03 -- Gemma nested score shots inside scenes
- promotion: BUG-11.45
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `649e1d99-c96d-485b-bce1-f68858f6d2d8`, 2026-07-12
- symptom: the run cleared P1-P4, then P5 returned `shots` arrays inside all four `scenes` rows; typed repair repeated the forbidden nesting after `PROMPT_GUARD` truncated its input from 4751 to 4592 tokens, and the run failed closed after 13:31
- root cause: the BroadcastScore seam and typed-repair rules specified scene and shot fields but did not explicitly require separate top-level scenes/shots/beats arrays; no deterministic structural repair handled declared score collections at the wrong depth
- fix: `54e159ec` -- exact top-level score ownership is stated in base and repair prompts; a P5-only deterministic repair retains authoritative top-level shots/beats or lifts nested rows verbatim when top-level is absent/empty, then requires strict schema and full score-graph validation
- verify idea: test top-level-authoritative cleanup, absent-top-level nested shots+beats lifting, non-list values, unknown fields, duplicate graph IDs, and a full runner with no extra LLM call; rerun the E4B/Mistral canonical smoke
- bible-worthy: yes -- a second typed artifact reproduced the wrong-depth collection class, showing the prevention law must cover every nested row boundary rather than one schema
- confidence: HIGH
- status: FIXED -- canonical E4B/Mistral prompt `fafd6306-cf0a-4c41-9bcb-11d2a8974864` cleared P5, froze the ledger, and published the episode; that run exposed the separate semantic false green below

## PBUG-20260712-04 -- Raiders of the Lost Prompts: opaque clue IDs let the spoken story abandon its source bank
- promotion: BUG-11.39
- surfaced: published canonical 30-word `original_codex56sol` episode `signal_lost_the_muted_melody_20260712_020438`, E4B creative + Mistral technical, prompt `fafd6306-cf0a-4c41-9bcb-11d2a8974864`, 2026-07-12
- symptom: history, freeze, audio identity, mux, and OBS publish all succeeded, but the immutable c03 draw (`parcel tag`, `brass button`, `choir note`, `clockwork display`, repair-and-return ending) became an ancient-artifact laboratory procedural speaking `protocol alpha`, `isotopic decay`, `resonance signature`, and `micro-vibrations`; none of the three lost possessions, the device, or the promised return survived into dialogue
- root cause: routing was correct and visual style never entered P1-P9; semantic provenance stopped at opaque clue IDs. P5 proved clue-ID coverage but not clue meaning, P6 received score+manifest without the draw/truth map, script validation checked graph/safety only, P7/P9 could bless a self-consistent replacement cause, and only response hashes survived for intermediate artifacts. The independently selected `sci_fi_radio` visual pack then amplified the already accepted story drift downstream
- fix: add a strict draw-derived grounding contract with literal lost-possession/device/resolution anchors; require object anchors on clue-carrying intents and spoken lines, the device anchor on reveal, and the resolution anchor on closure; thread truth+grounding into P5/P6/all retakes/P9; rerun the blind listener after a blocking retake; make P9 rejection fail closed; add an ordinary-world bank boundary and narrow incident-derived detour phrases; persist accepted intermediate artifacts plus line-level grounding evidence; prove visual-style changes leave every story message byte-identical
- verify idea: the exact seven-line `The Muted Melody` script must fail before P7; independently remove each object/device/resolution anchor and get its exact coordinate; switch only `visual_style` between `sci_fi_radio` and `video_art` and prove captured P1-P9 messages are identical; rerun deterministic c03 at 120 words and require the grounding receipt, frozen ledger, episode final, and OBS final
- bible-worthy: yes -- structured IDs can stay referentially valid while their semantic payload disappears between artifacts; an end-to-end media success is not a content-contract success
- confidence: HIGH
- status: FIXED IN CODE / AWAITING LIVE 120-WORD C03 REQUALIFICATION; the published 30-word episode is retained as a false-green regression artifact and does not qualify the bank

## PBUG-20260712-05 -- Every custom runner title was stamped as a Fable2 title
- promotion: BUG-12.49
- surfaced: forensic audit of the same Codex56 false-green ledger, 2026-07-12
- symptom: `meta.title_source` said `fable2_script_title` even though routing and authorship correctly identified `original_codex56sol`; the stale label could falsely implicate another story bank during incident diagnosis
- root cause: the shared writer tail hardcoded the Fable2 receipt whenever any custom runner supplied `final_title_override`
- fix: derive custom title provenance from `ctx.source_bank_row.source_bank_id`, preserve the established `fable2_script_title` value for the actual Fable2 lane, and stamp `<source_bank_id>_script_title` for every other custom runner without changing the pinned tail-context field contract
- verify idea: direct helper tests for Fable2 and Codex56 plus the existing title-override precedence suite
- bible-worthy: yes -- stale provenance labels turn correct routing evidence into a false root-cause lead
- confidence: HIGH
- status: FIXED IN CODE / AWAITING FAN-OUT

## PBUG-20260712-06 -- Gemma repeated invented music filenames through P5 repair
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification, prompt `7384fbe8-d1c9-4485-ba8e-b7f100329a12`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 reached the BroadcastScore on its first base call but added `opening_music.music_file=opening_music.mp3` and `closing_music.music_file=closing_music.mp3`; the typed repair repeated both forbidden fields, so strict validation failed closed after 12:32 and no ledger/media artifact was accepted
- root cause: the score seam closed the top-level, scene, shot, beat, and line-intent key sets but described music bookends only semantically; the shared schema instruction listed their required paths without explicitly forbidding nested extras, allowing a model to treat plausible production filenames as authored score fields
- fix: the bank prompt now states that each music bookend has exactly `description` and `generation_prompt` and explicitly forbids filenames/paths/cue metadata; the existing P5 structural normalizer now deletes only non-authoritative extra bookend keys, preserves every required LLM-authored value byte-for-byte, and still requires the complete strict score plus graph/content validators to pass before it can avoid another model call
- verify idea: inject the exact two `music_file` fields into an otherwise valid score, require deterministic repair with unchanged descriptions/prompts and zero extra LLM calls, pin the prompt wording, then rerun deterministic c03 at 120 words through canonical to ledger and OBS
- bible-worthy: yes -- required nested paths are not the same contract as exact nested key ownership, and a typed repair can faithfully repeat plausible but forbidden production metadata
- confidence: HIGH
- status: FIXED IN CODE / AWAITING LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-07 -- Gemma interleaved complete P5 beat blocks through repair
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification, prompt `d29b63d8-1890-40a4-a1ea-370bc9b02406`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 produced a strict BroadcastScore with complete typed beats but returned to an earlier `shot_id` after starting another shot; the typed repair repeated the same A/B/A topology and the run failed closed after 11:51 with `beats for each shot must form one contiguous block`
- root cause: the prompt named contiguous shot blocks and Python rejected interleaving, but the contract gave no concrete valid/invalid sequence example and the repair ladder had no safe deterministic ordering projection for otherwise valid authored beats
- fix: the base and repair prompts now state that the beats array is chronological and must never be reordered, give an A/A/B-valid and A/B/A-forbidden example, and require a fresh shot row/ID for a return cut; the P5 structural repair preserves the exact beat sequence and all authored beat content, clones only the reopened shot's mechanical row under a collision-safe ID, retags only the later run, and accepts only after the full score graph/content post-validator passes
- verify idea: interleave a valid score as shot_01/shot_03/shot_01 while keeping clues before reveal, require byte-identical beat-ID order and content with only the reopened-run shot IDs changed and zero additional LLM calls; force an ID collision and a hidden graph defect to prove deterministic naming and fail-closed behavior; rerun the identical c03 120-word seed through canonical to ledger and OBS
- bible-worthy: yes -- collection completeness does not imply ordered graph topology, and a typed repair can repeat a structurally plausible interleave indefinitely
- confidence: HIGH
- status: PARTIAL IN `09222618` -- the clone/retag projection was correct, but its repair-factory-only placement missed the typed-repair response; see PBUG-20260712-08

## PBUG-20260712-08 -- P5 deterministic repair did not run on the typed-repair response
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `09222618`, prompt `76cb5ca2-0ac7-4b2b-9b64-705b30f0cf75`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 base output again interleaved a closed shot; the repair-prompt factory could not accept the base after projection because another hidden validator defect remained, so it correctly requested typed repair. Gemma's typed-repair response repeated the same interleaving, then went directly to post-validation and failed after 12:33 without ever receiving the safe clone/retag projection
- root cause: deterministic P3/P5 structural normalization lived only inside `repair_prompt_factory`, which runs before the typed-repair model call. `structured_call` validates the typed-repair response directly; it does not call the factory a second time for a schema-valid content failure
- fix: hash each actual raw response for audit first, then run the same narrow P3/P5 projection inside the lane's slot-output wrapper on every ladder attempt. A projected model is serialized back to the ladder only when the complete pass post-validator clears; otherwise the original raw output and its real defect continue through the normal typed-repair path
- verify idea: make a base P5 response contain both a safe topology defect and a separate safety defect so it must reach typed repair; return a safe typed-repair response that still repeats A/B/A; require the per-attempt projection to preserve beat order, split the return shot, complete with exactly one repair model call, and produce resolving ledger boundaries
- bible-worthy: yes -- repair factories are not attempt-wide output middleware, so deterministic repairs placed only there can be bypassed by the response they requested
- confidence: HIGH
- status: FIXED IN CODE / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-09 -- raw P5 projection was not the schema-validated acceptance boundary
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `d024bc18`, prompt `51932200-9d57-499f-aae8-76f6fcf01631`, E4B creative + Mistral technical, 2026-07-12
- symptom: both the P5 base output and its typed repair were schema-shaped BroadcastScores with the same reopened-shot A/B/A defect; the slot-output projection did not accept either response, and the shared ladder failed closed after 12:36 with `beats for each shot must form one contiguous block`
- root cause: the clone/retag projection was still coupled to raw-string collection normalization before `structured_call` had created the strict `BroadcastScore`. That wrapper is useful for wrong-depth collections and nested extras, but it is not the guaranteed acceptance boundary for every schema-valid P5 response. A production response can therefore arrive at post-validation with the safe topology defect intact.
- fix: `P5` now applies the clone/retag projection inside its schema-validated post-validator. Every base, structural retry, and typed-repair response that parses as `BroadcastScore` must cross this hook. It mutates only the accepted in-memory score's mechanical `shots`/`beats` ownership, verifies the complete grounded score again, then runs authored-surface validation. The prompt also asks Gemma to silently scan the final beat sequence and mint a fresh shot row before emitting a return cut.
- verify idea: disable the older raw score normalizer in a mocked runner; a base A/B/A score must still produce a closed ledger with one extra cloned shot and no extra model call. Repeat with a separate safety failure on the base output so typed repair is required; its A/B/A response must clear through the same schema boundary. Run the identical c03 120-word seed to ledger and OBS.
- bible-worthy: yes -- a raw-output middleware hook is not a substitute for the strict typed object boundary where an artifact is actually accepted
- confidence: HIGH
- status: FIXED IN CODE / FULL SUITE + BUG BIBLE GREEN / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-10 -- Gemma repeated duplicate clue ownership through P5 repair
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `73de861a`, prompt `00196fd7-943a-4427-90f0-91dce04d4a4b`, E4B creative + Mistral technical, 2026-07-12
- symptom: the new P5 topology guard cleared the reopened-shot defect, exposing the next schema-valid error: a truth-map clue ID appeared in more than one `line_intent`. Gemma's typed repair repeated the duplicate and the run failed closed after 11:35 with `each truth-map clue must be assigned to exactly one line intent`.
- root cause: the safe first-placement-wins duplicate-clue projection still lived in raw-output/repair-factory helpers. Like the prior topology repair, it was not enforced over the strict `BroadcastScore` object that the shared ladder accepts.
- fix: move duplicate clue ownership into the same schema-validated P5 post-validator as reopened-shot topology. It keeps the first authored clue placement in beat order, removes only later duplicate references, reruns the complete grounded score validator, and leaves missing or unknown clues for the LLM repair path. The raw duplicate projection path is removed so tests cannot mistake it for the acceptance guard. Base and repair prompts now require a final exact-once `clue_ids` scan.
- verify idea: disable raw collection cleanup, inject a duplicate clue into a valid base score, and require a no-extra-call accepted ledger. Then add an unrelated forbidden authored phrase so typed repair is required and return a duplicate-clue repair response; require exact-once clues in the persisted accepted BroadcastScore with one repair model call.
- bible-worthy: yes -- ordered first-owner reconciliation is safe only after the full typed graph is available, not as speculative raw JSON cleanup
- confidence: HIGH
- status: FIXED IN CODE / FULL SUITE + BUG BIBLE GREEN / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-11 -- independent P5 repairs could not compose
- promotion: BUG-11.41
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `145e955b`, prompt `de6f4c1e-b021-4106-871e-8e4a3673bfa4`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 again returned a reopened-shot topology plus duplicate clue ownership. The topology guard was reached first but declined to apply because its helper demanded that the entire score, including the independent duplicate-clue invariant, already validate. Both the base and typed-repair responses therefore failed closed on the first reported topology error after 12:32.
- root cause: each safe normalizer was implemented as an all-or-nothing full-score repair. A valid artifact containing two independent, non-authoritative mechanical defects could not reach either repair's success path; the post-validator handled only the first reported defect rather than a bounded composition of disjoint projections.
- fix: split each P5 helper into a narrow projector and a full-validation wrapper. At the typed `BroadcastScore` acceptance boundary, apply at most the two proven-safe projections in deterministic order (reopened shot ownership, then duplicate clue ownership), preserve all authored prose/beat order/first clue placements, and run the complete grounded score validator only after the bounded composition. Any remaining or ambiguous defect remains a normal LLM failure.
- verify idea: create one base score and one typed-repair score with both A/B/A topology and a later duplicate clue, plus an unrelated forbidden phrase on the base so the typed call is mandatory. Disable raw cleanup and require the accepted score to retain beat order, mint the collision-safe return shot, keep first clue ownership, remove only the later duplicate, and use exactly one repair model call.
- bible-worthy: yes -- independently safe deterministic transformations must compose before a global validator can judge their shared result
- confidence: HIGH
- status: FIXED IN CODE / FULL SUITE + BUG BIBLE GREEN / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-12 -- full-score repair overflowed for a one-intent grounding omission
- promotion: BUG-11.42
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `ef6cd277`, prompt `cabf66f8-14d1-4de5-b043-d329b888df78`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5's typed structural guards cleared both reopened-shot topology and duplicate-clue ownership, but one non-announcer clue intent omitted the exact lost-object anchor `parcel tag`. The generic typed repair then attempted to regenerate the entire BroadcastScore, produced no decodable top-level JSON, repeated the same overflow on syntax retry, and failed closed after 14:39.
- root cause: the acceptance boundary treated a localized LLM-owned semantic omission as a whole-artifact repair. A complete score is too large and too fragile an output shape for a one-line intent correction, especially after the failed score and contract inputs are fed back through the repair ladder.
- fix: accept the P5 score after structural and safety validation, then immediately derive a bounded eligible-beat plan from the immutable grounding contract. Call a separate `ScoreIntentPatch` seam with only `{beat_id, current_intent, required_anchors}` targets. The LLM authors each replacement intent; Python accepts only one replacement for every and only planned beat, verifies literal anchors, merges no other field, and reruns the full grounded-score and authored-surface contracts. Prompt-pack and pipeline declarations make the tool auditable.
- verify idea: remove `stamp` from an otherwise valid P5 clue intent, require one nine-call runner where the sixth call is a `ScoreIntentPatch`, persist only the LLM-provided replacement intent, and reject both a missing literal anchor and an unplanned beat ID. Run the full suite and Bug Bible, then repeat the same c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- when a semantic defect is confined to an explicitly owned leaf, a whole-document repair is an avoidable reliability and context-window hazard. Create a small typed patch artifact, validate its exact scope, and retain full-artifact validation as the authority.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P5 grounding patch cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-13 -- grounding-intent patch could erase an already-valid anchor
- promotion: BUG-11.43
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `ec489787`, prompt `54c0a8bb-45f9-4cf2-bc56-49882fd16377`, E4B creative + Mistral technical, 2026-07-12
- symptom: P1-P5 cleared on their first attempts after P5 safely normalized reopened-shot topology and duplicate-clue ownership. The new `ScoreIntentPatch` first corrected four lost-object/device targets but omitted beat_09's required resolution anchor; its typed retry passed the patch-local anchor check yet the merged score still failed closed after 578.73 seconds.
- root cause: the patch plan listed only newly missing anchors. A selected target beat can already hold a different immutable anchor required elsewhere (especially reveal/closure beats that also carry clues); overwriting its complete `line_intent.intent` could silently remove that existing anchor. The patch post-validator checked local target coverage but did not validate the merged BroadcastScore before accepting the typed patch.
- fix: every selected target now carries forward every immutable anchor already present in its current intent, in addition to newly missing anchors. The patch's `structured_call` post-validator now applies the candidate in memory and rejects it unless the complete score grounding and authored-surface contracts clear; a typed repair receives that exact merged-contract error. The repair seam explicitly forbids visual direction, camera/scene/shot instructions, stage business, dialogue, and production metadata.
- verify idea: make a reveal beat carry the `stamp` clue and its already-valid `grille` device anchor while omitting `stamp`; require the plan to demand both literals, accept only a patch preserving both, and reject a patch that carries a banned phrase even when its anchors are complete. Run full suite, Bug Bible, and the same c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- a narrow patch must preserve every currently valid invariant in the field it replaces, not merely add the invariant that triggered repair. Patch-local schema acceptance is insufficient; validate the merged canonical artifact at the same structured-call boundary.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P5 grounding patch cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-14 -- full-script repair repeated one missing closure literal
- promotion: BUG-11.42
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `a9cf3bbe`, prompt `69a56fa7-afd3-4e72-b2b4-188a7afaac00`, E4B creative + Mistral technical, 2026-07-12
- symptom: P1-P5 cleared, including the bounded P5 grounding patch. P6 generated a structurally valid `PerformanceScript` but its closure line did not speak the exact immutable resolution anchor `returns everything`. The generic typed repair regenerated the whole script and repeated the identical omission, failing closed after 11:22.
- root cause: P6 treated a localized LLM-owned spoken-line grounding omission as a full-script repair. The prompt already named the literal, but the full artifact request was large enough that Gemma reproduced the otherwise-valid script and its one missing phrase instead of isolating the closure line.
- fix: P6 now accepts a structurally and safety-valid script, then derives a bounded `ScriptLinePatch` target only when full grounding fails. The LLM receives each affected `{line_id, current_text, required_anchors}` target and authors only replacement spoken text. Python requires exact line coverage and literals, preserves all immutable anchors already spoken on a targeted line, merges no other field, and validates the complete graph/text/grounding contract before accepting the patch. The new source-pack seam forbids labels, stage/camera/visual direction, production metadata, and wrappers.
- verify idea: remove the exact closure anchor from an otherwise valid P6 script, require a single extra `ScriptLinePatch` call that repairs only `line_005`, and preserve all other lines byte-for-byte. Also make a reveal line carry an object clue while already speaking the device anchor; require its patch plan and accepted replacement to retain both literals, then reject an otherwise anchored banned-phrase replacement. Run full suite, Bug Bible, and the identical c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- the localized semantic-repair law applies separately to each artifact boundary. A complete script is no more suitable than a complete score for correcting one owned leaf.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P6 grounding patch cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-15 -- later full-script retakes bypassed the bounded P6 grounding repair
- promotion: BUG-11.44
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `cb5166f8`, prompt `10438a88-66c6-400d-b7b9-d049b2f116f3`, E4B creative + Mistral technical, 2026-07-12
- symptom: P1-P6 cleared, including both bounded P5 and P6 grounding patches. The blind-listener loop requested P8; P8 returned a structurally valid replacement script but again omitted exact closure anchor `returns everything`. Its generic full-script typed repair repeated the omission and failed closed.
- root cause: the P6 local-line guard was attached only to initial script creation. P8, optional P8, and P9 retake paths continued to validate full grounding inside their full-script `structured_call`, so a later reauthoring route could reintroduce the same local defect and bypass the guarded acceptance boundary.
- fix: factor one `_call_grounded_script` acceptance path for every complete-script authoring call. It accepts only structural/safety-valid scripts, invokes a bounded `ScriptLinePatch` when and only when full grounding fails, validates the merged script before acceptance, and records a pass-specific patch journal entry such as `P8_grounding_patch`. P8 and P9 pipeline registry entries now declare the patch seam, keeping the dynamic repair route visible in the source-pack contract.
- verify idea: force P7 to request P8, make P8 omit the closure anchor, require a single P8 line patch and a clean blind-listener rerun before P9. Assert P8/P9 pipeline seam references include `codex56_script_anchor_patch`; run full suite, Bug Bible, and the identical c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- a validation/repair guarantee must cover every reauthoring route for an artifact, not only its first construction. Factor the guarded boundary rather than duplicating a one-off call-site fix.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P8 grounding patch cleared, listener rerun and P9 cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-16 -- detached soak monitor reported a blank exit code after canonical success
- promotion: BUG-12.50
- surfaced: same-seed c03 120-word live qualification, monitor log `logs/codex56_c03_120_after_156cb2e4`, 2026-07-12
- symptom: the detached PowerShell wrapper wrote `COMPLETE: SOAK_FAIL rc=` even though the canonical API reported `RESULT SUCCESS`, the final OBS MP4 was published, duration and byte-identical-audio checks passed, and the frozen ledger existed.
- root cause: the monitor trusted a blank `Process.ExitCode` from the detached PowerShell child as a failure without reconciling the canonical API's explicit terminal result.
- fix: the monitor now treats an empty child exit code plus a `RESULT SUCCESS` marker in the canonical runner log as success; a real nonzero code or missing success marker remains a fail.
- verify idea: the next detached canonical smoke with a successful API result must write `COMPLETE: PASS rc=0`; an absent success marker must remain a failure.
- bible-worthy: yes -- test orchestration must not turn an observed final asset plus explicit canonical success into a false-negative qualification verdict.
- confidence: HIGH
- status: FIXED IN HARNESS / AWAITING NEXT DETACHED-SMOKE CONFIRMATION

## HISTORICAL BACKFILL -- 2026-07-12 production-only Bug Log sweep

The archived `BUG_LOG.md` and `BUG_LOG_2026-06.md` contain many local labels,
including design notes, test-only findings, unresolved investigations, and
operator-pending visual observations. This backfill admits only historical
records with explicit live/published/GPU evidence, a grounded root fix, and a
current regression test. It does not promote an archived label merely because
its name contains `BUG`.

## PBUG-20260614-01 -- malformed post-blend filter silently dropped scopes and captions
- promotion: BUG-08.08
- surfaced: live look-QA of a published episode, 2026-06-14; server log recorded
  ffmpeg `gbrpformat` rejection and source-copy fallback while `obs_publish OK`
- symptom: a three-input procgen blend silently published without burned SDH
  captions or audio-reactive scopes
- root cause: an enabled green-overlay chain already ended in `,format=gbrp` and
  the next chain appended `format=gbrp` without a separator, producing the
  invalid token `gbrpformat`
- fix: commit `99320ae` adds the pixel-format pin exactly once; current
  `test_build_cmd_3input_scopes_no_double_format_gbrp_bug402` covers both
  overlay states and the caption burn
- verify idea: every enabled three-input filter combination has valid token
  separators, expected pixel-format pins, and its required visual layers
- bible-worthy: yes -- process success is not evidence that optional final
  compositing effects survived
- status: PROMOTED BUG-08.08

## PBUG-20260614-02 -- post-composition shortest input clipped the rolling-credits tail
- promotion: BUG-08.08
- surfaced: operator-verified fresh render, 2026-06-14; credits were absent from
  the published tail before the fix and visibly restored after it
- symptom: the final video stopped at the shortest upstream track, cutting a
  deliberately longer floor/HUD credits layer
- root cause: post-composition treated a short scopes track as the completion
  boundary despite the credits layer intentionally extending past master audio
- fix: preserve the intended long-form timeline; current
  `test_blend_cmd_does_NOT_use_shortest_for_c7_safety` guards the command
- verify idea: a credits/scopes fixture with a deliberately longer tail retains
  the complete post-roll in the final composition
- bible-worthy: yes -- final-output success must include duration and layer
  completeness, not merely an ffmpeg exit status
- status: PROMOTED BUG-08.08

## PBUG-20260626-01 -- LTX-AV activation spill caused a no-OOM multi-minute crawl
- promotion: BUG-07.22
- surfaced: GPU-validated live 30-word all-`ltx_audio_in` headless run,
  2026-06-26; 223 s/iteration spill reduced to steady roughly 11 s/iteration
- symptom: audio-conditioned video inference avoided OOM but fell into system
  memory spill with near-zero free VRAM and an extreme per-beat slowdown
- root cause: one VideoVAE stayed alive through both encode and decode, while
  no activation reserve protected the sampler from desktop VRAM contention
- fix: `ae8ec55e` splits encode/decode VAE lifetime; `bd5ffd23` scopes an
  `EXTRA_RESERVED_VRAM` minimum and restores it after the run
- verify idea: graph wiring has distinct VAE nodes; reserve scope raises,
  restores on exception, and never lowers a stricter existing reserve; GPU soak
  remains free of system-memory crawl
- bible-worthy: yes -- a slow no-OOM render is a real VRAM failure class, not a
  license to guess at quantization or offload changes
- status: PROMOTED BUG-07.22

## PBUG-20260702-02 -- orphaned one-shot environment hook poisoned later headless boots
- promotion: BUG-12.52
- surfaced: live all-`ltx_audio_in` probe, 2026-07-02; the report instead showed
  every shot rendered by HuMo from a crashed leg's stale force-engine override
- symptom: a canonically configured run silently inherited file-based engine
  overrides that were not present in the explicit new-run configuration
- root cause: normal post-leg cleanup did not run after a crash, leaving the
  sourceable environment hook to affect later boots
- fix: consume-once hook semantics plus canonical-wrapper stale-hook removal;
  `test_headless_wrapper_clears_stale_extra_env_hook_before_boot` pins the
  cleanup boundary
- verify idea: seed an override hook, run the canonical headless wrapper, then
  require hook removal and an engine receipt matching only explicit inputs
- bible-worthy: yes -- temporary file-based overrides must not become hidden
  persistent process defaults
- status: PROMOTED BUG-12.52

## BUG AUDIT RECEIPT -- 2026-07-12

Searched every repository filename containing `bug`, both historical bug logs,
all current PBUG entries, and bug-labelled commits. Promoted the July-11
canonical-smoke set plus the four historical incidents above only after locating
their real-run evidence and current regressions. Kept the unresolved July-2
VRAM diagnosis, the environmental Ollama outage, and the predicted (not yet
live) 720-word context risk out of the Bible; other archived local labels stay
out until they independently meet the same production-only admission rule.

## PBUG-20260712-17 -- Codex56 P6 grounding patch exhausted both live attempts
- surfaced: canonical 120-word `original_codex56sol` queue leg, prompt `e256be3f-69a0-495f-8a99-3bf9c06e01a8`, Gemma E4B creative + Mistral-Nemo technical, 2026-07-12
- symptom: the canonical API returned `RESULT FAIL`; node 1 stopped at `P6_grounding_patch` after two structured-call attempts, before ledger/media/OBS completion
- root cause: OPEN -- the queue wrapper preserved only the truncated terminal exception, not the exact messages, raw response, projection, and validator error needed to distinguish model omission from repair-contract or context failure
- fix: none yet; first reproduce or inspect the retained attempt artifacts after the code-ready Codex56 telemetry seam lands, then fix the owning representation/validator boundary rather than increasing retries
- verify idea: run the same 120-word model pairing with attempt telemetry; require the failing rung's exact raw/projected/error record, then add a focused regression for the isolated cause and a canonical rerun proving ledger, episode asset, `obs_publish OK`, and final OBS file
- bible-worthy: pending -- live admission is proved, but no reusable rule exists until the root cause and fix are known
- status: SUPERSEDED 2026-07-15 (baseline) -- the target lane
  `original_codex56sol` was ripped from the roster @ `3312aec7`, so the failure
  cannot recur as logged, and the code-ready telemetry seam this fix was gated
  on was retired with it. The diagnostic gap it names (no retained
  raw/projected/error attempt record) is carried forward as an engineering risk
  on the GO_FORWARD context/cap item, re-targetable at any surviving
  structured-call lane. Not Bible-eligible from this record.

## PBUG-20260712-18 -- Sci-Fi Codex P3 repair envelope rejected as the artifact root
- surfaced: canonical 120-word `scifi_codex` queue leg, prompt
  `cc9e0f8a-2a20-40a1-b5dc-da2fc8a400d6`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: `RESULT FAIL` at P3 after two structured attempts; the repaired
  `RadioScoreV4` was complete but nested under `resolved_artifact`, so strict
  validation reported every required root field missing and the wrapper extra
- root cause: the lane passed an exact single-key typed-repair transport
  envelope directly into the requested strict artifact schema
- fix: normalize only the exact `{"resolved_artifact": <object>}` transport
  shape at the Sci-Fi Codex response boundary, preserve original-wire hash and
  length, journal the normalization boolean, and keep mixed/non-object roots
  fail-loud
- verify idea: exact-wrapper, direct-root, mixed-root, non-object, original-wire
  telemetry, and prompt-seam exclusion regressions; then rerun the same canonical
  bank and require RESULT SUCCESS plus ledger and OBS final existence
- bible-worthy: pending -- live admission and reusable exact-envelope rule are
  proved; promote only through the standing Bug Bible fan-out
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-19 -- all-visualizer policy still invoked upstream image authoring
- surfaced: canonical 120-word `scifi_codex` queue leg, prompt
  `e5ded258-1f3d-4a6e-874a-ba89ce1e6a83`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: `RESULT FAIL` at node 89 `OTR_MetaBriefImagePromptGen`; the canonical
  all-visualizer policy (`viz_mxc_cpu`, `viz_mxc_mandala`, `viz_camera`) still
  resolved and used the writer visual-prompt path, and c03 failed the
  story-consistency gate even though no downstream video role consumed an init
  image. No OBS final was produced.
- root cause: effective-engine / `accepts_still` capability was checked only at
  downstream image dispatch. MetaBrief and ShotLock entered visual-authoring
  paths before that guard, so a proven no-consumer policy could still spend or
  fail in an upstream writer call.
- fix: make dispatcher-owned effective per-role still capability (including
  force-map and radio redirects) the shared authority. A complete all-false
  map returns an explicit empty v1 payload and bypasses MetaBrief/ShotLock
  writer resolution; mixed policy omits only roles proven procedural and keeps
  unknown roles conservative upstream. The dispatcher renders only roles
  proven to consume an init image and fails loudly for an unproven object role.
- verify idea: `test_roles_requiring_stills_needs_a_complete_resolvable_policy`,
  `test_meta_brief_all_visualizers_bypass_prompt_authoring`,
  `test_meta_brief_node_bypasses_before_writer_resolution`,
  `test_meta_brief_mixed_policy_authors_only_proven_consumer_roles`,
  `test_dispatcher_refuses_image_render_without_proven_consumer`,
  `test_dispatcher_preserves_proven_role_when_another_slot_is_unresolved`,
  `test_dispatcher_rejects_explicit_unknown_object_role`,
  `test_dispatch_skips_stills_for_all_visualizer_episode`, and
  `test_shotlock_all_visualizers_skip_writer_visual_directives`; then rerun the
  canonical bank and require RESULT SUCCESS, no image objects or visual-writer
  call, ledger, episode asset, `obs_publish OK`, and OBS final.
- bible-worthy: yes -- live failure plus reusable effective-consumer-capability
  contract and executable coverage
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-20 -- Sci-Fi Codex P3 typed repair silently lost its contract

- promotion: BUG-11.50
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `ffc354cc-febf-4ada-9ebd-2e3d27a057e8`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: the base P3 `RadioScoreV4` had five music cues (maximum three),
  then its typed repair logged `PROMPT_GUARD: Truncated 5273 -> 4592`
  (`context_cap=8192`, `max_new_tokens=3600`) and returned the request-shaped
  root `{artifact_inputs, validation_error}`. Strict validation correctly
  rejected the envelope; no ledger, media asset, or OBS final was produced.
- root cause: P3/P3_rewrite reserved a flat 3,600 output tokens without
  accounting for the fixed 8,192-token local context or the full failed-score
  repair prompt. The generic repair payload duplicated the original request,
  so the token wrapper left-truncated its leading system/schema/rules before
  calling Gemma. The model did not receive the contract it was expected to
  repair and echoed trailing input material instead.
- fix: calculate the RadioScoreV4 reservation from requested words and locked
  beat count; at the observed 120-word/12-beat case it reserves 2,800 tokens,
  leaving 5,392 input tokens. Mark P3 and P3_rewrite as `prompt_must_fit` so
  a future oversize graph fails before generation, and send P3 repair context
  as compact tagged references (failed score, rejection, locked graph,
  advisory; plus accepted score/review only for P3_rewrite) rather than a
  copyable JSON request envelope.
- verify idea: assert `8192 - radio_score_output_token_budget(120, 12) >=
  5273`, assert P3/P3_rewrite both use that dynamic budget and
  `prompt_must_fit=True`, assert the P3 repair prompt carries no
  `original_request`/`artifact_inputs` JSON envelope, then rerun the same
  canonical bank and require P3 clearance, zero all-visualizer image objects,
  saved ledger, episode asset, `obs_publish OK`, and final OBS file.
- bible-worthy: already promoted as BUG-11.50; added OTR executable coverage
  for the repair-prompt dimension.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-21 -- Sci-Fi source P0 could exhaust its bounded output before producing JSON

- promotion: BUG-11.50
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `b5341847-4635-4eeb-a5b8-4660136b0d78`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 selected a valid long RSS source, then both base/structural
  attempts ended at `generated_tokens=2000` with incomplete JSON
  (`prompt_tokens=2455`, `max_new_tokens=2000`). Its typed repair returned an
  otherwise literal index with `tone: []`, which strict validation correctly
  rejected because `tone` is model-owned scalar prose. No P1/P3 artifact,
  ledger, media asset, or OBS final was produced.
- root cause: FactIndexV4/FragmentDossierV4 allowed up to twelve facts,
  entities, and numbers while claims, quote spans, numeric-token lists, and
  several strings had no finite serialized surface. A fixed P0 output ceiling
  could therefore be too small by construction. The generic typed repair also
  replayed a copyable original-request envelope and did not explicitly require
  scalar `tone`, recreating the context/shape failure class from PBUG-20.
- fix: introduce one shared Sci-Fi P0 evidence contract for Codex, Gemini, and
  Sonnet: 1-6 facts, 0-4 entities/numbers, one literal span per fact/entity,
  bounded claim/name/quote/token/tone fields, and compact story-usable prompt
  seams. Reserve 2,800 output tokens for FactIndexV4 and 3,000 for Sonnet's two
  extra root strings, journal the bounds/source-size receipt, and retain
  `prompt_must_fit=True`. P0 repairs now receive tagged failed-artifact,
  rejection, source evidence, digest, and allowed-field references only; they
  explicitly require the exact artifact root and one nonempty scalar `tone`.
  Python never substitutes a tone value.
- verify idea: reject seven facts, a second evidence span, an overlong quote,
  and `tone: []`; assert all three source lanes use the shared bounded
  reservation and compact repair context without `original_request` or
  `artifact_inputs`; then rerun the same canonical Codex bank through P3 and
  require zero all-visualizer image objects, saved ledger, episode asset,
  `obs_publish OK`, and OBS final.
- bible-worthy: yes -- a live bounded-output failure with reusable
  model-facing artifact-surface and compact-repair requirements; promoted as
  BUG-11.50 with cross-lane executable coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-22 -- Sci-Fi Codex P3 whole-score transport exhausted its model window

- promotion: BUG-11.50 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `f26b727b-42c8-40d6-b3ee-001d7a869cf9`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: the initial bounded direct-score correction was rerun live at
  prompt `edbbac48-9aa8-4907-8086-f63134604604` (same Gemma E4B creative +
  Mistral-Nemo technical pairing). P0-P2 cleared, but P3 again produced no
  decodable top-level JSON on its 2,900-token base, lower-temperature, and
  typed-repair calls. The canonical queue ended `RESULT FAIL` before ledger,
  episode, image/still, or OBS work.
- root cause: finite `RadioScoreV4` bounds removed the original unbounded
  schema defect, but the model still had to serialize duplicate mechanical
  graph state it did not author: advisory rows, scene/shot/beat/line IDs,
  parents, order, speakers, roles, and canonical cue anchors. The direct
  whole-score transport remained too wide for the live model even with a
  2,900-token cap; increasing the cap alone would recreate repair-window risk.
- fix: replace direct P3/P3-rewrite score emission with bounded
  `RadioScoreDraftV4` plus a fail-closed compiler. The model authors only
  creative surface, local shot/cast/line-count/fact/cue choices; Python derives
  only uniquely determined mechanics from accepted P0/P2/advisory state and
  revalidates fresh `RadioScoreV4`. The three-call ladder restarts from trusted
  context after two decode failures and uses minified parsed semantic repair
  only for complete invalid drafts. Exact wrapped-root handling remains. The
  real Gemma tokenizer measured a max-width draft at 1,418 output tokens;
  reservation is 1,647 (`+ max(128, 15%) + 16`). Measured base, clean restart,
  semantic repair, rewrite base, and rewrite repair prompts were respectively
  1,110, 1,167, 2,664, 2,614, and 4,165 tokens, all within 8,192 with the new
  reservation.
- verify idea: compiler tests reject dynamic advisory/count/shot/cast/fact/cue
  defects and preserve `compile(project(score))` rewrite structure; actual
  tokenizer tests cover all six envelopes and require prompt plus reservation
  <=8,192; default schema injection remains unchanged for other passes; then
  rerun the canonical 120-word bank and require all-visualizer zero image
  objects, saved ledger, episode asset, `obs_publish OK`, and final OBS file.
- bible-worthy: yes -- BUG-11.50 now explicitly permits a compact
  authoring-draft/compiler boundary when it removes deterministic graph
  serialization rather than merely papering over absent bounds.
- status: ROOT REPLACEMENT IN TREE; LIVE REVERIFY PENDING

## PBUG-20260712-23 -- Sci-Fi P0 generic string clamp stranded an exact oversized source span

- promotion: BUG-11.50 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `81e0b0c9-2f20-4085-9fd0-e7f8034f75da`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 selected an eligible RSS source, but its first fact returned an
  exact literal `full_text` quote wider than the 240-character P0 source-span
  cap. The generic tolerant validator word-clamped the quote while retaining
  the model's old end coordinate; the literal-span validator correctly rejected
  the synthetic mismatch. The typed repair repeated the same oversized literal,
  so the canonical queue ended `RESULT FAIL` before P1/P3, ledger finalization,
  media, or OBS work.
- root cause: `repair_literal_source_metadata` safely reindexed exact quoted
  source text, but it first required the raw artifact to satisfy Pydantic's
  quote cap. It therefore could not repair an exact source quote that exceeded
  that cap. Meanwhile P0 still used the generic string clamp intended for
  compatibility fields, which may shorten source metadata at a word boundary
  without recomputing `end`. The shared P0 helper shape made the defect possible
  in Codex, Gemini, and Sonnet.
- fix: disable generic overlong-string clamping at all three Sci-Fi P0
  boundaries. Extend the shared metadata-only repair so it accepts an oversized
  quote only after proving the *entire raw quote* occurs literally in one legal
  source field, rehomes/reindexes it under the existing ambiguity rules, then
  replaces only the quote with that coordinate's exact finite source prefix and
  recomputes `end`. Claims, tone, and all nonliteral/ambiguous text remain
  model-owned and fail through the bounded typed-repair ladder.
- verify idea: exact oversized source quote repairs in one P0 call to the
  schema cap with byte-identical claim; an oversized quote with invented text is
  rejected; all three Sci-Fi P0 call sites disable generic clamping. Run focused,
  full, and Bug Bible gates, then rerun the same canonical Codex leg through
  P3/ledger/OBS proof.
- bible-worthy: yes -- repeatable bounded-source-metadata repair class at the
  shared Sci-Fi P0 fan-out; promoted as an executable BUG-11.50 extension.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-21 -- P1 repeated an overlong Aion dramatic question

- renumbered: 2026-07-15 baseline -- originally logged as PBUG-20260713-10,
  colliding with the P9-audit entry below (which keeps -10: it is the id cited
  by the contract-gap docs and commit `3a98a6f1`). BUG_BIBLE.yaml currently
  carries two `legacy_id: PBUG-20260713-10` rows (~:4357/:4379); at the next
  operator fan-out, re-point the BUG-11.54 row's legacy_id to PBUG-20260713-21,
  and verify the acronym-union rule (~:4357, also citing -10) against its true
  source entries (the acronym PBUGs -07/-09) -- the P9-audit entry that owns
  -10 is not an acronym bug.
- promotion: BUG-11.54
- surfaced: canonical 120-word `scifi_codex` smoke, prompt
  `2147f181-8821-461f-a5dc-8cb9bfefd48c`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P1 returned a question above the 160-character schema cap; the
  typed repair repeated the invalid field and exhausted the ladder before P2.
- root cause: the repair prompt described the cap but relied on a second model
  call to shorten authored text, so a reasoning model could copy the rejected
  question unchanged.
- fix: add a deterministic, word-boundary P1 repair only for overlong question
  or consequence fields, preserving the established semantic repair turn for
  ending-only overflow and rejecting malformed roots.
- verify idea: unit-test bounded shortening and run the canonical 120-word
  combination through P5, ledger, episode, `obs_publish OK`, and final OBS
  existence.
- bible-worthy: yes -- bounded typed repair must not depend on a model obeying
  a repeated length instruction; promoted as executable BUG-11.54.
- status: FIXED IN TREE; 120-WORD LIVE PASS

## PBUG-20260712-24 -- Sci-Fi Codex P3 compact draft omitted nested literal semantics

- promotion: BUG-11.38 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `fab1bbbe-cfc1-484b-8f5b-61dfc296de6e`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 repaired and P1/P2 cleared, but P3's base compact draft emitted
  numeric `arc_phase` values copied from advisory word centers and invented
  descriptive cue IDs (`TensionBuild`, `EquityStrain`, `DecisionPoint`). Its
  typed repair reduced unrelated overlength errors but repeated those seven
  invalid nested values, so the canonical queue ended `RESULT FAIL` before
  P4/P5, ledger finalization, media, or OBS work.
- root cause: P3 correctly omitted the large full Pydantic schema to preserve
  its measured 8,192-token repair window, but the compact model-facing contract
  named `arc_phase` and `cue_id` only by field and length. It did not preserve
  their nested literal/type semantics. The local model therefore treated
  `arc_phase` as a word-band number and treated `cue_id` as a creative title;
  the same incomplete surface was reused for typed repair.
- fix: make the shared compact P3/P3-rewrite base and repair contract state that
  `arc_phase` is a short narrative JSON string, never a number/word count/center
  or percentage, and enumerate `music_open`, `music_inter`, and `music_close`
  as the only cue IDs. Creative cue naming stays in `description`. No Python
  normalization is permitted: arc labels and cue choice remain model-authored.
- verify idea: drive the live failure shape (numeric arc plus descriptive cue
  ID) through base then typed repair and assert the accepted score returns only
  after the repair sees both literal rules; retain actual-tokenizer fit tests
  for base/restart/semantic-repair/rewrite envelopes. Then rerun the same
  canonical Codex leg through P3/ledger/OBS proof.
- bible-worthy: yes -- model-visible compact schemas must retain nested type and
  literal semantics, not merely field names and maximum lengths; promoted as an
  executable BUG-11.38 extension.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260712-25 -- Sci-Fi Codex P3 full typed repair repeated local prose overflow

- promotion: BUG-11.42 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `4b19f3ed-bd28-4f84-9b81-5fcddfb89dc0`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 repaired and P1/P2 cleared. P3 then returned a complete compact
  draft whose only surfaced defects were four model-authored strings over their
  finite caps. Its normal full typed repair shortened one field but repeated
  three over-cap fields, exhausting the bounded ladder before P4, ledger,
  media, or OBS work.
- root cause: generic clamping was correctly disabled at the author-owned P3
  boundary, but the only remaining repair transport resent the complete draft.
  That invited the local creative model to reauthor already-valid graph and
  prose surface instead of making the one bounded shortening decision. Pydantic
  length errors could also conceal a compiler-only defect, so a naive text patch
  would have incorrectly treated every string-only error report as local. A
  lazy scheduler wrapper also hid remote-provider markers, which would have
  let a remote slot take the local-only route; generic completion reporting
  could then mislabel a rejected direct patch as a decoded accepted draft.
- fix: on local P3/P3-rewrite only, derive a maximum-six exact whitelist of
  over-cap authored leaves from the real Pydantic locations; preflight a clone
  through the strict draft/compiler/signature/graph boundary; then request one
  strict one-for-one author-owned shortening patch at the common typed-repair
  temperature. Merge only through trusted locations, revalidate the complete
  draft, and record the real patch call in the existing P3 receipt. Unknown,
  broad, hidden-graph, malformed, or still-over-cap repairs fail closed. Remote
  OpenRouter slots retain the existing same-slot full repair because their
  virtual context metadata is not an exact tokenizer preflight; no model/router
  fallback or substitution is introduced. The scheduler carries exact catalog
  transport capability into its lazy closure and relays OpenRouter JSON-object
  mode; direct-patch receipts own their parse/schema/contract truth so the
  generic ladder cannot overwrite it with an empty factory result.
- verify idea: cover every eligible P3 leaf, mixed/broad errors, hidden compiler
  defects, malformed patch roots, unselected-field preservation, local base and
  rewrite receipt success at `.10`/512, a scheduler-wrapped remote same-slot
  JSON-mode full repair, truthful malformed-patch receipts, and actual Gemma
  E4B six-target prompt plus reservation fit. Run focused,
  full, Bug Bible, pack/registry, and canonical workflow gates, then rerun the
  fresh canonical Codex leg through ledger/OBS proof.
- bible-worthy: yes -- a live bounded-patch admission: localized authored prose
  needs one-for-one model replacement plus complete preflight/merged validation,
  never Python clipping or a broad retake. Promoted as executable BUG-11.42
  extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260712-26 -- Strict Sci-Fi RSS admission starved eligible inline bodies

- surfaced: canonical 120-word `scifi_codex` GUI/API run, prompt
  `59b9baa5-046f-4e4c-b313-8d18223ea716`, 2026-07-12
- symptom: the live feed pool contained qualifying literal inline RSS bodies,
  but the strict selector body-resolved only the first ten headline-ranked
  candidates. All ten fell back to thin summaries, so the writer failed before
  P0 with `No science RSS candidate met the v4 source floor`.
- root cause: the selector enforced the 400-character/80-word/12-unique-token
  floor only after its bounded body-fetch slice. It had no eligibility-aware
  ordering, and its legacy `rss_full > 300` shortcut did not match the stricter
  Codex/Gemini/Sonnet envelope contract.
- fix: define one stdlib-only v4 RSS predicate and route strict selection plus
  all three lane envelopes through it. In strict mode, stable-partition already
  qualified inline RSS bodies ahead of unresolved candidates while preserving
  prior rank inside each partition; admit inline text only through the shared
  predicate, retain the ten-candidate cap and URL-scrape path, and leave legacy
  non-strict behavior unchanged.
- verification: focused admission coverage passed; full Windows suite
  `7843 passed, 31 skipped, 1 xfailed`; Bug Bible `17 passed, 12 skipped,
  3 xfailed`. Canonical prompt `14af0787-f45c-4caa-8737-92d057855653`
  logged `Strict v4 admission prioritized 13/40`, resolved ten bodies with
  `10/10 candidate(s) passed content floor`, selected a 4,825-character MIT
  article, and crossed P0 into P1. A separate OpenRouter reasoning-capability
  error then stopped the episode; it does not reopen source admission.
- bible-worthy: yes -- a bounded selector must apply hard downstream
  eligibility before its truncating candidate slice and share the exact
  predicate with the accepting envelope. Promotion remains a separate review.
- status: FIXED AND LIVE VERIFIED

## PBUG-20260713-01 -- OpenRouter global reasoning-off rejected by mandatory endpoint

- promotion: BUG-12.53
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `14af0787-f45c-4caa-8737-92d057855653`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-12/13
- symptom: strict RSS admission and P0 cleared, then P1 stopped before creative
  generation with HTTP 400: `Reasoning is mandatory for this endpoint and
  cannot be disabled.` The process-wide `OPENROUTER_REASONING_EFFORT=none`
  had been sent unchanged to `aion-labs/aion-3.0-mini`.
- root cause: the OpenRouter cache discarded the live `/models` reasoning
  contract, and request construction applied one global effort to every model.
  A saved slug absent from the stale June cache also had no bounded way to learn
  the provider's precise mandatory-capability response.
- fix: retain sanitized per-model reasoning metadata in catalog schema v2 and
  resolve the global setting against the selected model. A mandatory model uses
  its lowest declared enabled effort (or `low` when the catalog omits effort
  levels), while ordinary models retain explicit `none`. For stale/cold cache
  only, the exact mandatory-reasoning 400 triggers one same-model corrected
  call, remembers the capability for the process, and does not consume the
  transient retry budget; every other 400 remains fail-fast.
- verify idea: prove proactive metadata resolution, exact-400 learning with
  zero transient retries, subsequent-call reuse, unchanged ordinary-model
  `none`, generic-400 fail-fast, and live catalog retention of Aion's
  `mandatory: true`; then rerun the same canonical 120-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: promoted as BUG-12.53 with executable OTR coverage and shared
  Bug Bible regression pins.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-02 -- Remote P3 whole-draft repair repeated ten local prose overflows

- promotion: BUG-11.42 extension
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `b98bef70-d5ae-4c60-9402-ce3adeccf26e`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: RSS admission, P0, P1, and P2 cleared. P3 returned an otherwise
  complete compact draft with ten `string_too_long` authored fields. The normal
  remote whole-draft typed repair fixed only two, repeated eight, and exhausted
  the ladder before ledger/media/OBS work.
- root cause: PBUG-20260712-25's one-for-one authored-text repair was restricted
  to exact-tokenizer local slots and six targets. OpenRouter already sent full
  messages or failed loudly, but its explicitly known transport was still forced
  through the broad retake. The live ten-target shape also exceeded the local
  patch schema, so merely enabling the remote marker would have remained dead.
- fix: declare behavioral patch transport explicitly on the lazy scheduler:
  exact-tokenizer local, full-message/fail-loud OpenRouter, or ineligible.
  Expand the one-call patch envelope to 12 targets/1024 output tokens, prove its
  actual tokenizer envelope, preserve complete preflight and merged validation,
  and record the chosen transport. OpenRouter honors the patch's strict output
  cap even when its global minimum-output floor is raised; JSON mode, mandatory
  reasoning, cost guard, routing, retries, and terminal provider errors remain
  in the shared backend. Thirteen-plus or any mixed/hidden/unproven shape keeps
  whole-draft repair.
- verify idea: reproduce the ten-target artifact with an explicit OpenRouter
  callable and require exactly base plus one patch, all ten exact paths,
  json_object mode, 1024 tokens, full-message transport receipt, and complete
  draft acceptance. Prove 12-row prompt/response fit, 13 targets retain broad
  repair, excluded/unmarked transports remain ineligible, and a raised global
  remote floor cannot inflate the strict patch budget. Then rerun the same
  canonical combination through ledger, episode asset, `obs_publish OK`, and
  final OBS existence.
- bible-worthy: yes -- bounded semantic repair eligibility depends on a proven
  transport behavior, not locality alone; promoted as executable BUG-11.42
  extension coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-03 -- P4 repair replaced a valid pass review with diagnostic-shaped JSON

- promotion: BUG-11.38 extension
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `a43a3e77-2ba4-4420-a4e1-1982bf0448cc`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P3 cleared. P4 returned the correct three-field review
  with `verdict: "pass"` and an empty string issue list, but its rationale was
  one character over the 240-character cap. The generic typed repair changed
  the verdict to the invalid literal `fail` and changed issues into objects
  shaped like validation diagnostics, then exhausted the ladder before
  ledger/media/OBS work.
- root cause: the compact P4 seam named field lengths but did not state the
  exact verdict literals or that issues are strings. Its generic repair turn
  also supplied the entire score-shaped original request beside Pydantic error
  diagnostics, allowing input and diagnostic shapes to compete with the small
  output contract.
- fix: repeat an exact StructureReviewV4 contract at the base and repair
  boundaries: exactly `verdict`, `issues`, and `rationale`; only `pass` or
  `rewrite`; a flat list of at most six bounded strings; and one bounded
  rationale. Give the repair only the failed review and bounded rejection,
  require valid fields to remain unchanged, and explicitly forbid copying
  error codes/messages/shapes. The model still authors any shortening; Python
  does not clip review prose.
- verify idea: inject a correct `pass` review whose rationale is 241
  characters, capture both calls, and require the repair to preserve `pass`
  and empty string issues while shortening only the rationale. Assert both
  system prompts carry the literal/type contract and the repair input omits
  the accepted score/original request. Then rerun the same canonical 120-word
  combination through ledger, episode, `obs_publish OK`, and final OBS
  existence.
- bible-worthy: yes -- compact typed contracts must preserve literal and item
  type semantics at every repair boundary; promoted as executable BUG-11.38
  extension coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-04 -- P3 patch aimed at the hard cap and crossed it

- promotion: BUG-11.42 extension
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `94a11e73-c7f8-47a1-b929-37c1cf7d63d6`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P2 cleared. P3 selected the bounded three-leaf text
  patch, but its patch schema failed. The outer ladder logged the stale base
  draft head, obscuring the direct patch error. Three exact live Aion probes
  then reproduced a premise replacement just over its 144-character cap.
- root cause: the request exposed the strict schema cap as the model's writing
  target, and its `original_text` field looked like an output value to copy.
  Approximate character counting crossed the edge; Aion also copied the
  over-cap source unchanged. The receipt collapsed every patch-schema error
  into `patch_root`.
- fix: expose only a conservative 75% model-facing `max_chars` for each leaf;
  keep the larger immutable schema cap private to validation so the model
  cannot anchor on the rejection edge. Root the input at `rewrite_tasks`, name
  the source `source_to_shorten`, and use one concise contract that forbids an
  unchanged copy. Never Python-clip authored prose. Classify replacement-
  string overflow separately without recording rejected prose.
- verify idea: require a model-facing 54-character ceiling for a scene whose
  private schema cap is 72, with no hard-cap field in the request. Require the
  action-shaped input and prove three exact live Aion probes pass. Inject a
  145-character replacement for a
  144-character target and assert a `replacement_over_schema_cap` receipt with
  no prose retention. Reprobe live Aion, then rerun canonical 120 through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- bounded authoring needs safety margin below its strict
  rejection cap; promoted as executable BUG-11.42 extension coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-05 -- P1 repair copied an overlong ending direction unchanged

- promotion: BUG-11.38 extension
- surfaced: canonical 42-word `scifi_codex` smoke, prompt
  `d1313994-753c-4748-bf8c-a4e09e15d8fe`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: RSS admission and deterministic P0 repair cleared. P1 returned the
  correct three-field DramaticQuestionV4 shape, but `ending_direction` exceeded
  its 120-character cap. The typed repair copied the same overlong value
  unchanged and exhausted the ladder before ledger/media/OBS work.
- root cause: P1 fell through to the generic graph-artifact repair contract.
  Although its generated JSON schema carried the numeric constraint, the
  authoring instruction did not repeat the three exact keys, per-field caps,
  or a safe rewrite target. Its repair input also repeated the full original
  request and fact index beside the tiny failed artifact, obscuring the only
  required edit.
- fix: give DramaticQuestionV4 its own compact repair boundary. Supply only the
  parsed failed question and bounded rejection; repeat the exact three root
  keys and hard caps; preserve valid fields byte for byte; require each
  rejected overlong field to be rewritten rather than copied or mid-word
  clipped; and give rewritten fields a conservative 75% authoring ceiling.
  Python never clips or authors the prose.
- verify idea: inject a valid question/consequence plus an overlong ending,
  require exactly one model-authored repair, and assert the repair prompt names
  the 160/160/120 hard caps plus 120/120/90 rewrite margins. Assert the repair
  payload omits the original request and fact index and preserves valid fields.
  Then rerun the same canonical 42-word combination through ledger, episode,
  `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- a tiny authored repair must repeat its exact contract,
  isolate the failed artifact, and target below the rejection edge; promoted
  as executable BUG-11.38 extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-06 -- P3 repair fixed total beats by overflowing one scene

- promotion: BUG-11.38 extension
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `a2b76223-c4be-49e3-945f-9fd1895a33a3`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P2 cleared. P3's base draft had fewer flattened beats
  than the locked six-row advisory. Its semantic repair restored all six beats
  but placed them in one scene, violating RadioScoreDraftV4's maximum of four
  beats per scene, and exhausted the ladder before ledger/media/OBS work.
- root cause: the compact schema said each scene had one to four beats, while
  the accepted advisory lived only in the input context. Neither the base nor
  repair instruction explicitly bound the locked global beat count to the sum
  of the scene-local arrays or derived the minimum scene count. The model fixed
  the named total-count rejection without preserving the independent local cap.
- fix: derive a bounded topology instruction from the accepted advisory before
  every P3/P3_rewrite call. State the exact locked flattened beat total, repeat
  the four-beat per-scene maximum, derive the minimum scene count, and require
  distribution across scenes. The same instruction is carried by base,
  restart, semantic repair, and rewrite boundaries; Python still derives only
  canonical mechanics after a complete valid authored draft.
- verify idea: use a six-row advisory and a schema-valid one-scene/four-beat
  draft, then require the repair to return a valid two-scene/six-beat draft.
  Assert both captured system prompts name exact total six, local maximum four,
  and minimum two scenes. Re-run the same canonical 42-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- compact nested contracts must explicitly relate locked
  global cardinality to local collection caps at every repair boundary;
  promoted as executable BUG-11.38 extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-07 -- Source-grounded acronym was rejected as shouting

- promotion: BUG-11.51
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `3627b61a-8174-43e5-95f1-1a0c8f0269ec`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P4 and P3_rewrite cleared, including the repaired compact
  topology. P5 used the acronym `RIO` from accepted source evidence. The spoken
  hygiene validator rejected it as an all-caps lexical word; the model repair
  merely moved the same grounded acronym to another line and exhausted the
  ladder before ledger/media/OBS work.
- root cause: spoken hygiene used a blanket uppercase-token regex. It had no
  connection to the accepted FactIndexV4 evidence, so a legitimate acronym and
  ungrounded shouting were indistinguishable at every script boundary.
- fix: derive the exact set of uppercase lexical tokens only from the literal
  source spans already accepted for facts, entities, and numeric evidence.
  Thread that immutable allowlist through P5, P7, P9, and final spoken
  validation. Continue rejecting every all-caps lexical token absent from the
  accepted evidence; do not lowercase or rewrite authored dialogue in Python.
- verify idea: accept `RIO` when it is present in a literal accepted fact span,
  reject `STOP` in the same line, and prove the source-grounded set reaches all
  script validators. Re-run the same canonical 42-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- lexical hygiene must distinguish source-grounded
  acronyms from ungrounded shouting at the validator boundary; promoted as
  executable BUG-11.51 coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-08 -- P3 rewrite overflowed more prose leaves than its patch envelope

- promotion: BUG-11.42 extension
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `3c4f6e67-8dda-47e8-802e-a37c6359e1b1`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P4 cleared and the repaired P3 topology held. The
  P3_rewrite response preserved structure but reauthored broadly, producing 13
  prose leaves just over their strict caps. That exceeded the proven 12-target
  local patch envelope, so the normal full-draft repair repeated all 13
  overflows and exhausted the ladder before script, ledger, media, or OBS.
- root cause: the base/rewrite authoring contract exposed every private schema
  rejection edge as the model's writing target. The rewrite instruction also
  allowed every creative prose leaf to change even when the review required a
  narrower correction. The bounded patch already used conservative targets,
  but it was only a downstream backstop after the broad overflow was created.
- fix: make conservative 75% prose ceilings the only model-visible limits at
  every P3 base, restart, full-repair, and rewrite boundary while retaining the
  larger immutable Pydantic caps privately. Require P3_rewrite to change only
  prose directly necessary for the review and preserve every other prior prose
  leaf byte for byte. Keep the proven 12-row patch envelope and fail-closed
  validation; do not expand an arbitrary capacity or Python-clip authored text.
- verify idea: capture base, full-repair, and rewrite system messages and assert
  they expose only the safe 48/108/60, 42/54/90, 48/21, and 60/90 ceilings.
  Require rewrite to preserve non-target prose byte for byte while retaining
  existing structure locks. Re-run the same canonical 42-word combination
  through ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- safety margin belongs at every authoring boundary, not
  only the local patch after overflow; promoted as executable BUG-11.42
  extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-09 -- P2 rejected an acronym-bearing canonical character name

- promotion: BUG-11.52
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `7997800e-b0f5-4201-ae2d-193a899ac6f4`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 and the compact P1 repair cleared. P2 returned `AI Unit 7`; its
  repair correctly removed the digit as `AI Unit Seven`, but the same validator
  rejected the legitimate `AI` acronym and exhausted the ladder before P3,
  script, ledger, media, or OBS.
- root cause: the cast-name grammar accepted only `[A-Z][a-z]+` tokens. The
  repair instruction said only "Title-Case," so the model fixed the visible
  numeric defect while the validator's hidden blanket acronym ban remained.
- fix: accept at most one short 2-3-letter acronym token inside a name that also
  contains at least one normal Title-Case word. Continue rejecting digits,
  lowercase labels, empty tokens, multiple acronyms, and all-uppercase full
  labels. State the exact grammar and `AI Unit Seven` example at the P2 repair
  boundary; do not rewrite character names in Python.
- verify idea: accept `AI Unit Seven` and `Dr. Amelia Hart`; reject `AI Unit 7`,
  `AI UNIT`, and lowercase names. Reproduce base `AI Unit 7` followed by the
  one-call authored repair and assert its exact model-facing grammar. Re-run
  the canonical 42-word combination through ledger, episode, `obs_publish OK`,
  and final OBS existence.
- bible-worthy: yes -- lexical validators must state and implement the same
  bounded acronym-aware name grammar; promoted as executable BUG-11.52.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-10 -- P9 audit blocked on a defect its only repair route could not touch

- surfaced: canonical 42-word `original_codex56sol` run, prompt
  `e0a03830-aa18-42c4-8c47-89c6cff51a46`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13. Logs
  `tmp\scifi_42_aion_original_pair_harness.out.log` and
  `tmp\scifi_42_aion_final_server.log`.
- symptom: the whole pipeline authored, validated, and grounded a complete
  script, then died at the last gate with `final contract audit rejected the
  script without actionable grounded findings` after 570s. The audit had
  returned `accepted=false` whose only findings named the manifest and a clue
  id (`manifest.lines[4].clue_ids`, `item_id=c4`) -- never a spoken line.
- root cause: the P9 seam asked the model to audit the script AND the manifest,
  truth map, and grounding contract, but the only correction the pass owns is a
  spoken-line retake, and `_audit_blocks` accepts a finding only when it names a
  script line and quotes an exact span. A finding about a derived artifact was
  therefore simultaneously authoritative enough to reject the episode and too
  unlocatable to repair -- a guaranteed dead end. The manifest is not even
  model-owned: Python compiles it from the accepted score and `_validate_manifest`
  already proves exact clue coverage, no duplicates, and landmark order.
- fix: state and enforce the audit's blocking authority. The seam prompt and the
  P9/P9_rerun repair rules now say only a finding whose `item_id` is a
  `script.lines` line_id and whose `exact_span` is copied verbatim from that
  line may block, and that manifest/truth/grounding concerns belong in
  `warnings`. `_validate_audit_envelope` runs as the P9 and P9_rerun
  post-validator: a blocking finding that names a real script line without a
  verbatim span, a rejection carrying no blocking finding, and an acceptance
  carrying one all return to the typed-repair ladder and fail closed if it
  exhausts. `_audit_advisories` demotes findings aimed at derived artifacts --
  a mechanical classification, never a judgment of authored meaning -- and
  records them verbatim in the new `final_audit_disposition` receipt. The dead-
  end raise is gone because the state is now unreachable.
- verify idea: assert a manifest-only `accepted=false` completes the episode
  with zero retakes and an advisory receipt row; a quoted script-line block
  still triggers exactly one retake; an unquotable script-line block reaches
  typed repair; an `exact_span` array stays a typed failure and is never
  normalized into index semantics. Re-run the canonical 42-word combination
  through ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- an audit's blocking authority must not exceed what its
  repair route can re-author, and a validator that cannot be overruled must not
  be re-litigated by a model. Candidate for fan-out.
- status: FIXED at `3a98a6f1`; LIVE REVERIFIED 2026-07-13, prompt
  `28fe3cdf-e652-4db6-ab59-b7ddda6786ae` (same canonical 42-word
  `original_codex56sol` leg, Aion 3.0 Mini creative + Mistral-Nemo technical).
  `RESULT SUCCESS`, `obs_publish OK`, asset confirmed on disk (84,092,039 bytes,
  `output\otr\obs\signal_lost_waiting_room_whispers_20260713_122501_silent_procgen_blended_captioned_with_credits_final.mp4`).
  The live receipt proves the seam, not just the gate: P9 ran ONCE with no
  retake and no repair (`call_journal` `P9x1`), and the audit model met the very
  situation that killed the prior run -- a concern it could not act on -- and
  self-classified it into `warnings` verbatim: "The 'resolution_anchor' in the
  grounding_contract is missing. This is a compile-time issue and cannot be
  corrected during this pass." `accepted=true`, `findings=[]`,
  `blocking_script_findings=0`. That warning is itself a model misread -- the
  deterministic grounding validator proved the anchor IS spoken on the closure
  line -- which is exactly why a model's opinion about a Python-owned artifact
  must never hold a blocking vote.

## PBUG-20260713-11 -- P1 slate lost a clue per object with no rule to repair it

- surfaced: canonical 42-word `original_codex56sol` reverify of PBUG-20260713-10,
  prompt `0bca5788-9da4-4d23-b7da-c49984956bec`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13. Log `tmp\p9_verify_42_server.log`.
- symptom: the run died 121s in, before P9 was ever reached:
  `P1 failed ... after 2 attempt(s)`. Three of four possibility cards returned a
  two-entry `clue_plan` against `Field(min_length=3)`. The typed repair returned
  the same defect (4 errors, then 3), so the ladder exhausted.
- root cause: the clue-per-lost-object contract existed only as a bare pydantic
  `min_length=3`. The P1 seam never stated it, `_validate_slate` never checked
  coverage, and `_repair_rules` had no `P1` branch at all -- so the repair prompt
  carried the raw pydantic text plus a generic "repair only the typed contract
  error" and no rule telling the model to author the missing clue. The model
  read its own merged two-object clue as correct and kept it. The schema minimum
  is also not the real invariant: one clue per lost object means a four-object
  draw needs four clues, which `min_length=3` would silently pass.
- fix: state the same contract at all three surfaces. The seam and the new
  `_repair_rules("P1")` branch require 4-6 possibilities, verbatim immutable
  fields, and one distinct clue for EVERY lost object, in order, never merging
  two objects and never dropping a card to repair another. `_validate_slate` now
  derives the required count from the accepted draw
  (`len(clue_plan) >= len(draw.lost_objects)`) and reports the exact shortfall.
  Python never authors a clue: a clue is story, so the defect returns to the
  model and fails closed if the ladder exhausts.
- verify idea: assert a four-object draw rejects a three-clue card with the exact
  shortfall message; assert the P1 repair rules and seam both state the
  per-object rule; drive a runner where the base slate is clue-short and the
  authored repair restores it. Re-run the canonical 42-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- a cardinality invariant that lives only in a schema
  minimum, with no matching prompt rule and no repair branch, is an unrepairable
  contract. Same class as PBUG-20260713-10. Candidate for fan-out.
- status: FIXED at `58983363`; LIVE REVERIFIED 2026-07-13, prompt
  `28fe3cdf-e652-4db6-ab59-b7ddda6786ae` (same leg and models). P1 passed on its
  FIRST attempt with no repair rung (`call_journal` `P1x1`) and the run carried
  through to `RESULT SUCCESS`, `obs_publish OK`, and the final OBS asset.
- coverage limit (stated, not hidden): `constraint_deck.json` ships 3 draws and
  every one has exactly 3 lost objects, so live production has only ever
  exercised the 3-object case, where the coverage rule and the bare
  `min_length=3` happen to coincide. The wider-draw behaviour this fix adds
  (4+ objects) is proven by unit test only. `ConstraintDraw` permits up to 6.

## PBUG-20260713-12 -- P4 wrote `blocking` as prose because no seam ever demanded the field

- surfaced: canonical 42-word `original_codex56sol` run, prompt
  `6c73745f-e639-4d4e-b46a-1cfeb7df3716`, Aion 3.0 Mini creative + Mistral-Nemo
  technical, 2026-07-13. Log `tmp\p245_verify_42_server.log`.
- symptom: `P4 failed after 2 attempt(s) -> ValidationError: findings.1.blocking
  Field required`. The model emitted the flag as PROSE inside the detail string --
  `"detail": "All lost objects are mundane items, blocking=false"` -- and omitted
  the actual boolean field. It also wrote `field_path` with bracket indexing
  (`caller_threads[0].lost_object`).
- root cause: a seam that PHRASED A FIELD AS PROSE, in a seam shipped hours
  earlier at `81336eca`. The fair-play seam said "report them with
  blocking=false" -- which reads as text to WRITE -- so the model wrote the
  literal string into `detail` and dropped the real boolean field.
  **Correction to an earlier hypothesis in this log:** the required path WAS in
  the model-visible contract. `schema_shape_instruction` DOES emit required
  nested paths (verified: `Required paths: accepted, findings,
  findings[*].category, findings[*].detail, findings[*].blocking`). What it does
  NOT emit is any BOUND -- no `min_length`, `max_length`, or `pattern`. So the
  invisible-contract class is real but narrower than first written: it covers
  bounds (PBUG-20260713-11's `clue_plan` `min_length=3`), not required-ness.
  This entry is a phrasing defect, not an unstated-field defect.
  Separately, `_corroborated_fair_blocks` derived the collection root with
  `split(".", 1)[0]`, so the model's bracket-indexed `field_path`
  (`caller_threads[0].lost_object`) resolved to `caller_threads[0]`, matched no
  collection, and would have silently failed to corroborate a real defect.
- fix: demand `blocking` as a real JSON boolean FIELD in the P4 seam and repair
  rules, and forbid writing it as prose inside `detail`/`category`. Apply the
  same wording to P7, whose `ListenerFinding` seam had the identical prose-invite
  exposure. Add `_field_path_root`, which strips an index suffix so
  `audible_clues[0].x` and `audible_clues.0.x` name the same collection (a
  mechanical read of a coordinate, not a reinterpretation of the finding).
  Add the CLASS GUARD that is actually true: every field carrying a schema BOUND
  must have that field named in the seam or the repair rules, across all 11
  structured passes. Running it surfaced two schema caps that forbade the only
  artifact their own validator would accept, both fixed here:
  `PossibilityCard.callers` capped at 4 while a draw may carry 6 lost objects
  (one caller each), and `ScoreIntentPatch.replacements` capped at 6 while the
  plan can target 6 anchors + reveal + closure = 8 and
  `_validate_score_intent_patch` demands every planned target. Its script-side
  twin was already correctly sized at 8.
- verify idea: the bounds guard itself (every bounded field named in the
  model-visible contract, all 11 passes); assert the seam demands a boolean field
  and forbids prose in detail; assert bracket and dot coordinates corroborate
  identically; assert no cap is below what its validator demands. Re-run the
  canonical 42-word combination through ledger, episode, `obs_publish OK`, and
  final OBS existence.
- bible-worthy: yes, as two separate rules. (1) A prompt must demand a structured
  field AS a field; phrasing a key's value in prose ("set x=false") invites the
  model to emit it as prose and drop the field. (2) A schema bound is invisible
  to the model -- the shape instruction emits required paths but never
  min/max/pattern -- so every bound must be restated in the model-visible
  contract, and no bound may be lower than what its own validator demands.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-13 -- my own fair-play validator fail-closed on a benign path prefix

- surfaced: canonical 42-word `original_codex56sol` run, prompt
  `d5f66b1a-e85a-46b3-a447-7bf4f22d6e4e`, Aion 3.0 Mini creative + Mistral-Nemo
  technical, 2026-07-13. Log `tmp\final_42_server.log`. Self-inflicted by the
  validator shipped at `810369ff`.
- symptom: the new P4 retake route worked -- P4 corroborated a block, `P3_rerun`
  re-authored the truth map -- and then `P4_rerun` exhausted its ladder on my own
  error: `blocking finding for item 's4' sets field_path root 'truth_map', which
  does not own that item`. Episode dead at 408s.
- root cause: the model wrote `field_path` as `truth_map.causal_steps[...]`,
  prefixing the path with the key its input arrived under -- which is exactly
  what the payload calls it (`inputs={"truth_map": ..., "grounding_contract":
  ...}`). My root extraction took the FIRST dotted segment, got `truth_map`,
  found it owned no item, and classified a perfectly clear finding as ambiguous.
  It was never ambiguous: `item_id='s4'` resolves in exactly ONE collection. I
  had made `field_path` the identity when the `item_id` is the identity, and then
  fail-closed on a cosmetic disagreement -- turning a working retake into a kill.
- second failure, same gate (prompt `07725d30-0014-4da8-a9df-137663c3ad37`): the
  first fix classified by identity but still fail-closed when an item_id resolved
  in more than one collection. It then died on `item_id 't1' exists in more than
  one collection (caller_threads, resolution_links)` -- because `_truth_item_ids`
  keys BOTH of those collections by `thread_id`. Every thread-level finding is
  "ambiguous" BY DESIGN. The premise was wrong, not the branch.
- fix: DELETE the coordinate gate. `_truth_item_exists` asks the only question
  Python needs -- does this id name a real item anywhere in the accepted truth
  map? -- and that is all corroboration requires. The retake receives the finding
  verbatim and re-authors the whole truth map, so the owning collection cannot
  change the repair, and for a thread id it is not even a well-posed question.
  `field_path` is now a hint for the model, never a gate. The envelope keeps only
  the checks that catch the model contradicting ITSELF: accepted=true carrying a
  blocking finding, accepted=false carrying none, and a blocking finding on a
  real item with no category/detail.
- verify idea: assert every coordinate spelling corroborates the same defect --
  `audible_clues[0].x`, `audible_clues.0.x`, `truth_map.audible_clues[0].x`, a
  bare `truth_map`, an empty path, and a prose path -- including the two exact
  coordinates that killed prompts `d5f66b1a` and `07725d30`; assert `thread_id`
  resolving in two collections is not an error. Re-run the canonical 42-word
  combination through ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes, and this is the sharpest lesson of the day. **A guard that
  cannot change the outcome must not be able to cause an outage.** I added a
  coordinate gate whose verdict the repair path never reads, and it killed two
  production episodes on cosmetic disagreements while improving nothing. Before
  adding a fail-closed check, name what a caller would DO differently with its
  answer; if nothing, it is not a guard, it is a liability. The corollary: an ID
  is the identity, a path is a hint, and Python must never fail closed on a
  coordinate it can resolve -- or on one it does not need.
- status: FIXED at `fdb5c433`; LIVE REVERIFIED at prompt
  `ee452c84-7bd7-4dba-9e45-ad15a255f8ab` -- the coordinate gate no longer fires;
  P4 corroborated, `P3_rerun` ran, and `P4_rerun` reached a real verdict for the
  first time. That verdict exposed PBUG-20260713-14 below.

## PBUG-20260713-14 -- the fair-play audit graded a property its artifact cannot express

- surfaced: canonical 42-word `original_codex56sol` runs, prompts `d5f66b1a`,
  `07725d30`, and `ee452c84` (Aion 3.0 Mini creative + Mistral-Nemo technical,
  2026-07-13). The last one reached the retake verdict and failed closed with
  `fair-play audit rejected the retaken truth map`.
- symptom: P4 raised a corroborated blocking finding on **3 of 3** live runs, and
  blocked the RETAKEN truth map as well. A repair route that fires every time,
  and that a retake cannot satisfy, is not a repair route.
- root cause: the P4 seam ordered the model to check "clue-before-reveal order"
  and "audible sufficiency" -- but `AudibleTruthMap` carries **no line order and
  no reveal position**. Nothing in it is "before" anything. The audit asked the
  model to judge a property the artifact cannot state, so it manufactured a
  defect every run, and `P3_rerun` could never fix what it could not represent.
  The property is not even unowned: it is already tested where it IS
  representable, by the P7 blind listener, which reads only the pre-reveal lines
  and must infer the mundane cause. P4 was duplicating a downstream gate on an
  artifact that cannot answer it.
- fix: narrow P4's charter to what a truth map can express -- causal closure,
  separate mundane possessions, the declared device as sole cause, benign safety,
  declared-name closure, and the helpful ending. The seam now explicitly forbids
  judging clue ordering, clue timing, clue-before-reveal placement, pacing, or
  listener experience, and says those are decided later and audited downstream.
  The retake machinery is unchanged and correct; it was aimed at an impossible
  question. Also persist the retake's CAUSE: `fair_play_disposition` was writing
  a hardcoded `"corroborated_blocking_findings": 0` and dropping the initial
  blocking findings, so the retake rate could not be calibrated from the receipt
  (v2 now records `initial_blocking_findings` and a real count).
- verify idea: assert `AudibleTruthMap` declares no ordering/timing field and the
  P4 seam forbids grading one; assert the blind-listener seam still owns
  "before the declared reveal"; assert the disposition records the initial
  blocking findings and a true count. Re-run the canonical 42-word combination
  and confirm P4 accepts without a retake, through ledger, episode,
  `obs_publish OK`, and final OBS existence.
- bible-worthy: yes. **An audit may only grade properties its artifact can
  express.** If the check needs an ordering, a timing, or a coordinate the
  artifact does not carry, the model will invent a verdict and the repair cannot
  converge. Audit each artifact for what it can say, and put ordering checks
  where ordering exists. Found by asking why a repair route fired 100% of the
  time -- a repair path that always fires is a design smell, not a safety net.
- status: SUPERSEDED by the 2026-07-13 rip. P4 no longer exists: fair play is a
  deterministic contract (`_validate_script_grounding` -- the device anchor is
  spoken on a clue line before the reveal line), and P7's blind listener is gone
  with it. Closed by the green 30-word canonical leg, prompt `fb34bf4f`.

## PBUG-20260713-15 -- the score's repair prompt did not fit the window it was sent to

- surfaced: canonical 30-word `original_codex56sol` run, prompt
  `a89a46a4-196b-41ad-89fa-1bbac4bb496d`, Mistral-Nemo both slots, 2026-07-13.
- symptom: P5 rejected a valid announcer cast row filed under `char_id: "a"`, and
  then the ladder collapsed: the typed repair returned prose and a JSON fragment,
  and the syntax retry returned ONE CAST ROW as the whole score (16 validation
  errors: 12 missing top-level fields, 4 extra).
- root cause: the P5 repair prompt was **5,772 tokens against a 4,592-token usable
  window** (context_cap 8192, max_new_tokens 3600). PROMPT_GUARD truncated it, and
  what fell off the end was the instruction to return a complete artifact. The
  model answered with the last thing it could still see.
- fix: `_repair_inputs` -- the P5/P6 repair no longer re-sends the full truth map
  and grounding contract; the failed artifact already carries the graph, so it
  sends only the anchors and the clue inventory. Plus `_project_announcer_char_id`:
  an id is a coordinate, not authored content, so it is canonicalized at the
  attempt boundary and the rejection never happens.
- bible-worthy: yes. **A repair prompt that does not fit is worse than no repair.**
  Silent left-truncation deletes the contract and the model answers from the
  fragment it can still see. Measure the repair prompt against
  `context_cap - max_new_tokens`, and bound the repair context to what the failed
  artifact does not already carry.
- status: FIXED at `b286c478`; LIVE REVERIFIED by the green 30-word leg, prompt
  `fb34bf4f` (RESULT SUCCESS + obs_publish OK + asset).

## PBUG-20260713-16 -- the lane died on echoes of its own inputs

- surfaced: canonical 30-word `original_codex56sol` runs, prompts
  `efafc6fa` (P1), `5bd46a5e` and `d199a783` (P3), `55756bac` (P5), Mistral-Nemo
  both slots, 2026-07-13.
- symptom: four separate deaths, all the same shape. P1: "lost_objects and
  acoustic_device must be copied verbatim" -- the model wrote the right story
  about the right device and re-worded the field that only ECHOES the immutable
  draw. P3: wrote `timetable` for `folded timetable`, then dropped the third
  caller thread entirely. P5: wrote `closure` for the `closing` enum and the
  schema threw out the whole score.
- root cause: the lane asked a 12B model to copy immutable strings back verbatim
  into typed fields, and compared exactly. Every one of those strings was an INPUT
  Python already owned. Worse, `caller_threads` carries `min_length=2` while the
  real rule is one thread per lost object -- the schema and the contract said
  different things, and the model believed the schema.
- fix: restore the input instead of dying on the echo, but ONLY when the
  correction is FORCED (exactly one value possible):
  `_restore_slate_immutables`, `_restore_thread_lost_objects`,
  `_project_arc_phases`, `_drop_unknown_clue_ids`. P3 is now handed the caller-
  thread ROWS as data (`required_caller_threads`) rather than asked to remember
  how many to write. Ambiguity still goes back to the model.
- bible-worthy: yes. **Restoring an input is not authoring.** When a model is asked
  to echo a string the program already owns, a mismatch is a coordinate error, not
  a story decision -- restore it when the correction is forced, and never let a
  schema bound contradict the real invariant.
- status: FIXED at `b286c478` + `f3f88cb0` + `5879d6ef`; LIVE REVERIFIED by the
  green 30-word leg, prompt `fb34bf4f`.

## PBUG-20260713-17 -- a proxy gate with a repair the model could not perform

- surfaced: canonical 30-word `original_codex56sol` runs, prompts `41faff33`,
  `6fe52216`, `522e1581`, `6a325375` (P5 anchor patch) and `ec428576` (P6 anchor
  patch), Mistral-Nemo both slots, 2026-07-13.
- symptom: five deaths in the bounded anchor patch. It returned truncated JSON at
  the creative temperature; then it wrote two of three required anchors into one
  intent and failed; then, at the script level, it rewrote two of three planned
  lines and the whole batch was rejected -- the two good lines went in the bin
  with the missing one.
- root cause: TWO design errors. (1) The score's intent-anchor rule was a PROXY: a
  `line_intent` is a private note nobody hears, and the anchors are real in the
  SPOKEN SCRIPT. The proxy's repair asked the model to fit two or three immutable
  strings into one sentence. (2) The script patch asked for every planned line in
  ONE call, so a partial success was a total failure.
- fix: rip the score-intent anchor patch and its seam entirely -- the anchors are
  proven in the script, whose patch rewrites dialogue the model is good at. The
  script patch now rewrites ONE line per call, at 0.25 temperature, with a token
  budget derived from the plan. A shortened anchor ("the grille" for "ventilation
  grille") is RESTORED, since the model decided where it belongs and the exact
  wording was never its decision.
- bible-worthy: yes. **A bounded repair must ask for the unit the model can
  deliver.** Batch a repair and a partial success becomes a total failure; enforce
  a property on a proxy artifact and the repair fights the wrong object.
- status: FIXED at `f3f88cb0`; LIVE REVERIFIED by the green 30-word leg, prompt
  `fb34bf4f` -- four per-line patches, each accepted on its FIRST attempt.

## PBUG-20260713-18 -- a 30-word broadcast with a nineteen-beat score

- surfaced: canonical 30-word `original_codex56sol` run, prompt
  `717f3a4f-53e4-47fc-9992-0aaedeb5fd72`, Mistral-Nemo both slots, 2026-07-13.
- symptom: P6 returned undecodable JSON -- the script was cut off mid-object.
- root cause: the P5 seam gave the score a beat FLOOR ("at least 5 beats") and NO
  ceiling, so the model built a nineteen-beat graph for a thirty-word broadcast.
  Every beat is a line the script must then write, and the script's token budget
  was computed from the word target alone -- which knows nothing about how many
  lines exist.
- fix: `_validate_score_scale` (a broadcast of N words holds N/4 beats, floor 8,
  cap 40) and a P6 token budget derived from the MANIFEST LINES. `max_beats` is
  supplied to the score author as data.
- bible-worthy: yes. **A generation budget must be derived from the artifact that
  will be generated, not from a proxy.** A word target does not bound a line
  count; a floor without a ceiling is not a size contract.
- status: FIXED at `f3f88cb0`; LIVE REVERIFIED by the green 30-word leg, prompt
  `fb34bf4f` (6 lines, arc_verdict=strong).

## PBUG-20260713-19 -- rerolled ledger text kept stale skip state
- surfaced: canonical 30-word `shakespeare` Aion creative leg, prompt
  `bfad7f51-042b-4733-ad8f-1257442148ae`, 2026-07-13
- symptom: the deterministic freeze audit rejected `b004` because its row had
  `skip=True` and non-empty text; `OTR_CastLock` stopped the run before render
  and the queue recorded `RESULT FAIL` / `QUEUE_BLOCKED shakespeare`
- root cause: the bounded reroll wrote replacement text through
  `Ledger.update_line_text()` but that mutator updated counts without clearing
  the row's old `skip`, `tts_skip_reason`, and `reviewer_skip_reason` fields
- fix: `c25d63c6` clears stale skip metadata at the meaningful text-write seam
  and preserves it for empty/whitespace text; focused regression coverage was
  added for both transitions
- verify idea: rerun the same canonical `shakespeare` Aion/local-Mistral leg
  and require `RESULT SUCCESS`, `obs_publish OK`, duration/audio checks, and a
  non-zero asset under `output\otr\obs`; unit-test a skipped row receiving
  meaningful replacement text and an empty replacement
- bible-worthy: yes -- paired authored state must be repaired at the seam that
  writes its partner; relaxing the deterministic validator would hide a real
  render contract violation
- promotion: BUG-05.11
- status: FIXED at `c25d63c6`; live requalification pending

## PBUG-20260713-20 -- a remote model's context window was read from the static row
- surfaced: live headless OpenRouter legs (`tmp/final2_42_server.log`,
  `final3_42_server.log`, `final4_42_server.log`, 2026-07-13), each of which
  logged `[OpenRouter] load slot=A ... slug=aion-labs/aion-3.0-mini
  route=default ctx=8192 (remote, 0 VRAM)`. The same lines appear for
  `tencent/hy3:free`. Aion advertises **131,072** tokens and HY3 **262,144**;
  both ran the whole episode against an effective **8,192**.
- symptom: no crash and no warning -- a SILENT 16x-to-32x understatement of the
  usable window on every remote call. Short legs never noticed, because the
  request stayed under `8192 - prompt` anyway. The damage was latent and
  scheduled: `original_codex56sol` P6 budgets
  `240 + 160*beats + 4*target_words`, and at 720 words (beat ceiling 40) that
  is **9,520** output tokens. `fit_output_tokens` would have reduced it to
  whatever 8,192 minus the prompt left, the performance script would have come
  back cut off mid-JSON, and the ladder would have reported a bare
  `JSONDecodeError` three times -- blaming the frontier model for a defect that
  was a constant in our own catalog row.
- root cause: `OpenRouterBackend.load()` took `context_window` from the
  CuratedModel row. The two OpenRouter rows (`openrouter:slot-a|b`) are VIRTUAL
  and STATIC: one row stands in for every slug an operator may bind to it, so
  its `context_window` cannot describe the model actually selected. It carried
  `DEFAULT_CONTEXT_WINDOW = 8192` -- a LOCAL, VRAM-shaped number that is simply
  false for a remote model. The catalog cache ALREADY stored each slug's real
  `context_length` (`_slim_model`), so the truth was on disk the whole time and
  was never read.
- fix: `32e680b2` adds `resolve_context_window(slug)`, which reads the resolved
  slug's advertised `context_length` from the catalog cache. A cold/stale cache
  has no entry -- that is a genuinely unknown window, so it falls back to the
  row default and says so LOUDLY rather than inventing a confident number.
  Also at the local transport: `OUTPUT_TRUNCATED` now logs the full arithmetic
  at ERROR whenever generation stops at a ceiling that was itself a clamp (a
  reader must never have to reconstruct why the JSON was cut), and the
  unreachable PROMPT_GUARD left-slice is deleted rather than left as a dead
  lever for the next reader to repair.
- verify idea: a canonical leg with `creative_writing_model=openrouter:slot-a`
  and `openrouter_slot_a_model=aion-labs/aion-3.0-mini` must log
  `ctx=131072`, not `ctx=8192`; and a 9,520-token output request must reach the
  wire whole (`max_tokens=9520`) instead of being clamped.
- bible-worthy: yes -- a capability constant that stands in for a FAMILY of
  models describes none of them. When a per-instance truth is already cached,
  a static row default is not a fallback, it is a lie with a default value.
  Measure a budget against the window of the model that will actually serve it.
- promotion: queued for operator fan-out (overlaps the BUG-11.50 structured-
  capacity family but is distinct: that family is about artifact size, this is
  about the WINDOW the artifact is measured against).
- status: FIXED at `32e680b2`; LIVE REVERIFIED by the green 30-word
  `original_codex56sol` Aion leg, prompt `411c2f17-c05a-4af4-a6cf-c578183c072b`
  -- server log shows `slug=aion-labs/aion-3.0-mini route=default ctx=131072`,
  `RESULT SUCCESS`, `obs_publish OK`, 65.5 MB asset.

## PBUG-20260716-01 -- writer-model dropdown mislabeled on-disk models NOT DOWNLOADED + emitted a contradictory "[LOCAL HF] [NOT DOWNLOADED]" double suffix
- surfaced: live ComfyUI writer-model dropdown (OTR node INPUT_TYPES), operator-observed 2026-07-16; reproduced on the box with the Windows venv against the real HF cache (labels matched the operator report 6-for-6)
- symptom: on-disk Gemma rows shown `google/gemma-4-E2B-it [LOCAL HF] [NOT DOWNLOADED]` (contradictory double badge) and `google/gemma-2-2b-it` / E2B shown `[NOT DOWNLOADED]` though their HF snapshots (4.9 GB / 9.8 GB) are on disk; detection inconsistent across peers (E4B labeled correctly)
- root cause: TWO independent defects. (1) cache-root + completeness: `_hf_hub_root()` read only `HF_HOME` + the legacy `HUGGINGFACE_HUB_CACHE`, never the modern `HF_HUB_CACHE` that huggingface_hub itself honors, so a process without those two vars fell through to the stale `~/.cache/huggingface/hub` default (partial coverage: E4B present, E2B/gemma-2-2b hollow); and `on_disk` was set from "a `snapshots/<hash>` dir exists", not from a materialized weight blob, so a config-only / tokenizer-only pull read as downloaded. (2) label composition (regression `e412e84b` "Disambiguate local Gemma model labels"): `_display_label_for_local_row` began adding `[LOCAL HF]` UNCONDITIONALLY for any `google/gemma*` row while `build_dropdown_choices` still appended `[NOT DOWNLOADED]` on a scan miss -> the two badges stacked
- fix: same commit as this entry -- `_hf_hub_root()` now honors `HF_HUB_CACHE` (then `HUGGINGFACE_HUB_CACHE`, then `HF_HOME/hub`, then default; read live from os.environ, matching huggingface_hub precedence so scanner == loader); `scan_local_llm_cache()` sets `on_disk` only when the chosen snapshot carries a materialized weight blob (`_snapshot_has_weights` -> symlink-resolved `*.safetensors`/`*.bin`, size > 0), preferring the newest weighted snapshot; `build_dropdown_choices()` makes the state suffix EXCLUSIVE ([LOCAL HF] XOR [NOT DOWNLOADED]); parametrized per-item label-vs-disk regression added to `tests/test_model_catalog_scan.py`
- verify idea: for each curated local row, assert the emitted dropdown label carries exactly one state suffix for a given fixture cache state (materialized weight blob -> `[LOCAL HF]` for gemma / bare id otherwise; config-only OR absent -> `[NOT DOWNLOADED]`; never both); assert `_hf_hub_root()` returns the `HF_HUB_CACHE` path and wins over `HF_HOME/hub`; assert `_snapshot_has_weights` is False for a config-only snapshot and True for one with a weight blob
- bible-worthy: yes -- generic rule: a model-picker that scans the HF cache must (a) resolve the cache the LOADER uses (honor HF_HUB_CACHE, not just HF_HOME/legacy alias), (b) gate "downloaded" on a materialized weight blob, not a bare snapshot dir, and (c) keep UI state badges mutually exclusive. Hits any custom node that labels a model dropdown from a cache walk
- operator env note (NOT code-fixable): on this box the ComfyUI process resolves to `~/.cache/huggingface/hub` because it has no HF_* var set; the User-registry `HF_HUB_CACHE=C:\ComfyUI-Models\huggingface` points at the CONFIG-ONLY parent (weights live in `...\huggingface\hub`). For the dropdown to show the real weights, launch ComfyUI with `HF_HOME=C:\ComfyUI-Models\huggingface` (yields `/hub`) or `HF_HUB_CACHE=C:\ComfyUI-Models\huggingface\hub`. The code fix makes the label HONEST for whatever cache the process actually uses
- follow-up (operator directive 2026-07-16, same day): after seeing the corrected labels the operator observed the download-state STILL depends on each user's HF cache layout ("has to work out of the box for every user regardless of where they store their files"), which no scanner can guarantee. Per that directive the download-state badges were REMOVED entirely: `build_dropdown_choices` now emits the bare repo id / handle with NO `[LOCAL HF]`/`[NOT DOWNLOADED]`/`[LOCAL GGUF]` badge (the dead `_display_label_for_local_row` + `_is_google_gemma_local_row` helpers were ripped). `on_disk` is still tracked internally (recovery hint + auto-download short-circuit) and the HF_HUB_CACHE + weight-completeness fixes are retained; the SUFFIX CONSTANTS + `_strip_label_suffix` stay so a value saved by an older badge-bearing workflow still normalizes. Selection is never gated -- a not-cached model is fetched by `auto_download_if_missing` on first Queue
- status: OPEN (badge-label surface removed; underlying cache-resolution/completeness fix stands)

## PBUG-20260717-01 -- codex P0 FactIndex literal-span rejects a whitespace-polluted RSS source
- surfaced: live 30w headless `scifi_codex_v4` leg, 2026-07-17, canonical prompt `ac027c36-4aab-412b-9844-6239cf561d4f` (RESULT FAIL at node 1 OTR_LedgerScriptWriter, pass P0)
- symptom: P0 fails after 2 attempts -- `fact F01 has a non-literal source span: full_text[11:54] expected exact slice '\n\t\t\t\t\t\t\t\tThe Growing Crescent of Mars as NA'; returned quote 'The crescent of Mars grow as the spacecraft approached the planet...'`
- root cause: NEW upstream (ingestion) root -- the INVERSE of the S5 P0 evidence-contract family (PBUG-20260710-10/BUG-11.35, -20260711-02/BUG-11.37, -04/-07/BUG-11.46, -20260712-23/BUG-11.50 ext), all of which assume a clean payload and wrong MODEL metadata. Here `A0.full_text` carried leading `\n`+8 tabs from the RSS source, so the literal-span offset `[11:54]` landed inside the whitespace run and cut a word -- a slice no model can reproduce verbatim, so it paraphrased and the exact-literal validator rejected it. No prior PBUG/Bible rule normalizes source whitespace at INGESTION (BUG-11.26 is a comparison-time whitespace fix for the key_term verbatim test, not offset-bearing-payload cleaning).
- fix: same commit as this entry -- normalize the four span-bearing fields (headline/summary/full_text/seed_text) to single-spaced text in `_otr_scifi_codex.validate_payload_envelope`, AT ADMISSION and UPSTREAM of the digest + the P0 evidence projection + the literal-span validator (cleaned text is the sole coordinate system, so no accepted offset shifts -- the BUG-11.37 constraint); point the P0 post-validator at `env.payload` (the normalized A0) not the raw input `payload`. Codex-scoped -- shared `validate_source_payload` stays byte-identical for the science ledger stamps. New helper `_normalize_span_source_text`.
- verify idea: inject a fact whose source field has leading `\n\t...`; assert `env.payload` span fields are single-spaced/stripped, a literal span into the cleaned full_text passes `_span_ok` first-try, and normalization is idempotent on an already-clean control (no offset shift). Covered by `tests/test_scifi_codex_lane.py::test_p0_source_spans_survive_whitespace_polluted_source`.
- bible-worthy: yes (operator decides at fan-out) -- generic rule: when a contract validates a model quote against a literal offset slice of a source payload, normalize the source whitespace at ingestion UPSTREAM of offset assignment; NOT a whitespace-tolerant validator (leaves dirty offsets stored for every downstream consumer -- the BUG-11.50/PBUG-20260712-23 anti-pattern) and NOT a seam nudge (model-obedience gamble, BUG-11.54). Nearest kin = the S5 family + BUG-11.50; new UPSTREAM root. (Cross-check window ruling, 2026-07-17.)
- status: **LIVE-VERIFIED** -- full episode green on leg `c1f3891f` (RESULT SUCCESS + obs_publish, "The Whisker Effect" 56.6 MB on disk). P0 cleared on the same whitespace-polluted source class that aborted pre-fix (leg 90f22b15 first proved P0 clears; c1f3891f proves the whole episode). Bible-promote at the next operator fan-out.

## Regression watch (2026-07-17 -- NOT a new PBUG) -- codex P3 string_too_long on `premise` re-occurred on scifi_codex_v4
- RE-OCCURRENCE of PBUG-20260713-04 (BUG-11.42), not a new class (cross-check window ruling). Live 30w `scifi_codex_v4` leg (prompt `6883758f`) failed P3 `string_too_long` on premise >144 -- same field, same 144 cap, same lane, same mechanism (model writes over-cap; text-patch never clips prose). The -04 verified recipe (conservative ~75% model-facing `max_chars` with the true cap PRIVATE + `source_to_shorten`/forbid-unchanged-copy + never Python-clip) is present in tree (`_otr_scifi_codex.py:1752/:1754/:1800` + surface instruction premise<=108). A same-session kibitz had re-added the literal 144 cap to the base seam; per -04 that is the anti-pattern (exposing the rejection edge makes the model aim at it and cross it), so it was REVERTED (same commit as PBUG-20260717-01). Untestable end-to-end until PBUG-20260717-01 (P0) clears; sequence: P0 clears -> a live 120w leg exercises the P3 premise cap. If premise still overruns for the v4 proof-pressure density AFTER -04's recipe, the BUG-11.54 deterministic word-boundary shortener (already used for question/consequence) is the design precedent. No new PBUG until a live failure survives the -04 recipe.
- **RESOLUTION (2026-07-17, operator decision "allow longer text"):** after P0 cleared, a live leg (`90f22b15`) failed P3 `string_too_long` on BOTH `premise` (>144) AND scene `description` (>72) -- the -04 recipe IS insufficient for the verbose v4 proof-pressure lane. Operator chose to RAISE the caps rather than clip prose. Raised the non-spoken metadata caps: `premise` 144->240, scene/shot `description` 72->144 (draft+final models + `_p3_text_patch_cap` + the text-patch `replacement_text` schema bound + the receipt). These caps are **load-bearing** (they size the P3 draft to the model's 8192 context+output budget), so the output reservation was resized `1647->1829` and every exact-token guard updated (max-width draft 1418->1576 tokens; envelope re-verified prompt+output=5935<=8192). Full suite 8144 / Bible 17. **LIVE-PROVEN**: leg `c1f3891f` RESULT SUCCESS + obs_publish -- premise+description now fit the raised caps end-to-end (obs asset on disk). NOT promoted to a new PBUG (re-occurrence of -04, resolved by the cap raise, not a novel class).

## PBUG-20260718-01 -- scifi_fable2_v3 was a runnable=True bank that could never complete a leg (fable2 revision_contract hardcodes rules_id == 'scifi_fable2')
- surfaced: live cross-bank Sonnet bake-off render window, 2026-07-18, baseline HEAD `60c73618`; the `scifi_fable2_v3` story-only leg logged `RESULT FAIL canonical_runner_exit=1` at t=22s, before any generation, and is model-independent (reproduced under creative=`anthropic/claude-sonnet-4.5`). Full causal record: `docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md`.
- symptom: `!!! [scifi_fable2] pass 'revision_contract' failed: story_rules.rules_id must be 'scifi_fable2', got 'scifi_fable2_v3' (no fallback to legacy_many_pass)` -> `nodes._otr_scifi_fable2.Fable2ScriptError`.
- root cause: the fable2 lane (`nodes/_otr_scifi_fable2.py:2307`) hardcodes the expected `rules_id` to the literal `"scifi_fable2"`. The 2026-07-17 roster trim (`499386aa`) made every lane own its `story_rules` by EXACT id, so `scifi_fable2_v3`'s rules carry `rules_id = "scifi_fable2_v3"` while its pipeline `fable2_multipass_v3` still routes into `_otr_scifi_fable2` -- which then rejects the v3 id. Net: a `runnable=true` bank that can never finish a leg. (`scifi_fable2` base is unaffected -- rules_id == 'scifi_fable2'.)
- fix: RETIRED the bank rather than patch the contract (Sonnet-bake-off verdict, `docs/2026-07-18-sonnet-bakeoff-analysis.md` + `docs/2026-07-18-rip-4-banks-plan.md`). The `scifi_fable2_v3` bank row, pack dir, `story_rules`, and its `fable2_multipass_v3` pipeline (removed from BOTH `pipelines.json` and `_RUNNER_BY_PIPELINE`) plus the writer's fable2 target-word gate entry were all deleted in this change, alongside `media_archive_v3` / `scifi_codex_v3` / `scifi_sonnet_v3`. No live route to the defective contract remains.
- verify idea: `scifi_fable2_v3` no longer appears in `_otr_story_routing._ensure_loaded().pipelines`, `banks.json`, or `_RUNNER_BY_PIPELINE`; the source-only retired-id scan over `nodes,tests,workflows` returns zero; full suite + Bug Bible stay green with the bank gone.
- bible-worthy: no -- resolved by removal, not a reusable code contract. If the fable2 family re-adds a `_v3`, re-open the NEWBUG fix-direction: accept the lane's DECLARED rules_id, never a single literal.
- status: **CLOSED-BY-RIP** at this commit. NEWBUG doc marked CLOSED-BY-RIP and RETAINED (the only causal record of the live failure -- never deleted).

## PBUG-20260720-01 -- official Gemma 4 12B HF writer was stranded behind an obsolete architecture/catalog gate
- surfaced: offline Gemma recovery probe plus canonical headless requalification on the RTX 5080 16 GB box, 2026-07-20. The complete official checkpoint was already under `C:\ComfyUI-Models\huggingface\hub`, but the installed Transformers 5.5.0 did not recognize `model_type=gemma4_unified`; the catalog separately hard-rejected `google/gemma-4-12b-it` and steered users to the unconstrained GGUF row.
- symptom: the official 12B model could not be selected on OTR's in-process Transformers/HF lane, so the writer could not use that lane's lm-format-enforcer token grammar. The optional GGUF lane instead reached character-zero JSON failures in structured work.
- root cause: the catalog tombstone outlived the runtime limitation that prompted it. Correct inference requires native `Gemma4UnifiedForConditionalGeneration` support, not the retired text-tower remap. This machine's HF cache also splits the materialized weights and the newer `chat_template.jinja` across two revisions, while the old resolver assumed the newest snapshot directory was complete.
- fix: require Transformers >=5.10.4, restore the curated `google/gemma-4-12b-it` row, remove its hard reject, resolve the newest materialized-weight snapshot plus newer compatible local chat-template metadata, and keep tokenizer/config/model loads `local_files_only=True` with no in-loader HTTP fallback. The canonical workflow now selects the row in both writer slots with `cuda` / `sdpa` / `bnb_nf4`, context 8192. Exact result schemas are bound at the local P0-P9 scheduler boundary; P3's authored-text patch keeps its narrower schema.
- verify idea: in a zero-network process require the official Gemma4Unified class, `is_loaded_in_4bit=True`, coherent prose, and LMFE JSON that decodes and validates. In the real canonical workflow require P0's raw head to begin with `{` and reach semantic validation instead of character-zero parsing.
- bible-worthy: yes -- architecture capability and coherent split-revision cache resolution are reusable model-admission contracts.
- promotion: BUG-02.16.
- status: **FIXED; LIVE-REQUALIFIED THROUGH P5**. The doctor measured 7.152 GiB allocated / 7.286 GiB peak and returned coherent prose plus parsed constrained JSON. Canonical prompts `4a89df7e-c8e1-407f-ab10-c3159eb17b6b` and `ee0d4743-11bc-4367-9e19-5422afa2c95f` both loaded offline NF4 at a 7.15 GiB model delta; P0 began with valid JSON, decoded, and needed only deterministic source-span repair. The second leg reached P5 with a complete schema-valid artifact. Full media publication remains unclaimed because that leg later exhausted the existing P5 spoken-hygiene semantic repair.

## PBUG-20260720-02 -- an open P5 scene dictionary crashed LM Format Enforcer mid-object
- surfaced: first real canonical Gemma/HF leg, prompt `4a89df7e-c8e1-407f-ab10-c3159eb17b6b`, 2026-07-20. P0-P4 and P3 rewrite had already cleared under hard constraints.
- symptom: every P5 attempt began valid JSON and stopped at `..."scenes":[{`; LMFE logged `AttributeError: 'bool' object has no attribute 'anyOf'`, after which the retry ladder misleadingly reported character-zero JSON because no complete top-level object remained.
- root cause: `ScriptArtifactV4.scenes: list[dict[str, Any]]` compiled to `items: {type: object, additionalProperties: true}`. LMFE 0.11.3 accepted that schema initially but treated the boolean wildcard as a schema object when generation reached the first arbitrary scene key, then terminated token enforcement.
- fix: replace the wildcard with the real closed contract, `ScriptSceneV4(scene_id, env, description)`. Hard enforcement stays enabled; no unconstrained fallback or output stripping was added. A regression feeds a complete production-shaped P5 artifact through `JsonSchemaParser` one character at a time and requires every character to be allowed plus `can_end()` at completion.
- verify idea: assert the P5 scene schema has exactly the three required properties and `additionalProperties: false`; scan every bound P0-P9/P3-patch schema for boolean wildcards; run P5 live and require a complete artifact to reach post-validation without an LMFE internal error.
- bible-worthy: yes -- validate generated schemas against the grammar compiler's supported subset before binding them to local structured generation.
- promotion: BUG-11.55.
- status: **FIXED; LIVE-REQUALIFIED AT P5** by prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f`: P5 produced a complete, schema-valid JSON artifact and entered the ordinary spoken-text post-validator. The later semantic repair exhaustion is not a recurrence of this grammar/compiler bug.

## Regression watch (2026-07-20 -- NOT a new PBUG) -- Gemma P5 repeated a spoken-hygiene defect after bounded repair
- prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f` produced a complete constrained P5 artifact but line `l001` contained stage direction, markup, or a role label. The existing Axis-6 route from `docs/2026-07-18-codex-short-leg-convergence.md` correctly selected the spoken-reword repair rule; Gemma repeated the same defect and the lane failed closed after the bounded model repair. This is model non-compliance at an existing semantic gate, not a JSON/LMFE regression and not evidence for a new deterministic code defect. It blocks a full-episode promotion claim, so the handoff records runtime/grammar qualification only.

## PBUG-20260720-03 -- a craft-only spoken-line reject could kill the episode
- surfaced: canonical Gemma/HF requalification prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f`, 2026-07-20, after P0-P4 had cleared and P5 had produced a complete schema-valid artifact
- symptom: P5 line `l001` failed with `spoken text contains stage direction, markup, or a role label`; Gemma repeated the defect on typed repair and `_otr_structured_call` raised after two attempts, so no accepted/frozen ledger, TTS, video, or OBS asset was produced. Code grounding also found the shared freeze path could translate craft/quality exhaustion into a terminal-skip disposition with downstream readiness phases stamped `terminal_skipped`
- admission note: this supersedes the preliminary "NOT a new PBUG" regression-watch classification immediately above. That note assessed only model noncompliance; grounding exposed the distinct deterministic workflow-liveness defect: a sanitizable quality reject controlled episode completion
- root cause: spoken craft exhaustion had no total post-model repair boundary. Quality-budget exhaustion shared terminal liveness semantics with genuinely invalid structure, and content-owned lanes could validate authored text before delivery normalization rather than the exact TTS surface. One stubborn but sanitizable row could therefore kill the whole episode
- fix: all six runnable banks now use a total spoken-hygiene ladder. The established repair/recompose, lower-temperature CRITICAL, and alternate-slot rungs remain the opening A/B/C phase; unresolved rows then enter a dynamic fresh repair/rejudge loop that rotates every callable same/alternate writer lane with new defect feedback and temperatures. Every candidate is rescored on the exact projected spoken surface. A finite model-pass budget ends at an idempotent validated SFW floor, so a stubborn quality model cannot hang or kill the episode. Repaired rows stamp the gate and resolving rung. Craft exhaustion continues through freeze/readiness, while a truly empty mechanical row is isolated locally. Structural ambiguity and the deterministic G9 SFW ship-stop remain fail-closed
- sibling quality paths: `scifi_news` (the renamed Codex implementation) now repeats its P6/P7 listener judge+creative-retake and P8/P9 final-audit+retake cycles until clean or the validated quality floor; its typed spoken patch path uses the same dynamic cross-slot policy. `original` likewise alternates fresh creative/technical outro repairs and independent technical re-judges for subjective epilogue findings, then keeps the best structurally valid close with a nonterminal `quality_floor` receipt if its dynamic 3-6-cycle budget is exhausted. Inline Story QA feeds each MICRO/REJECT concern into a fresh scoped creative repair and an independent technical rejudge under the same liveness rule. The source-adapter family (`media_archive`, `public_domain`, `shakespeare`, plus the registered news interpreter) retains separate bank prompts/schemas/truth validators but now shares a 12-model-call liveness chain: technical and creative slots alternate with the exact prior rejection, then a validated bank-specific brief is derived only from the fetched payload, source hash, and source-side cast hints. Broken feeds, manifests, rights/config, backends, and interpreter contracts still fail loud. `scifi_news_pro` (the renamed Fable2 implementation) keeps its existing content ownership and seal/rebuild contract; only the shared total spoken repair boundary applies when one of its sealed rows is defective
- verify idea: force every craft gate and whole-line stage cue to survive all model rungs; require a non-empty clean floor result plus `hygiene_repaired_after_reroll:<gate>:<rung>`, normal Phase 7/8/10 completion, and no quality verdict in `FREEZE_TERMINAL_FAILURE_VERDICTS`. Cover `media_archive`, `original`, `public_domain`, `shakespeare`, `scifi_news`, and `scifi_news_pro`; assert delivery projection is repaired before content-owned seals; retain an unsafe-line test that Phase 10 refuses to freeze
- bible-worthy: yes -- generic rule: an LLM's refusal to satisfy a non-safety wording gate must not own workflow liveness when a bounded deterministic clean spoken projection exists
- promotion: BUG-11.56
- status: **ROOT-FIXED / LIVE-VERIFIED**. Final canonical prompt
  `f3770246-2d6a-4302-90af-153120edddf2` exercised the new boundary twice:
  P5 repaired four `one_breath` rows and P7 repaired a
  `spoken_format` / `stage_direction` row; both receipts stamp
  `trigger=craft_only_post_validation` and
  `shared_artifact_repair_bypassed=true`. The ledger froze
  `frozen_with_warns` (cosmetic word-count receipts only), all Phase 7/8/10
  telemetry remained non-skipped, four clean lines / 45 words reached TTS and
  video, and `obs_publish OK` wrote the 22,892,541-byte final asset at
  `output/otr/obs/signal_lost_the_weight_of_height_20260720_221418_silent_procgen_blended_captioned_with_credits_final.mp4`.
  A later P9 score-graph mismatch correctly stayed outside the craft-only
  boundary; its full-artifact retries hit the separate 8K structured-capacity
  limit and the already accepted clean script still completed normally.

## PBUG-20260720-04 -- alias-blind media consumers dropped the sentinel announcer identity
- surfaced: the published Fable2 Einstein and Butterfly episodes audited in `docs/2026-07-10-fable2-s2-QA-ANALYSIS-r2.md`. Einstein captions omitted an ANNOUNCER label around the sentinel; Butterfly labeled the intro sentinel but omitted the coda sentinel
- symptom: the ledger and rendered episode completed, but a normalized/cast-keyed sentinel could lose its canonical speaker label in captions. Static sibling grounding found that credits could resolve the alias-aware display name yet miss the same row's voice receipt, HuMo could reject the normalized radio face unless `char_id` remained the literal `announcer`, and captions could consume a canonically skipped row instead of filtering it row-locally
- root cause: downstream media consumers independently rebuilt raw exact-`char_id` maps instead of using the central alias-aware ledger-consumer resolver and canonical skip semantics. ShotLock correctly normalizes the sentinel to a cast identity, but HuMo's later stale guard still tested the pre-normalization literal ID
- fix: captions now filter canonical skips before ordering, preserve canonical caption text, and resolve speakers through the shared alias-aware cast lookup; credits use that same lookup for both display name and voice; HuMo recognizes the sentinel by role/source-family/portrait identity after ShotLock normalization. No ledger ownership, readiness, seal, hash, node, widget, or canonical-workflow surface changed
- verify idea: feed intro/coda sentinel aliases plus an ordinary similarly named cast row through captions, credits, ShotLock, and HuMo; require both announcer labels and the voice receipt, exclude skipped rows without a timing clamp, accept the normalized sentinel portrait, and still reject the ordinary stale mismatch
- bible-worthy: covered by existing BUG-12.43 (namespace aliases must resolve at every consumer) and BUG-05.11 (canonical skip state is row-local); no new portable rule
- status: **ROOT-FIXED / FOCUSED-GREEN; canonical six-bank live qualification pending**

## PBUG-20260720-05 -- caption suffix made the terminal mux publish into a fake sibling episode
- surfaced: production-artifact audit of the completed `media_archive` episode `signal_lost_reel_history_20260720_102732`, 2026-07-20. Its ledger named a 105,782,049-byte final that existed on disk, but the path was under the invented sibling directory `signal_lost_reel_history_20260720_102732_silent_procgen_blended_captioned` rather than the episode root
- symptom: every media stage could complete and OBS could receive a playable copy while the archival final escaped `meta.paths.episode_root`. A success/file-exists check alone therefore blessed a structurally wrong output tree and left the real episode directory without its terminal final
- root cause: `OTR_MasterAudioMux._default_out` reconstructed the episode id by peeling a hard-coded suffix list. Captions were inserted before credits, but `_captioned` was absent from that list; after `_with_credits` was removed, the remaining enriched stem was reinterpreted as a new episode id. Any future terminal enrichment could repeat the class
- fix: the mux now treats the in-flight ledger path as the canonical episode-directory authority and accepts it only when it is a direct child of the configured episodes root and the incoming stem begins with that episode id (rejecting a stale prior-episode singleton). Ordered suffix peeling, now including captions, remains only as the no-ledger fallback. The fully enriched filename is preserved. No node, widget, link, or canonical-workflow change was needed
- verify idea: point the in-flight ledger at `otr/episodes/ep042/audio/ep042_ledger.json`, feed an input with an unknown future terminal-enrichment suffix, and require the final parent to remain exactly `otr/episodes/ep042`; require a mismatched stale ledger to be rejected and the caption/credits fallback chain to peel to the correct episode root
- bible-worthy: yes -- portable rule: when an accepted manifest/ledger already owns an artifact directory, downstream enrichments must consume that authority rather than reverse-engineering identity from an open-ended filename suffix grammar
- status: **ROOT-FIXED / FOCUSED-GREEN; canonical six-bank live qualification pending**

## PBUG-20260721-01 -- selected RSS provenance disappeared behind a blank request widget
- surfaced: first canonical six-bank qualification leg, `media_archive` prompt
  `12f3df7f-298e-411c-9fe2-59ef3ac62ae2`, published episode
  `signal_lost_the_casting_reels_20260721_010623`, 2026-07-21
- symptom: the fetched media payload carried a real selected article link,
  outlet, date, and embedded source hash, but the final ledger stamped a blank
  `meta.source_ref` and empty source sidecars. The story and media rendered, so
  an output-only check could not detect that the ledger no longer identified
  the item it had adapted
- root cause: the RSS fetchers returned the strict seven-key payload as a raw
  dict. `normalize_fetch_result` intentionally treats a legacy raw dict as
  having no provenance sidecars, and `_resolve_inputs` then wrote the optional
  request widget as `source_ref`. Both RSS families ignore that widget and
  choose an item dynamically, so the request coordinate could never name the
  selected source. The sibling defect covered `media_archive`, `scifi_news`,
  and `scifi_news_pro`; manifest-backed and synthetic banks already owned their
  provenance explicitly
- fix: the two known RSS wrappers now preserve the exact seven-key payload but
  return `SourceFetchResult` with selected URL/label/date metadata and explicit
  unknown rights. The writer resolves the canonical ledger `source_ref` in
  owner order (fetcher-selected ref, selected payload link, requested widget)
  and stores a differing request separately as `requested_source_ref`. It does
  not invent a license or fair-use claim
- verify idea: make each RSS wrapper select a link different from the supplied
  request; require the selected link at `meta.source_ref`, in source metadata,
  and in rights provenance, with the request retained only as a request. Keep
  raw-dict legacy normalization and both manifest-backed banks unchanged
- bible-worthy: yes -- BUG-12.54
- status: **ROOT-FIXED / FOCUSED-GREEN; live requalification pending**

## PBUG-20260721-02 -- the frozen master WAV had no durable byte identity or final pointer
- surfaced: same completed `media_archive` qualification artifact as
  PBUG-20260721-01. During video generation every per-beat slice logged that
  `ledger.audio.master_audio_sha256` was absent
- symptom: the master WAV existed and the final archival MP4 was proven
  byte-identical to it, but the final ledger had no top-level `audio` section,
  no full master hash, and a blank `final_audio_path`. The video slice cache
  therefore fell back to path identity and would reuse stale slices if new WAV
  bytes later landed at the same path
- root cause: EpisodeAssembler wrote and closed the authoritative master but
  recorded only a first-kilobyte waveform tripwire in `audio_gates`.
  `render_driver` already consumed a full-file hash that no production owner
  produced, and a later `Ledger.save()` did not preserve an externally stamped
  `audio` section. The terminal mux owned the final video pointer but never
  stamped the re-resolved master path
- fix: EpisodeAssembler computes a streaming SHA-256 after the WAV header is
  closed and stamps it with `ledger_frozen=true` in the owned `audio` section.
  ProductionLedger now initializes and preserves that section. Hash receipt
  failure remains loud but nonterminal. The terminal mux stamps the resolved
  master path together with the final video and OBS pointers after successful
  publication
- verify idea: hash a multi-chunk closed WAV and require the full digest to
  survive a later ProductionLedger merge; require per-beat slice keys to bind
  that digest, and require the final ledger's audio path to exist. A simulated
  hash failure must not erase or relabel an otherwise usable master asset
- bible-worthy: yes -- BUG-12.55
- status: **ROOT-FIXED / FOCUSED-GREEN; live requalification pending**

## PBUG-20260721-03 -- ledger save rebuilt the exact OBS deliverable into a nonexistent alias
- surfaced: same completed `media_archive` qualification artifact as
  PBUG-20260721-01
- symptom: `final_video_path` named the existing archival final and
  `meta.obs_final_path` named the existing playable OBS copy, but
  `meta.paths.obs_final` named a shorter nonexistent
  `<episode_id>.mp4`. Two official ledger surfaces therefore disagreed after a
  successful publish
- root cause: the terminal mux stamped the exact OBS path and then called the
  shared ledger owner. Every save unconditionally rebuilt `meta.paths` from a
  pre-publication filename plan, discarding the terminal publisher's enriched
  caption/credits/final filename. The planned alias outranked the observed
  artifact
- fix: the ledger owner accepts a terminal published OBS path only when it is
  an existing MP4 for the current episode under the inferred canonical OBS
  root or the explicit `OTR_OBS_DIR`. That validated exact path drives both
  `meta.obs_final_path` and `meta.paths.obs_final`; before publication the
  historical planned path remains only a plan. The mux stamps all terminal
  asset pointers in one owner-layer save
- verify idea: publish an enriched filename and require every final path
  surface to name an existing asset after save. Reject a missing path, wrong
  episode prefix, wrong extension, or path outside authorized OBS roots, then
  verify a later ProductionLedger save cannot regress the accepted filename
- bible-worthy: yes -- BUG-12.56
- status: **ROOT-FIXED / FOCUSED-GREEN; live requalification pending**

## PBUG-20260721-04 -- the post-audio ledger owner never reached the video wire
- surfaced: canonical `media_archive` requalification prompt
  `7a6618ec-dd00-4711-93c6-43573d5e9580`, episode renamed to
  `signal_lost_the_municipal_ledger_20260721_020231`, 2026-07-21. The run was
  stopped before publication at the first repeated render warning, per the
  cross-bank qualification protocol
- symptom: the closed master WAV and its disk ledger carried full SHA-256
  `2f8f4a196c28343d28904f4381ca1632c66f6ff00fef79307ef2c564dc217e93`,
  but the exact `VideoRenderBatch` input capture had `audio={}`. Every shot
  therefore warned that `_slice_master_audio` had been called without the
  master content hash and built an under-invalidated slice identity. The same
  capture omitted disk-only `audio_gates` and `transitions`
- root cause: `OTR_ShotLock` is the graph's intended post-audio join, but
  `overlay_audio_timing` copied only missing row-local timing/WAV fields from
  the newest ledger. It returned before reading disk whenever any wire row
  already had a timing hint, never copied the producer-owned top-level audio
  state, and selected by newest mtime without proving episode identity. The
  freeze-cascade wire is intentionally pre-audio, so the canonical graph could
  be correctly gated yet still deliver an empty audio section to every image
  and video consumer
- fix: ShotLock now resolves the active ledger through
  `in_flight_ledger_path`, proves same-episode identity with the immutable Phase
  10 `meta.freeze_timestamp` (which survives pending-to-final rename) or an
  exact non-empty episode id for older ledgers, and rejects mismatches loudly.
  It always visits the post-audio owner despite existing row timing. Matching
  disk truth replaces the complete producer-owned `audio` section, carries
  `audio_gates`, `transitions`, and `radio_bookend_path`, additively fills empty
  metadata, then performs the established missing-only row merge. The image
  dispatcher remains a wire-preserving consumer; no workflow link/widget/node
  change is required
- verify idea: give ShotLock a pending-id wire with existing timing and a
  renamed disk ledger sharing the same freeze timestamp; require disk's full
  master hash and post-audio sections to survive ShotLock and ImageDispatcher
  JSON serialization while populated writer metadata remains unchanged. Give
  it a different freeze timestamp and require no field to cross the boundary
- bible-worthy: yes -- BUG-12.57
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-05 -- split dialogue rows replaced accepted beat identity with synthetic child ids
- surfaced: canonical `media_archive` requalification prompt
  `c96e268d-0b8a-4bb5-8e10-2aacb8459680`, episode
  `signal_lost_soot_and_signature_20260721_025707`, 2026-07-21. The run was
  stopped during video before publication when the final ledger's deterministic
  consistency receipt reported nine beat-id defects
- symptom: split rows had unique child ids such as `b003_s1`, but both
  `line_id` and `beat_id` were set to that synthetic id. The accepted outline
  owned only `b001` through `b009`, so every child appeared to reference a
  nonexistent beat. The same legacy ledger had empty top-level `beats[]`,
  leaving no durable parent-to-line membership even for unsplit rows
- root cause: `_clone_voiced_row` treated line identity and narrative beat
  identity as one namespace. `production_ledger.init_lines_from_outline`
  initialized only `lines[]`, despite already owning the accepted outline beat
  set, and structural apply never refreshed denormalized beat membership after
  split/cut/merge operations. Downstream render stages correctly key shots by
  unique `line_id`; changing those consumers to parent beat ids would collapse
  sibling split rows and was rejected
- fix: split children now mint only a unique `line_id` and retain the exact
  parent `beat_id`. Outline initialization materializes the accepted top-level
  beat collection with initial `line_ids`; every structural apply rebuilds
  only those retained beats' final exact line membership, leaving a fully cut
  accepted beat present with `line_ids=[]`. Repeated repair passes allocate the
  first free child suffix across the ledger, so a second split cannot reuse an
  existing child line id. Structural telemetry is line-named,
  with deprecated beat-named aliases carrying the same unique line ids for
  compatibility. ShotLock, TTS, timing, image/video, captions, credits,
  readiness, hashes, and OBS retain their existing unique-line consumer keys
- verify idea: split one accepted beat and cut another; require two unique
  child line ids mapped to the first parent beat, the cut parent retained with
  an empty list, a clean outline/ledger consistency result, and no collapse at
  line-keyed consumers. Split the same parent again and require a new unique
  child id. Initialize directly from an outline and require
  `beats[].line_ids` to match the initial lines before any timing stage
- bible-worthy: yes -- BUG-12.58
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + offline self-test + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-06 -- the radio editor overrode a requested 180-word story with a hard-coded 350-word target
- surfaced: same stopped canonical `media_archive` run as
  PBUG-20260721-05. The writer produced a good 148-character-word body plus 67
  announcer words for the requested 180-word episode, but the editor declared
  it short and expanded it. The final receipt reported 252 character words
  (`actual_ratio=1.4`, advisory drift) and 316 total spoken words
- symptom: a story already inside the live receipt's `[0.7, 1.3]` band was
  needlessly sent through length normalization. The model claimed an in-range
  `projected_word_total`, but deterministic application produced a different
  total and still passed. Separately, a row-local quality repair could be
  rejected solely because the whole episode carried advisory word drift
- root cause: `_otr_radio_editor` hard-coded 350 +/-20%, counted announcer
  overhead in the episode target despite the writer's character-only contract,
  and validated the LLM's arithmetic claim rather than the plan's applied
  result. Its shared validator also made global length conformance a
  prerequisite for unrelated micro repair
- fix: the live `meta.word_budget` receipt is now the single authority: a
  positive target plus its two ratio multipliers. Only an absent pre-receipt
  ledger uses the historical 350/[0.8,1.2] fallback; a malformed present
  receipt records `SKIPPED_INVALID_BUDGET` without an LLM call or mutation.
  Budget accounting counts non-skipped character rows only, while every
  character and announcer row still owns the one-breath cap. Length-plan
  validation deep-copies the ledger, applies the proposed edit
  deterministically, and gates on the resulting character total; the model's
  projection is retained as forensic evidence. Micro repair disables only the
  global band check and retains structural, noun, line-cap, anchor, action, and
  row-scope guards. The two content-owned sci-fi routes keep their independent
  budget/seal contracts
- verify idea: at target 180, require 148 character words plus 67 announcer
  words to skip normalization, but an over-cap announcer row to trigger it.
  Accept a good simulated result despite a false model projection and reject a
  bad result despite a claimed 180. Permit a scoped repair during advisory
  episode drift while still rejecting an over-cap replacement; prove malformed
  present receipts make no mutation
- bible-worthy: yes -- BUG-12.59
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + offline self-test + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-07 -- the protected news-coda fact bypassed final spoken-surface validation
- surfaced: completed and published canonical `media_archive` qualification
  prompt `f7bffc53-bada-45c1-9256-4a27a3caed51`, episode
  `signal_lost_the_diary_keys_20260721_040039`, 2026-07-21
- symptom: strict audit rejected exact TTS coda row `b009`. The canonical row
  contained all three episode anchors and expanded to 44 delivery words after
  normalization, producing `anchor_stuffing` and `one_breath`, even though the
  bridge itself had passed hygiene. The Phase 7 record reported a failed row
  count but exposed no corresponding failure receipt
- root cause: both first-pass composition and the later shared spoken scour
  validated only the authored bridge, then reattached the protected factual
  suffix without rescoring the assembled surface TTS would consume. The
  composer could also character-cut the factual suffix, manufacturing a false
  sentence boundary. Phase 7 built failure detail by filtering the successful
  repair receipts, so row-local failure evidence was always empty
- fix: one shared coda finalizer now assembles the bridge with the exact source
  fact, projects it through the authoritative delivery normalizer, and scores
  the complete spoken row. A dirty full fact may reduce only to the longest
  clean exact complete-sentence prefix; if no such prefix exists, the spoken
  row points truthfully to credits while the full source note remains in
  `meta.news.news_close_brief`. Models never receive or rewrite factual prose,
  and no character truncation is permitted. First-pass composition receives
  injected canon and the live breath range; later scour stamps only hash
  receipts for any reduction. The mutator itself refuses both content-owned
  sci-fi policies, preserving their accepted rows, seals, and hashes. Phase 7
  now carries explicit row-local failure details
- verify idea: replay the exact live `b009` note and require its projected TTS
  surface to pass the final row scorer without exposing the fact to any model.
  Exercise multi-sentence, initials/version, and single-sentence facts: permit
  only whole exact sentences or a truthful credits deferral, never fragments.
  Require the same behavior in the `media_archive`, `public_domain`, and
  `shakespeare` legacy routes; keep `original` empty-coda behavior and direct
  `scifi_news`/`scifi_news_pro` shared-scour inputs byte-identical. A forced
  row-local failure must appear in the Phase 7 receipt
- bible-worthy: yes -- BUG-12.60
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + cross-bank + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-08 -- generic cast roles were mistaken for forbidden character names
- surfaced: canonical `public_domain` qualification episode
  `signal_lost_inheritance_of_desolation_20260721_060315`, 2026-07-21. The
  cast contained `THE TRAVELER` and `THE WITNESSES`; the technical story-brief
  model returned `A weary traveler faces a skeptical assembly...`. An older
  canonical public-domain 720-word production log reproduced the same class
  with `THE SCIENTIST` and `A scientist...`
- symptom: the story-brief content gate reported `named_character`, sent a
  repair that rejected the ordinary role noun, received the same truthful role
  again, and exhausted to the explicit failed sentinel. `ShotLock` and the LTX
  scene opener then received blank story-brief metadata (`status=failed`,
  `0/0/0`) and used only their non-authoring visual defaults. The episode could
  continue mechanically, but it was not a clean configured image/video brief
- root cause: one lexical splitter served two incompatible jobs. It treated
  every word in a cast label as both an input-anonymization alias and a
  forbidden output name. Thus generic identity labels such as `THE TRAVELER`
  made `traveler` illegal and even mapped the article `the` as if it were a
  person. The validator could identify only the broad reason code, so repair
  was not told which surface triggered the rejection
- fix: a bounded shared cast-identity grammar now distinguishes generic roles
  from personal names. Input anonymization maps a generic full label and its
  role noun to one stable identity, never an article; output validation permits
  those generic role forms. Personal labels still protect the full name plus
  meaningful components while excluding articles and honorifics as standalone
  tokens. The public reason code remains stable, while the private repair seam
  receives the exact matched surface and asks for environment, light, color,
  texture, space, material, weather, or objects. Genuine exhaustion still
  returns the observable failed sentinel, and downstream deterministic visual
  defaults remain non-authoring
- verify idea: replay the live `THE TRAVELER`/`A weary traveler...` case in one
  call and require a successful brief. Exercise article-bearing, ordinal, and
  multiword roles such as `First Witch`; Unicode, hyphenated, apostrophe, and
  honorific-bearing personal names; and representative personal-name shapes
  from all six banks. Assert that input substitution preserves one identity,
  ordinary `the` survives, real names remain forbidden, private repair names
  the exact surface while the public code stays `named_character`, and a
  genuine failed brief still produces a valid non-authoring downstream visual
  prompt
- bible-worthy: yes -- BUG-12.61
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + cross-bank + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-09 -- quality retakes repeatedly requested an artifact that could not fit the real context
- surfaced: canonical `scifi_news` qualification prompt
  `e67869e2-6ed5-43d2-b522-094e96ea94c0`, source article
  `3 Questions: Neural transparency and the future of AI design`, 2026-07-21.
  The run was stopped before ledger assembly after the third identical P7
  capacity cycle
- symptom: P7 serialized the complete score, prior `ScriptArtifactV4`, review,
  and complete result schema into an approximately 6,182-token prompt, then
  requested 2,970 output tokens from an 8,192-token local context. The
  transport reduced that output to approximately 2,010 tokens, truncating the
  complete artifact. Typed repair expanded the prompt to approximately 7,501
  tokens and left only 691 output tokens. The quality loop restored the
  unchanged prior script, re-audited it, and repeated the same mathematically
  impossible work. Roughly forty minutes elapsed with zero ledger rows; the
  GPU was busy generating, not video-rendering or memory-thrashing
- root cause: P7/P9 shared P5's whole-artifact schema, post-validator, retry
  ladder, and dynamic script budget even though quality findings owned only
  line text. Shared context fitting treated every output request as a
  reducible ceiling and had no signal that a bounded patch must arrive whole.
  Provider wrappers hid the capacity type behind backend errors, and the
  quality loop continued after restoring an unchanged artifact
- fix: P5 is now the only complete `ScriptArtifactV4` pass. P7/P9 derive a
  closed write set from valid finding line IDs (null means all voiced rows;
  invented IDs are discarded), request a compact typed line-text patch, merge
  only `line.text`, and run the complete script validator. A successful merge
  always returns to a fresh P6/P8 judgment. Malformed creative output gets one
  colder technical-slot attempt; two failures keep the best valid script and
  stop without rejudging unchanged input. A full-output marker is captured
  before normalization and enforced by writer-local, model-loader/polish,
  OpenRouter, Comfy Credits, Google, and GGUF transports, including provider
  output caps. Proven capacity failure is a no-call quality floor. P6/P8 model
  or transport failure is advisory and cannot kill an already valid story.
  Final hashes, authorship receipt, ledger rows, readiness, media consumers,
  and OBS paths are still built only after quality converges or floors
- verify idea: replay the live `6182 + 2970 > 8192` arithmetic and require zero
  generation/network calls when the complete patch cannot fit. Exercise every
  backend and provider cap; prove unmarked calls retain ordinary clamping.
  Require exact target coverage, immutable non-text fields, full merged-script
  validation, creative-to-technical rotation, fresh rejudgment only after a
  successful merge, and no second audit after a two-slot failure. Assert P5 is
  the only complete-artifact pass and all six source-bank routes retain their
  existing ledger/media/OBS ownership
- bible-worthy: yes -- BUG-12.62
- status: **LIVE-ADMITTED / ROOT-FIXED; focused cross-backend + lane tests GREEN; full-suite, Bug Bible, workflow gates, and live requalification pending**

## PBUG-20260721-10 -- redundant JSON-schema constraints disabled every compact local repair
- surfaced: canonical `scifi_news` qualification prompt
  `c8277cf6-dbb8-41ec-bcc4-ac5671080022`, episode
  `signal_lost_the_fortress_of_reason_20260721_095038`, 2026-07-21
- symptom: the P5 spoken-hygiene repair and every P7/P9 quality patch failed
  before emitting one token with `LMFormatEnforcerException: String schema
  contains both a pattern and a min/max length`. Reusing the technical slot
  then failed through `NoneType.allowed_tokens`; the run fell to deterministic
  hygiene/quality floors even though both local models were available
- root cause: both compact patch row schemas declared `line_id` with exact regex
  `^l\d{3}$` and redundant `min_length=1,max_length=16`. LM Format Enforcer
  explicitly rejects that JSON-Schema combination. Its token enforcer caches
  an output state before allowed-token calculation, so the first schema
  exception can leave an incomplete cached state for the reused prefix
- fix: retain the exact regex as the sole line-id constraint in both patch
  schemas. Do not catch or suppress the formatter exception. Character-feed
  the production JSON for each complete patch schema through LMFE and retain
  Pydantic rejection coverage for wrong-prefix and wrong-length ids
- verify idea: drive valid P5-hygiene and P7/P9/P10 patch JSON one character at
  a time through `JsonSchemaParser`, require `can_end()`, assert neither line-id
  schema contains `minLength/maxLength`, and reject `l1000`/`a100`
- bible-worthy: yes -- BUG-12.63
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-11 -- content-owned sci-fi omitted and then ignored its real character-word contract
- surfaced: same canonical `scifi_news` artifact as PBUG-20260721-10
- symptom: a requested 180-word episode sealed only 143 character-story words
  (146 including announcer). `meta.word_budget` lacked the target and band, and
  the shared final stamp marked `actual_drift=false` under its broad global
  `0.7..1.3` tolerance. The other qualified banks landed at 166--184 character
  words, inside the operator's approximately 163--200 campaign window
- root cause: Scifi Codex treated its advisory P3 word blueprint as sufficient
  and had no deterministic post-hygiene word-fit owner. Its final hygiene floor
  could shorten rows after the taste/factual loops. Separately, the shared tail
  read a producer target but always judged actual drift against global
  constants, ignoring a producer-stamped band even when one existed
- fix: Scifi Codex stamps an inclusive target-relative character-word contract
  before mutation. After every quality pass and the final hygiene scour, a
  bounded P10 compact line patch extends or compresses only selected character
  rows, runs the full merged-script graph/hygiene validator, and gets a fresh
  deterministic recount. Creative then technical attempts continue under a
  finite dynamic budget; exhaustion keeps the closest valid artifact with a
  truthful floor/drift receipt. Only then are ledger rows, counts, authorship
  hashes, and seals minted. The shared final stamp honors a valid producer band
  and uses `0.7..1.3` only as the legacy fallback. Announcer overhead remains
  separate
- verify idea: require target 180 to resolve to the relative 163--200 integer
  window, repair a 15-word character artifact into the 30-word window with a
  compact patch and fresh recount, then exhaust both slots and require the
  original valid story plus explicit drift. Prove the shared receipt consumes
  valid producer ratios and rejects malformed/reversed bands to legacy fallback
- bible-worthy: extends BUG-12.59
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-12 -- a real zero-second dialogue onset was reported as missing timing
- surfaced: same canonical `scifi_news` artifact as PBUG-20260721-10. ShotLock
  successfully overlaid eleven timed rows; the first spoken row had
  `start_s=0.0` and positive duration
- symptom: video logged the BUG-404 missing-overlay warning and ran the volume
  envelope fallback even though timing was present. That could manufacture an
  opening title window over immediate dialogue
- root cause: `_resolve_title_timing` correctly converted the onset to frame
  zero, then accepted it only when `first_dialogue_f > 0`. The valid zero
  sentinel shared the same branch as `None`
- fix: any non-`None` first-dialogue frame is known timing. Clamp a known onset
  to the nonnegative title window; zero yields no opening-card gap, preserves
  the zero receipt, and emits no missing-timing warning. Only `None` reaches the
  envelope fallback and BUG-404 diagnostic
- verify idea: pass a character row with `start_s=0.0` and require
  `first_dialogue_f=0`, no opening bounds, and no BUG-404 warning; retain the
  existing positive-head-gap test and missing-timing fallback
- bible-worthy: yes -- BUG-12.64
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-13 -- rendered music had no durable ledger timeline or downstream mirrors
- surfaced: canonical `scifi_news` artifact
  `signal_lost_the_fortress_of_reason_20260721_095038`, followed by a
  cross-bank inspection of the four already-qualified inline ledgers. The
  scifi ledger carried `music_open/music_inter/music_close` rows with
  `open/inter/close` placements and null path/timing. Each inline bank rendered
  three cue WAVs and audibly mixed its opening/closing bookends, but retained
  zero `music[]` rows, dropped the interstitial, and minted zero
  `mirrored_from=music` timeline rows
- symptom: the audio renderer could produce valid cue bytes while the durable
  ledger, video/title consumers, and OBS-bound wire ledger had no coherent
  account of which cue played where. A dialogue-anchored scifi interstitial was
  never inserted because SceneSequencer only resolved dedicated
  `music_inter` sentinel rows. Legacy manifest materialization in
  EpisodeAssembler would have been too late for SceneSequencer timing even if
  it had existed
- root cause: banks exposed different cue ID/placement dialects; four legacy
  producers authored only sentinel lines; the rendered cue manifest was not
  reconciled into the ledger before timeline mutation; SceneSequencer keyed
  interstitial timing only by sentinel anchor; EpisodeAssembler recognized
  bookends by cue ID instead of canonical placement; and ShotLock rehydrated
  lines/audio sections but not identity-gated music rows or assembler-owned
  mirrors
- fix: all content producers now cross the durable boundary using
  `opening/inter_NN/closing`, canonical placements, and explicit anchors.
  StableAudioTheme accepts historical aliases and gives synthesized legacy
  cues deterministic cue-spec identities plus ordered sentinel anchors. A
  shared manifest reconciler materializes or path-refreshes `music[]` before
  SceneSequencer writes timing and rejects any authored identity mismatch.
  SceneSequencer inserts interstitials before either a sentinel or ordinary
  dialogue anchor without consuming the dialogue's voice slot, then writes by
  cue ID. EpisodeAssembler idempotently reconciles, promotes even zero-offset
  scene timing, places bookends and chooses mirror roles by placement, and
  remains the sole mirror minter. ShotLock proves same-episode identity, lets
  disk win render-owned music fields only on a recomputed cue-spec match,
  appends valid legacy rows, and replaces only mirrors belonging to matched
  cues
- verify idea: cover canonical producer mappings and aliases; materialize a
  legacy manifest twice and require idempotence; reject a stale authored cue
  hash; insert an interstitial before ordinary dialogue while consuming every
  voice clip; position a sentinel cue; materialize/place/mirror all three cues
  in EpisodeAssembler; and require ShotLock to append valid legacy music and
  mirrors while rejecting a changed cue and its mirror
- bible-worthy: yes -- BUG-12.65
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-14 -- title rename moved assets but stranded their ledger identity
- surfaced: canonical `scifi_news` episode
  `signal_lost_the_chemical_throne_20260721_121728`, 2026-07-21. The audio
  assembler rendered and timed three canonical cues, persisted three music
  mirrors, and produced a byte-sealed master before SignalLostVideo renamed
  `pending_20260721_112330` to the final title
- symptom: the final ledger lived under the correct renamed directory, but all
  three `music[].wav_path` values still pointed into the deleted pending
  directory. The cue files themselves had moved and existed under the final
  directory. A later singleton save reduced the durable line set from twelve
  authored rows plus three assembler mirrors back to twelve rows. ShotLock
  retained the stale wire episode id and path block, while image dispatch and
  master mux escaped only through newest-ledger rescues. The video rendered,
  but the ledger was not a truthful map of the assets that produced it
- root cause: `Ledger.rename_episode` treated directory movement and identity
  movement as different operations: it renamed bytes and the ledger filename
  without recursively rebasing episode-local absolute JSON values. The
  singleton merge iterated only rows already present in memory, so
  assembler-owned disk-only mirrors had no join. The post-audio wire join did
  not let a same-freeze durable final id and path block replace stale pending
  values. ShotLock had no graph dependency on rename completion, and consumer
  recovery chose the newest sibling ledger instead of the active owner
- fix: rename now recursively rebuilds every absolute JSON string contained by
  the old episode root, for both the durable ledger and singleton, and atomically
  saves durable truth before advancing in-memory identity. External paths and
  prefix siblings remain unchanged; retry after a partial move is idempotent
  and durable-save failure is loud. Singleton saves preserve only validated
  EpisodeAssembler music mirrors: same immutable freeze, unique matching cue,
  recomputed cue-spec hash, legal role, master-timeline coordinates, and unique
  line id. One-sided, mismatched, reauthored, malformed, and ordinary disk-only
  rows are rejected. ShotLock lets the proven durable final episode id,
  `meta.paths`, media sections, and terminal paths win. Canonical link 284 gates
  ShotLock on SignalLostVideo rename completion. Image, clip, and master-audio
  recovery use the active in-flight ledger with available freeze/directory
  identity checks; no newest-mtime sibling selection remains
- verify idea: parameterize the rename over all six source banks. Move real cue,
  clip, master, path-block, and nested receipt values; require every episode
  pointer to resolve under the final root while an external path stays exact.
  Require freeze/readiness/authorship/master/cue hashes to remain unchanged and
  assembler mirrors to survive a later singleton save. Reject foreign or
  one-sided freezes, reauthored cues, malformed mirrors, ordinary disk-only
  dialogue, and mismatched consumer directory identities. Validate canonical
  link 284 and live-qualify both sci-fi banks through terminal OBS publication
- bible-worthy: yes -- BUG-12.66
- status: **LIVE-ADMITTED / ROOT-FIXED; focused cross-bank + consumer tests, full-suite (8,297 passed), Bug Bible, and canonical workflow gates GREEN; live `scifi_news` + `scifi_news_pro` requalification pending**

## PBUG-20260721-15 -- split word-count ownership falsely dirtied a correct live ledger
- surfaced: canonical `scifi_news` qualification prompt
  `1435f170-78fa-45ec-81a7-779b44533eb7`, pending artifact
  `pending_20260721_131620`, 2026-07-21. The source was the MIT News article
  `Study finds cell memory can be more like a dimmer dial than an on/off
  switch`. The run was stopped at TTS immediately after the freeze warning
- symptom: line `l003` contained the canonical surface `'off'—it's`. Its stored
  regex-derived `word_count=21` was correct, but Phase 0 and Phase 10 counted
  whitespace fields and reported only 20. The ledger froze
  `frozen_with_warns`; root `total_word_count=186` disagreed with the durable
  meta total `185`. The text, character count, and authored hashes were valid
- root cause: derived ledger metrics had several owners. Production row
  writers used an ASCII word regex, Scifi Codex used a slightly different
  smart-apostrophe regex, readiness/freeze/meta stamps used `str.split()`, and
  multiple repair/scrub/editor paths changed `text` without atomically
  refreshing both counts. Save aggregated stored fields instead of deriving
  them from canonical text, so stale or merely differently-tokenized values
  could survive into cast, scene, root, and budget receipts on every bank
- fix: one stdlib-only text-metrics leaf now owns character and word boundaries:
  ASCII hyphens plus straight/smart apostrophes remain intra-word, while en/em
  dashes are boundaries. Every confirmed durable text mutator calls the atomic
  text/count setter. Production save re-derives every row before rolling up
  cast, scene, root, and character/announcer meta totals, clearing stale zero-
  line aggregates. The freeze cascade preserves the raw Phase-0 diagnosis,
  then performs one count-only refresh after all permitted text mutation and
  before Phase 10. It does not alter canonical text, `text_for_tts`, hashes,
  authorship receipts, or seals. The freeze audit consumes the same helper
- verify idea: pin straight/smart apostrophe, ASCII-hyphen, en-dash, and em-dash
  counts including the exact live sentence. Parameterize all six banks through
  a save with deliberately corrupted row/root/cast/scene/meta counts and
  require complete self-healing. Show Phase 0 retaining an incoming mismatch
  while Phase 10 is clean after the final refresh. AST-audit production nodes
  so direct `row['text']` writes cannot bypass the atomic owner
- bible-worthy: yes -- BUG-12.67
- status: **LIVE-ADMITTED / ROOT-FIXED; exact six-bank + writer/freeze focused tests, full suite (8,315 passed), Bug Bible, and canonical workflow gates GREEN; live six-bank requalification pending**

## PBUG-20260721-16 -- whole-artifact P5 transport exhausted an otherwise healthy local writer
- surfaced: canonical `scifi_news` qualification prompt
  `569b20e5-0e28-4472-a04d-637ab019f19f`, pending artifact
  `pending_20260721_144919`, 2026-07-21. The source was the NASA NISAR /
  Hummingbird Antarctica item. The episode stopped in P5 before ledger or media
  production after 39 minutes of active local inference
- symptom: P5 attempt one reached its 2,970-token caller cap and returned
  truncated JSON. Attempt two returned a complete `ScriptArtifactV4` but
  invented line ID `l013`. The full typed-repair prompt then occupied 5,807
  tokens; the 8,192-token local context could reserve only 2,385 of the required
  2,970 output tokens, so the repair truncated and the three-attempt ladder
  exhausted. No OOM or idle GPU thrash occurred, but a structurally repairable
  story killed the episode
- root cause: the initial script pass made the model reserialize the accepted
  score's title, scene, cue, speaker, graph, boundary, fact, and neutral delivery
  metadata beside the only fields it actually authored: line IDs and spoken
  text. Its repair turn then reinjected the failed whole artifact plus almost
  the whole original request and duplicate schema authority. Output and context
  budgets therefore scaled with compiler-owned metadata, and a fresh LLM ladder
  could only retry the same oversized transport
- fix: P5 now transports a strict compact `ScriptTextDraftV4` containing only
  `{line_id,text}` rows. Python requires an exact unique bijection to the
  accepted line graph, maps by ID rather than response position, and compiles
  every mechanical `ScriptArtifactV4` field from the accepted score before the
  unchanged full graph, roster, fact, spoken-hygiene, and craft validation.
  Typed repair carries only the compact draft plus story, line-graph, fact, and
  word-steer authority; malformed prefixes are omitted. Every P5 call requires
  the complete prompt and full dynamic output reservation to fit. Exhaustion is
  a flat, truthful creative ladder followed by at most one fresh technical
  ladder, never recursion; the existing row-local A/B/C/deterministic spoken
  floor remains the final craft-liveness boundary
- verify idea: at the full supported 900-word, 24-row surface, tokenize the real
  base and semantic-repair chats with the exact on-disk Gemma 4 12B tokenizer
  and require prompt plus the full 3,208-token output reservation to fit its
  8,192-token context. Require byte-preserved text and compiler-owned graph
  fields; reject missing, unknown, and duplicate IDs; prove typed repair omits
  whole-request/schema echo and malformed raw prefixes; and prove the restart
  runs creative then technical exactly once before truthful exhaustion
- bible-worthy: yes -- BUG-12.68
- status: **LIVE-ADMITTED / ROOT-FIXED; exact-tokenizer maximum envelope, 165 focused lane/route tests, full suite (8,325 passed), Bug Bible, and canonical workflow gates GREEN; live six-bank requalification pending**

## PBUG-20260721-17 -- positioned video double-counted two audio crossfades at terminal mux
- surfaced: canonical `scifi_news` qualification prompt
  `a5e6e996-8f1e-4eb4-aff2-29486d4fd28c`, episode
  `signal_lost_the_fire_ant_bridge_20260721_163825`, 2026-07-21. The compact P5
  path passed on its first attempt and the run completed story, TTS, music,
  fifteen shots, silent composition, captions, and credits before the terminal
  master-audio mux rejected the body video. No OBS artifact was published
- symptom: the master audio was 114.5433 seconds / 5,498,077 samples at 48 kHz,
  while the silent body was 115.5600 seconds / 2,889 frames at 25 fps. With the
  valid 53.517-second credits declaration, video exceeded the allowed
  audio-plus-credits duration by 0.8997 seconds. The GPU remained around
  4.1--4.4 GB during the tail; this was deterministic timeline arithmetic, not
  VRAM thrashing
- root cause: the durable post-audio ledger correctly positioned the first
  drama row 0.5 seconds before the opening music ended and the closing music
  0.5 seconds before the last drama row ended. The render driver nevertheless
  defined final video length as the sum of every full per-shot render request,
  and the positioned planner emitted each full request even after a later row
  owned an earlier start boundary. The two intentional audio crossfades were
  therefore duplicated as one extra second / 25 visual frames. The filesystem
  master probe could grow the bad total but was forbidden to shrink it. The mux
  and credits declaration correctly refused to misclassify body drift as a
  credits tail
- fix: the clip manifest now separates full `render_target_frames` from the
  authoritative positioned `timeline_total_frames`. When every row has a
  position and the post-audio ledger owns `total_episode_dur_s`, the output
  boundary is `ceil(duration * fps)`; sparse legacy manifests retain their
  sequential sum. Positioned planning is stable by `(start_s, manifest order)`
  and gives each row only the visible interval ending at its requested end, the
  next row's start, or the timeline boundary, whichever comes first. This trims
  overlaps without stretching real gaps. QA reports requested, rendered,
  planned-visible, and overlap-trimmed frames separately. The actual master
  probe may reconcile a positioned total downward or upward, while sequential
  behavior remains grow-only. Terminal mux tolerance and credits ownership are
  unchanged
- sibling audit: exact Antigravity `gemini-3.5-flash-high` R2/R3 review in a
  clean detached worktree confirmed the shared tail affects all six banks.
  Sol grounded every claim against the real Windows files, discarded incorrect
  bank-specific and file-path claims, and retained sole coding/judgment
  authority. No workflow wiring change was required
- verify idea: build a positioned manifest whose full requests sum to 563
  frames but whose two crossfades and authoritative boundary yield 538 visible
  frames. Require stable slot ownership, no duplicated frames or stretched
  gaps, truthful overlap-trim telemetry, and a green visible-frame QA result.
  With real ffmpeg media, give a positioned 80-frame manifest a 2.1-second
  master and require exactly `ceil(2.1 * 25) = 53` output frames. Retain a
  legacy no-position manifest that preserves full sequential requests
- bible-worthy: yes -- BUG-12.69
- status: **LIVE-ADMITTED / ROOT-FIXED / PUSHED at `651118ef`; exact failed-ledger replay, 96 focused CPU/ffmpeg tests, full suite (8,328 passed), Bug Bible, and canonical workflow gates GREEN; resumed qualification surfaced PBUG-20260721-18 before media**

## PBUG-20260721-18 -- requested story length escaped every producer as advisory drift
- surfaced: canonical scifi_news qualification prompt
  f62c1177-a40a-4f9e-a9ac-f9c3bcfad717, pending episode
  pending_20260721_172001, 2026-07-21. The selected Ars Technica source was
  Let Tom Hiddleston be your guide to Pompeii's final day. The run completed
  P0/P3/P4/P3-rewrite/P5 and the dynamic quality passes, then was stopped before
  TTS/media when the writer logged the measurable delivery miss. The first root
  fix resumed qualification, but prompt 38d83284-49aa-48ba-aada-344b32f57110
  live-admitted the remaining liveness defect after 41:38: scifi_codex exhausted
  17 row-local cycles at 206 words against the same 289..356 ledger band
- symptom: the first final composed body held only 190 words for an explicit
  320-word request. The writer labeled the miss ADVISORY ONLY, stamped drift,
  and entered story reflection. After strict delivery became fatal, the second
  live run correctly stopped before media, but it still failed the whole episode
  when one model candidate exhausted its local word-fit attempts
- root cause: every producer family originally had a different length escape.
  The first pass repaired the shared integer contract, row-local progress, and
  pre-media hard stop, but conflated candidate liveness with episode liveness.
  A finite per-candidate repair ladder could still raise out of Codex, Fable, or
  the four inline banks. Fable also counted with split() instead of the canonical
  ledger tokenizer, and append-only hygiene receipts could leave a repaired row
  marked row_failed_mechanical so assembly silently skipped it
- fix: one dependency-light contract owns target, inclusive bounds, producer,
  canonical character count, exact text hash, and the final receipt. A
  WordFitLivenessController permits unlimited strict progress but retires a
  candidate after four consecutive stalls. Producers escalate row repair to the
  alternate LLM, then author a fresh complete producer-owned candidate. There is
  deliberately no outer model-output retry ceiling: generation or provider
  exhaustion remains pending/retryable/non-ready until a deterministic ledger
  candidate passes or the operator cancels. Codex fresh P5 candidates alternate
  producer priority; Fable rerolls and reseals its complete P3/P5 proof surface;
  inline banks re-author a complete staged row set before committing it. Fable
  now uses canonical_word_count throughout and stores current line-id hygiene
  state rather than stale append-only failure history. All lanes reject filler,
  repetition, fake commercials/products, markup, unsupported numeric/visual/
  canon claims, and Python-authored story padding. Subjective quality remains
  fail-open. No readiness or media consumer receives the candidate until the
  assembled ledger passes stamp_actual(require_in_band=True)
- live qualification continuation: prompt
  32b374e2-7c89-4d4a-bb8c-42e180571ecc remained queue-running beyond the
  temporary observer's two-hour wall clock and retired more than a dozen P5
  candidates without leaking a partial ledger downstream. It also proved that
  producer-priority alternation was not sufficient when both logical slots
  resolved to the same seeded Gemma backend: the two fixed P5 prompts replayed
  the same failures. Each complete reroll now carries a model-visible,
  monotonically unique candidate prompt nonce. The canonical observer accepts
  timeout zero as wait-until-terminal so monitoring cannot kill qualification
- sibling audit: exact Antigravity gemini-3.6-flash-high R2/R3 review covered all
  six banks in a clean worktree. Sol grounded the findings against the real
  Windows checkout, retained the candidate/episode liveness, canonical Fable
  count, and stale hygiene-state defects, and discarded the proposed five-reroll
  ceiling and soft ledger stamp because both violated the operator law. Hidden
  Opus produced no usable grounded findings and was discarded
- verify idea: pin 180 -> 163..200 and 320 -> 289..356; prove unlimited strict
  progress, four consecutive stalls per candidate, candidate retirement rather
  than episode failure, alternate-slot complete rerolls, and survival beyond
  five outer generation failures. Prove row-local immutability, canonical Fable
  counting, stale failure clearing, proof/hash resealing, final reflection
  ordering, fake-commercial/new-claim rejection, and hard downstream gating.
  Freeze must remain a read-only last backstop. Then qualify all six banks at
  320 words through audio, video, captions, credits, mux, and OBS publication
- bible-worthy: yes -- BUG-12.70
- status: **LIVE-ADMITTED / ROOT-FIXED / OFFLINE-GREEN IN WORKTREE; 177
  focused producer tests, full suite (8,348 passed / 33 skipped / 1 expected
  failure), BUG-12.70 (17 passed / 23 route-local skips / 3 expected failures),
  and canonical workflow gates (48 passed; byte-identical SHA-256
  f9d9c2c3a101ec607c9658456f6e191a164d8214be7b6d560bc68975d0511e9a) GREEN;
  commit/push and six-bank 320-word OBS qualification pending**

## PBUG-20260722-01 -- scifi_news codex seam lookup bypassed prompt_stages
- surfaced: canonical six-bank sweep `six_bank_sweep_20260722_162943_317`,
  `scifi_news` at 120 words, 2026-07-22
- symptom: the episode failed before model inference with
  `CodexPackContractError: P0 missing nonempty prompt seam
  'codex_fact_index_system'`; no episode or OBS asset was published
- root cause: the shared Codex-lane seam resolver used `getattr(pack, seam,
  None)` even though production seams are stored in the `StoryPack.prompt_stages`
  mapping. The resolver therefore returned `None` for every valid Codex seam.
- fix: the resolver now reads `pack.prompt_stages.get(seam)` and fails closed
  only when that mapped seam is absent or empty; a regression test proves a
  valid prompt-stage seam reaches the structured pass.
- verify idea: load every runnable bank's declared seams through its selected
  runner, assert the exact prompt text reaches the structured-call system
  message, and require the canonical `scifi_news` 120/320 legs to publish.
- bible-worthy: yes -- BUG-12.72; production pack seams must be accessed
  through the canonical mapping owner, not dataclass attribute guesses;
  executable regression coverage is present
- status: **LIVE-ADMITTED / ROOT-FIXED IN WORKTREE; post-fix scifi_news live
  requalification pending**

## PBUG-20260722-02 -- scifi_news P0 fact spans still exhausted the bounded repair ladder
- surfaced: fresh post-seam-fix focused qualification runs
  `six_bank_sweep_20260722_200609_509` (`scifi_news` at 120 words, prompt
  `59256a76-bd44-447e-88a2-fab5fe2c350f`) and
  `six_bank_sweep_20260722_201449_793` (`scifi_news` at 320 words, prompt
  `4b9f096b-3d8c-4c89-9f00-8924ad0e177c`), 2026-07-22
- symptom: P0 failed after two structured attempts on both distinct source
  payloads because `F01` returned a quote that did not equal the declared
  `full_text[start:end]` slice; no ledger or OBS asset was published. The
  320-word run was the same class after the prompt-stage seam fix and the
  120-word run also had no word/length gate involvement
- root cause: the accepted source payload is already normalized at admission
  and the literal-span validator is correctly fail-closed, but the technical
  model plus bounded repair still returned a non-literal or unrelated quote
  instead of repairing the exact field/start/end/quote identity. This extends
  the existing source-span contract defect covered by BUG-11.35; it is not a
  new whitespace-ingestion defect and not a prose/length gate
- fix: **KIBITZ-HARDENED / IMPLEMENTED IN WORKTREE**. The first mechanical fix
  is an explicit literal identity instruction
  `payload[field][start:end] == quote`. The shared structured-call boundary
  then gets one direct, bounded alternate repair owner with a hard context
  ceiling, original post-validator reuse, owner/backend/rung/nonce journal
  fields, and explicit terminal disposition. P0 now wires the creative owner
  through that one-shot branch. The remote RTX 4060 Qwen worker
  at `10.55.0.2:1234` completed the four scoped read-only reviews; the RTX
  5080 remains reserved for ComfyUI and must not load this worker. Any patch
  stays out of live qualification until the accepted-object boundary is
  proven against the captured production payloads
- verify idea: replay both captured P0 failures with the exact normalized A0
  payloads, require a repair whose quote is byte-identical to the selected
  slice, preserve the payload digest, and then requalify both canonical
  `scifi_news` legs through `RESULT SUCCESS`, `obs_publish OK`, and exact
  episode/OBS assets
- bible-worthy: extends BUG-11.35; no new portable rule
- status: **LIVE-ADMITTED / ROOT-FIXED IN WORKTREE; focused/canonical offline
  gates green, live 120/320 requalification pending**

## PBUG-20260723-01 -- six-bank campaign trusted exit 0 over canonical RESULT FAIL
- surfaced: live six-bank 120-word viz campaign
  `six_bank_viz_120_20260723_20260723_011138`, `scifi_news` leg, prompt
  `cde10c6d-3b70-4732-8179-4b18c8bcd933`, 2026-07-23
- symptom: the child stdout contained `[canonical-api] RESULT FAIL`, but the
  campaign receipt recorded `status=PASS`, `exit_code=0`, and `queue_empty=true`.
  The campaign therefore reported `6/6 PASS` despite a live P0 fact-span
  failure and no valid qualification evidence for that leg
- root cause: the PowerShell campaign wrapper inferred terminal success from a
  zero child exit code and an empty ComfyUI queue. It did not consume the
  canonical runner's explicit terminal result, so a contradictory `RESULT FAIL`
  was invisible to the receipt owner
- fix: the wrapper now delegates verdict construction to
  `scripts/otr_campaign_receipt.ps1`. A leg is PASS only when exit code is zero,
  the queue is empty, and the latest explicit terminal marker is
  `RESULT SUCCESS`; missing/contradictory markers are recorded as FAIL with the
  observed terminal line and reason
- verify idea: feed captured `RESULT FAIL` stdout with exit code zero, missing
  terminal output, `RESULT SUCCESS` with a nonzero exit, and a clean success
  through the helper; require truthful verdicts and nonzero failure exits
- bible-worthy: extends BUG-12.50's terminal-evidence contract; no new
  portable rule
- status: **LIVE-ADMITTED / ROOT-FIXED IN WORKTREE; helper regression GREEN;
  six-bank live requalification pending**

## PBUG-20260723-02 -- the 8GB Wan tier's low-VRAM launch contract never reached a production leg
- surfaced: 2026-07-23 overnight media qualification, matrix leg
  `wan_8gb__lumina_image__media_archive` (`model_coverage_wan/receipts.json` +
  `server_wan.log`; staged in `docs/2026-07-23-video-failure-inventory.md`)
- symptom: terminal `FAIL` at `OTR_VideoRenderBatch` -- `wan_ti2v` received a
  177-frame request while the cost model afforded 30 frames at the observed
  free VRAM. No silent resize happened, which is correct; the requested
  832x480 / 17-frame low-VRAM lane simply never applied to the leg
- root cause: the 17-frame ceiling existed only in the profile's
  `launch.env.OTR_WAN_TI2V_MAX_FRAMES`, and `eng_wan_ti2v._floor_length` read
  that env var as its ONLY channel. A production episode leg is submitted to an
  ALREADY-BOOTED server, so `launch.env` can never reach it -- any leg not
  booted through `scripts/otr_headless_canonical.ps1 -Profile otr_8gb_wan`
  inherited the 177-frame engine max. The profile's other declaration,
  `render.frame_budget: 17`, maps to `OTR_VideoRenderBatch.frame_count`, which
  is diagnostic-harness-only ("Ignored in mode=episode" per its own tooltip),
  so the tier's contract was inert in production on both channels
- fix: new OPTIONAL profile key `video.max_render_frames` (0/absent =
  unpinned) travelling the same proven channel the device/dtype policy uses --
  profile -> `OTR_VideoDirector.max_render_frames` widget (appended LAST,
  canonical ships 0) -> v2 policy -> ShotLock ledger `video` section ->
  `render_driver.build_episode_render_policy` -> `MotionEngineBase.prepare` ->
  `_floor_length`. Env pin still outranks it; every other tier omits the key
  and is byte-for-byte unchanged. Beat frame targets are untouched: the ceiling
  caps what the ENGINE renders (then ping-pong-extended to the beat's full
  audio length), never what the episode plays
- verify idea: with free VRAM affording ~30 frames at 832x480, a 177-frame beat
  must raise `MotionBudgetError` UNPINNED and return 17 with the tier ceiling
  on the ledger; and an unpinned tier must still return 177 (no lane capped by
  the fix). Covered by `tests/test_remaining_video_contracts.py`
- UPDATE 2026-07-27 (B3): the "then ping-pong-extended to the beat's full audio
  length" clause above is still exactly right for WAN and is now only HALF the
  meaning of `video.max_render_frames`. For `ltx_8gb` -- the sole member of
  `frame_contract.PLANNING_CAP_ENGINES` -- the same ledger key is a coverage
  PLANNING cap: it narrows the contract `otr_shot_lock._stamp_coverage_plan`
  partitions against, so the beat is covered by real chained clips of at most
  that length instead of one short render padded out. WAN is deliberately
  excluded, because applying it before `partition_beat()` would turn every WAN
  beat into a pile of 17-frame renders and undo this very fix. Anyone reading
  this entry as the definition of the key should read
  `docs/2026-07-27-b3-qa-findings.md` and `frame_contract.effective_frame_contract`
  alongside it
- UPDATE 2026-07-27 (B6): SECOND application of this entry's portable rule to
  the same tier, with the OPPOSITE remedy, because there was no channel to fix.
  B3 gave the ltx_8gb CEILING a profile -> ledger channel. The tier's RECIPE --
  T5 device, tiled decode, the sampling knobs, the negative conditioning, the
  tile geometry -- has no channel at all: the profile schema accepts only
  `device_policy`, `dtype_policy` and `max_render_frames`, and
  `otr_8gb_ltx.json`'s `launch.env` is `{}`. So the recipe is now FROZEN IN
  CODE (`eng_ltx_8gb.LTX8_RECIPE_V1`); those env vars bind only under an
  explicit `OTR_LTX_8GB_PREQUALIFICATION` consent act, and a run that sets it
  stamps a `+prequalification` recipe receipt so a measurement artifact is
  never mistaken for a production one in `meta.render_engines`.
  A NEW TESTABLE COROLLARY this produced, which the original entry does not
  state: **a knob that cannot bind must be IGNORED, never FATAL.** The first
  draft parsed the demoted vars before discarding them, which meant a stale
  `OTR_LTX_8GB_STEPS=not-a-number` in a long-booted server's environment would
  raise MALFORMED_CONFIG and kill a leg over a value with no effect on it --
  the same defect wearing the opposite mask. Presence is named in a warning;
  nothing outside the consent act is parsed
- bible-worthy: yes -- the portable rule is that a contract declared only in a
  process-launch environment cannot bind work submitted to an already-running
  server; a per-tier constraint has to ride the artifact the run loads. B6 adds
  the corollary above (ignore, never fail, on a knob that cannot bind) and the
  receipt rule (a run under a consent act must mark its own artifacts)
- status: **ROOT-FIXED + suite/contract GREEN; live 8GB requalification leg
  still owed (no GPU run authorized in this window)**

## PBUG-20260729-01 -- P5 markup defect hid behind the compile refusal, and the one repair shot died on it
- surfaced: the live 45-word campaign, leg `ltx_8gb` (2026-07-29 06:46), headless
  canonical run. `P5 failed: ... disposition=primary_ladder_exhausted; last
  error -> PostValidationError: l001: spoken text is production markup`
- symptom: the writer died before any video engine ran. Attempt 1 (base) was
  told only "P5 compact draft line IDs do not exactly cover the accepted graph
  (missing=[], unknown=['l011','l012','l013'])". Attempt 2 (typed repair) did
  exactly what it was told -- dropped the three invented IDs -- and was then
  refused for `l001: spoken text is production markup`, a defect that was
  sitting in attempt 1's output and had never been mentioned. The ladder was
  spent: `structured_call` deliberately does not retry a repair that was
  schema-valid but content-invalid.
- root cause: the P5 post-validator surfaced ONE defect at a time.
  `compile_script_text_draft` raises on ID coverage before any markup check can
  run, so a compile refusal hid every markup defect behind it; and
  `_validate_p5_structure` returned on the first offending line, so even a
  clean-ID draft with three bad lines would have burned the shot on line one.
  A validator that reports serially is incompatible with a ladder that grants
  one repair attempt.
- fix: `3b49d3f8` -- `_validate_p5_structure` reports EVERY offending spoken
  line (a single finding still yields the bare historical message, so existing
  pins hold); and when `compile_script_text_draft` refuses, the RAW draft rows
  are scanned by the new `_p5_raw_spoken_findings` and those findings ride
  along with the compile refusal. Only rows the score owns and marks spoken are
  judged -- an invented ID has no speaker_role, and judging its text would be
  inventing a contract.
- verify idea: drive the P5 post_validator with a draft that BOTH misses the
  graph and speaks production markup, and assert the returned string names both
  defects. `tests/test_p5_repair_sees_every_defect.py` does this; mutation E9
  (the compile refusal drops the markup findings again) and E8 (the structure
  validator reports only the first bad line) both die against it.
- bible-worthy: yes -- the portable rule is that **a validator feeding a
  bounded repair budget must report every defect it can see in one pass.**
  Serial reporting silently converts an N-defect artifact into N required
  attempts, and any ladder shorter than N then fails for a reason that looks
  like a model problem and is actually a reporting problem.
- status: **ROOT-FIXED; suite/mutation GREEN; live requalification owed --
  `ltx_8gb` must be re-run and reach a video engine**

## PBUG-20260729-02 -- a degenerate P5 generation burns 24 minutes and bypasses the whole retry ladder
- surfaced: the live 45-word campaign, leg `ltx_audio_in` (2026-07-29 06:46 ->
  07:11, 1449s), headless canonical run
- symptom: `P5 failed: prose generation exhausted the full remaining
  provider/context capacity (14697 output tokens after a 1687-token prompt);
  the partial artifact is not eligible for a prose or structural reroll`
  (`PromptContextOverflowError`). One leg, 24 minutes, no video engine reached.
- root cause: TWO layers, and only the second is in doubt.
  (a) The model never stopped adding lines. `ScriptTextDraftV4.lines` declares
      `max_length=_SCRIPT_TEXT_DRAFT_MAX_LINES`, but that ceiling is the GLOBAL
      one (`_RADIO_SCORE_MAX_BEATS * _RADIO_SCORE_MAX_LINES_PER_BEAT`), not this
      episode's accepted line count, and the constrained decoder did not force
      the array closed at it. Nothing told the decoder the real ceiling. This is
      the same pathology as PBUG-20260729-01's `unknown=['l011','l012','l013']`,
      one step worse -- there the model invented three extra lines, here it
      never stopped. `repetition_penalty` was already at its gentle 1.03, so
      this is a constrained-decoding ceiling problem, not a sampling one.
  (b) The refusal is raised on attempt 1 of 3 and `PromptContextOverflowError`
      is a `RuntimeError`, which `structured_call` does not catch -- so a
      runaway consumes one attempt and then BYPASSES the remaining two rungs of
      a ladder that exists to absorb exactly this. The refusal's own text says
      the partial ARTIFACT is not eligible for a reroll, which is right; it does
      not say a fresh call at a lower temperature is ineligible, and the
      structural-retry rung (0.32) is the standard remedy for a degenerate loop.
- fix: **NOT FIXED.** Deliberately left open rather than changed unattended:
  every candidate touches a ratified fail-loud contract or the writer's
  sampling, and getting it wrong is worse than one lost leg the campaign
  watchdog already re-runs. Two candidates, for whoever picks this up:
    1. Bound the constrained decoder by the ACCEPTED line count for this
       episode rather than the global product ceiling, so a runaway is
       structurally impossible instead of merely caught. Preferred -- it
       removes the failure rather than recovering from it.
    2. Let a runaway under `_otr_reserve_remaining_output_capacity` advance the
       ladder instead of being terminal. Note the trap: the typed repair
       factory would be handed the ~14,700-token truncated output as
       `failed_output`, so the repair prompt itself must be bounded first.
  Do NOT "fix" this by capping P5's output budget to the word target -- THE LAW
  is explicit that requested word length and actual word count are telemetry
  only and may never reject or block an episode, and `output_budget_mode:
  "provider_capacity"` is that decision written down.
- verify idea: a fake slot_fn that returns exactly `effective_max_new_tokens`
  tokens without an EOS, asserting the pass makes a SECOND call at the
  structural-retry temperature instead of raising on the first.
- bible-worthy: yes -- the portable rule is that **a failure raised as a
  RuntimeError inside one rung of a retry ladder silently cancels the rungs
  below it.** Any bounded-retry design has to classify its own terminal errors
  explicitly, or the budget it advertises is not the budget it spends.
- status: **OPEN -- diagnosed, not fixed. Live: 2 occurrences in 13 legs
  (`ltx_audio_in` 1450s, `still_word` 1420s -- both ~24 minutes of GPU time
  spent inside a single P5 base call before the refusal).** Both ran to the
  full remaining provider capacity (14697 and 14359 output tokens), which is
  the signature: the model never stops adding lines, and the array's declared
  `max_length` is the GLOBAL product ceiling rather than this episode's
  accepted line count, so nothing forces the array closed.
- **OPERATOR RULING 2026-07-29 (supersedes the "candidates" framing above):**
  "the writer should not be allowed to kill the run, it just needs to fix the
  ledger" -- restated: "the writer should never veto, the writers should keep
  on passing in a loop to agents to clean up the ledger." Candidate 2 is
  therefore the RULED DIRECTION, not an open design question, and the rule is
  general rather than runaway-specific: a writer pass failure must degrade to a
  workable ledger and never terminate the episode. The recorded trap still
  binds -- the typed-repair factory would be handed the ~14,700-token truncated
  output as `failed_output`, so the repair prompt must be bounded before that
  path opens -- and PBUG-20260729-03's hard-limit refusal is the same trap from
  the other side, so both belong in one design. THE LAW is untouched: word
  length stays telemetry. PARKED by the operator until the video pipe engines
  work as expected; queued as the next step in `docs/GO_FORWARD_PLAN.md`
  ("THE WRITER NEVER VETOES").

## PBUG-20260729-03 -- a P0 repair is refused for being too big to attempt
- surfaced: the live 45-word campaign, legs `mesh_stage` (07:23, 182s) and
  `still_flat` (07:26, 208s), headless canonical runs
- symptom: `P0 failed: P0 repair context is 16796 bytes, over the hard limit
  14336` -- and on `mesh_stage`, `P0 failed: [OTR_StructuredCall]
  'scifi_codex:P0' failed after 3 attempt(s); disposition=repair_owner_exhausted`.
  Two legs, no video engine reached.
- root cause: the repair context the pass BUILDS is larger than the bound it is
  allowed to spend, so the repair is refused before it is attempted rather than
  being trimmed to fit. The bound exists for a good reason (an unbounded repair
  prompt is how a context window gets eaten); what is missing is the step that
  makes the context fit it. Same family as PBUG-20260729-02: a budget that
  refuses instead of degrading, discovered only on live GPU time.
- fix: **NOT FIXED.** Folded into the operator's "writer never vetoes" ruling
  above and parked with it -- it is the same design, and fixing the two
  separately would produce two different answers to one question.
- verify idea: build a P0 repair context deliberately over the bound and assert
  the pass still returns a workable artifact, with a receipt naming what was
  trimmed -- rather than raising.
- bible-worthy: yes -- same portable rule as PBUG-20260729-02, seen from the
  other side: **a bound that refuses is not a budget, it is a veto.** A limit
  on a repair context has to come with the trim that makes the context fit it.
- **CORRECTION 2026-07-29 (this entry's first draft was wrong on two counts;
  the original text is kept below so the error is auditable).** A grounded read
  of the source established:
  1. **There are TWO checks with TWO different bounds, not one.** INNER bound
     **14336** in `compact_p0_repair_context`
     (`nodes/_otr_scifi_p0_contract.py:223-226`) -- and 14336 is not a constant
     at all, it is `max(1024, max_bytes - 2048)` computed at
     `nodes/_otr_scifi_codex.py:2253` from
     `P0_REPAIR_CONTEXT_MAX_BYTES = 16_384`. OUTER bound **16384** at
     `nodes/_otr_structured_call.py:1197-1201`, measured AFTER
     `_prompt_with_schema_contract` appends the schema instruction (`:1192`).
     `still_flat` (16796) hit the INNER check; `still_pan` (16735) hit the
     OUTER one. They are NOT the same failure, and the original claim that both
     sat "within 61 bytes of each other against a fixed 14336 bound" was an
     artifact of reading two different bounds as one.
  2. **`mesh_stage` and `viz_camera` are a DIFFERENT ROOT CAUSE and do not
     belong to this bug.** `disposition=repair_owner_exhausted` is set when
     `repair_attempted = True` (`nodes/_otr_structured_call.py:1180`) and the
     ALTERNATE owner's own output then fails validation -- meaning the repair
     context FIT, the alternate model ran, and its answer was rejected. That is
     model quality. `still_flat`/`still_pan` never reach the alternate model:
     their raw `ValueError`s skip the `except StructuredCallFailedError` clause
     (`nodes/_otr_scifi_codex.py:1685`) and land in the generic handler
     (`:1698-1712`), where the journal disposition resolves to a THIRD value,
     `repair_context_builder_failed` (`:1703-1706`).
  So this bug's live count is **2 occurrences, not 4**.
- **THE ACTUAL STRUCTURAL DEFECT, measured rather than inferred:** the reserve
  at `nodes/_otr_scifi_codex.py:2253` is a literal `2048`, but the overhead it
  is reserving for is `schema_shape_instruction(FactIndexV4)` at **3809 bytes**
  plus the fixed CRITICAL P0 REPAIR system text at **302 bytes** plus a join
  newline = **4112 bytes**. The reserve under-provisions by **2064 bytes**.
  Arithmetically certain: any inner render above ~12,272 bytes passes the inner
  check BY CONSTRUCTION and is then guaranteed to fail the outer one. A guessed
  literal reserve drifted out of sync with the thing it reserves for.
- **AND THE TRIM HELPERS ALREADY EXIST, UNWIRED.** `p0_source_char_budget`
  (`nodes/_otr_scifi_p0_contract.py:58-70`) and `p0_source_chunks` (`:72-121`)
  are defined, documented and exported, with **ZERO call sites** in the
  non-vendored codebase. `_p0_evidence_projection`
  (`nodes/_otr_scifi_codex.py:1104-1138`) dedupes by substring containment only
  and caps nothing, and `nodes/_otr_source_payload.py:317` collapses whitespace
  "without truncating authored source text" -- so a long RSS article body
  reaches the repair context unbounded. `failed_artifact` is likewise echoed
  untruncated, where the generic `default_repair_prompt_factory` truncates to
  `failed_output[:400]`.
- status: **OPEN -- diagnosed, parked under the 2026-07-29 operator ruling.
  Live: 2 occurrences in 17 legs** (`still_flat` 208s, inner check;
  `still_pan` 173s, outer check).
  ORIGINAL (WRONG) TEXT, kept for the record: "Live: 4 occurrences in 13 legs
  -- mesh_stage 182s and viz_camera 165s (disposition=repair_owner_exhausted),
  still_flat 208s and still_pan 173s. The two measured sizes land within 61
  bytes of each other against a fixed 14336 bound."
- **CAMPAIGN-WIDE RATE, 2026-07-29 (the number that should inform when this
  gets unparked):** across the first 13 legs of the live 45-word run, SEVEN
  died inside the writer -- 2x PBUG-...-02 and 4x PBUG-...-03 plus one
  PBUG-...-01 -- against 5 engine-side failures (all since fixed) and ONE leg
  that produced a finished episode. That is a **54% writer failure rate**, and
  it is the dominant blocker on the campaign, not the video engines. The narrow
  fix here (trim the repair context to fit its own bound) is SEPARABLE from the
  full "writer never vetoes" redesign and is the single highest-yield unblock
  available; it is not being taken because the operator parked writer work
  until the engines are proven, and that call stands until the operator changes
  it. Recorded here so whoever unparks it does not have to re-derive the cost.

## PBUG-20260801-01 -- the gemma row understated its own model by 32x, so the writer could never fit
- surfaced: the live 45-word campaign, every `otr_g4_*` leg, headless canonical
  runs. Six engines, zero episodes -- each leg died at `OTR_LedgerScriptWriter`
  before any video engine was reached.
- symptom: `GGUF unsloth/gemma-4-12b-it-GGUF cannot fit the complete requested
  output: requested_output=2800, provider_output_cap=512`, and when the context
  was raised to compensate, `effective n_ctx 8192 (from policy.gguf_n_ctx) is
  outside [512, 4096] for this row -- NO clamp`.
- root cause: TWO placeholder defaults, each below the pipeline's own contract.
  1. The catalog row declared `context_window=DEFAULT_CONTEXT_WINDOW` (4096)
     while the GGUF file itself declares `gemma4.context_length = 262144` -- the
     row understated the model by 32x. P0 needs `_P0_PROMPT_OVERHEAD_TOKENS`
     2600 + `_P0_BASE_OUTPUT_TOKENS` 2800 = 5400, so P0 was STRUCTURALLY
     impossible on this row: no setting could satisfy it.
  2. `DEFAULT_OUTPUT_TOKENS_CAP` was 512 against P0's 2800 request. The other
     backends never had this -- `_otr_comfy_backend` 8192, `_otr_openrouter_backend`
     16384. 512 was the outlier, not the rule.
- fix: row `context_window` 4096 -> 8192 (P0's own `_P0_LOCAL_CONTEXT_CAP`, the
  value its contract was written against, not a guess), and
  `DEFAULT_OUTPUT_TOKENS_CAP` 512 -> 4096 (bounded by the window: 8192 - 2600
  overhead = 5592 usable). Cost checked rather than assumed: KV at 0.7 GB/1k is
  5.60 GB, plus 6.63 GB of Q4_K_M weights = 12.23 GB, ~2.3 GB under the 14.5 GB
  tier ceiling. Commits 805123ea + 76c9f565.
- verified: `fastwan_8gb` 45-word canonical leg, RESULT SUCCESS in 2433 s,
  published 1920x1080 / 3036 frames / 121.44 s / AAC stereo, coverage 70.68 s
  audio vs 71.72 s video across 7 clips.
- **the part worth remembering:** between the first fix and the second, the ONLY
  thing keeping the writer alive was exporting `GEMMA4_12B_MAX_NEW_TOKENS=3072`
  at server boot. That is a dead channel of the PBUG-20260723-02 class -- the
  env binds at BOOT, so the next restart that forgets it silently restores the
  failure, and the symptom comes back looking like a NEW bug. A live pass that
  depends on remembering an export is not a fixed bug. The second boot was run
  deliberately WITHOUT the var to prove the default carries it alone.
- also worth remembering: two fixes failed before this one because both turned
  knobs that could not bind -- `n_ctx` when the limit was the output cap, then
  `n_ctx` past a ceiling the row would not allow. The row was never questioned
  until the GGUF metadata was read directly. **When two settings in a row fail
  to move a limit, stop tuning and go read what the artifact itself declares.**
- bible-worthy: yes -- **a placeholder default that sits below the caller's own
  contract is a structural refusal, not a configuration problem.** Any registry
  row describing a model's capacity must be derived from, or checked against,
  the artifact's declared metadata; a hand-set default silently caps a model at
  a fraction of what it can do, and no amount of caller-side tuning can reach
  past it.
- verify idea: assert every GGUF catalog row's `context_window` is <= the
  context length its own file declares AND >= what the P0 contract requires, so
  a row that cannot host the pipeline's own pass fails at test time rather than
  on live GPU minutes.

## PBUG-20260802-01 -- ltx_video declared 21 legal lengths for an engine that renders exactly one
- surfaced: the live 45-word campaign, leg `ltx_video` (2026-08-01 23:20, headless
  canonical run). Died at 11.8 minutes -- AFTER the writer, the cast, the TTS and
  the music had all been rendered and paid for. No obs asset.
- symptom: `RenderError: shot shot_music_opening_001 segment 1 rendered 169
  frame(s) but its plan asked for 89 (a surplus of 80). NO FALLBACK -- the plan's
  count is what this segment's audio slice was cut against, so assembling a
  segment of any other length makes the beat drift against its own audio.`
  Preceded in the same log by the engine's own warning:
  `[eng_ltx_video] frame ask 89 below the decode floor 169 -- raising`.
- root cause: the adapter's DECLARATION disagreed with its own RUNTIME.
  `frame_contract` declared `min_frames=9, max_frames=169, quantum=8` -- 21 legal
  rungs -- while `_ltx_frame_length` raises every ask below
  `_LTX_DECODE_FLOOR_DEFAULT` (169) up to it, and `_LTX_MAX_FRAMES_DEFAULT` is
  ALSO 169. The floor equals the cap, so the adapter emits exactly ONE length and
  20 of its 21 declared rungs do not exist. The planner believed the declaration,
  split a beat into 89-frame segments, and the engine could not produce them.
  The refusal was CORRECT; what it was checking against was wrong.
- why it read as a regression: `ltx_video` shipped for months in single-clip
  mode, where nothing ever asked it for a non-169 length. Only coverage planning
  (2026-07-25) can ask, so only coverage planning could expose it. The operator's
  "ltx_video always worked, check a week ago" was accurate.
- fix: declare the truth -- `min_frames=169, max_frames=169`, as LITERALS. Not
  derived from the constants: a FrameContract is STATIC because stills are minted
  against it before the render phase, so it must never track a value that can
  move underneath it (`test_the_LTX_ceilings_do_not_silently_follow_their_env_overrides`
  rejected a first draft that did exactly that). Commit 53fcebff.
  Then TWO more channels that could reintroduce it, both found by the kibitz
  panel (codex gpt-5.6-sol + antigravity), both closed:
  1. `assert_env_matches_contract` raises `ContractEnvConflict` when
     `OTR_LTX_MAX_FRAMES` / `OTR_LTX_MIN_DECODE_FRAMES` disagree with the
     declaration -- wired into BOTH graph builders, since either can resolve a
     length. Commit 8c5449db.
  2. `render_canvas = (832, 480)` declared, because the decode floor's own
     comment ties it to "this canvas". Without a declaration,
     `OTR_LTX_RENDER_CANVAS` could move the canvas at boot and invalidate the
     static contract with no code change; `declared_render_canvas` is applied
     LAST in `build_request_from_shot` precisely so a declaration wins.
- verified: plan-vs-engine agreement on the PRODUCTION call path (`join_mode_for`,
  not a forced mode) -- beats 17/89/168/169 take `single`, 170/250/338/442/530
  take `chain` with 2-4 segments, and every segment satisfies
  `_ltx_frame_length(render_frames) == render_frames`. Suite 8253 passed.
  **A live leg has NOT yet re-run -- the fix lands in code the running server
  loaded hours earlier, so it is proven in arithmetic only until the overnight
  driver restarts the server and re-runs it.**
- bible-worthy: yes. **A capability declaration is a promise the runtime must
  keep, and an OVERSTATED one is worse than none.** An understated contract
  merely wastes capability; an overstated one converts a plannable component
  into a GUARANTEED late failure, because the planner commits work against the
  declaration and only the render discovers the lie. Three channels can break the
  promise and all three need closing: the declaration itself, an environment
  override read at runtime, and a second dimension (here canvas) the bound
  silently depends on.
- verify condition (automatable, and implemented): feed each adapter's own
  declared minimum and maximum through its own length resolver and require them
  to come back unchanged -- `test_a_declared_MINIMUM_is_a_length_the_adapter_can_actually_render`.
  Currently covers `ltx_video` only, because each adapter resolves length
  privately; the general version needs the shared `resolve_render_frames`
  interface both panel lanes converged on.

## PBUG-20260802-02 -- the writer casts two characters and writes lines for one
- surfaced: the live 45-word campaign, 2026-08-02. TWO legs, two different
  symptoms, one underlying fault:
  * `wan_ti2v` (02:35, 2.7 min): `[scifi_fable2] pass 'script' failed after 4
    attempt(s): markup ladder exhausted`, with `UNKNOWN_SPEAKER` on every line
    of both characters AND `CAST_MEMBER_SILENT: Commander Vance` /
    `CAST_MEMBER_SILENT: Pilot Elara`.
  * `ltx_video` (02:47, 2.2 min): `OTR_CastLock: freeze cascade stamped
    freeze_verdict='needs_full_rerun'`, from
    `[LFC] read-only structural validation failed under content_owned_readonly:
    content_authorship: line proof coverage mismatch: missing=[]
    extra=['shot_001_b2', 'shot_001_b4', 'shot_002_b2', 'shot_002_b4',
    'shot_003_b2']`.
- root cause: the ledger shows those five "extra" rows are EXACTLY the second
  character's lines, and every one carries `len(text)==0`:
      shot_001_b1  len=111  speaker=c02
      shot_001_b2  len=0    speaker=c03   <- extra proof
      shot_001_b3  len=196  speaker=c02
      shot_001_b4  len=0    speaker=c03   <- extra proof
  The phase-2B skeleton allocates dialogue slots for BOTH cast members, the
  composition fills only c02, and c03's rows are left empty. `_voiced_rows`
  (`_otr_content_authorship.py:28`) excludes a row with empty text, so the
  authorship proofs -- minted while those rows still had text -- no longer
  match the live voiced set, and the read-only structural validation refuses.
  So the writer produces an effectively SINGLE-character play from a TWO-
  character cast, and two different downstream gates catch it in two different
  ways.
- **the two symptoms are the same fault, which is why this is one entry.** The
  `UNKNOWN_SPEAKER` half was a separate, real parser gap (the role parenthetical,
  fixed in afe53c7c); fixing it did not fix this, it merely let a script that
  previously died at the parser reach the freeze gate, where the silent second
  character is what fails. Fixing a blocker upstream does not fix the thing it
  was hiding.
- fix: **NOT FIXED.** Recorded at 03:00 with the operator asleep and the GPU
  mid-campaign. It is a story-QUALITY defect in the composition pass, not a
  renderability bug, and the right fix is upstream of everything touched
  tonight.
- verify idea (automatable): after composition and before the freeze gate,
  assert every cast member the skeleton allocated a slot for has at least one
  non-empty line -- and fail there, by name, rather than letting an empty row
  reach an authorship proof and surface as a coverage mismatch five stages
  later. The current failure names `shot_001_b2` when the real answer is
  "character c03 never got any dialogue".
- bible-worthy: probably -- **an artifact minted from state that a later stage
  can still invalidate is a proof of nothing.** The authorship receipt is built
  from rows that are voiced AT THAT MOMENT; nothing stops a later pass emptying
  one. The portable rule is to build such a proof at the same barrier that
  freezes the state it describes, or to re-derive it at the gate.

## PBUG-20260802-02 CORRECTION (same day, before any fix was written)
The entry above claims the two legs were "the same fault, which is why this is
one entry". **That is not established, and the difference changes the fix.**
Grounded from the ledgers and the server log:

* `wan_ti2v` ran the **`scifi_fable2`** lane. It failed with `UNKNOWN_SPEAKER`
  plus `CAST_MEMBER_SILENT` -- and that lane's own gate is what caught it
  (`_otr_scifi_fable2.py:2306`, "speaker set != cast rows", plus the parser
  defect). The gate WORKED. What failed upstream of it was the writer producing
  a play in which a cast member never speaks, and the repair ladder exhausting.
* `ltx_video` ran the **`scifi_news_pro`** lane, whose ledger meta says in as
  many words: `"pack for bank 'scifi_news_pro' declares NO line_composer_system
  seam -- the lane owns its own content loop"`. There is NO equivalent gate on
  that path, so the empty rows travelled all the way to the freeze gate and
  surfaced as a line-proof coverage mismatch naming `shot_001_b2`.

And the cast row that was silent is the tell: `c01=ANNOUNCER, c02=Elias,
c03=**The Relay**`. The lane cast a RELAY -- a machine, not a speaking part --
and then, reasonably, wrote it no dialogue. So the `scifi_news_pro` root cause
is most likely CASTING a non-speaking entity, not a composition pass dropping
lines it was asked to write.

What survives from the original entry: an artifact minted from state a later
stage can still invalidate is a proof of nothing, and the named-gate verify
condition is right and lane-agnostic. What does not survive: "one fault, two
doors", and the implication that fixing the fable2 parser gap had anything to do
with the `scifi_news_pro` failure. Two lanes, two mechanisms, one shared
symptom.

## PBUG-20260805-01 -- every adaptation cast rolled its gender, so 44 published rows contradict the source
- surfaced: published episodes, measured 2026-08-05 across every adaptation
  ledger under `output/otr` (88 ledgers, 176 non-announcer rows). Visible in
  shipped episodes -- e.g. `signal_lost_malvolios_yellow_stockings_20260804_192850`.
- symptom: MALVOLIO and LEAR cast female; MIRANDA, CORDELIA and ROSALIND cast
  male. 44 of 176 rows (25%) carry a gender that contradicts the shipped
  provenance sidecar. Also confirmed on MARIA, ROMEO, JULIET, CELIA, MACBETH,
  BENEDICK, VIOLA, MARCELLUS, FERDINAND, TITANIA, BANQUO, HERO, PROSPERO.
- root cause: `precompute_ensemble_slots` assigned every open slot a gender from
  a 40/40/20 largest-remainder roll (`_plan_gender_distribution`,
  `nodes/_otr_casting.py`), including slots whose NAME had just been popped off
  the source's own cast list. The roster truth was already on disk -- 14
  provenance sidecars carry a `characters` list -- and `source_meta_from_scene`
  never loaded it, so nothing downstream could know MALVOLIO was a man.
  The row gender is not only a voice field: it feeds the description LLM
  (`_otr_casting.py:777`), the outline prompt (`OTR_LedgerScriptWriter.py:4144`),
  the dialogue cast block (`_otr_line_composer.py:446`) and the image prompt's
  gender anchor (`otr_meta_brief_image_prompt.py:78-90`), so the defect reached
  the script and the portrait as well as the voice.
- fix: new `nodes/_otr_roster_gender.py` joins each source-owned slot name to the
  sidecar roster through an abstaining tier ladder, backed by a committed
  10-entry curated supplement for the names no tier reaches; the resolved gender
  OVERRIDES the drawn value at pinned indices while
  `_plan_gender_distribution` is left completely untouched -- same count, same
  priors, same rng, same post-call stream. Source-owned slots are also exempted
  from the name-coherence rename. Stamped as `meta.cast_source_contract` with
  per-name evidence.
- verify idea: for any adaptation ledger, every non-announcer row whose name
  appears in the source's `characters` roster must carry the roster's gender.
  Machine-checkable against the shipped sidecars with no render.
- bible-worthy: yes -- "a generator that rolls a value the source already
  records" is a reusable defect class, and the fix shape (join, abstain
  honestly, override in place rather than re-allocating) is portable.
- confidence: HIGH -- measured across every published adaptation ledger.
- status: OPEN (fix landed 2026-08-05; live proof leg pending)

## PBUG-20260805-02 -- LATENT: the bark voice replay rebuilds a different ensemble than the writer cast
- surfaced: NOT a production failure. Reproduced by probe against the shipped
  modules on 2026-08-05 at cast_seed 424242, and recorded on operator direction
  rather than promoted to a fix. **This entry does not meet the log's usual
  live-artifact admission bar and must not be fan-out-promoted to the Bug Bible
  on its own evidence.**
- symptom: at cast_seed 424242 with source names
  `['Antipholus','Dromio','Adriana']` the writer's ensemble is
  (ANTIPHOLUS male, DROMIO other, ADRIANA female) while `replay_voice_assignment`
  reconstructs (ERIN MARTIN female, FABER SATO other, KANE SIRIKIT male). The
  gender SEQUENCE diverges on 149 of 200 seeds (74%).
- root cause: an asymmetry in what the replay is told. `assemble_pre_locked_rows`
  accepts `source_character_names`, and on an adaptation lane it POPS those names
  off a queue for zero rng draws; `replay_voice_assignment(*, cast_seed,
  num_characters, lemmy_hit)` cannot accept them, so the replay takes the pool
  path and burns `pick_first_last` draws the writer never spent. Every later draw
  is then off by that much.
- why it is latent: `CastLock._assign_bark_voices` writes only
  `row["voice_preset"]` (`nodes/cast_lock.py:355`) and never a gender, and the
  shipped workflow runs indextts2 (node 80), which takes its audible reference
  from the ROW gender at `cast_lock.py:563`. Nothing a listener hears depends on
  the replay's reconstruction today. It becomes real the day the operator
  switches character voices back to bark.
- fix: none. The corrective step was CUT from the 2026-08-05 continuity build as
  not worth the surface it touches; forwarding `source_character_names` +
  `source_bank_id` into the replay restores the match exactly (probed), if it is
  ever wanted.
- verify idea: `lock_cast` then `_assign_bark_voices` at cast_seed 424242 with
  those three source names; the reconstructed per-slot gender must equal the row
  gender.
- bible-worthy: no -- latent, single-project, and no production artifact behind
  it. Recorded so the next reader does not re-derive it.
- confidence: HIGH on the mechanism, N/A on production impact (there is none today).
- status: OPEN (deliberately unfixed; reproducing seed 424242)

## PBUG-20260805-03 -- the announcer was planned but never scheduled, so scifi_news went 0-for-4
- surfaced: batch v2 headless run, 2026-08-05, 28 episodes over the canonical
  workflow. `scifi_news` failed 4 of 4 legs (008, 013, 015, 016) while every
  other lane was perfect: shakespeare 12/12, public_domain 9/9, original 2/2.
  Leg 013 burned **45.1 minutes** before dying; the others died at 4-5 minutes.
- symptom: `RESULT FAIL ... exception_message: "cast voice coverage failed for
  bank 'scifi_news': 1 of 4 cast member(s) have no SAYABLE line"`, raised from
  `_otr_cast_voice_coverage` at `stamp_receipt` -- AFTER the whole script had
  been written. The live server logs name the uncovered member as the ANNOUNCER.
- root cause: `compile_radio_score_draft` treated an uncovered cast member as
  ADVISORY -- it logged `cast_coverage advisory: N/M planned cast own a beat`
  and returned the score, on the reasoning that "an uncovered cast member simply
  carries no lines". The hole is therefore CREATED at P3 and only DISCOVERED
  after P5. It cannot be repaired downstream: P5 authors only `line_id` and
  `text` INTO a graph whose beat/shot/scene ownership is already compiled from
  the accepted score, so a member with no beat can never be given a line. A
  P5-level check could only burn the retry ladder and fail anyway -- which is
  what turned a 5-minute failure into a 45-minute one.
- fix: the advisory became a RECOVERABLE `RadioScoreDraftCompileError` with the
  new `cast_coverage` code, naming the missing ids. `validate_draft` converts it
  to a `PostValidationError`, which `_candidate_error_is_recoverable` already
  accepts, so the retry ladder and then the fresh-candidate loop redraft the
  score rather than failing the episode -- preserving the earlier ruling that
  cast_coverage must not become a fatal successor to the removed beat-count
  gate. Two prompt surfaces stated the OPPOSITE and were corrected: the pack's
  `codex_radio_score_system`, and `_RADIO_SCORE_DRAFT_SURFACE_INSTRUCTION`,
  which is appended LAST and still said "an unused planned cast member is not a
  story failure" (kibitz r3, Codex).
- why the loop cannot spin: the invariant is provably satisfiable.
  `num_characters` is clamped to 1..6 and the announcer makes at most 7;
  `_RADIO_SCORE_MAX_BEATS` is 3 scenes x 4 beats = 12; `_codex_target_beat_count`
  is called with `len(p2.cast)` and returns `max(cast_count, 3, min(12, ...))`.
  7 <= 12, so every planned member can always own a beat.
- coverage: `tests/test_p3_cast_coverage_invariant.py` (6 tests, mutation-
  verified: reverting the raise fails exactly the two coverage assertions).
- generalizable rule for the Bible fan-out: a producer stage that PLANS a
  resource must not defer the check that the resource was SCHEDULED to a
  consumer stage that cannot create it. Where the plan and the schedule are
  written by different passes, the schedule-time check is the only one that can
  still be repaired.
- live receipt: **PAID 2026-08-05 16:24.** One `scifi_news` leg at the exact
  failing coordinates (180 words, 2 characters) through
  `workflows/otr_canonical.json`, on a server booted FRESH after the fix
  (the batch v2 server had booted at 08:30, before the 14:30 commit, and held
  the old module in memory -- which is why the batch could not have proven it).
  All three gates: `RESULT SUCCESS`, `obs_publish OK`, and the asset on disk --
  `output/otr/obs/signal_lost_echoes_of_bias_20260805_161414_silent_procgen_blended_captioned_with_credits_final.mp4`,
  16.5 MB. `Prompt executed in 00:22:31`.
- what the log proves, and it is the whole point: the defect FIRED and was
  RECOVERED. P3 attempt 1 failed `draft.cast_coverage` ("3/4 covered, missing:
  announcer"), attempt 2 (typed repair) failed identically, the retry ladder
  exhausted -- and instead of killing the episode it logged `P3 candidate cycle
  1 exhausted (PostValidationError); abandoning it and starting cycle 2`, whose
  first attempt passed. Two coverage failures, one cycle abandonment, one
  published episode. Pre-fix that same sequence was terminal, four times.
- the recovery is bounded in practice: cycle 2 succeeded on attempt 1, so the
  unbounded fresh-candidate loop did not need a cap on this leg. The open
  cycle-cap question (agy for, Codex against) stays open on the same reasoning
  -- the invariant is provably satisfiable at cast <= 7, beats <= 12 -- but
  there is now one live data point rather than none.

## PBUG-20260805-04 -- the announcer read the source URL and licence aloud, and the captions burned it into the video

- surfaced: the PUBLISHED corpus, not a review. A scan of all 1,587 ledgers under
  `output/otr/episodes` finds spoken lines carrying a URL, a bare domain, a
  licence identifier or our own prompt labels. By `speaker_role` and line
  position: **84 leaked lines, 100% announcer, 100% at the LAST announcer
  position (the coda row), 0 non-announcer.** 30 distinct episodes leak on or
  after 2026-08-04, the most recent at 2026-08-05 14:22. The reusable predicate
  (`scripts/audit_spoken_citations.py`, which also matches shortened licence
  forms the sidecar string cannot) reports **69 episodes** with findings.
- worst shipped example, `2026-08-05 08:42` -- the announcer reads our own
  interpreter scaffold on air:
  `From tonight's echoing "Nothing," let us turn our ears to the silent archives:`
  `Source: Folger Shakespeare. Date/Rights: c. 1606 | CC BY-NC 3.0. URL:`
  `https://www.folger.ed...` -- `Source:`, `Date/Rights:` and `URL:` are verbatim
  the field labels built at `_otr_shakespeare_sources.py:586-589`.
- second surface, and the reason this is not merely an audio defect:
  `_otr_captions.py:283-286` copies RAW `lines[].text` into the ASS cue
  ("RAW line text, deliberately") and CaptionBurn is enabled in
  `workflows/otr_canonical.json`, so the URL is **burned into the delivered
  video**. NOT the still prompt (announcer rows take `scene_beat` at
  `otr_meta_brief_image_prompt.py:1117`, whose target carries no line text) and
  NOT the i2v motion clause (default OFF, `_otr_motion_clause.py:13-14`) -- both
  were claimed as surfaces in the inherited spec and both were disproved.
- mechanism: the interpreter is handed the source URL
  (`_otr_public_domain_sources.py:635`, `_otr_shakespeare_sources.py:589`) and
  asked for an attribution note in the SAME payload (`:665`, `:624`). The writer
  hoists that reply (`OTR_LedgerScriptWriter.py:4895-4897`) and
  `compose_news_coda` appends it VERBATIM (`_otr_line_composer.py:1285`,
  contract at `:1255` "never score, shorten, or replace it"). The append is
  deliberate -- it exists so a weak model cannot blend the fact away -- so the
  one thing engineered to survive unedited is the one carrying the URL.
- root cause of the RECURRENCE, which is the important part: the deterministic
  replacement already existed. `meta["provenance_coda_line"]` is composed by
  `_otr_provenance.spoken_coda_line` and stamped at
  `OTR_LedgerScriptWriter.py:3595`, and `_otr_provenance.py:112-118` records that
  the licence was removed from the spoken line on 2026-08-04 for exactly this
  reason. **That fix was applied inside `spoken_coda_line()`, a function with
  ZERO readers** -- grep returns the write and one docstring. The live path was
  never touched, so 30 more episodes leaked after the fix "landed".
- fix: select the effective spoken fact at the writer call site, keyed on
  `"provenance" in meta` (stamped unconditionally at `:3592`; the coda key was
  NOT, so presence of the coda is an invalid ownership test). A provenance-owned
  lane always takes the deterministic append regardless of `_style_grammar_on`;
  owned-but-empty goes straight to `fallback_announcer_outro("")` with neither
  composer entered. `news_close_brief` keeps its value and its owner -- it is
  also the treatment "Sign-off" line (`video_engine.py:1866`). The URL is also
  removed from both interpreter prompts (it never grounded anything; grounding is
  the source text) with `PROMPT_VERSION` bumped to
  `public_domain_interpreter_v3` / `shakespeare_interpreter_v2`.
- found in passing, same call, fixed with it: `compose_news_coda` was never
  passed `source_bank_id` (`OTR_LedgerScriptWriter.py:5491-5497`), so EVERY lane
  resolved media_archive's `coda_system` prompt -- while the sibling
  `compose_announcer_outro` call has passed it since Stage 4, and
  `tests/test_closing_seams_bank_routing.py:123-137` already proved the composer
  routes correctly when given it.
- receipt: `meta["spoken_coda_source"]`, closed vocabulary
  (`provenance` | `news_close_brief` | `none`) validated at write time, so a
  corpus audit can JOIN on what was spoken instead of inferring it from prose.
  Inferring it from prose is how this survived.
- coverage: `tests/test_spoken_citation_audit.py` (22 tests) pins the predicate
  itself, including that the deterministic coda PASSES its own audit and that the
  empty `license_label` on the public-domain sidecar is dropped as a needle -- an
  empty needle is a substring of every string and would report the whole corpus.
- generalizable rule for the Bible fan-out: **a fix applied to a function with no
  callers is not a fix.** When correcting a defect on a live surface, prove the
  edited symbol is REACHED from that surface before claiming the defect closed --
  grep for callers, not just for the symbol. This is the fourth armed-consumer-
  without-producer defect found on 2026-08-05.
- live receipt: **PAID 2026-08-05 evening.** Seven canonical legs on a server
  booted after `3943dd38`, across every lane the fix touches:

  | leg | bank | `spoken_coda_source` | the announcer's closing line |
  |---|---|---|---|
  | 01 | public_domain 320w | `provenance` | "Tonight's tale was adapted from a work in the public domain." |
  | 02 | shakespeare 320w | `provenance` | "Tonight's tale was adapted from Folger Shakespeare." |
  | 03 | media_archive 320w | **`news_close_brief`** | its own factual note, verbatim -- the CONTROL held |
  | 05 | public_domain 520w x3 | `provenance` | deterministic coda |
  | 06 | shakespeare 520w x3 | `provenance` | deterministic coda |
  | 07 | original 320w | `none` | fictional close, no attribution -- correct for an unowned lane |

  **Zero leaked lines across all of them.** Leg 02 is the one that matters most:
  that lane used to read "CC BY-NC 3.0" aloud on essentially every episode, and
  now says only the edition name -- exactly what `_otr_provenance.py:25-27`
  specifies ("names the SOURCE, never the licence identifier").

  The control is the other half of the proof: `media_archive` still speaks its
  news note verbatim, so the fix did not silence the lanes that are supposed to
  carry one.

- corpus verdict: `scripts/audit_spoken_citations.py --root <output>/otr/episodes`
  scanned **1,595** ledgers (8 more than the pre-fix baseline) and reports
  **69 findings -- unchanged**. Every new episode is clean; the number did not
  move because nothing new leaked. Pre-fix, a shakespeare leg leaked essentially
  every time.
- not covered by this receipt, and correctly so: `scifi_news` never traverses this
  code. It dispatches to `scifi_news_circuit` and returns before the coda block
  (`OTR_LedgerScriptWriter.py:3663-3717`), so its ledger carries no
  `spoken_coda_source` key at all -- confirmed live on
  `shadows_of_phobos_20260805_193430`. That is why the acceptance control is
  `media_archive` and not `scifi_news`.

## PBUG-20260807-01 -- the announcer asked the operator to write the opening, and 23 episodes shipped with it as their first line

- status: **FIXED AND LIVE-PROVEN 2026-08-07** (5/5 qualification legs; receipts below)
- promotion: BUG-12.86 (survival-guide `7a5fb88`, entry count 261 -> 262, `otr_coverage_index.yaml` row appended in the same commit). Promoted by the window under the 2026-08-07 amendment above, after checking the class against the index and the 261-entry Bible and finding it uncovered.
- found: corpus scan of shipped ledgers under `output/otr/episodes`, 2026-08-07,
  while investigating a DIFFERENT reported defect (`--premise` allegedly not
  reaching the writer). The premise wiring turned out to be sound; this was next
  door and worse.
- symptom, verbatim from `lines[].text` on shipped episodes:

  > "Please provide the SETTING, TIME, HOOK, and the cast list so that I may
  > write the opening for you."
  > "Please provide the cast list and setting details so I may begin the
  > broadcast."

- blast radius: **23 ledgers**, all `line_id b001`, `speaker_role announcer`,
  compose_flags `['announcer_intro', 'announcer_intro_rewritten']`. Range
  2026-07-22 .. 2026-08-07 across `original` (6), `shakespeare` (9),
  `public_domain` (6), `media_archive` (2). It is the FIRST line the listener
  hears, it is spoken by TTS, `_otr_captions.py` burns raw `lines[].text` into
  the ASS cue, and because the rewrite runs BEFORE the outro pass the poisoned
  text was also fed forward as `intro_text` / `OPENING TONE` into the close.
  A 24th corpus hit (`the_caretakers_clause`, `shot_001_b2`, scifi lane) is
  in-story machine dialogue and is NOT this defect.
- **four independent faults, none of which failed loudly:**
  1. `_otr_line_composer.compose_announcer_intro` read
     `getattr(safe_open_brief, 'hook', '')`. `SafeOpenBrief` has never defined
     `hook` -- its fields are `setting`, `time_of_day`, `opening_status_quo`,
     `cast`, `era`. The getattr default made it silent, and
     `opening_status_quo`, `cast` and `era` were constructed at two call sites
     and read by no prompt builder.
  2. its `"\n".join(filter(None, (...)))` could never drop anything: every
     element was an f-string with a literal label prefix, so always truthy. A
     starved brief therefore shipped as bare labels -- `"SETTING: \nTIME: \n
     HOOK: \nWrite the opening now."` -- which reads to a model as a form.
  3. `_otr_story_brief._validate_produced_open` accepted a brief with an EMPTY
     CAST (it iterated `model.cast` only to reject off-roster names), while all
     four banks' `announcer_intro_safe_system` seams end "Use ONLY the proper
     names in the cast list below; invent none". The prompt promised a roster it
     never sent.
  4. the rewrite could not have recovered: a failed compose does NOT raise --
     `_announcer_generate` converts the exception to `None` and the composer
     returns `fallback_safe_open()`, non-empty canned text -- so the writer
     stamped `announcer_intro_rewritten` and overwrote a real composed opening.
     The documented keep-the-in-loop-intro posture only ever fired on a raise.
- **the origin is NOT the obvious commit.** `314dd481` (2026-07-24) rewrote the
  safe-open path and severed faults 1 and 2, but **10 of the 23 legs predate
  it** -- proven from the git HEAD each ledger stamps at render time
  (`341545ec` x6, `f150213f`, `2129ce84`), not from dates. At `341545ec` the
  composer already read all five fields, already emitted each only when
  non-empty, and already sent a cast line -- and `_validate_produced_open` is
  BYTE-IDENTICAL there to HEAD. So fault 3 is the older cause and the one that
  explains the pre-314 legs, whose replies lead with the cast list.
- fix: one shared viability predicate --
  `(setting OR opening_status_quo) AND at least one CLEANED cast name` --
  defined once in `_otr_line_composer` and imported by `_otr_story_brief`, so
  the validator and the composer cannot disagree about what a usable brief is.
  Direct attribute access replaces every `getattr` default. Labels emit only
  with a value behind them. A starved brief raises a typed
  `AnnouncerBriefStarvedError` BEFORE the model call; the rewrite caller
  declines and keeps the existing line, the in-loop caller ships the
  deterministic open and records the fallback. A returned structural fallback is
  no longer stamped as a rewrite. Shipped `a200b6f1` + `615de993`.
- **two dead receipts found in passing, same defect class, fixed with it:**
  `meta["open_safe_fallback"]` and `meta["news_coda_fallback"]` each tested for
  a flag string no producer has ever emitted, so both read False on every
  episode including the ones that fell back. These are STATIC findings -- no
  live artifact demonstrates their impact -- and are recorded here only because
  they rode this fix, NOT as production incidents in their own right.
- **the class, which is the reusable part:** a receipt or prompt-context field
  keyed on a producer string or attribute the producer never emits, hidden by
  `getattr(x, "name", default)` or an `in flags` test that silently reads False.
  Four instances now: `hook`, `open_safe_fallback`, `news_coda_fallback`, and
  BUG-LOCAL-255's `_speaker_role`. It fails in the SAFE direction, so nothing
  ever complains.
- why no test caught it: `tests/test_closing_seams_bank_routing.py` asserted the
  SYSTEM message only, `tests/test_announcer_intro_rewrite.py` stubbed the
  compose entirely, and `test_intro_requires_nonempty_structural_context` had a
  parametrize list with exactly ONE case -- the sibling `script_brief` path --
  so the safe-open branch carried the same invariant and none of its cases.
  Nothing had ever asserted the brief's content REACHES the prompt.
- receipts: suite 9177 passed / 111 skipped / 1 xfailed; Bug Bible 17 at
  survival-guide `3759ae5`; `workflows/otr_canonical.json` byte-identical.
  Ten mutations of the shipped code each confirmed to turn the new tests red.
- **LIVE PROOF OWED, and one trap to avoid when running it:**
  `workflows/otr_canonical.json` node 1 has `widgets_values[23] == 'scifi_news'`,
  a lane that dispatches to `scifi_news_circuit` and RETURNS BEFORE this code.
  A leg from the unchanged canonical JSON proves nothing here. Every leg must
  load that exact file with a per-leg RUNTIME bank override and assert the
  resolved bank is one of `original`/`shakespeare`/`public_domain`/
  `media_archive`. Per `PRODUCTION_SPRINT_LESSONS.md:106-113` this is
  model-sensitive work: 30-word smokes on two local model families plus one
  cloud/frontier lane, the same at 120, only then 720.
- OPERATOR DECISION OWED: the 23 shipped episodes are in canonical ledgers and
  delivered audio/captions. Rerender/republish, or tombstone as known-bad and
  exclude from publication. Not a build gate; recorded here so it is not lost.

### PBUG-20260807-01 -- LIVE QUALIFICATION, 5/5 PASS (2026-08-07)

Ladder per `docs/PRODUCTION_SPRINT_LESSONS.md:106-113`. Every leg loaded the
UNCHANGED `workflows/otr_canonical.json` with a per-leg RUNTIME bank override,
and the resolved bank was asserted from `meta.source_bank` before the leg
counted -- the canonical graph is pinned to `scifi_news`, which returns before
this code, so an un-overridden leg would have been green and meaningless.

| Leg | Bank | Words | Writer | b001 (opening line, verbatim) |
|---:|---|---:|---|---|
| 1 | shakespeare | 30 | `mistralai/Mistral-Nemo-Instruct-2407` | "In the royal court of Britain, King Lear demands an accounting from his daughter, Cordelia." |
| 2 | public_domain | 30 | `google/gemma-4-12b-it` | "The sun hangs heavy over the garden as Rikki-tikki-tavi keeps a watchful eye on the grass..." |
| 3 | original | 120 | `mistralai/Mistral-Nemo-Instruct-2407` | "In the hushed confines of Spender Manor, as the grandfather clock strikes midnight, Malcolm Sirikit and Clarisse Spender..." |
| 4 | media_archive | 120 | `google/gemma-4-12b-it` | "From the dust of a forgotten archive, we find Sailor Burns and Rod Howard standing in a silent hallway..." |
| 5 | shakespeare | 30 | CLOUD `~anthropic/claude-haiku-latest` | "Good evening, friends, and welcome back to Signal Lost -- tonight we find ourselves on the battlements of Elsinore Castle..." |

**Every leg:** `meta.announcer_intro_rewrite == {"status":
"announcer_intro_rewritten", "reason": null}`; schema `l4-2026-08-07`;
`obs_publish OK` with the asset on disk; and **no leg asked the operator for
input**. Four affected banks, three model families (Mistral / Gemma /
Anthropic-remote), both 30 and 120 words.

Leg 5's cloud arm is proven from the server log, not inferred:
`[OpenRouter] load slot=A handle=openrouter:slot-a
slug=~anthropic/claude-haiku-latest route=default ctx=200000 (remote, 0 VRAM)`
followed by `[OpenRouter] call accounted ~1239 tokens`.

**Two things this qualification did NOT establish, stated so nobody reads more
into it than it earned:**
1. **No exhaustion rate.** Five legs is not a rate. No leg hit
   `reason: derive_failed`, so the starvation path itself was never exercised
   live -- the guard is proven present and non-interfering, not proven to fire
   correctly in production. Its unit coverage is the evidence for that.
2. **A meta-recording gap found in passing:** `meta.openrouter_slot_a_model` is
   `null` on leg 5 even though the slot demonstrably resolved and served the
   run. Routing worked; the RECEIPT is incomplete. Not this defect, not fixed
   here, and static -- so it does not get its own PBUG.

---

## PBUG-20260811-01 -- forcing the LEMMY cameo kills the scifi_news_pro writer

- surfaced: two live canonical headless legs, 2026-08-11 (`PROBE B_90w_forced`,
  and the `BANKSWEEP scifi_news_pro` leg of the six-bank sweep the night before)
- symptom: node 1 `OTR_LedgerScriptWriter` raises
  `[scifi_fable2] pass 'script' failed after 4 attempt(s): markup ladder
  exhausted; last defects: - BAD_LINE`. The run dies before any casting; no
  episode, no assets.
- root cause: NOT ESTABLISHED. What IS established is the trigger and that one
  plausible explanation is ruled out. Reproduced at BOTH 30 and 90 target words
  with `lemmy_cameo="always include"`, so it is not a word-budget squeeze; the
  same lane at 30 words with the cameo on its natural roll gets the writer
  through cleanly (it then fails elsewhere -- see PBUG-20260811-02). The
  pre-locked LEMMY row is what the `scifi_fable2` script pass cannot satisfy.
- fix: NONE YET. Recorded, not repaired.
- verify idea: run the `scifi_news_pro` lane with `force_lemmy=True` through the
  writer's script pass and assert it does not exhaust the markup ladder. A
  cheaper unit-level version: assert the lane's script prompt/validator can
  accept a pre-locked cameo row at all.
- bible-worthy: probably not on its own -- it reads as one lane's prompt/validator
  not tolerating a pre-locked row, rather than a portable contract. If a SECOND
  lane shows the same shape, the class ("a pre-locked cast row the writer pass
  cannot honour fails the whole render") would be.
- status: OPEN

**Reachability note, stated because it is my own change.** `lemmy_cameo` was
whitelisted for headless drivers in `baf338ee` (Chunk D) so a qualification run
could force the cameo deterministically. That commit did not CREATE this defect
-- the widget has always existed and the GUI could always set it -- but it made
the failure reachable from the sanctioned headless runner, which is how it was
found. Four other banks force the cameo fine, so the whitelist is not the thing
to revert.

---

## PBUG-20260811-02 -- scifi_news_pro dies at video render with no still for the closing-music beat

- surfaced: live canonical headless leg, 2026-08-11 (`PROBE A_30w_noforce`,
  profile `otr_w45_still_flat`, 30 words)
- symptom: node 92 `OTR_VideoRenderBatch` raises `still-spine handoff missing
  materialized scene still for shot shot_music_closing_001 beat
  music_closing_001 engine still_flat`. The writer, casting and the whole audio
  chain succeeded first (executed list includes nodes 1, 62, 63, 80-83).
- root cause: NOT ESTABLISHED. The closing-music beat reached the still-spine
  handoff without a materialized still. Five other banks on the SAME profile
  produced one and published normally, so it is lane- or beat-topology-specific
  rather than a profile defect.
- fix: NONE YET. Recorded, not repaired.
- verify idea: assert every beat the still-spine hands off has a materialized
  still, naming the beat when one is missing -- the current message already
  names it well, so the gap is a pre-handoff completeness check, not better
  reporting.
- bible-worthy: possibly. "A handoff consumed a per-beat artifact that was never
  produced" is a portable shape. Hold until the root cause is known.
- status: OPEN

**Seen once.** Recorded because it is a live production failure with a named
node and a named beat, which the admission rule admits; but it has not been
reproduced, and the ONE observation came from a diagnostic leg rather than a
normal render. Re-run before treating the cause as understood.

---

## PBUG-20260811-03 -- scifi_news LOST the Lemmy cameo it was built for

- surfaced: live canonical headless leg, 2026-08-11 (`BANKSWEEP scifi_news`,
  profile `otr_w45_still_flat`, 30 words, `lemmy_cameo="always include"`)
- symptom: the forced cameo produced NO Lemmy row and recorded NO reason. The
  episode's `cast_contract` is **empty** -- no `cast_seed`, no `cast_seed_source`,
  no `casting_attempts`, no `lemmy_hit`, no `lemmy_policy`,
  no `num_characters_locked`. `num_characters` was also ignored (asked 2, got 3:
  Ada, Kai, Dr. Elara). Compare `original` on the same sweep, which stamped all
  seven keys.
- root cause: **ESTABLISHED 2026-08-11.** `scifi_news` is a CONTENT-OWNED lane
  (`delivery_mode_for_meta(meta) == CONTENT_OWNED`, verified against the sweep's
  own ledger; `original` on the same sweep is `legacy`). Content-owned lane
  runners build their own cast rows and stamp their own voice presets, so they
  never run the writer's seeded cast picker -- `OTR_LedgerScriptWriter` says so
  in as many words at the content-owned tail. `lock_cast()` is what applies the
  cameo, so the cameo cannot happen there. The empty `cast_contract` is the same
  decision: that block deliberately stamps `meta.episode_seed` and NOT
  `cast_contract.cast_seed`, because cast_seed is a claim the writer's picker
  produced this cast and can replay it -- and a lane-owned cast has no
  `num_characters_request` to replay with. Claiming it detonated CastLock's
  replay in a prior bug (`num_characters must be 1-6, got 0`).
- **so this is a capability lost to an ARCHITECTURAL change, not a careless
  break.** scifi_news predates the content-owned redesign, worked under the
  legacy picker (which is why the operator remembers it working), and lost the
  cameo when it became content-owned. Nobody removed Lemmy from it.
- **THE OBVIOUS FIX IS THE WRONG ONE.** Routing content-owned lanes back through
  `lock_cast()` is exactly what the writer's comment warns detonates the replay.
  The repair belongs in the LANE RUNNER: either it offers the cameo itself when
  building its cast, or it stamps an explicit declined-policy so the ledger
  records a decision instead of a silence. Which of those is an operator call --
  it decides whether Lemmy can appear in scifi_news again at all.
- **why this is a REGRESSION and not a design choice:** the operator confirmed
  2026-08-11 that scifi_news "was built with Lemmy in mind and always used to
  work -- it was the first Lemmy plan". This lane is the cameo's ORIGINAL home.
  An earlier draft of the finding doc recorded it as a possible
  "lane owns its cast" design decision; that reading is WITHDRAWN.
- fix: NONE YET.
- verify idea: assert every runnable bank's episode records a cameo DECISION --
  `lemmy_policy` present with some value -- even on content-owned lanes. Absence
  of the key, not absence of Lemmy, is the detectable defect. Do NOT assert a
  non-empty `cast_contract.cast_seed` on content-owned lanes; that is the field
  whose false claim caused the earlier replay detonation.
- bible-worthy: likely yes as a class -- "a pipeline silently bypassed the one
  function that records a decision, so the ledger cannot distinguish 'declined'
  from 'never asked'". That shape is portable well beyond this cameo.
- status: OPEN

**Ranking note.** Of the three defects the sweep found this is the one that has
been shipping longest and most invisibly: nothing failed, nothing logged, and
every scifi_news episode since the regression simply has no cast contract. It was
only visible because the cameo was FORCED and then did not appear.

## PBUG-20260812-01 -- a shared module reached its sibling by an absolute `nodes.` import, so the Sage probe could not read Sage on ANY server

- surfaced: LIVE headless leg, 2026-08-12 (`_otr_single_engine_smoke.py --engine
  minimax_h3_video --frames 129` against the sage-free `h3` boot on :8000, lane
  19's solo smoke). The render refused before any weight loaded with:
  `BootContractError: the running server does not satisfy boot contract 'h3' ...
  needs SageAttention ABSENT, but the probe could not determine it
  (ModuleNotFoundError) -- an unverifiable Sage state is not a pass on a lane
  Sage silently corrupts`. The server WAS sage-free; nothing about Sage was
  wrong.
- symptom: a lane that requires a Sage-constrained boot contract is
  UNRENDERABLE on every server, and the refusal names Sage rather than the
  import that actually failed.
- root cause: `nodes/_otr_shared/boot_contracts.py:running_server_boot_state`
  reached its sibling package with
  `from nodes._otr_video_engines.motion_common import sageattention_patched`.
  **`nodes` resolves against `sys.path`.** Under pytest the repo root is on the
  path, so `nodes` IS the OTR package and the probe worked in every test. Inside
  a running ComfyUI server `nodes` is ComfyUI's OWN top-level node-registry
  module and OTR lives under `custom_nodes/ComfyUI-OldTimeRadio`, so the import
  raised `ModuleNotFoundError`, was caught, and left `sage_attention = None`.
  `check_running_server` correctly treats UNKNOWN as not-satisfied, so the
  contract could never be met.
- **why it shipped dormant:** the probe's error is only CONSULTED when a
  contract constrains Sage, and `h3` (2026-08-12) is the first one that does.
  `default` and `humo_diet` both say "don't care" about Sage, so the broken
  import sat behind them harmlessly since the S8 boot-contract mechanism landed.
- **TWO MORE INSTANCES of the same class, swept in the same commit** -- neither
  independently live-verified, both the identical import defect:
  `_otr_shared/content_oracle.py:family_for_engine` failed SOFTLY into a bare
  `except: pass` and answered from the `_FAMILY_FALLBACK` table on every call,
  so the live registry was "the source of truth when present" only OFF the
  runtime. That table stops at 2026-07-05, so `ltx_8gb`, `fastwan_8gb`,
  `still_word`, every cloud lane and `minimax_h3_video` resolved to family `""`
  in production -- which is not in `MOTION_FAMILIES` -- making
  `motion_required_for_engine` answer False and those lanes silently
  MOTION-EXEMPT. `_otr_shared/slot_matrix.py:eligible_engines_for_role` raised
  outright.
- fix: all three are relative imports (`from .._otr_video_engines import ...`),
  which resolve through the package's own `__name__` and are correct under both
  names. Committed `be4aadff` with lane 19.
- verify: `tests/test_minimax_h3_video.py::
  test_no_shared_module_reaches_a_sibling_by_an_absolute_nodes_import`
  AST-walks both shared packages and fails on any `nodes.`-prefixed
  `Import`/`ImportFrom`. **AST, not a text grep** -- the first draft grepped the
  source and failed on the comment that explains the fix, which necessarily
  quotes the broken line. Plus `::test_the_family_oracle_answers_from_the_
  REGISTRY_not_the_stale_table` for the soft-failure half, which asserts the
  CONSEQUENCE (every registered engine's family matches the registry) rather
  than the import.
- bible-worthy: **yes, and strongly** -- "a module works in the test environment
  and raises in production because an absolute import resolves against a
  `sys.path` that differs between them" is portable to every plugin/extension
  architecture where the host owns a top-level module name. The three-way split
  in how it failed (caught-and-unknown, swallowed-into-stale-fallback, raised)
  is the instructive part: only one of the three was visible at all.
- status: FIXED, live-proved (the same smoke rendered PASS after the fix:
  129 frames at 864x480, exactly 5.160 s, zero audio streams)

## PBUG-20260812-02 -- a Pydantic field named `register` silently becomes a BOUND METHOD, and the writer dies serializing it

- surfaced: LIVE headless leg, 2026-08-12 06:08, the first leg of the 45-word
  every-visual-path campaign (`otr_w45_campaign.py --only still_flat,...`,
  profile `otr_w45_still_flat`, source bank rolled to `scifi_fable2`). The node
  `OTR_LedgerScriptWriter` failed with
  `TypeError: Object of type method is not JSON serializable`, 78 s in, before
  any video work. Leg verdict: `FAIL (exit=1, 1.3 min) no new file in otr/obs`.
- symptom: an episode dies in the WRITER, at
  `_otr_scifi_fable2.py:1532` in `_script_user_prompt`, on
  `json.dumps(treatment.model_dump())`. Nothing about the message names the
  field or the model, so the failure reads as a generic serialization bug.
- **root cause, REPRODUCED exactly.** `CastShape.register`
  (`_otr_scifi_fable2.py:281`) shadows an attribute that exists on
  `BaseModel` -- Pydantic's `ModelMetaclass` inherits `ABCMeta`, so
  `BaseModel.register` is a bound metaclass method. Pydantic does NOT reject
  this field name (the clash is on the metaclass, not the class body), and a
  fully-validated instance is fine because the value lives in `__dict__`.
  **But any instance whose `register` is absent from `__dict__` falls through
  to the class attribute, and `model_dump()` then returns the bound method.**
  Minimal reproduction on this box:

      ok  = CastShape(name="Ada", role="lead", want="w", pressure="p",
                      register="dry")
      ok.model_dump()            # {'register': 'dry'}  -- fine
      bad = CastShape.model_construct(name="Ada", role="lead", want="w",
                                      pressure="p")
      bad.model_dump()           # {'register': <bound method ...>}
      json.dumps(bad.model_dump())
      # TypeError: Object of type method is not JSON serializable

  Pydantic even warns on the dump --
  `PydanticSerializationUnexpectedValue(Expected 'str' ... field_name='register',
  input_value=<bound method ModelMetacl...>, input_type=method)` -- and that
  warning goes to stderr where nothing reads it.
- **why it is INTERMITTENT, which is what makes it nasty:** the campaign rolls
  `--source-bank "roll (any eligible bank)"` per leg, so only legs that roll
  `scifi_fable2` can hit it, and only when a `CastShape` reaches the prompt
  builder without its `register` set. A re-run can pass and look like a flake.
- fix: NONE YET -- deliberately. Two candidate directions, and the choice
  belongs to the writer lane's owner because both touch the LLM contract:
  (a) RENAME the field (`register` -> e.g. `vocal_register`), which removes the
  shadow entirely but changes the structured-output schema the model is
  prompted against and the `register:` label in the cast block at
  `_otr_scifi_fable2.py:1527`; or (b) give it a DEFAULT (`register: str = ""`),
  which makes the construct path fill it instead of leaking the method, at the
  cost of weakening a currently-required field. **(a) is the root fix; (b) is a
  containment.** Not attempted unattended: a wrong move here breaks story
  generation on a shipping bank.
- **the production TRIGGER is still unidentified.** There is no
  `model_construct` or `CastShape(...)` call anywhere in `_otr_scifi_fable2.py`
  -- the treatment is built by `structured_call(schema=Treatment, ...)`, which
  validates -- so something in the structured-output/repair path is producing a
  partially-populated model. That search is the next step and it is where the
  fix has to be aimed.
- verify idea: a test that asserts NO Pydantic model in the writer path has a
  field name for which `hasattr(BaseModel, name)` is true. That is a one-line
  structural check over `model_fields`, it catches the whole CLASS of this bug
  rather than this one field, and it would have failed the day `register` was
  added. (`Treatment` itself is clean; `CastShape.register` is the only hit in
  the two models on this path.)
- bible-worthy: **yes, strongly, as a class.** "A Pydantic field whose name
  collides with an attribute of BaseModel/its metaclass serializes as a bound
  method whenever the instance is built without it" is portable to every project
  using Pydantic v2, the failure is silent until a `json.dumps`, and the error
  message names neither the model nor the field.
- status: OPEN -- root cause PROVEN and reproduced, production trigger not yet
  located, no fix attempted.
