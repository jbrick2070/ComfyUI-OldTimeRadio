# Production Bug Log (staging pre-Bible)

**Contract (operator, 2026-07-10):** Claude appends entries here AUTONOMOUSLY, but
ONLY for bugs that actually failed in a live/prod run (live render, headless lane,
soak, published episode). Dev/audit/review catches get fixed, never logged. NO entry
here touches the Bug Bible directly -- at ship time the operator triggers a BUG
FAN-OUT over this log, which promotes approved entries into the survival-guide
Bible in bulk under the Three-File Contract (YAML + README count + regression
test, one commit). Promoted entries get a `- promotion: BUG-...` mapping;
rejected ones get marked `REJECTED` and stay for the record. Append-only,
newest last.

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
- status: FIX IN TREE (unit-verified); LIVE end-to-end reverify PENDING -- the same-lane P3 premise cap (regression of PBUG-20260713-04, below) is downstream; sequence: P0 clears -> P3 premise leg.

## Regression watch (2026-07-17 -- NOT a new PBUG) -- codex P3 string_too_long on `premise` re-occurred on scifi_codex_v4
- RE-OCCURRENCE of PBUG-20260713-04 (BUG-11.42), not a new class (cross-check window ruling). Live 30w `scifi_codex_v4` leg (prompt `6883758f`) failed P3 `string_too_long` on premise >144 -- same field, same 144 cap, same lane, same mechanism (model writes over-cap; text-patch never clips prose). The -04 verified recipe (conservative ~75% model-facing `max_chars` with the true cap PRIVATE + `source_to_shorten`/forbid-unchanged-copy + never Python-clip) is present in tree (`_otr_scifi_codex.py:1752/:1754/:1800` + surface instruction premise<=108). A same-session kibitz had re-added the literal 144 cap to the base seam; per -04 that is the anti-pattern (exposing the rejection edge makes the model aim at it and cross it), so it was REVERTED (same commit as PBUG-20260717-01). Untestable end-to-end until PBUG-20260717-01 (P0) clears; sequence: P0 clears -> a live 120w leg exercises the P3 premise cap. If premise still overruns for the v4 proof-pressure density AFTER -04's recipe, the BUG-11.54 deterministic word-boundary shortener (already used for question/consequence) is the design precedent. No new PBUG until a live failure survives the -04 recipe.
