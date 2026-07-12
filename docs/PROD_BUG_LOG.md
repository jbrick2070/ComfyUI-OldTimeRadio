# Production Bug Log (staging pre-Bible)

**Contract (operator, 2026-07-10):** Claude appends entries here AUTONOMOUSLY, but
ONLY for bugs that actually failed in a live/prod run (live render, headless lane,
soak, published episode). Dev/audit/review catches get fixed, never logged. NO entry
here touches the Bug Bible directly -- at ship time the operator triggers a BUG
FAN-OUT over this log, which promotes approved entries into the survival-guide
Bible in bulk under the Three-File Contract (YAML + README count + regression
test, one commit). Promoted entries get marked `PROMOTED <bible-id>`; rejected
ones get marked `REJECTED` and stay for the record. Append-only, newest last.
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
- surfaced: scifi bake-off canonical 30w smoke, Codex P0, 2026-07-11
- symptom: a full headline quote was returned with `headline[0:55]`, so the validator saw only a truncated payload slice and halted the lane
- root cause: the model supplied a stale end offset and, in some artifacts, source-field labels did not identify the field containing the exact quote
- fix: `55f3cf17` rehomes an exact quote only when exactly one allowed payload field contains it, then recomputes start/end; absent or ambiguous evidence still fails closed
- verify idea: fixture with wrong field and offset rehomes to the unique literal field; fixture with absent or duplicate quote returns no repair
- bible-worthy: yes -- live source-evidence metadata failure, cross-lane helper
- confidence: HIGH
- status: OPEN

## PBUG-20260711-05 -- JSON parser salvaged a nested fact from a broken outer artifact
- surfaced: scifi bake-off canonical 30w smoke, Codex P0, 2026-07-11
- symptom: malformed outer fact JSON was scanned past its first brace; the parser returned the first nested fact object, producing misleading missing-top-level-key errors and preventing the intended repair path
- root cause: shared fallback scanning treated a nested child as a valid top-level object when the response began with an invalid outer object
- fix: `5489baa8` fails closed when a response begins with malformed outer JSON instead of salvaging nested children; all source packs use the shared parser
- verify idea: malformed outer-with-valid-child fixture raises a top-level parse error; valid leading prose plus a valid object still parses normally
- bible-worthy: yes -- shared structured-call integrity defect across source packs
- confidence: HIGH
- status: OPEN

## PBUG-20260711-06 -- Codex P3 omitted required nested scene graph fields
- surfaced: scifi bake-off canonical 30w smoke, Codex P3, 2026-07-11
- symptom: score JSON had the correct top-level artifact but omitted required nested `scene_id`, `shot_id`, and `visual_prompt` fields; strict validation halted before script/dialogue/media work
- root cause: the prompt named top-level keys but hand-described no complete nested required-field tree, so the local model repeated an incomplete graph
- fix: `b9cfc508` generates a compact required-path inventory from each Pydantic model's `model_json_schema()` and injects it into all three lane prompt builders
- verify idea: assert `scenes[*].shots[*].scene_id` and equivalent nested paths appear in generated prompts; live smoke must pass P3 graph validation
- bible-worthy: yes -- live nested-schema contract failure, same family as PBUG-20260711-03
- confidence: HIGH
- status: OPEN

## PBUG-20260711-07 -- Codex P0 overclaimed beyond the supplied RSS payload
- surfaced: scifi bake-off canonical 30w smoke roll 6, Codex P0, 2026-07-11
- symptom: the model returned a quote longer than the literal `full_text` payload; typed repair repeated it and the evidence validator halted before downstream work
- root cause: the model treated a claim-like sentence as source evidence even though the supplied payload did not contain that exact span
- fix: `6e6ff57b` drops unsupported facts/entities/numbers during metadata-only repair and retains only literal evidence; if no supported fact remains, the pass still fails closed
- verify idea: mixed fixture keeps literal facts and drops paraphrased facts; all-paraphrase fixture remains invalid
- bible-worthy: yes -- live grounding overclaim, same evidence-contract family as PBUG-20260711-01/02/04
- confidence: HIGH
- status: OPEN

## PBUG-20260711-08 -- Codex P3 generic repair repeated an incomplete graph
- surfaced: scifi bake-off canonical 30w smoke roll 7, Codex P3, 2026-07-11
- symptom: base and generic typed repair both omitted required nested scene graph fields despite a valid top-level score object
- root cause: non-P0 passes used the generic repair factory, which did not present the failed artifact and validation error with lane-specific graph-preservation instructions
- fix: `a27206df` routes typed repair for every Codex/Gemini/Sonnet pass through a schema-aware failed-artifact/error prompt while preserving premise, beats, cast, and authored content
- verify idea: force a nested graph validation failure and assert the repair prompt includes the failed artifact, exact validation error, schema paths, and context-preservation rule
- bible-worthy: yes -- live repair-contract failure, cross-lane by construction
- confidence: HIGH
- status: OPEN

## PBUG-20260711-09 -- Codex P3 repair omitted cast-locked speaker fields
- surfaced: scifi bake-off canonical 30w smoke roll 8, Codex P3, 2026-07-11
- symptom: schema-aware repair reduced the failure to two missing `speaker` fields on beats; the lane halted before script/media work
- root cause: nested graph repair did not explicitly bind each beat's speaker to its cast row by `char_id`
- fix: `fca99a5a` adds the cast-lock mapping rule to typed repair prompts for all three lanes
- verify idea: force missing beat speakers and assert the repair prompt requires cast-row lookup by `char_id`; live Codex P3 must clear
- bible-worthy: yes -- live cast/graph integrity contract failure, cross-lane prevention
- confidence: HIGH
- status: OPEN

## PBUG-20260711-10 -- Codex P5 repair omitted ScriptLine boundary metadata
- surfaced: scifi bake-off canonical 30w smoke roll 9, Codex P5, 2026-07-11
- symptom: full script artifact was otherwise shaped, but all eight lines omitted required `boundary` values; strict validation halted before audio/media work
- root cause: the repair contract named nested fields but did not define the boundary derivation from shot/beat order
- fix: `94331eb2` adds the structural rule: first line in shot = `shot_start`, first line in beat = `beat_start`, otherwise `continue`
- verify idea: force missing boundaries and assert the repair instruction contains the three-way derivation rule; live P5 must clear
- bible-worthy: yes -- live script graph metadata failure
- confidence: HIGH
- status: OPEN

## PBUG-20260711-11 -- Canonical RSS selector delivered a thin science payload
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
- surfaced: scifi bake-off canonical 30w smoke roll 11, Codex P5, 2026-07-11
- symptom: both P5 attempts returned prose or a score-shaped object instead of `ScriptArtifactV4`; the prompt guard reported `Truncated ... -> 1692 tokens` before each call
- root cause: P5 reserved a fixed 6500 output tokens inside an 8192-token context even for a 30-word script, leaving too little input budget for the failed artifact, graph, schema paths, and repair instructions
- fix: `fdc413ed` scales Codex whole-script P5/P7/P9 output reservation from the requested word steer (30w = 2200 instead of 6500), keeps every generated required path, removes the duplicate full schema from typed repair, and records token-budget/raw-size receipts; eight Kibitz reviews converged on the exact call-site wiring
- verify idea: 30w P5 prompt is not truncated, required ScriptArtifactV4 paths remain in the effective prompt, and canonical Codex reaches publish
- bible-worthy: yes -- live context-budget/structured-output contract failure
- confidence: HIGH
- status: OPEN

## PBUG-20260711-13 -- Codex P5 typed repair retained two forbidden legacy metadata values
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
- surfaced: scifi bake-off canonical 30w smoke roll 12, Codex CastLock, 2026-07-11 (first roll to survive P5 + the voice gate)
- symptom: the lane cleared every structured pass, stamped 13 delivery lines, rendered, and then died ~14 minutes in with `ValueError: num_characters must be 1-6, got 0` (cast_lock.py:189 -> _assign_bark_voices -> _otr_casting.replay_voice_assignment -> assemble_pre_locked_rows:1211)
- root cause: `meta.cast_contract.cast_seed` is not a generic episode seed -- it is a claim that the WRITER's seeded cast picker produced this cast and can be REPLAYED from it. Content-owned lanes build their own cast rows and stamp their own voice presets in the lane runner, so the picker never ran and the contract carries no `num_characters_request` -> `int(None or 0)` -> 0 -> ValueError. The PBUG-20260711-15 credits fix stamped `cast_seed` as a generic receipt and thereby CLOSED the `cast_seed is None` escape hatch these lanes had always relied on. A fix for one producer gap opened another.
- fix: the shared writer tail stamps `meta.episode_seed` ONLY (otr_credits_roll.py:279-284 already accepts it as the seed receipt, so credits provenance holds without asserting a replayable cast); and cast_lock._assign_bark_voices VERIFIES instead of REPLAYING for a content-owned lane -- it preserves the lane's `voice_preset` values and still runs the Gate 1 invariants, so such a lane can never ship duplicate or non-`v2/` bark voices. The legacy replay path is untouched (test_cast_voice_replay_parity pins it byte-for-byte).
- verify idea: a content-owned meta carrying a cast_seed must NOT enter the replay; a content-owned cast with two identical bark voices must still raise; the fable2 tail test asserts episode_seed is present AND cast_contract.cast_seed is absent
- bible-worthy: yes -- a receipt key that silently doubles as a behavior switch; the "my fix opened the next gap" class
- confidence: HIGH
- status: FIXED (awaiting fan-out)

## PBUG-20260711-17 -- P7 echoed the request envelope and truncated against its own output cap
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
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `0c1bb246-fae0-41c6-8f12-4cd8cccd27f3`, 2026-07-12
- symptom: P3 emitted `lost_object_2`, `lost_object_3`, and `resolution_links_2`; typed repair renamed them to `lost_object_secondary` / `lost_object_tertiary` instead of removing the schema violations, so the run failed closed after 459 seconds
- root cause: the P3 prompt named the collections but never stated that every selected lost object owns one separate `caller_threads` row with one singular `lost_object`, nor that every thread owns exactly one resolution row; Python also did not validate exact cross-artifact lost-object coverage
- fix: `5fd661ab` makes the base and repair contracts explicit, forbids numbered/suffixed pseudo-fields, validates the selected-object multiset, requires clue coverage per thread, and requires exactly one resolution per thread
- verify idea: validate a three-object selected possibility against a truth map with exactly three caller rows, at least one clue per thread, and exactly one resolution per thread; reject packed/suffixed fields, missing objects, duplicate resolutions, and repair-only renames; run the same canonical 30-word bank through Mistral and Gemma families
- bible-worthy: yes -- cross-model structured-output ownership ambiguity is reusable beyond OTR and survived a typed repair by changing only the illegal field names
- confidence: HIGH
- status: FIXED (the next E4B run used one row per object with no suffixed fields; it exposed the distinct nesting bug below; awaiting fan-out)

## PBUG-20260712-02 -- Gemma nested top-level truth collections inside caller rows
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `fc362a77-ec2f-4bf0-a4fc-ac9017eeec53`, 2026-07-12
- symptom: P3 returned a schema-complete top-level truth map but also put a `causal_steps` array inside each of three `caller_threads` rows; typed repair repeated the forbidden nesting unchanged, and the run failed closed after 461.82 seconds
- root cause: the P3 seam and typed-repair rules described collection contents but did not state the exact top-level collection placement or exact caller-row field set; the repair ladder had no safe deterministic relocation for declared collections placed at the wrong depth
- fix: `8f5b3d4d` -- the P3 seam and repair rules name exact nesting, and a P3-only deterministic repair treats an existing top-level collection as authoritative or lifts nested rows verbatim only when top-level is absent/empty; strict schema plus full truth-graph validation must pass or the normal typed LLM repair runs
- verify idea: test authoritative top-level plus nested extras, absent top-level plus verbatim nested rows, non-list nested values, unknown fields, duplicate graph IDs, and a full mocked ladder proving the deterministic repair spends no additional LLM call; repeat Gemma/Mistral canonical smoke
- bible-worthy: yes -- strict item schemas do not prevent a model from placing a valid declared collection at the wrong depth, and typed repair may reproduce the nesting unchanged
- confidence: HIGH
- status: FIXED (next E4B run cleared P3/P4 and exposed the distinct P5 nesting bug below; awaiting fan-out)

## PBUG-20260712-03 -- Gemma nested score shots inside scenes
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `649e1d99-c96d-485b-bce1-f68858f6d2d8`, 2026-07-12
- symptom: the run cleared P1-P4, then P5 returned `shots` arrays inside all four `scenes` rows; typed repair repeated the forbidden nesting after `PROMPT_GUARD` truncated its input from 4751 to 4592 tokens, and the run failed closed after 13:31
- root cause: the BroadcastScore seam and typed-repair rules specified scene and shot fields but did not explicitly require separate top-level scenes/shots/beats arrays; no deterministic structural repair handled declared score collections at the wrong depth
- fix: `54e159ec` -- exact top-level score ownership is stated in base and repair prompts; a P5-only deterministic repair retains authoritative top-level shots/beats or lifts nested rows verbatim when top-level is absent/empty, then requires strict schema and full score-graph validation
- verify idea: test top-level-authoritative cleanup, absent-top-level nested shots+beats lifting, non-list values, unknown fields, duplicate graph IDs, and a full runner with no extra LLM call; rerun the E4B/Mistral canonical smoke
- bible-worthy: yes -- a second typed artifact reproduced the wrong-depth collection class, showing the prevention law must cover every nested row boundary rather than one schema
- confidence: HIGH
- status: FIXED -- canonical E4B/Mistral prompt `fafd6306-cf0a-4c41-9bcb-11d2a8974864` cleared P5, froze the ledger, and published the episode; that run exposed the separate semantic false green below

## PBUG-20260712-04 -- Raiders of the Lost Prompts: opaque clue IDs let the spoken story abandon its source bank
- surfaced: published canonical 30-word `original_codex56sol` episode `signal_lost_the_muted_melody_20260712_020438`, E4B creative + Mistral technical, prompt `fafd6306-cf0a-4c41-9bcb-11d2a8974864`, 2026-07-12
- symptom: history, freeze, audio identity, mux, and OBS publish all succeeded, but the immutable c03 draw (`parcel tag`, `brass button`, `choir note`, `clockwork display`, repair-and-return ending) became an ancient-artifact laboratory procedural speaking `protocol alpha`, `isotopic decay`, `resonance signature`, and `micro-vibrations`; none of the three lost possessions, the device, or the promised return survived into dialogue
- root cause: routing was correct and visual style never entered P1-P9; semantic provenance stopped at opaque clue IDs. P5 proved clue-ID coverage but not clue meaning, P6 received score+manifest without the draw/truth map, script validation checked graph/safety only, P7/P9 could bless a self-consistent replacement cause, and only response hashes survived for intermediate artifacts. The independently selected `sci_fi_radio` visual pack then amplified the already accepted story drift downstream
- fix: add a strict draw-derived grounding contract with literal lost-possession/device/resolution anchors; require object anchors on clue-carrying intents and spoken lines, the device anchor on reveal, and the resolution anchor on closure; thread truth+grounding into P5/P6/all retakes/P9; rerun the blind listener after a blocking retake; make P9 rejection fail closed; add an ordinary-world bank boundary and narrow incident-derived detour phrases; persist accepted intermediate artifacts plus line-level grounding evidence; prove visual-style changes leave every story message byte-identical
- verify idea: the exact seven-line `The Muted Melody` script must fail before P7; independently remove each object/device/resolution anchor and get its exact coordinate; switch only `visual_style` between `sci_fi_radio` and `video_art` and prove captured P1-P9 messages are identical; rerun deterministic c03 at 120 words and require the grounding receipt, frozen ledger, episode final, and OBS final
- bible-worthy: yes -- structured IDs can stay referentially valid while their semantic payload disappears between artifacts; an end-to-end media success is not a content-contract success
- confidence: HIGH
- status: FIXED IN CODE / AWAITING LIVE 120-WORD C03 REQUALIFICATION; the published 30-word episode is retained as a false-green regression artifact and does not qualify the bank

## PBUG-20260712-05 -- Every custom runner title was stamped as a Fable2 title
- surfaced: forensic audit of the same Codex56 false-green ledger, 2026-07-12
- symptom: `meta.title_source` said `fable2_script_title` even though routing and authorship correctly identified `original_codex56sol`; the stale label could falsely implicate another story bank during incident diagnosis
- root cause: the shared writer tail hardcoded the Fable2 receipt whenever any custom runner supplied `final_title_override`
- fix: derive custom title provenance from `ctx.source_bank_row.source_bank_id`, preserve the established `fable2_script_title` value for the actual Fable2 lane, and stamp `<source_bank_id>_script_title` for every other custom runner without changing the pinned tail-context field contract
- verify idea: direct helper tests for Fable2 and Codex56 plus the existing title-override precedence suite
- bible-worthy: yes -- stale provenance labels turn correct routing evidence into a false root-cause lead
- confidence: HIGH
- status: FIXED IN CODE / AWAITING FAN-OUT

## PBUG-20260712-06 -- Gemma repeated invented music filenames through P5 repair
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification, prompt `7384fbe8-d1c9-4485-ba8e-b7f100329a12`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 reached the BroadcastScore on its first base call but added `opening_music.music_file=opening_music.mp3` and `closing_music.music_file=closing_music.mp3`; the typed repair repeated both forbidden fields, so strict validation failed closed after 12:32 and no ledger/media artifact was accepted
- root cause: the score seam closed the top-level, scene, shot, beat, and line-intent key sets but described music bookends only semantically; the shared schema instruction listed their required paths without explicitly forbidding nested extras, allowing a model to treat plausible production filenames as authored score fields
- fix: the bank prompt now states that each music bookend has exactly `description` and `generation_prompt` and explicitly forbids filenames/paths/cue metadata; the existing P5 structural normalizer now deletes only non-authoritative extra bookend keys, preserves every required LLM-authored value byte-for-byte, and still requires the complete strict score plus graph/content validators to pass before it can avoid another model call
- verify idea: inject the exact two `music_file` fields into an otherwise valid score, require deterministic repair with unchanged descriptions/prompts and zero extra LLM calls, pin the prompt wording, then rerun deterministic c03 at 120 words through canonical to ledger and OBS
- bible-worthy: yes -- required nested paths are not the same contract as exact nested key ownership, and a typed repair can faithfully repeat plausible but forbidden production metadata
- confidence: HIGH
- status: FIXED IN CODE / AWAITING LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-07 -- Gemma interleaved complete P5 beat blocks through repair
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification, prompt `d29b63d8-1890-40a4-a1ea-370bc9b02406`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 produced a strict BroadcastScore with complete typed beats but returned to an earlier `shot_id` after starting another shot; the typed repair repeated the same A/B/A topology and the run failed closed after 11:51 with `beats for each shot must form one contiguous block`
- root cause: the prompt named contiguous shot blocks and Python rejected interleaving, but the contract gave no concrete valid/invalid sequence example and the repair ladder had no safe deterministic ordering projection for otherwise valid authored beats
- fix: the base and repair prompts now state that the beats array is chronological and must never be reordered, give an A/A/B-valid and A/B/A-forbidden example, and require a fresh shot row/ID for a return cut; the P5 structural repair preserves the exact beat sequence and all authored beat content, clones only the reopened shot's mechanical row under a collision-safe ID, retags only the later run, and accepts only after the full score graph/content post-validator passes
- verify idea: interleave a valid score as shot_01/shot_03/shot_01 while keeping clues before reveal, require byte-identical beat-ID order and content with only the reopened-run shot IDs changed and zero additional LLM calls; force an ID collision and a hidden graph defect to prove deterministic naming and fail-closed behavior; rerun the identical c03 120-word seed through canonical to ledger and OBS
- bible-worthy: yes -- collection completeness does not imply ordered graph topology, and a typed repair can repeat a structurally plausible interleave indefinitely
- confidence: HIGH
- status: PARTIAL IN `09222618` -- the clone/retag projection was correct, but its repair-factory-only placement missed the typed-repair response; see PBUG-20260712-08

## PBUG-20260712-08 -- P5 deterministic repair did not run on the typed-repair response
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `09222618`, prompt `76cb5ca2-0ac7-4b2b-9b64-705b30f0cf75`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 base output again interleaved a closed shot; the repair-prompt factory could not accept the base after projection because another hidden validator defect remained, so it correctly requested typed repair. Gemma's typed-repair response repeated the same interleaving, then went directly to post-validation and failed after 12:33 without ever receiving the safe clone/retag projection
- root cause: deterministic P3/P5 structural normalization lived only inside `repair_prompt_factory`, which runs before the typed-repair model call. `structured_call` validates the typed-repair response directly; it does not call the factory a second time for a schema-valid content failure
- fix: hash each actual raw response for audit first, then run the same narrow P3/P5 projection inside the lane's slot-output wrapper on every ladder attempt. A projected model is serialized back to the ladder only when the complete pass post-validator clears; otherwise the original raw output and its real defect continue through the normal typed-repair path
- verify idea: make a base P5 response contain both a safe topology defect and a separate safety defect so it must reach typed repair; return a safe typed-repair response that still repeats A/B/A; require the per-attempt projection to preserve beat order, split the return shot, complete with exactly one repair model call, and produce resolving ledger boundaries
- bible-worthy: yes -- repair factories are not attempt-wide output middleware, so deterministic repairs placed only there can be bypassed by the response they requested
- confidence: HIGH
- status: FIXED IN CODE / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-09 -- raw P5 projection was not the schema-validated acceptance boundary
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `d024bc18`, prompt `51932200-9d57-499f-aae8-76f6fcf01631`, E4B creative + Mistral technical, 2026-07-12
- symptom: both the P5 base output and its typed repair were schema-shaped BroadcastScores with the same reopened-shot A/B/A defect; the slot-output projection did not accept either response, and the shared ladder failed closed after 12:36 with `beats for each shot must form one contiguous block`
- root cause: the clone/retag projection was still coupled to raw-string collection normalization before `structured_call` had created the strict `BroadcastScore`. That wrapper is useful for wrong-depth collections and nested extras, but it is not the guaranteed acceptance boundary for every schema-valid P5 response. A production response can therefore arrive at post-validation with the safe topology defect intact.
- fix: `P5` now applies the clone/retag projection inside its schema-validated post-validator. Every base, structural retry, and typed-repair response that parses as `BroadcastScore` must cross this hook. It mutates only the accepted in-memory score's mechanical `shots`/`beats` ownership, verifies the complete grounded score again, then runs authored-surface validation. The prompt also asks Gemma to silently scan the final beat sequence and mint a fresh shot row before emitting a return cut.
- verify idea: disable the older raw score normalizer in a mocked runner; a base A/B/A score must still produce a closed ledger with one extra cloned shot and no extra model call. Repeat with a separate safety failure on the base output so typed repair is required; its A/B/A response must clear through the same schema boundary. Run the identical c03 120-word seed to ledger and OBS.
- bible-worthy: yes -- a raw-output middleware hook is not a substitute for the strict typed object boundary where an artifact is actually accepted
- confidence: HIGH
- status: FIXED IN CODE / FULL SUITE + BUG BIBLE GREEN / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION
