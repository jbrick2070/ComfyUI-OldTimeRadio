# r2 judgment -- dynamic-story-visual (CODING PLAN). Claude, sole judge.

Panel: codex `gpt-5.6-sol` @ reasoning `ultra` (confirmed:
codex_model_selected.txt = gpt-5.6-sol, codex_reasoning_selected.txt = ultra);
antigravity `gemini-3.5-pro`. Driver anchor written FIRST (driver_anchor.md).
Both panelists returned VERDICT: no. So did I. Doc rev 2 is not buildable.

Every panel claim below was checked against the REAL Windows files (file tools /
Desktop Commander). Misreads are discarded, not folded.

---

## 1. Grounding verdicts -- codex (sol ultra)

| # | Claim | Verdict | Grounding I did |
|---|---|---|---|
| 1 | Doc conflates the LLM RESPONSE with the Python-ASSEMBLED artifact; needs two models (draft vs final); `structured_call` needs a pydantic model, contradicting the "stdlib-only" claim in §7.3 | **CONFIRMED** | §7.3 of the doc says `nodes/_otr_visual_direction.py` is "(pure/stdlib, lazy)"; `structured_call`'s schema param is `type[T]` bound to `BaseModel` (nodes/_otr_structured_call.py:551-563). The doc's own 2.1 shape carries derived receipts + hashes the model must never author. |
| 2 | "2 attempts, matching max_reseed" is not a failure-class ladder AND is numerically wrong (max_reseed=2 = 3 calls); use `structured_call(max_attempts=3)` + typed repair factory + post_validator; persist an attempt journal | **CONFIRMED** | Independently found in my anchor M2. `_DEFAULT_MAX_ATTEMPTS = 3` (:69); rungs at :668-689, :700-721, :724-775, :783-811; `StructuredCallFailedError` :819-823. The attempt-journal addition is codex's and is accepted. |
| 3 | Projection / evidence universe / hashes not in lockstep: brief is fed verbatim but is NOT in the projection, so brief-derived evidence cannot resolve; "exactly the merge-owned fields" is FALSE | **CONFIRMED** | `_MERGE_OWNED_ROW_FIELDS` is 19 fields (char_count, word_count, arc_phase, compose_flags, beat_intent, target_words, dialogue_slot_id, shot_id, reviewer_*, needs_render_realign ...), NOT the doc's 9-field list (nodes/production_ledger.py:1441-1459). The doc's 2.3 claim is a misstatement I wrote. Fix = one explicit `DirectionSourceV1` DTO, serialized into the prompt AND hashed AND the sole evidence universe. |
| 4 | `semantic_sha256` preimage self-contradicts: includes `story_binding`, which contains `freeze_timestamp`, while claiming timestamps are excluded; read path never recomputes it | **CONFIRMED** | Doc 2.1 lines: preimage = "... + story_binding (timestamps and model_receipt EXCLUDED)". `story_binding.freeze_timestamp` is a timestamp. Real contradiction. |
| 5 | **MetaBrief cannot consume the artifact as designed** -- `derive_image_prompts` has NO ledger param and calls the meta-only resolver; MetaBrief is missing from §7 | **CONFIRMED (must-fix)** | `def derive_image_prompts(cast: list, meta: dict, *, llm_fn=None, ..., lines=None, ...)` (nodes/otr_meta_brief_image_prompt.py:1570-1574) -> `_vstyle = _resolve_style(meta)` (:1609). `generate()` parses `led` but passes only cast/meta/lines (:2137-2168). The doc's 4.3 "consumers all hold the parsed ledger" is FALSE for the deepest consumer. |
| 6 | `shots[]` overloads `beat_id`; `b000_music_open` does not exist when Direction runs | **CONFIRMED** | There is NO top-level `beats` array: beats are DERIVED from lines inside ShotLock, `beat_id = line_id` (nodes/otr_shot_lock.py:260-288). `OPENING_MUSIC_BEAT_ID = "b000_music_open"` is SYNTHESIZED later by `derive_opening_music_beat(ledger, fps)` (:291-301). So the doc's projection row "beats \| beat_id ... as shipped" describes an array that does not exist, and a shots[] row keyed to the music beat could never validate at direction time. |
| 7 | **`render_driver` is an unfixed dynamic consumer** -- calls `get_visual_style(meta)` directly and uses `get_story_brief_ltx` as the runtime prompt core | **CONFIRMED (must-fix)** | `_vstyle = get_visual_style((ledger or {}).get("meta") or {})` (nodes/_otr_video_engines/render_driver.py:1248) -- raises on the sentinel. `core = get_story_brief_ltx(_meta)` (:2069) then `finish_visual_prompt(...)` (:2080) -- a SECOND look authority in the video lane, exactly the 4.4 problem the doc only fixed for stills. render_driver is absent from §7. |
| 8 | Model API + budget unresolved; `make_generate_fn` does not enforce `context_cap`; `max_new_tokens: 0` and a string `top_p` are not budgets; ShotLock's `callable(prompt)->str` is NOT the GenerateFn signature | **CONFIRMED** | Independently found (anchor M1): nodes/_otr_model_loader.py:1108-1137 has no context guard. The two-signature trap is real: `structured_call` wants `slot_fn(messages, *, temperature, max_new_tokens)`; ShotLock's `llm_fn` is `callable(prompt:str)->str` (nodes/otr_shot_lock.py:513-516). The doc must not mix them. |
| 9 | **The VRAM guarantee is false**: MetaBrief reloads the writer LLM right after Direction unloads it | **CONFIRMED (must-fix)** | `_resolve_writer_llm(meta, warnings)` in MetaBrief delegates to ShotLock's resolver (nodes/otr_meta_brief_image_prompt.py:2087-2096) and is called at :2158. So the doc's live-smoke assertion "VRAM returns to baseline after the direction stamp BEFORE image dispatch" is not an invariant of the graph. |
| 10 | Exact JSON delta: ids 96 / 284; `[284,62,1,96,0,"STRING"]`; repoint 252/255 to source 96; freeze fan-out becomes `[16,231,232,233,284]`; and **the dropdown test that pins choices == registry must change** | **CONFIRMED** | Independently grounded: `last_node_id=95`, `last_link_id=283`; node 62 out[1].links = `[16,231,232,233,252,255]` (16/231/232/233 = SignalLost + the three audio nodes -- they must KEEP reading the raw freeze json). `tests/test_visual_style_widget_3c.py:62-66 test_choices_are_exactly_the_registry` asserts `choices == list(vs.list_style_ids())` -- the sentinel breaks it. r3 owns the full delta; the ids and the fan-out list fold now. |
| 11 | Qualification ladder incomplete | **CONFIRMED** | Same as my anchor M6 / Lesson 6. |
| 12 | **The live replay leg cannot pass**: writer + cascade `IS_CHANGED` return `time.time()`, so a requeue writes a FRESH story | **CONFIRMED (must-fix)** | `IS_CHANGED` -> `return _t.time()` with the comment "always re-execute" (nodes/OTR_LedgerScriptWriter.py:3023-3028). A canonical re-queue can never be an "unchanged replay". Replay moves to a deterministic CPU test over a CAPTURED frozen ledger. |
| 13 | No PROD_BUG_LOG / sprint receipt closeout; verify the file under `otr/obs/`, not just `obs_publish OK` | **CONFIRMED** | Lessons 7 + 9 + receipt. Same as anchor M8/M10. |
| S1 | Receipts must be provider-truthful (local hardcodes do_sample/top_p; remote may override) | **CONFIRMED** | nodes/_otr_model_loader.py:1122-1129. |
| S2 | The cited canonical hasher falls back to `repr()` and never raises -- wrong for a fail-closed seal | **CONFIRMED** | nodes/production_ledger.py:292-302. Fold: a strict JSON-only canonicalizer that REJECTS non-JSON values / NaN / inf / non-string keys. |
| S3 | `OTR_TEST_MODE=1` makes `stamp_durable` skip disk writes -> split the merge-survival test | **CONFIRMED** | nodes/production_ledger.py:408-452 + tests/conftest.py:38. |
| S4 | Put `visual_direction_semantic_sha256` on image rows / video creative sidecars | **ACCEPTED** (promoted from SHOULD to REQUIRED) | Without it the 6.6 audit walk breaks the moment a re-direction happens. |
| S5 | Capture pre-feature named-pack prompt/hash/request-key fixtures BEFORE implementation | **ACCEPTED** | The byte-identity test in 8.3 has no baseline otherwise. |
| S6 | Verify `peek_ledger()` episode id matches the wire ledger before `stamp_durable` | **ACCEPTED** | Cheap guard against a stale process singleton. |
| CUT 1 | Cut `gate_in`/`done`/`direction_report` | **REJECTED (partially)** | `done` as an unwired opaque ordering STRING is the shipped repo idiom (ShotLock out[3] `done` has `links: []` in the live canonical file; node 89 carries an unwired `gate_in`). Zero JSON cost, zero code cost, and it is what a later ordering wire needs. `direction_report` is the operator's only human-readable surface for a taste feature -- KEEP. Ruling: keep all three, and the r3 delta states explicitly that they ship UNWIRED. |
| CUT 2 | D9: Python constant, no JSON packaging fork | **ACCEPTED** (matches anchor S5) |
| CUT 3 | D5: pin typography/backdrop to the safety base for v1 | **ACCEPTED** (matches anchor S4) |
| CUT 4 | Cut `rationale.composition_notes` | **ACCEPTED** -- non-executable by construction, pure budget. Cut. |
| CUT 5 | Cut `authored_fields` + `content_mutations` | **REJECTED** | Both become DERIVED (Python-written) receipts, not LLM-authored fields, so they cost zero output tokens. `authored_fields` is the machine-checkable proof that the assembly obeyed the whitelist -- it is exactly the "durable storage and replay receipt" Lesson 1 demands. Keep, reclassified. |
| CUT 6 | No test-only workflow mutation / replay node | **ACCEPTED** -- consistent with codex 12. |

## 2. Grounding verdicts -- antigravity (gemini-3.5-pro)

| # | Claim | Verdict |
|---|---|---|
| 1 | MetaBrief + render_driver missing from §7; both call `get_visual_style` and crash on the sentinel | **CONFIRMED** -- independent convergence with codex 5 + 7 and with my own reads. This is the round's strongest finding: TWO panels and the judge found it separately. |
| 2 | Five representations missing (base prompt, worked fixture, repair prompt) | **CONFIRMED** (= anchor M3). |
| 3 | Model-diversity ladder missing | **CONFIRMED** (= anchor M6). |
| 4 | D8 budgets unresolved; require a must-fit fail-loud | **CONFIRMED** (= anchor M1/M4). Its `prompt_must_fit=True` phrasing matches the real marker (`_otr_scifi_codex.py:308-311`). |
| 5 | Repair ladder undefined; "two-rung ladder" proposed | **CONFIRMED as a defect; its FIX is SUPERSEDED** -- the repo already has a 4-rung ladder (`structured_call`). Do not build a two-rung one. Codex's version folds. |
| 6 | Sprint receipt missing | **CONFIRMED**. |
| 7 | PROD_BUG_LOG expectation missing | **CONFIRMED**. |
| S1 | Label every field authored/derived/measured; nested rows as closed sets | **CONFIRMED** (= anchor M9, Lesson 1). |
| S2 | Exact ids 96 / 284 | **CONFIRMED** -- matches the real counters (`last_node_id=95`, `last_link_id=283`). |
| S3 | `freeze_timestamp` breaks semantic-hash stability | **CONFIRMED** (= codex 4). |
| S4 | Protect the `_load_all()` directory scan from the sentinel | **MISREAD** | The design adds NO pack file (the placeholder-pack option was REJECTED in r1), so the registry sweep (nodes/_otr_visual_styles.py:329-336) never sees `dynamic_story`. The REAL adjacent defect is the dropdown-equality test (codex 10), which is folded. |
| CUT | "None -- the plan is already streamlined" | **NOTED**; codex's cuts 2/3/4/6 are taken anyway. |

**Panel disagreement, adjudicated:** antigravity says the plan needs no cuts; codex proposes six. Judge: four cuts land (composition_notes, D5 typography, D9 JSON fork, test-only replay node), two are rejected (the node's standard `done`/`gate_in`/`report` surface; `authored_fields`/`content_mutations` as derived receipts). Neither panel is wrong about scope -- they weight "dead scope" (codex) against "receipt completeness" (Lesson 1) differently, and the reclassification to DERIVED dissolves the conflict.

## 3. Findings the panel MISSED that the anchor keeps

- **The unguarded-context seam itself** (anchor M1). Codex 8 gets close ("the direct `make_generate_fn` path does not enforce `context_cap`") but neither panel states WHERE the guard actually lives -- only the writer's own slot wrapper has it (nodes/OTR_LedgerScriptWriter.py:664-699), so ANY new node that copies the ShotLock idiom inherits zero protection. That is a repo-wide latent class, not a doc bug, and it goes in the doc as a named requirement.
- **The one-pass-does-not-fit sizing argument with real numbers** (anchor M4) -> the P-A / P-B split. Both panels flagged the budget as unresolved; neither proposed the decomposition. Folded as the design's answer to Lesson 5, with ShotLock's `batch_size=15` (nodes/otr_shot_lock.py:499-508) as the precedent.
- **D2 answered from precedent, not taste** (anchor M5): fable2 runs authorship passes on `creative` and extraction/audit passes on `technical`, both through `structured_call` (nodes/_otr_scifi_fable2.py:1129-1137 technical; :1166-1174, :1201-1209, :1394 creative). D2 = `creative`, CLOSED.
- **Registration must be the literal `_NODE_MODULES` dict** (anchor M7): the canonical-workflow contract test AST-parses that literal (tests/test_workflow_contract_validation.py:41) and never executes the `_otr_class_registry` merge (__init__.py:335-349). A class-registry-only node is invisible to the workflow gate.

## 4. Rulings folded into the doc (rev 3)

1. Two typed models: `VisualDirectionDraftV1` (what the LLM returns, strict, extra=forbid) and `VisualDirectionArtifactV1` (what Python assembles + seals). §7.3's "stdlib-only" is dropped -- pydantic is required.
2. `structured_call` ladder replaces "2 attempts", with a failure-class table + attempt journal + `post_validator` carrying the semantic checks.
3. `DirectionSourceV1` DTO: ONE closed serialization that is the prompt input, the hash preimage, and the evidence universe. Brief fields included. `beats` row DELETED from the projection (no such array exists).
4. `shots[]` re-keyed to `line_id`; `b000_music_open` explicitly excluded from authored rows.
5. Semantic-hash preimage enumerated key-for-key; `freeze_timestamp` excluded; both hashes recomputed on read.
6. §7 gains `nodes/otr_meta_brief_image_prompt.py` and `nodes/_otr_video_engines/render_driver.py` as REQUIRED surfaces; 4.4's look-authority rule extends to the video lane (`get_story_brief_ltx` core).
7. VRAM: per-node `finally` teardown barrier; the smoke assertion is rewritten to "no local LLM resident when the Dispatcher's GPU work begins", which requires MetaBrief and ShotLock to unload too.
8. Context budget: must-fit slot_fn, the real cap (8192), the per-pass token equation, P-A/P-B split.
9. Section 8: model-diversity ladder, PROD_BUG_LOG expectation, sprint receipt, pre-feature baseline capture, replay moved off the live leg.
10. D2 CLOSED (creative). D5 CLOSED (safety base). D8 CLOSED (equation). D9 CLOSED (Python constant). D10 CLOSED (flag on the resolved VisualStyle).

Deliverable: docs/2026-07-12-dynamic-story-visual-scope.md rev 3. final.md in this
folder is the rev-3 copy.
