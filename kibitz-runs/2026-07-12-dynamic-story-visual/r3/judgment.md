# r3 judgment -- dynamic-story-visual (WIRING). Claude, sole judge.

Panel: codex `gpt-5.6-sol` @ `ultra` (confirmed in codex_model_selected.txt /
codex_reasoning_selected.txt); antigravity `gemini-3.5-pro`. Driver anchor written
FIRST. Codex: VERDICT no. Antigravity: VERDICT yes-with-fixes. Judge: the WIRING
math (node 96 / link 284 / repoint-don't-renumber) is CONFIRMED correct by both
panels and by my own probe of the real file -- but the round surfaced two defects
in the CONSUMPTION path that would have shipped dead LLM output, plus a silent
content-coercion path. Doc goes to rev 4.

---

## 1. The wiring itself: CONVERGED

Nobody disputed the delta. Both panels independently reproduced it:
`last_node_id 95 -> 96`, `last_link_id 283 -> 284`, new link
`[284, 62, 1, 96, 0, "STRING"]`, repoint `[252, 96, 0, 90, 0, "STRING"]` and
`[255, 96, 0, 89, 0, "STRING"]`, node 62 `outputs[1].links`
`[16, 231, 232, 233, 252, 255] -> [16, 231, 232, 233, 284]`, new node
`outputs[0].links = [252, 255]`, `widgets_values: []`. Both flagged the same
single hard test pin (`tests/test_google_video_sfx_workflow.py:41`,
`last_link_id == 283`). Codex adds the one thing my anchor left implicit: the doc
must carry the LITERAL node record (pos/size/order/mode/flags/properties/slot
indexes/title), not a prose description. Folded.

## 2. Grounding verdicts -- codex (sol ultra)

| # | Claim | Verdict |
|---|---|---|
| 1 | Gate the pass on `meta.freeze_unload_ok`; add an ordering edge `96.done -> 81.gate_in` (link 285) so direction runs before the audio lane | **SPLIT.** The `freeze_unload_ok` PRECONDITION is **CONFIRMED and folded** -- a resident writer LLM plus a fresh creative LLM is an OOM on a 16 GB card, and the cascade already stamps the receipt (nodes/OTR_LedgerFreezeCascade.py:453-478). The ORDERING EDGE is **REJECTED for v1** (codex flagged it `[ASSUMPTION]` itself): ComfyUI executes nodes SERIALLY, the direction node unloads in `finally`, and the LLM-after-audio pattern ALREADY exists in production today -- MetaBrief re-resolves the writer LLM (nodes/otr_meta_brief_image_prompt.py:2087-2096, called :2158) with no audio ordering gate. The feature introduces no new interleaving hazard, and an inbound edge from the visual lane into the audio lane is a coupling I will not add on an assumption. Recorded as decision **I4** with the trigger that would reopen it (a live OOM). |
| 2 | **`parse_validate_tolerant` SILENTLY CLAMPS over-long authored strings and accepts** | **CONFIRMED -- the round's best find.** `validate_tolerant_data` catches `ValidationError`, calls `_clamp_overlong_strings`, re-validates, and proceeds with only a `log.warning` (nodes/_otr_structured_call.py:422-434). A 260-char `subject_note` or a verbose `era_tail` would be TRUNCATED and accepted -- Lesson 3's "never silently coerce authored content", violated by the machinery the r2 plan adopted. **Judge's fix is simpler than codex's:** do NOT declare `max_length` on authored strings in the vd schemas. No `max_length` -> no `ValidationError` -> the clamp arm never fires. Enforce every bound in `post_validator`, which raises `PostValidationError` and routes to the TYPED REPAIR rung. Zero changes to shared machinery; the authored text either comes back correct or fails closed. |
| 3 | Wrong-depth repair examples are impossible (P-A has no `shots`, P-B has no `motifs`); a factory-returned model that later fails `post_validator` exhausts with no cached repair prompt | **CONFIRMED.** The r2 test list mixed the two draft models. Folded: PER-PASS wrong-depth batteries, and the deterministic relocation is permitted ONLY within the same pass, only on a unique verbatim destination, and the factory must validate the COMPLETE contract (schema + post_validator) before returning -- the Lesson 3 rule "a deterministic repair must validate the complete downstream contract before returning; a partially valid schema object can consume the repair rung". |
| 4 | `DirectionSourceV1` is not closed: P-A uses an unspecified truncated projection, P-B a different one; `story_brief_terms` is duplicated under `meta` and `brief`; blanket `null -> ""` is invalid for boolean `skip` | **CONFIRMED** -- all three are defects I introduced in rev 3. Folded: the DTO is the closed hash + evidence universe; each pass gets a NAMED, DEFINED VIEW of it; `story_brief_terms` lives under `brief` only; normalization is per-type. Added rule: a pass may only cite evidence PRESENT IN ITS OWN VIEW (a model cannot cite what it never saw). |
| 5 | Preflight order cannot execute: P-B's prompt depends on P-A's output, and no cache entry exists at step 3; 15 x 240-char notes exceed a 900-token reservation | **CONFIRMED.** Folded: load inside the teardown-protected region; exact-preflight P-A; SCHEMA-MAXIMUM preflight for P-B before any generation; then exact-preflight each real batch. Batch size and output reservation are DERIVED FROM MEASURED MAXIMA, not the placeholder 15/900. |
| 6 | Provider propagation: remote entries have no HF tokenizer (so the must-fit helper cannot measure them); OpenRouter/Comfy floor output tokens; **Google has no branch in `make_constrained_generate_fn`** | **CONFIRMED.** The remote `response_format` mapping covers openrouter / comfy_credits / gguf_native only (nodes/_otr_constrained_generate.py:207-238) -- Google, which section 9.2 advertises for the cloud qualification leg, is absent. Folded: ONE provider-effective config + token-counting interface shared by preflight, invocation and receipt; and Google is either given a branch or explicitly excluded from constrained generation for the qualification leg. |
| 7 | `resolve_visual_direction(ledger) -> VisualStyle` cannot carry the shots index or the digest; the per-kind consumption matrix is undefined | **CONFIRMED** (and independently forced by antigravity 4). Folded: the seam returns a typed `ResolvedDirection` bundle (`style`, `shots_by_line_id`, `semantic_sha256`, artifact meta), and the doc now carries an explicit PER-KIND consumption matrix. |
| 8 | `attempt_journal` has no producer: `structured_call` returns only the accepted model and exposes no rung/outcome hook | **CONFIRMED.** Folded: an optional attempt-event sink on `structured_call` (a new, additive surface in section 8) -- a caller-side wrapper can hash raw calls but cannot see which rung fired or why. Receipts store ACCEPTED-state hashes + reasons, never rejected raw outputs. |
| 9 | Lifecycle order contradicts itself: step 7 seals + stamps BEFORE step 8 teardown, while 5.5 requires post-teardown serialization | **CONFIRMED** -- my own text. Folded to the cascade order: generate -> validate -> unload in `finally` -> on teardown failure persist the failed receipt and RAISE -> only then seal, persist, serialize, emit `done`. Also CONFIRMED: MetaBrief has NO ledger output, so the image-row digest cannot be stamped by MetaBrief -- it rides in the `image_prompts_json` payload and the DISPATCHER persists it. And every new receipt field is DYNAMIC-LANE-ONLY, or the named-pack byte-identity test breaks. |
| 10 | The audit chain is not durable: cache HITS clone the old row and overwrite provenance | **CONFIRMED.** The hit path does `fresh = dict(ref_row or {})` then `fresh.update({... "provenance": {"source": "cache_hit"}})` (nodes/otr_image_gen_dispatcher.py:627-632) -- a cloned row would inherit the PREVIOUS artifact's digest. Folded: on the dynamic lane the digest and the composed prompt are set EXPLICITLY on both the hit and miss paths, never inherited. |
| 11 | The delta promises a "FULL record" but omits it; add a topology test pinning `62:1 -> 96:0 -> {89:0, 90:0}` | **CONFIRMED.** Folded (the literal record + the topology test). |
| 12 | Five representations still disagree: two schemas but one fixture; motif/clue evidence duplicated inline AND in `field_evidence`; `era_cues` has no evidence; **dotted-path keys are simultaneously required and forbidden** | **CONFIRMED on the substance.** The last one is a real self-contradiction in rev 3 (4.1 forbids "schema-path strings used as field names" while `field_evidence` is keyed by dotted paths). Folded: `field_evidence` keys are a CLOSED ENUM of literal key names; row-shaped items (`motifs[]`, `clue_visual`) carry their evidence INLINE only; `era_cues` gets an entry; one fixture per pass. **Its "Lesson 11/12/13" citations are MISREADS** -- PRODUCTION_SPRINT_LESSONS.md has ten lessons plus the receipt; there is no Lesson 11, 12 or 13. The underlying asks are judged on their merits, not on the invented lesson numbers. |
| 12b | Evidence needs quote anchors, not bare IDs ("ID existence != support") | **ACCEPTED in a LIGHT form.** Full source-field/start/end offsets are over-engineering for a taste artifact. Folded: a `factual` entry must carry a short verbatim `quote` (<= 120 chars) that must be a SUBSTRING of the cited line's text in the DTO -- machine-checkable in the `post_validator`, and it kills the "cited but unsupported" hole r1 opened. |
| 13 | The test plan does not prove CHANNEL ISOLATION: run the same mocked story twice changing ONLY `visual_style` | **CONFIRMED and folded** -- a genuinely better test than the "before/after the feature" baseline, because it also proves the sentinel never leaks into the story lane. Both tests stay. |
| S1 | Node 96 has no `IS_CHANGED`, so identical frozen JSON could be served from ComfyUI's cache | **ACCEPTED as an explicit decision, not a defect.** Immutable-per-source caching is CORRECT and desirable here (same frozen story -> reuse the artifact), and it is only reachable when the upstream did not re-execute -- which the writer's `IS_CHANGED` (`time.time()`) makes rare. Folded as a stated, tested policy. |
| S2 | `direction_report` is claimed as the operator surface but is unwired | **ACCEPTED:** it must ALSO be logged, so the observability claim is true whether or not the output is wired. |
| S3 | Pin a concrete cloud model + auth/rate policy; Google treats every 429 as terminal | **ACCEPTED** into section 9.2 as a qualification precondition. |
| CUT 1 | Cut the unwired `gate_in` on node 96 | **ACCEPTED.** No producer, no sequencing use -- the `script_json` edge already orders the node. Dead scope (feedback_rip_legacy_dead_code). Node 96 has exactly ONE input. |
| CUT 2 | Cut `direction_report` if unwired | **REJECTED** -- kept, but S2's logging requirement makes it honest. A taste feature with no human-readable surface is unauditable. |
| CUT 3 | Cut persisted `story_binding.content_mutations` (always the constant 0) | **ACCEPTED.** The CHECK stays (refuse the stamp on any delta); the always-zero FIELD goes; the recheck outcome is recorded in the attempt journal instead. |
| CUT 4 | Cut the "survives a later re-direction" provenance promise | **ACCEPTED.** Reroll is out of v1 scope, so artifact HISTORY is too. The digest still ships (it is what makes the audit walk work at all), scoped to the current artifact. |

## 3. Grounding verdicts -- antigravity (gemini-3.5-pro)

| # | Claim | Verdict |
|---|---|---|
| 1 | `semantic_sha256` preimage names `binding` while the artifact names `story_binding` | **CONFIRMED** (my typo-level defect with real consequences). Folded: the preimage's `binding` is now explicitly a DERIVED two-key object, named and enumerated, not the `story_binding` object itself. |
| 2 | Preflight must budget the WORST-CASE REPAIR call, not the base call: `input + 2*max_new_tokens + envelope` | **CONFIRMED** -- independent convergence with codex 5. Folded as a computed envelope (the repair directive is a known string; measure it) rather than a magic 500. |
| 3 | ShotLock AND MetaBrief must unload the writer LLM in a `finally` | **CONFIRMED** -- converges with codex r2 M9 / r3 9. Both nodes are now named explicitly as required surfaces. |
| 4 | **`shots[]` notes would reach ONLY still_word cards** -- the cited seam is inside `compose_still_word_prompt`, which no other prompt kind calls | **CONFIRMED -- co-best find of the round.** Read the real file: :1004-1008 sits inside `compose_still_word_prompt` and applies to `character_video` word cards only. Character portraits, `_compose_char_scene_prompt` and `compose_still_prompt` would ignore the notes entirely -- i.e. rev 3 would have shipped an LLM authoring per-line notes that almost nothing consumes, in a doc whose own rule says an unconsumed note is dead scope. This forces the per-kind matrix (codex 7) to become the LAW: **`shots[]` rows exist ONLY for lines that actually yield a scene still or a word card, and the row universe is computed by the SAME pure target-derivation helper MetaBrief uses** (`derive_scene_still_targets` / `_iter_beat_lines`, nodes/otr_meta_brief_image_prompt.py:1016+). Portraits, radio-host and plates are per-episode/per-character surfaces and take NO per-line note. |
| 5 | `test_google_video_sfx_workflow.py:41` breaks | **CONFIRMED** (also in my anchor). |
| S1 | `peek_ledger()` can return `None` -> `AttributeError` | **CONFIRMED-plausible; folded** as an explicit None-guard. |
| OPT | Parameterized `get_era_tail` tests asserting the pack tail wins when `is_dynamic` | **ACCEPTED** into 9.1. |
| CUT | none | Noted; codex's cuts 1/3/4 are taken anyway. |

## 4. Panel disagreement, adjudicated

- **Verdict split** (codex "no" vs antigravity "yes-with-fixes"): codex is right on
  severity. Two of its findings (the silent clamp; the impossible preflight order)
  and one of antigravity's (the still_word-only note seam) each independently make
  rev 3 unbuildable as written. rev 3 was not "yes-with-fixes".
- **The ordering edge** (codex 1) is the only place I overrule a panel on a
  substantive design point rather than a grounding error, and I do it on evidence
  the panel did not weigh: MetaBrief already loads an LLM post-audio in production
  today, so the hazard predates the feature and the edge buys determinism, not
  safety. Recorded as I4 with an explicit reopen trigger.

## 5. Folded into rev 4

Wiring: the literal node-96 record; `gate_in` cut (one input); the topology test;
the re-baseline procedure (validator -> round-trip -> widget/INPUT_TYPES audit ->
link reconciliation -> master-hash stamp); the two breaking test pins; the
`strict_unknown_types=False` caveat (a green contract test does NOT prove the class
registered).

Contract: no `max_length` on authored strings (bounds move to `post_validator`);
per-pass wrong-depth batteries; the closed DTO + named per-pass VIEWS + the
"cite only what you saw" rule; quote anchors on factual evidence; `field_evidence`
as a closed key enum with inline evidence on row-shaped items; the corrected
preflight (worst-case repair envelope, measured maxima); the provider-effective
config/counting interface + the Google constrained-generation gap; the attempt-event
sink; the corrected lifecycle order (teardown BEFORE seal/persist);
`freeze_unload_ok` as a precondition; ShotLock + MetaBrief teardown; the per-kind
shots consumption matrix; dynamic-lane-only receipt fields; explicit digest on both
dispatcher cache paths; the channel-isolation test; `content_mutations` cut.

Deliverable: docs/2026-07-12-dynamic-story-visual-scope.md rev 4. final.md in this
folder is the rev-4 copy.
