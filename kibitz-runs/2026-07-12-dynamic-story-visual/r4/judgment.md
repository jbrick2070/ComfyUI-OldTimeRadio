# r4 judgment -- dynamic-story-visual (CONVERGENCE). Claude, sole judge.

Panel: codex `gpt-5.6-sol` @ `ultra` (confirmed in the run folder); antigravity
`gemini-3.5-pro`. Codex: VERDICT no (13 must-fix). Antigravity: VERDICT
yes-with-fixes (3 must-fix, all small closures).

## Convergence call

**The arc CONVERGES at r4 and stops here** (CLAUDE.md section 8: re-loop only on
new material; do not grind passes). The evidence for convergence is not that the
panel said "looks good" -- codex said "no" -- but that the CHARACTER of the
findings changed. In r2 and r3 the panel was overturning ARCHITECTURE (the artifact
was unimplementable; the repair ladder was the wrong mechanism; two consumers were
missing; the notes reached nothing). In r4 **not one finding proposes a different
design.** Every codex item is a precision fix INSIDE the agreed architecture --
exact field paths, exact bounds, an exact node `order`, an exact test -- and codex
itself closed with a VERIFY-AT-BUILD checklist rather than a redesign. Antigravity
independently landed on "yes-with-fixes". That is what convergence looks like on a
design of this size.

Six r4 findings were nonetheless REAL and build-blocking; three of them I confirmed
by reading the code myself. They fold into rev 5. The residue becomes an explicit
VERIFY-AT-BUILD checklist that Codex owns at implementation time -- not a fifth
round.

---

## 1. Grounding verdicts -- codex (sol ultra)

| # | Claim | Verdict |
|---|---|---|
| 6 | **`style_pack.is_dynamic = true` would FAIL `_validate_row`** | **CONFIRMED -- the worst defect in rev 4, and I introduced it.** `VisualStyle` is a `@dataclass(frozen=True)` with an EXACT field list (nodes/_otr_visual_styles.py:138-166) -- there is no `is_dynamic` field, and v2 validation rejects unknown keys. My own artifact would have failed my own validation step (5.2 step 6). FIX: `is_dynamic` is a RUNTIME-ONLY dataclass attribute (default `False`, set by the resolver after a strict v2 validation of the assembled payload); it is NEVER a persisted pack key. |
| 8 | **`derive_scene_still_targets` DELIBERATELY EMITS `b000_music_open`**, which vd-1 excludes -- so "shots set == target set" can never pass | **CONFIRMED.** The helper adds the opening-music target explicitly, and even emits it OPTIMISTICALLY on a pre-audio ledger (nodes/otr_meta_brief_image_prompt.py:1072-1086). This is my own r4 anchor concern B2, sharpened into a defect. FIX: the shots universe is a NAMED, LINE-BACKED PROJECTION of the target set -- targets MINUS the synthetic `scene_open` -- and the fail-closed gate compares against THAT projection. Also CONFIRMED-in-principle: a policy-dependent mesh role replaces the ordinary still with mesh fodder + a plate, and the 7.4 matrix consumes the note on neither -- so a note authored for such a line would be unconsumed. FIX: the projection excludes lines whose role resolves to a mesh/plate-only kind, or the matrix must consume it there; either way the rule is stated, not assumed. |
| 9 | **The talking-safety claim is FALSE**: the LLM-authored `positive_tail` / `era_tail` DO enter talking-head prompts | **CONFIRMED.** `text_prompt = finish_visual_prompt(meta, text_prompt)` runs on EVERY talking-head path (nodes/otr_shot_lock.py:634), and the finisher appends the era + style tails. D6 ("talking lane pinned to the safety base") was true only of the SUBJECT/MOUTH/FRAMING vocabulary. FIX: state the law precisely -- the safety base pins `portrait_look_talking`, `announcer_subject_*` and `motion_registers` (what actually protects lip-sync); the authored TAILS reach talking prompts exactly as a NAMED pack's tails do today, and the Python-owned anti-geometry lint (2.4) is what prevents an authored tail from injecting framing/mouth vocabulary. Add a HOSTILE-TAIL test. Also CONFIRMED: `get_story_brief_ltx` is SEMANTIC SCENE CONTENT -- 5.4 must suppress its LOOK authority only, never delete the narrative core. |
| 1 | P-A head-truncation contradicts "reads the final accepted story" and "NEVER truncates" | **CONFIRMED** (my anchor B3, resolved the strict way). FIX: no truncation. P-A receives the complete line spine; if it does not fit, the preflight ABORTS. If a 720-word episode cannot fit an 8192 cap, that is a real constraint the operator must see -- not something to paper over with a silent head-slice. |
| 3 | `DirectionSourceV1` has no unique ledger projection: `story_brief` is stored as a STRING with flat fields plus a nested `story_brief_terms`; `episode_id` is TOP-LEVEL, not meta; `cast[].traits` str-vs-list is ambiguous | **ACCEPTED as a defect; the exact paths are VERIFY-AT-BUILD.** I did not personally read `_otr_story_brief.py:841-860` / `production_ledger.py:551-571`, so I will not assert codex's exact shapes as fact. The REQUIREMENT folds: one canonical representation, exact source paths, closed types, per-field normalization, no duplicated lighting -- and Codex confirms the real paths against the live ledger before coding. |
| 12 | **The live control leg cannot byte-match a committed pre-feature prompt** (the writer always re-executes; local generation samples without a seed); and the canonical default visualizers may IGNORE stills, so a dynamic smoke could publish without exercising story-directed pixels | **CONFIRMED on the first half** (nodes/OTR_LedgerScriptWriter.py:3023-3028; nodes/_otr_model_loader.py:1122-1129 -- both already grounded in this doc). **ACCEPTED on the second half** as a live-proof design flaw: a smoke that does not consume stills proves nothing about this feature. FIX: byte-identity lives ONLY in captured-ledger/injected tests; the live legs run a still-consuming configuration and assert a generated still carrying artifact-authored clauses. |
| 11 | AST presence in `_NODE_MODULES` does NOT prove registration -- the loader catches import failure and silently omits both mappings | **CONFIRMED** (`__init__.py:351-367`). FIX: a COLD PACKAGE-IMPORT test asserting the class and label are actually in `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`, plus the node-contract pins. |
| 2 | Bounds are unbounded; the repair equation double-counts -- the real repair echoes only 400 chars of failed output | **CONFIRMED.** `_compose_repair` echoes `failed_output[:_FAILED_OUTPUT_ECHO_CHARS]` (nodes/_otr_repair_prompts.py:128-152), so `input + 2*max_new_tokens` overstates it. FIX: preflight `max(base_call_tokens, repair_call_tokens) <= configured_safe_cap`, where the repair call = original prompt + the BOUNDED echo + the directive, generating `max_new_tokens`. Every string, list cardinality, evidence count and `mood` gets an explicit bound. |
| 4 | The wrong-depth defense regressed: P-A's ownership prose omits top-level `era_cues`; test 9.1.3 puts `era_tail_2` in P-B, which has no era field | **CONFIRMED** (my sloppiness). FIX: ONE exact parent/key table covering every object and collection in BOTH passes, shared by schema, prompt, fixture, validator and repair. |
| 5 | Executable fields can evade grounding by declaring `kind="rationale"`; inline evidence has no `kind` | **CONFIRMED as a hole; folded in the LIGHT form.** Every EXECUTABLE `look.*` field and every shots row must carry at least one `factual` anchor with a verbatim quote. The full `field/path/index/start/end` coordinate scheme is REJECTED as over-engineering for a taste artifact -- a substring check against a named target text is machine-checkable and sufficient. |
| 7 | Stored provenance objects are open placeholders; `policy_from_meta` returns a dataclass, not JSON; a failed Direction teardown happens BEFORE `meta.visual_direction` exists, so `*_unload_ok` persistence is not implementable there | **CONFIRMED.** FIX: closed receipt DTOs with owners; a teardown failure is a NAMED EXCEPTION + a structured log, not a stamp on an artifact that does not exist yet; `runtime_policy` is a serialized snapshot, explicitly converted. |
| 10 | `order` is still a placeholder; output slots lack `slot_index`; the variant stamps need regenerating | **ACCEPTED.** `order = 4` with every order >= 4 shifted by one; `slot_index` on all three outputs; the variant/stamp regeneration is a VERIFY-AT-BUILD item (scripts/build_variants.py). |
| 13 | The sprint receipt omits the Dispatcher/ShotLock as durable provenance writers; no mechanical PASS criteria; no build sequence | **ACCEPTED.** |
| CUT: `direction_report`, `done` | **REJECTED (final ruling, third time asked).** Both have EXACT unwired precedents in the same live graph -- ShotLock ships `shot_report` (out[2]) and `done` (out[3]) with `links: []` today. They cost zero JSON, zero code, and `done` is the affordance the I4 reopen trigger needs (a one-link change, not a node-contract change). The panel's "dead scope" instinct is right in general and wrong here; recorded as a standing dissent rather than relitigated a fourth time. |
| CUT: `ResolvedDirection.artifact_meta` | **ACCEPTED** -- no listed consumer. |
| CUT: persisted `source_recheck` | **ACCEPTED** -- a STORED artifact can only ever say "clean" (a failed recheck refuses the stamp), so the field is tautological. The CHECK stays; its outcome is a log line and an exception, not a field. |

## 2. Grounding verdicts -- antigravity (gemini-3.5-pro)

| # | Claim | Verdict |
|---|---|---|
| 1 | The Google constrained-generation lane is left as an unresolved "state which" | **CONFIRMED and CLOSED by the judge:** the Google lane runs **UNCONSTRAINED**, with the typed-repair ladder as its sole schema defense, and `make_constrained_generate_fn` returns the plain Google closure for `provider == "google_api"`. That is the honest option: adding a Google schema branch is a separate piece of work and does not belong in this feature's blast radius. |
| 2 | The quote-substring check never defines the TARGET TEXT for non-line IDs | **CONFIRMED** -- a real hole in 2.2/7.1. Its mapping table is adopted verbatim (`line:` -> `lines[id].text`; `cast:` -> `character_description`; `brief:` -> the brief value; `meta:` -> the meta value; `title` -> `meta.episode_title`). |
| 3 | `clue_visual` is mandatory, but an episode may have no clue mechanism | **CONFIRMED.** Folded: `clue_visual.evidence` may be empty and `treatment` may state the absence -- and a `factual` anchor is therefore NOT required for it. (Only executable `look.*` fields and shots rows carry the mandatory factual anchor.) |
| S1 | `order = 4`, shift the rest | **CONFIRMED** -- converges with codex 10. |
| S2 | Nothing specifies how batched P-B shots are re-sorted into source-line order before hashing | **CONFIRMED** -- a real determinism hole: without it the semantic hash depends on batch completion order. Folded. |
| OPT | `slot_index` on outputs 1 and 2 | **ACCEPTED.** |

## 3. What r4 did NOT surface

No panelist proposed a different architecture, a different storage location, a
different slot, a different seam, a different workflow delta, or a different
failure posture. The disagreements are about EXACTNESS, not DESIGN. Both panels
accepted: the two-model split, `DirectionSourceV1`, the `structured_call` ladder,
the P-A/P-B split, the three-consumer resolve seam, the teardown barrier, the
node-96/link-284 delta, the fail-closed matrix, and the qualification ladder.

## 4. Folded into rev 5 (the final revision of this arc)

`is_dynamic` off the persisted pack; the shots universe as a named line-backed
projection (minus the synthetic open, minus mesh/plate-only lines); the corrected
talking-safety law + hostile-tail test; `get_story_brief_ltx` preserved as semantic
content; no P-A truncation; `max(base, repair)` preflight with the bounded 400-char
echo; every bound explicit; one parent/key table for both passes; mandatory factual
anchors on executable fields; the evidence target-text mapping; `clue_visual`
absence allowed; the Google lane declared unconstrained; closed receipt DTOs;
teardown failure = exception + log; `source_recheck` and `artifact_meta` cut;
`order = 4` + shifts + `slot_index`; deterministic re-sort of batched shots; the
cold package-import registration test; the live legs made still-consuming and
byte-identity moved off them; sprint-receipt PASS criteria + build sequence; and a
VERIFY-AT-BUILD checklist.

**ARC COMPLETE.** r1 (arc) -> r2 (coding plan) -> r3 (wiring) -> r4 (convergence).
6 agent calls (2 agents x 3 rounds), codex on gpt-5.6-sol @ ultra throughout.
Deliverable: docs/2026-07-12-dynamic-story-visual-scope.md rev 5. final.md in this
folder is the rev-5 copy.
