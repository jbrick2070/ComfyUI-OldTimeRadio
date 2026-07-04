# Widget Surface Audit v2 -- otr_scifi_16gb_full.json
Finalized 2026-07-04 at HEAD 8c3e4911 (post credits tail-chain + no-fallback rip). Supersedes v1 (2026-07-03).
Method: static AST inventory + consumption grep (widget_audit_raw.json / _v2.json) -> sonnet semantic pass -> kibitz 4-round arc (codex gpt-5.5 x4, Claude anchor+judge) + independent Antigravity manual review. Every claim below is judge-grounded against the real files; round artifacts in kibitz-runs\2026-07-03-widget-audit\ and \2026-07-04-widget-audit\.

## Headline
- 23 nodes, ~125 surfaced widgets. Zero positional drift (re-confirmed at HEAD; node 95 CreditsRoll exposes ZERO widgets).
- Zero dead widgets; no dropdown offers retired options. The clutter is confusion: 4 single-option placeholder dropdowns, 2 mode-conditional widgets that go silently inert, env-shadowed writer dials, 1 duplicated toggle pair.
- Retracted from v1: node 87 "duplicate alias options" (aliases live only in the pick-parse path, never displayed) and "node 93's burn is a legacy remnant" (tests deliberately pin 93 as owner; the CW-4 tear-out simply never finished -- it is an OPERATOR DECISION, not dead code).

## Findings (verdict != KEEP)
| Node | Widget | Verdict | Evidence (verified at HEAD) |
|---|---|---|---|
| 80 CastLock | delivery_profile | REMOVE SURFACE ONLY | cast_lock.py:100-103 single option "neutral"; kwarg stays (validated+stamped :128-176) |
| 81/82/83 voice nodes | stereo_policy x3 | REMOVE SURFACE ONLY | _otr_voice_node_common.py:235 + stable_audio_theme.py:119 single option "mono_safe"; kwarg stays (mono conversion :362 etc.) |
| 1 ScriptWriter | refine_target_grade | TOOLTIP (env-shadowed) | OTR_STORY_REFINE_BAR/PASSES override in headless |
| 1 ScriptWriter | 4x slot-model dropdowns | TOOLTIP | inert without the paired "openrouter:slot-a"-style pick in creative_writing_model/technical_model (:2098-2170) |
| 1 ScriptWriter | story_scaffold | OK AS-IS (downgraded) | already tooltipped :2196-2213; bidirectional env resolver :1662-1682 |
| 62 FreezeCascade | protagonist_only | TOOLTIP | manual_line_ids silently supersedes it (:246-266) |
| 86 CaptionBurn | burn_captions (+3) | OWNER DECISION | duplicate of node 93's live toggle; see Batch 3 |
| 92 VideoRenderBatch | engine / oom_index | TOOLTIP | engine read only in mode=="single" (:152-158); oom_index only in mode=="soak" |

Stats: 6 hard-removable widget entries (4 in Batch 1 + 2 on node 93 under the 86-owner path); ~8 tooltip fixes; everything else KEEP.

## Build plan (converged r4)
### Batch 1 -- surface-only removal (build-ready, ~1-2h, confidence ~95%)
- Remove INPUT_TYPES entries only; KEEP kwargs with defaults (behavior byte-identical).
- Exact vectors: node 80 ["default","auto_registry","neutral",true]->["default","auto_registry",true]; 81 ["indextts2","mono_safe"]->["indextts2"]; 82 ["kokoro","mono_safe"]->["kokoro"]; 83 ["stable_audio_3","mono_safe"]->["stable_audio_3"]. Validator length gate _otr_workflow_validator.py:175-179 hard-fails drift.
- Tests: test_cast_lock.py:62-64, test_announcer_voice.py:87, test_batch_character_voices.py:94-95, test_stable_audio_theme.py:80.
- Stale docstrings: cast_lock.py:18-20, _otr_voice_node_common.py:176-182.
- Gate: validator + JSON round-trip + suite + Bug Bible + test_audio_byte_identical.

### Batch 2 -- tooltip-only, NO key renames (~30min, zero risk)
Renames are schema migrations (_otr_workflow_apply.py addresses widgets by name) -- permanently cut from this effort.

### Batch 3 -- caption single-owner migration (operator gates direction; ~half day, confidence ~75-80%)
OPERATOR DECISION: 86-owner (finish CW-4) or 93-owner (ratify status quo).
CURRENT wiring (sonnet-verified at HEAD, do NOT "fix" if 93-owner is chosen): 12 -> 84 -> 86(pass-through OFF) -> 93(burns) -> 95 -> 85. Today's owner (93) already keeps credits caption-free; the rewire below applies ONLY if 86-owner is chosen.
If 86-owner, ALL of:
1. Chain: 84 -> 93 -> 86 -> 95 -> 85. CONVERGED INDEPENDENTLY by codex r4 AND antigravity reverify. Four grounded reasons (agy): (a) credits-first outputs *_with_credits, a suffix the ledger resolver does NOT strip -> CaptionBurn silently passes through, zero captions burned; (b) CreditsRoll appends via concat -c copy (lossless/fast) -- captioning after it would re-encode the whole concatenated video; (c) node 95 must stay terminal-before-mux so its declared_credits_tail_s feeds 85 slot 6 alongside the video path (the duration guard v_dur <= a_dur + tail + tol); (d) SDH captions caption dialog; credits have none. Full link referential integrity: links[] + input link fields + output links arrays + node order.
2. Port into otr_caption_burn.py: _procgen_blended suffix strip + sibling audio/ ledger fallback (from otr_post_upscale_procgen_blend.py:98-115); resolver today strips only _silent/_captioned/_final/_blend (:70-86).
3. Fix _default_out (:183-192): write beside input video / under otr\episodes\<ep>\, never the flat episodes root.
4. Enablement via canonical workflow + profiles ONLY (env-only path cut): set node 86 widget true where profiles say true; retarget widget_mapping.json:95-112 + 3 profile JSONs to OTR_CaptionBurn.
5. Strip node 93's whole caption path (widgets :827-845, kwargs, ASS routing; vector 13->11); clean stale positional comment :823-826.
6. Tests: invert test_workflow_live_passes_validator.py:56-87; rework test_post_upscale_procgen_blend.py:150-163 (_ass_filter_arg moves); add test_capability_profiles.py + test_workflow_apply.py to the gate; optional caption-free-credits visual smoke.
If 93-owner: fix otr_caption_burn.py's docstring; leave node 86 registered-but-unwired (do NOT delete the file: __init__.py registers it, test_caption_burn_cw4.py imports it).
Per section 9: one Fable final-gate pass before this batch merges.

### Standalone fix (new, r4): validate_workflow_links.py --strict-types false-flags nodes 80-83
CLI parses only literal _NODE_MODULES keys (tools\validate_workflow_links.py:61-81) and misses the dynamic new_node_modules_table() merge (_otr_class_registry.py via __init__.py). Runtime registration is fine (suite green, /object_info has all nodes). Fix the CLI to include the registry keys without importing node modules; then require strict-types green in CI. Related (sonnet): _otr_workflow_validator.py's own standalone fallback (:352-362) tries importlib.import_module("custom_nodes.ComfyUI-OldTimeRadio") -- an illegal dotted path (hyphen) -- so outside the ComfyUI app it always lands on an empty mapping; same artifact class, fix alongside.

### Verify-at-build checklist (r4)
Validator + round-trip + link integrity; re-run the widget-audit script expecting zero unintended drift; strict-types after the CLI fix; fresh-restart /object_info check (80-83, 86, 93, 95, 85); assets under otr\episodes\<ep>\ and final under otr\obs\; suite + Bug Bible + test_audio_byte_identical + test_capability_profiles + test_workflow_apply.

## Sequencing note
Batch 3 is derived AGAINST the landed credits tail (node 95, 5f510ebe/e346eeb4). Antigravity's re-verify (antigravity_reverify.md, HEAD 8c3e4911): STILL-VALID across all cites (one shift: _otr_voice_node_common docstring now ~:177-182), zero new must-fix, zero misreads, tail order CONFIRMED identical to codex r4, strict-types independently confirmed as a CLI-context artifact (static AST parse of __init__.py cannot see the dynamic _otr_class_registry merge).

## Panel record
codex r1: caption-ownership contradiction, hide!=delete, renames=migration. codex r2: 5-surface atomicity, exact vectors, test pins. codex r3 (verdict "no" on 86-owner as then specified): stage order, _procgen_blended, enablement, output contract, don't-delete-file. codex r4: node-95 chain re-derivation, sibling-audio fallback, strict-types root cause, cite refreshes. antigravity (manual, r1-r3 scope): independently verified all sonnet cites + exact vectors + widget_mapping retarget; zero misreads found in our draft. sonnet (operator-run, reverify): confirmed all vectors/node-95 wiring/mapping targets; added the _otr_workflow_validator fallback-import artifact; its "tail-order MISREAD" flag REJECTED by judge (the packet's question was explicitly the hypothetical 86-owner placement, which codex+agy answered identically) but its don't-fix-unbroken-wiring warning is folded above. Judge retractions of own v1 claims: 2 (alias duplicates; 93-as-remnant framing).
