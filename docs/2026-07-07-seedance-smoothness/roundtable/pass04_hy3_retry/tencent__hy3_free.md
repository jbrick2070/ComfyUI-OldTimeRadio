<!-- requested_model: tencent/hy3:free | resolved_model: tencent/hy3-20260706:free -->

VERDICT: yes-with-fixes — ambiguous softener regex for standalone handheld and unspecified softeners_applied content block a single correct build.

MUST-FIX BEFORE BUILD:
1. [Softener Order] Rule 7 "standalone `handheld`" has no regex; implementors may choose `\bhandheld\b` vs `^handheld$` vs space-delimited. Fix: specify exact pattern `r"\bhandheld\b"` (case-insensitive), applied after rule 2 so "handheld dolly" already replaced; replacement `stabilized` lowercase.
2. [Helper Contract / Metadata] `softeners_applied: list[str]` undefined what strings to store. Fix: specify list contains the matched source substring for each rule triggered, in order of application (e.g., "dynamic dolly push", "whip-pans", "white-hot", "aggressively"), exactly as captured by regex match (pre-case-normalization). Makes test "softener names are stable" checkable.
3. [Helper Contract] Empty prompt behavior unstated despite "caller must pass non-empty". Fix: add explicit `if not prompt.strip(): raise ValueError("empty prompt")` (or return unchanged with changed=False). Recommend raise to fail fast. [ASSUMPTION] raising won't break caller guarantee.
4. [Helper Contract] Marker check "First check for this stable marker" doesn't state match type (substring vs end-anchored). Fix: specify `if MARKER in prompt:` (substring anywhere) before softening; if true return (prompt, meta with changed=False). Prevents divergent implementations.

SHOULD-FIX:
1. [Tests] "second and third helper calls are byte-identical" ambiguous which input. Fix: state test calls helper twice on same original grounded prompt and asserts outputs equal; additionally call helper on its own output and asserts idempotent (changed=False).
2. [Adapter Integration] "Log one structured, sanitized line" – sanitization undefined. Fix: define sanitized = meta dict plus `duration` field, no raw request, no PII; use existing logger.
3. [Manual QA] Claim "Seedance seed is documented as non-deterministic" is [ASSUMPTION] not in grounding. Add verify step: confirm in provider API docs or code comments before relying on 2-3 samples.

OPTIONAL / NICE-TO-HAVE:
- Extract MARKER and SMOOTH_MOTION_CLAUSE as module constants to avoid drift.
- Add test that asserts clause appended exactly once and ends with marker.

CUT THESE:
None — plan is lean and scoped; all sections serve smoother Seedance motion without editing style JSON.

VERIFY-AT-BUILD checklist (items depending on code not shown / earlier UNVERIFIABLE):
- verify: `CloudSeedance2Engine._text_prompt_input(request)` exists and returns final `text_prompt` string (not in grounding excerpts).
- verify: `_partner_inputs()` currently builds top-level `{"model": ..., "seed": ..., "watermark": False}` shape (grounding partial; confirm in eng_cloud_video.py).
- verify: `model["prompt"]` is correct dict key sent to partner node (grounding shows `model.prompt` field; confirm assignment path).
- verify: `OTR_CLOUD_SEEDANCE_MODEL` env parsing accepts "Seedance 2.0" alias (grounding says accepted aliases normalize; confirm in cloud_model_ids.py).
- verify: duration clamp 4..15 in `_duration_seconds()` (grounding states; confirm code).
- verify: softener regexes compile and transform grounded `music_open` string to expected output (run test).
- verify: `ByteDance2ReferenceNode` exposes no camera-smoothing field (grounding confirms; confirm at build we set no unknown field).