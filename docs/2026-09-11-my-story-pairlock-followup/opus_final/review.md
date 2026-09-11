## Verdict

**Approve as bounded repair.** No must-fix defect is demonstrated in the supplied delta on a reachable current path. The three dispositions are consistent with the code shown, and the residual concerns I can raise are either evidence-limited or below the must-fix bar.

## Point-by-point

**1. Age join (`_scene_source_context`, lines 1708–1710).** The normalization is correct as written: both candidates are `str(...).strip()`, the filter is `casefold() not in {"", "n/a"}`, row precedes planned, and a double-unknown yields `""` rather than a fabricated band. `matches[0] if len(matches) == 1` preserves the "unique exact normalized name only" rule; ambiguous names fall back to `{}` and the projection simply reports absence. The final `json.loads(json.dumps(...))` snapshot is a genuine mutation barrier.

*Residual, not must-fix:* line 1714 still uses raw truthiness for gender — `row.get("gender") or planned.get("gender") or ""`. A legacy row carrying `"n/a"`/`" "` would mask a known planned gender, which is the exact failure mode just normalized one line above. Root's own disposition 1 states production `set_cast` strips these fields, so I have no reproducer and cannot call this confirmed. It is a one-token symmetry fix (`_norm` the pair as done for age) if root wants the projection internally consistent; leaving it is defensible under the bounded scope.

**2. CastLock fallback (lines 1202–1249).** The distinction is sound and non-overlapping: `not gender` raises `"source gender unspecified"` *inside* the try so it lands on the same fallback path, but only after the `google_tts` hard raise at 1176–1180, so no engine escapes the Google refusal. `gender_unservable` therefore cannot be stamped on a blank-gender row and `gender_unspecified` cannot be stamped on an explicit-but-unservable one. `fallback_ref is None` still reports "NOT cast" and `continue`s without stamping, so the no-reference row is not falsely provenanced. Re-raise at 1221–1222 keeps Google exceptions live for all other unservable genders — no dead branch. I agree with root that stamping `presentation_gender` from the *chosen* reference is the fix, not a regression: `_stamp` (1653) reads `ref.gender`, i.e. the voice that actually speaks, which is precisely what the row label cannot answer for ANNOUNCER/`other`.

*Limitation (cannot verify):* `_stamp` is defined at 1628 as `_stamp(entry, ref, *, fallback="")`, but all call sites shown use `_stamp_row(entry, ..., fallback=...)` / `_stamp_row(entry, ref)`. The wrapper/alias binding `_stamp_row` is outside the supplied window. Given the full suite passes (14,399) including the fallback legs, this is almost certainly an intentional wrapper; but a keyword-signature mismatch there would be a hard `TypeError` on the fallback path. One-line confirmation, not a claimed defect.

*Low-severity, evidence-limited:* `provider_voice_id` is written only when truthy (1659–1661). If an entry arrives already carrying a provider id (legacy cloud ledger) and is re-cast to a local ref, the stale id survives next to a non-matching `voice_ref_id`. I have no reproducer that a same-entry engine switch occurs in production, and the comment's byte-identity goal is legitimate; noting only so root can decide whether an explicit `pop` on the local branch is wanted later.

**3. `structured_call` (rejected gates).** Verified against the shown exits: all six returns (975, 1034, 1084, 1122, 1183, 1257) route through `_parse_and_validate(... post_validator ...)` except the typed-repair return, which applies `post_validator` inline at 1085–1088 and raises `PostValidationError` on failure — so the prebuilt instance is not a bypass. `validate_tolerant_data` (684–689) does a real `model_validate` then the structural check; no `model_construct`, no success cache, no zero-attempt path is visible. `notify_attempt(None)` now precedes every success return including 1106, matching the stated mutation test. `last_raw = ""` is reset before each `_invoke_slot`, so a stale head cannot leak into a later ladder WARNING. I concur that requiring identity declarations on every schema/list, or rejecting unused mappings, would break valid current callers (P2 dialogue has no stable key; P3 lists are strings) — correctly not implemented.

## Qualification limits

- Scope is the shown delta only; full raw source remains authority.
- 51 inherited failures are unchanged and out of scope; the +20 pass delta is consistent with added controls.
- Prompt-and-pixel qualification stays **OPEN** pending fresh canonical; nothing in this delta changes graph/interface (23/63/37 unchanged), so the rerun is a confirmation, not a gate on the code.

The bounded code is sound.