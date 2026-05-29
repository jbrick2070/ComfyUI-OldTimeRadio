# Build 1 — tension floor fix spec (DRAFT — do not apply until integration session)

**Recommended approach:** widen the schema floor from `ge=1` to `ge=0`. Mirrors the existing BUG-LOCAL-282 precedent for `length_target_words`, smallest honest change, `tension` is advisory (not consumed by the render path today). Alternative (clamp generator output to >=1) rejected: more code for a field nothing currently reads.

## Exact edit (spec only — do NOT apply yet)
- **File:** `nodes/_otr_stage1_plan.py`
- **Old (lines ~260-265):**
```python
    tension: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Sprint 2: 1..5 tension level. Strictly monotonic NOT required.",
    )
```
- **New:**
```python
    tension: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Sprint 2: 0..5 tension level. 0 = no/idle tension. Strictly monotonic NOT required.",
    )
```

**Uniqueness caveat:** there are multiple `ge=1,` occurrences in the file — match the full `tension` block (with the "Sprint 2: 1..5 tension" description) so the edit is unique. The `AxisScore` `conint(ge=1...)` lives in a different module; no collision there.

## Draft test
`test_stage1_tension_floor_draft.py` (this folder). Verified: pre-fix (ge=1) the bug is `xfail` + post-fix assertion `skip`s; post-fix (ge=0) -> 4 passed + 1 xpassed, proving `tension=0` validates while `tension=-1`/`tension=6` still reject and `tension` stays Optional. Source was restored to pristine `ge=1` after simulation.

## Regression gate for this change
Bug Bible + core + audio byte-identity must stay green. Add the draft test to `tests/` (rename off `_draft`). This is measurement hygiene only — do NOT expect a score change.
