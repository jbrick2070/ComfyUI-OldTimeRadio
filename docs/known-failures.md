# Known test failures

NONE in `EXPECTED_FAILED_NODEIDS`. The suite is green.

## The strict xfail opened 2026-08-12 is CLOSED

`tests/test_writer_model_field_shadowing.py::
test_no_writer_model_field_shadows_a_BaseModel_attribute` was marked
`xfail(strict=True)` on 2026-08-12 for PBUG-20260812-02, and the marker was
deleted the same day when the field was fixed.

**The mechanism worked exactly as designed, and is worth keeping as the pattern.**
`strict=True` meant that fixing `CastShape.register` made the test PASS, which
failed the suite and forced the marker out in the same change. The bug could not
be quietly forgotten and the marker could not outlive its cause. That is the
lane matrix's strict unexpected-pass discipline applied to a single defect.

What replaced it is stronger than the original rule: the general check now
sweeps EVERY pydantic model reachable under `nodes/` (92 of them) and asserts no
field default can fail `json.dumps`, with a companion test proving the sweep
actually found the models so a broken import cannot make it vacuously green.
See `docs/PROD_BUG_LOG.md` PBUG-20260812-02 (status FIXED).

The suite's headline is therefore back to **1 xfailed**, the long-standing one,
after briefly reading 2 xfailed on 2026-08-12.

`EXPECTED_FAILED_NODEIDS` in `tests/conftest.py` is the executable authority;
this file is its human-readable ledger, and the conftest guard names both.
When a nodeid enters that set, document it here IN THE SAME COMMIT: the
nodeid, why it fails, the tracking item, and the exit condition. When the set
empties again, this file returns to NONE.

(Restored 2026-08-07: the conftest referenced this file in seven places but it
was never created -- the guard's own instructions pointed at a dead path.)
