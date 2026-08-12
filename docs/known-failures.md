# Known test failures

NONE in `EXPECTED_FAILED_NODEIDS`. The suite is green.

## One STRICT XFAIL, which is a different thing and is tracked here anyway

`tests/test_writer_model_field_shadowing.py::
test_no_writer_model_field_shadows_a_BaseModel_attribute`

- **why it xfails:** PBUG-20260812-02. `CastShape.register`
  (`nodes/_otr_scifi_fable2.py:281`) shadows `BaseModel.register`, which exists
  because Pydantic's `ModelMetaclass` inherits `ABCMeta`. An instance built
  without that field dumps a BOUND METHOD, and the writer dies on the next
  `json.dumps` with `Object of type method is not JSON serializable`.
- **found by:** a LIVE headless leg, 2026-08-12 -- the first leg of the 45-word
  every-visual-path campaign, 78 s in, before any video work.
- **why xfail and not EXPECTED_FAILED_NODEIDS:** `strict=True` means fixing the
  field makes this test PASS, which fails the suite and forces the marker to be
  deleted in the same change. That is the lane matrix's strict unexpected-pass
  discipline applied to a single defect, and it keeps the suite green today
  without letting the bug be forgotten.
- **tracking:** `docs/PROD_BUG_LOG.md` PBUG-20260812-02 (status OPEN -- root
  cause proven and reproduced, production trigger not yet located).
- **exit condition:** the writer lane renames the field (root fix) or gives it a
  default (containment). Delete the `xfail` marker in that same commit; the
  three characterization tests beside it stay, and
  `::test_the_rule_finds_exactly_the_known_offender_and_no_other` will need its
  expected list updated to `[]`.

The suite's headline therefore reads **1 xfailed -> 2 xfailed** from
2026-08-12. The other is long-standing.

`EXPECTED_FAILED_NODEIDS` in `tests/conftest.py` is the executable authority;
this file is its human-readable ledger, and the conftest guard names both.
When a nodeid enters that set, document it here IN THE SAME COMMIT: the
nodeid, why it fails, the tracking item, and the exit condition. When the set
empties again, this file returns to NONE.

(Restored 2026-08-07: the conftest referenced this file in seven places but it
was never created -- the guard's own instructions pointed at a dead path.)
