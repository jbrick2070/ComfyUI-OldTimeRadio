# Known test failures

NONE. The suite is green.

`EXPECTED_FAILED_NODEIDS` in `tests/conftest.py` is the executable authority;
this file is its human-readable ledger, and the conftest guard names both.
When a nodeid enters that set, document it here IN THE SAME COMMIT: the
nodeid, why it fails, the tracking item, and the exit condition. When the set
empties again, this file returns to NONE.

(Restored 2026-08-07: the conftest referenced this file in seven places but it
was never created -- the guard's own instructions pointed at a dead path.)
