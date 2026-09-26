# Known test failures

`EXPECTED_FAILED_NODEIDS` in `tests/conftest.py` is an **empty frozenset on
purpose**. An empty set makes every fail print as NEW, so a window diffs the
failing SET, never the count.

Do not fill that set with a remembered 45-49 list. A guessed ledger hides a
real break.

## Two gates (operator 2026-09-17)

A ~10 minute full suite is accepted as the **chunk gate** on a mature row.
Iteration uses the **working gate** (seconds).

Working gate, from the pack root:

```
$env:PYTHONUTF8=1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_working_gate.py
```

Add the nodeid you are in:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_working_gate.py tests\test_cast_lock.py::test_auto_registry_stamps_voice_refs
```

Print the default set: `scripts\otr_working_gate.py --list`

Chunk gate, once per green row (~10 min):

```
$env:PYTHONUTF8=1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
```

`scripts/otr_working_gate.py` DEFAULT_NODEIDS holds closed-row pins still
worth re-running plus any OPEN [GO_FORWARD](GO_FORWARD_PLAN.md) CODE verify
that is already green. Drop a path in the same commit that closes its row.
Do not add a still-red verify -- that is the row, not the gate.

Worktree-only (not a live-checkout finding):
`test_installed_pack_not_stale` and the two `test_w45_campaign_bank_pinning`
rows.

## Last chunk-gate SET (2026-09-26, HEAD `88211ed2`)

16965 passed, 0 failed (after the voice-route deletion: 16973 at `eadb8ab2`,
less 10 tests removed with it, plus 2 renamed). The SET is empty: no
known-failure class is open.
The 34-failure set recorded at `90e75d33` (2026-09-17) is fully closed, and
the four portable-bank route tests left with the voice-route subsystem.

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
See `apple/PROD_BUG_LOG.md` PBUG-20260812-02 (status FIXED).

When a nodeid enters `EXPECTED_FAILED_NODEIDS`, document it here IN THE SAME
COMMIT: the nodeid, why it fails, the tracking item, and the exit condition.
When the set empties again, this file returns to the empty-set paragraph above.

(Restored 2026-08-07: the conftest referenced this file in seven places but it
was never created -- the guard's own instructions pointed at a dead path.)
