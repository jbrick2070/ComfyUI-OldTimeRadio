# r1 JUDGMENT -- audition evidence guards

Driver: Claude (Cowork), sole judge. Reviewer lane this round: **Antigravity
(`agy`, gemini-3.7-flash-high)**. Codex is **not installed on this box**
(`codex` is not a recognised command), so the Codex seat was filled per the
operator's 2026-08-17 substitution directive -- a **Fable** subagent ran the same
round COLD, without seeing the driver anchor, per the standing r1 rule. Its
verdict is folded in separately below.

Every claim below was checked against the real Windows files at HEAD `345ea230`
before being accepted or discarded.

---

## ACCEPTED -- verified against the files

### A1. `render()` does not resume; it re-renders and overwrites. (antigravity MUST-FIX 1)

**CONFIRMED, and it corrects the driver anchor.** Anchor section 2B called the
script "deliberately resumable" without qualification. The truth is narrower:

* `render()` (`scripts/otr_lemmy_cross_engine_audition.py:288-297`) builds a
  fresh `row` with `"clips": {}` and assigns `manifest["engines"][engine] = row`
  **unconditionally** -- there is no "already complete, skip it" branch.
* `_write_clip()` (`:216-235`) ends in `os.replace(part, path)`, an
  unconditional overwrite.

So what actually resumes is the **manifest merge across runs**: `--engine dia`
preserves the *other* engines' rows because `_load_manifest()` reads the existing
file. Within the selected set, everything is re-rendered. A bare `--render`
re-run destroys all eight clips.

**And the driver found the sharper half while grounding this:** `_save_manifest()`
(`:268`) sets `manifest["generated_utc"] = _now()` on **every** call, and it is
called in the per-engine `finally` block (`:329`). So **any** run against the
cited directory rewrites `MANIFEST.json` with a new timestamp and breaks the
sha256 three records cite -- even a run that renders nothing at all. A guard that
only protects clip bytes would still let the citation rot.

### A2. Drop the runtime `cast_pools` import from the instrument. (antigravity MUST-FIX 2 / CUT 2)

**ACCEPTED, with a condition the reviewer did not state.** The fragility premise
is real and the driver verified it independently: `config/cast_pools.py` carries
**four different citation field shapes** --
`artifact_path`/`artifact_sha256` (`:389-390`, `:874-876`),
`audition_manifest_path`/`audition_manifest_sha256` (`:537-538`, `:1094+`), and
the nested `audition_manifest.{path,sha256}` dict (`:860`, `:995`). A cite-scanner
inside the instrument would silently fail to cover a fifth shape, which is the
same "partial guard reads as complete" defect 12.111 is about.

**The condition:** the reviewer's replacement -- refuse non-empty unless
`--resume` -- leaves a hole it did not name. An operator resuming into the
**cited** directory still destroys evidence, because `--resume` is exactly the
flag that permits writing there. That hole closes only if A1 is fixed properly:
**resume must be idempotent** -- skip any engine already complete whose clips
exist and hash-match, and **write no manifest at all when nothing was rendered**.
Then a resume against a complete cited directory is a genuine no-op and cannot
rot the citation. Accepted on that basis; the driver's Option D is withdrawn in
favour of it, because it is simpler and it removes a coupling.

### A3. The listen page is in scope after all. (antigravity MUST-FIX 3)

**CONFIRMED.** `scripts/otr_lemmy_listen_page.py:36` hardcodes
`_CAMPAIGN_DIR = os.path.join(_EPISODES, "lemmy_cross_engine")`. Anchor section 4
scoped the listen page out; that was right about `DECISIONS.json` and
`LISTEN.html` (neither is cited by hash, and `:334-336` already refuses to
clobber `DECISIONS.json`) and **wrong** about the campaign directory. Adding
`--out-dir` to the audition while the listen page can only ever look at
`lemmy_cross_engine` ships a new campaign nobody can listen to.

### A4. G1 must reach parity. (antigravity MUST-FIX 4)

**CONFIRMED -- independently, before the fan-out.** `otr_g1_lemmy_audition.py`
guards on `MANIFEST.json` alone (`:253-254`), creates `_KEY_DIR` with
`exist_ok=True` (`:156`), and offers `--overwrite` (`:236`). Its manifest backs
the *superseded* qualification whose own comment (`config/cast_pools.py:985-988`)
promises the audition "can still be re-verified byte for byte".

### A5. A roster pin, not an AST scanner. (antigravity CUT 1)

**ACCEPTED.** The anchor asked for a test that trips when a fourth unguarded
writer appears; the reviewer is right that a general scanner is brittle. A
parameterized test over the three known instruments plus a pinned roster -- so a
new evidence writer trips the pin and forces a decision -- gets the coverage
without the heuristics.

---

## REJECTED -- checked and false

### R1. "makedirs runs before preflight and the resident-server gate." (antigravity SHOULD-FIX 2)

**MISREAD.** `main()` calls `preflight(engines)` at `:377` and the resident-server
gate at `:382-386`, and only then calls `render()` at `:387`. `os.makedirs` is the
first line *inside* `render()` (`:282`), so both gates have already passed. No
dirty directory is left by an aborted preflight. Discarded.

### R2. "Receipt paths use forward slashes and will not resolve on Windows." (antigravity SHOULD-FIX 1)

**MISREAD, and disproved by execution.** The existing check already performs
exactly the join the reviewer warns about
(`tests/test_lemmy_provisional_tier.py:776`,
`os.path.join(root, "otr/episodes/.../MANIFEST.json")`). Driver ran it in
isolation on this box: `test_a_rendered_receipt_names_artifacts_that_exist_and_still_match`
**PASSED**, not skipped. Windows accepts forward slashes. Discarded.

### R3. The reviewer's stated assumption about `bark_preset_audition.py`.

**PARTLY WRONG, conclusion survives.** It does not write only to temp: `:38`
writes `manifest.json` and a CSV to `docs/2026-06-17-bark-voice/audition/` (only
the WAVs go to `tempfile.gettempdir()`). But nothing cites that manifest by
sha256 -- the driver's exhaustive grep of `config/` found citations only in the
audition family -- so it is correctly out of scope. No action.

---

## DRIVER CORRECTIONS TO ITS OWN ANCHOR

1. **Anchor section 2B overstated resumability.** Corrected by A1.
2. **Anchor section 5 claimed verify step 4 as new work.** It is not: the standing
   re-hash check **already exists** for the cross-engine routes
   (`tests/test_lemmy_provisional_tier.py:758-796`) and passes today. **The real
   gap is G1**: `tests/test_voice_identity_fix.py:740-761` asserts the record's
   *declared* hash string is unchanged and never opens the file to re-hash it --
   partial coverage that reads as complete. That is now the step-4 work item.
3. **Anchor section 4 wrongly scoped out the listen page.** Corrected by A3.
4. **Anchor Option D is withdrawn** in favour of A2's simpler shape.

## STANDING FACT ESTABLISHED THIS ROUND

All four cited manifests and six cited clips were re-hashed and **all match**:
cross-engine `ac55c90c...`, G1 `34dd4c9d...`, production-ceiling `344ccdf8...`.
**Nothing has rotted. This work is preventive** and must be written up that way.

## MEASURED BASELINE

Full suite on the settled tree at `345ea230`: **11028 passed, 110 skipped,
1 xfailed** (331.65s). This confirms the `HANDOFF_LOG` figure and shows
`GO_FORWARD_PLAN.md`'s BASELINES block (`10913`) is **stale by 115 tests** --
the fourth drift of that "single authority" receipt. Correct it in this window.
