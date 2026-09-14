# BRIEF: 4060 confirmation pass on main @ 2.1.2

**For the Claude window running ON the 4060.** Self-contained — assume no
memory of anything before this file. Written 2026-09-14 after ~90 commits
landed since the last verified 4060 pass, including a branch promotion
(`v2.0-alpha` retired, `main` is now the live branch) and a real bug that
shipped in 2.1.0/2.1.1 and was fixed in 2.1.2.

## 0. WHY THIS RUN EXISTS

Since the last 4060 confirmation:
- `v2.0-alpha` is retired. **`main` is the branch now** — CLAUDE.md 0B's pull
  ritual targets `main`, not `v2.0-alpha`. If your local remote's fetch
  refspec is still narrowed to `v2.0-alpha` only (check
  `git config --get-all remote.origin.fetch`), widen it:
  `git config --add remote.origin.fetch '+refs/heads/main:refs/remotes/origin/main'`
- **2.1.0 and 2.1.1 shipped a real defect**, fixed in 2.1.2: the canonical's
  `custom_premise` widget held a hardcoded "toy boots" story that silently
  overrode four of six banks' fetchers (`scifi_news_pro`, `media_archive`,
  `public_domain`, `shakespeare`). A stranger installing either of those two
  versions got the same story on four of six bank picks, with no error.
- Stable Audio 3 became the music default "everywhere it can run" (was already
  true on the canonical since alpha.28; this widened it).
- The writer node's 36 widgets were reordered and the workflow repaired to
  match — the single riskiest class of change this repo has (positional
  `widgets_values` drift). It went through a QA round that found and fixed a
  blocker before landing.
- The gallery now lists **exactly one** template: `otr_canonical`. The 92
  hardware-specific variant files are gone from the picker (16 variant JSONs
  remain on disk for headless/profile use, not as gallery entries) — the
  install experience is now "load the one graph, it resolves your device."
- **The canonical's video roles now read `viz_mxc_cpu` / `viz_green` /
  `viz_camera`** — the zero-download, no-still-consuming visualizer lanes,
  each carrying `(audio-reactive, no scene image)` right in the saved widget
  label. This is the FLOOR configuration proven on this card 2026-09-07
  (16:16, ~12 GB total download, image field `none`). It is now the shipped
  default rather than a hand-picked test config.

**None of the above has been confirmed with a real render on this card since
it landed.** That confirmation is this brief's job.

## 1. FIRST ACTIONS

```
git fetch origin main
git log --oneline HEAD..origin/main
git pull --ff-only origin main
```

Say what came down. If your checkout is still on `v2.0-alpha`, switch:
`git checkout -B main origin/main`.

**Sync the live ComfyUI install from the checkout.** The live pack is a
Manager/registry install (plain files, no `.git`), not a symlinked checkout —
find it (`find` for a directory named `comfyui-old-time-radio` under
`AppData\Local\Comfy-Desktop`), then:

```bash
# check for shipped-surface deletions since whatever is currently live
# (compare pyproject.toml version in $LIVE against git log to find the commit)
git diff --diff-filter=D --name-only <live-commit>..HEAD -- . ':!tests' ':!docs' \
  ':!scripts' ':!.github' ':!kibitz-plugin' ':!kibitz-runs' ':!apple' ':!viewer' ':!tools' ':!assets'
# remove any listed files from $LIVE, THEN:
git archive HEAD | tar -x -C /tmp/otr_export
cp -rf /tmp/otr_export/. "$LIVE"/
find "$LIVE" -name "__pycache__" -type d -exec rm -rf {} +
```

Verify: `grep "^version" "$LIVE/pyproject.toml"` matches `pyproject.toml` in
the checkout, and `diff` on `workflows/otr_canonical.json` is empty.

**Restart ComfyUI.** A file sync alone does not take effect — Python already
has the old modules loaded in-process.

## 2. THE CONFIRMATION RENDER — the main event

1. Open ComfyUI. Confirm the console banner reads *"the only one the gallery
   lists"* and that Browse Templates → EXTENSIONS → `comfyui-old-time-radio`
   (or `ComfyUI-OldTimeRadio` on a git-clone install) shows **exactly one**
   entry: `otr_canonical`. More than one, or a missing one, is a regression —
   report it before doing anything else.
2. Load `otr_canonical`. **Change nothing.** No dropdown, no widget.
3. Queue Prompt.
4. Watch the console for `[OTR.assets] READY engines=...` — confirm
   `z_image_turbo` does **not** appear in that line (it's nominally the image
   dropdown value, but all three video roles are no-still lanes, so
   PBUG-20260907-03's fix should mean it is never actually fetched). If it
   downloads 19 GB of z_image weights here, that fix regressed somewhere
   between alpha.28 and now — stop and investigate before continuing.
5. Let it run to completion.

## 3. PASS CRITERIA — all four, together (section 6 of CLAUDE.md, restated)

1. `RESULT SUCCESS` in the log.
2. `obs_publish OK`.
3. **The file actually on disk** in `otr/obs/` — check it, do not infer it.
4. Zero hand steps between "load the template" and that file.

On top of the standard four, verify these specifically because they are what
this run exists to confirm:

- **Runtime near 16 minutes**, not 27–42. A much longer runtime means it did
  not take the viz lane it's supposed to.
- **The published filename's image field reads `none`**, not `zimg` — same
  logic as the console check in step 4, confirmed at the artifact.
- **The music field reads `sa3`**, not `mgen`.
- **The story is not about toy boots.** Skim the ledger/script text
  (`otr/audio/*_ledger.json` or the episode's own captions). If it's the toy-
  boots premise regardless of which bank fired, the 2.1.2 fix did not actually
  land the way the commit claims — report that as a live-artifact contradiction
  of a code-level fix, which is exactly the class of thing a real render on
  real hardware catches that a unit test cannot.

## 4. THE FULL SUITE, ON THIS BOX'S OWN STACK

A lot of new test coverage landed with the widget surgery
(`tests/test_combo_defaults_are_real_choices.py`,
`tests/test_no_shipped_graph_carries_a_premise.py`,
`tests/test_widget_migration_pairs_values_by_name.py`,
`tests/test_widget_schema_order_matches_live_input_types.py`,
`tests/test_widget_surgery_tool.py`). It was QA'd and proven on the 5080's
stack. Prove it independently on THIS box's Python/torch stack — a green
suite on one machine's environment is not evidence for another's, and this
card has caught real environment-specific defects before that the dev box's
own suite could not see.

The venv has no pytest installed; use a scratch `--target` install rather than
touching the venv (established pattern this session):

```bash
PY="<the live install's venv python.exe>"
"$PY" -m pip install --quiet --target <scratchdir>/pytest-lib pytest
PYTHONUTF8=1 PYTHONPATH=<scratchdir>/pytest-lib "$PY" -m pytest -q -p no:cacheprovider tests/
```

Compare the failing-nodeid set against a clean baseline (stash any local
changes, rerun, restore) rather than trusting a raw pass/fail count — the
guard in `tests/conftest.py` (`EXPECTED_FAILED_NODEIDS`) will tell you plainly
if something regressed versus what main declares expected.

## 5. REPORT

Append to `docs/4060_DRILL_LOG.md` with the same rigor every entry this
session used: the verified filename, byte size, runtime, and the console
line proving no wasted image download. Commit and push together (section 7)
— `main` now, not `v2.0-alpha`.

If ANY of section 3's four confirmation checks fails, that is the headline,
not a footnote — it means ~90 commits of QA on the 5080 missed something a
real 8 GB card caught, which is this box's entire reason for existing.

## 6. CONSTRAINTS THAT DO NOT CHANGE

- Nothing is ever hidden from a dropdown. A fix adds a row; it never filters
  or hides one.
- Do not save new profile JSONs speculatively — if a gap is found, name it in
  the report and let the operator decide the fix, per the same standing rule
  that killed the three-tier profile-saving idea until "all testing is done
  with all machines."
- Report the actual numbers. If the "16:16 / ~12 GB" figures above turn out
  wrong on the current build, say the real ones — this brief's estimates come
  from a run nine days and 90 commits old, and confirming or correcting them
  is the point of the exercise, not a formality.
