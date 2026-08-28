# Dead code and slop — hunt v4, the deep sweep

**Hand to each reviewer INDEPENDENTLY; never show them each other's answers.**
Three prior rounds have stripped this repo's easy findings — v4 is for what
survives a good audit, so shallow sweeps will return nothing and that is the
wrong answer. Go where the prior rounds did not.

    REPO:   C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
    BRANCH: v2.0-alpha
    PYTHON: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe  (set PYTHONUTF8=1)
    TESTS:  python -m pytest -q -p no:cacheprovider tests   (~12,400 tests, ~7 min)
    CORPUS: C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\*\audio\*_ledger.json
            (~2000 frozen production ledgers — REAL EVIDENCE, use them)

**READ-ONLY. No edits, no renders, no server** — a GPU queue is often active.
CPU greps, AST parses, corpus scans, git archaeology and running tests are all
encouraged. Work from the CURRENT working tree.

---

## THE FOUR QUESTIONS, in the order that catches the most

**1. Does it actually DO anything?** A function that is called, computes, and
whose return nobody reads survives every reachability check ever written. Same
for a parameter accepted and never read, a receipt field no consumer reads, a
log value nobody greps. This shape has produced the best finding of every
round so far.

**2. Three layers.** A blocking dependency is a claim: A lives because B
references it, B lives because C references it — if C is dead, the chain is
dead, and at one hop it looks perfectly referenced. Walk to something
genuinely live (a registered node, a string in `workflows/*.json`, a CLI entry
point, a test asserting PRODUCTION behaviour) or call it dead. Report chains
as `symbol -> caller -> its caller -> reached: <live thing or "nothing">`.

**3. Is a mention a use?** An import never called, a name in a comment, a
string literal, recursion inflating grep counts. AST for the final call, every
layer.

**4. Do the COMMENTS tell the truth?** This repo's audits keep finding
confident comments that lie — "used by X" where X was deleted, "dormant" on a
module that runs every episode, wrong node ids and wrong counts in
freshly-written tombstones. Grep `used by`, `called from`, `consumed by`,
`wired into`, `dormant`, `deprecated`, and verify each against a call graph.
**Also spot-check NUMBERS in comments** (slot indices, file counts, node ids)
— wrong numbers in fresh comments bit this project four times in one day.

---

## WHERE TO HUNT — the unswept territory, in priority order

**A. `scripts/` — 90+ files, NEVER audited.** Every prior round excluded it.
`docs/LEAN_MEAN_CLEANUP.md` order 11 says "RE-GROUND PER FILE — do not reuse
the old bulk kill list", which is an instruction for exactly this pass, not a
protection. For each script establish: does anything invoke it (a test, a
launcher, another script, a documented operator command in docs/)? Does it
target machinery that still exists (engines, nodes, flags it references)? A
script whose target was deleted is dead however clean it looks. Note:
`.comfyignore` excludes scripts/ from the SHIPPED pack, so nothing here is a
user-facing surface — but the repo still carries the maintenance cost.

**B. Receipt and ledger fields nobody reads.** The ledger carries dozens of
stamped fields. Pick the writers (grep `meta\[`, `setdefault`, receipt
builders) and for a sample of ~15 fields, find the READER. A field written on
every episode and read by nothing is question 1 at the data layer. The corpus
tells you what is actually stamped; the code tells you what is actually read.

**C. Env vars.** Build the full inventory of `os.environ` reads across
nodes/ and scripts/ (there are many). For each: does anything SET it — a
launcher (`scripts/*.cmd`, `*.ps1`), a profile, a test, documentation telling
the operator to set it? An env var read in one place and set nowhere is a
knob that never turns; one SET somewhere but read nowhere is debris on the
other side. Report both lists.

**D. Test-support debris inside `tests/` itself.** Fixtures no test uses,
helper functions orphaned by retired tests, `EXPECTED_FAILED_NODEIDS` entries
for tests that no longer exist, patch targets pointing at deleted symbols.
The suite is ~12,400 tests; its own dead weight has never been audited.

**E. `config/` and data files.** `config/cast_pools.py` and friends, JSON/YAML
under the repo: keys nothing reads, entries for retired engines, voice rows
for presets no pool includes.

---

## WHAT IS NOT CRUFT — you will be wrong about these

* **Long explanatory comments are the house style and load-bearing.** Flag one
  only when it is factually WRONG about the code beside it.
* **Tombstones are intentional** — and they must carry `legacy` / `deleted` /
  `removed in` on the line naming a dead symbol, or the legacy-audit guard
  trips. Do not propose deleting tombstones.
* **Video engine lane duplication is a RULING** (2026-08-23), scoped to
  `nodes/_otr_video_engines/eng_*.py`. The TTS sidecar seam is separately
  authorized for consolidation. Check scope before invoking either.
* **`acceptance.py` imports nothing but `__future__`** — ratified, tested.
* **Per-node try/except in `__init__.py` is partial-install resilience.**
* **`docs/2026-*/` and `kibitz-runs/` are gitignored working notes.**
* **A test-only symbol is not automatically dead** — parked capability, safety
  tool, or unwired fix. Say WHICH. An unwired fix whose defect is still
  reachable is a BUG REPORT, not a cleanup item.

## Widget/socket findings — the full recipe or nothing

Inert widgets get removed WITH their migration (operator ruling). A removal
touches FOUR things: (1) `widgets_values`, (2) the `inputs` descriptor array,
(3) **every link whose `dst_slot` indexes past the removed descriptor** —
links index into the same array — and (4) **every Python call site passing the
removed name as a kwarg to the node method**, including tests. Variants are
GENERATED: fix `otr_canonical.json` (+ `otr_story_only.json`), then
`scripts/build_variants.py --all`, verify with `--check` plus
`tests/test_widget_value_alignment.py`, `test_canonical_widget_input_parity.py`
and `test_workflow_link_target_indexes.py`. Name the widget's index and
whether it is LAST (free) or mid-list (full re-index). See CLAUDE.md
"REMOVING A WIDGET TOUCHES THREE THINGS" — and note its list is one short:
the Python-kwarg fallout is the fourth thing, found the hard way.

## Already removed or ruled — do not re-report

`_load_canon_for_writer` · `_otr_voice_resolver` · `compute_cache_key` · the
cast-consolidation cluster · `nsfw_frame_qc` · `refine_target_grade` ·
`optimization_profile` (both sites) · `_OPTIMIZATION_PROFILE_CHOICES` ·
`normalize_dbfs`/`_normalize` · `_generate_bark_for_line` + helpers ·
`GemmaHeartbeatStreamer` · `_normalize_dialogue_names` · `_pick_accent` ·
the Bark preset-health cluster (migrated to `BarkSilentOutputError` in
`eng_bark.py`, then deleted) · `clean_one_line`/`validate_announcer_line`
length params · FreezeCascade's four compat widgets · both
`consistency_gate_warn_only` widgets · ShotLock's `image_done` socket ·
`_TIMEOUT_CTX` · the `force_vram_offload` import.

Still OPEN by design, leave alone: `_otr_scifi_p0_contract`'s cap slice and
`compact_p0_repair_context` (standing-ruling gated), `slot_matrix`/
`content_oracle` (REMOVE-AFTER-MIGRATION, migration unstarted), the
Chatterbox/Dia `_load_wav` consolidation (authorized, unexecuted),
`_voice_backends` (verified test-only, deletion pending), the OpenRouter GBNF
surface (RE-GROUND gated).

## Output format

    ### <short title>
    CATEGORY: stale-claim | unwired-fix | unreachable | debris | duplicate | inert-control
    CONFIDENCE: CONFIRMED | LIKELY | UNVERIFIED
    WHERE: path:lines
    WHAT: one sentence
    CHAIN: symbol -> caller -> its caller -> reached: <...>
    CONSUMED: does anything read what it produces?
    EVIDENCE: the scans you ran and their real numbers
    ATOMIC-WITH: every file that must change in the same commit
    RISK: what breaks if wrong, and the fast check
    PAYOFF: ~N lines, mechanical | needs-a-decision

End with **"WHAT I COULD NOT CHECK"** — honest and unpadded.

Twelve defensible findings beat forty guesses. In territory A-E, even five
solid ones justify the pass — and one unwired fix whose defect still occurs
outranks everything else you could find.
