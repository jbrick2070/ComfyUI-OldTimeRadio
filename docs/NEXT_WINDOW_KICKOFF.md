# NEXT WINDOW -- paste this whole block

resume the OTR build as a CODER window. Repo:
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch
`v2.0-alpha`, HEAD `75932b76` == origin. FIRST verify both repos are pushed --
OTR `75932b76` == origin/v2.0-alpha and survival-guide `02e8bcb` == origin/main.
Two `git ls-remote` calls.

Read `docs/GO_FORWARD_PLAN.md` -- "HOW TO TALK TO THE OPERATOR", then "QUEUE
STATE AT THE 2026-08-17 LATE CLOSE", the REVIEW ROUTING block, and BASELINES --
plus the top entry of `docs/HANDOFF_LOG.md`. State your MODEL & CREDIT BUDGET
rung first (the table is EMPTY -- header row, separator, no rows -- so cite the
per-window mapping paragraph beneath it) and the dated REVIEW ROUTING you read.

**BASELINES:** suite **10860 passed / 110 skipped / 1 xfailed**, Bible
**20/26/3** at **289** entries. The trailing `1` is an xfail, not a failure.

## YOUR JOB: CLEAR THE TITLE/IDENTITY FAMILY, THEN A GPU PROOF

Everything open is one family -- a title or identity being wrong, or its receipt
lying about it. Work it in this order. **Push per green chunk.**

**1. BUILD THE PBUG-20260817-05 FIX. It is already decided and grounded; just
build it.**
   * Remove `"--title", title` from `scripts/otr_gpu_soak_matrix.py` `leg()`.
     Verified safe: `leg()` correlates nothing by title (success comes from
     `"RESULT SUCCESS"` in stdout) and nothing reads the receipt's `title` key.
   * **Rename that receipt key `title` -> `leg_label`** and also record the
     ledger's real `episode_title`. Keeping a key named `title` holding a run
     label is the same one-field-two-meanings defect as Bible `12.110`/`11.61`.
   * **Add the cheap guard:** assert a headless run never produces
     `title_source == "user"`. It is a contradiction on its face and would have
     caught all 17 published harness titles on day one.
   * Accepted cost: soak legs now run `_generate_title_from_script`, an LLM call
     they never made. That is "mimic the entire workflow" and closes a real hole.

**2. BISECT ITEM I BEFORE BUILDING ANYTHING.** The wrong-person
`character_description` went **6.8% (Jul) -> 50% (Aug)** among pitch-bearing
ledgers -- measured twice independently. Find what changed in early August
first; you may be about to fix a symptom of a recent change. Everything else
about item I is in `docs/2026-08-17-item-I-wrong-person-description/roundtable/pass00_judgment.md`,
including five corrections to the driver's own anchor. **Cut at `casting_brief`
(98% carrier), not the pitch (81%, and it misses 7 of 31 affected episodes).**

**3. PBUG-20260817-04 -- the public_domain announcer invents a work title even
when handed the real one.** The fix worked and the model ignored it, so this is a
different root from item F. Needs a panel; do not guess at seam wording, which is
the mechanism item F already disproved.

**4. TITLE PROVENANCE** -- `docs/2026-08-17-title-provenance-SPEC.md`, reviewed
right-sized. Append-only stamp chain (value, source, stage, symbol, UTC
timestamp, replaced-value), because `title_source` is one slot and records only
the LAST writer.

**5. WHEN THE ABOVE IS GREEN, RUN THE GPU PROOF.** Reset per CLAUDE.md section 4,
boot headless, and run `scripts/otr_writer_bank_gate.py --banks
shakespeare,public_domain,original,media_archive --acts 1`. Read each episode's
announcer line and title card. **The bar: a real story title on the card, and the
announcer naming the work it actually performed.**

## HARD CONSTRAINTS -- disqualifying, not advisory

1. **`otr/obs/` publication may never be reduced, gated or relocated.** It is how
   he reads success: *"a test is not complete unless published to obs... if I see
   it in obs then it's somewhat a success"*, and *"if I don't see it in obs and it
   took more than 5 minutes it's a fail."* A previous tidy-up was reverted within
   minutes. Harness runs BELONG there.
2. **An automated run is a REAL EPISODE, start to finish** -- same code, same
   canonical graph, ending in a published artifact. Testing one part is HIS
   explicit exception to ask for, never one a fix imposes.
3. **No saved harness graphs.** Every run regenerates from
   `workflows/otr_canonical.json`. A fix may add a run-time PARAMETER; never a
   stored workflow.
4. **A missing reviewer never blocks an arc.** Substitute Fable / Sonnet / Opus /
   `anthropic-skills:roundtable` and state the roster honestly. Both agy lanes
   were quota-held this session; Codex has been out since 08-19.
5. Sonnet 5 QA on the finished diff BEFORE the push. THE LAW: an audit may never
   FAIL an episode. A render must degrade, never raise.

## TRAPS THAT BIT THIS SESSION -- all four are live

* **CITE SYMBOLS, NEVER LINE NUMBERS.** The driver's marquee citation
  (`build_description_prompt`) did not exist; the real symbol is
  `_build_user_prompt`.
* **`git check-ignore -v` EVERY new artifact path.** It bit four times:
  `kibitz-runs/`, `docs/2026-*/`, a plain new test file that was simply never
  added, and `kibitz/` being a NESTED GIT REPO whose commits never appear in OTR
  history.
* **Never write a commit message as a PowerShell inline `-m` string** --
  apostrophes shred it. Write the message to a file and use `git commit -F`.
* **A BROKEN REVIEWER RETURNS A CONFIDENT REPORT, not an error.** One lane
  produced a trace with `summary | summary | standard processing applied` and
  invented a file and class. kibitz now fails that structurally, but read every
  panel claim against the real files anyway.
