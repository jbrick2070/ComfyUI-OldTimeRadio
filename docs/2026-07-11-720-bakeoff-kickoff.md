# 720-Word All-Banks Bake-off -- Kickoff Prompt (self-regrounding)

**Written 2026-07-11. The code is changing fast; every concrete fact below is a
snapshot, not gospel. GATE -1 exists precisely because this doc will be stale by
the time it runs.**

## GATE -1 -- REGROUND (do this before anything else, every time)

Re-derive every volatile fact from the live repo; where this doc disagrees with
the repo, THE REPO WINS:

- Bank list: read `nodes/story_packs/banks.json` + the pipelines registry for
  the CURRENT runnable banks (snapshot list below may have gained/lost lanes).
  `custom_source_bank` stays excluded (fails loud by design) unless its row says
  otherwise.
- Receipts: re-cite which banks have live >=120w receipts from actual ledgers /
  HANDOFF_LOG -- the Gate-1 waiver list is receipt-driven, never memory-driven.
- Harness surfaces: confirm the smoke status file name/format
  (`scifi_smoke_status.txt`, KEY=EXITCODE lines at snapshot time), the canonical
  launcher path, and the watchdog script still exist under the same names.
- Model pins: read the CURRENT canonical writer widget values for the
  creative/technical slots -- pin whatever canonical ships that day, identically
  across all legs. Record the resolved labels.
- Laws: re-read `docs/PRODUCTION_SPRINT_LESSONS.md`, `AGENTS.md`, `CLAUDE.md` --
  they are authoritative over this doc.
- Suite baseline: whatever is green at HEAD is the bar; numbers in old docs are
  history.
- If any structured-pass prompt/schema changed since a lane's last green smoke,
  that lane's 30w gate result is STALE -- re-run its 30w before counting it green.

## GATE 0 -- Preconditions (hard)

All three scifi lanes 30-word GREEN (status file all zeros AND obs artifacts
Test-Path-proven). Claim the coder slot in `docs/GO_FORWARD_PLAN.md` (scope +
base SHA); parallel windows read-only (Lesson 10); stage only owned changes.

## GATE 1 -- The 120-word rung (Lesson 6: nobody skips the ladder)

Banks WITHOUT a cited >=120w live receipt run one 120-word canonical leg first,
full proof chain. At snapshot time that means the three new scifi lanes +
scifi_fable2; re-derive per GATE -1. Two failures on the rung = DNF for the 720
batch (log the PBUG, move on). Before every 120/720 leg, size context from the
REAL artifact (Lesson 5): line counts, evidence rows, graph width, repair
envelope (often the largest call); fail loud if a provenance-sensitive prompt
cannot fit the measured safe context.

## GATE 2 -- The 720 batch

One 720-word episode per qualifying bank. Snapshot list (RE-DERIVE):
science_news, media_archive, public_domain_story, shakespeare, original_radio,
scifi_fable2, scifi_codex, scifi_gemini, scifi_sonnet.

- ONE VARIABLE (Lesson 8): the bank is the experiment. Identical pinned model
  slots across ALL legs; record model labels, slot assignments, prompt IDs.
- Reset per CLAUDE.md section 4 (selective CIM kill, port 8000 empty, baseline
  VRAM); boot ONCE via the UTF-8 canonical launcher loading the REAL
  `workflows/otr_canonical.json`; one server log + one leg log per run; watchdog
  on every leg.
- Per leg, patch `source_bank` + `target_words=720` through the headless
  widget-patch path -- never edit canonical.
- Full proof chain per leg (Lesson 7 -- API SUCCESS is not proof): saved ledger
  + receipts, asset under `otr/episodes/<ep>/`, `obs_publish OK`, final file
  under `otr/obs/`. Record requested-vs-actual words (720 is a one-use advisory
  steer -- record, never reject), discard/repair/reroll counts, wall time.
- Never read ledger files mid-leg (WinError-5 class). Live failure = PBUG entry
  (Lesson 9) + `/kibitz` fan-out + root fix; any structured-pass fix audits all
  FIVE representations in lockstep (Lesson 2) before the bank's single re-roll.

## GATE 3 -- Blind judging

Extract final scripts from frozen ledgers; strip ALL lane identifiers
(bank/engine/provenance/HUD); randomize as Story A..N; seal the letter->bank key
where judges cannot see it. TWO independent Fable judges score each story /10:
premise originality, radio-craft structure, character distinctness, dialogue
quality, ending payoff, source-fact integration, 720-length pacing -- paragraph
rationale each + ranked top three. Anchor judge reconciles on argument quality
(never averaging), unseals, writes `docs/<date>-720-bakeoff/VERDICT.md`: ranked
table, rationales, identities, word-count table, and one page on what the
winning lane's design got right.

## GATE 4 -- Close out

Suite + Bug Bible green; commit AND push every green chunk to v2.0-alpha;
HEAD==origin. End with the full SPRINT RECEIPT block from
PRODUCTION_SPRINT_LESSONS.md, every field filled (30/120/720 receipts, model
pairings, live ledgers, published assets, prod bug entries, remaining risks).
Release the coder slot. Final crown = OPERATOR EYEBALL blind listen against the
verdict board -- the doc's ranking is advisory until Jeffrey listens.
