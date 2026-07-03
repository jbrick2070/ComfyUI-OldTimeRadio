# Credits Enrichment — fresh-window KICKOFF

**Read this + `GO_FORWARD_CREDITS.md` (the source of truth) at the start of the execution window.**
Plan status: v4, SHIP-WITH-FIXES, build-ready. Reviewed by codex + agy + Fable + operator. All fixes folded.

## What this campaign does
Rip the credits (cast / voices / image+video engine receipts) OUT of node 12 `OTR_SignalLostVideo` (renders too early, before nodes 91/92 -> today prints blank voices + `(not recorded)`) and render ONE unified viewer roll LATE in a NEW terminal node `OTR_CreditsRoll`, fed by durable ledger stamps (S2) + the clip manifest. Cleanbreak, no fallbacks.

## Operator directives (hard)
- Cleanbreak: rip -> paste -> fix seams up+downstream in the SAME change. No dual path.
- No fallbacks during the rip; accepted temporary red between rip and paste; green at each commit.
- Audio model = SILENT tail: master audio stays body-length; credits are appended SILENT video; VIDEO ENDS at credit-end (no blank after). Make the mux guard credits-AWARE (declare tail duration), never blind-widen.
- Every node/widget/wiring change lands IN `workflows/otr_scifi_16gb_full.json` same change.
- Suite + Bug Bible after every green chunk; commit AND push to v2.0-alpha same session.

## Execution order (start here)
1. **S2 first, alone, green** — durable singleton stamps: image_engines (dispatcher), CastLock cast+voice, music_engine. GOTCHA: copy from the LOCAL wire ledger into `get_ledger().data` before `save()`, and check `save()`'s return (it returns None on failure — raise). No JSON change. Commit+push.
2. **Parallel with S2:** scaffold `OTR_CreditsRoll` as a new UNWIRED file + spec tests (relocated dossier tests + node-84 bug410 test + duration/loop/backdrop). Inert until wired.
3. **S3 + S1 = ONE ATOMIC commit (guaranteed-RED window):** rip node-12 HUD (dossier + credits-music loop + treatment merge) + rip node-84 tail-fill look contract + wire CreditsRoll (`93->CreditsRoll->85`, preserve link 263, add manifest + declared-tail links) + reproduce the looped-last-clip backdrop in CreditsRoll + JSON validator. Move the broken tests to the new node's spec in the same commit.
4. **S0** font +50% + credits-aware duration budget.
5. **S4** polish (footer :598 only; other HUD polish dies with the rip).

## Verification (every chunk + final)
Suite + Bug Bible + OTR_WorkflowValidator (after any JSON change). Frame-level LIVE smoke on a SHORT and a LONG episode: last frame is a credit frame (not black), video ends with the roll, body audio byte-identical, no mux ValueError on the long roll. S3 = full kibitz + Fable final gate before merge (per CLAUDE.md §9).

## Grounding + panel record
All anchors + the panel reviews (codex r1, agy manual, Fable final gate) are captured in `GO_FORWARD_CREDITS.md` (Grounding + Revision log) and `kibitz-runs/2026-07-03-credits-enrichment/`.
