# OTR Better Story — Execution Plan v10 (FOUR BUILDS, canonical)

**Date:** 2026-05-28
**Status:** CANONICAL. Supersedes v7/v8/v9. Incorporates Jeffrey's hard cut + 15-point critique.
**Rule:** build one, prove it with a real gate, then build the next. Never have two half-built systems in flight — or you can't tell which change moved (or broke) quality.

---

## Reframed thesis [corrected]

The commit gate is **safety, not artistry.** It rejects broken/flat lines; it cannot author good ones. The actual quality lift comes from the **slot contract** (a real obligation per line) and the **exchange writer** (room to build subtext across lines). The validator just stops bad output from committing. Do not expect the gate to raise the score on its own.

---

## Build 1 — Measurement integrity (must come first; nothing is provable without it)

Per critiques 14 & 15, these are **not** side quests — they decide whether any later number means anything.

- **Resolve the critic key.** Read the critic code. Confirm whether `stage7_shadow_critic.mean_score` is the same basis as the baseline's `whole_episode_critic.mean_score`, or a different pass. Lock the canonical score source + the real baseline number.
- **Resolve the Stage-1 shadow bug.** The plan emits `tension=0`, schema floor is 1, so the shadow plan is discarded every run. Determine if that discarded plan is in the **active render path** or genuinely inert. If active → fix (widen floor to 0 or clamp the generator) — it may be part of the plateau. If inert → document the proof and move on.
- Then run an **N=3 smoke baseline** on the confirmed key.

**Gate:** canonical baseline source confirmed + 3 means recorded; shadow bug either fixed or proven inert with evidence.

## Build 2 — Slot-formatted output + immediate Tier-A integrity gate

Per critique 2, the integrity gate lands **here**, before any exchange writing can make a mess.

- Story Room emits `d###|SPEAKER: ...`, **one slot = exactly one committed text block** (a block may contain internal pauses, but it is one ledger row — critique 8). One row in, one row out, same slot IDs, same speakers, no added/skipped slots.
- **Tier-A validator** (format/integrity only, deterministic, hard-fail): slot count, slot order, speaker match, empty line, per-line word floor. Nothing semantic.

**Gate:** extract + commit still clean (rows committed == draft rows, no fallback); validator produces **zero false-reds** on a known-good draft; 100% deterministic (same input → same verdict).

## Build 3 — `slot_drama_contract` + contract validation

Specified properly this time (critiques 4 & 5).

- **Who generates it:** one **technical-slot** LLM pass per episode (tagged per rule 6), but **derive deterministically wherever possible** to limit garbage-in: `speaker`, `state_before`, `state_after`, `must_turn` from `DramaticState` + beat position; `concrete_detail_required` candidates from the continuity ledger's `active_props` ∪ news `key_terms`; `costly_choice` flag from `DramaticState.costly_choice_beat`. The LLM only writes `line_job` and `hidden_pressure`.
- **Schema (pydantic):** `dialogue_slot_id, speaker, line_job, hidden_pressure, concrete_detail_required[], state_before, state_after, must_turn:bool`.
- **Contract validator (deterministic, runs before the contract is trusted):** schema-valid; every field non-empty; `concrete_detail_required ⊆ (active_props ∪ key_terms)`; `state_before != state_after` for any `must_turn` slot; exactly one slot carries the costly-choice flag. On fail → regenerate once → else fall back to a **deterministic minimal contract** built purely from DramaticState + beat intent.

**Gate:** on N=3, every contract is schema-valid and passes the deterministic sanity checks; no garbage contract reaches the writer.

## Build 4 — `compose_exchange` + repair-by-exchange-group

- Replace isolated `compose_line` with `compose_exchange` over a **beat group (2–3 slots)**, prior committed lines in context. **Hard rule:** the exchange returns exactly one text block per slot ID in the group — internal pauses allowed, slot count fixed (critique 8).
- **De-exposition lives here, in the writer prompt, weighted to avoid object spam** (critique 6): require **one concrete grounding per *exchange*, not per line.** Forbidden-word list is a **soft hygiene nudge inside the prompt, not a gate** (critique 7).
- **Repair by exchange group, not single slot** (critique 11): if any slot in a group fails Tier-A, re-run the whole exchange once with the failure reason; preserve slot IDs + speakers; one repair attempt; still failing → fail loud / legacy fallback. No multi-cycle churn.

**Gate (concrete checklist, no "subjective read" — critique 9):** vs baseline, fewer exposition hits; **each exchange contains ≥1 interruption / refusal / reversal**; **zero slot drift**; VRAM ≤ 14.5 GB; commit still clean.

---

## Explicitly deferred (everything else waits)
- Tier-B **semantic** validator (costly-choice realized, scene-turn, EXPOSITION_DUMP). When revisited it needs **named thresholds** — measured false-positive / false-negative rates against a **labeled example set** — before any check is promoted to hard-fail (critique 10). Not "looks right after 3 runs."
- Standalone banned-language phase (folded into Build 4's prompt).
- Best-of-N tuning, music-interlude policy, longer-episode act counts.

## Definition of done [revised — critiques 12 & 13]
- **N=3 only for phase sanity.** Declaring the plateau crossed requires **N=6–10**, compared on means, because of the ±0.2–0.4 critic jitter.
- Target: mean ≥ 4.0 with **naturalness, emotional_arc, specificity each ≥ 4**, SFW still 5, commit clean, no new churn —
- **AND at least one human listen/read gate** (Jeffrey signs off on listenability). The critic can be gamed if prompts/validator are tuned to its axes; a human ear is the backstop against optimizing the score instead of the story.

## Non-negotiables
Audio is king (revert if commit→audio destabilizes; re-run audio byte-identity each build). 14.5 GB ceiling; `_flush_vram_keep_llm()`. Bug Bible + core + audio regression after every change. No "dummy"; UTF-8 no BOM; SFW. Wire any node/widget/socket change into the workflow JSON (most builds here are internal).
