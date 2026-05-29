# OTR Go-Forward Plan v11 -- Model Router first, then Cleanup; Spine-First held as backup

**Date:** 2026-05-28 (late). **Status:** ROADMAP (Jeffrey's "next card" + back-pocket).
Do NOT execute tonight. Capture only. Author: handoff from the live use_exchange round.

## Where we are (end of the v10 round)
- All four v10 builds shipped + proven live (HEAD on v2.0-alpha; commits ddfa4c9, 830f2fe,
  05397bc, f0d283e, a373fb2, 174bf67, 97833b5, 9f29c0a, 839b7a0).
- use_exchange + Build 3 contracts run end-to-end and produce dialogue with GOOD
  line-craft (subtext, refusal, concrete grounding). Graded ~6.5/10.
- The quality gap is the ARC RESOLUTION, not the dialogue mechanics: episodes build
  tension then end without the costly choice actually being made (freeze critic:
  "uneven" / MISSING_COSTLY_CHOICE). That is the real plateau-holder now.
- Current "best story" workflow config (otr_scifi_16gb_full.json): use_exchange=ON,
  Story Room commit=OFF (dormant), freeze halt bypassed (smoke), shadow/multiturn/
  fanout OFF, Stage-3/polish/news ON. All local Mistral-Nemo.

---

## CARD 1 (next, do FIRST): prove the model router -- ARCHITECTURE RISK ONLY
Goal: each LLM slot can be a LOCAL model OR an API model, with the SAME workflow wiring
and the SAME ledger commit path. Do NOT mix this with any story redesign. Just prove the
router works.

Scope:
```
creative slot  = local or API
technical slot = local or API
same workflow wiring
same ledger commit path
no secret keys in JSON   (keys live in env, never in the workflow)
episode meta records backend/model/slot
```

Minimum test matrix (all four must run):
```
1. local creative  + local technical
2. API   creative  + local technical
3. local creative  + API   technical
4. API   creative  + API   technical
```

Pass/fail:
```
- all four combinations run
- no node assumes Hugging Face / local-only model IDs
- API JSON retries fail LOUD, not silently
- local constrained JSON still works
- StoryRoomCommit still lands cleanly
- meta proves which backend generated what
```

Clean fallback: the current local-only path stays the proven fallback while the API
backends are tested. If a slot's API path is unset/unreachable, fall back to local.

> NB / flag for Jeffrey: this revisits the standing global directive
> "100% local, open source, offline-first; no cloud/API/paid services." Adding API
> backends is a deliberate shift -- confirm it's intentional (API as an *option* with
> local as default/fallback), and keep keys in env per the "no secret keys in JSON" rule.

---

## CARD 2 (roadmap; pair with/after Card 1): remove code that isn't used or helping the story
"A good story is paramount." Strip everything that doesn't earn its place:
- Retire the Story Room subsystem (OTR_StoryRoom / Extract / Commit + DirectorBrief feed).
  use_exchange is the keeper dialogue path; the Story Room re-write is dormant + flaky.
  Rewire writer script_json -> Freeze Cascade directly. Keep the craft-floor MODULE
  (_otr_craft_floor; Build 4's adapter uses it) -- only its Story-Room wiring goes.
- Harden or remove the brittle "ask small model to pick from a list, then crash" gates
  (style chooser already fixed -> BUG-LOCAL-295; audit news/critic/constraint choosers
  for the same retry+graceful-fallback treatment).
- Prune dead/unused node paths + modules surfaced by the above.
- Deliberate cleanbreak sprint, one concern per commit, full Bug Bible + core + audio
  regression after each. Not a midnight teardown.

---

## CARD 3 (BACKUP, in the back pocket): spine-first story redesign
Pull this out ONLY if, after Card 1, story quality still stinks. Targets the arc-
resolution gap directly:
```
spine -> scenes -> dialogue -> ledger
```
A top-down story architecture (episode spine first, then scenes, then dialogue, then the
ledger) instead of beat-by-beat outline -> dialogue. This is the structural answer to
"builds tension but never lands the costly choice." Do NOT start this until the model
router (Card 1) is proven -- it is the next card, not today's distraction.

---

## Sequencing
1. Finish the local/API testing round -> prove the model router (Card 1).
2. Cleanup pass (Card 2) -- can run alongside/after Card 1.
3. Re-judge story quality. If still weak -> spine-first redesign (Card 3).
