# HuMo improvements -- DRAFT for kibitz (operator 2026-06-30, eyeball of record_of_betrayal)

> Run the /kibitz panel on this AFTER the current engine soak completes (the local agents starve a live
> HuMo render's CPU -- proven: a concurrent kibitz dragged a 1.7B leg to ~40 min). Plan now, harden later.

## OPERATOR FEEDBACK (grounded in the eyeballed episode)
- The HuMo CHARACTER portrait (a real lip-synced face, headphones) looks GOOD ("kk"). Keep it.
- At ~0:45 the clip goes to "ALL MUSH" -- a murky held frame. Root cause (known, GO_FORWARD S-A): HuMo
  per-clip frame ceiling underruns a long beat -> the composite HOLDS the last frame for the remainder ->
  the held static plate reads as murk. `CLIP UNDERRUN ... will HOLD the last frame` in the logs.
- Title sequence / MUSIC bookends: open (music_open) + the pure-music END. Operator unsure whether these
  should be an ANIMATED VINTAGE RADIO (no person face) or a person face. Likely a radio (on-brand,
  faceless host), but TBD.
- Dropdown should SAY which HuMo a row is (1.7B vs 14B + VRAM tier).

## GOALS (hard constraints)
- Improve HuMo portrait QUALITY **without changing the HuMo model class (stay 1.7B) and without raising the
  VRAM budget** (<= 14.5 GB single resident). No new model.
- Fix the clip-underrun mush (delivery-quality floor) so no beat ships a held murky plate.
- Decide + implement the music-bookend visual (animated radio vs face).
- Label HuMo dropdown rows with model + VRAM tier.

## CANDIDATE LEVERS (for the panel to ground + harden -- NOT decided)
1. **Portrait quality (same 1.7B / same VRAM):** better init portrait (the flux/z_image face the HuMo
   I2V conditions on -- resolution, framing, prompt: "35mm, soft key, shallow DoF, period"); HuMo
   inference knobs that don't cost VRAM (cfg, steps, sampler, frame count, the LoRA weight); the
   audio-conditioning strength; face crop/align of the init. Confirm which are exposed
   (`eng_humo.py` / the wrapper) and which actually move quality.
2. **Clip-underrun mush (GO_FORWARD S-A clip-fill):** when HuMo delivers fewer frames than the beat needs,
   LOOP / ping-pong-extend to target (the composite's own recommendation) instead of holding the last
   frame; + a legibility guard (sharpness ratio vs source) that flags/handles a dead plate; record
   attempted/delivered frames in the ledger. HuMo phrase-chunking for long dialogue (vs the 49-frame cap)
   is the upstream root fix.
3. **Music bookends:** route music_open / music_close to an ANIMATED RADIO visual (a still of a vintage
   radio given subtle motion / a dedicated radio motif) rather than a faceless HuMo render -- ties to the
   Route-A "music bookend = radio-themed still" note. Decide: still+motion vs a small bespoke radio
   animator; how the open title card overlays it.
4. **Dropdown labels:** per GO_FORWARD S-E -- every HuMo row states model + VRAM tier (1.7B = LOW-VRAM
   ~3.3 GB fast / 14B = HIGH-VRAM ~15.9 GB, spills on 16 GB). Auto-derived label only (the
   `_engine_id_from_pick` `" ("` round-trip; no custom labels).

## OPERATOR ADDITIONS (2026-06-30, second eyeball)
- **RADIO, NOT A FACE, FOR THE HOST (hard preference).** The announcer + music bookends (open title +
  pure-music end) should be a RADIO -- ideally an ANIMATED TALKING RADIO -- not a person's face. This is
  the "the radio IS the host" OTR aesthetic (matches `_NEVER_HUMO_ROLES` in `_otr_speaker_role.py`, which
  already keeps announcer/music/sfx off HuMo). Character DIALOGUE beats keep HuMo faces; the host does not.
  So an all-HuMo bookend (the "woman in headphones" as the host) is WRONG -- route announcer/music to a
  radio visual. (An "animated talking radio" is a candidate new small engine / motif -- scope it.)
- **HuMo-ISOLATION SMOKE (tooling; later, not now).** HuMo is the slow pole (~40 min/episode; drags under
  any CPU contention). Build an S-F-style fixture that BAKES one good episode's story + master audio +
  ledger + portraits ONCE, then re-renders ONLY the HuMo character beat while swapping optimization knobs
  (init portrait / cfg / steps / sampler / frame-cap / LoRA weight / audio-cond strength) -- fast, low-VRAM,
  apples-to-apples HuMo A/B, no full-episode re-run. This is the TEST HARNESS that makes the portrait-quality
  + clip-fill work above measurable in minutes. Keep 1.7B class + <= 14.5 GB.

## OPEN QUESTIONS FOR THE PANEL
- Which 1.7B-class, VRAM-neutral levers ACTUALLY improve the portrait (grounded in eng_humo + the wrapper)?
- The exact clip-fill mechanism (loop vs ping-pong vs parallax-still) + where it lands (composite vs
  render_driver) + how it stays no-fallback-compliant + LOUD-stamped.
- Music bookend: animated-radio approach + its engine/slot routing + capability implications.
- Label mechanics that don't break engine-id resolution.
- Build order + which pieces are content-only vs workflow-JSON.
