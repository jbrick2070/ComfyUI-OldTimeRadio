# R1 JUDGMENT -- creative arc (Claude, grounded)

Panel: GPT-5.5-20260423, Gemini-3.1-pro-preview, DeepSeek-v4-pro-20260423.
Spend this pass ~$0.0927. Convergence: VERY HIGH on the three roots.

## CONVERGED ROOTS (anchor + all 3 panel) -- ACCEPTED, all CONFIRMED vs code
1. Outro system prompt contradicts the news coda (`_ANNOUNCER_OUTRO_SYSTEM`
   :2536-2555 forbids news-summary; :2854 forces fictional outcome). -> pass01 §2
   Job3: rewrite outro voice under flag + gate off the fiction branch + deterministic
   lead-in. (pass00 had wrongly called the coda "largely wired/just framing".)
2. Open no-spoiler cannot be prompt-only; sever `script_brief`, input-starvation
   primary (`compose_announcer_intro` :2709 reads only script_brief; fallback :2614
   echoes it). -> pass01 §2 Job1.
3. KILL 2 is the single-prior trap without teeth. -> pass01 §1.

## JUDGE OVERRIDE OF THE PANEL (grounded)
- DeepSeek MUST-FIX#3 + GPT SHOULD#3 proposed a per-LINE "style-marker present"
  gate. REJECTED as ungrounded-harmful: the catalog docstring (`:5-7`) makes
  `sound_world` AUDIO vocabulary; gating dialogue lines for those markers would
  PUSH the stage-direction leak the repo already fixed (L3/L4 strip,
  docs/2026-06-22-stage-direction-leak). Instead: inject story_engine at the
  OUTLINE, sound_world at MOOD/RENDER (its documented home), keep lines to compact
  register + the gateable conflict_object. KILL 2 scoped as a structural STEER +
  existing conflict-object teeth, measured at re-soak. (pass01 §1.)
- Gemini "DELETE the resolved/ending branch" -> SOFTENED to GATE it OFF under the
  flag (GPT CUT#4 framing) so byte-identical-off holds. (pass01 §2 Job3 + §5.)

## ACCEPTED ADDITIONS (panel caught, anchor missed or under-weighted)
- Outro/climax COUPLING: pass `climax_character_line` not `final_character_line`
  (GPT#5, Gemini#3, DeepSeek#6). GROUNDED LOW-COST: `_climax_beat_id` already
  exists (:3266) and climax==last today -> byte-identical now. (pass01 §2 Job2.)
- Do NOT delete `consequence` enrichment -- KILL 3 makes it reachable (GPT#6).
  pass04 said "cut, unreachable"; corrected to DEFER. (pass01 §3.)
- Explicit byte-identical flag boundary + OFF-flag golden tests (GPT#8). (§5.)
- Pipeline/data-handoff spec; verify cast-lock precedes OutlineRequest (GPT#4).
  (§8.)
- Coda-specific length budget (GPT#4/Gemini), feature-specific telemetry (GPT#5).
- News-coda lead-in: deterministic PREFIX, fixed/small-closed-set (all 3). Honor
  operator's "The real story:" but flag the OTR-fiction break; recommend in-voice
  variants. (§2 Job3 + ask #1.)

## DEFERRED / DOWNGRADED
- "Premise-specific conflict objects beyond the domain pool" (GPT CUT#2, DeepSeek
  optional) -> deferred; the seeded domain slot + KILL-1 grounding suffice. (§1.)
- DeepSeek "merge KILL 3 into the announcer chunk / drop the re-soak" -> PARTIAL:
  pull the outro climax-decoupling forward (announcer needs it, byte-identical),
  keep the KILL-3 validator relaxation DEFERRED per operator; KEEP the re-soak
  (operator standing discipline after a structural change). (§4, §7.)
- DeepSeek SHOULD#4 (rewrite the hardcoded open intent :1591 for truthfulness) ->
  DOWNGRADED: it does not drive content and changing the constant could shift the
  ledger when the flag is OFF -> only if flag-gated; lean skip for byte-identity.

## VERIFY-AT-BUILD (carried)
cast-lock before OutlineRequest; `select_style` other-callers before deletion;
outline can carry `opening_status_quo` (new field) + structured era/setting/cast;
`news_close_brief` never empty for any path + distinct from `ending_change`;
off-flag golden outputs.

## CONVERGENCE CALL
R1 converged on the creative arc. Material design changes remain (the coda/outro
rewrite, open input-starvation, KILL-2 layering) -> proceed to R2 (coding plan /
implementability) to harden the mechanism, NOT exit early.
