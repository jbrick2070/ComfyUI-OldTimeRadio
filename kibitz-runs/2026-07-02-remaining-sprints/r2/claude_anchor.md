# Claude anchor review -- r2 (coding plan / implementability)

VERDICT: BUILD-READY WITH MUST-FIXES. The sprint list is correct and correctly
ordered, but three items are underspecified at the coding level.

## MUST-FIX

1. **[CONFIRMED] Sprint A scope is bigger than the plan admits.** render_driver.py
   greps confirm the scaffolding (UNIVERSAL_FLOOR :56, SYNTH_FALLBACKS :63,
   EXPECTED_OOM_TRAIL :117, make_fallback_of :153, consumers :2101/:2527) BUT
   :2527 is inside what looks like a soak/acceptance verifier that ASSERTS the
   oom trail -- E1 must rewrite that verifier's contract (assert NO trail /
   LOUD failure), not just delete constants. eng_character_3d.py references the
   chain; its oom_index behavior (A-S7.5 soak: character_3d OOM->floor restamps)
   was a PROVEN production behavior -- the plan must state explicitly that a
   character_3d OOM now FAILS THE EPISODE LOUD (operator directive) and the soak
   fixtures that expected restamps get rewritten, not baselined.
2. **[CONFIRMED] E2 widget surface: allow_auto_fallback default is `True`**
   (otr_video_director.py:241). Flipping the DEFAULT to False changes INPUT_TYPES
   but the SAVED workflow JSON carries its own positional value -- the JSON must
   be re-audited in the same change (widgets_values positional, BUG-LOCAL-097),
   and the plan should pin: widget STAYS (no mid-list removal), default False,
   runtime IGNORES True with a LOUD deprecation log (a True from a stale JSON
   must not resurrect fallbacks).
3. **[CONFIRMED] Sprint B names the wrong module layout risk correctly but must
   resolve it BEFORE coding:** nodes/_otr_image_engines EXISTS -- the adapters
   go there on the existing image-engine registry pattern (mirror eng_cloud_video
   on the video registry). The plan's "or verify" hedge becomes a concrete first
   task: read the image registry's engine protocol (assert_usable/mint contract
   differs from video's render_clip).
4. **[CONFIRMED-BY-DESIGN] The portrait_mint_3d "stay 2D" clause skirts the
   no-fallback directive.** It is defensible ONLY as a pre-spend GATE (the 3D
   flag never activates), but the plan must require the beat's ledger stamp to
   name the gate decision (mint_3d=REJECTED:<reason>) so it is auditable and
   cannot silently become an engine swap.

## SHOULD-FIX

5. Sprint B conformance test: parametrize over partner_nodes.yaml ROWS (14),
   not adapter modules, so a future row cannot ship unconformed.
6. Sprint D: the AudioEngine protocol is FROZEN -- the plan should name the
   concrete adapter seam (per-line clip contract like bark/kokoro/indextts2)
   and forbid any new widget in the static shell (V-11); the voice dropdown is
   an existing widget whose option list grows from the registry.
7. Sprint E: define where the profile is computed (render_driver per-beat slice
   point already exists -- the slicer logs "per-beat audio: sliced") and stamp
   it into the request dict; engines MUST fail LOUD if the field is absent
   rather than recomputing (drift guard).

## UNVERIFIABLE (verify at build)

- Whether content_oracle.check_manifest requires fallback-trail fields today.
- The exact V3-expansion pin mechanics for seedance_2 dynamic inputs.
