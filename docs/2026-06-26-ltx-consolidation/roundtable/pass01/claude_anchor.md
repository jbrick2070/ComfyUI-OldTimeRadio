CLAUDE ANCHOR REVIEW -- R1 (high-level arc / creative coherence)
Grounded against eng_ltx_av.py, render_driver.py, registry.py, 16gb_full.json (read directly).

VERDICT: yes-with-fixes. The arc is coherent -- "one audio-in LTX engine + still
routing driven by a declared capability, not a name patchwork" is a real
simplification that matches the operator ask. But the plan bundles a low-risk
mechanical consolidation with a high-risk refactor of a load-bearing file and does
not sequence them, and it asserts a capability rule SUBSUMES three battle-tested
branches without proving behavior-preservation. Both are arc-level risks.

MUST-FIX BEFORE BUILD:
1. [Part B / F1] SEQUENCE the two changes; do not bundle. The consolidation
   (delete 2 engines, repoint profile/workflow/registry/tests) is mechanical and
   independently shippable. The capability-driven still UNIFICATION is a refactor
   of render_driver.py where each of the three scene-still branches
   (_SCENE_INIT_FAMILIES @479/842, flux_still/flat_still @869, ltx_video @906) is
   tied to a named shipped bug (BUG-LOCAL-403, the 2026-06-20 portrait pillarbox,
   the LK-1a I2V restore). Fix: land Phase 1 = consolidation + a NARROW still fix
   (an ltx_audio_in branch cloning the proven ltx_video I2V branch) so rendering
   is UNBLOCKED green; land Phase 2 = the capability unification as a separate,
   behavior-preserving commit. CONFIRMED the three branches exist and are distinct.
2. [Part B / Invariant 5-6] PROVE preservation: the plan's rule
   "accepts_still AND not _requires_fodder AND family != audio_driven_face -> wide
   scene still" must be shown to reproduce EACH deleted branch's exact behavior,
   including: the portrait-CLEAR so a 832x1216 portrait can't leak into a wide
   frame (CONFIRMED @873-883); station_card is static_image_gen but is
   deliberately NOT scene-still-conditioned (CONFIRMED @863-865) -- the rule must
   NOT pull it in; the OTR_ENABLE_LTX_I2V kill-switch + the _i2v_still_missing
   trace stamp (CONFIRMED @905-929). Add a per-branch behavior-preservation table
   to Part B before coding.
3. [Part A] Default-role transfer is load-bearing, not cosmetic. ltx_av_music
   carries default_roles=(music_visual, announcer_visual) (CONFIRMED @588-589) and
   the 16gb_full.json profile pins both roles to ltx_av_music (CONFIRMED @12-13).
   Deleting it WITHOUT moving the default + repointing the profile silently drops
   the audio-in default for the bookend roles. Plan says to do both -- elevate to
   MUST and verify the default-resolution path actually consults default_roles
   (verify-at-build).

SHOULD-FIX:
1. [Invariant 3 / SYNTH_FALLBACKS] The legacy SYNTH_FALLBACKS entries
   (ltx_av_talk->humo, ltx_av_music->ltx_video, CONFIRMED @63) CONTRADICT the
   engines' fallback_engine=None. Removing them is correct, but state explicitly
   that NO ltx_audio_in SYNTH_FALLBACK entry is added (no-fallbacks) so a future
   reader does not "restore" one.
2. [Part C / canvas clamp] The 1082 canvas clamp keys on the two names; after
   deletion ltx_audio_in must still get the 512x288 AV clamp or it renders at the
   wrong canvas and busts the 14.5 GB ceiling. Make the clamp follow the
   OTR_ENABLE_LTX_AV requires_flag / the engine class, not a name set.

CUT THESE:
1. [F3] Rename ltx_audio_in -> ltx_av. Pure churn across JSON/profile/tests/soak
   for a cosmetic id; safe to cut -- keep ltx_audio_in, document "ltx_video =
   no-audio, ltx_audio_in = audio-in".

[ASSUMPTION] The VISUALIZER is family audio_conditioned_video with
accepts_still=False, so a capability gate on accepts_still excludes it -- stated
in the plan as verify-at-build; treat as UNVERIFIED until the registry is
enumerated.
[ASSUMPTION] The canonical workflow references the two names ONLY at node-87
(grep-supported, not exhaustively verified across every widget).
