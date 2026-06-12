# pass01 architecture -- Claude panelist review (written BEFORE reading the panel)

MUST-FIX

1. (role_compat.py, eng_ltx_av design) The single flat `required_inputs`
   tuple cannot express "init_image required for talking roles, optional for
   music_visual". Resolution: ONE adapter, required_inputs =
   ("text_prompt", "audio_ref"); init_image CONSUMED when the role supplies
   it (all three roles' supply sets include init_image, so I2V engages for
   announcer/character via supplied portrait; music_visual may pass a scene
   still). The unavoidable delta is role_compat MUSIC_VISUAL supply +=
   "audio_ref" -- additive one-liner, but ONLY valid if the render driver
   actually attaches the beat's audio slice on music beats
   (VERIFY-AT-BUILD: render_driver.py request assembly; wiring pass).
2. (fallback chain) ltx_av -> humo -> latentsync -> still_kenburns is the
   right SINGLE chain only because role-aware resolver pruning exists
   (AS-1/AS-2): for music_visual, humo (face) and latentsync (lipsync
   overlay) are role-incompatible and must be PRUNED so music degrades
   ltx_av -> still_kenburns; talking roles degrade with sync preserved.
   MUST-GROUND: confirm resolver-prune semantics in _otr_shared (fallback.py
   walks a single-linked chain; resolver.py was kept separate in CW-7). If
   pruning does NOT exist on the fallback path, the chain for music is
   BROKEN mid-episode -- this becomes the pass's biggest finding.
   ltx_av -> ltx_video is REJECTED: ltx_video lacks the character_video role
   and granting it would touch eng_ltx_video.py (out of scope).
   Aspect change on degrade (landscape -> humo pillarbox) is acceptable ONLY
   as a LOUD restamped swap; note it in the ledger reason string.
3. (family) NEW token `audio_conditioned_video`. Reusing `audio_driven_face`
   for a music-reactive non-face role is semantically wrong and risks any
   family-driven branching elsewhere. VERIFY-AT-BUILD: grep ALL consumers of
   `family` (schemas.py validation, role_compat reasoning, ledger stamps,
   image-director 3D-role detection) before finalizing; the additive edits
   are registry.py docstring + schemas.py family list (if enum-validated).
4. (isolation STOP rule) In-process per the ltx_video precedent, with a
   concrete gate: M0 records `pip freeze` BEFORE/AFTER installing whatever
   the A2V graph needs; ANY new package in the cu130 venv beyond the
   already-installed ComfyUI-native/Lightricks stack = STOP, write the
   finding, evaluate the cu128 sidecar (latentsync precedent). No silent
   venv drift.

SHOULD-CONSIDER

5. MotionEngineBase fits without protocol changes: request already carries
   audio_ref (humo proves it); audio conditioning is adapter-internal;
   canonicalize reuses the humo has_audio=False pattern. No new lifecycle
   member; reject any panel proposal to widen the Protocol.
6. Yvann-Nodes lane: CUT from this sprint. New custom-node dependency (b7
   sweep + license + V-12 review), music_visual-only payoff, different
   mechanism than the lane being proven. Revisit ONLY if M0's verdict is
   INERT for music conditioning. Record as appendix, not a milestone.
7. Dims validator goes in a NEW dep-free shared helper (e.g.
   _otr_shared/av_dims.py), not motion_common.py, to keep the no-touch
   promise on shipped files; existing engines adopt it in a later sprint.
8. Engine-count/dropdown enumeration tests will shift by one engine --
   updating TESTS is in scope and expected; updating engine code is not.

OPEN QUESTIONS (for later passes)

9. Does the IA2V graph hard-require the gemma text encoder, and what is its
   resident cost on the 14.5 GB budget (hardware pass)?
10. Per-clip wall time at 1472x832 for 22B distilled vs the ~6 min/clip the
    2B engine already costs (hardware pass; episode time budget).
11. Music beats: does conditioning on a MUSIC slice (not speech) actually
    modulate motion usefully at this scale, or does it collapse to T2V?
    (M0 P1 matrix cell c answers; testing pass owns the gate.)
