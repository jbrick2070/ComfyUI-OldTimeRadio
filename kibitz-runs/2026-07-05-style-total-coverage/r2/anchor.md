# ANCHOR REVIEW (Cowork Claude) -- style total-coverage r2 (coding plan)

Doc: STAGE3_TOTAL_COVERAGE_SUBPLAN.md v2. Extra grounding: render_driver.py :529-622 read
directly this round.

VERDICT: SHIP WITH FIXES.

MUST-FIX:
1. **Motion-register KEY MAPPING stays Python; packs own VALUES only.** CONFIRMED
   :529-544 (the four keys) + :597-622 (_ltx_motion_role_key with the synthetic-open
   structure check and the OTR_LTX_OPEN_MOTION_KEY env retarget). The plan must state:
   the role->key selector + env override are Python; `motion_registers` supplies the four
   VALUE strings; the BUG-LOCAL-112 240-char budget applies to pack values at LOAD time
   (lint) so a long pack phrase cannot silently drop the brief fragment.
2. **The announcer TALKING motion prompt is PROBE-PROVEN VERBATIM wording** (:558-560:
   a paraphrase scored HALF the articulation, "do not improve it"). This HARDENS the r1
   AG CUT: _IA2V_TALKING_PROMPT_ANNOUNCER and _IA2V_TALKING_CLAUSE_CHARACTER are
   physics-locked constants -- MUST NOT become pack fields, and the plan should carry the
   probe citation so a future round doesn't reopen it. The talking-lane style surface is
   ONLY the init still (announcer_subject_ltx_mouth + portrait_look_talking), and the
   pack's ltx_mouth subject MUST preserve the mouth-prominence contract (load lint: the
   field must mention a mouth; geometry clause stays Python).
3. **sci_fi byte-identity for motion registers must include the composed scene_prompt
   path** (:1656-1657 fallback `or _LTX_MOTION_PROMPT_BY_ROLE["announcer"]`) -- the
   fallback-to-announcer read must go through the SAME pack values (no half-routed dict).

SHOULD-FIX:
4. Chunk A is large (11 fields x 5 packs + 6 composer files). Split A into A1 (portrait
   look trio + announcer subjects + open subjects -- the image lane) and A2 (motion
   registers + plate/object look + emblem -- the video/mesh lane), each independently
   byte-identical + green. Chunks B, C unchanged.
5. The prompt-metadata visual_style stamp (r1 codex S3): specify WHERE -- the shot's
   creative dict alongside text_prompt (write-time surfaces) and the request metadata
   (render-time motion registers); pin one test each.

CONFIRMED-OK: build_request_from_shot is the right substitution point (:955, :1656 sits
inside it); the env-key retarget survives pack routing (keys unchanged); get_open_subject
optional-style approach.
