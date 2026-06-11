<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. Critical gaps in per-scene prompt mapping, composite fill mechanics, and a contradictory acceptance requirement make the plan unbuildable as written.

MUST-FIX BEFORE BUILD:
1. [R5-1 / composite] The LTX frame cap alone does not produce a filled opening. The document says “render the cap and FILL the remainder in the composite by hold-last-frame with the existing slow-zoom (Ken Burns) treatment.” There is no description of how the composite (OTR_SilentComposite? elsewhere?) receives the cap, detects the shortfall, and generates the hold‑last‑frame segment. Without this specification, the composite may error, produce black, or drop the opening. Fix: either (a) add a concrete composite‑side mechanism (e.g., a new shot type or manifest extension) that accepts a short clip and a fill duration, or (b) adopt the alternative of splitting the beat into multiple capped LTX renders to cover the gap without composite changes.
2. [R5-2 / D2] No beat‑to‑scene mapping exists. The plan says “restore the legacy per‑scene layer: LTX scene core becomes scene_visual_prompt (from meta.visual_plan, per‑beat/scene) else get_story_brief_ltx. Map beats‑>scenes via the visual_plan's scene index (fall back to scene 0 / nearest).” The grounding excerpts show that meta.visual_plan scenes have scene_id and visual_prompt, but there is no field linking a beat/line to a scene. Without a mapping, the per‑beat prompts will remain identical across announcer/music beats, so the D2 problem persists. Fix: define a deterministic assignment (e.g., use scene_id from the writer’s per‑line metadata if present, else partition beats among scenes by index) or provide a per‑beat variation not dependent on visual_plan scenes.
3. [Acceptance / R5‑5] The acceptance gate demands “no self‑vocative/mis‑attributed line ships” for the re‑render. The audio is frozen, and changing the `char_id` of b004 would mismatch the pre‑recorded voice. The plan itself states “R5‑5 runs BEFORE the freeze in future episodes, never rewrites a frozen ledger.” Therefore, for this render, the mis‑attributed line cannot be corrected without violating the frozen‑audio invariant. Fix: either (a) remove the “no self‑vocative/mis‑attributed line” requirement from acceptance, or (b) acknowledge that this line will ship with the current voice and log a LOUD warning (the plan already proposes R5‑6 logging), and accept that outcome.

SHOULD-FIX:
4. [R5‑1] Consider splitting the long synthetic beat into multiple capped LTX clips (e.g., 121f + 117f) instead of a single clip plus a Ken‑Burns fill, to preserve motion variety and avoid a static 4‑second hold that may look unnatural.
5. [D2 / prompt variety] For beats where per‑scene visual_prompt is unavailable or mapping is impossible, generate per‑beat variation by appending a beat‑index‑specific clause (e.g., “opening establishing shot,” “mid‑scene continuation”) to the brief‑composed core, so prompts still differ.
6. [D3 / b002] The M4 person‑anchor guard (R5‑4) should also reject non‑person descriptions (like “console”) even if a person token is absent, and fall back to a portrait‑anchored template, to prevent HuMo from animating objects.

OPTIONAL / NICE-TO-HAVE:
- R5‑3 (brightening the OPEN clause) is a trivial string change and safe.
- R5‑6 (LOUD warning for missing portrait index and announcer char_id normalisation) is low‑risk and adds safety.

CUT THESE (over‑engineering):
7. [R5‑5 / writer‑side] The writer‑side self‑vocative scrub for future episodes is appropriate but cannot affect this render; defer to a future sprint. For today’s re‑render, rely only on a LOUD log of the misattribution.

[ASSUMPTION]
- The OTR_SilentComposite node or the video assembly stage is capable of holding last frame and applying Ken Burns; the grounding excerpts do not show its API. If that capability does not exist, point 1 becomes a hard blocker that requires building that feature.
- The writer stamps a `scene_id` on each line or the visual_plan includes an implicit order that matches beat order; the plan assumes this but the provided code does not confirm it. If absent, point 2 must be resolved by an alternative variation strategy.