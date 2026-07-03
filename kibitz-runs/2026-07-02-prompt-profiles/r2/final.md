# prompt-profiles codex pass -- judgment (single pass by design)

Operator: "optimize prompts/inputs/seeds/temps later" -- one codex
pass, factual corrections folded, tuning deferred. No further rounds.

CONFIRMED + FOLDED into PROMPT_PROFILES.md v2 (grep-verified against
partner_nodes.yaml): kling_lipsync seed_supported=false; negative_
prompt only on Recraft (inline elsewhere, final prompt in cache key);
720p = base-clip precondition not a lipsync param; flux prompt_
upsampling=false policy; COMBO values = adapter-build verify;
_flash/_tts = one adapter, model selector; profile->schema conformance
test required at S1.

MAJOR FINDINGS (beyond the doc):
1. DYNAMICCOMBO_V3 pin-depth limit: cloud_seedance_2's REAL inputs
   (reference images, audio) nest inside the dynamic model schema --
   pinned required shows only model/seed/watermark. Pinner upgrade
   (V3 expansion) scheduled with S1/S3. Wan pins NO text-prompt input
   (prompt_extend BOOLEAN; audio OPTIONAL -- the V5 audio-conditioning
   probe input exists).
2. eng_cloud_video.py (landed from the video window mid-day) emits
   kwargs (image/prompt/audio) that do NOT exist in the pinned
   schemas -- it will fail closed at invoke. Flagged in
   GO_FORWARD_PLAN for the video team; the conformance test catches
   this class permanently.
3. Codex must-fix #1 (registries lack cloud image/audio rows,
   canonicalize_image/audio NotImplemented) = CORRECT and BY DESIGN:
   those are S1/S2 deliverables; the profiles doc is their spec, not
   their proof.

REJECTED: none. Codex's read was clean.
