# CREATIVE_FORMATS_PLAN -- r1 synthesis (anchor + codex + claude; agy benched)

Judgment: codex 6 MUST-FIX (all CONFIRMED against yaml/plan/director),
claude 4 MUST-FIX (eng_character_3d collision CONFIRMED via repo E1
open item; Meshy-not-pinned CONFIRMED -- dump != pin; one MISREAD:
"Flux fill/expand IS pinned" -- only Flux2ProImageNode is pinned, fill/
expand are separate unpinned classes), anchor 5 MUST-FIX (all stand).
Convergent x3: format switch undecided; still->clip missing; rig cut.

## r1 RESOLUTIONS (fold into the plan; r2 reviews the result)

1. TOKENS vs PINNED NAMES (codex#2, claude#1, anchor#2): descriptor
   tokens (base_clip_ref/audio_ref) are role-compat CAPABILITY tokens;
   every cloud adapter maps tokens -> pinned field names from
   partner_nodes.yaml at invoke time (kling_lipsync: video/audio/
   voice_language). One mapping contract, stated once. F1-c gains the
   explicit LOCAL still->silent-base-clip step (ffmpeg loop at role
   fps) before any lipsync call.
2. NO NEW UNPINNED DEPENDENCIES IN MVP (codex#1/#3, claude#2/S-5):
   - Rig/animate (Meshy): CUT from F2 MVP (convergent w/ pass04
     Appendix A "future-lane"). F2 MVP = STATIC mesh + Blender
     turntable/dolly + kling mouth -- same pattern as the Prop Shot.
   - Ideogram: CUT; tin_toy_v1 concept sheets run on the PINNED stills
     rows (recraft / nano_banana_2). Ideogram = optional future pin.
   - LTX outpaint: CUT (it rode the cancelled Surface B). Board v1 =
     PURE LOCAL COMPOSITING: cork backdrop still + polaroid stills
     (all pinned gen rows) pasted locally; 4K canvas, not 8K
     (claude CUT-2, codex CUT-3: crops deliver at 1472x832 -- 4K is
     ample headroom). FluxProFillNode = optional future pin if seams
     ever demand generative blending.
   Any future row (rig, fill, Ideogram) goes through the FULL S0
   pinning pipeline (script + drift test + pricing + ToS) -- being in
   the 214-node dump is NOT pinned (claude#2's distinction, adopted
   verbatim).
3. FORMAT = ENGINE (anchor#5 + codex#5 + claude#4, decided NOW):
   formats register as VIDEO ENGINE ROWS (`fmt_evidence_board`,
   `fmt_tin_toy`) in the existing registry -- local, zero-cloud-cost
   rows whose render_clip implements the format (camera desk lives in
   `nodes/_otr_video_engines/eng_evidence_board.py`; toy plates in
   `eng_tin_toy.py`). ShotLock/ledger/reactivity policy see every shot
   (no bypass; board pans classify as format-local shots, stamped).
   UX sugar: ONE `visual_format` widget APPENDED at the END of
   OTR_VideoDirector's optional widgets (BUG-LOCAL-097 rule: append-
   only, same change as workflow JSON) with values
   standard|evidence_board|tin_toy; selecting a format flips all three
   per-role defaults to the format row; headless override
   `OTR_VISUAL_FORMAT`. Explicit picks per role still win
   (universal-slot rule).
4. LOCAL 3D SCAFFOLD (claude#3): F2 SUPERSEDES the parked
   eng_character_3d lane for character presentation. It does not touch
   that code; the dark scaffold's removal continues under open item E1
   (no-fallback migration). NO coexistence of two talking-3D paths:
   character_3d stays parked/dark; F2 is the live path when built.
5. V1 FALLBACK TRUTH (claude S-3): if Kling lipsync mangles CG-rendered
   faces generally (not just tin), BOTH F2 and the Prop Shot mouth path
   die together. Honest fallback: the LOCAL audio-driven mouth lane
   (HuMo audio-driven face / ltx_audio_in RECIPE_IA2V) applied to toy
   renders -- or F2 parks. V1 probe therefore tests BOTH a tin face
   AND a photoreal CG render, plus mouth READABILITY at real shot
   sizes (codex S-3), before any F2 build.
6. BOARD CACHE LAYERS + KEY (codex#6, anchor#4): (a) cast-polaroid
   layer keyed by portrait-hash set; (b) episode dressing layer keyed
   by episode_evidence_hash (clue content). Same-cast episodes reuse
   (a), never (b).
7. ASSET PLACEMENT DIRECTIVE stands as written (operator): everything
   materializes under otr\episodes\<ep>\ (formats get subdirs); only
   the final mp4 goes to otr\obs\. Storage tradeoff acknowledged
   (claude S-4): GB-scale mesh duplication across episodes is an
   ACCEPTED operator choice (visibility > disk); cache remains
   billing-dedup only.
8. IDENTITY + ACCEPTANCE (codex S-1, both OPT): post-paste face-
   similarity check against the portrait hash chain (not just +/-2px
   geometry); each format ships a GOLDEN 30-SECOND SAMPLE acceptance
   before episode wiring; F1 acceptance reuses the 30w smoke script in
   board format.
9. COSTS (codex S-2, claude OPT-1): section 6 numbers marked
   [ASSUMPTION] until S0 pricing stamps the rows; estimate report
   gains format rows when formats land.
10. STANDING [ASSUMPTIONS] recorded: S0 invoke bridge ships first (or
    neither format builds); catalog keeps the nodes we pin; V4
    spatial-stability probe gates the crop-paste architecture.

VERDICT after synthesis: the plan advances to r2 with these folded.
