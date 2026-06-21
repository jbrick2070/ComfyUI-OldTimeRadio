# OTR 3D Image Streams -- design to harden (pass00)

## Problem (grounded against the code, 2026-06-21)

OTR's `mesh_stage` engine (the only live 3D path; `nodes/_otr_video_engines/eng_mesh_stage.py`)
reconstructs a 3D mesh from a SINGLE image via Hunyuan3D-2mv, then renders it through Blender
Workbench. The mesh source is the per-character PORTRAIT (`render_driver._portrait_index(ledger)` ->
`ledger['images']['images']`, fed as `init_image`).

That portrait is minted with a CINEMATIC prompt (`render_driver.py` ~:575:
`"close-up cinematic portrait of a person speaking, face centered, subtle [dramatic lighting]"`).
A cinematic portrait -- dramatic side-light, a hood, partial/occluded face, atmospheric background --
is excellent 2D/HuMo fodder but TERRIBLE single-view-3D fodder: Hunyuan3D cannot separate subject
from environment, so it fuses the whole frame into a clay blob (observed: `signal_lost_ancient_rhythm_
20260620_223552` -- a ghostly clay head double-exposed over the scene). On the announcer/music slots
there is no character at all, so the fodder is worse.

The per-beat SCENE still is currently used only as the composite background (`bg_still_path`, the C1
manifest stamp), and the composite GHOSTS the mesh over it (`_silent_procgen_blended_final`) rather
than placing the subject opaquely.

## The design (operator-directed, to harden)

When a beat routes to a 3D engine (gated on the engine's `requires_mesh_portrait` / `character_3d`
or `mesh_stage` capability -- NEVER a hardcoded engine-name check), the image stage forks into TWO
distinct streams, both with prompts QUITE DIFFERENT from the 2D cinematic path:

1. **MESH FODDER (the subject).** A story-driven clean still of the character OR a story object to be
   meshed: single centered subject, plain/neutral seamless background, even diffuse/studio light, full
   unoccluded front-or-3q view, no hood/hands-over-face, no hard shadows, no environment. Goal: a
   reconstruction-friendly image so Hunyuan3D produces a real 3D object, not a blob. Fed to the mesher
   as `init_image`.

2. **BACKGROUND PLATE (the world).** A complementary environment image with NO subject in it, matched
   to the scene's mood/era/palette, designed to sit BEHIND the composited 3D subject. Fed to the
   composite as the background; the subject is placed OPAQUE in front (the mesh already renders
   straight-alpha), not ghosted.

## Open questions for the panel

- **Where does the fork live?** Candidates: `OTR_MetaBriefImagePromptGen` (writes the image prompts),
  `OTR_ImageDirector` (per-role policy + granularity lock that already reads `requires_mesh_portrait`),
  or the image dispatcher (`otr_image_gen_dispatcher.py`). Pick the seam that already knows BOTH the
  per-beat role AND the selected engine's capability, with least plumbing.
- **Subject selection (character vs object).** A 3D beat's subject may be a character OR a story
  object/artifact. How is it chosen from the ledger (speaker char_id? a story-object register?) and
  prompted? Announcer/music slots have no character -- do they get a story-object, a generic subject,
  or fall back to a 2D engine?
- **Ledger `images` taxonomy.** Today rows carry a `kind` + `role` + `engine_id`. Add a `mesh_fodder`
  kind and a `background_plate` kind? Keep cache keys distinct (the fodder must not collide with the
  cinematic portrait of the same character).
- **Mesh cache.** `mesh_cache_key` keys on the CANONICAL portrait content-hash. Switching the source to
  a mesh-fodder still changes the hash -> a clean cache migration (old cinematic-portrait meshes must
  not be reused). Per-character fodder so a stable cast reuses its mesh across beats/episodes.
- **Prompt templates.** Exact mesh-fodder + background-plate prompt scaffolds (and negative prompts) that
  reliably yield isolated subjects + subject-free plates on the validated image engines (flux_gen1,
  z_image_turbo, flux2_klein, lumina_image).
- **Determinism + invariants.** Seed-keyed (C7); additive ledger keys only (schema `l3-2026-05-14`
  frozen); audio byte-identical (this is image-only); content-only where possible; UTF-8/SFW.
- **Aspect.** Mesh fodder wants a near-square/portrait isolated subject; the plate wants the 16:9 scene
  canvas. Reconcile with the engine `render_aspect` contract.

## Invariants (do not let any "fix" break these)
- Ledger schema `l3-2026-05-14` unchanged (additive keys only); audio byte-identical.
- No hardcoded engine-name routing -- gate on the capability field.
- Single resident heavy engine <= 14.5 GB; 100% local/offline; deterministic (seed-keyed);
  LOUD fallbacks; UTF-8 no BOM; SFW.
