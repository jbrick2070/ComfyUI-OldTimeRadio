# ideo_word razzle-dazzle -- 3D / video-engine escalation (high-level, r1)

**STATUS: DRAFT (pre-panel)**
Candidate plan -- ideas window 2026-07-02. Third tier of the ideo_word family:
- `ideo_word` (still cards, CODE-READY/S1-gated)
- `ideo_word_vid` (2D kinetic typography over an Ideogram plate, NEEDS-DECISION)
- THIS: the showpiece tier -- words in MOTION IN DEPTH: 3D typography or a real video engine,
  for hero beats/bookends. "Razzle dazzle."

## Problem / goal

2D kinetic type is flat by construction. The operator wants a tier where the typography itself
lives in a moving 3D space: extruded period letterforms, camera pushes/orbits through the words,
dimensional lighting (neon marquee, brass-and-bakelite, projected-serial titles), or a video
engine that convincingly fakes the same. Reserved for hero moments -- episode open, title
reveal, act breaks -- not per-beat.

## Candidate approaches (panel to rank -- high level)

- **A. Blender 3D typography (LOCAL, deterministic).** Blender 4.5.10 is ALREADY shipped +
  selftested in the repo's 0-E track. Blender text objects: extrude the excerpt/title, period
  fonts, keyframed camera path (push/orbit/dolly), 3-point or neon lighting, render N frames at
  role canvas, existing mux. Fully seed-keyed (camera path + light rig presets), zero cloud
  cost, exact text guaranteed. Headless `blender -b -P script.py` fits the harness model.
  Render time on CPU/GPU to be measured; frame counts are short (3-6s hero clips).
- **B. Ideogram plate -> cloud 3D chain (Meshy).** Meshy image-to-model rows were cataloged in
  the cloud-engines ideation (not yet pinned). Chain: Ideogram mints a dimensional
  title-card still -> Meshy lifts to a mesh -> rig/animate -> render. VERIFY: whether Meshy
  rows are pinnable via the S0 pin flow; text fidelity through mesh-lift is unproven and
  likely poor (meshing letterforms = melted type risk). Panel to confirm reject/park.
- **C. Depth-parallax fake-3D (2.5D, LOCAL).** Depth-map the Ideogram card (DA-V2-S depth model
  is ALREADY sha-verified in the repo from the 0-E track), displace into layers, parallax camera
  drift. Classic motion-graphics 2.5D. Cheaper than A, more dimensional than flat kinetic type;
  text stays exactly as Ideogram rendered it. Rides the same overlay/compositing skills as
  ideo_word_vid.
- **D. Cloud video engine on a 3D-styled still.** Mint an Ideogram card that LOOKS 3D (Ideogram
  renders convincing dimensional/neon type in a STILL) then animate with the same blocked i2v
  rows as everything else -- inherits the full companion-doc blocker list (no promptable i2v).
  Park with A-of-companion.
- **E. Hybrid A+C:** Blender renders ONLY the 3D text pass (transparent background), composited
  over the Ideogram plate or a depth-parallaxed plate. Best-of-both: Ideogram look + true 3D
  type + exact text. More moving parts.

## Working preference (to pressure-test)

A (pure Blender) or E (Blender text over Ideogram plate) as the razzle tier; C (2.5D parallax)
as the mid-tier that might make ideo_word_vid dazzling enough on its own; B likely reject
(melted type); D parked on known blockers.

## Constraints / notes

- Hero-beats only (episode open, title reveal, act breaks) -- a per-beat 3D render is cost/time
  madness and visually exhausting.
- LOCAL Blender/depth work needs the same operator ruling as the ideo_word_vid overlay (local
  lane in an ideogram-cloud family). The 3D toolkit plan (docs/2026-06-09-3d-toolkit/) is
  PARKED for character_3d -- this is NOT that; text objects need no meshes, no rigs, no
  ARKit; do not entangle the two.
- Single resident heavy <= 14.5 GB; Blender render must not co-reside with a heavy engine
  (sequential residency like Wan/HuMo).
- Deterministic/seed-keyed camera+light presets; SFW; fail LOUD; EMPTY defaults, selectable
  only; dark/fail-closed registration; no new widgets expected (V-11, verify).

## Risks / open questions (high level)

- Blender render time per hero clip on the 5080 (measure in a spike); CPU fallback acceptable?
- Period-authentic 3D look is a craft problem (materials/lighting presets need iteration).
- Depth-map quality on typographic stills (DA-V2-S was tuned for scenes, not flat type) -- C
  spike needed.
- Where does this register: video engine consuming dialogue payload (like ideo_word_vid) with
  an internal Blender subprocess? Sidecar isolation (V-12) vs in-process?
- Does the 0-E Blender install carry licensing/env constraints for render use? (It shipped
  with a selftest -- presumed fine, verify.)

## Rough size (complexity, not time)

A: medium -- Blender headless script + presets + engine adapter + residency guard; the craft
iteration is the long tail. C: small-medium -- depth + layer displacement + drift camera in the
existing compositing stack. E: A + compositing seam. Spike first: one hardcoded Blender hero
clip + one 2.5D parallax clip, operator eyeball picks the tier.
