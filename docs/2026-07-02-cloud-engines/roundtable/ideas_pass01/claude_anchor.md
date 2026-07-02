# Ideation anchor (Claude, panelist)

1. NAME: The Persistent Playhouse.
   PITCH: the show gets ONE physical home -- a 3D radio-station diorama
   the camera lives inside; every beat is a camera position in the same
   set, forever.
   CHAIN: set-dressing still (stills lane, wide interior) -> Recraft
   background removal per prop + image-to-3D (Tripo/Rodin) for 6-10 key
   props + walls -> assemble once in Blender (already shipped) as a
   saved .blend "stage" -> per-beat camera positions rendered locally
   (cheap CPU/GPU) -> beats composited w/ episode audio-driven lighting
   flicker (audio_motion_profile rms -> lamp intensity keyframes) ->
   kling_lipsync only on face-bearing shots.
   WHY CLOUD-ONLY: minting 6-10 clean textured meshes locally = the
   parked toolchain; cloud mints the set in minutes.
   IDENTITY: the SET is the identity -- spatial continuity no I2V chain
   can fake; props never drift.
   AUDIO-REACTIVE: lighting/needle-meters keyframed from per-beat rms/
   onsets (S-C C1 audio_motion_profile ties in directly).
   RISK: mesh seams/scale mismatch when assembling; one-time cost.
   COST: medium once, then near-zero per episode (mesh cache is global).

2. NAME: Coverage, Like a Real Director.
   PITCH: every character beat gets wide/medium/close coverage cut on
   dialogue rhythm -- filmed-drama editing from ONE mesh.
   CHAIN: minted character mesh (character-variant POC) -> 3 Blender
   camera renders per beat (free after mesh) -> per-line cut list from
   the ledger's line timings -> close-ups get kling_lipsync; wides stay
   mute (universal-slot rule) -> composite over Playhouse backdrop.
   WHY CLOUD-ONLY: the mesh mint; multi-angle I2V without a mesh
   destroys identity.
   IDENTITY: one mesh = perfect cross-angle identity, the strongest
   lock in the plan.
   AUDIO-REACTIVE: cuts land on line boundaries + onsets; lipsync on
   close coverage.
   RISK: cut-list heuristics feeling mechanical; needs taste knobs.
   COST: low per episode (renders local; lipsync only close-ups).

3. NAME: The Ensemble Photograph.
   PITCH: a period cast photo that breathes -- every episode opens on
   the whole cast posed around the station microphone, camera craning
   in as the announcer speaks.
   CHAIN: CastLock roster -> minted mesh per cast member (cached
   globally by portrait hash; amortizes across episodes) -> one Blender
   group scene + crane move -> Magnific upscale pass for the poster
   look -> kling_lipsync on the announcer as the crane lands.
   WHY CLOUD-ONLY: N character meshes.
   IDENTITY: ensemble identity locked forever; recurring characters
   LOOK recurring across episodes for free (global mesh cache).
   AUDIO-REACTIVE: crane timing follows announcer cadence.
   RISK: group compositions reading waxwork-stiff; mitigate w/ subtle
   Seedance pass over the render [verify: img2video on a render].
   COST: high first episode (N meshes), near-zero after (cache).

4. NAME: Marquee Physics.
   PITCH: episode titles as real neon-and-tin 3D signage that swings,
   buzzes, and lights to the theme music.
   CHAIN: Tripo TEXT-to-3D (episode title styled as 1940s marquee) ->
   Blender turntable/swing w/ light keyframes from music onsets ->
   composite over title-card still; music beds already exist per beat.
   WHY CLOUD-ONLY: text-to-3D.
   IDENTITY: consistent title-card design language per season.
   AUDIO-REACTIVE: literally lit by the theme (onset -> flicker).
   RISK: text legibility in generated 3D typography; fall back to
   extruded text in Blender w/ cloud-generated materials only.
   COST: low (one small mesh per episode).

5. NAME: The Wireless Ghost (hybrid render-to-video).
   PITCH: 3D-true geometry with living film grain -- Blender renders
   become Seedance/Kling INIT frames so the model adds smoke, dust,
   tube-glow life while geometry stays locked.
   CHAIN: any Playhouse/coverage render -> frame as init_image +
   episode audio slice -> seedance_2 (identity-preserving, audio-ref)
   -> canonicalize (strip provider audio, mux LAST).
   WHY CLOUD-ONLY: the I2V pass itself on 62GB-class models.
   IDENTITY: geometry from the mesh render constrains drift; best of
   both worlds.
   AUDIO-REACTIVE: seedance audio-ref drives the added motion.
   RISK: I2V wandering off the init frame on long clips; keep clips
   short (existing per-beat clip budget).
   COST: medium (one I2V call per beat that wants it).
