# Cloud-creative ideation -- judged synthesis (2026-07-02)

One live ideation pass (GPT-5.5 + Gemini 3.1 Pro + DeepSeek v4-pro,
temp 0.9, ~$0.0962) + Claude anchor (5 ideas). 14 ideas total; every
node family in the picks verified against the live 214-node dump
(IdeogramV1-V4, MeshyImageToModel/MultiImage/Rig/Animate/Refine/
Texture, Tripo multiview, LTX 2.3 outpaint template, Kling avatar/
lipsync, Seedance 2.0 -- all present). Convergence signal: the
persistent-set concept was invented independently by 2 of 4 panelists.

## TOP 3 (operator asked for ~3)

### 1. THE TIN-TOY THEATRE (Gemini; judge's #1)
Characters are 1940s tin-toy automatons on a dim stage; their painted
metal faces lip-sync the dialogue. CHAIN: Ideogram (tin-toy character
concept in period trade-catalog style) -> Tripo/Meshy image->multiview
->mesh -> MeshyRigModelNode + MeshyAnimateModelNode (idle sway --
REAL nodes, verified) -> local Blender noir-lit renders (cheap, reused
all season) -> kling_lipsync on the rendered face per line.
WHY IT WINS: it is the operator's character-beat 3D vision WITH the
uncanny valley designed OUT -- a tin toy is ALLOWED to look slightly
wrong; artifacts read as charm, not failure. Season-level mesh cache =
lowest running cost of any character treatment (Kling per line only).
Era-perfect. IDENTITY: mesh = zero drift, one mint per character per
season. RISK (real, from panel): Kling forcing skin texture onto metal
during lip-sync -- S0-adjacent probe: one tin-face lipsync clip before
committing the aesthetic. VERDICT: designate as the CHARACTER-BEAT
STYLE of the Prop Shot POC line.

### 2. THE LIVING EVIDENCE BOARD (Gemini; judge's #2)
An endless 1940s corkboard -- suspect polaroids, red string, pinned
clues; the camera prowls it, and whichever photo the camera finds
MOUTHS ITS OWN LINES. CHAIN: Recraft/Flux polaroid stills per
character (portrait-hash reuse) -> Flux fill/expand + LTX 2.3 outpaint
(cloud template, verified) stitch an 8K board -> LOCAL pan/crop
viewport per beat (free) -> kling_lipsync ONLY on the cropped active
speaker -> paste back at known coords.
WHY IT WINS: cheapest wow in the whole set -- video credits only on
tiny speaker crops; identity is mathematically locked (the base photo
never changes); deeply radio-noir. AUDIO-REACTIVE: photos come alive
exactly on their TTS lines; pans cut on line boundaries. RISK: crop/
paste coordinate drift vs Kling output framing -- we control the crop,
low. Also the perfect LOW-BUDGET EPISODE FORMAT (a whole episode can
live on the board).

### 3. THE PERSISTENT PLAYHOUSE (anchor + GPT convergence, merged)
The show gets one physical home: a 3D radio-station set the camera
lives inside forever. CHAIN: set-dressing stills -> Recraft bg-removal
per prop -> Tripo/Rodin mesh x 6-10 key props + walls -> assembled
ONCE in Blender as the saved stage -> per-beat camera positions
(GPT's addition, adopted: FIXED COORDINATES PER ROLE -- announcer
booth / orchestra pit / stage flats map 1:1 onto announcer_visual /
music_visual / character_video) -> local renders + audio-driven
practicals (rms/onsets -> lamp flicker, needle meters via the S-C C1
audio_motion_profile) -> optional Seedance/Wan "life pass" over
renders (anchor's Wireless Ghost, folded in) -> kling_lipsync on
face-bearing shots.
WHY IT WINS: spatial continuity no I2V chain can fake; the set
amortizes to near-zero per episode via the global cache; the
role->coordinate mapping drops straight onto OTR's 3-role structure.
RISK: mesh scale/seam coherence at assembly -- one-time curation cost.

## HONORABLE MENTIONS (recorded, not designated)
- FOLEY RESURRECTION (GPT): visible foley pit performing the SFX.
  Gorgeous concept; trigger-matching heuristics risky. FOLDED: its
  audio-transient -> practical-effect trick joins the Playhouse.
- SHADOW CAST (GPT + Gemini convergent): characters as hard-light
  silhouettes/reflections. Cheap identity-via-silhouette; kept as an
  episode STYLE option (works inside the Playhouse booth glass).
- VTO WARDROBE DEVOLUTION (Gemini): beat-by-beat costume decay in a
  locked frame (Flux VTO verified in catalog). Special-episode tool;
  very high per-beat cost.
- MARQUEE PHYSICS (anchor): 3D episode-title signage lit by the theme.
  Cheap garnish; build whenever Blender text + cloud materials feel
  worth an afternoon.
- LIVING PORTRAIT / IDENTITY-LOCKED REPERTORY (DeepSeek): both
  essentially validate the already-designated character POC and the
  shipped seedance_2 row defaults -- convergence, not novelty.

## Verify-at-build additions
V1. Kling lipsync on non-photoreal (tin/painted) faces -- texture-morph
probe clip. V2. MeshyAnimateModelNode animation inventory (idle/sway
presets). V3. LTX outpaint template stitch quality at 8K board scale.
V4. Kling output framing vs crop/paste round-trip (Evidence Board).

Order of adoption (judge): Tin-Toy Theatre rides the existing POC line
(after S1+S3); Evidence Board next (cheapest, needs only S1 + kling
row); Playhouse last (biggest one-time asset build, biggest payoff).
