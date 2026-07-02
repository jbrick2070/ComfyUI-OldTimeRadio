<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

NAME: The Living Evidence Board
PITCH: An endlessly panning, highly-detailed 1940s corkboard where pinned crime scene photos silently mouth the dialogue as the camera sweeps past them.
CHAIN: Recraft (generate individual 1940s suspect polaroids/clues) -> BFL Flux (fill/expand to generate pushpins and strings tying them) -> Cloud template: LTX 2.3 outpaint (stitch clues into a massive 8K spatial corkboard) -> Local (pan and crop viewport across the board image) -> Kling (avatar + lip-sync applied *only* to the cropped viewport containing the active speaker's photo).
WHY CLOUD-ONLY: The VRAM required to outpaint/stitch an 8K continuous spatial background via LTX, plus running Kling's high-tier avatar model, would immediately OOM a 16GB card.
IDENTITY/CONTINUITY ANGLE: The suspects are literal photographs pinned to a board; identity is locked mathematically because the base image never changes, only the local facial animation does.
AUDIO-REACTIVE ANGLE: The static faces in the photos suddenly come alive and lip-sync precisely to the frozen master TTS track exactly when the camera finds them.
RISK: The local pan/crop math desyncing from the Kling facial tracking coordinates, causing the animated mouth to float off the photograph.
COST SKETCH: Medium. Heavy on image generation & outpainting credits upfront, but video credits are saved by only running Kling on short, cropped TTS segments instead of full-frame renders.

NAME: The Shadow-Cast Foley
PITCH: The physical scene is just a smoldering cigarette in a broadcast booth, but the hard-light shadows cast on the acoustic wall act out the entire noir sequence in body-sync with the sound effects.
CHAIN: ElevenLabs (voice isolation to strip TTS, leaving only the music/Foley from master audio) -> Nano Banana 2 (generate 1940s broadcast booth master still) -> Wan I2V (animate the booth still into a looping, smoky background) -> Recraft (generate flat, bold vector silhouettes of the characters) -> Seedance 2.0 (audio-ref + identity preservation to drive the movement and collision of the vector silhouettes using the isolated Foley track) -> Local (composite Seedance output as a distorted multiply-blend shadow onto the Wan I2V background).
WHY CLOUD-ONLY: Running heavy I2V ambient generation alongside a specialized audio-to-motion video model (Seedance 2.0) requires massive concurrent VRAM and compute speed completely unavailable locally.
IDENTITY/CONTINUITY ANGLE: By restricting the acting characters to Recraft vector silhouettes, the show bypasses AI face-morphing entirely—the silhouette remains structurally rigid while Seedance bends its posture.
AUDIO-REACTIVE ANGLE: The characters' physical shadow bodies twitch, recoil, or swagger based entirely on the volume and transients of the isolated Foley/music track.
RISK: Seedance 2.0 interpreting abstract sound effects (like a gunshot or tire screech) as chaotic, unusable full-body spasms instead of dramatic reactions.
COST SKETCH: High. Two heavy video API calls per beat (Wan for the background, Seedance for the shadow cast), plus ElevenLabs isolation preprocessing.

NAME: The Tin-Toy Theatre
PITCH: The characters are literal 1940s tin-toy automatons on a dimly lit stage, their rigged metal jaws clacking to the dialogue.
CHAIN: Ideogram (generate vintage tin-toy character concepts) -> Tripo (single-image to orthographic multiview to mesh) -> Meshy (auto-rig and [SPECULATIVE] apply basic idle animation) -> Local (render static/idle 3D passes in Blender with heavy noir lighting) -> Kling (avatar + lip-sync applied to the rendered 3D tin toy face, driven by the master TTS).
WHY CLOUD-ONLY: Deep multi-view generation, instant meshing, and auto-rigging are heavily optimized for cloud pipelines; doing this locally requires chaining multiple specialized models that take hours per character.
IDENTITY/CONTINUITY ANGLE: Perfect temporal lock. The mesh is generated once per season. The 3D asset is reused locally for every shot, completely eliminating diffusion hallucination between cuts.
AUDIO-REACTIVE ANGLE: Kling forces the rigid, rendered 3D face to articulate its "metal" jaw and painted lips in sync with the human voice actor.
RISK: Kling's avatar model trying to enforce hyper-realistic human skin onto a rendered tin-toy texture during lip-sync, causing an uncanny valley texture morph on the mouth.
COST SKETCH: Low. The 3D generation is a one-time upfront cost per character; the episode-to-episode cost is just Kling lip-sync on pre-rendered local video.

NAME: The VTO Wardrobe Swap
PITCH: A static, tense character confrontation where the camera doesn't move, but the characters' clothing and physical state violently devolves—from clean suits to bloodstained rags—beat-by-beat to match the escalating dialogue.
CHAIN: Recraft (hero portrait generation) -> BFL Flux (VTO / Virtual Try-On and erase/fill prompted by LLM script state: "add blood", "torn lapel", "sweaty") -> Seedance 2.0 (audio-ref to create subtle breathing/head-bobs on the VTO frames) -> Kling (avatar + lip-sync on the Seedance output driven by TTS) -> Local (seamless dissolve transitions between the progressive damage states directly on the audio beat).
WHY CLOUD-ONLY: Chaining high-res VTO with audio-driven motion (Seedance) and high-fidelity lip-sync (Kling) across three different proprietary architectures would crash a 16GB GPU instantly.
IDENTITY/CONTINUITY ANGLE: Relies on BFL Flux's VTO feature mapping new clothing/damage onto the exact underlying character geometry, altering the wardrobe without hallucinating a new face or shifting the background.
AUDIO-REACTIVE ANGLE: The intensity of the master audio drives Seedance's head movement, Kling handles the dialogue articulation, and the local dissolve triggers right on the music's downbeat.
RISK: BFL Flux's VTO subtly altering the ambient lighting of the coat between generations, causing a distracting strobe effect during the local dissolve.
COST SKETCH: Very High. Requires Flux VTO, Seedance, and Kling generation for every single visual beat in the scene.