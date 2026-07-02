# OTR cloud-creative ideation brief (2026-07-02)

CONTEXT: OTR is an autonomous 1940s-style radio-drama video show built
on ComfyUI (Windows, 16GB RTX 5080 local). Pipeline: LLM writer ->
per-line TTS voices + theme music (frozen master audio, mux LAST) ->
per-beat visuals (in-character portraits, scene stills, video clips)
-> episode assembly. Beats have roles: announcer_visual, music_visual,
character_video. Audio reactivity is DEFAULT-ON for video (operator
directive): visuals driven by/synced to episode audio wherever
possible. A cloud engine lane is being built (Surface A = ~214 hosted
partner API nodes billed via Comfy credits; zero local VRAM): pinned
rows include Kling avatar + lip-sync (audio-driven face / lipsync
overlay on a base clip), Seedance 2.0 reference-to-video (audio-ref +
identity preservation), Wan I2V, ElevenLabs TTS, Stability/Sonilo
music, Recraft / Flux2 Pro / Nano Banana 2 stills.

CLOUD CATALOG (verified on the live install): VIDEO x91 (Kling x21,
Vidu x13, Wan x12, Luma Ray x8, Seedance x7, Runway, PixVerse, Sora 2,
Veo/GeminiVideoOmni), IMAGE x72 (Recraft x12 incl. remove/replace
background + vectorize, BFL Flux x10 incl. fill/expand/erase/VTO,
Stability, Magnific upscale, Ideogram, Gemini/Nano Banana 2, Grok),
AUDIO x12 (ElevenLabs x7 incl. speech-to-speech + voice isolation,
Stability audio x3 incl. AudioInpaint, Sonilo x2 incl. VideoToMusic),
3D x32 (Tripo x12 text/image-to-3D + multiview, Rodin x8, Meshy x7
incl. rig/animate, Hunyuan3D x5). Plus comfy.org cloud templates:
single-image -> orthographic multiview -> mesh; LTX 2.3 outpaint /
object-removal LoRAs on cloud GPUs.

ALREADY DESIGNATED (do NOT repeat): "The Prop Shot" POC -- hero still
-> image-to-3D mesh -> Blender turntable -> period-backdrop composite
-> Kling lip-sync overlay mouths the episode audio; character-beat
variant with a mint-grade portrait profile.

THE ASK: propose exactly 3-5 NOVEL cloud-native creative workflows for
this show that a 16GB local GPU could never do (or could never do
well). Bias toward: real 3D character/people work, persistent spatial
worlds, audio-driven visuals, 1940s radio-theater aesthetics, and
beat-to-beat identity/continuity locks. Each idea MUST name a concrete
chain of real node families from the catalog above.
