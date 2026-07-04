# viz_rainbow_cpu + viz_rainbow_gpu -- two-tier rainbow audio-reactive visualizers

> Draft spec for kibitz hardening (2026-06-30). Goal: a creative "rainbow" audio-reactive
> visual to fill the fun slot the retired `abstract` engine vacated, in TWO tiers so it runs
> for EVERYONE (CPU, any OS / AMD / Mac) AND looks great for users with a capable GPU.

## CONTEXT / GROUNDING
- The shipped `visualizer` engine (`nodes/_otr_video_engines/eng_visualizer.py`, engine_id
  `visualizer`, `family="abstract"`) is ffmpeg/CPU-only: it reuses torch-free audio analysis
  (FFT / RMS / onsets) in `nodes/_otr_shared/scope_draw.py` and renders a CRT scope AS the
  per-beat picture. `accepts_still=False`, `required_inputs=("audio_ref",)`,
  `render_aspect="wide"`, `fallback_engine=None` (no fallbacks). It fits announcer/music/
  character by capability (needs `audio_ref`), NOT scene_broll/background.
- The slot-audit sprint (C0-C5, shipped) made any engine usable in any capability-fitting slot;
  `abstract` (procedural floor) + `station_card` were retired. visualizer_rainbow is the
  operator-requested creative replacement, two tiers.
- Operator constraints (2026-06-30): MUST run easily on AMD/Mac/any box (CPU tier = no GPU, no
  shaders, no exotic deps); GPU tier is the "cool audio-reactive shader" version for those who
  can run it; 100% local/offline; SFW; UTF-8 no BOM.

## TIER 1 -- viz_rainbow_cpu (universal, the default rainbow)
- engine_id `viz_rainbow_cpu`; `family="abstract"`; `accepts_still=False`;
  `required_inputs=("audio_ref",)`; `render_aspect="wide"`; `commercial_clean=True`;
  `fallback_engine=None`; `declared_isolation` in-process; cold-import clean (lazy ffmpeg/numpy
  inside render_clip). default_roles=() (selectable, never auto-default).
- RENDER APPROACH (pure CPU, ffmpeg + numpy only -- ffmpeg is already a hard dep):
  - audio-reactive BASE from ffmpeg's built-in CPU visualizers: `showcqt` (constant-Q, colorful)
    or `showspectrum mode=combined:color=rainbow` / `showwaves` / `avectorscope`, sized to the
    beat canvas + fps, fed the per-beat audio slice.
  - RAINBOW + generative layer via CPU filters: `hue` rotation over time, `pseudocolor` /
    `lutrgb` palettes, `gradients`, a `tblend`/feedback loop for plasma-ish trails.
  - OPTIONAL numpy plasma / flow-field generator (the kind scope_draw already does in numpy),
    its parameters (palette shift / flow speed / bloom) driven by the SHARED audio analysis
    (FFT bass -> bloom, onsets -> palette jumps, RMS -> speed). Composited under/over the ffmpeg
    base.
  - Output = the platform silent CanonicalClip contract (h264/yuv420p/bt709/fps, has_audio=False;
    audio only via OTR_MasterAudioMux). Same shape eng_visualizer emits.
- OPTIONAL `no_audio` procedural mode: a time-seeded generative rainbow that needs NO audio and
  NO image -> the accessible no-image floor for scene_broll/background (what abstract used to be).
  If wired, this RELAXES required_inputs for those slots -- design TBD (capability implications).

## TIER 2 -- viz_rainbow_gpu (the cool shader version, opt-in)
- engine_id `viz_rainbow_gpu`; same family/roles/capability as the CPU tier.
- RENDER APPROACH: real fragment shaders (plasma / flow-fields / bloom / feedback) driven by the
  SAME audio analysis as uniforms. Candidate stacks (kibitz to choose): moderngl offscreen GLSL
  (needs a headless GL context -- EGL on Linux/NVIDIA, may be fragile on AMD/Mac); a torch-based
  compute shader (runs wherever torch+CUDA/ROCm/MPS does); or ffmpeg GL filters.
- MUST be GPU-capability-gated: detect a usable GPU/GL context at assert_usable; if absent, FAIL
  CLOSED (no silent CPU swap -- the registry has no fallbacks) with a clear "select viz_rainbow_cpu"
  message. Default-OFF (opt-in) until validated on the 5080.
- Cross-vendor goal: "most GPUs" -> prefer a stack that works on NVIDIA + AMD + Apple, not CUDA-only.

## SHARED / WIRING (both tiers)
- Reuse `scope_draw` audio analysis (FFT / RMS / onset) -- ONE analysis source for both tiers; do
  NOT duplicate. Extract a shared helper if needed.
- registry: `@register` each; CAPABILITIES rows (cpu tier: vram_class cpu / cpu_ok True; gpu tier:
  light/medium + cpu_ok False + required GL/torch). node-87 OTR_VideoDirector dropdown is built
  from all_engine_names() (no JSON option edit needed); add aspect-derived labels
  ("Rainbow visualizer (CPU, any hardware)" / "Rainbow visualizer (GPU, shader)").
- Capability (role_compat): both `required_inputs=("audio_ref",)` -> fit announcer/music/character,
  NOT scene_broll/background (unless the no_audio mode changes that for the CPU tier).
- Tests: registry+CAPABILITIES consistency; capability matrix (fit the 3 audio roles); accepts_still
  False (mint no still); cold-import clean; an offline render-contract test (mock ffmpeg) for the CPU
  tier; GPU tier gated/skipped without a GPU. Suite + Bug Bible + B7 green; push per chunk.
- Both default-OFF / selectable until operator look-QA promotes them.

## OPEN QUESTIONS FOR THE PANEL
1. CPU tier: ffmpeg-filter-only vs ffmpeg+numpy-plasma -- which gives the best rainbow look at the
   lowest complexity/dep cost? Exact filtergraph?
2. GPU tier stack: moderngl/EGL vs torch-compute vs ffmpeg-GL -- which is genuinely cross-vendor
   (AMD/Mac/NVIDIA) AND headless-safe AND low-dep? Or is the GPU tier not worth it vs a richer CPU tier?
3. The no_audio procedural mode: should the CPU tier carry it (to refill the scene/background
   no-image floor abstract left), and what are the capability/required_inputs implications?
4. Build order + how to keep it cold-import clean + no-fallback-compliant.
