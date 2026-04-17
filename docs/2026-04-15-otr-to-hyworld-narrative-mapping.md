# OTR -> HyWorld Narrative Mapping (Design Ideas, Pre-Build)

**Date:** 2026-04-15
**Branch:** `v2.0-alpha`
**Status:** Ideas doc for visualization + review. Not an implementation spec.
**Owner:** Jeffrey A. Brick
**Companions:**
- `docs/2026-04-15-hyworld-poc-design.md` (architecture / install / nodes)
- `docs/2026-04-15-hyworld-integration-plan-review.md` (keep/discard triage)
- `docs/OTR_PIPELINE_EXPLAINER.md` (layman's pipeline)

---

## 0. Why this doc exists

The PoC doc says **how** HyWorld plugs in (sidecar, JSON contract, three new nodes).
This doc answers the creative half: **what OTR already knows about a story, and what you could point HyWorld at to make meaningful art out of it.**

OTR's output is a radio drama: dialogue-first, SFX-heavy, environments painted in sound. HyWorld 2.0 is a 3D reconstruction / world-generation stack. The interesting question isn't "can we draw a picture of the script" — it's "what does it look like when a radio play dreams."

Jeffrey plans to visualize the mapping tables in this doc and pass them around for feedback before we lock a schema.

---

## 1. What OTR already parses today (ground truth)

From `nodes/story_orchestrator.py::_parse_script` and the downstream `scene_sequencer`, every generated episode produces a structured `script_lines` array. These are the "Canonical Audio Tokens" — the only shapes of information OTR guarantees are present in every episode.

| Token type      | Fields                                                  | Example                                              |
|-----------------|---------------------------------------------------------|------------------------------------------------------|
| `title`         | `value`                                                 | `"The Last Signal from Vault 7"`                     |
| `scene_break`   | `scene` (label / number)                                | `=== SCENE 2 ===` -> `scene = "2"`                   |
| `environment`   | `description` (3-4 descriptors, prose)                  | `[ENV: fluorescent hum, distant traffic, rain]`      |
| `sfx`           | `description` (single event, prose)                     | `[SFX: heavy wrench strike on metal pipe]`           |
| `pause`         | `kind=beat`, `duration_ms=200`                          | `(beat)`                                             |
| `dialogue`      | `character_name`, `voice_traits`, `line`                | `[VOICE: COMMANDER, male, 50s, weary] "Hold the line."` |
| `direction`     | raw non-tag prose kept for flavor                       | stage directions the LLM emitted between tags        |
| `music`         | `description` (extracted from `[MUSIC: ...]`)           | `[MUSIC: Opening theme]`                             |

Two derived structures are also available per episode:
- **Character roster** — unique `character_name` set, each with accumulated `voice_traits` strings (gender, age, tone). This is a natural source for character cards.
- **Scene table** — every `scene_break` splits the timeline. Environments and SFX between two breaks belong to that scene; dialogue clusters inside scenes.

And two timing structures the audio pipeline builds:
- **Audio offsets** — each token gets an absolute `audio_offset_s` after SceneSequencer renders. This is how HyWorld shots will sync to the WAV.
- **Duration per scene** — derived by summing dialogue + SFX + pauses inside a scene.

**This is the raw material HyWorld can eat.** Nothing else is guaranteed to exist. Anything richer (shot lists, camera motion, mood beats) has to be *inferred from these tokens*, either by us in deterministic code or by a small LLM pass that reads the script and emits a `shots[]` array. Both options are discussed below.

---

## 2. What HyWorld 2.0 actually wants as input (current, 2026-04-15)

Only one model is shipped right now — everything else on the HY-World 2.0 page is "Coming Soon." We must design for today's reality and mark forward-looking entries as speculative.

### 2.1 Shipped today — WorldMirror 2.0

Multi-view / video -> 3D reconstruction.

**Wants:**
- 5-20 images of a scene from different viewpoints (or a short video clip).
- Resolution roughly 512x512 and up, consistent enough to correspond.

**Produces:**
- 3D geometry (point cloud or 3DGS), depth maps, camera poses.

**So the creative question is: how do we get 5-20 coherent images per scene when the source material is a radio play?** Several candidate answers are in Section 3.

### 2.2 Coming Soon — HY-Pano-2.0

Text / image -> 360 panorama.

**Speculated input:** a prose environment description (perfectly what `[ENV:]` tokens already are) plus optional reference image.
**Produces:** equirectangular panorama of a place.

When this ships, it's a near-direct match to OTR's `environment` tokens — probably the cleanest mapping in the entire stack.

### 2.3 Coming Soon — WorldStereo 2.0

Panorama -> navigable 3DGS.

**Speculated input:** a pano from HY-Pano-2.0.
**Produces:** walkable Gaussian splat of the panoramic scene.

This is the model that could turn an OTR scene into a space the viewer can move through while the dialogue plays. High creative payoff, but blocked on Pano shipping.

### 2.4 Coming Soon — WorldNav

Trajectory planning inside a 3DGS scene.

**Speculated input:** a scene + a high-level intent ("dolly forward", "look up at the light").
**Produces:** a camera path.

Matches one-to-one with a `camera` field on the Director `shots[]` array we already reserved in the PoC doc.

---

## 3. Three candidate mappings, ordered from "actually buildable today" to "aspirational"

### 3.1 Mapping A — Single anchor image per scene, WorldMirror does the rest (PoC target)

**Idea:** for each scene, generate one "anchor" image from the `[ENV:]` description using any text-to-image model available locally (existing OTR toolchain already has ComfyUI + SDXL / Flux). Then synthesize 8-12 virtual viewpoints around it by small camera perturbations and feed them into WorldMirror 2.0 to lift to 3D.

**Why this first:** everything needed ships today. No dependency on Pano / Stereo / Nav.

**OTR token -> HyWorld input:**

| OTR token                      | Becomes                                                   |
|--------------------------------|-----------------------------------------------------------|
| `environment.description`      | Text prompt for the anchor image                          |
| First dominant `sfx` in scene  | Adjective modifier on the anchor prompt ("rain-soaked")   |
| `scene.duration_s` (derived)   | Clip length budget (capped at 12 s per C4)                |
| `character_name` + roster      | **Not used in the image** (C6: IP-Adapter environments only, never characters) |
| `scene.title` / `scene_break`  | Filename / anchor key                                     |

**Creative framing:** scenes become "places that were heard, now seen." The listener hears dialogue in a room they have never visited but can now walk through. Not a shot of the Commander — a shot of the room the Commander was in.

**Honest limits:** WorldMirror lifts geometry from images; if the anchor image is wrong, the 3D is wrong. Hallucinated anchor + WorldMirror = a coherent but wrong room. Acceptable for PoC. Sell the weirdness — it is on-brand for SIGNAL LOST.

### 3.2 Mapping B — Pano-first, when Pano ships

**Idea:** feed `[ENV:]` descriptions directly into HY-Pano-2.0 -> panorama -> WorldStereo -> navigable splat. Dialog timestamps drive camera paths via WorldNav.

**Why this is the eventual goal:** the `[ENV:]` token was already designed to be a 3-4 descriptor prose chunk ("fluorescent hum, distant traffic, rain on concrete"). That is exactly the shape a text-to-pano model wants. Zero glue code between OTR parse output and HyWorld input.

**OTR token -> HyWorld input:**

| OTR token                      | Becomes                                                   |
|--------------------------------|-----------------------------------------------------------|
| `environment.description`      | HY-Pano-2.0 text prompt (verbatim, no rewriting)          |
| Secondary `environment` tokens in same scene | Pano style anchor (shifting mood mid-scene)    |
| `scene_break`                  | Pano boundary                                             |
| `sfx` cluster density          | Implies visual busyness - feeds pano "detail" knob if exposed |
| `dialogue` clusters            | WorldNav keyframes ("dolly to CHARACTER position")        |
| `character_name` positions     | Named anchors in the 3DGS for nav targets (positions only, not faces — C6) |

**Creative framing:** every OTR episode becomes a walkable audio drama. Listeners move through the pano as dialogue plays. Think the old-radio-drama equivalent of a VR space that exists only because a voice described it.

**Blocker:** Pano + Stereo + Nav are "Coming Soon." Revisit when they ship. Do not block PoC on this.

### 3.3 Mapping C — Emotional geometry (speculative, aesthetic)

**Idea:** stop treating the 3D output as a realistic depiction of the script's setting, and start treating it as a **visual analogue of the emotional state.** Character moods, scene tension, and SFX density drive geometry parameters; the dialogue audio plays underneath.

Example: a tense argument between two characters produces a cramped, high-contrast, low-ceilinged geometry; a quiet reflective monologue produces open spaces with soft light. The scene doesn't depict a real place; it depicts the *feeling* of the place.

**OTR token -> HyWorld input:**

| OTR token                                                | Maps to                                    |
|----------------------------------------------------------|--------------------------------------------|
| `voice_traits` keywords (angry, weary, calm, frantic)    | Anchor prompt adjectives                   |
| Dialogue density (lines/sec)                             | Camera cut rate                            |
| SFX density                                              | Scene complexity / particle density        |
| Unique speaker count in scene                            | Depth of field / scene scale               |
| Ratio of `(beat)` tokens to dialogue                     | Tempo of virtual camera                    |
| Presence of `[MUSIC:]` tokens                            | Color palette shift                        |
| Episode title                                            | Global style anchor (consistent across scenes) |

**Creative framing:** this is the Codex-Olympia-style move. Radio drama is already non-visual; forcing literal visuals is a step down. But visualizing the *emotional architecture* of a scene is a step sideways into new territory. LA28 anthology-friendly. Gallery-friendly.

**Honest risk:** can slide into generic "abstract visuals" territory if the mappings are not specific enough. Counter: rigid, deterministic rules per token type. No LLM second-guessing on mood; pick the adjectives from the trait string verbatim.

---

## 4. Concrete per-token mapping ideas (detailed tables)

These are starter values — they are the "first draft" Jeffrey will critique. Do not treat them as canonical.

### 4.1 `environment.description` (cleanest case)

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Anchor image prompt (Mapping A)        | Verbatim, prepended with "wide establishing shot of " |
| HY-Pano-2.0 prompt (Mapping B)         | Verbatim, no modifier                                 |
| Style anchor token for episode         | Hash of first `environment` in Act 1, reused throughout |
| SFX enrichment                         | Append dominant `sfx` adjective when pano detail is low |

**Why `[ENV:]` is the anchor:** Jeffrey's own prompt rules already require `[ENV:]` to be 3-4 concrete descriptors, not abstractions. That is the exact shape a generative vision model wants. Reuse what's already there; do not re-author.

### 4.2 `dialogue.character_name` + `voice_traits`

Characters do **not** become faces (C6). Instead:

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Named position in 3DGS (Mapping B)     | Place a named null / marker at a generated position; WorldNav can dolly to it |
| Mood tag for scene camera              | Map `voice_traits` first adjective -> camera style (see 4.5) |
| Character card sidecar file            | `chars.json` with `{name, traits}` for future use — kept but not yet consumed |

The constraint "no character faces" is not a limitation, it is the interesting aesthetic: audience hears the Commander, sees their empty chair. The chair is what we render.

### 4.3 `sfx.description`

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Visual accent on timeline              | Flash / zoom / particle burst at `audio_offset_s`     |
| Tag into anchor prompt (Mapping A)     | Only the first scene-dominant SFX is prompted in; the rest are timed effects |
| Camera shake intensity (Mapping C)     | Density of SFX per 10 seconds -> shake amplitude      |

### 4.4 `scene_break` + per-scene timing

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Chunk boundary in render pipeline      | One HyWorld scene per OTR scene                       |
| Crossfade duration                     | 0.75-1.5 s at scene boundaries (matches audio bed)    |
| Clip duration cap                      | Enforce C4: if a scene's audio exceeds 12 s, split into multiple shots |
| Max `shots[]` per scene (Director schema extension) | 4 (keeps VRAM predictable; tune after smoke test) |

### 4.5 `voice_traits` -> camera adjective (Mapping C detail)

| Trait keyword (first match, case-insensitive) | Camera treatment                       |
|-----------------------------------------------|----------------------------------------|
| weary, tired, old                             | slow handheld, close                   |
| angry, hostile, sharp                         | fast dolly, canted                     |
| calm, warm, gentle                            | locked off, wide                       |
| frantic, panicked                             | whip-pan, short focal length           |
| announcer, formal                             | clean push-in, centered                |
| child, young                                  | low angle, looking up                  |
| whisper, hushed                               | macro detail, shallow focus            |
| (no match)                                    | slow drift, medium lens                |

### 4.6 Beats and silence

`(beat)` and long gaps between dialogue tokens are narratively load-bearing. In an OTR radio drama, silence is the punchline. In HyWorld:

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Each `(beat)` token                    | 200 ms camera hold                                    |
| Gap > 2 s between dialogue tokens      | Insert a beauty shot (env-only, no cut)               |
| Gap > 5 s                              | Consider title card / intertitle for drama            |

Silence earns its own frame.

---

## 5. How much of this does OTR *have* to do, vs. HyWorld?

Two approaches, same mapping:

### 5.1 Deterministic mapping in OTR (recommended for PoC)

A new pure-Python helper, `otr_v2/hyworld/shotlist.py`, reads `script_lines` and produces a `shots[]` array using the tables above. No LLM. No creativity. Pure lookup + timing math.

**Pros:** deterministic, testable, free of LLM hallucination, easy to diff across runs, byte-stable for the same input. Respects the "audio is king" rule trivially because shotlist generation is a separate, side-effect-free pass on already-written script text.

**Cons:** mappings stay literal. No surprise, no poetry.

### 5.2 LLM shotlist pass (later, when we want poetry)

A small Gemma / Mistral pass reads the parsed script + environment tokens and *writes* the `shots[]` array with camera / duration / mood as structured JSON. Guardrailed by schema validation.

**Pros:** poetic, surprising, handles genre shifts better.
**Cons:** another LLM phase, another VRAM event, non-deterministic. Not for PoC.

**Recommendation:** ship deterministic first. It's the floor. LLM shotlist becomes a toggle later.

---

## 6. Where this plugs into the existing Director schema

From the PoC doc we already reserved optional `shots[]` and `style_anchor_hash` fields. This narrative-mapping doc gives a filling for those fields:

```json
{
  "shots": [
    {
      "shot_id": "s01_01",
      "scene_ref": "1",
      "duration_sec": 9,
      "camera": "slow handheld, close",
      "env_prompt": "fluorescent hum, distant traffic, rain on concrete",
      "sfx_accents": [
        {"at": 3.2, "desc": "metal clatter"},
        {"at": 7.8, "desc": "door slam"}
      ],
      "dialogue_line_ids": ["line_14", "line_15"],
      "mood": "weary"
    }
  ],
  "style_anchor_hash": "a1b2c3d4e5f6"
}
```

All fields remain optional. A v1.7 run produces none of them and is byte-identical to baseline (C7 honored). A v2.0-alpha run with HyWorld enabled produces the full structure.

`env_prompt` is the literal `environment.description` for the scene the shot belongs to. `sfx_accents` carries offset-relative timings (relative to shot start) so renderer can trigger flashes/zooms without reading the WAV. `mood` is the picked camera adjective from 4.5.

---

## 7. Questions for Jeffrey to visualize / debate

These are the gaps I can't close alone. Answer them in whatever medium you want (sketch, voice memo, doc comment) and the final schema will fall out.

1. **Faces or no faces.** C6 says environments only. But do you want a silhouette / named marker / empty chair per character in the 3D scene, or is the scene truly character-free and all emotion is carried by camera + geometry? (Both are interesting. They are different aesthetics.)
2. **One world per episode, or one world per scene?** Mapping A does one-per-scene. Mapping B can do one-per-episode with scene-level camera moves. The second is more expensive but more cinematic.
3. **How literal is the visual?** If the script says "the Mars dome under storm light," do you want that depicted, or do you want a visual analogue that evokes claustrophobia (Mapping C)? This is the single biggest creative axis.
4. **What stays off-screen?** Radio drama's strength is implication. Rendering too much might flatten it. Is there a rule for "this kind of [SFX:] is heard but never shown"? (Candidate: anything gory, anything out-of-frame per narrative convention.)
5. **Color palette source.** Does `[MUSIC:]` drive it? Episode title? Genre? A hash of the script for determinism? We can pick any one for PoC and iterate.
6. **When Pano ships, do we retrofit old episodes?** Consequences for Codex Olympia, Detroit album, Nam June Paik anthology — those already have scripts. They'd each become walkable. That's a creative decision, not a technical one.
7. **Character consistency across episodes.** Same COMMANDER appearing in three episodes — should their visual motif (chair, workstation, light) be consistent? If yes, we need a per-character anchor image cache.
8. **"Broadcast static" aesthetic.** SIGNAL LOST is your brand. Does the HyWorld output get degraded / CRT-filtered / interlaced after rendering, to match the radio's aesthetic? (Probably yes, but should it be an optional post-effect or baked into the prompt?)

---

## 8. Recommended first visualization to pass around

If you want one picture that captures the whole idea: take one existing OTR episode (pick a scene with a strong `[ENV:]` token and a single dominant mood), fill in the `shots[]` array by hand from the tables in Section 4, and sketch three storyboard panels: anchor image, pano speculation, and emotional-geometry version. Present the three next to the audio waveform. That's the conversation starter.

---

## 9. Scope fence

This doc is **creative mapping only.** It does not:
- Choose the anchor-image generator model (that is a separate decision).
- Specify the HyWorld subprocess protocol (covered in PoC doc).
- Modify v1.7 files or behavior.
- Commit to Mapping A, B, or C — it lays out all three so we can pick.
- Propose any additional soak / supersoaker testing.

---

## 10. Next steps (for Jeffrey)

1. Read Section 3 and decide: A, B, C, or a hybrid.
2. Answer Section 7 questions in any medium.
3. Pick one pilot episode for the visualization in Section 8.
4. Separately, in the PoC track: install conda, download WorldMirror 2.0 weights, run the Gate 0 smoke test (see PoC doc Section 8).

Once we have (1), (2), and (3), we freeze a schema and implement `otr_v2/hyworld/shotlist.py` plus the three new nodes.

---

## 11. Interim local stack — stand-ins for Pano / Stereo / Nav

HY-Pano 2.0, WorldStereo 2.0, and WorldNav are all marked "Coming Soon" on Tencent's HY-World 2.0 page with no release date. Tencent's own interim recommendation is HunyuanWorld 1.0, but that model wants 24-48 GB VRAM (OOMs on consumer cards without heavy quantization) — not viable on a 16 GB 5080.

The plan below swaps in open-source, local, ComfyUI-friendly replacements for each slot so Mapping B becomes buildable today. When Tencent ships the real components, we swap stand-ins out one at a time without changing the sidecar contract.

### 11.1 Pano slot — `[ENV:]` text -> 360 equirectangular image

**Primary pick: `Diffusion360_ComfyUI` (wraps `SD-T2I-360PanoImage`)**

- **Why:** Already a ComfyUI plugin. Takes a text prompt, returns a PNG equirectangular panorama. Clean match for the `[ENV:]` description -> pano mapping in 3.2.
- **Upstream model:** `ArcherFMY/SD-T2I-360PanoImage` (SDXL-based pipeline, uses `py360convert` for equirectangular handling).
- **Inputs:**
  - Text-to-pano: `prompt` (string) -> PNG pano.
  - Image-to-pano: `image` (512x512), `mask`, `prompt` -> PNG pano.
- **Output:** PNG, equirectangular projection. Resolution tunable; typical 4096x2048 for downstream splat work.
- **Dependencies (version-locked):** `diffusers >= 0.20.0, <= 0.26.0` (higher versions cause oversaturated SR). Also needs `torch`, `transformers`, `accelerate`, `RealESRGAN`, `py360convert`, `triton`, `xformers`.
- **VRAM:** not stated in README, but SDXL + float16 fits comfortably in 16 GB. Safe for Jeffrey's 5080.
- **Runs where:** sidecar is optional. Because it's a ComfyUI plugin and stays on torch 2.10, we can run it **in-process** in the main ComfyUI graph if we want — no subprocess hop required, unlike HyWorld. That simplifies Mapping B's first leg.
- **Node shape (confirmed from repo presence, exact fields require code inspection):** `Text2Pano` and `Image2Pano` nodes exposed; workflow files `Text2Pano.json` / `Image2Pano*.json` in the repo show wiring.
- **Swap target:** drop this out when HY-Pano 2.0 ships; replace input/output wrapper, keep downstream.

**Backup pick: any SDXL + 360 LoRA ComfyUI workflow**

If `Diffusion360_ComfyUI` has version-pin headaches with our torch 2.10 main env, fall back to a plain SDXL workflow with a 360 LoRA and post-process through `ComfyUI-PanoTools` to fix the seams. Lower fidelity, but no new torch/diffusers constraint.

**Seam check / preview:**
- `ComfyUI_preview360panorama` node renders the equirectangular image as a spherical preview so seams and distortions are visually verifiable before feeding downstream. Not a generator; it's the QA pane.

### 11.2 Stereo slot — pano -> navigable 3DGS

**Primary pick: `SPAG4D` (cedarconnor)**

- **Why:** Purpose-built for exactly this conversion. Equirectangular pano in, PLY (3D Gaussian Splat) out. Multiple backends with known VRAM budgets — some fit inside our 14.5 GB ceiling.
- **Input:** equirectangular JPG/PNG, typical 4096x2048 (accepts other resolutions).
- **Output:** `.ply` file, sRGB SH0 encoding (standard 3DGS PLY consumed by most splat viewers and SuperSplat).
- **VRAM per backend (measured by project):**

  | Backend | VRAM | Notes |
  |---|---|---|
  | DA360 (depth-based) | 6 GB | Lightest; first pick |
  | DAP (depth-based) | 6 GB | Alt depth backend |
  | SHARP 360 (perspective face crops) | 8 GB | Higher fidelity, still safe |
  | GSFix3D refinement | 16 GB | On the ragged edge; skip for PoC |
  | OmniRoam v2 refinement | 48 GB | Not viable on 5080 |

  We use **DA360 or SHARP 360** for PoC. Both fit comfortably under the 14.5 GB ceiling.
- **Invocation (three options):**
  - CLI: `python -m spag4d convert panorama.jpg output.ply [options]`
  - Python API: `SPAG4D(device="cuda").convert("panorama.jpg", "output.ply")`
  - Web UI: `python -m spag4d serve --port 7860` (not needed for our pipeline; nice for debugging)
- **No ComfyUI node exists.** That's fine — SPAG4D runs as a sidecar subprocess just like HyWorld will (reuses the pattern we already designed for C3). A minimal wrapper node `OTR_SPAG4D_Convert` reads pano path, writes PLY to `io/splat_out/<scene_id>.ply`, returns the path.
- **Dependencies:** PyTorch + CUDA. Compatible with a torch 2.4 or 2.10 env; no hard pin known.
- **Swap target:** replace with WorldStereo 2.0 when it ships.

**Alternative: `Splatter-360`**

- Higher-quality panoramic 3DGS (generalizable, wide-baseline). Academic project; VRAM profile less well-documented. Keep as a **later upgrade**, not PoC day-one.

### 11.3 Monocular image -> 3DGS (supports Mapping A)

**Primary pick: `ComfyUI-Sharp` (wraps Apple SHARP)**

- **Why:** Mapping A does not need a pano — it needs a splat from a single anchor image. SHARP predicts Gaussians from one perspective image in under a second. Native ComfyUI node. Very low VRAM.
- **Inputs (nodes exposed):**
  - `Load SHARP Model` — downloads/loads weights; first run auto-fetches to `ComfyUI/models/sharp/`.
  - `Load Image with EXIF` — loads an image and extracts focal length from EXIF; focal length also accepts manual override.
  - `SHARP Predict` — the core node: `image` + `focal_length` -> 3D Gaussians.
- **Output:** PLY (3DGS), consumed by `ComfyUI-GeometryPack` for visualization or by SPAG4D's viewer.
- **VRAM:** not documented, but Apple's reference implementation runs sub-1-second on a single card; safe assumption is well under 8 GB.
- **Dependencies:** model weights auto-downloaded from HF.
- **Fit in the narrative mapping:** Mapping A's "anchor image -> lifted 3D" step. This replaces the "synthesize 8-12 virtual viewpoints + feed into WorldMirror 2.0" workaround with a one-shot call that's faster and uses less VRAM. WorldMirror 2.0 still wins when we have genuine multi-view coverage (e.g. generated video frames); SHARP wins when we have a single anchor still.

**Umbrella bundle worth knowing:** `ComfyUI-3D-Pack` (MrForExample) packages InstantMesh, TripoSR, 3DGS, and NeRF nodes in one extension. If we end up needing more than two of these primitives, switch to the pack instead of installing them piecemeal.

### 11.4 Nav slot — camera trajectory across a 3DGS scene

**Primary pick for PoC: hand-authored camera paths in the Director `shots[]` array**

Every shot already carries `camera` (string like `"slow handheld, close"`) and `duration_sec`. For PoC we turn those strings into discrete keyframes (start pose + end pose + easing) in deterministic code and hand them to any splat renderer that accepts a camera path (SuperSplat, Polycam, a custom pyrender driver). No new model dependency. Good enough to ship the first real episode.

**Primary pick for "real" Nav: `SplaTraj`**

- **Why:** Formalizes camera path generation as trajectory optimization over a semantic splat. You specify regions/objects to visit; it plans an occlusion-aware, object-centered path. Closest match to WorldNav's stated intent.
- **Inputs:** a 3DGS scene (PLY or equivalent), a target list (semantic anchors: region labels or object IDs), a set of constraints (duration, smoothness).
- **Outputs:** a camera trajectory (sequence of poses) that can be rendered against the splat.
- **Fit with OTR:** `dialogue_line_ids` and `character_name` positions from Mapping B become the semantic target list. "Dolly from the COMMANDER marker to the ANNOUNCER marker while line 14-17 plays" becomes one SplaTraj call.
- **Status:** academic code; may need wrapping. Not day-one. Queue behind pano + stereo.

**Interactive backup: `SuperSplat` + `KIRI 3DGS Render v4.0`**

- `SuperSplat` — PlayCanvas's MIT-licensed web editor. Loads PLY, supports camera animation, fly-throughs, hotspots (up to 25 per scene). Great for hand-authoring a signature episode opener.
- `KIRI 3DGS Render v4.0` — free Apache-2.0 Blender add-on that keyframes camera paths through splats. If we want final composite through Blender (good for matching SIGNAL LOST's broadcast-static aesthetic in post), this is the hook.

### 11.5 Updated pipeline diagram (interim stack)

```
OTR script (v1.7)
    |
    v
_parse_script() -> script_lines (dialogue, environment, sfx, scene_break, ...)
    |
    v
otr_v2/hyworld/shotlist.py (deterministic rules, Section 4 tables)
    |
    +--> per-scene: [ENV:] prompt, shots[] (camera, duration, dialogue_line_ids)
    |
    v
FORK A (anchor image + lifted 3D)      FORK B (pano -> navigable splat)
   |                                        |
   SDXL anchor gen                          Diffusion360_ComfyUI (Text2Pano)
   |                                        |
   ComfyUI-Sharp (SHARP Predict)            SPAG4D (DA360 backend, subprocess)
   |                                        |
   PLY                                      PLY
   |                                        |
   +----------------+-----------------------+
                    |
                    v
   OTR_HyworldRenderer
     - hand-authored camera path from shots[].camera (PoC)
     - SplaTraj optimizer (later) for semantic trajectories
                    |
                    v
   per-scene MP4 (muxed with untouched v1.7 WAV)
```

### 11.6 What this adds to the Director schema

One new optional field at the shot level, so we can track which model produced which output (helpful when we swap stand-ins for real components):

```json
{
  "shot_id": "s01_01",
  "scene_ref": "1",
  "duration_sec": 9,
  "camera": "slow handheld, close",
  "env_prompt": "fluorescent hum, distant traffic, rain on concrete",
  "dialogue_line_ids": ["line_14", "line_15"],
  "mood": "weary",
  "visual_backend": {
    "pano": "diffusion360",
    "stereo": "spag4d_da360",
    "nav": "hand_authored"
  }
}
```

All three `visual_backend` fields are free-form strings and optional. When HY-Pano 2.0 ships, the value just changes to `"hy_pano_2"`; no schema migration.

### 11.7 Install order (interim stack)

Do not execute yet — this is the plan:

1. **In main ComfyUI env (torch 2.10):** install `ComfyUI_preview360panorama`, `ComfyUI-PanoTools`. Both are viewer / utility, low risk.
2. **New light sidecar env `otr_pano` (torch 2.4, CUDA 12.4, Python 3.10):** install `SD-T2I-360PanoImage` with the pinned diffusers range. Reason for a separate env: the `diffusers <= 0.26.0` pin conflicts with newer ComfyUI features. Worth isolating.
3. **Same `otr_pano` env:** install `SPAG4D`. Minimal additional deps beyond torch.
4. **In main ComfyUI env:** install `ComfyUI-Sharp` + `ComfyUI-GeometryPack`. Weights auto-download on first run.
5. **Later:** `SplaTraj` into the `otr_pano` env when we move past hand-authored paths.
6. **HyWorld 2.0 (`hyworld2` env, torch 2.4)** stays its own sidecar as designed in the PoC doc.

That's three envs total: main (torch 2.10), `otr_pano` (torch 2.4 + pinned diffusers), and `hyworld2` (torch 2.4 + HY-World). Each one isolated so a bad pip install in one cannot break the others.

### 11.8 Known unknowns / things to verify at install time

- Exact ComfyUI node parameter names for `Diffusion360_ComfyUI` (README doesn't enumerate; need to inspect `Diffusion360_nodes.py` after clone).
- VRAM of `SPAG4D SHARP 360` backend in practice on Blackwell (documented 8 GB; verify on 5080).
- Whether `ComfyUI-Sharp` needs a focal length we don't have from a synthetic anchor (likely workaround: pass a sensible default like 28mm equivalent).
- Whether SDXL + 360 LoRA works well enough as a fallback if `SD-T2I-360PanoImage`'s diffusers pin fights ComfyUI.

Flag these as smoke-test items; none block design.

---

## 12. Round 1 brainstorm — OTR outputs <-> visual model inputs

This section lays every OTR output shape next to every visual-model input shape in one place, then proposes three ways to wire them together. The three lanes are not mutually exclusive — an episode can be Lane 1 overall with a Lane 3 cold open, or Lane 2 for chapter beats and Lane 3 for interstitials. They are **modes**, not **forks**.

### 12.1 Side-by-side: what OTR produces vs. what each model eats

**Left column:** every shape of data OTR already emits per episode (from `_parse_script` + sequencer + director).
**Right column:** every input slot across the interim stack and HyWorld 2.0.

| OTR output (shape, source)                                             | Matches directly | Matches with rewrite | Matches only chaotically |
|------------------------------------------------------------------------|------------------|----------------------|--------------------------|
| `title.value` — episode title string                                   | style-anchor seed | pano mood modifier  | pano prompt itself       |
| `scene_break.scene` — scene label / number                             | shot boundary    | shot_id suffix       | scene shuffle key        |
| `environment.description` — 3-4 descriptor prose                       | **HY-Pano / Diffusion360 prompt (verbatim)** | SDXL anchor prompt (prefix with "establishing shot of") | SHARP focal length derived from string hash |
| `sfx.description` — single-event prose                                 | timeline accent  | SDXL negative prompt ("no {sfx}") for surreal inversion | pano prompt substitute |
| `pause` (beat / duration_ms)                                           | camera hold      | trajectory easing    | splat density modulator  |
| `dialogue.character_name` — uppercase token                            | 3DGS anchor name | semantic target for SplaTraj | pano seed via name hash |
| `dialogue.voice_traits` — "male, 50s, weary" etc.                      | camera adjective (Section 4.5) | LLM -> pano mood phrase | SDXL LoRA selector by hash |
| `dialogue.line` — the spoken text                                      | subtitle only    | LLM-distilled visual metaphor | **direct pano prompt** |
| `direction` — leftover stage prose                                     | ignored          | LLM-augmented shot description | feedstock for Lane 3 |
| `music.description` — opening/closing theme tag                        | palette cue      | LLM -> color grade LUT | palette inverter |
| Audio offsets (seconds, per token)                                     | shot start/end times | trajectory keyframes | scrubbed; offsets swapped between tokens |
| Character roster (derived)                                             | anchor placements | per-character SDXL embedding | permuted between episodes |
| Scene durations (derived)                                              | clip length cap (C4) | LLM-balanced pacing | uniform random 3-12 s |

The first column is the literal-translation lane (Lane 1). The second is the LLM-shaped lane (Lane 2). The third is the chaos lane (Lane 3). They are discussed below.

### 12.2 Lane 1 — Faithful (as-is, deterministic)

Every OTR token maps to the most direct visual input through pure lookup or trivial string math. No LLM second pass. Same script in, same visuals out, every run.

**Feature of the lane:** reproducibility. We can regenerate an episode's visuals after any code change and diff byte-by-byte (modulo model nondeterminism, which we minimize by fixing seeds). Regression-testable.

**Canonical wiring (Mapping B flavor):**

```
environment.description  -> Diffusion360 Text2Pano (verbatim prompt)
scene.duration_s         -> shot duration (clamped [3, 12])
voice_traits[0]          -> camera adjective (Section 4.5 table)
dialogue_line_ids        -> SplaTraj semantic targets (future) / hand-authored camera (PoC)
character_name           -> named anchor position in 3DGS
sfx.description @ offset -> visual accent (flash/zoom) at that time
(beat) / gap > 2s        -> held beauty shot
```

**Strength:** legible. A reader of the script can predict what they'll see. Good for trailers, documentation, festival submission cuts.

**Weakness:** risks being too literal. A radio drama that was *evocative* in audio becomes *explanatory* in visuals. This is the Uncanny Caption problem: "the listener imagined a cathedral; we rendered a parking lot." Counter with Lane 2 or selectively with Lane 3.

### 12.3 Lane 2 — Translated (LLM-shaped, semantic)

Between OTR's parse output and each visual model's input sits one small local LLM pass (Gemma 3 4B / Mistral Nemo, already part of the stack). The LLM reads tokens in context and **rewrites** them to fit each model's idiom, without inventing story content. It is a translation layer, not an author.

**Four concrete LLM jobs in this lane:**

1. **Pano prompt expansion.** Input: `environment.description` + the scene's dialogue lines + `voice_traits`. Output: a single sentence in the idiom the pano model prefers, enriched with lighting, camera height, and lens. Rule: the LLM may only reorder, enrich, and add atmosphere — never change the place or the objects named. Validator rejects if named nouns disappear.
2. **Camera brief.** Input: a scene's dialogue arc + `voice_traits` + mood. Output: a `shots[]` array with camera strings, durations, and dialogue_line_ids groupings — the same shape the deterministic rules in Section 4 produce, but allowed more variety. Feeds SplaTraj when SplaTraj is online.
3. **Visual metaphor for dialogue.** Input: a standout dialogue line. Output: a short visual phrase ("an empty chair where the voice is speaking from"). Used for the 1-3 "hero frames" per episode. Does not run on every line. Opt-in per scene.
4. **SFX -> negative prompt / accent descriptor.** Input: `sfx.description`. Output: an enriched accent brief for the renderer, or a negative prompt for the pano (e.g. SFX of breaking glass -> "no intact windows"). Optional; off by default.

**Feature of the lane:** the pano feels lived-in, the cameras feel intentional, the script's voice comes through in the frame. It is the lane that makes the difference between "script + stock visuals" and "this looks like SIGNAL LOST."

**Cost:** +1 LLM pass per scene (seconds, cached by hash of inputs). Not byte-stable across runs unless we fix the LLM seed and use a frozen model revision — we do both, because C7 tolerance matters.

**Guardrails:**
- LLM output passes through the same JSON schema validator the Director uses. Off-spec output is discarded; fallback to Lane 1 mappings.
- LLM may only consume tokens from within one scene at a time. No cross-scene spoilers.
- LLM cannot invent `character_name` values. It can enrich `voice_traits` but not add new names.
- Cache keyed on `sha256(prompt_template + inputs + model_revision + seed)`. Hit rate should be near 100% across reruns.

### 12.4 Lane 3 — Chaotic (avant-garde, intentionally misaligned)

Deliberate mismatches between what a token *is* and what model input it *drives*. The goal is not incoherence; it is **productive strangeness** — the kind of friction that a Nam June Paik anthology or a Codex-Olympia piece earns. Every choice below is misaligned on purpose and each run is seeded so the mismatch is reproducible.

**A menu of chaos operators (pick 0-3 per episode):**

- **Swap.** Feed `dialogue.line` (the spoken text) into the pano prompt, and `environment.description` into the subtitle track / on-screen text. The image is what was said; the text on screen is where it was said.
- **Shuffle.** Pair scene `n`'s pano prompt with scene `n+k`'s shot list, where `k` is a seeded integer per episode. Rooms go with someone else's camera moves.
- **Permute roster.** Across an anthology, rotate each character's anchor position through every episode. COMMANDER becomes ANNOUNCER's spot in episode 3, reverts in episode 5. Visual recurrence without narrative recurrence.
- **Offset drift.** Use `audio_offset_s` values, but rotated by a seeded amount. Visual accents land *almost* on their SFX — a quarter-second early or a beat late. Sells a disoriented feel.
- **Random-parts selection.** Every shot randomly picks one of `{environment, dialogue, sfx, direction, title}` from a *different random scene* as its pano prompt. One episode sees its own Rosetta stone scrambled.
- **Voice-traits wins.** Ignore `environment.description` entirely; drive pano generation purely from the combined `voice_traits` strings of every character in the scene ("male 50s weary + female 30s frantic + announcer formal"). The room looks like the people.
- **Beat is the prompt.** `(beat)` tokens become full shots. The pano prompt for a beat is the previous dialogue line run through a negation ("the opposite of what was just said"). Silence rendered.
- **Music is the weather.** `[MUSIC:]` tag becomes the only weather/time-of-day cue for the pano. Otherwise ignored.
- **Scene sum.** A whole episode compressed into a single pano via concatenation of all `[ENV:]` descriptions. One pano, played under the entire runtime, camera drifts according to dialogue cadence.

**Feature of the lane:** makes the *process* the artwork. The audience can't re-derive visuals from the script, and they shouldn't want to — Lane 3 is where OTR stops being a visualization of a radio play and starts being a separate piece that shares materials with the play.

**Constraints kept even in chaos:**
- Audio path is never touched (C7).
- C6 still holds: no character faces, even under swap.
- Per-episode seed logged so any Lane 3 run is reproducible. Gallery submission needs this.
- Each chaos operator is a toggle in the `chaos_ops[]` array on the Director extension. Default is empty (Lane 1). Opt-in per episode.

### 12.5 How the three lanes cohabit in one workflow

```
                  +------------------+
  script_lines -->|  shotlist.py     |-->  deterministic shots[]      (Lane 1 floor)
                  +------------------+
                           |
                           v
                  +------------------+
                  |  llm_shaper.py   |-->  enriched shots[]           (Lane 2, optional)
                  | (Gemma / Mistral)|     pano_prompts, camera briefs
                  +------------------+
                           |
                           v
                  +------------------+
                  |  chaos_ops.py    |-->  mutated shots[]            (Lane 3, opt-in ops)
                  | (seeded rng)     |
                  +------------------+
                           |
                           v
                  visual backends (Pano, Stereo, Nav / WorldMirror)
```

Lanes stack. Lane 1 is the floor and always runs (determinism safety net). Lane 2 enriches on top if enabled. Lane 3 mutates on top if chaos ops are picked. Fallback on any failure collapses one layer down — Lane 3 failure -> Lane 2 output; Lane 2 failure -> Lane 1 output; Lane 1 is the last line of defense and cannot fail.

### 12.6 Episode-level preset ideas

A few "director's presets" that bundle lane choices into named modes so Jeffrey can pick one per project:

| Preset                 | Lanes active | Chaos ops                                   | Typical use                    |
|------------------------|--------------|---------------------------------------------|--------------------------------|
| `signal_lost_clean`    | 1            | none                                        | flagship episodes, reference   |
| `signal_lost_dream`    | 1 + 2        | none                                        | most episodes; "default art"    |
| `nam_june_paik_mode`   | 1 + 2 + 3    | swap, shuffle, offset drift                 | Nam June Paik anthology        |
| `codex_olympia_mode`   | 1 + 2 + 3    | scene sum, voice-traits wins                | LA28 cinematic anthology       |
| `experimental_raw`     | 1 + 3        | random-parts selection, permute roster      | gallery one-offs, no LLM cost  |

Presets are a single field on the Director extension. Each preset maps to a fixed set of lanes + chaos op toggles. Reproducible.

### 12.7 Questions this brainstorm raises for Section 7

Adding to the creative questions list:

9. **Which lane is "default"?** Lane 1 is safest; Lane 2 is prettier; Lane 3 is the art-forward statement. Any new episode should pick a lane explicitly.
10. **Is chaos per-episode or per-scene?** Can one scene be Lane 3 while the rest is Lane 1 (cold-open-as-statement), or is it all-or-nothing?
11. **How much scripted chaos is acceptable?** i.e., is the chaos operator list itself part of the Director JSON (reproducible), or is it an on-the-fly knob in the ComfyUI UI (exploratory)? Probably both — JSON is the record of truth; the UI is a scratch pad.
12. **How do we flag Lane 3 runs to the audience?** A watermark? A title card? Nothing? ("Nothing" is the Nam June Paik answer.)

### 12.8 Minimum viable path through all three lanes

- **Week 1 of work:** ship Lane 1 deterministic mapping + Mapping A (anchor + SHARP) to get first real frames.
- **Week 2:** add Lane 2 LLM shaper as an optional pass. Same data contract.
- **Week 3:** implement two Lane 3 chaos operators (swap + shuffle) behind a single Director toggle. Demo one anthology piece with chaos on.
- **Later:** bring in SPAG4D for pano-based splats and SplaTraj for nav, uplifting Mapping B to real.

No HyWorld 2.0 or "Coming Soon" component blocks any of the above. Every week ships a watchable artifact.
