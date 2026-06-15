# Cloud Engines — Coding & Wiring Plan (v1)

Date: 2026-06-14
Status: BUILD PLAN (planner window — for a coder window to execute). No code here.
Scope locked by operator (2026-06-14): add **3 cloud engines** behind one opt-in
flag, default-OFF.

- **`cloud_flux_pro`** — image engine (Flux Pro 1.1 via Comfy API node). All
  image roles.
- **`cloud_ltx2`** — video engine (LTX-2 fast via Comfy API node).
  image-to-video / text-to-video. All video roles.
- **`cloud_kling_avatar`** — video engine (Kling Avatar via Comfy API node).
  audio-driven talking face.

Grounded against: `nodes/_otr_image_engines/{registry,flux_gen1,__init__}.py`,
`nodes/_otr_video_engines/{registry,eng_humo,eng_ltx_video,__init__}.py`,
`nodes/_otr_shared/{role_compat,capability_profiles}.py`,
`nodes/otr_image_gen_dispatcher.py`, `nodes/_otr_video_engines/schemas.py`.

---

## 0. The one thing that changes the wiring: the three engines take DIFFERENT inputs

This is why a roundtable on wiring is warranted. `role_compat.py` offers an
engine in a role **only if the role can supply every one of the engine's
`required_inputs`**. The roles supply:

| Role (token) | text_prompt | init_image | audio_ref | base_clip_ref |
|---|:--:|:--:|:--:|:--:|
| `announcer_visual` | ✅ | ✅ | ✅ | ✅ |
| `character_video` | ✅ | ✅ | ✅ | ✅ |
| `music_visual` | ✅ | ✅ | ❌ | ✅ |
| `scene_broll` | ✅ | ✅ | ❌ | ✅ |
| `background_abstract` | ✅ | ❌ | ❌ | ❌ |

So each engine's `required_inputs` decides where it can appear — automatically:

| Engine | family | required_inputs | Appears in roles |
|---|---|---|---|
| `cloud_flux_pro` | static_image_gen | `("text_prompt",)` | all image roles |
| `cloud_ltx2` | image_to_video | `("init_image",)` | announcer, music, character, scene (NOT background_abstract — no init_image) |
| `cloud_kling_avatar` | audio_driven_face | `("audio_ref","init_image")` | **announcer + character ONLY** |

**Operator note (important):** you asked for Kling Avatar to be selectable on
"announcer, music, or other beats." Kling Avatar is **audio-driven** — it
animates a portrait *to speech*. The music and background-abstract beats carry
no `audio_ref`, so there is nothing for it to lip-sync. By the existing
fail-closed rule it will be offered on **announcer** and **character** beats
(the "other beats" that have a voice), and excluded from music/abstract. For
motion on music / scene / background, **`cloud_ltx2`** is the cloud option there
(it fits all of those). This keeps the dropdowns honest instead of offering a
pick that can't run. If you instead want a *generic* (non-talking) Kling clip on
music beats, that's a different engine (Kling t2v/i2v) — say so and we add it as
`cloud_kling_video` with `required_inputs=("init_image",)`.

The mapping the operator gets, then:

| Dropdown slot | Cloud options offered |
|---|---|
| `announcer_video_model` | `cloud_kling_avatar`, `cloud_ltx2` |
| `music_video_model` | `cloud_ltx2` |
| `other_beats_video_model` | `cloud_ltx2` (+ `cloud_kling_avatar` on character beats) |
| `announcer_image_model` / `music_image_model` / `other_beats_image_model` | `cloud_flux_pro` |

---

## 1. The adapter pattern (what every engine already does — copy it)

Grounded from `flux_gen1.py` (image) and `eng_humo.py` / `eng_ltx_video.py`
(video). A new engine is one `@register` class in its namespace, imported
(guarded) in the package `__init__.py`. Cold-import clean (V-12): module scope
imports only stdlib + the dep-free registry; the network/SDK call is lazy inside
`render_*`.

Declared fields (registry core): `name`, `roles`, `default_roles=()` (opt-in,
never a silent default), `commercial_clean`, `requires_flag`, `required_inputs`,
`engine_version`. Image adds `render_image(request)->png path|uint8 array` +
`assert_usable`. Video extends `MotionEngineBase`, adds `family`,
`declared_isolation`, `target_fps`, `fallback_engine`, `render_clip` +
`canonicalize`.

The dropdown auto-populates: the Director's COMBO is the full registry (V-6) and
`role_compat` filters at execute time. **No Director edit is needed for the
dropdown to show the new engines** — registering + importing is enough.

---

## 2. Per-engine wiring spec (the different inputs/outputs)

### 2a. `cloud_flux_pro` (image) — file `nodes/_otr_image_engines/cloud_flux_pro.py`
- Mirror `FluxGen1ImageEngine`. `roles = ROLES`, `default_roles = ()`,
  `required_inputs = ("text_prompt",)`, `requires_flag = "OTR_ENABLE_CLOUD"`,
  `commercial_clean = False` (verify BFL/Comfy ToS; default False).
- **Inputs from the request** (already produced today): `prompt`, `width`/`w`,
  `height`/`h`, `seed` (advisory only on cloud). Map to the Flux Pro API node's
  prompt + aspect/size.
- **Output:** write the returned PNG to `ComfyUI/output/...` and **return its
  path** (the dispatcher's `_coerce_pixels` PATH branch handles it). Do NOT
  return a raw ComfyUI IMAGE tensor — `_coerce_pixels` raises `TypeError` on it.
- `assert_usable`: fail closed (raise `EngineUnusable`) when the master flag is
  off, no API key, or the cost guard would be exceeded; else return `self.name`.

### 2b. `cloud_ltx2` (video) — file `nodes/_otr_video_engines/cloud_ltx2.py`
- Mirror `eng_ltx_video` shape but cloud. `family = "image_to_video"`,
  `roles = ("announcer_visual","music_visual","character_video","scene_broll")`,
  `default_roles = ()`, `required_inputs = ("init_image",)`,
  `requires_flag = "OTR_ENABLE_CLOUD"`, `fallback_engine = "still_kenburns"`
  (zero-VRAM floor; cloud has no local sibling to fall to),
  `target_fps` per LTX-2, `declared_isolation` = a NEW network value (see §4).
- **Inputs:** `init_image` (the ST-3 scene still OTR already mints), `text_prompt`,
  and a **clip duration** — derive from the beat's audio/budget window the same
  way the local LTX adapter sizes frames; LTX-2 is billed **per second**, so the
  duration must be bounded by the cost guard.
- **Output:** an MP4 written locally; return it through the existing
  `render_clip` → `canonicalize` path exactly like a local motion engine.

### 2c. `cloud_kling_avatar` (video) — file `nodes/_otr_video_engines/cloud_kling_avatar.py`
- Mirror `eng_humo` (the audio-driven template). `family = "audio_driven_face"`,
  `roles = ("announcer_visual","character_video")`,
  `required_inputs = ("audio_ref","init_image")`, `default_roles = ()`,
  `requires_flag = "OTR_ENABLE_CLOUD"`, `fallback_engine = "still_kenburns"`.
- **Inputs:** `init_image` (the portrait) + `audio_ref` (the per-line voice WAV
  OTR already feeds HuMo). Kling Avatar is billed **per second** of the output,
  which equals the audio length — duration is the audio's duration; bound by the
  cost guard.
- **Output:** silent MP4 (V-1: only `OTR_MasterAudioMux` adds audio) through
  `render_clip` → `canonicalize`.

---

## 3. Registry + capability rows (required, or the engine is excluded)

For EACH engine add a `CAPABILITIES` row in its registry
(`_otr_image_engines/registry.py`, `_otr_video_engines/registry.py`):

```
"<engine>": {"vram_class": "cpu", "vram_estimate_mb": 0,
             "required_toolchain": None, "requires_sidecar": False,
             "cpu_ok": True, "model_requirements": []}
```

`vram_class="cpu"` ranks 0 in `VRAM_CLASS_RANK`, so it fits **any** profile's
`max_model_class` with **no `capability_profiles.py` validator change**
(confirmed: `max_model_class ∈ {cpu,light,medium,heavy}`; a cpu engine always
fits). The cloud-ness (network + auth + cost) is enforced at runtime by the flag
/ auth probe / cost guard, not by the profile.

Register each adapter import (guarded `try/except`) in the package `__init__.py`
next to the existing engines.

---

## 4. Dispatcher / lease changes (the NOT-zero-change part)

1. **Skip the GPU lease for network engines.** `dispatch_images`
   (`otr_image_gen_dispatcher.py`) calls `_lease.acquire()` *unconditionally*
   before `gen_fn`. Add a check: if the resolved engine's `vram_class` is `cpu`
   AND it is a network engine (new `declared_isolation == "network"` marker, or
   a `is_network = True` adapter attribute), skip `acquire`/`release` and the
   post-gen NVML probe. Do the equivalent on the video render path's lease site.
   Rationale: a blocking cloud call must not hold the 16 GB for the network wait.
2. **Cost guard (request-context, not `assert_usable`).** Add an
   `estimate_cost(request, policy)` step the dispatcher calls before dispatch
   (it has w/h/duration/engine there), or enforce in `render_*`. Minimal v1: a
   per-episode credit ceiling in the image/video policy, deterministic
   reserve/spent accounting, idempotent on retry; over-budget ⇒ fail closed
   (skip → radio floor, LOUD). Estimate from a **dated price table** keyed by
   engine→model→unit (per-run / per-second); unknown/stale price ⇒ fail closed.
3. **Auth probe.** Dep-free helper: is a Comfy API key present
   (`OTR_COMFY_API_KEY` env, then Comfy account)? Surface a missing key as
   `EngineUnusable` in `assert_usable` so the dropdown/report says why.
4. **Per-adapter network timeouts:** connect + total wall-clock + poll interval
   + 429/5xx retry + cancel. Separate from the local render watchdog.

---

## 5. The Partner-Node invocation seam (verify-at-build spike — do FIRST)

Before coding the three adapters, settle ONE question with a throwaway probe:
**can a ComfyUI API/Partner node be driven programmatically from inside an
adapter method (like `flux_gen1` drives core nodes via `wrapper_bridge`), or
does it require the graph executor?** Options, in preference order:
1. Import the partner node class and call its function in-process (matches the
   existing `wrapper_bridge.run_graph` pattern). Verify it honors the Comfy API
   key + credit billing outside a full prompt run.
2. POST to the running ComfyUI server's API with a tiny sub-workflow.
3. Direct provider HTTP call with the Comfy API key (fallback; bypasses the
   node layer but loses Comfy's unified billing).

Pin: the API-key source + precedence, the polling contract, and **output-file
discovery** (where the node writes the PNG/MP4 so the adapter can return the
path). Keep the registry import cold-import clean — the SDK/HTTP client is lazy,
inside `render_*` only.

---

## 6. Workflow JSON wiring (`workflows/otr_scifi_16gb_full.json`) — CLAUDE.md §0

- The six dropdowns (`announcer_video_model`, `music_video_model`,
  `other_beats_video_model`, `announcer_image_model`, `music_image_model`,
  `other_beats_image_model`) are COMBOs built from the registry at
  `INPUT_TYPES` time, so the new engine ids appear **automatically** once
  registered + imported — **no JSON edit needed to make them selectable.**
- `widgets_values` is POSITIONAL and append-only (BUG-LOCAL-097). v1 ships
  default-OFF, so the SAVED selections stay on the local engines; the operator
  opts in by changing a dropdown to a `cloud_*` value at runtime. Only if you
  want a cloud engine as the SAVED default do you edit that widget's value —
  and then re-run `OTR_WorkflowValidator` + the link/widget audit.
- No new node, no new link for v1 (the engines plug into the existing Director +
  dispatcher). If the cost guard adds a Director widget (e.g. a per-episode
  credit ceiling), that is ONE new optional widget APPENDED at the end of the
  Director's widget list — never inserted mid-list.

---

## 7. Tests (run the suite + Bug Bible after every change — CLAUDE.md §3)

- Cold-import: `test_cold_import_no_heavy_libs` still passes with the 3 adapters
  registered (no torch/SDK at module scope).
- `all_engine_names()` lists the 3 cloud rows with `OTR_ENABLE_CLOUD` off.
- `role_compat`: `cloud_kling_avatar` is offered ONLY for announcer + character;
  `cloud_ltx2` for those + music + scene; `cloud_flux_pro` for all image roles.
- `assert_usable` fails closed when flag off / no key / over budget (no silent
  local substitution — the BUG-LOCAL-405 contract).
- Lease-skip: a network engine does not call `_lease.acquire` (mock).
- Output coercion: cloud image adapter returns a path; `_coerce_pixels` accepts
  it; a returned IMAGE tensor is rejected LOUD.
- Cost guard: over-ceiling episode fails closed; price-table miss fails closed.

---

## 8. Build order (sprints)

1. **S0 spike:** the §5 invocation-seam probe (1 model, 1 call) — settle how to
   call a Partner node + key + output discovery. Gates everything.
2. **S1:** dispatcher lease-skip + `declared_isolation="network"` marker +
   cold-import/role_compat/lease tests. No engine yet.
3. **S2:** `cloud_flux_pro` (simplest: prompt→png) end to end + cost guard +
   auth probe + dated price table.
4. **S3:** `cloud_ltx2` (init_image + duration → mp4) through render_clip.
5. **S4:** `cloud_kling_avatar` (audio_ref + init_image → mp4).
6. **S5:** commercial_clean table + dropdown-UX decision + README/wizard note;
   live smoke on the operator's credits (small budget).

---

## 9. Open decisions for the operator

1. **Kling on music beats?** Confirm: Kling Avatar is talking-face only
   (announcer + character). For music/scene motion the cloud option is LTX-2.
   Want a separate generic `cloud_kling_video` for music too? (default: no.)
2. **Per-episode credit ceiling** default value + behavior when hit (skip to
   radio floor vs hard stop). Recommend: skip that beat to the floor, LOUD.
3. **Dropdown UX:** cloud rows always visible-but-disabled (with help text) when
   `OTR_ENABLE_CLOUD=0`, vs hidden until enabled. Recommend: visible-disabled
   (discoverability for the newbie audience).
4. **commercial_clean defaults** — set all three `False` until ToS confirmed?
   Recommend: yes, with a dated source URL in adapter metadata.
