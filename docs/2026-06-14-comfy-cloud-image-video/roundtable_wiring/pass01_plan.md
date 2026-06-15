# Cloud Engines — Coding & Wiring Plan (v1)

Date: 2026-06-14
Status: BUILD PLAN (planner window — for a coder window to execute). No code here.
Hardened: roundtable pass 01 (GPT-5.5 + Gemini 3.1 Pro + DeepSeek-v4-pro,
grounded). Corrections folded below + summarized in Section 10; see
`roundtable_wiring/pass01_judgment.md`.
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
| `cloud_ltx2` | text_to_video (i2v opportunistic) | `("text_prompt",)` | **all 5 video roles** (uses `init_image` when the beat supplies one; else t2v) |
| `cloud_kling_avatar` | audio_driven_face | `("audio_ref","init_image")` | **announcer + character ONLY** |

> Roundtable correction: `cloud_ltx2` is declared on `text_prompt` (not
> `init_image`) so it fits every video role including `background_abstract`
> (text-only); the adapter conditions on the scene still when one exists. An
> `init_image`-required declaration would have been excluded from
> `background_abstract` — contradicting "all video roles."

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
| `other_beats_video_model` | `cloud_ltx2` (+ `cloud_kling_avatar` on character beats only — see below) |
| `announcer_image_model` / `music_image_model` / `other_beats_image_model` | `cloud_flux_pro` |

> Roundtable correction (per-beat partiality): `other_beats_video_model` is ONE
> saved selector spanning three roles (character / scene / background). If it is
> set to `cloud_kling_avatar`, only **character** beats can run it; **scene and
> background beats fail closed to the fallback, LOUD** (ShotLock/role_compat).
> Recommendation: offer Kling Avatar primarily on `announcer_video_model`; a
> dedicated `character_video_model` selector is a possible later refinement.

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
- **Method signature (roundtable):** `render_image(self, request,
  prepared=None)` — `_inprocess_gen_fn` calls `render_image(request, prepared)`,
  so a one-arg signature would `TypeError`.
- **Inputs from the request** (already produced today): `prompt`, `width`/`w`,
  `height`/`h`, `seed` (advisory only on cloud). Map to the Flux Pro API node's
  prompt + aspect/size.
- **Output (roundtable):** prefer returning an **in-memory uint8 numpy array**
  (convert the API node's IMAGE tensor, like `flux_gen1` returns
  `images_to_uint8(...)[0]`) — `_coerce_pixels` accepts it and you avoid the
  `wait_for_file_ready` disk round-trip. If the node only yields a file, return
  its `.png` path. Never return a raw torch IMAGE tensor (`_coerce_pixels`
  raises `TypeError`).
- `assert_usable`: fail closed (raise `EngineUnusable`) ONLY for flag-off /
  missing API key; else return `self.name`. **The cost guard is NOT here** — it
  runs in the dispatcher after the request is built (§4.2).

### 2b. `cloud_ltx2` (video) — file `nodes/_otr_video_engines/cloud_ltx2.py`
- Mirror `eng_ltx_video` shape but cloud. `family = "text_to_video"`,
  `roles = ROLES` (all five), `default_roles = ()`,
  `required_inputs = ("text_prompt",)` (roundtable: NOT `init_image` — that
  would exclude `background_abstract`; the adapter uses `init_image` when the
  request has one, i2v, else t2v), `requires_flag = "OTR_ENABLE_CLOUD"`,
  `fallback_engine = "still_kenburns"` (zero-VRAM floor; cloud has no local
  sibling), `is_network = True`, `target_fps` pinned to the LTX-2 output rate.
- **Inputs:** `text_prompt` (always), `init_image` (the ST-3 scene still, when
  present → i2v), and a **clip duration** = `timing.target_frame_count /
  target_fps`, clamped to a pinned min/max and bounded by the cost guard (LTX-2
  is billed **per second**).
- **Output:** an MP4 written locally; return it through the existing
  `render_clip` → `canonicalize` path exactly like a local motion engine.

### 2c. `cloud_kling_avatar` (video) — file `nodes/_otr_video_engines/cloud_kling_avatar.py`
- Mirror `eng_humo` (the audio-driven template). `family = "audio_driven_face"`,
  `roles = ("announcer_visual","character_video")`,
  `required_inputs = ("audio_ref","init_image")`, `default_roles = ()`,
  `requires_flag = "OTR_ENABLE_CLOUD"`, `fallback_engine = "still_kenburns"`,
  `is_network = True`.
- **Inputs:** `init_image` (the portrait) + `audio_ref` (the per-line voice WAV
  OTR already feeds HuMo). Kling Avatar is billed **per second** = the audio
  length; **duration = the `audio_ref` duration** (probe the WAV header),
  clamped and bound by the cost guard.
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

1. **Single network marker.** All three adapters set a class attribute
   `is_network = True`. The lease sites read `getattr(engine, "is_network",
   False)` — NOT a new CAPABILITIES key (keeps the capability-decl validator
   untouched) and NOT a new `declared_isolation` enum (avoids motion_common
   constant changes).
2. **Skip the GPU lease AND the post-gen NVML probe for network engines.**
   `dispatch_images` calls `_lease.acquire()` *unconditionally* before `gen_fn`,
   AND after the `finally` it *unconditionally* calls
   `_lease.wait_until_below_mb(15000)`. Resolve the engine before the `try`,
   compute `skip = is_network(engine)`, and skip `acquire` / `release` / the
   NVML probe when true (roundtable: the probe stalls polling the local GPU
   after every cloud render otherwise). Apply the same branch at the video
   render lease site. A blocking cloud call must not hold the 16 GB.
3. **Cost guard — dispatcher-level, request-context, NOT `assert_usable`.** A
   `reserve_cloud_cost(engine_id, request, episode_id, request_id)` runs AFTER
   the request is assembled and BEFORE `render_*`. Idempotent by `request_id`;
   **cache hits are free**; estimate from a **dated price table**
   (engine→model→unit: image = per-run, video = per-second); unknown/stale price
   ⇒ fail closed. Track a running spent total in **`ledger["billing"]`** so the
   image phase and the later video phase share ONE episode ceiling
   (`OTR_CLOUD_CREDIT_CEILING`, env/config — no Director widget in v1). Over
   ceiling ⇒ skip that beat to the radio floor, LOUD.
4. **Auth probe.** v1: `OTR_COMFY_API_KEY` env ONLY (drop the un-grounded
   "Comfy account" fallback). Missing key ⇒ `EngineUnusable` from
   `assert_usable` so the report says why.
5. **Per-adapter network timeouts:** connect + total wall-clock + poll interval
   + 429/5xx retry (idempotency key; a retry must not double-reserve) + cancel.
   Separate from the local render watchdog. Lazy-import the HTTP client
   (`httpx`/`requests`) inside `render_*` only (cold-import V-12).

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
3. **Dropdown UX (corrected by roundtable):** the COMBO is the full static
   registry (V-6 forbids dynamic widget mutation), so cloud rows **cannot be
   greyed out** — they are always visible and `assert_usable`/the report
   explains unusability at execute time. No decision needed; just document it.
4. **commercial_clean defaults** — set all three `False` until ToS confirmed?
   Recommend: yes, with a dated source URL in adapter metadata.

---

## 10. Roundtable pass 01 — folded corrections (grounded)

GPT-5.5, Gemini 3.1 Pro, DeepSeek-v4-pro reviewed this plan against the real
dispatcher/registries. Architecture validated; the following were corrected
(full log: `roundtable_wiring/pass01_judgment.md`):

1. `render_image(self, request, prepared=None)` — two-arg, matches
   `_inprocess_gen_fn` (§2a).
2. `cloud_ltx2` declared on `text_prompt` (fits all 5 video roles), uses
   `init_image` opportunistically (§0, §2b).
3. Cost guard moved OUT of `assert_usable` to a dispatcher-level idempotent
   `reserve_cloud_cost`; cache hits free; episode total in `ledger["billing"]`
   (§4.3).
4. Lease-skip must ALSO skip the unconditional post-gen `wait_until_below_mb`
   NVML probe (§4.2).
5. Image cost = per-run; only video is per-second (the image request has no
   duration) (§4.3).
6. ONE network marker: `is_network = True` class attribute, read at both lease
   sites; no CAPABILITIES/validator change (§4.1).
7. Image adapter returns in-memory uint8 numpy when possible (avoids the disk
   round-trip); path otherwise; never a torch tensor (§2a).
8. Auth = `OTR_COMFY_API_KEY` env only in v1 (§4.4).
9. `other_beats_video_model = cloud_kling_avatar` is per-beat partial —
   character beats run, scene/background fail closed LOUD (§0).
10. Guarded `__init__.py` imports must log LOUD on failure + a test asserts the
    three rows register with the flag off (§7).
11. Dropdown rows are static-visible, not disable-able (§9.3).

Recommended defaults so the coder isn't blocked: episode ceiling ≈ $5.00
(env `OTR_CLOUD_CREDIT_CEILING`), over-ceiling ⇒ skip beat to radio floor LOUD;
commercial_clean = False for all three until ToS confirmed.

Still gating (verify-at-build): the §5 invocation-seam S0 spike (how to call a
Partner node + key + output discovery), the `MotionEngineBase` return contract,
and confirming the video Director builds its COMBO from the registry.
