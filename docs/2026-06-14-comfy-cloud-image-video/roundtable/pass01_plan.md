# Comfy Cloud Image + Video Options for OTR Dropdowns — Research

Date: 2026-06-14
Status: RESEARCH ONLY (no code). Author window: planner.
Hardened: roundtable pass 01 (GPT-5.5 + Gemini 3.1 Pro + DeepSeek-v4-pro,
grounded against the real dispatcher/registries) — see Section 9 and
`roundtable/pass01_judgment.md`.
Purpose: scope what it would mean to add **opt-in cloud image-gen and
video-gen choices** to the OTR per-role dropdowns (announcer visual, music
visual, character, other-beats), so a user can spend ComfyUI credits to run a
generation step in the cloud instead of on their own GPU.

---

## 0. TL;DR / recommendation

- The right integration surface is **ComfyUI native "API / Partner Nodes"**
  (cloud models invoked *from inside your local graph*, billed by Comfy
  credits), **not** full "Comfy Cloud" (the whole workflow hosted remotely).
  API Nodes keep the entire OTR pipeline — ledger, caching, mux, captions —
  local; only the single gen step goes over the wire.
- OTR is **already built for this.** The image and video engines are
  model-agnostic registries (`nodes/_otr_image_engines/registry.py`,
  `nodes/_otr_video_engines/registry.py`). Each engine self-registers and
  declares `roles` / `default_roles` / `commercial_clean` / `requires_flag` /
  `family` / `required_inputs`. The Director nodes build the per-role dropdowns
  from the registry; the Dispatchers `assert_usable` (fail-closed, no silent
  fallback). **A cloud option is just a new adapter row** that calls an API
  node under the hood. No architecture rethink.
- Input/output mapping is clean. OTR already produces exactly what the cloud
  nodes consume: a **text prompt** (image still), a **still init_image**
  (image-to-video), and an **audio reference** (audio-driven talking face).
  Cloud nodes return a file (png / mp4) — which is exactly what the existing
  adapter contract already returns and content-addresses.
- The closest cloud match to each role exists today:
  - Still image (Flux local today) → **Flux Pro / Flux.2 / Nano Banana 2 /
    Luma Photon** cloud nodes.
  - Audio-driven character face (HuMo local today) → **Kling Avatar** and
    **Kling Lip Sync** cloud nodes (audio + portrait → talking video).
  - Scene / b-roll / music visual motion (LTX / Wan local today) → **LTX-2,
    Luma Ray, Kling, Seedance, Pika, MiniMax/Hailuo** cloud nodes.
- The new work is **policy and guardrails, not models**: an API-key /
  credit-presence check, a per-episode **credit cost guard** (mirror the
  shipped OpenRouter cost-guard), per-provider `commercial_clean` flags, and
  one master opt-in flag (default OFF). Everything else the registry already
  does.

---

## 1. Two cloud surfaces — and why we pick API Nodes

| | **ComfyUI API / Partner Nodes** (RECOMMEND) | **Comfy Cloud** (hosted) |
|---|---|---|
| What runs remotely | One generation step (the node call) | The entire workflow |
| Where OTR runs | Local (your 5080), unchanged | Remote container |
| Billing | Comfy **credits**, pay-per-run (211 credits = US$1) | Credits by **GPU-seconds** (0.266 cr/s) + monthly plan |
| Hardware | Your box + provider's cloud for the one model | RTX 6000 Pro, 96 GB VRAM, 30–60 min run cap |
| Fit with OTR | **Native** — drops into the existing registry as one adapter | Would require shipping the whole OTR graph + assets to the cloud |
| Offline story | Local pipeline stays; only the chosen step needs network | Fully online |

**Decision: API / Partner Nodes.** OTR's value (ledger, content-addressed
cache, audio-byte-identical mux, captions, fallback ladder) is in the *local*
superstructure. We want to swap **one model call** out to the cloud per role,
which is exactly what a Partner Node is. Comfy Cloud is the wrong granularity —
it would mean uploading the full model stack + every asset and giving up the
local pipeline.

### How an API Node works (input → output)
- Auth: user signs into a Comfy account and pastes a **Comfy API Key**
  (Settings → User → Sign in → Comfy API Key). Since frontend PR #8041 the key
  can drive Partner Nodes headlessly (matters for OTR's headless API runs).
- Credits: prepaid via Stripe; **only consumed on a successful run**; "ComfyUI
  charges the same as the original API price." No idle/subscription charge for
  API Nodes themselves.
- Execution: the node takes a **prompt** (and optionally a ref/init image,
  audio, aspect ratio), calls the provider, **polls internally**, and on
  completion writes the result to `ComfyUI/output/` and passes an IMAGE/VIDEO
  downstream. From the graph's point of view it is a **blocking node that
  returns a file** — same shape as a local engine, just network-bound.

---

## 2. What's available that maps to our stack (with credit prices)

Prices from docs.comfy.org/tutorials/partner-nodes/pricing (211 credits =
US$1). Figures are current as of 2026-06-14 and will drift — treat as
order-of-magnitude.

### 2a. Image (maps to OTR's local Flux still gen → `flux_still` / `flux_gen1`)

| Cloud model | Credits | ≈ USD | Notes |
|---|---|---|---|
| Luma `photon-flash-1` | 0.57 / run | $0.003 | cheapest; good for bulk scene stills |
| Luma `photon-1` | 2.2 / run | $0.010 | |
| **`flux-dev`** | 5.28 / run | $0.025 | direct cloud sibling of the local Flux dev |
| `flux-2-pro` | 6.33 / run (+3.17/extra MP) | $0.030+ | newest Flux |
| **`flux-pro-1.1`** | 8.44 / run | $0.040 | strong default portrait quality |
| `flux-kontext-pro` | 8.44 / run | $0.040 | **edit/consistency** — good for keeping a character's face across beats |
| `flux-pro-1.1-ultra` | 12.66 / run | $0.060 | |
| `flux-kontext-max` / `flux-2-max` | 16.88 / 14.77 | $0.08 / $0.07 | |
| **Nano Banana 2** (Gemini 3.1 flash image) | 21.31 / run (2K), 32.49 (4K) | $0.10 / $0.15 | best-in-class character consistency / instruction following |
| GPT-Image-1/1.5/2 | token-based ($30–40 / 1M out) | varies | |

### 2b. Audio-driven character face (maps to OTR's local HuMo / latentsync)

This is the special one — OTR drives faces from **audio**. Cloud equivalents:

| Cloud model | Credits | ≈ USD | Notes |
|---|---|---|---|
| **Kling Avatar** (std) | 11.82 / sec | $0.056/s | portrait + audio → talking avatar |
| **Kling Avatar** (pro) | 23.63 / sec | $0.112/s | higher fidelity |
| **Kling Lip Sync** | 14.77 / run | $0.070 | lip-sync an existing clip to audio |
| MiniMax `s2v-01` | 137.15 / run | $0.650 | speech-to-video |
| Kling `v3-omni` (sound) | 23.63–29.54 / sec | — | omni model w/ sound |

> A 6-second talking announcer beat on Kling Avatar std ≈ 71 credits ≈ **$0.34**.

### 2c. Scene / b-roll / motion (maps to OTR's local LTX / Wan i2v + t2v)

| Cloud model | Credits | ≈ USD | Notes |
|---|---|---|---|
| Luma `ray-flash-2` | 0.66 / sec | $0.003/s | cheapest motion; great for "other beats" b-roll |
| Luma `ray-2` | 1.93 / sec | $0.009/s | |
| **LTX-2 fast** (i2v/t2v 1080p) | 8.44 / sec | $0.040/s | **direct cloud sibling of local LTX**; native audio in LTX-2 |
| LTX-2 pro (1080p) | 12.66 / sec | $0.060/s | up to 4K |
| Kling `v2-5-turbo` | 73.85 / run | $0.350 | |
| Kling `v3` (1080p, no sound) | 23.63 / sec | $0.112/s | |
| Seedance 1.0 lite / pro | token-based (~$1.8–2.5/1M tok) | varies | |
| Pika 2.2 i2v (5 s, 1080p) | 94.95 / run | $0.450 | |
| MiniMax Hailuo-02 (6 s, 1080p) | 103.39 / run | $0.490 | |
| Google Veo 3 (with audio) | 337.6 / run | $1.60 | premium; expensive |

**Direct-sibling wins:** local Flux → cloud **Flux Pro/2**; local LTX → cloud
**LTX-2**. These are the lowest-friction adds (same family, same mental model
for the user) and answer the user's "if Comfy has cloud versions of the best
models in our stack" question with a clear yes for Flux and LTX.

---

## 3. Mapping to the dropdown slots

The operator confirmed (2026-06-14 screenshot of `OTR_VideoDirector`): **all
six** model dropdowns should gain a cloud option, plus **one easy "just use
Comfy credits" default**. The exact widgets, with their current local defaults:

| # | Dropdown widget | Current default | Generates |
|---|---|---|---|
| 1 | `announcer_video_model` | flux_still | announcer talking / motion clip |
| 2 | `music_video_model` | flux_still | music-bed / title-card motion |
| 3 | `other_beats_video_model` | flux_still | character + scene b-roll motion |
| 4 | `announcer_image_model` | flux_gen1 | announcer portrait still |
| 5 | `music_image_model` | flux_gen1 | music / open title still |
| 6 | `other_beats_image_model` | flux_gen1 | character + scene stills |

(Characters route through the `other_beats` slots per
`otr_image_gen_dispatcher.py` `_ROLE_TO_IMAGE_SLOT`.) Proposed cloud picks:

| Slot | Local engine today | Suggested cloud option(s) |
|---|---|---|
| `announcer_video_model` | humo / flux_still | **Kling Avatar** (audio-driven talking) / LTX-2 fast |
| `music_video_model` | abstract / ltx / wan | **Luma Ray Flash** or **LTX-2 fast** (cheap motion) |
| `other_beats_video_model` | ltx / wan_i2v / still_parallax | **LTX-2 fast** / **Luma Ray Flash** / Seedance lite |
| `announcer_image_model` | flux_gen1 | **`flux-pro-1.1`** / Nano Banana 2 |
| `music_image_model` | flux_gen1 | **`flux-pro-1.1`** / Luma Photon Flash |
| `other_beats_image_model` | flux_gen1 | **`flux-kontext-pro`** (consistent face) / Luma Photon Flash |

Because the dropdown is registry-driven, each is just an adapter row declaring
which `roles` it serves. A user picks "cloud Flux Pro" on a slot the same way
they pick "humo_1.7B" today.

### 3a. The "easy option" — one `cloud_auto` choice per slot

Per the operator ask ("an easy option using Comfy credits"), each of the six
dropdowns gets a single zero-config entry — call it **`cloud_auto`** — that:

- requires only a signed-in Comfy account + credits (the master flag + auth
  probe gate it; no per-model decision for the user);
- routes to a curated, sensible cloud model for that slot's kind:
  - **image slots →** `flux-pro-1.1` (≈ $0.04/still), the closest cloud sibling
    of the local Flux the user already knows;
  - **motion video slots** (music / other-beats) → **LTX-2 fast**
    (≈ $0.04/sec), the cloud sibling of local LTX;
  - **announcer video slot →** **Kling Avatar std** (≈ $0.056/sec) when an
    `audio_ref` is present (talking announcer), else LTX-2 fast;
- shows an estimated credit cost before it runs and is governed by the
  per-episode cost guard (Section 4).

This gives the user exactly two tiers per slot: pick a **specific** cloud model
(power user) **or** pick the easy auto entry and let OTR choose (easy mode).

**Implementation note (roundtable):** do NOT build one universal `cloud_auto`
that infers its slot — `slot` is not in the dispatch request (only `role` is),
so slot inference is fragile. Instead ship **three per-kind auto engine IDs**,
each hard-bound to its curated model so there is no inference and each is
independently testable:

- `cloud_auto_image` → `flux-pro-1.1` (the image slots);
- `cloud_auto_motion` → LTX-2 fast (music + other-beats video);
- `cloud_auto_talking` → Kling Avatar when `audio_ref` present, else LTX-2 fast
  (the announcer video slot).

Each resolves to its concrete model **before** the cache key is computed, and
the resolved model id is folded into `engine_version` so a future model swap
can't silently reuse a stale cached result. Same adapter pattern as the
power-user rows; just pre-bound.

---

## 4. How it fits the existing architecture (no rethink)

The model-agnostic registries already do 90% of the work:

1. **Adapter rows.** Add cloud engines, e.g. `cloud_flux_pro` (image),
   `cloud_kling_avatar` (video, `family=audio_driven_face`),
   `cloud_ltx2` (video, `family=image_to_video`/`text_to_video`),
   `cloud_luma_ray` (video). Each declares `roles`, `default_roles=()` (never
   a silent default — opt-in only), `commercial_clean=<per provider ToS>`,
   `requires_flag="OTR_ENABLE_CLOUD"`, `required_inputs` (text_prompt /
   init_image / audio_ref as appropriate).
2. **Dropdowns auto-populate.** `OTR_ImageDirector` / `OTR_VideoDirector`
   build the dropdown from the full static registry and filter by
   `role_compat` — the cloud rows appear automatically once registered.
3. **Fail-closed selection.** Dispatchers call `assert_usable`. A cloud
   adapter's `assert_usable` should fail closed when: the master flag is OFF,
   **no API key is present**, or the **credit cost guard** would be exceeded.
   No silent Flux/HuMo substitution (matches the BUG-LOCAL-405 contract
   already enforced in `otr_image_gen_dispatcher.py`).
4. **Output handoff (mostly supported, one coercion caveat).** The image
   dispatcher's `_coerce_pixels` accepts **either** a decoded uint8 numpy array
   **or a `.png` PATH** (the cu128 sidecar path). A cloud node writes a file to
   `ComfyUI/output/` → the adapter returns that path → the dispatcher
   content-addresses it, stamps the ledger, materializes into
   `episodes/<ep>/stills/`. Caveat (roundtable): `_coerce_pixels` raises
   `TypeError` on a raw ComfyUI **IMAGE tensor**, so the cloud adapter must
   return a path or convert tensor→uint8 numpy itself. Video adapters return a
   clip + `canonicalize` exactly as local ones do.
5. **GPU lease — needs a dispatcher change (corrected by roundtable).** Cloud
   gen uses **no local VRAM**, but the lease is NOT skippable per-adapter as
   first written: `dispatch_images` calls `_lease.acquire()` *unconditionally*
   right before `gen_fn(request)`, so a cloud call would hold the local GPU
   lease for the entire network wait. The dispatcher must check the engine's
   `vram_class` (a new network class, or `cpu`/`vram_estimate_mb=0`) and skip
   `acquire` / `release` / the post-gen NVML probe for network engines. Small,
   contained dispatcher edit — not a superstructure rethink, but not "zero
   change." See Section 9.

### New pieces to design (the actual scope)
- **Auth probe.** A dep-free helper that reports whether a Comfy API key is
  configured (env var or Comfy account), surfaced via `assert_usable`.
- **Credit cost guard — NOT in `assert_usable` (corrected by roundtable).**
  The dispatcher calls `_ireg.assert_usable(engine_id, role)` with no request,
  dims, duration, or budget state, so the guard cannot live there. It needs the
  full request context: either build the request before the usability check and
  pass it to a dedicated `estimate_cost(request, policy)` step, or enforce
  inside the adapter's `render_image`/`render_clip`. Keep it **minimal for v1**
  — a per-episode credit ceiling with deterministic reserve/spent accounting
  (idempotent across retries), NOT a full clone of the OpenRouter guard.
  Estimate from a **dated** price table; a stale/unknown price entry fails
  closed (never an unbounded run).
- **`commercial_clean` per provider.** Cloud model ToS vary on commercial use
  and output ownership; set the flag conservatively per provider and surface
  it in the wizard so a commercial-clean render never silently picks a
  non-clean cloud model.
- **Network failure ladder (corrected by roundtable — differs image vs video).**
  `_otr_shared/fallback.py` is real but is wired into the **video** render
  path, not the image dispatcher. `dispatch_images` catches a render exception
  and **skips the object → radio floor**; it does NOT try a local engine.
  - *Images (v1):* accept the existing behavior — a cloud image failure skips
    to the radio floor, LOUD. Do not build a stateful image-fallback loop in v1.
  - *Video:* a cloud failure can degrade through the existing fallback resolver
    (cloud → local → floor), each swap LOUD + ledger-restamped (CW-7). Verify
    the resolver is reachable from the cloud video adapter before relying on it.
  - Every cloud adapter still needs its own timeouts: connect + wall-clock +
    poll interval + 429/5xx retry + cancel.
- **Determinism caveat.** Cloud models are not bit-reproducible across runs
  the way the local seed scheme is; the content-addressed cache still works
  (a cloud result is hashed and reused), but document that re-gen may differ.

---

## 5. Inputs / outputs contract (the user's specific question)

What each cloud node needs vs. what OTR already produces:

| Cloud node input | OTR already produces it? | Source |
|---|---|---|
| text prompt | **Yes** | `OTR_MetaBriefImagePromptGen` / story-brief helpers |
| aspect ratio / w,h | **Yes** | image policy carries `w`/`h` (landscape stills) |
| init / reference image | **Yes** | the still the image dispatcher already mints (init_image for i2v) |
| audio reference (talking face) | **Yes** | the per-line voice render OTR feeds HuMo today (`audio_ref`) |
| seed | Yes (advisory only on cloud) | `resolve_object_seed` |

| Cloud node output | OTR consumes it as | Handling |
|---|---|---|
| PNG file/path | image still | `_coerce_pixels` PATH branch → content-hash → ledger |
| MP4/clip | video clip | video adapter `render_clip` → `canonicalize` → mux |

**Conclusion:** inputs map 1:1 with no new producers; outputs land on code paths
that already exist. The integration is genuinely "input → cloud → file → existing
pipeline."

---

## 6. Risks / open questions (seed the roundtable)

1. **Auth UX in a "newbie/vibe-coder" README world.** OTR's audience targets
   ComfyUI newbies. Pasting an API key + buying credits is a friction step —
   how much hand-holding belongs in the node vs. the wizard vs. the README?
2. **Cost guard granularity.** Per-episode ceiling vs. per-role vs. per-run?
   Where does the estimate live (Director policy vs. dispatcher)? What's the
   default ceiling and the fail-closed behavior (local fallback vs. hard stop)?
3. **commercial_clean truth source.** Per-provider ToS change; do we hardcode
   a conservative table, fetch it, or make it operator-declared?
4. **Async/timeout model.** API nodes block + poll; a slow provider could
   stall a headless soak. Do we need a wall-clock timeout per cloud beat and a
   LOUD degrade, separate from the local watchdog?
5. **Determinism / cache semantics.** Is hashing a non-reproducible cloud
   output the right cache key, or do we key on (prompt, model, params) and
   accept the first result as canonical?
6. **Which models to surface v1.** Recommend a *small* curated set (Flux Pro,
   LTX-2 fast, Kling Avatar, Luma Ray Flash) rather than all 60+ — fewer
   choices, clearer mapping, less ToS surface. Confirm.
7. **Mixed local+cloud episode.** Announcer cloud, characters local — does the
   ledger / fallback / mux handle a per-role mix cleanly? (Believed yes; the
   registry is per-role already, but worth a verification pass.)

---

## 7. Recommended next steps (post-research, NOT in this doc's scope)

1. Roundtable this doc with 3 frontier models on architecture + the
   input/output contract (Section 8).
2. Fold the synthesis into a short go-forward sprint plan: v1 = curated cloud
   set, master flag, auth probe, cost guard, per-provider commercial_clean,
   fallback ladder.
3. Hand the sprint plan to a coder window (this is the planner window; it does
   not write production code).

---

## 8. Roundtable seed questions

Send this doc to GPT, Gemini, and DeepSeek and ask each, independently:

1. Is "API/Partner Nodes as registry adapters" the right architecture for
   per-role opt-in cloud gen, or is there a cleaner seam? Critique the adapter
   boundary.
2. Pressure-test the **input/output contract** (Section 5). What breaks when a
   cloud node is blocking + async + network-failing inside a node-graph render?
3. Where should the **credit cost guard** live and how should it fail closed?
   Per-episode vs per-role budget?
4. Cache semantics for non-deterministic cloud output — hash-the-result vs
   key-on-request? What are the failure modes of each?
5. Minimum viable curated model set for v1 across the four roles — what would
   you cut or add, and why?
6. Biggest risk we're under-weighting (auth UX, ToS/commercial_clean,
   determinism, cost runaway, or something we missed)?

---

## 9. Build constraints from roundtable pass 01 (grounded)

GPT-5.5, Gemini 3.1 Pro, and DeepSeek-v4-pro independently reviewed this doc;
every claim below was verified against the real files. Full log:
`roundtable/pass01_judgment.md`. These are the hard constraints a coder window
must honor — they correct the doc's earlier "zero dispatcher change" optimism.

1. **Dispatcher must skip the GPU lease for network engines.** `dispatch_images`
   acquires the lease unconditionally; add a `vram_class`/network check so a
   cloud call doesn't hold the 16 GB during the network wait.
2. **Cost guard takes request context, not `assert_usable`.** Add an
   `estimate_cost(request, policy)` step (or enforce in `render_*`); minimal
   per-episode ceiling with reserve/spent accounting, idempotent on retry.
3. **Every cloud engine needs a `CAPABILITIES` row** in its registry
   (`vram_class="cpu"`, `vram_estimate_mb=0`, `cpu_ok=True`,
   `requires_sidecar=False`) or `capability_profiles.py` may exclude it.
4. **Image failure → radio floor (no local fallback in v1).** Only the video
   path has the `_otr_shared/fallback.py` resolver; don't claim local image
   fallback. Each cloud adapter still needs connect/wall-clock/poll/retry
   timeouts.
5. **Specify the Partner-Node invocation seam (verify-at-build spike).** Decide
   among: direct provider HTTP call with the Comfy API key / import the
   partner-node class and call it / submit a sub-graph. Define API-key source +
   precedence, polling, output-file discovery, and keep registry import
   cold-import-clean. Confirm partner nodes can run programmatically outside the
   graph executor before committing to the class-import path; direct HTTP is the
   fallback.
6. **Image adapter returns a path or uint8 numpy** — never a raw IMAGE tensor
   (`_coerce_pixels` raises on it).
7. **Easy option = three per-kind auto engine IDs**, not one slot-inferring
   `cloud_auto` (Section 3a). Resolve to a concrete model before the cache key;
   fold the resolved model id into `engine_version`.
8. **commercial_clean = hardcoded dated table, unknown ⇒ False**, with a pinned
   source URL in adapter metadata. No runtime ToS fetching in v1.
9. **Curated v1 set only** — one image, one motion, one talking-face, plus the
   three easy entries. Don't surface all 60+ models.
10. **Registration/import + cold-import test.** Wire cloud modules into the
    existing adapter import path; add a test that `all_engine_names()` lists the
    cloud rows with `OTR_ENABLE_CLOUD` off.
11. **Dropdown UX decision:** cloud rows visible-but-disabled (with help text)
    vs hidden until `OTR_ENABLE_CLOUD=1` — pick one, implement in Director
    validation.

Resolved by grounding (no action): the video `VideoRequest` already carries
`role` + `audio_ref` + `init_image`, so Kling Avatar gets its inputs and the
LTX-2 announcer fallback gets its `init_image`; and the cache already keys on
the request (`request_cache_key`), so the only cache risk is the cloud_auto
resolved-model nuance in #7.

## Sources

- ComfyUI Partner Nodes pricing — https://docs.comfy.org/tutorials/partner-nodes/pricing
- Native API/Partner Nodes intro — https://blog.comfy.org/p/comfyui-native-api-nodes
- API Nodes Wave 2 (model list) — https://comfyui.org/en/comfyui-api-nodes-wave-2
- 62 new API nodes — https://comfyui.org/en/comfy-gets-major-boost-with-new-api-nodes
- Login via Comfy API Key (PR #8041) — https://blog.comfy.org/p/api-nodes-login-via-comfyui-api-key
- Comfy Cloud features + pricing — https://blog.comfy.org/p/comfy-cloud-new-features-and-pricing
- Comfy Cloud pricing — https://comfy.org/cloud/pricing/
- Cloud API Nodes (DeepWiki, internals) — https://deepwiki.com/Comfy-Org/ComfyUI/10-cloud-api-nodes
- Kling Motion Control / partner node docs — https://docs.comfy.org/tutorials/partner-nodes/kling/kling-motion-control
- OTR internals: `nodes/_otr_image_engines/registry.py`, `nodes/_otr_video_engines/registry.py`, `nodes/otr_image_gen_dispatcher.py`
