# S0 Invocation-Seam Spike + Build-Ready Sprint

Date: 2026-06-14
Status: PLAN + SPIKE SPEC (planner window — no production code here). The on-box
spike script is a coder-window task (§3); this doc resolves the seam by desk-
grounding ComfyUI's real `comfy_api_nodes` and pins what the spike must confirm.
Companion: `WIRING_PLAN_cloud_engines.md` (the engine wiring) + its pass01/02
judgment logs.

---

## 1. What the cloud API nodes actually are (grounded)

From ComfyUI `comfy_api_nodes` (commit `fed4ac03`, indexed 2026-04-14 — recent):

- Nodes are **API clients**, not local torch nodes. They authenticate, upload
  assets, poll a remote task, download the result.
- They use the **V3 API** `define_schema` pattern from `comfy_api.latest`.
- Request orchestration lives in **`comfy_api_nodes/util/client.py`**:
  - `sync_op` — immediate request (good for images: Flux Pro);
  - `poll_op` — submit + poll a long-running task (video: LTX-2, Kling Avatar);
  - `sync_op_raw` — lower-level.
- Asset helpers: `util/upload_helpers.py`
  (`upload_image_to_comfyapi`, `upload_video_to_comfyapi`) and
  `util/download_helpers.py`
  (`download_url_to_image_tensor`, `download_url_to_bytesio`), plus
  `util/conversions.py` (`tensor_to_base64_string`, `bytesio_to_image_tensor`).
- Requests route through a **`/proxy/` endpoint** on `api.comfy.org` that handles
  credit billing — so calling the util client IS the billed path.
- Provider request/response models live in `comfy_api_nodes/apis/<provider>.py`
  (e.g. `apis/kling.py`); node classes in `comfy_api_nodes/nodes_<provider>.py`.

## 2. The seam decision (S0 resolved) + the one big gotcha

**Chosen seam:** OTR's cloud adapters call the **`comfy_api_nodes.util.client`**
layer (`sync_op` for image, `poll_op` for video) directly inside `render_*`,
using `upload_helpers` for `init_image`/`audio_ref` and `download_helpers` for
the result — **NOT** by instantiating the V3 node classes (their execute
signature + hidden-auth injection is the brittle part) and **NOT** raw provider
HTTP (loses Comfy's unified billing). This mirrors OTR's existing pattern of
driving Comfy internals in-process via `wrapper_bridge`.

**THE GOTCHA that de-risks the whole feature (grounded in ComfyUI issues
#13222, #8344, #11481):** the API nodes normally get their key from **hidden
inputs** `auth_token_comfy_org` / `api_key_comfy_org`, which the SERVER injects
from the logged-in web-UI session. **When a workflow runs headless via `/prompt`
— exactly how OTR runs — those hidden values are `None`, and the call fails with
"Unauthorized: Please login first to use this node."** Therefore OTR MUST pass
the key explicitly:

- Primary: pass the Comfy API key from **`OTR_COMFY_API_KEY`** straight into the
  `util.client` call's auth argument (the util layer accepts an explicit
  auth/`comfy_api_key`), bypassing the hidden-input path.
- Equivalent alternative: inject `api_key_comfy_org` into the prompt's
  `extra_data` at `/prompt` submission time (OTR controls its own submission).

The spike's #1 job is to confirm WHICH auth argument name the installed build's
`util.client` accepts and that an explicit key works headless.

**Output handling:**
- Image (`cloud_flux_pro`): `download_url_to_image_tensor` → IMAGE tensor →
  convert to **uint8 numpy** in the adapter (`(t[0]*255).clamp(0,255).byte()
  .cpu().numpy()`), return that (per the wiring plan §2a — no `wrapper_bridge`).
- Video (`cloud_ltx2`, `cloud_kling_avatar`): `download_url_to_bytesio` → write a
  local `.mp4`, return the path through `render_clip` → `canonicalize` (strip any
  provider audio; `has_audio=False`).

## 3. The S0 spike (coder-window task — HARD STOP gate before adapters)

A throwaway script (delete after; do NOT commit) run with the ComfyUI venv,
against the user's installed Comfy Desktop build, that proves the seam end to
end for the CHEAPEST image model:

1. Read `OTR_COMFY_API_KEY` from env.
2. `import comfy_api_nodes.util.client` (+ the relevant `apis/*` model) and call
   `sync_op` for one Flux Pro image with an explicit key (no hidden inputs).
3. Confirm: the call authenticates headless, returns a result URL, and
   `download_url_to_image_tensor` yields a tensor; convert + save a PNG; print
   its path and the credits charged.
4. Repeat once with `poll_op` for a 2-second LTX-2 clip to prove the
   submit→poll→download path + a video file out.

**Spike deliverable = a checked-in NOTE** (`S0_RESULTS.md`) pinning, for the
installed build: the exact module path(s), the `sync_op`/`poll_op` signatures,
the auth argument name, the upload/download helper signatures, the result type,
and the per-call credit cost observed. The adapters are coded against THOSE
pinned facts. **If none of the auth paths work headless, STOP** — the feature is
blocked until login/headless auth is solved (raise with the operator).

Cost: 1 image (~$0.04) + 1 tiny clip (~$0.10). Trivial; run live.

## 4. Risks the spike must close (grounded)

1. **Headless auth** (the §2 gotcha) — the single biggest risk. Prove explicit
   key works before anything else.
2. **API churn** — `util/client.py` (current) vs the older `apis/client.py`
   `SynchronousOperation`. Comfy Desktop pins a specific build; the spike reads
   the INSTALLED signatures, the plan does not assume them.
3. **Executor-thread requirement** — OTR already runs its render in ComfyUI's
   executor thread (the `OTR_VideoRenderBatch`/dispatch nodes via `/prompt`); the
   cloud calls inherit that. Confirm the util client is safe to call from there
   (it is network I/O, not CUDA, so it should be — verify no event-loop clash;
   `sync_op`/`poll_op` may be async under the hood → may need `asyncio.run` or a
   sync wrapper).
4. **Cold-import** — `comfy_api_nodes.util.client` pulls comfy/torch; import it
   **lazily inside `render_*`**, never at adapter module scope (V-12).
5. **Proxy/billing** — calls must go through `/proxy/` (api.comfy.org) so credits
   are billed; a raw provider call would bypass billing. The util layer does
   this; confirm.

## 5. Build-ready sprint (supersedes WIRING_PLAN §8 ordering)

- **S0 — invocation-seam spike (HARD STOP).** §3 above. Output `S0_RESULTS.md`.
  Nothing else starts until this is green.
- **S1 — platform glue, no engine yet.**
  - `is_network = True` marker; image dispatcher lease-skip (restructure the
    `dispatch_images` lease bracket) + skip the post-gen NVML probe;
    `MotionEngineBase.prepare()` override path for network video engines.
  - `OTR_COMFY_API_KEY` auth probe (dep-free) surfaced via `assert_usable`.
  - `reserve_cloud_cost(...)` + `ledger["billing"]` schema + dated price table
    (reserve→commit/release; unknown/stale ⇒ fail closed).
  - Tests: cold-import, role_compat (the three engines land in the right roles),
    lease-skip (image + video), guarded-import-logs-LOUD + rows present with flag
    off, cost reserve/refund.
- **S2 — `cloud_flux_pro`** (prompt→uint8, simplest) end to end on the seam +
  cost guard + auth, live smoke (small budget).
- **S3 — `cloud_ltx2`** (text_prompt + opportunistic init_image → mp4 via
  `poll_op`), all video roles incl. `background_abstract`.
- **S4 — `cloud_kling_avatar`** (audio_ref + init_image → silent mp4; WAV-probe
  duration), announcer + character roles.
- **S5 — polish:** commercial_clean dated table; README/wizard note on
  `OTR_ENABLE_CLOUD` + `OTR_COMFY_API_KEY` + the ceiling; full live smoke on the
  operator's credits.

Per CLAUDE.md: run the suite + Bug Bible after every change; commit+push per
green chunk; the saved `workflows/otr_scifi_16gb_full.json` needs NO edit for v1
(engines auto-appear in the COMBO; default-OFF keeps saved selections local).

## 6. Open operator decisions (unchanged from the wiring plan)

Kling on music beats (no — talking-face only), episode ceiling default (~$5,
skip-to-floor), commercial_clean = False until ToS confirmed. None block S0.
