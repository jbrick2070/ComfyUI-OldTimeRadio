# S0 Invocation-Seam Spike + Build-Ready Sprint

Date: 2026-06-14
Status: PLAN + SPIKE SPEC (planner window — no production code here). The on-box
spike script is a coder-window task (§3); this doc resolves the seam by desk-
grounding ComfyUI's real `comfy_api_nodes` and pins what the spike must confirm.
Hardened: roundtable (GPT-5.5 + Gemini 3.1 Pro + DeepSeek-v4-pro, grounded) —
2 yes-with-fixes, 1 no-on-rigor; fixes folded + summarized in §7; see
`roundtable_s0/pass01_judgment.md`.
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
- Fallback only (roundtable: NOT equivalent): injecting `api_key_comfy_org` into
  the prompt's `extra_data` does NOT feed a direct util-client call — pursue it
  only if the explicit-key route proves fragile, and have the spike verify the
  exact path before relying on it.

The spike's #1 job is to confirm WHICH auth argument name the installed build's
`util.client` accepts and that an explicit key works headless.

**Output handling (roundtable: keep torch out of the network path):**
- Image (`cloud_flux_pro`): `download_url_to_bytesio` → `PIL.Image` → **uint8
  numpy `(H,W,3)`** (or save a `.png` and return the path) — NOT
  `download_url_to_image_tensor` (that needlessly pulls torch into a plain HTTP
  download). `render_image` asserts the array is `(H,W,3)` uint8 with nonzero
  dims before returning (`_coerce_pixels` does not validate dtype/shape).
- Video (`cloud_ltx2`, `cloud_kling_avatar`): `download_url_to_bytesio` → write to
  a temp file → flush/fsync → **atomic rename** to the final `.mp4` → verify
  nonzero size → return the path through `render_clip` → `canonicalize` (strip
  any provider audio; `has_audio=False`).

## 3. The S0 spike (coder-window task — HARD STOP gate before adapters)

Run with the ComfyUI venv against the installed Comfy Desktop build. **Gate: the
script aborts unless `OTR_RUN_LIVE_CLOUD_SPIKE=1` and prints the planned billed
calls first.** Two parts (roundtable: a standalone script alone does NOT prove
the real context):

**Part A — import/signature probe (standalone script, throwaway):**
1. Read `OTR_COMFY_API_KEY` from env.
2. `import comfy_api_nodes.util.client` (+ the relevant `apis/*` model); pin the
   exact `sync_op`/`poll_op` signatures, the auth argument name, and the
   upload/download helper signatures. If they are async, document the exact sync
   wrapper that does NOT clash with Comfy's event loop.
3. Call `sync_op` for one Flux Pro image with an explicit key (no hidden inputs);
   `download_url_to_bytesio` → save a PNG.

**Part B — real-context proof (in-graph, the part that actually de-risks):**
4. Call the SAME util-client code from inside a ComfyUI `/prompt` execution (a
   throwaway debug node or OTR hook) AND from a plain `threading.Thread` with no
   event loop — to mirror `PromptExecutor`. This is what proves the headless
   executor-thread seam, not the standalone script.
5. Repeat once with `poll_op` for a 2-second LTX-2 clip (submit→poll→download →
   `.mp4`).

**Spike deliverable = a checked-in NOTE** (`S0_RESULTS.md`) pinning, for the
installed build: module path(s), `sync_op`/`poll_op` signatures + auth arg name,
the async-wrapper method, upload/download helper signatures, the result type,
whether the response exposes a billed-cost field, and the cost source.
**SECRETS HYGIENE:** redact the API key, auth headers, signed URLs, task ids, and
local user paths — the key is NEVER logged or put in an error message. The
adapters are coded against THOSE pinned facts. **If no auth path works headless,
STOP** — blocked until headless auth is solved (raise with the operator).

Cost: 1 image (~$0.04) + 1 tiny clip (~$0.10). Trivial; run live (gated).

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
  - **Auth probe (roundtable):** cold `assert_usable` = `OTR_ENABLE_CLOUD` on +
    `OTR_COMFY_API_KEY` non-empty + `importlib.util.find_spec("comfy_api_nodes")`
    present (catches key-set-but-package-missing without importing/executing).
    Live key validation happens at smoke; render handles 401/403 fail-closed.
  - **Cost guard (roundtable):** `reserve_cloud_cost(...)` runs **per object /
    per clip** AFTER cache-hit + `assert_usable`, BEFORE the first billed call;
    commit on success; **release on ANY failure** (auth / upload / submit /
    poll-timeout / download / canonicalize). The deterministic gate uses the
    **dated price-table ESTIMATE** (unknown/stale ⇒ fail closed); recording the
    real billed cost (if the response exposes it) is OPTIONAL telemetry via a
    `gen_fn -> (result, meta)` channel — a SHOULD, not required for v1.
    `ledger["billing"]` schema per the wiring plan §4.3.
  - **Error contract (roundtable):** define adapter exceptions for auth /
    timeout / HTTP / rate-limit, all mapped to the existing fail-closed path
    (dispatcher warns + skips → radio floor). The key never appears in an error.
  - Tests: cold-import, role_compat (the three land in the right roles incl.
    `cloud_ltx2` in `background_abstract`), lease-skip (test FAILS if
    `_lease.acquire` runs for a network engine; cloud video `prepare()` returns
    `lease=None`), guarded-import-logs-LOUD + rows present with flag off, cost
    reserve→commit→**release-on-failure**, return-array shape/dtype assert.
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

---

## 7. Roundtable — folded corrections (grounded)

GPT-5.5, Gemini 3.1 Pro, DeepSeek-v4-pro reviewed this spec; 2 yes-with-fixes,
1 no-on-rigor. Seam validated. Full log: `roundtable_s0/pass01_judgment.md`.

1. **Spike now has two parts** — a standalone import/signature probe AND an
   in-graph `/prompt` + `threading.Thread` proof, because only the latter proves
   the headless executor-thread seam (§3).
2. **Torch kept out of the image path** — `download_url_to_bytesio` + PIL →
   numpy, not `download_url_to_image_tensor` (§2).
3. **`assert_usable` also checks `comfy_api_nodes` is importable** via
   `find_spec` (key-set-but-package-missing) (§5 S1).
4. **Cost reserve is per-object/clip, releases on ANY failure**; gate uses the
   price-table estimate; observed billed cost is optional telemetry (§5 S1).
5. **Error-handling contract** (auth/timeout/HTTP/rate-limit → fail-closed) (§5).
6. **Async wrapper** for `sync_op`/`poll_op` documented by the spike if needed
   (§3).
7. **Secrets hygiene** — `S0_RESULTS.md` redacts key/URLs/task-ids/paths; key
   never logged (§3).
8. **Spike gated** behind `OTR_RUN_LIVE_CLOUD_SPIKE=1` + prints planned spend
   (§3); **atomic video write** + nonzero check (§2); **return-array shape/dtype
   assert** (§2).
9. **`extra_data` auth demoted** to a fallback to verify-in-spike, not an equal
   path (§2).
10. **`cloud_ltx2.required_inputs = ("text_prompt",)`** reaffirmed so
    `background_abstract` is not excluded (§5 S3).

Deferred (panel): separate `/proxy/` gate (unnecessary once util signatures
pinned); `commercial_clean` table polish (stays S5).
