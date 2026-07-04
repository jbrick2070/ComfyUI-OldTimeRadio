# Cloud-auth wiring (Bug A) + video-not-moving (Bug B) -- FIX PLAN

Two root causes found live on the operator's logged-in Comfy Desktop (:8000, 2,399 credits).
Goal: make the cloud engines (ideo/seedream/recraft/flux_pro/nano_banana image + word_razzle/
kling/seedance video) actually authenticate + render on the logged-in Desktop, and make the
heavy video engines' MOTION reach the shipped final. Wan = PARKED (local ckpt missing).

Invariants (must not break): NO fallbacks / dropdown-only defaults; workflow-JSON same-change
rule (BUG-LOCAL-097 positional widgets); audio spine byte-identical (mux LAST); single resident
heavy <= 14.5 GB; UTF-8 no BOM; SFW; suite + Bug Bible + B7 + push per green chunk; prod/main GATED.

---

## BUG A -- cloud auth fails EVEN WHEN LOGGED IN (fix first; unblocks all cloud)

### Evidence (live, Desktop run, prompt executed 478s)
`otr_image_gen_dispatcher.dispatch_images -> eng_cloud_image.render_image ->
invoke_partner_node (cloud_media_invoke.py:578) -> get_or_create_session ->
cloud_media_backend.resolve_auth (:149)` raised:
`CloudMediaError: cloud media: auth -- no credentials: set OTR_COMFY_API_KEY, or run with a
logged-in Comfy account (hidden inputs api_key_comfy_org / auth_token_comfy_org)`.

### Root cause (grounded)
ComfyUI injects the Comfy Cloud credentials ONLY into nodes that DECLARE the hidden inputs
`api_key_comfy_org: "API_KEY_COMFY_ORG"` + `auth_token_comfy_org: "AUTH_TOKEN_COMFY_ORG"` in
their `INPUT_TYPES()["hidden"]`. The pinned PARTNER node classes declare them -- but OTR does
NOT place partner nodes in the graph; it invokes them PROGRAMMATICALLY via
`invoke_partner_node`. The OTR nodes that trigger those invokes -- `OTR_ImageGenDispatcher`
(images) and `OTR_VideoRenderBatch` (video) -- do NOT declare the two hidden inputs, so the
logged-in server never injects the credentials into them, and `resolve_auth` finds nothing
(env `OTR_COMFY_API_KEY` also unset). Grep confirms the hidden-auth tokens appear only in
`cloud_media_invoke.py` / `cloud_media_backend.py` (+ the writer's OpenRouter path), never in
the dispatch nodes.

### Fix (proposed -- panel to harden the threading seam)
1. Add to `OTR_ImageGenDispatcher.INPUT_TYPES()["hidden"]` and
   `OTR_VideoRenderBatch.INPUT_TYPES()["hidden"]`:
   `"api_key_comfy_org": "API_KEY_COMFY_ORG"`, `"auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG"`
   (+ `"unique_id": "UNIQUE_ID"` / `"prompt": "PROMPT"` if the bridge needs the prompt id;
   `bind_prompt_id()` already exists for headless).
2. Capture the two injected values in each node's execute/dispatch fn and hand them to the
   invoke bridge -- via a contextvar the backend reads at `resolve_auth` time (mirror the
   existing `bind_prompt_id()` contextvar seam in cloud_media_invoke), NOT a new parameter on
   every adapter. resolve_auth precedence stays: explicit OTR_COMFY_API_KEY env > injected
   hidden auth > raise.
3. NO workflow-JSON change expected (hidden inputs are auto-injected, never wired/widgets) --
   VERIFY V-11 (no new visible widgets) holds after the schema change; if the validator counts
   hidden inputs, update the pin/audit in the SAME change.
4. Tests: unit -- a node exposing the hidden inputs threads them to resolve_auth (monkeypatch);
   resolve_auth prefers env then hidden then raises. Live -- RESTART Desktop, re-run
   `scripts/_otr_cloud_desktop_probe.py cloud_seedream_2 still_flat 30`; expect the Seedream
   still to mint + credits to decrement.
5. Restart required: Python module cache -- the Desktop app must be relaunched to load the code.

### Open questions for the panel
- Does the ComfyUI version on this box inject `API_KEY_COMFY_ORG`/`AUTH_TOKEN_COMFY_ORG` by
  TYPE-name in hidden, and is that the exact seam the partner nodes use? (confirm vs the pinned
  rows in partner_nodes.yaml, which record `auth_hidden_present`.)
- Is `OTR_VideoRenderBatch` the node that runs the cloud VIDEO invoke, or does another node
  (director / assembler) own it? Ground the exact node whose execute calls invoke_partner_node
  for video.
- Contextvar vs threading through the session table: which is the clean seam given the
  backend-owned loop thread?

---

## BUG B -- heavy video engines' MOTION does not reach the final (separate)

### Evidence
Operator: humo/ltx finals show STILL frames. Desktop log shows the LEGACY procgen video path
shipping: `[Video] Starting render: 54.6s audio -> 1366 frames @ 25fps (1920x1080)` ...
`[Video] Encoder: NVIDIA h264_nvenc` ... `[Video] Credits music: ...` ...
`signal_lost_*_silent_procgen_blended_final.mp4`. That is the old `video_engine.py` HUD +
rolling-credits + scopes over the scene STILL. The per-beat MOTION platform
(`OTR_VideoRenderBatch`: ltx/humo motion clips) is not the base layer of the shipped final.
Overnight heavy legs DID burn GPU (ltx 99% 15 min) -> motion clips WERE rendered.

### Hypotheses (panel to ground against the real graph + compositor)
- The canonical `otr_scifi_16gb_full.json` still routes the final through the legacy
  `video_engine` HUD/credits node instead of (or blended over) the per-beat motion clips.
- OR the per-beat motion clips render to the episode dir but the compositor/EpisodeAssembler
  blend uses the scene still, not the clips, as the base.

### Fix (direction -- confirm before coding)
Trace ONE episode end-to-end: per-beat manifest (are motion `.mp4`s present in
`otr/episodes/<ep>/`?) -> the compositor/blend node -> final mux. Make the motion clips the
BASE layer; procgen (HUD/credits/scopes) becomes an OPTIONAL overlay, not the whole picture.
Any node/wiring change goes IN otr_scifi_16gb_full.json in the SAME change (hard rule 0).
Keep the audio spine byte-identical (mux LAST). Do NOT start coding until the trace confirms
which of the two hypotheses is real.

### Sequencing
Bug A first (small, unblocks all cloud + the operator's credits). Bug B second (bigger; needs
the episode-dir trace). Each ships as its own green chunk (suite + Bug Bible + B7 + push).
