# R3 Claude anchor review (wiring / integration / sequencing)

New grounding this round: otr_image_gen_dispatcher.py line 38 imports
`_otr_image_engines.registry` (S1 assumption CONFIRMED -- dispatcher is
registry-driven); OTR_LedgerScriptWriter.py:2219-2220 declares
`"hidden": {"auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG", ...}` and
run() accepts `auth_token_comfy_org=None` (verify-item #10 RESOLVED:
exact constant names proven in-repo; additive hidden inputs are the
established pattern and do not touch widgets_values).

VERDICT: yes-with-fixes. The spec is coherent; the wiring gaps are
session lifetime across nodes, reuse of the SHIPPED fallback machinery,
and sprint-scoping of the node surgery.

MUST-FIX BEFORE BUILD:
1. [3] SESSION LIFETIME: CloudMediaSession must span MULTIPLE node
   executions (3a fires, then 3c, then video batch -- separate execute
   calls, one episode). Unspecified = each node builds its own session =
   budget/semaphores/ledger fragment. Fix: backend session table keyed
   by prompt_id (+ episode_id once known); lazy-init at first cloud call
   from whichever node runs first (all cloud-capable nodes declare the
   hidden auth inputs); torn down on the assembler done signal or
   prompt end; budget accumulator lives in the table entry, not the node.
2. [3/8] FALLBACK MUST REUSE THE SHIPPED CHAIN MACHINERY:
   `nodes/_otr_shared/fallback.py` (humo -> latentsync -> still_kenburns,
   LOUD restamp) is the production in-render fallback. The cloud
   fallback resolver EXTENDS it (cloud row -> cloud row -> local row ->
   abort) rather than introducing a second system. Two fallback systems
   in one render path is how silent divergence happens.
3. [7] REACTIVITY ENFORCEMENT POINT: the matrix must be wired into the
   existing queue-time gate (`OTR_ShotLock` calls `assert_usable`; video
   registry delegates role fit to role_compat). A mute_only row picked
   for a talking beat must fail THERE with a named error -- not at
   render time after money is spent. Fix: reactivity is role_compat
   metadata consumed by the same engine_fits_role path; matrix tests via
   descriptor_for_engine.
4. [8-S0] NODE-SURGERY SCOPING: hidden-input declarations belong to each
   LANE's sprint (S1 dispatcher, S2 audio nodes, S3 video batch), NOT
   all-at-once in S0. S0's smokes need NO node surgery: smoke #1/#2
   invoke partner nodes directly through the backend with INJECTED auth
   (env/server config), proving the transport before any node changes.
   Also: `otr_save_to_episode_workspace.py:145` shows nodes may ALREADY
   have a hidden dict (PROMPT/EXTRA_PNGINFO) -- additions MERGE, never
   replace.
5. [4] CACHE DIR PATH DISCIPLINE: `otr\cache\cloud_media\` must derive
   from the SAME output-base resolution as `otr\episodes\` (repo rule:
   otr\ may resolve to ComfyUI's real output base). Fix: one path helper,
   config key for override, .gitignore entry, and the cache is EXCLUDED
   from obs_publish sweeps.
6. [adapters] IMPORT LINE = LIVENESS: each new adapter file requires its
   one import line in the package __init__ (the registry pattern);
   forgetting it is dead code (repo 2026-06-13 lesson). Fix: per-lane
   checklist item + a registration test asserting every curated row id
   is registered when the flag is on.

SHOULD-FIX:
1. [ledger] Billing JSONL vs production ledger: define linkage
   (request_id stamped in BOTH; production ledger remains the artistic/
   render source of truth; billing ledger is financial). No dual-write
   of the same fact.
2. [8-S0] Operator setup step named in S0 acceptance: logged-in Comfy
   Desktop OR OTR_COMFY_API_KEY on the render box, before smokes.
3. [6] Canonicalizer golden files live under tests/fixtures/cloud/ with
   pinned schema versions (offline suite needs them from day one).
4. [8-S4] Profile DEFAULT-OVERRIDE map location: profiles module beside
   the enable-set derivation, one source of truth for role -> row id.

OPTIONAL: doctor CLI first (S0), it de-risks every later wiring question.

CUT: none new.
