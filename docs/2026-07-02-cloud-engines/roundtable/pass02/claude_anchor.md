# R2 Claude anchor review (coding plan / implementability)

Grounding: same four files as R1 + registry line-151 confirmation + A-ship
executor-thread lesson (repo memory) + live install facts.

VERDICT: yes-with-fixes. The architecture is now honest about surfaces and
gating; the implementability gaps are concurrency/async bridging, cache
key ordering, and an untested assumption that the image lane is really
"lowest risk."

MUST-FIX BEFORE BUILD:
1. [5] In-process invocation of bundled partner-node classes: modern
   comfy_api_nodes classes are ASYNC and expect auth from the execution
   context. Our adapters run inside node execution (executor thread).
   Fix: specify the sync-bridge (dedicated event-loop thread owned by the
   shared backend; adapters submit coroutines and block with timeout) and
   wire long polls into watchdog heartbeats (verify-at-build #9). Do not
   leave "invoke in-process" as a one-liner -- this is where the build
   stalls first.
2. [5] Schema pinning via /object_info is the WRONG capture point when we
   are in the same process: hidden inputs may be filtered from
   /object_info. Fix: pin schemas by importing the node classes and
   reading INPUT_TYPES()/RETURN_TYPES directly; serialize pinned schemas
   to a fixtures file checked into tests.
3. [2] Billing-cache key includes audio-slice hash, but slices derive from
   upstream cloud TTS output which is NONDETERMINISTIC -> every re-run of
   audio invalidates all downstream video cache entries unless audio is
   cache-hit first. Fix: document the cache dependency DAG (voice/music
   cache -> slice hash -> video cache) and make S2 acceptance include a
   re-run producing 100% CACHED audio + video.
4. [3] Per-line loudness normalization may fight the existing master-mix
   loudness handling. verify: where loudness is applied today (mixer vs
   per-line). Fix: canonicalization normalizes to the SAME reference the
   local lane produces at that stage, not a new LUFS convention invented
   in this plan.
5. [7-S1] "S1 STILLS lowest risk" is an [ASSUMPTION]: the image registry
   (C1) exists per the video registry docstring, but whether
   OTR_ImageGenDispatcher resolves engines through it (vs a hardcoded
   FLUX path) is unverified from what was read. Fix: R3 wiring round must
   confirm the image dispatch path; if the dispatcher bypasses the
   registry, S1 inherits registry-wiring work and the sprint risk order
   should be revisited (voice may be the true lowest-risk lane).

SHOULD-FIX:
1. [2] Cost accounting: partner responses may not return actual USD. Fix:
   ledger stamps ESTIMATED (pricing table x units) vs ACTUAL (when
   response metadata provides it); episode cost report labels which.
2. [7-S0] Name the test list: budget-matrix units, cache-key stability,
   canonicalization golden files, auth-broker fail-closed (missing auth),
   registration flag on/off (rows absent when off), pinned-schema drift
   test (import fails loudly when a partner node signature changes).
3. [5] Fixture pattern for the offline suite: canned provider responses
   keyed by pinned schema version; no live network in tests, ever.
4. [4a] Preset->stock-voice table: state where it lives (curated table in
   the adapter file, audited like COMFY_LLM_MODELS pinned catalog).

OPTIONAL:
- Per-provider client caching (Kling token handshakes) inside the backend.

CUT: none new.
