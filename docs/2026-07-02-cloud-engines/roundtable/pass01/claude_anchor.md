# R1 Claude anchor review (code-grounded panelist)

Grounding actually read: nodes/_otr_model_catalog.py (head),
nodes/_otr_comfy_backend.py (head), nodes/_otr_audio_engines/registry.py
(head), nodes/_otr_video_engines/registry.py (head), live list_api_nodes
dump (214 nodes, digested), Comfy Cloud template search results,
workflow dropdown screenshots from operator. Claims below labeled
CONFIRMED / MISREAD / UNVERIFIABLE against those.

VERDICT: yes-with-fixes. The arc (provider axis + thin adapters + one
transport + cloud profile) matches the shipped LLM-lane pattern
[CONFIRMED in _otr_model_catalog.py / _otr_comfy_backend.py], but the
cheapest lanes rest on the least-verified transport and the
audio-reactive requirement is stated as blanket while the CHEAP video
pick violates it.

MUST-FIX BEFORE BUILD:
1. [3d] Blanket "AUDIO-REACTIVE REQUIRED" contradicts the CHEAP row
   (wan_i2v "mute b-roll"). Fix: replace blanket requirement with a
   per-role reactivity matrix: announcer/talking beats REQUIRE
   audio_ref or lipsync_overlay family; music/b-roll beats MAY be mute
   I2V (existing still/parallax families are already non-reactive
   [CONFIRMED: families list in video registry]). Keeps operator intent
   (the radio TALKS) without pricing every beat like a talking beat.
2. [3a+4] CHEAP voice row (chatterbox on Comfy Cloud) depends on
   transport (B), flagged possibly-vapor in Q4. A plan whose cheapest
   lane may not exist is not build-ready. Fix: state IN 3a the named
   fallback row (ElevenLabs flash/turbo cheap tier) + expected cost
   delta, fail-closed switch, so S2 cannot stall on transport (B).
3. [4] "partner API nodes via the api.comfy.org proxy (primary)" is an
   [ASSUMPTION] generalized from the CHAT partner node only
   [CONFIRMED for chat: DEFAULT_CHAT_PATH in _otr_comfy_backend.py;
   UNVERIFIABLE for media nodes]. Media partner nodes bundle their own
   client + async task polling in comfy_api_nodes. Fix: adapters INVOKE
   the bundled partner-node classes in-process (they own endpoints,
   polling, retries); our shared backend wraps only auth capture,
   budget guard, cost ledger, and fail-closed classification. Do not
   hand-roll per-provider HTTP.
4. [4] Missing concept: idempotent re-runs / billing cache. A crashed
   or re-run episode must not re-bill every completed asset. Fix: cloud
   asset cache keyed by content hash (model row + prompt + params +
   audio-slice hash) resolving into otr\episodes\<ep>\; transport
   checks cache before submit; ledger marks CACHED vs BILLED.
5. [3a] Voice-clone bank (ElevenLabsInstantVoiceClone) inside the S2
   critical path adds ToS/consent audit + per-voice cost + CastLock
   coupling. Fix: demote to post-S2 enhancement; S2 ships preset ->
   curated stock-voice mapping (CastLock still assigns presets
   [CONFIRMED pattern: bark discrete presets precedent]).

SHOULD-FIX:
1. [3b] "prompt + style continuity" between local stable_audio_3 and
   cloud StabilityTextToAudio is a hypothesis, not a fact (different
   model generations). Mark as verify-at-build listening test.
2. [5-S2] "master WAV byte-stable across mux" is the wrong acceptance
   for a nondeterministic cloud lane. Byte-identical applies to the
   LOCAL default baseline only [CONFIRMED invariant]. Cloud acceptance:
   structural (per-line duration tolerance, loudness lint, mux-LAST
   audio equality between pre-mux master and muxed track).
3. [4/S0] Rate limits / concurrency absent: batch legs will parallel-
   hit providers. One serialization + per-provider rate-limit knob in
   the shared transport, S0.
4. [4] Pre-run cost estimate (rows x beat counts printed before submit)
   in addition to post-hoc ledger -- operator trust + budget guard
   pairing.
5. [2] Pricing verify-at-build should name its source (Comfy partner
   pricing table) and stamp approx_cost per row BEFORE S1 promotion,
   not "before promotion" generically.

OPTIONAL / NICE-TO-HAVE:
- Keep GeminiVideoOmni/Sora2 as mute-I2V generic rows later; their
  native audio stays discarded (mux-LAST invariant), panel may argue.
- SoniloVideoToMusic: no consumer in pipeline; do not list.

CUT THESE:
1. [3a] ElevenLabsTextToDialogue as BEST-2: dialogue synthesis bypasses
   the per-line ledger/caption machinery (captions, per-line delivery
   vectors, per-beat audio slices all assume line-granular audio
   [CONFIRMED: per_line/clip contract in audio memory + registry
   roles]). A whole-conversation blob breaks caption timing. Either
   demote to experiment-flag or replace BEST-2 with a second per-line
   TTS row. This is the strongest cut candidate in the doc.
2. [3e] Meshy rig/animate "reopen ARKit keystone" sentence -- delete
   the aspiration; keeps 3D seam honest (nothing consumes rigs).

ASSUMPTION SURFACE (biggest hidden ones, now explicit):
- A1: Comfy account credit balance is shared across chat + media
  partner nodes (single billing surface). UNVERIFIABLE from repo;
  verify at first live call.
- A2: Partner nodes work from a HEADLESS ComfyUI (auth hidden inputs
  populate without the desktop login UI) -- the whole no-GPU story
  rides on this. Must be S0 smoke test #1.
- A3: Cloud jobs returning within watchdog heartbeat windows (5-min
  stall detector) -- long queue times on partner side could false-kill
  legs; transport must heartbeat while polling.
