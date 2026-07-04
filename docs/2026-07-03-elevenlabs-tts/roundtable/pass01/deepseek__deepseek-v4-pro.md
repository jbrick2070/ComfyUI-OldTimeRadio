<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: **yes-with-fixes** — The model-selection choices are consistent with the grounded facts, but the plan omits how the ElevenLabs voice-id will flow from the curated bank into the `ELEVENLABS_VOICE` input of the TTS node, does not specify the new `runtime` value for cloud engines, and makes untested assumptions about the voice-bank schema.

MUST-FIX BEFORE BUILD:
1. **[A2 / Shared Contract]** The plan maps each ElevenLabs voice to a bank entry with `ref_path=voice_id`, but the TTS node requires a typed `ELEVENLABS_VOICE` input (normally produced by the `cloud_elevenlabs_voice_selector` AUX node). It must be decided how a voice_id string from the bank becomes that typed input: (a) by including the voice selector in the workflow and feeding it the voice_id (how?), (b) by direct construction of the `ELEVENLABS_VOICE` object inside the engine dispatch, or (c) by modifying the node to accept a string. Without this, the casting integration is underspecified.

2. **[Grounded facts / Integration surface]** The plan acknowledges that no `cloud` runtime exists in `audio_engine_profiles.yaml` but does not propose a value (e.g., `cloud_partner`) or sketch how the dispatch layer will route a profile with that runtime to `invoke_partner_node`. This is a missing concept that must be resolved before any profile can be built.

3. **[A2 / assumption]** The claim that an ElevenLabs entry can place the voice_id in `ref_path` and omit `ref_sha256` assumes that `voice_bank_entry_schema.json` allows a non‑file path and a missing hash. Verify against the actual schema; if it enforces a file path or requires a checksum, the bank format must be extended (e.g., a `cloud_identity` field) or the schema relaxed — otherwise the build will break.

SHOULD-FIX:
4. **[A4 / Per‑line delivery]** The plan correctly notes only `stability` is exposed, but it does not propose how the per‑line delivery vector (emotion, style) will be mapped solely to `stability` + `seed`. Even at the model‑selection stage, a short mapping table (e.g., “angry → low stability 0.3‑0.5”) would ensure the later implementation can honor the existing delivery profiles.

5. **[B3 / Length handling]** “Must NOT re‑open the just‑ripped credits‑music loop” is vague. Define what “ripped” means (the existing stable_audio_3 open‑close pair) and at which stage trimming occurs; otherwise this requirement is uncheckable.

6. **[Shared Contract / scope creep]** “`test_audio_byte_identical` stays green through the mux” is an implementation test, not a model‑selection decision. Move it to the wiring round to avoid pre‑committing test expectations that may change.

OPTIONAL / NICE‑TO‑HAVE:
- Once the V3 dynamic expansion of `model` is known, provide a mapping from the COMFY_DYNAMICCOMBO_V3 options to the engine profile’s `default_params.model` so the tier can be selected declaratively.

CUT THESE (scope / over‑engineering):
- **Every node/widget/wiring change lands in `workflows/otr_scifi_16gb_full.json`** — this is a detailed implementation directive; drop it from the model‑selection pass.

- **`canonicalize_audio` analog** — mentioning it is sufficient; do not enumerate “SR/format” at this stage.

- **Explicit reference to “billing JSONL”** — the reuse of the cloud backend already implies it; cut to avoid prescribing backend internals.

Mark [ASSUMPTION]:
- [ASSUMPTION] The `cloud_media_invoke.invoke_partner_node` backend already supports async audio node classes (`ElevenLabsTextToSpeech`, `SoniloTextToMusic`) and handles hidden auth, budget, and the AUDIO return type.
- [ASSUMPTION] The voice‑bank entry schema can be extended to accept a non‑file `ref_path` and missing `ref_sha256`; if not, a new field must be added.
- [ASSUMPTION] CastLock’s `assign_voice_for_slot` and `announcer_voice_ref` will work with `engine="elevenlabs"` entries that have no real reference clip, provided the entry carries the metadata (gender, timbre, roles, age_band).
- [ASSUMPTION] The COMFY_DYNAMICCOMBO_V3 for `model` will expand at pin time to a stable list of tier identifiers that can be used as engine‑profile defaults.