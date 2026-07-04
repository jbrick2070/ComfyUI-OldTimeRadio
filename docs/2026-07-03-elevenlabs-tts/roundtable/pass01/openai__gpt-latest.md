<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan calls itself a model-selection pass but leaves core selections open, overclaims determinism/fail-loud behavior against the shown CastLock behavior, and lacks the schema/runtime contract needed for cloud audio to be selectable safely.

MUST-FIX BEFORE BUILD:
1. [Open operator decisions vs A1/A3/A5/B1/B2/Shared contract] The plan has unresolved decisions that are central to model selection: announcer pinned vs shuffled, voice-pool size/licensing, cloning in/out, Sonilo vs Stability as v1 cloud music default, all-three music roles vs open/close only, and unset-key behavior. This contradicts the stated purpose: “hardens the MODEL-SELECTION decisions first.” Concrete fix: close v1 as a minimal decision set before build, e.g. library-only, pinned announcer, Sonilo cloud music candidate/default only if operator promotes cloud, all three music cues or explicitly only open/close, fail-loud on missing ElevenLabs key. Move true experiments to a later pass.

2. [A1/A4 + partner_nodes.yaml] ElevenLabs “default model = multilingual/quality tier” is not actually selectable from the shown facts because `model` is `COMFY_DYNAMICCOMBO_V3` and `combo_options_excluded: true`; the real model options are hidden. The plan also omits required non-model inputs: `output_format`, `apply_text_normalization`, and `language_code`. Concrete fix: make V3 expansion/pin of ElevenLabs `model` and required combo defaults a prerequisite to build; name the exact default model, flash/turbo alternative, output format, text-normalization setting, and language policy. Until then, do not claim a specific ElevenLabs tier/default.

3. [A2 + Shared contract + partner_nodes.yaml] The ElevenLabs voice identity path is conceptually incomplete. A2 says put `voice_id` into `voice_reference_bank.json` as `ref_path` with no disk clip/no sha, while the grounded schema fields include `ref_path` and `ref_sha256`, and the current local meaning is “disk clip.” Also the TTS node requires `voice: ELEVENLABS_VOICE`; the only shown converter is `cloud_elevenlabs_voice_selector` taking a COMBO and returning `ELEVENLABS_VOICE`. Passing a raw `voice_id` through `ref_path` is not a coherent typed contract. Concrete fix: add an explicit cloud voice schema/adaptor contract before build: e.g. `provider_voice_id`, `provider_voice_label`, `engine=elevenlabs`, `ref_sha256=null/"" with schema allowance`, and a resolver that converts bank entries to the `ELEVENLABS_VOICE` input through the selector or verified direct typed value. Verify the selector accepts stable IDs, not only mutable display labels.

4. [Shared contract “No fallbacks / no hidden promotion” + cast_lock.py] The fail-loud contract conflicts with the shown CastLock behavior. `CastLock._resolve_character_voices_fail_soft` explicitly “NEVER raises,” repairs missing voices, and may leave lines “for the node-81 engine fallback.” `_auto_registry` also reports “NOT cast” and continues for some failures. That is not compatible with “missing key / voice / quota = LOUD stop” for cloud voice. Concrete fix: add a cloud-specific preflight/admission gate after CastLock and before cloud render that validates every cloud-selected character/announcer has a resolvable provider voice and required auth/quota; fail hard there. Do not rely on existing CastLock to enforce cloud completeness.

5. [Integration surface + Shared contract] “New runtime value” is acknowledged but not designed. The grounded profiles only allow/comment `runtime: in_graph | oop_venv`; there is no `cloud` runtime today. The plan does not define the profile fields that map an audio engine profile to a partner-node row, provider inputs, billing labels, or canonicalizer. Concrete fix: define the minimal `runtime: cloud` profile contract before build: `partner_row`, provider/provider_id, required param defaults, auth requirement, billing category, output canonicalization target, error policy, and whether the engine is usable for `char_voice`, `announcer_voice`, or `music`.

6. [B1/B2 + audio_engine_profiles.yaml] The music routing story mismatches the current profile model. B2 names cue roles `music_open` / `music_close` / `music_inter`, but the shown `audio_engine_profiles.yaml` has one profile role: `role: music`. `meta.music_engine` is singular. Concrete fix: either keep one `music` engine profile and explicitly map all cue types to it, or introduce separate cue-role profiles/stamps. Also define whether stamps are per episode (`meta.music_engine`) or per cue (`cue.music_engine`, `cue.seed`, `cue.duration`).

7. [A1] “flash/turbo option for cheap soak lanes” contradicts the pricing section and the plan’s own grounded fact that ElevenLabs pricing is flat across model tiers. Concrete fix: rename this to latency/quality soak or low-quality/fast soak; do not describe flash/turbo as cheaper unless a separate verified price source says so.

8. [Shared contract “Determinism” + partner_nodes.yaml] The plan treats `seed_supported: true` as enough for reproducibility. That only proves the node exposes a seed; it does not prove provider byte-identical determinism across time, model revisions, or library changes. Concrete fix: state the weaker contract: deterministic request construction and durable logging, not guaranteed byte-identical cloud audio. Stamp provider row, model string, voice id, seed, duration, prompt hash/text hash, and any provider/version metadata available. Keep `test_audio_byte_identical` scoped to local/default paths unless cloud fixtures are mocked.

SHOULD-FIX:
1. [B1] The Sonilo-vs-Stability argument is too thin for a default decision. “BEST music” is just a note in `partner_nodes.yaml`; price comparison is 60s Sonilo vs one Stability run, not necessarily the actual cue mix. Concrete fix: define expected cue durations and evaluate total episode cost/quality target. If no audition exists, call Sonilo “recommended first candidate,” not settled default.

2. [B3 + Shared contract] Length handling is underdefined. “If provider ignores a short request, OTR trims inside the FROZEN assembler” assumes trimming is always acceptable and ignores too-short returns, silence tails, fades, loop points, and cue boundaries. Concrete fix: define cloud music normalization behavior: min/max duration tolerance, trim/fade policy, pad/retry/fail behavior for short outputs, and where this happens without reopening the credits-music loop.

3. [A3] Announcer “operator env to shuffle” undermines the stated stable show identity and deterministic credit stamping unless fully specified. Concrete fix: for v1, make pinned announcer mandatory; defer shuffling until there is a stamped announcer pool, seed derivation, and credits behavior.

4. [A2] Voice-pool curation is doing too much work without an acceptance criterion. “Small licensed pool” is not enough for the existing scorer, which weighs gender/timbre/role/age. Concrete fix: define minimum pool coverage by gender/age/timbre and the behavior when the pool cannot satisfy unique voices without reuse.

5. [Shared contract] Budget behavior is mentioned but no user-facing budget estimate is part of the selection story. [ASSUMPTION] If operators choose between local/cloud via dropdown, they need a pre-render estimate. Concrete fix: add a required estimate/report path for chars, announcer chars, and music seconds before invocation.

6. [A4] “Confirm at V3 expansion whether the DYNAMICCOMBO hides more knobs” is misframed. The shown pinned node required inputs expose no `similarity_boost`, `style`, or `speed`; V3 combo expansion may reveal model options, not new required input sockets. Concrete fix: separate “dynamic model options” from “node input surface”; do not plan delivery mappings to nonexistent sockets.

7. [Shared contract] `canonicalize_audio analog` is necessary but too vague. Concrete fix: pick the assembler-facing audio contract now: sample rate, channels, dtype/container, loudness/peak normalization ownership, and whether voice/music share one canonicalizer.

8. [Shared contract] The workflow JSON requirement is an implementation constraint, not a model-selection decision, and risks forcing graph churn before the cloud runtime/profile contract is stable. Concrete fix: gate workflow edits until profile/runtime/schema decisions are closed.

OPTIONAL / NICE-TO-HAVE:
- Add an audition matrix for ElevenLabs model tiers and Sonilo/Stability using the same script/cue prompts before promotion.
- Add a “cloud dry-run” mode that resolves voice/model/music choices and estimates cost without invoking partner nodes.
- Add provider terms/licensing stamp fields distinct from local `commercial_clean`.

CUT THESE (scope / over-engineering):
1. [A5] Cut InstantVoiceClone from v1 planning. It is already marked deferred, needs separate consent/licensing/user-flow decisions, and is not needed for library-based cloud TTS.

2. [A3] Cut announcer seed-shuffle/env support from v1. It conflicts with stable show identity and adds determinism/credits complexity without serving the basic cloud TTS goal.

3. [A1] Cut “flash/turbo cheap soak lane” language and any separate cheap-lane behavior. Pricing is flat per `PRICING.md`; keep only one ElevenLabs engine with a model/tier param after V3 expansion.

4. [Shared contract] Cut “fold the `meta.cast_voice_slots` durable-stamp gap” from this model-selection pass unless it is directly required for cloud voice credits. It is a ledger persistence repair, not a model-selection decision.

5. [B1] Cut Stability Audio as a fully wired v1 alternate if Sonilo is chosen as the first cloud music engine. Keeping both in the initial build doubles routing/default/test surface; Stability can remain documented as the next candidate.

6. [Shared contract] Cut mandatory workflow JSON edits from this pass. First close schema/runtime/model defaults; then wire the graph in a later implementation pass.