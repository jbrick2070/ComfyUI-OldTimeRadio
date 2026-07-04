# Kibitz r2 (coding) — judgment + hardened deltas

Panel: codex (read-only sandbox) + Claude anchor/judge. Antigravity BENCHED
(no review produced; empty log, hung — consistent with the prior 22-min hang).
No cloud spend (local agents).

## ACCEPTED — codex findings CONFIRMED against real files (spot-verified)
- **K1 `runtime: cloud` needs model + dispatch changes, not just YAML
  (CONFIRMED).** `_otr_engine_profiles.py:35` `_VALID_RUNTIMES={"in_graph",
  "oop_venv"}`; `:67` `EngineProfile` is `extra="forbid"`; `:95` the validator
  RAISES on an unknown runtime. And dispatch ignores `profile.runtime` — voice
  goes through `adapter.generate_voice()` (`_otr_voice_node_common.py:354/:559`),
  music through `stable_audio_theme.py:203/:224`. FIX: register real cloud
  ADAPTERS (`elevenlabs`, `cloud_sonilo_music`) whose `generate_voice` /
  `generate_clip` call `invoke_partner_node`, add `"cloud"` to `_VALID_RUNTIMES`
  and declare the new fields on `EngineProfile`. (Corrects BUILD_PLAN C1's
  "resolver routes" wording.)
- **K2 `partner_row` must be a REAL pinned key (CONFIRMED).** `cloud_elevenlabs`
  is not a row; real keys are `cloud_elevenlabs_tts` / `_flash` /
  `_voice_selector`. `invoke_partner_node` rejects unknown keys
  (`cloud_media_invoke.py:603/:620`). Use `cloud_elevenlabs_tts`.
- **K3 `provider_voice_id` is dropped through the whole chain (CONFIRMED).**
  Schema allows extras but `VoiceBankEntry` has no field (`_otr_voice_bank.py:77`),
  `_entry_from_dict()` copies only known fields (`:158/:166`), CastLock `_stamp()`
  persists only voice_ref_id/voice_engine/commercial_clean (`cast_lock.py:650-654`).
  FIX: thread `provider_voice_id` through schema + dataclass + loader + `_stamp`
  + durable cast stamp + admission gate.
- **K4 custom validator has no oneOf/if/then (CONFIRMED).** `_validate_entry()`
  (`_otr_voice_bank.py:121`) checks only required/type/minLength/minItems, and
  `_entry_from_dict()` indexes `ref_path`/`ref_sha256` unconditionally
  (`:166/:167`). FIX: non-empty cloud SENTINEL (`ref_path="cloud:elevenlabs/
  <voice_id>"`, `ref_sha256="cloud"`) + skip disk/sha in the cloud path — do NOT
  try JSON-Schema conditionals. (Corrects C2's "relax minLength".)
- **K5 canonicalize_audio is a NotImplementedError stub (CONFIRMED).**
  `cloud_media_canonical.py:127` -> `_not_built_yet("audio","S2")`;
  `LOUDNESS_REFERENCE_SOURCE="UNRESOLVED"` (`:68`). This campaign IS that S2:
  build `canonicalize_audio` (WAV 44.1k, stereo policy, loudness matched to the
  LOCAL lane's real reference) BEFORE any adapter returns audio. First
  prerequisite of S1.
- **K6 budget cap is INERT unless adapters pass estimates (CONFIRMED-by-cite).**
  `invoke_partner_node` defaults `estimated_usd=0.0` and reserves the passed
  estimate (`cloud_media_invoke.py:603/:605/:623`); backend cap exists
  (`cloud_media_backend.py:110/:287/:292`). FIX: cost estimators — ElevenLabs
  = chars*($0.24/1K), Sonilo = duration*($0.15/60s) — passed as nonzero
  `estimated_usd`. Also resolve auth: hidden auth injects from `session.auth`
  (`:363/:379`) so confirm these audio adapters receive Comfy hidden auth or
  require `OTR_COMFY_API_KEY`.
- **K7 conformance tests currently XFAIL these exact rows (CONFIRMED-by-cite).**
  `tests/test_cloud_partner_conformance.py:28-33` `KNOWN_UNADAPTERED` xfails
  `cloud_elevenlabs_tts/_flash` + `cloud_sonilo_music`; `_engine_by_node_key()`
  scans only image/video (`:24/:53`). Add the audio registry + remove those
  xfails in the SAME sprint that adds the adapters.

## ACCEPTED (should-fix)
- **K8** `StableAudioTheme` hard fallback tuple omits `stable_audio_3`
  (`stable_audio_theme.py:36`); include it + the cloud music engine once
  registered. Minor.
- **K9** add a ledger request-hash test (text/prompt hash, resolved model,
  provider_voice_id, seed, duration, partner_row) — the practical determinism
  contract (matches C7).

## CUTS (accepted)
- One ElevenLabs adapter/profile only; tier via resolved `model` combo default
  (no separate `_flash` engine).
- No JSON-Schema cloud branch (use sentinel + skip logic).

## Net effect on BUILD_PLAN
Sprint order tightened: **S0 now = canonicalize_audio (K5) + EngineProfile cloud
runtime/fields (K1) + provider_voice_id threading (K3) + real partner_row (K2) +
V3 combo expansion + conformance-xfail removal (K7)** — all pure-code/no-render,
gating everything else. Budget estimators (K6) fold into S1/S5 adapters. The
fail-loud admission gate (C4) stays S3. No invariant broken.
