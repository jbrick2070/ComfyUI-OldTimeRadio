# Cloud-audio implementation — code-scan REVIEW PROMPT (paste-ready)

Hand this to any repo-reading agent to review the ACTUAL code once (or while) the
cloud ElevenLabs TTS + Sonilo music lanes are built. It scans the real files and
reports whether the implementation matches the plan and reintroduces none of the
identified build-breakers.

- Run via codex: `codex exec --sandbox read-only -m gpt-5.5 -` then paste below.
- Run via Claude Code: `claude -p` then paste below.
- Run via the kibitz skill (all local agents): `/kibitz` on BUILD_PLAN.md.

---

You are a read-only code reviewer for the OTR ComfyUI custom-node pack
(`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch
`v2.0-alpha`). GROUND EVERY CLAIM against the REAL files with your own reads — do
not trust this prompt or the plan; cite `file:line` for each finding. The design
contract is `docs/2026-07-03-elevenlabs-tts/BUILD_PLAN.md`. Your job: verify the
cloud ElevenLabs voice + Sonilo music implementation matches it and breaks no
invariant. Read the plan first, then the code.

INVARIANTS THAT MUST STILL HOLD (fail the review if any is broken):
- Audio spine frozen: `test_audio_byte_identical` still passes; the master mix +
  mux-LAST path is unchanged; cloud is dropdown-opt-in so defaults never change.
- No fallback / fail-loud: a missing ElevenLabs key/voice/quota LOUD-stops; there
  is NO silent swap to a local engine anywhere on a cloud-selected line/cue.
- Determinism: casting keys on `OTR_CAST_SEED`; a curated checked-in voice pool
  reproduces; the request is deterministic + ledger-hashed (not provider
  byte-identity).
- Append-only positional widgets in `workflows/otr_scifi_16gb_full.json`.

CHECK EACH CONTRACT AGAINST THE CODE (report CONFIRMED / VIOLATED / MISSING with
file:line):
1. runtime `cloud`: is `"cloud"` in `_VALID_RUNTIMES` and are the new profile
   fields declared on `EngineProfile` (it is `extra="forbid"`)? Do real cloud
   ADAPTERS (`elevenlabs`, `cloud_sonilo_music`) exist whose
   `generate_voice`/`generate_clip` call `invoke_partner_node` with the REAL
   pinned rows `cloud_elevenlabs_tts` / `cloud_sonilo_music`?
   (`nodes/_otr_engine_profiles.py:35/63-95`, `nodes/_otr_shared/cloud_media_invoke.py`)
2. CAPABILITIES parity: are matching rows added so
   `set(CAPABILITIES)==set(_REGISTRY)` holds
   (`tests/test_capability_profiles.py:213`,
   `nodes/_otr_audio_engines/registry.py:184-207`), landed in the SAME COMMIT as
   registration?
3. Selectability: is `elevenlabs` APPENDED (not inserted) to
   `_LEGACY_FIRST_ENGINES` char_voice+announcer_voice and the music engine to
   music (`_otr_engine_profiles.py:42-51`, index 0 unchanged)? Is a new cloud
   bank id in `_VOICE_BANKS` (`cast_lock.py:39`) allowed ONLY on the elevenlabs
   profile (`_resolve_char_engine` walk `:657-676`)?
4. Voice identity: does the elevenlabs adapter set `requires_voice_ref=False`,
   NO `missing_ref_fallback`, `voice_ref_field="provider_voice_id"`? Confirm the
   bark fallback path (`_otr_voice_node_common.py:472-552`) is UNREACHABLE for
   cloud lines. Is `provider_voice_id` threaded through schema + `VoiceBankEntry`
   dataclass + `_entry_from_dict` + CastLock `_stamp` (`cast_lock.py:650-654`) +
   the durable stamp + the admission gate?
5. Announcer: is it pinned by the ADAPTER via `announcer_voice_ref("elevenlabs")`
   in `begin_episode` (`_otr_voice_node_common.py:383-385`), with an
   `"announcer_voice"`-role manifest entry — NOT via CastLock (which is
   kokoro-hardcoded, `cast_lock.py:41`)?
6. Admission gate: is there a cloud gate AFTER CastLock and BEFORE any cloud
   invoke that fails LOUD on missing auth/quota, any cloud-selected line with no
   `provider_voice_id`, or budget exhausted? (`_resolve_character_voices_fail_soft`
   `cast_lock.py:187/386` never raises — the gate cannot rely on it.)
7. Budget: do the adapters pass a nonzero PER-LINE/PER-CUE `estimated_usd`
   (ElevenLabs chars*$0.24/1K; Sonilo duration*$0.15/60s)? Confirm it is per-line
   scale (not episode-total per line). (`cloud_media_invoke.py:605/:623`)
8. canonicalize_audio: is `cloud_media_canonical.py:127` implemented (not the
   `_not_built_yet` stub) with `LOUDNESS_REFERENCE_SOURCE` resolved to the local
   lane's real reference, 44.1kHz, stereo policy, tolerance/padding? Is the
   elevenlabs YAML row `sample_rate: 44100`?
9. Music: ONE `music` cloud profile for all cues; `meta.music_engine` durably
   stamped and read by `OTR_CreditsRoll` (`otr_credits_roll.py:161-207`);
   `stable_audio_3` at index 0 of the `StableAudioTheme` fallback tuple
   (`stable_audio_theme.py:36`)?
10. Conformance test: is `_engine_by_node_key()`
    (`tests/test_cloud_partner_conformance.py:50-59`) extended to the audio
    registry, and are xfails removed ONLY in the sprint that registers each
    adapter (elevenlabs S1, sonilo S5) — never before?
11. Workflow JSON: does S6 select via widget VALUES only, with the MASTER engine
    widgets LEFT at local defaults (`test_capability_profiles.py:173-202` asserts
    master==profile)? Confirm NO real ElevenLabs/Sonilo partner graph nodes were
    added (they would bypass `invoke_partner_node`).
12. Acceptance harness: is the S7 cloud run PROFILE-LESS (or a dedicated cloud
    profile), since `config/profiles/16gb_full.json:17-22` + `widget_mapping.json`
    would otherwise stamp the widgets back to local engines? Is `OTR_COMFY_API_KEY`
    required/documented for headless?

ALSO scan for: any NEW silent local fallback; any positional-widget insert (must
be append-only); any test the change breaks that the plan doesn't mention; any
adapter method whose signature does not match the base protocol
(`nodes/_otr_audio_engines/base.py`).

DELIVER: VERDICT (ready-to-merge / merge-with-fixes / not-ready), then numbered
MUST-FIX and SHOULD-FIX, each with exact `file:line` evidence and the precise
change. If everything checks out, say so plainly and list only the genuine
verify-at-build items. Do not edit any files. Be specific, not thorough-looking.
