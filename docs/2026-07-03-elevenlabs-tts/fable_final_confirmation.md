# Fable final confirmation — 100% code-ready gate

VERDICT: **GO** (after 2 fixes folded, now in BUILD_PLAN.md).

Fable re-verified all seven prior cross-cutting fixes against the real files —
all correctly integrated (F-MF1 profile-revert / F-MF2 CAPABILITIES parity /
F-MF3 three tuples / F-MF4 adapter-pinned announcer / F-MF5 no silent bark /
F-MF6 xfail order / F-MF7 dropdown-values). Hidden-auth question RESOLVED: auth
is bridge-level (`cloud_media_invoke.py:363-384` from `session.auth`;
`resolve_auth` `cloud_media_backend.py:135-154`, `OTR_COMFY_API_KEY` precedence
#1) — no per-adapter work; headless S7 needs `OTR_COMFY_API_KEY` set. Also
confirmed `generate_clip(prompt, duration_s, seed)` shape, adapter metadata attrs
on the base (`_otr_audio_engines/base.py:48-60`), `stamp_durable` section copy,
credits-roll reads.

Two remaining fixes (now folded):
- **FIX-1** (S1): `_engine_by_node_key()` (`test_cloud_partner_conformance.py:50-59`)
  scans only image/video, so removing the xfail without extending it to the audio
  registry raises `test_billed_row_has_adapter_or_explicit_xfail` at `:88-90`.
  Extend the map + give adapters `node_key`/`_partner_inputs`, same commit as the
  CAPABILITIES row.
- **FIX-2** (S6): `test_capability_profiles.py:173-202` asserts master engine
  values == profile slot_overrides; S6 must NOT flip the master's engine widgets.
  Cloud pick is per-run (harness/operator), master stays local defaults.

Verify-at-build (runtime only, not defects): ElevenLabs voice-id vs display-label;
`LOUDNESS_REFERENCE_SOURCE` existence; V3 re-pin row-drift diff.
