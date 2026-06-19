# OTR Cast Voice-Engine Selection -- Decision to Harden

## The decision
Pick the **1-2 voice (TTS) engines** OTR should standardize on for its **cast
(character voices)** that best deliver, in priority order:
1. **Voice variety** -- many distinct, castable character voices.
2. **Best interaction with the cast** -- each character gets a *distinct,
   deterministic* voice; supports per-line *emotion/delivery* so scenes play.
3. **Best fit for OTR's writing + coding/wiring integration** -- cleanly slots
   into the existing casting spine and per-line render interface with the least
   new, fragile code.
Constraints: prefer commercial-clean licensing; fit the single-resident-heavy
**14.5 GB** VRAM ceiling (voice is small, but the 8 GB distribution tier matters);
100% local/offline; Blackwell RTX 5080 (torch 2.10 / cu130) main venv.

## OTR context the choice must respect (the wiring reality)
- **Render interface:** voice engines implement a `per_line` adapter:
  `generate_voice(text, voice_ref, delivery_vector, seed) -> {waveform,
  sample_rate}`; `begin_episode(meta)` optional. Engines declare `roles`
  (`char_voice` / `announcer_voice` / `music`), `default_roles`,
  `commercial_clean`, `requires_voice_ref`, and a `voice_ref_field`
  (`voice_ref_id` / `voice_preset` / `wav_path`).
- **Casting spine = `OTR_CastLock`:** two assignment mechanisms today:
  (a) **preset engines** (bark): `_assign_bark_voices` replays a deterministic,
  cast-seed-keyed picker and stamps a distinct `voice_preset` per `char_id`
  (uniqueness asserted by `_assert_unique_bark_voices`).
  (b) **clone engines** (indextts2/chatterbox/dia): `cast_voice_policy=
  auto_registry` -> `_otr_voice_bank.assign_voice_for_slot(role="char_voice",
  engine=..., char_id, gender, timbre, age_band, episode_seed, ...)` picks a
  per-character voice from the bank with no reuse.
- **Per-line emotion:** `_otr_delivery_vector.py` derives a deterministic 8-dim
  emotion vector per line (text + scene tension); *expressive* engines
  (indextts2/chatterbox) consume it, others ignore it.
- **Sidecar/dep isolation (V-12):** engines with dep pins that fight the Blackwell
  venv run in their OWN isolated venv as a subprocess worker (indextts2 already
  does: torch 2.8/cu128 worker driven over line-delimited JSON).
- **Voice bank:** CC0 reference WAVs live at
  `C:\ComfyUI-Models\TTS\refs\indextts2\vz_*.wav`; the operator just curated a
  named cast there (ms_dee, mr_derno, mr_spacey, mr_buck, mr_jeffrey USA/UK/UK-exp).

## Candidate engines (grounded against the repo)
| engine | role(s) today | variety mechanism | clone? | emotion/delivery | VRAM | license | wiring cost |
|---|---|---|---|---|---|---|---|
| **indextts2** (default char) | char_voice | zero-shot clone of any ref WAV (unbounded cast via the vz_* bank) | yes (`voice_ref_id`->ref wav) | per-line delivery vector + emo_vector/emo_alpha | ~4-6 GB, isolated venv worker | bilibili **non-commercial** | already wired (bank + worker) |
| **chatterbox** | char_voice, announcer_voice | zero-shot clone of any ref WAV | yes (`requires_voice_ref`) | expressive (delivery vector) | ~2-3 GB, sidecar | **MIT** | wired; re-smoke needed |
| **dia** | char_voice | zero-shot clone | yes | dialogue-oriented | ~3-4 GB, sidecar | **Apache-2.0** | wired; re-smoke needed |
| **bark** | char_voice | fixed preset pool (v2/en_speaker_0..9) | no | none (stochastic) | ~1.5 GB, in-venv | Suno (unconfirmed) | wired; runaway-length fixed 2026-06-18 |
| **kokoro** | announcer_voice only | fixed preset pool (~dozens am_/af_/bm_/bf_) | no | none | **<1 GB**, in-venv (CPU-capable) | **Apache-2.0** | char_voice NOT wired (needs roles + char pool + CastLock assigner) |
| **Qwen3-TTS** (not installed) | -- | Base=zero-shot clone; CustomVoice/VoiceDesign=instruct-emotion fixed/designed | yes (Base) | instruct-emotion only on CustomVoice/VoiceDesign, NOT the clone path | ~7 GB standup (isolated venv) | **Apache-2.0** | new isolated venv + qwen-tts lib + Blackwell torch risk |

## Starting recommendation (the position for the panel to attack)
**Primary: keep `indextts2` as the cast engine** -- it already gives unbounded
variety (clone any of the curated vz_* refs incl. the operator's own voice),
consumes the per-line delivery vector, and is fully wired. Its only weakness is
the **non-commercial license**.
**Secondary (commercial-clean cloning): promote `chatterbox`** (MIT, ~2-3 GB,
zero-shot clone, expressive) as the commercial-safe + 8 GB-tier cast engine, so
OTR is not license-locked to indextts2.
**Add `kokoro` as a low-VRAM, commercial-clean PRESET cast option** (Apache,
<1 GB) for the 8 GB floor and crowd/bit-part characters -- requires the
roles+pool+CastLock-assigner work.
**Defer `Qwen3-TTS`** unless a quick-taste A/B beats indextts2 on the operator's
own ref -- its emotion knob does not apply to the clone path, and it is a 7 GB
isolated-venv standup with Blackwell torch risk; low marginal value over
indextts2 for cloning.
**Demote `bark`** to last-resort fallback only (no per-character variety, no
delivery vector, license unconfirmed).

## Open questions for the panel
1. For *cast variety + interaction*, is clone-any-ref (indextts2/chatterbox) the
   right primary, or do preset engines (kokoro pool) materially help (bit parts,
   determinism, VRAM)? Recommend the 1-2 final picks.
2. **kokoro-as-char_voice wiring:** is mirroring `_assign_bark_voices` (a
   deterministic per-character kokoro-preset assigner + a gendered char pool +
   roles+=char_voice) the right minimal design, or is there a cleaner unification
   with the bank's `assign_voice_for_slot` so preset and clone engines share ONE
   assignment path?
3. Is the non-commercial indextts2 license an acceptable risk for the cast
   default, or should the commercial-clean pair (chatterbox + kokoro) be primary?
4. Coding/wiring: smallest, least-fragile change set to land the chosen 1-2 +
   kokoro, honoring determinism (seed-keyed), uniqueness-per-character, the
   per-line interface, and dep isolation -- without destabilizing the frozen
   audio spine / byte-identical golden.
5. Best writing integration: how should the writer/CastLock map character
   archetype + gender + scene emotion onto each engine's controls (clone ref vs
   preset vs delivery vector) for the most believable cast?

## Invariants (a "fix" that breaks one is rejected)
Single resident heavy <= 14.5 GB; 100% local/offline; determinism (seed-keyed,
per-character unique); per-line interface unchanged; dep isolation (V-12); frozen
audio spine + `test_audio_byte_identical` stays green; UTF-8/no BOM; SFW.
