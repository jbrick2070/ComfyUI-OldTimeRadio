# Voice-casting architecture: hardening plan (grounded)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro (spend $0.29). All four
returned "not build-ready" -- the design is the right SHAPE (engine-parameterized
caster + per-engine `voice_ref_field`) but has real gaps. Every item below is
grounded against the real files by Claude.

## DONE now (immediate, low-risk, proven)

**Canonical workflow CastLock (node 80) flipped to `auto_registry` +
`allow_voice_reuse=True`.** Headless proof: this assigns an indextts2 reference
to 6/6 characters of a 3M/3F cast (the 3 females share `vz_caro_davy`) -> zero
bark. `preserve_ledger` left it 4/6 (2 females -> bark). No code change; loads on
workflow reload. Trade-off: same-gender characters SHARE a voice until the bank
grows (see "Reference-bank expansion").

## MUST-FIX backlog (grounded; review before I implement -- these touch C7)

1. **`voice_ref_path` is never stamped by CastLock.** `_stamp` (`cast_lock.py:390`)
   writes `voice_ref_id`/`voice_engine`/`commercial_clean` only, but the dispatch
   reads `voice_ref_field="voice_ref_path"` for clip engines -- so even
   `auto_registry` works only because the render-time resolver re-derives the path
   from `voice_ref_id`. Fix: in `_auto_registry`, after `_stamp`, set
   `entry["voice_ref_path"] = ref.ref_path` for clip engines. Makes it robust and
   lets the fragile resolver be retired. (DeepSeek, GPT, grounded.)

2. **`commercial_clean` is wrong for IndexTTS2.** The adapter is
   `commercial_clean=False` (`eng_indextts2.py:47`, bilibili non-commercial model
   license) but the CC0 bank refs are `commercial_clean=true`, and the per-line
   request uses `profile.commercial_clean`. **Effective cleanliness = engine AND
   ref** -- IndexTTS2 output is NOT commercially clean regardless of the CC0 clip.
   This matters for shipping your films commercially. Fix: stamp
   `engine_commercial_clean`, `ref_commercial_clean`, `commercial_clean_effective`
   and make any release gate read the effective value. (GPT, Grok, grounded.)

3. **Gender is re-derived at render with no guarantee it survives onto the cast
   row.** `_resolve_clone_ref_path` reads `cast.get("gender")`; we observed it
   return None -> bark even with refs on disk. `auto_registry` (now default)
   sidesteps this by pre-stamping `voice_ref_id`, but the `preserve_ledger` path
   stays fragile. Fix: CastLock validates/stamps `gender` on every char row; the
   resolver logs a structured warning when it falls back so bark is never silent.
   (All four, grounded.)

4. **Kokoro announcer preflight can trip on a missing voice.**
   `ANNOUNCER_VOICE_POOL = ["bm_george","bm_fable","bf_emma","bf_lily"]`
   (`eng_kokoro.py:26`) but only `bm_george` is installed; `begin_episode`
   random-picks then `os.path.exists`-checks (`:85`) -- ~75% chance of hitting a
   missing file. Fix: pick only from installed voices (or always fall back to
   `bm_george`); preflight the voice actually used, after the cast `voice_ref_id`
   is resolved. (GPT, grounded.)

5. **Sample-rate normalization only covers the bark fallback.** The shipped
   `resample_audio` fixes the fallback clip, but primary-adapter clips are
   appended unchanged at the profile `sr`. A future engine whose real output rate
   != its profile `sr` would still mix. Fix: resample EVERY clip to the batch `sr`
   after `generate_voice` (or fail at profile-validation time on a rate mismatch).
   Generalizes the just-shipped fix. (GPT, Grok, Gemini, grounded.)

6. **The model-agnostic dispatch still hard-codes engine names.**
   `_OTR_CLONE_ENGINES = ("indextts2","chatterbox")` + membership branches in
   `_render_per_line` -- a future clip engine gets no ref resolution or fallback
   and violates the stated "no per-engine ladders" invariant. Fix: replace with
   adapter metadata (`requires_voice_ref=True`, `voice_ref_kind="wav_path"`,
   `missing_ref_fallback="bark"`) and branch on that. (GPT, grounded.)

## SHOULD-FIX (future-proofing)

- **Loud reuse/fallback visibility.** `assign_voice_for_slot` / `_auto_registry`
  should emit a structured WARNING per reused or fallen-back row (`char_id`,
  `gender`, `engine`, `voice_ref_id`, reason=`bank_exhausted`), and
  `_render_per_line` should log per-line bark fallbacks (which char, why) instead
  of one aggregate line. So "distinctness degraded" is never silent.
- **Gender model v2 (don't break C7).** Add an `"unspecified"` ladder tier +
  treat gender as a weighted attribute rather than a hard floor, but ONLY under a
  bumped `CASTING_POLICY_VERSION` so existing seeds/assignments are byte-stable.
- **Canonical "add a voice engine" checklist** (GPT's, grounded): adapter
  (`name/roles/interface/sample_rate/commercial_clean/requires_flag`) + engine
  profile (matching rate/params/license/token checks) + `voice_ref_field` +
  `voice_ref_kind` + bank entries (clip engines) + deterministic
  `generate_voice(text, ref, vec, seed)` + local-only `load`/preflight +
  best-effort `unload` + missing-ref policy. Publish it in docs.
- **Bank-health report** (CLI): counts by engine/role/gender, installed-vs-missing
  WAVs, commercial-clean status, and "minimum refs needed for this cast".

## Reference-bank expansion (your sourcing list, license-filtered)

Goal: more indextts2 refs, **especially female** (only `vz_caro_davy` today), so
characters get distinct voices instead of sharing. Your rules adopted: generic
timbre only, never impersonate real/living people, 10-20s clean mono WAV,
`ffmpeg -i in.mp3 -ac 1 -ar 24000 ref.wav`.

License triage of your list:
- **USE (CC0 / public-domain, generic):** LJ Speech (PD, clean female English --
  best single female grab), M-AILABS (PD LibriVox/Gutenberg, male+female folders),
  Mozilla Common Voice (CC0, has gender/age/accent metadata -> easy balancing),
  Kokoro JP set (PD, Japanese tone). LibriVox audiobook readers are PD audio and
  consistent with the existing bank (`vz_bill_boerst` etc. are LibriVox) -- fine
  as GENERIC timbre, not labelled/marketed as the person.
- **AVOID:** LibriSpeech (CC-BY, needs attribution -- not pure PD), Wikimedia
  human-voice category (mixed licenses), and the named-person clips (Freud /
  Armstrong / JFK) -- those are identifiable real individuals; even with PD audio
  it reads as impersonation, against your own rule.

Per new voice the pipeline is: download (you, or I ask per file) -> trim to a
clean 10-20s mono span -> `ffmpeg -ac 1` -> sha256 -> drop under
`C:\\ComfyUI-Models\\TTS\\refs\\indextts2\\` -> add a bank entry:
```
{ "voice_ref_id": "vz_<short>", "engine": "indextts2", "gender": "female",
  "timbre": ["warm","alto"], "roles": ["char_voice"], "age_band": "adult",
  "ref_path": "models/TTS/refs/indextts2/vz_<short>.wav",
  "ref_sha256": "<sha>", "commercial_clean": true }
```
Recommended first grab: 2-3 more FEMALE voices (kills the reuse) + 1-2 male for
timbre variety. I can build a `scripts/_otr_dl_indextts2_refs.py` that fetches a
curated CC0 set, trims, hashes, and wires the bank -- but downloading is gated on
your go-ahead (I won't pull files autonomously).

## Invariants guarded
C7 determinism (seeded casting; new behavior only under a bumped policy version);
PD1 always-renders; frozen legacy Bark path byte-identical when selected;
model-agnostic dispatch (engines self-describe); C-5 import-time clean; 16 GB VRAM.

## What I implemented vs deferred
Implemented now: the workflow `auto_registry`+reuse flip (gets you 100% index).
Everything in MUST-FIX/SHOULD-FIX is grounded and ready, but each touches casting
behavior + needs a ComfyUI restart and a live render to validate -- so they're
staged for your go-ahead, not applied blind while you're away.
