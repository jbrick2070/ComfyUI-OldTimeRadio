# BUILD SPEC -- ONE VERSIONED "VOICE IDENTITY" FIX (PBUG-20260817-09)

**Status: QA-corrected 2026-08-18. GO for coding.** The QA lane returned a
NO-GO on the first brief and eight corrections; all eight are folded in below
and marked **[QA-n]**. Operator directive is the spec; the corrections make it
buildable.

**NO CHANGELOGS.** Per operator instruction this brief carries no final
PBUG / Bible / README / GO_FORWARD update step. Do not add one.

## THE FIX -- both causes, one version

* **(A) Character-stable engine seed** for the CHARACTER CLONE profiles.
* **(B) IndexTTS2 effective emotion-mass cap `0.4` at the ADAPTER BOUNDARY**,
  so hand-edited / pre-stamped vectors are covered too.
* **(C) Alpha default `0.4`** in `nodes/_otr_audio_engines/eng_indextts2.py` and
  `config/audio_engine_profiles.yaml`.
* **(D) CAP AFTER ALPHA** -- never before; wrong order double-softens delivery.
* **(E) Bump `engine_impl_version`** so stale audio is invalidated honestly.
* **Do NOT** change the global delivery table. **Do NOT** add a workflow widget.
  The canonical graph is already wired correctly.

## THE EIGHT QA CORRECTIONS -- build to these

**[QA-1] The legacy fallback formula is not the raw seed.** It is
`_seed_to_int64(engine, request.stable_line_seed)`. The `line_v1` policy and the
blank-`char_id` fallback must preserve **that exact formula**, not
`request.stable_line_seed` on its own.

**[QA-2] Capture ONE per-line runtime context and use it everywhere.** Alpha,
seed policy, cache params, P-OBS, cap metrics and the worker payload must all
read the SAME resolved values. **Normalize alpha to three decimals AFTER
clamping** -- otherwise two different alphas can collide on one cache key.

**[QA-3] Cap the ACTUAL vendor weights, not an idealized calculation.**
IndexTTS2 conditionally truncates after alpha, and rounding a rescaled vector
can yield `.401`. Compute the metrics from the **exact outbound list** and
enforce `effective_mass <= .4` **after serialization**.

**[QA-4] Sanitize before P-OBS.** `nodes/_otr_voice_node_common.py` currently
calls `float(...)` on raw stamped values before IndexTTS2 sanitation. Route
P-OBS state and metrics through the new safe vector-preparation helper.

**[QA-5] Resolve the real assigned reference BEFORE deriving the character
seed.** On the non-cache path the request can be built before fallback reference
resolution, so the seed must use the **durable resolved reference id**, never a
blank or stale request value.

**[QA-6] Scope, contradiction resolved.** Apply character-stable seeding to all
**three character clone profiles -- IndexTTS2, Chatterbox, Dia** -- and bump
those three `char_*` implementation versions. **Keep the emotion cap
IndexTTS2-only. Do not change announcer profiles.**

**[QA-7] Lemmy is UNQUALIFIED after the adapter change.** Preserve the old
record, but require a **new versioned qualification after re-audition**. Current
validation does not compare its stored adapter/worker fingerprint against live
code, so the existing qualification cannot be trusted through this change.

**[QA-8] The evidence plan was wrong and is replaced.** Four ordinary canonical
runs cannot guarantee frozen Nag dialogue or a frozen reference, and **each
environment arm needs a fresh server boot.** Split QA into three:
1. **deterministic fixture / audio QA** for the exact `b003` / `b005` cases;
2. **canonical publish smoke per arm** (every arm still publishes to `otr/obs/`);
3. **a true frozen-ledger canonical replay ONLY if a separately designed replay
   path is approved** -- not assumed here.

**CAMPPlus is NON-GATING** unless a diagnostics implementation is explicitly
authorized. **Blind listening plus seed / mass / reference / hash evidence
remains mandatory.**

## THE 2x2 STILL STANDS

line-vs-character seed x alpha `1.0` vs `0.4`, every arm published to
`otr/obs/`, fresh server boot per environment arm [QA-8]. Re-audition the
existing Lemmy route afterward [QA-7].

## THE EVIDENCE THIS ANSWERS

Operator, on a published episode: *"Nag 1 sounded good, Nag beat 2 was another
voice."* NAG (`c03`) in `signal_lost_mongooses_stand_20260817_234050`:

```
b003  ref=vz_donor_glenn  alpha=1.0  delivery=v2:nonzero(derived)  seed=532084266468738542
b005  ref=vz_donor_glenn  alpha=1.0  delivery=v2:nonzero(derived)  seed=5038394939402288039
```

Same reference, different seed per line. `b003` leaves a `.266` pre-vector
residual; neutral `b005` has `calm=1.0`, leaving `0.0`. **Operator's caveat, to
be carried verbatim: that is the EMOTION-LATENT BLEND, not literally "26% of his
vocal tract."** The seed change is independently audible -- which is why an
alpha-only flip was rejected and both fixes ship together.

**The reference IS applied.** Refs resolve through
`_otr_audio_engines/base.resolve_voice_ref_path` to the migrated root
`C:/ComfyUI-Models/TTS`; `vz_donor_glenn.wav` hash-matches its bank
`ref_sha256`. The driver's earlier "162 of 206 refs missing" claim was wrong
(wrong models root) and is retracted.

## CONSTRAINTS -- disqualifying

* **THE LAW:** an audit may never FAIL an episode; a render degrades, never raises.
* **`otr/obs/` volume may never drop** -- every arm publishes.
* **Byte-exact replay of shipped episodes** is preserved behind the version
  bumps, never silently broken.
* **No saved harness graphs** -- every arm regenerates from `workflows/otr_canonical.json`.
* **Story QUALITY is not being chased.** This is a correctness fix: a
  character's identity must survive his own dialogue.

## RELEASE GATE

**Do not release until a new Lemmy qualification exists AND the evidence is
reproducible** [QA-7, QA-8].
