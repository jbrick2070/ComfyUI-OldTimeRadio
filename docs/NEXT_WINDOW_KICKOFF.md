# NEXT WINDOW -- paste this whole block

resume the OTR build as a CODER window. Repo:
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch
`v2.0-alpha`, HEAD `f2eeb6fd` == origin. FIRST verify both repos are pushed --
OTR `f2eeb6fd` == origin/v2.0-alpha, survival-guide `02e8bcb` == origin/main.
Two `git ls-remote` calls.

**YOUR JOB IS ONE THING: BUILD THE VOICE IDENTITY FIX. The spec is written, the
panel has run, and QA has already corrected it. Do not re-design it. Code it.**

Read `docs/2026-08-18-voice-identity-fix-ANCHOR.md` FIRST -- it is the corrected
BUILD SPEC, every item tagged `[QA-n]`. Then `docs/GO_FORWARD_PLAN.md` ("HOW TO
TALK TO THE OPERATOR", then the QUEUE STATE block) and the top entry of
`docs/HANDOFF_LOG.md`. State your MODEL & CREDIT BUDGET rung first (the table is
EMPTY -- header, separator, no rows -- so cite the per-window mapping paragraph
beneath it) and the dated REVIEW ROUTING you actually read.

**BASELINES:** suite **10913 passed / 110 skipped / 1 xfailed**. Bible
**20/26/3** at **289** entries. The trailing `1` is an xfail, not a failure.

## THE DEFECT, IN ONE LINE

A character's voice changes between his own two lines. The operator heard it:
*"Nag 1 sounded good, Nag beat 2 was another voice."* Same voice reference,
**different generation seed per line**, `alpha=1.0`.

```
b003  ref=vz_donor_glenn  alpha=1.0  delivery=v2:nonzero(derived)  seed=532084266468738542
b005  ref=vz_donor_glenn  alpha=1.0  delivery=v2:nonzero(derived)  seed=5038394939402288039
```

Episode: `signal_lost_mongooses_stand_20260817_234050` (NAG is `c03`).
`b003` leaves a `.266` pre-vector residual; neutral `b005` has `calm=1.0`,
leaving `0.0`. **Operator's caveat, carry it verbatim: that is the
EMOTION-LATENT BLEND, not "26% of his vocal tract."**

## WHAT TO BUILD -- both causes, one version, per the ANCHOR

* **(A)** character-stable engine seed for the **three `char_*` clone profiles:
  IndexTTS2, Chatterbox, Dia**; bump those three `engine_impl_version`s.
* **(B)** IndexTTS2 **effective emotion-mass cap `0.4` at the ADAPTER BOUNDARY**
  (covers hand-edited / pre-stamped vectors). **IndexTTS2 ONLY.**
* **(C)** alpha default `0.4` in `nodes/_otr_audio_engines/eng_indextts2.py` and
  `config/audio_engine_profiles.yaml`.
* **(D)** **CAP AFTER ALPHA**, never before.
* **Do NOT** touch the global delivery table, announcer profiles, or add a
  workflow widget. The canonical graph is already wired correctly.

## THE EIGHT QA CORRECTIONS -- these are why the first brief was a NO-GO

Full text in the ANCHOR; do not build without reading them.

1. Legacy fallback is `_seed_to_int64(engine, request.stable_line_seed)` --
   NOT the raw seed. `line_v1` and blank-`char_id` fallback preserve that formula.
2. ONE per-line runtime context shared by alpha / seed policy / cache params /
   P-OBS / cap metrics / worker payload. **Normalize alpha to 3 decimals AFTER
   clamping** or two alphas collide on one cache key.
3. Cap the **actual outbound vendor weights after serialization**, not an
   idealized vector -- a rescale can round to `.401`.
4. Sanitize BEFORE P-OBS. `_otr_voice_node_common.py` calls `float(...)` on raw
   stamped values before IndexTTS2 sanitation; route P-OBS through the new safe
   vector-prep helper.
5. Derive the seed from the **durable resolved reference id** -- the non-cache
   path can build the request before fallback reference resolution.
6. Seeding = all three `char_*` clone profiles. Cap = IndexTTS2 only. Announcer
   profiles untouched.
7. **Lemmy is UNQUALIFIED after this change.** Preserve the old record; require
   a NEW versioned qualification after re-audition. Validation does not compare
   its stored adapter/worker fingerprint to live code.
8. Evidence plan is three-way, NOT four canonical runs:
   (a) deterministic fixture/audio QA for the exact `b003`/`b005` cases;
   (b) canonical publish smoke per arm, every arm publishing to `otr/obs/`;
   (c) frozen-ledger canonical replay ONLY if a replay path is separately
   designed and approved.
   **Fresh server boot per environment arm.** CAMPPlus is NON-GATING unless
   diagnostics are explicitly authorized. Blind listening + seed/mass/reference/
   hash evidence is mandatory.

## THE PROOF WHEN IT IS GREEN

The 2x2: **line-vs-character seed x alpha `1.0` vs `0.4`**, every arm published
to `otr/obs/`, fresh server boot per environment arm. Then re-audition Lemmy.

**RELEASE GATE: do not release until a new Lemmy qualification exists AND the
evidence is reproducible.**

## HARD CONSTRAINTS -- disqualifying, not advisory

1. **THE LAW: an audit may never FAIL an episode.** A render degrades, never raises.
2. **`otr/obs/` publication may never be reduced, gated or relocated.** Every arm
   publishes. It is how he reads success.
3. **Byte-exact replay of SHIPPED episodes** stays intact behind the version
   bumps -- never silently broken.
4. **No saved harness graphs.** Every run regenerates from
   `workflows/otr_canonical.json`.
5. **Sonnet 5 QA on the finished diff BEFORE the push.**
6. **Story QUALITY is DONE.** This is a correctness fix only.

## TRAPS THAT BIT THE LAST WINDOW -- all five are live

* **A GREEN SUITE CAN PROVE A FIX FAILS SAFELY, NOT THAT IT WORKS.** A fix
  shipped calling `_OTRLC` from `_run_writer_tail`, where that name is not in
  scope; it raised `NameError` on every episode and its own `except Exception`
  swallowed it. All 10,905 tests passed. **If your fix is wrapped in a guard,
  add a test that asserts the PRODUCT exists, not that nothing raised.**
* **CHECK THE MODELS ROOT.** It is `C:\ComfyUI-Models`, NOT
  `ComfyUI\models`. A driver claim that "162 of 206 voice refs are missing" was
  pure wrong-root error, retracted in `f2eeb6fd`.
* **CITE SYMBOLS, NEVER LINE NUMBERS.**
* **`git check-ignore -v` EVERY new artifact path.** `docs/2026-*/` is ignored;
  `kibitz/` is a NESTED git repo whose commits never reach OTR history.
* **Never write a commit message as an inline PowerShell `-m` string** -- use
  `git commit -F`. And never a Windows path inside a non-raw Python string
  (`\U` in `C:\Users` is a unicode escape error).

## ALSO OPEN, do not start these before the voice fix

`PBUG-20260817-06` Doyle names spoken in a Leacock parody · `-07` stage
directions in captions (WILL-NOT-FIX by operator ruling) · `-08` Lemmy cameo
voice on 10.2% of all cast rows · **PBUG-04 is HALF closed** -- the announcer
names the real work, then can still embellish in sentence 2 (the join residue
the Fable seat predicted) · **24 of 34 logged bugs triaged as ALREADY FIXED** and
awaiting closure.
