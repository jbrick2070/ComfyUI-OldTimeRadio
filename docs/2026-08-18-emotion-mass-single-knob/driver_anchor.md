# Driver anchor -- collapse the emotion blend onto ONE knob at 0.560

**Driver:** Claude (Cowork), 2026-08-18. Repo
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch
`v2.0-alpha`, HEAD on origin `d910b4ae`.

Every claim below was read off the real Windows files or measured by running the
real code. Line numbers are as of this anchor.

---

## 1. The operator decision this implements

He listened to `otr/episodes/lemmy_emotion_ladder_logodds_2026-08-18/`, a ladder
that **pinned alpha at 1.0 and varied only the effective-mass ceiling**, and
said, verbatim:

> "IF I WERE A KID I'D LIKE MORE BUT AS AN ADULT ARM0P560 IS PERFECT."

and, of the uncapped arm:

> "its not a real emtion ist coimputer emoption"

`arm0p560_mass-0.560-PLUS16POINTS.wav` is effective emotion mass **0.560**,
speaker embedding retained **0.440**.

## 2. The mechanism, read not assumed

`nodes/_otr_audio_engines/eng_indextts2.py`:

* `EMO_ALPHA_DEFAULT = 0.4` (line 79)
* `EFFECTIVE_EMOTION_MASS_CAP = 0.4` (line 86)
* `EMOTION_MASS_CAP_DISABLED = 8.0` (line 91) -- 8 dims x 1.0, the "no ceiling"
  sentinel the pre-fix control arm needs.

`emotion_payload` (line 340) applies them **in this order**, and the docstring
says so explicitly: *"THE CEILING IS APPLIED AFTER ALPHA, NEVER BEFORE"*.

1. `_apply_vendor_alpha(vector, alpha)` -- mirrors the vendor's own pre-scaling,
   truncating each weight to 4 decimals, and **only when `scale != 1.0`**.
2. `mass = sum(applied)`; if `mass > cap`, rescale the RAW vector by `cap/mass`
   with a **floor** to 3 decimals, re-serialize through JSON, re-measure, then a
   bounded 0.001-per-step shave loop.

So the effective mass is `min(alpha * sum(raw), cap)`.

The vendor spends that mass against the speaker:
`emovec = emovec_mat + (1 - sum(weight_vector)) * emovec`. Mass 0.56 leaves
0.44 of Lemmy's own emotional embedding. Mass 1.0 leaves zero of him -- which is
the "computer emotion" he heard.

## 3. Why `EFFECTIVE_EMOTION_MASS_CAP = 0.56` alone does NOT ship what he approved

Alpha binds first. With alpha left at 0.4:

| line | raw sum | alpha 0.4 | vs cap 0.56 | effective |
|---|---|---|---|---|
| neutral (`calm=1.0`) | 1.000 | 0.400 | under | **0.400** |
| emotional (measured) | 0.934 | 0.374 | under | **0.374** |

The ceiling would never bind and the ladder rung he chose would never be heard.
This is the knob interaction the task names, and it is real.

## 4. The proposed change

* `EMO_ALPHA_DEFAULT` 0.4 -> **1.0**
* `EFFECTIVE_EMOTION_MASS_CAP` 0.4 -> **0.56**
* `config/audio_engine_profiles.yaml` `char_indextts2_v1`
  `default_params.emo_alpha` 0.4 -> **1.0** (line 99)

Alpha stops being a taste knob and becomes a pass-through; the ceiling becomes
the single knob. `EMOTION_MASS_CAP_DISABLED` stays -- it is the control arm.

## 5. MEASURED: the ceiling binds on 100% of real production lines

I ran `deterministic_delivery_vector` over **57 character lines from the 6 most
recent real episode ledgers** under `output/otr/episodes/*/audio/*_ledger.json`:

```
raw sum    : min=0.733  median=1.000  max=1.333
raw < 0.56 : 0 of 57 lines land BELOW the new ceiling
OLD eff    : min=0.293  median=0.400  max=0.400
NEW eff    : min=0.560  median=0.560  max=0.560
ratio      : min=1.40x  median=1.40x  max=1.91x
```

Two consequences, stated honestly:

* **This is not an extrapolation.** Every real line lands exactly on the
  ceiling, which is exactly the rung he heard on the ladder. The "what about
  lines below the cap, which he never auditioned" objection has no instances.
* **Total mass becomes uniform.** Under the old config the low end still varied
  (0.293 .. 0.400); now every character line is 0.560. The vector SHAPE still
  varies -- which emotions, in what proportion -- only the total budget is now
  constant. He approved that sound on the ladder, but it is a genuine change in
  character and it should be named, not buried.

## 6. Questions I want the panel to break

1. **Pin alpha, or delete it?** At 1.0 `_apply_vendor_alpha` short-circuits
   (`if scale != 1.0`), so the 4-decimal truncation never runs on the default
   path. Is a permanently-inert knob honest, or should alpha leave OTR's surface
   entirely? Deleting touches the cache key (`render_time_params`, line 411),
   the per-line receipt, `_begin_line_runtime` in
   `nodes/_otr_voice_node_common.py:490-556`, the profile schema, the acceptance
   checker's `--expect-alpha`, and the worker payload. My lean: **keep it as an
   env override defaulted to 1.0** -- the blast radius of deletion is large and
   the rollback value is real -- but say so in the docstring instead of leaving
   the old "alpha is the taste knob above it" line, which becomes false.
2. **Do the load-bearing docstrings become lies?** `current_emo_mass_cap` (line
   285) justifies the cap knob by the 2x2's alpha-axis degeneracy -- an
   experiment that is now concluded. `emotion_payload` (line 340) says "alpha is
   the taste knob above it". Both need rewriting to the shipped design, and I
   would rather the panel tell me which other comments I have not found.
3. **Does anything read alpha as a MEANINGFUL value rather than a recorded
   one?** I found `_begin_line_runtime` writing it into the cache key and the
   receipt, and `otr_voice_identity_acceptance.py:185` comparing it as a string.
   Is there a consumer that would behave differently at 1.0 -- e.g. a branch
   that treats 1.0 as "no blend requested"?
4. **The cache key.** `render_time_params` returns both `emo_alpha` and
   `emo_mass_cap`, so changing either constant changes every indextts2 line's
   key and no stale audio can replay. I believe this is already correct and I
   want it checked, not assumed.
5. **Re-qualification order.** Changing `eng_indextts2.py` moves
   `live_engine_impl_version("indextts2")`, which demotes the shipped Lemmy
   route by design (`RUNTIME_FINGERPRINT_SOURCES`,
   `nodes/_otr_voice_route.py:159`). So the audition MUST be re-rendered AFTER
   the constants land, and the record re-written with the post-change
   fingerprint. The `c18df292a41d3ddc` value currently in the working tree
   belongs to the PRE-change adapter and will be wrong the moment I edit the
   file. Is there any ordering trap here I have not seen?
6. **The audition script's own arms.** `scripts/otr_lemmy_production_audition.py`
   hardcodes `ARMS["shipped"] = {alpha 0.4, cap 0.4}` (line 58). After the
   collapse, shipped vs pre-fix differ only in the seed policy and the ceiling,
   not alpha. Does that weaken the A/B, or clarify it?
7. **The refusal that made this expensive.** That script refuses to render into
   a directory that already holds `MANIFEST.json` (line 168) -- the d910b4ae
   fix, because a re-render in place destroyed the evidence the qualification
   record cites by sha256. The re-render therefore needs a NEW directory name
   and the record must cite it. Confirm I have not left a second instrument with
   the same overwrite hazard.

## 7. The 8 tests that go stale, and their INTENT

They currently assert Lemmy is withdrawn. He is about to be qualified again, so
the assertions flip -- but each test's intent must survive.

* `tests/test_cast_lock_policy_repin.py`
  * `test_the_shipped_receipt_is_no_longer_SELECTED` (226) -- intent: the
    demotion happens at SELECTION, and returns `None` rather than raising.
    **Needs a synthetic stale record**, because the shipped one will no longer
    be stale.
  * `test_a_re_qualified_shipped_route_proves_end_to_end` (249) -- intent: the
    real record proves against the real bank with the reference re-hashed off
    disk. Can now use the shipped record directly.
  * `test_the_route_survives_the_DURABLE_stamp` (279) -- intent: `voice_route`
    reaches the durable ledger. Uses `_pin_requalified`; can drop the pin, and
    the pinned `route_id` becomes `...-v2`.
* `tests/test_voice_identity_fix.py`
  * `test_the_shipped_lemmy_route_is_no_longer_selected` (596) -- same flip,
    same need for a synthetic stale record to keep the gate covered.
  * `test_the_lemmy_record_itself_is_preserved_unedited` (607) -- intent:
    withdrawn evidence is preserved, not deleted. **Retarget to
    `superseded_native_routes`**, asserting the 2026-08-10 record verbatim
    (`g1-test-a-2026-08-10`, `b965453f355661a3`).
  * `test_a_re_qualified_record_selects_again` (622) -- intent: only a matching
    runtime recovers a route.
  * `test_the_demotion_degrades_and_never_raises` (640) -- **THE LAW**: an audit
    degrades, never raises. Must keep covering a DEMOTED route, so it needs a
    synthetic stale record too.
* `tests/test_otr_dialogue_policy.py`
  * `test_the_indextts2_route_is_approved_and_carries_its_receipt` (75) --
    asserts `audited_on == "2026-08-10"`; becomes the new audition date.

`test_the_shipped_policy_now_carries_a_REAL_qualified_route`
(`test_cast_lock_policy_repin.py:244`) asserts both runtime fields match
`[0-9a-f]{16}`; that stays true.

## 8. What I am NOT proposing

* No change to the seed policy, the delivery table, or the vendor call.
* No change to `EMOTION_MASS_CAP_DISABLED`; the control arm keeps working.
* No content filtering, no word-count gate, no story-quality work.
* No relocation or gating of `obs_publish`.
