# ITEM F: `otr_upscale_ship` fails at its WRITER, and never reaches the stage it exists to test

Observed live, 2026-08-23 ~02:45, first leg of this profile ever run.

## What happened

    [canonical-api] RESULT FAIL prompt_id=c37247e3-...
    node 1 (OTR_LedgerScriptWriter) raised NewsProScriptError
    [scifi_news_pro] pass 'script' failed after 4 attempt(s):
    markup ladder exhausted; last defects:
      - BAD_LINE_SHAPE: SCENE 1: (line 4)
      - SKELETON_BREAK: character line (Dr. Hijung Shin) before SCENE 1 (line 5)
      - SKELETON_BREAK: character line (Nikolas Martelaro) before SCENE 1 (line 6)
      ... 8 more SKELETON_BREAKs, all downstream of the same cause ...
      - SKELETON_BREAK: no scenes before END. (line 16)
      - CAST_MEMBER_SILENT: Dr. Hijung Shin
      - CAST_MEMBER_SILENT: Nikolas Martelaro (no fallback to legacy_many_pass)

**One root defect, thirteen symptoms.** The writer emitted `SCENE 1:` in a shape
`_run_markup_ladder` rejects (`BAD_LINE_SHAPE`), so the parser never registered
a scene opening, and every character line after it read as "before SCENE 1".
The two `CAST_MEMBER_SILENT` rows are the same cause reaching the cast gate.
The ladder retried FOUR times and exhausted.

**Nothing on the upscale path ran.** The profile exists to exercise
`upscale_stage.engine = spandrel_esrgan`, and the episode died in the script
pass, hours upstream of it. So this leg tells us nothing yet about the thing
item F actually wanted proved.

## The cause is the WRITER, and the comparison is clean

`otr_upscale_ship` pins **Mistral-Nemo** for both `creative_model` and
`technical_model`. The `otr_g4_wan_ti2v` leg that succeeded ~40 minutes earlier
ran **the same `_otr_scifi_news_pro` code path, the same markup ladder, on the
same server build** and produced a complete 2:47 episode. The material
difference between the two profiles' writers is Mistral-Nemo vs
`unsloth/gemma-4-12b-it-GGUF`.

This matches a measurement already in `GO_FORWARD_PLAN.md` from the Ghost Prompt
v2 work: put to the real batch prompt directly, **gemma-4-12b scored 8/8
accepted and Mistral-Nemo 4/8**. Same weakness, different surface.

## NOT caused by this session's cleanup -- checked, not assumed

The only change to `nodes/_otr_scifi_news_pro.py` in the whole campaign is the
deletion of `_RETIRED_SCENE_COUNT_TABLE`, a module-level constant with ZERO
references repo-wide, replaced by a comment (`git diff 4060a942..HEAD --
nodes/_otr_scifi_news_pro.py` is that block and nothing else). It cannot reach
`_run_markup_ladder`. The full suite is green at 12193 and an independent
verification pass rated the campaign SAFE FOR EPISODES.

## Honest limits of this finding

* **Observed ONCE**, though with four internal ladder attempts inside that once.
  A different news seed or a different roll might pass. It is not yet proven to
  be a permanent property of the profile.
* The cast names in the failing script (`Dr. Hijung Shin`, `Nikolas Martelaro`)
  look like real people pulled from the news seed. Noted, not investigated --
  it is not the failure.

## What was done about it, and what was deliberately NOT

**NOT done: editing the profile.** Promoting gemma on `otr_upscale_ship` would
change the SCRIPTS that lane produces, and the standing rule is that
`technical_model` is pinned PER-LEG with the exact dropdown label, never edited
into a shipped profile unasked. That call is the operator's.

**Done: a per-leg pin** to test the upscale stage without touching the profile --

    scripts/otr_canonical_api_run.py --profile otr_upscale_ship \
      --creative-model "unsloth/gemma-4-12b-it-GGUF" \
      --technical-model "unsloth/gemma-4-12b-it-GGUF" \
      --timeout 0

That separates the two questions cleanly: *does the upscale stage work* (the
per-leg run answers it) and *does the profile work as shipped* (it did not,
here).

## THE OPERATOR'S CALL

`otr_upscale_ship` cannot currently complete a leg on its own writer. Options:

1. **Promote gemma-4-12b on the profile.** One line. Makes the profile
   self-sufficient. Changes the scripts that lane writes.
2. **Leave it and always pin per-leg.** No script change; every future run of
   this profile needs the two flags or it fails again.
3. **Treat the Mistral-Nemo markup failure as its own bug** and fix the ladder /
   prompt so Mistral-Nemo produces a legal `SCENE 1:` shape.

Option 3 is the only one that helps every Mistral-Nemo lane rather than this
profile alone -- but story output is a closed subject by operator directive, and
a prompt change to satisfy a weaker model is close to that line.

---

## THE PER-LEG WORKAROUND DOES NOT EXIST. This is blocked on the operator.

Three attempts were made to prove the upscale stage WITHOUT editing the shipped
profile. Each failed for a DIFFERENT reason, and every guard behaved correctly --
none of these is a defect:

1. **`--creative-model "unsloth/gemma-4-12b-it-GGUF"`** -> refused at patch time.
   The combo wants the EXACT dropdown label including the size suffix,
   `unsloth/gemma-4-12b-it-GGUF (17.4 GB)`. The runner failed loud and printed
   all fourteen legal choices. (My error; the standing rule already says to pin
   with the exact dropdown label.)

2. **With the exact label** -> `GGUFNativeConfigError: Insufficient VRAM for
   GGUF n_ctx=4096`. Pinning the MODEL leaves the PROFILE's quant in place, and
   `otr_upscale_ship` carries `gguf_quant: "Q8_0"` from its Mistral-Nemo
   configuration. gemma-4-12b at Q8_0 is ~11.8 GB of weights plus KV; the
   working `otr_g4_wan_ti2v` uses Q4_K_M at 7.12 GB. The guard refused rather
   than silently downgrading the context, and its own comment says why: *"NO
   silent context downgrade (the old 4096->2048 downgrade truncated
   generations)."* Correct behaviour.

3. **Also pinning `--set OTR_LedgerScriptWriter.gguf_quant=Q4_K_M`** -> refused
   by design:

       patch_creative: widget 'gguf_quant' is not on the creative whitelist;
       managed widgets are patched ONLY via apply_profile_to_workflow(--profile)

   `gguf_quant` is a MANAGED widget. It belongs to the profile and cannot be
   overridden per leg. That is deliberate: it is what stops a leg from claiming
   a profile's identity while running a different engine configuration.

**So the architecture is unambiguous: on this lane the writer and its quant are
profile-owned, and there is no legitimate per-leg route to a passing
`otr_upscale_ship` leg.** Proving the upscale stage requires an operator
decision, which is exactly where this stops.

**The decision, restated with what is now known:**

| option | cost |
|---|---|
| Promote `unsloth/gemma-4-12b-it-GGUF (17.4 GB)` + `gguf_quant: Q4_K_M` on `otr_upscale_ship` | One profile edit. Makes the profile self-sufficient and lets the upscale stage finally be tested. CHANGES THE SCRIPTS this lane writes, which is why it is not being done unasked. |
| Leave the profile alone | `otr_upscale_ship` stays unprovable and stays "unexercised" in the queue -- honestly, rather than by an unnoticed gate bug as before. |
| Fix the Mistral-Nemo markup failure itself | Helps every Mistral-Nemo lane, not just this profile. But it is a prompt/ladder change in service of a weaker model, and story output is a closed subject by directive. |

**What item F DID achieve:** `otr_g4_wan_ti2v` is proven and published
(`docs/2026-08-23-item-F-g4-wan-ti2v-receipt.md`), and the sweep found and fixed
two real defects (`b11a4269` the preflight gate, `cebe7c75` the false-timeout
report). The second profile is now blocked on a decision instead of on a bug,
which is a better place for it to be.
