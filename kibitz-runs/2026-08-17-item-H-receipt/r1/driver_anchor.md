# Driver anchor -- item H (Claude, Cowork). Written BEFORE fan-out.

Every claim below is labelled against files I actually read at HEAD `9804f7a2`.

## VERDICT

**Do Option A (fix the receipt). Do NOT do Option B (give lumina a floor) without
the operator.** But Option A as item H words it is under-specified in a way that
matters, and my confidence that "rename the arm" is the whole job is lower than
the measured blast radius suggests. I want the panel to attack point 2 hardest.

Reasoning, shortest path first: the fourth arm is the only one making a claim the
dispatcher cannot substantiate, because `engine_id` is not bound until ~56 lines
later in the same iteration. That is not a lumina bug wearing a dispatcher
costume -- it is a dispatcher bug that lumina happens to expose. Fixing lumina
would leave the label still unverified on every other engine, including any
engine added later. So the receipt is the root and the floor is a separate
question about one model's artifact profile.

## MUST-FIX

**MF1 -- item H names a function that does not exist. CONFIRMED.**
`lumina_image` has no `_resolve_negative`; the negative is resolved inline in
`_lumina_params`. Only `z_image_turbo` has that function. Any plan phrased as
"add the floor to `_resolve_negative`" is unimplementable as written. I have
already corrected the notes in `lumina_image` and the item H body.

**MF2 -- the receipt is unverified for ALL engines, not wrong for one.
CONFIRMED.** Verified by reading `otr_image_gen_dispatcher.py`: the
`_neg_source` ternary is computed from `_pack_negative` / `_obj_negative` only,
and `resolve_engine_for_role(...)` binds `engine_id` about 56 lines further down
inside the same per-object `for` loop (whole block at one indent level). So
`engine_hygiene` is a claim about an unchosen engine. True of z_image by
coincidence. **This reframes the item and it is the single most important thing
the panel should either confirm or break.**

**MF3 -- "do not leave the receipt wrong" is satisfiable without touching a
recipe, and that asymmetry should decide the order.** Option A is zero-pixel;
Option B changes conditioning at cfg 4.0 and therefore owes a render under the
standing trap (green gates are not a working fix). Doing B first spends a render
to fix half a problem while leaving the label unverified elsewhere.

## SHOULD-FIX

**SF1 -- the blast radius is smaller than the project's own caution implies, and
I measured it rather than assuming. CONFIRMED.** `engine_hygiene` lives in two
code sites (a comment and the value). The ledger key is `negative_source`
(written at two call sites). A scan of 4,795 JSON files under the real ComfyUI
output base found ZERO `negative_source`, ZERO `self_veto_resolved`, ZERO
`_style_spread`, while finding `visual_style` in 770 files and `prompt_hash` in
1,022 -- so the scan is sensitive and the field has simply never been written.
There are no historical rows to become inconsistent with.

**SF2 -- a corollary the plan does not record: the entire `visual` ledger
section from the one-style-authority work is unproven on a live render.**
CONFIRMED by the same scan (three sibling keys absent together, not one).
This upgrades D-BIS finding 5 to "no tests AND no live observation", and it means
the operator's "lock them in the ledger" ask has never been demonstrated
end-to-end. The cheapest honest close for item H may therefore be *one render*
that writes the section at all, rather than any code change.

**SF3 -- naming. UNVERIFIABLE from the code alone; a judgement.** My instinct is
that the arm should say what was observed (no composed negative contributed)
rather than predict what an engine will do. But I can see an argument that
downstream readers want to know whether a floor WILL apply, in which case the
honest version requires the stamp to move after engine resolution -- a production
reordering in a loop that also computes cache keys, seeds and the banana
transform. I do not want to make that call alone.

**SF4 -- D-BIS finding 4 is adjacent and might be cheaper to fold in now.**
It asks for the resolved cfg (or a `negative_live` bool) on the per-row record,
because at cfg 1.0 a logged negative conditioned nothing. If the arm is being
touched anyway, recording whether the negative was LIVE may be more useful than
naming where it came from. Flagged, not assumed.

## Claims I deliberately did NOT verify

* Whether any consumer outside this repo reads `negative_source`. I checked this
  repo and the on-disk ledgers; a consumer living elsewhere (the 4060 box, an
  external analysis script) is outside what I can see.
* Whether lumina at cfg 4.0 actually produces worse artifacts with an empty
  negative than with a floor. That is a pixel question and no render has been
  spent on it. Option B rests on it and it is currently unmeasured.
