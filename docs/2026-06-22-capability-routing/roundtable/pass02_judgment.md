# R2 judgment -- capability-routing coding plan (Claude judge)

Panel: gpt-5.5 + gemini-3.1-pro + deepseek-v4-pro + grok-4.3 + Claude grounded. CONVERGED on a SMALLER,
SAFER fix than the input proposal.

## THE CORRECTION (Gemini MUST#1, echoed by GPT#3/#4, DeepSeek, Grok)
**Do NOT change wan_i2v's `required_inputs` to `("text_prompt",)`.** wan IS image-to-video -- it needs a
still. The announcer/music/scene_broll/character_video roles ALL supply `init_image`, so wan's
`("init_image",)` ALREADY satisfies the input match for them; the ONLY blocker is the `roles` whitelist.
Downgrading wan to `text_prompt` would falsely fit `background_abstract` (text-only, NO still) and CRASH
the i2v at `_assert_family_inputs_satisfiable`. So KEEP wan's `("init_image",)`; fix ONLY the whitelist.
wan then fits exactly the still-supplying roles (announcer included = what the operator wants); it
correctly still can't do background_abstract (no still) -- ltx can (text-to-video). That's fine.

## ACCEPTED
- engine_fits_role whitelist made OPTIONAL: `if required is None: return False`; `if roles and role not in
  tuple(roles): return False`; keep `required <= INPUT_TOKENS` + `required <= available`. (Gemini#2, Grok#1.)
- Empty wan's eligibility `roles` (capability governs); KEEP its `required_inputs=("init_image",)`.
- DROP the FAMILY_REQUIRED_INPUTS change (wan keeps init_image -> family gate unchanged). (Falls out of the
  correction.)
- CUT `optional_inputs` -- dead code, nothing consumes it. (Unanimous: GPT CUT#1, Gemini CUT#5, DeepSeek, Grok.)
- before/after eligibility test needs a BASELINE SNAPSHOT mechanism (capture pre-edit results / old-algorithm
  helper) -- can't compute "before" from post-edit descriptors. (GPT#6, Grok#4.)
- AUTO-SELECTION non-regression is the REAL risk (GPT#7, DeepSeek#3): expanding the eligible pool must not
  change which engine auto-PICKS for an existing slot -- golden test per slot; confirm the selection algo.

## RESOLVED IN R3 (wiring)
- **The descriptor `roles` source (l.132 `getattr(eng,"roles",())`) vs engines declaring `default_roles`**
  -- ALL FOUR flag this MUST be confirmed: is wan's blocking value a `roles` attr, `default_roles`, or
  derived? Emptying the wrong one won't propagate. Reconcile the naming.
- ASPECT (wide vs portrait): enforced downstream (director `_role_aspects`), or was `roles` hiding it? Test
  a wide engine not auto-picked into a portrait slot.

## REJECTED
- wan -> text_prompt downgrade (would crash i2v in background_abstract).
- assert-equal FAMILY sync as a hard step (Gemini SHOULD#4 prefers derive; Grok CUT -- and moot now that
  FAMILY is unchanged).
