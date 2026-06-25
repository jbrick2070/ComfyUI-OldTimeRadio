# R1 JUDGMENT -- dynamic coda segue (Claude, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.0898. Convergence: HIGH
on a redesign that beats my going-in plan.

## ACCEPTED (folded into pass01)
- **SPLIT the coda** (GPT CUT#1, DeepSeek optional template): LLM writes ONLY the
  bridge; append `news_close_brief` deterministically. Reliability becomes
  STRUCTURAL (the fact is never generated) -- the cleanest answer to the weak-model
  blend history. This is the core pivot.
- **DROP the `ending_change`-overlap gate** (Gemini MUST#1, the killer catch): a real
  bridge MUST reference the fiction to pivot from it, so a no-overlap gate rejects
  the good bridges. CONFIRMED -- my anchor's gate was self-defeating; removed.
- **Separate `_NEWS_CODA_SYSTEM` prompt** + suppress the resolved-fiction branch
  (all 3): the current outro prompt forbids news-summary -> can't reuse it.
- **EXCLUDE `ending_change` from the coda LLM call** (DeepSeek#4): removes the blend
  temptation at the source.
- **Anti-generic = news-token specificity + a generic-phrase blacklist** (GPT#8,
  DeepSeek#2): position/shape alone won't stop a weak model defaulting to "And now,
  the real story" -- force a real news token in the bridge.
- **Empty `news_close_brief` => don't fake a real story** (GPT#7): skip news-coda
  mode + LOUD flag.
- **Rotating-prefix deterministic FALLBACK pool** (Gemini SHOULD#1): the floor is
  varied (cast_seed-keyed) not one wooden phrase -- gives the operator variety even
  on the safety path.

## JUDGE NOTES
- Gemini's "teachability is an illusion for weak models" is RIGHT about the MODEL
  (it doesn't learn cross-episode) but the teaching target is the LISTENER, not the
  model -- consistent position + the appended real fact teach the AUDIENCE. So I keep
  "teach via structure" for the listener while NOT relying on the model to learn it
  (the model is steered per-prompt). Both panel framings reconciled.
- The split design also honors the operator BETTER than my gate-heavy plan: the
  bridge is genuinely free/crafted (no fixed prefix on the happy path), and it's safe
  precisely because it isn't carrying the fact.

## OPEN -> R2
Bridge references the news SUBJECT vs pivots from the DRAMA only (fact-copy/dup
risk); exact `_NEWS_CODA_SYSTEM` rules; specificity threshold; brief cleaning/length;
rotating pool; compose_flags taxonomy.

## CONVERGENCE CALL
The ARCHITECTURE converged hard at R1 (split bridge + deterministic payload). Proceed
to R2 to pin the build details; likely a short campaign (the hard call is settled).
