# Problem statement -- BLIND title trace (paste this to agy)

**Deliberately half-blind.** The driver has already traced this and is holding
its findings back so an independent trace can either confirm or contradict them.
Nothing below names the cause, the file, or the function. **Do not ask for the
driver's answer before reporting yours.**

Everything between the lines is the paste.

---

## DO NOT WRITE OR CHANGE ANY CODE

This is a **read-only diagnosis**. Do not edit files, do not write a patch, do
not produce a diff, do not propose an implementation. If you find yourself
writing code, stop. **Explanation only.** A fix will be decided separately,
after your diagnosis is checked.

## THE SETTING

Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
(real Windows files -- read them directly; any Linux-mount view of this tree
lags and is stale).

It generates old-time-radio drama episodes as videos. Each finished episode
opens with a **hero TITLE CARD** -- large text burned onto the video showing the
episode's title.

Produced episodes live under
`C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\<episode_id>\`, each with a
machine-readable record at `audio\<episode_id>_ledger.json` plus other sidecar
files.

## THE SYMPTOM

Some published episodes display a **title card that is not a story title at
all** -- it reads like internal engineering shorthand. The operator noticed it
on screen while watching finished episodes.

Two real episodes to compare:

* **WRONG:** `signal_lost_chunkb_accept_forced_lemmy_scifi_news_pr_20260816_185234`
  -- its title card reads `CHUNKB ACCEPT FORCED LEMMY SCIFI_NEWS_PRO`.
* **RIGHT:** `signal_lost_the_blackwood_enigma_20260817_172553`
  -- its title card reads `THE BLACKWOOD ENIGMA`, which is a proper story title.

Both episodes completed successfully and both published. Neither errored.

## WHAT TO DO

**Start at the title card on screen and trace the words backwards through the
code until you reach the place they originally came from.** Then explain why
these two episodes ended up different.

Answer these, in this order:

1. **The chain.** Every hop the title text passes through, from the burned pixels
   back to its origin. For each hop: the file, the SYMBOL (function, class or
   constant), what the value is called at that point, and one plain sentence on
   what that step does.
2. **The divergence.** The exact point where these two episodes stop following
   the same path. Name it, quote the deciding logic verbatim, and say what each
   branch does. **This is the most important thing in your report** -- be
   specific enough that someone could put a finger on it.
3. **Why each episode went the way it did.** Read both ledgers and show the
   actual values that made each one take its branch.
4. **Reach.** Does this same title string end up anywhere other than the title
   card? List every place you can prove it lands.
5. **Origin.** Where does the wrong-looking text physically come from -- what
   produces a string like that, and is it a one-off or something that happens
   systematically? Prove it if you can; say "not proven" if you cannot.
6. **Is anything actually malfunctioning?** State plainly whether any code is
   behaving incorrectly, or whether every step is doing what it was designed to
   do. Do not assume a bug exists just because the output is undesirable.

## GROUND RULES

* **Cite SYMBOLS -- function, class and constant names -- never line numbers.**
  Line numbers in this repo go stale within the hour; a report citing them
  cannot be checked later.
* **Quote verbatim** when you quote. Short quotes.
* **Write "not proven" rather than inferring.** A confident wrong answer is worse
  than an acknowledged gap. The driver's own trace was wrong about several things
  today; assume you can be too, and mark what you could not verify.
* Note anything that looks like corruption or a rendering artifact and say
  whether you can actually prove it is one -- some visual oddities in this
  pipeline are intentional effects.

## OUTPUT FORMAT

```
THE CHAIN (last hop first):
  <n>. <step name> | <file> | <SYMBOL> | <field name> | <one plain sentence>

THE DIVERGENCE:
  where: <file + SYMBOL>
  deciding logic (verbatim): <quote>
  branch A: <what happens>
  branch B: <what happens>

WHY EACH EPISODE WENT ITS WAY:
  <table: the actual values from each ledger that decided it>

REACH: <every surface the same string reaches, proven>

ORIGIN OF THE WRONG TEXT: <what makes it, one-off or systematic, proof>

IS ANYTHING MALFUNCTIONING? <plain answer>

[ASSUMPTION] <everything you could not verify at the files>
```

---

## FOR THE DRIVER ONLY -- do not paste below this line

Verification targets. The independent trace should land on the same divergence
point and the same systematic origin. Points of interest when its report comes
back:

* Does it find the divergence in the writer rather than the video renderer?
  (The renderer is a decoy -- both episodes take the identical path there.)
* Does it correctly rule the title-card scramble effect INTENTIONAL rather than
  reporting it as corruption?
* Does it find that the same string becomes filenames, the canon record and the
  credits, not just the card?
* Does it find the systematic producer for the `SOAK##`-shaped titles, and
  correctly decline to claim proof for the one-off?
* Does it answer question 6 with "nothing is malfunctioning"?

**Disagreement is the valuable outcome, not agreement.** If it lands somewhere
else, one of the two traces is wrong and that is worth knowing before any code
is written.
