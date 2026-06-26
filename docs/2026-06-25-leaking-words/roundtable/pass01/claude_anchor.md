<!-- Claude R1 anchor review (grounded). Leaking-words strategy. -->

VERDICT: yes-with-fixes. The problem framing + leak taxonomy are correct and the
option set is the right search space, but as a *plan* it is still a MENU, not a
decision. A roundtable plan must commit. My grounded position: ship E (cheap
deterministic floor) now + make D (frontier = recommended clean default) the
quality path; scope A (LLM-cleaner) as an OPT-IN layer that fails OPEN to E; CUT
B (constrained generation). The deepest truth in the doc is correct and under-
weighted: the scrubs are a downstream mop for an upstream (weak-local-model)
generation problem, so the durable answer is "clean floor + better writer," not
a smarter mop.

MUST-FIX BEFORE BUILD:
1. [strategic question] The doc lists A-E but commits to nothing. Decide. Proposed
   commitment: (E1) relax the leading stage-direction detector beyond the
   `_NARRATION_VERBS` whitelist; (E2) add a deterministic caps-name-vocative
   scrub; (E3) a news-proper-noun body guard; all three always-on. (D) document
   the frontier writer as the recommended clean lane. (A) an OPTIONAL LLM-cleaner
   layered on top, off by default, fail-open to E. That is the plan; the rest is
   sequencing.
2. [Option A] As written, A threatens THREE hard invariants: byte-identical audio,
   determinism, offline-capability. It is only safe if scoped: the cleaner must run
   in the WRITER, BEFORE TTS/freeze, so the audio is synthesized from the cleaned
   text (consistency preserved); it must FAIL OPEN -- if the cleaner model is
   absent/offline/errors, the deterministic floor (E) still runs and the episode
   still ships; and it must be deterministic-seeded or temp=0. Without all three,
   A breaks offline-first and cannot be the baseline. (verify: the freeze->TTS
   order -- the cleaner must sit upstream of audio synthesis.)
3. [Option E1 / leak class 1] The "Gasping," fix cannot be "add gasping to the
   whitelist" (that IS the whack-a-mole). Relax the detector to a SHAPE rule:
   leading capitalized `-ing`/`-ed` participle + comma + opening quote, EXCLUDING
   any clause containing a 1st/2nd-person pronoun. BUT this risks scrubbing real
   dialogue ("Running to the door, I shouted...") -- so it MUST be measured over
   the shipped-ledger corpus (the 2026-06-22 sprint's method: ~489 ledgers, count
   would-mutate vs false-positive, require 0 FP) BEFORE it ships. A relaxed scrub
   with an unmeasured FP rate is more dangerous than the leak.

SHOULD-FIX:
1. [leak class 2 / news-bleed] This is NOT a stage direction and no shape rule
   catches it. The cheap deterministic version: flag a body line that contains a
   proper noun present in the episode's news-seed named-entity set but ABSENT from
   the cast + setting allowlist (e.g. "President Trump" in a fictional NASA drama)
   -> reroll once with a "no real-world names" directive. The complete version is
   A. Pick the cheap guard as the floor, A as the upgrade. (verify: the news_seed
   exposes a named-entity / key_terms set the guard can read.)
2. [E2 / caps-vocative] Cheapest, safest win: a token that is ALL-CAPS and matches
   a cast first/last name in a vocative position -> title-case it (or drop the
   vocative). Very low FP risk; wire it as its own scrub. Do this regardless.
3. [sequencing] State the order explicitly: deterministic floor (E1+E2+E3) is the
   always-on baseline at compose+freeze; A is an opt-in pass layered above it and
   gated like the existing OpenRouter writer (default-off, fail-closed-to-floor).

OPTIONAL / NICE-TO-HAVE:
- A tiny regression corpus of the exact 4 leaks seen today (the "Gasping," line,
  the Trump line, the "YUKI MARTIN" line, the unclosed-quote line) as frozen test
  fixtures, so any future pass proves it catches these and stays green.

CUT THESE:
1. [Option B -- constrained generation / GBNF] Cut. Memory + the Ollama GBNF
   hardening work established Ollama's /v1 cannot take raw GBNF (the inventor-GBNF
   flag is a no-op there); the local lanes (Ollama / in-process / llama.cpp) have
   inconsistent grammar support, so a grammar floor is not portable. Worse, "emit
   only spoken words, no real-world names" is not expressible as a context-free
   grammar (it is a semantic constraint, not a syntactic one). B cannot do the job
   and is not model-agnostic.
2. [Option C as a standalone] Do not cut prompting, but do not treat C alone as
   the answer -- a weak model under a better prompt still leaks (that is the
   premise). Fold a one-line "spoken words only; no real names" reminder into the
   compose prompt as cheap insurance, but the floor must be downstream (E), not
   the prompt.

[ASSUMPTION] audio is synthesized from the frozen post-scrub text, so a cleaner
upstream of TTS preserves byte-identity (verify the freeze->audio order).
[ASSUMPTION] the local lane is the leak source; frontier (GPT) shipped 0 leaks
today, a 1-episode sample -- not yet proof across many episodes (verify with more
frontier renders).
