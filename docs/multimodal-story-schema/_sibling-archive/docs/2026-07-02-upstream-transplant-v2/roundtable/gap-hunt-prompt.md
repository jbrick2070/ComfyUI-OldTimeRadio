You are one reviewer on a 3-model panel. A plan and its implementation have
already survived a rigorous 4-round review by three LOCAL file-reading agents
(Claude Code, OpenAI Codex, Google Antigravity) that crawled the real
repository. Your job is DIFFERENT: hunt for what THEY missed.

The briefing document below is self-contained, and several REAL source files
from the repository are attached as grounding. The briefing includes a ledger
of findings the local panel already caught and fixed - do NOT re-report
those; finding them again scores zero.

Hunt specifically in the blind-spot zones of file-crawling code reviewers:

1. CONTENT QUALITY, not structure: the 12 story packs' prompt prose will be
   fed to LLMs to write radio drama. Are the prompts dramaturgically weak,
   contradictory, or likely to produce flat/samey episodes? Will the
   forbidden-pattern metadata approach actually prevent drift?
2. DESIGN-LEVEL blind spots: failure modes that only appear at RUNTIME or at
   OPERATOR scale (one person, local Windows machine, 16GB VRAM, offline
   LLMs) that code inspection cannot see.
3. SPEC-LEVEL contradictions between the documents and the code excerpts
   that a crawler dismissed as style.
4. WHAT IS ABSENT: things nobody wrote down anywhere - missing failure
   handling categories, missing operator workflows, missing tests that
   matter more than the ones written, second-order effects of the
   architecture (e.g., what happens after 50 packs, after 6 months of
   production drift, when the operator forgets the rules).
5. The COMPATIBILITY MIRROR strategy: pinned shapes + AST drift tests
   against a mirrored copy of production. What failure mode does this
   strategy itself have?

Output format (strict):

VERDICT: <one line - is the work transplant-ready, and your confidence>

NEW FINDINGS (things the local panel plausibly missed):
- [area] finding, why it matters, concrete evidence from the briefing or
  attached files, and a concrete fix. Number them. Quality over quantity -
  three sharp new findings beat ten recycled ones.

CONTENT CRITIQUE (the story-pack prose itself):
- specific packs/prompts that will underperform and rewrites you would make.

ABSENT THINGS:
- what is missing entirely, and the smallest version worth adding.

DO-NOT-DO:
- anything in the plan you would actively remove or simplify.

If you cannot ground a claim in the briefing or attached files, label it
[HYPOTHESIS]. Do not fabricate file contents or line numbers.
