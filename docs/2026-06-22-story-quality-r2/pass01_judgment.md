# Story quality R2 -- pass01 judgment + converged plan (Claude = grounded judge)

## Panel
GPT-5.5 (rigorous, 63 lines), Gemini-3.1-pro (tight), Grok-4.3 (sharp). ~$0.19. Strong convergence;
they upgraded the problem statement into a build-ready, ledger-safe, targeted fix set.

## CONVERGED FIX SET (4 chunks; ledger {cast,lines,meta} FIXED, audio frozen, craft-only,
## TARGETED so it never blanket-rewrites the already-good strong-model output)

### Chunk 1 -- A: stop the music_inter placeholder leaking as a spoken/caption line
- ROOT (GPT, decisive): the defect is that a NON-VOICED `music_inter` beat renders as a visible
  line -- NOT the wording. Wording-only fixes (Gemini's "[Music: transitional bridge]") leave the
  row rendering. FIX = role-based SUPPRESSION: suppress voiced/caption TEXT for
  `speaker_role == "music_inter"` while KEEPING the beat + its timing + the music row.
- Do NOT key on `dialogue_slot_id is None` (music_open/close/sfx also have none) -- key on the role.
- `Beat.intent` has a `min_length=4` validator, so keep a valid neutral internal intent (e.g.
  "Bridge to the next phase with music only.") -- never blank it.
- TEST: no rendered transcript/caption line contains "Musical interlude bridging"; the music_inter
  ROW COUNT + voiced slot ids are unchanged before/after.

### Chunk 2 -- B: announcer CLOSE dramatizes, never summarizes
- `_otr_outline._assemble_outline` stamps the close intent as "Close the episode and tag the
  broadcast." -> change to a concrete-final-image contract: "Close on a concrete final image showing
  what changed; no moral, thesis, or news-summary tag." (outline content only -- ledger-safe).
- ADD a deterministic banned-thesis-phrase scan on the close/meta line ("Tonight's revelation",
  "the lesson", "reminding us", "proving * right", "this shows", "* is now shared") -> reroll
  through the DEDICATED ANNOUNCER composer (the critic excludes announcer lines, so the character
  reroll path won't catch it). Reroll/reject, NOT regex deletion.
- TEST: the grounded close failures (the three real quotes) reroll/reject.

### Chunk 3 -- C: lift the WEAK end (cliche + meandering stage-business) -- the big one
- INJECT the opposed wants into the LINE-COMPOSER prompt for voiced beats -- BUT only when
  `DramaticState` wants are SOURCE-DERIVED / NON-DEFAULT (GPT+Grok: the helper often emits generic
  "honor the established commitment" vs "force a compromise" defaults; injecting those doesn't help).
- Require the line prompt to turn each beat intent into an ACTION VERB UNDER PRESSURE (reveal /
  refuse / demand / bargain / accuse / conceal / choose) -- safer than "write better prose", which a
  weak model satisfies with cliche.
- ADD a SMALL deterministic reject gate (grounded, high-signal -- NOT a big ban-list): exact/near
  cliches ("you're playing with fire", "this changes everything", "we're not leaving anything to
  chance") + pure stage-business with no pressure/reveal/refusal ("I'll go check...", "I'll
  double-check...", "I'll lock down...", "I've got this, no need..."). Flagged lines -> TARGETED
  reroll only (the beat intent + speaker + opposed want + prev/next context; include the reject
  REASON, not the ban-list). CAP at ~3-5 character rerolls/episode.
- CUT (all 3): a full extra-LLM rewrite over EVERY line (expensive, regresses opus, unnecessary).

### Chunk 4 -- gating + invariants (Q4/Q5)
- TARGETED only: music suppression always; announcer-close scan always; character-line reroll ONLY
  on a deterministic flag (or a very low critic score). NO blanket rewrite -> opus is protected.
- Craft-ONLY: do NOT touch EpisodeBudget / beat count / word allocation / ledger schema.
- INVARIANT TEST: music_inter row count + voiced slot ids stable before/after; track the four
  failure metrics SEPARATELY (music-placeholder / meta-close / cliche / stage-business counts) --
  never one collapsed "quality" score.

## CUT (panel)
- Full per-line LLM rewrite; a large global cliche ban-list (overfits, breeds alternate cliches);
  critic-as-primary-gate (a weak local critic is unreliable -> deterministic flags first, LLM critic
  optional); word/beat-count tuning (anti-goal); redesigning the _otr_outline Path C architecture.

## Build order
1 (music suppression -> kills the universal wart) -> 2 (announcer close) -> 3 (weak-end lift) -> 4
is woven through 1-3 (gating + the invariant tests). Each its own green chunk (suite + Bug Bible) ->
commit + push. Then a short re-soak (a few weak-local + one frontier leg) and re-read the scripts to
confirm the four metrics dropped without regressing opus.

## Convergence
Three models, the same four fixes, no contradictions; every claim code-grounded. Build-ready.
