# scifi_fable2 S1b live-smoke hardening -- panel problem statement

- Date: 2026-07-10 (mid-execution review; S1b code is IN the tree, uncommitted
  hardening deltas on top of pushed commit a24b75c4)
- Question for the panel: **are the S1b runner's content gates CONVERGING
  (keep iterating the current design) or STRUCTURALLY WRONG (a gate or pass
  boundary needs to move)?** Ground every claim in the real files:
  `nodes/_otr_scifi_fable2.py`, `nodes/_otr_fable2_markup.py`,
  `nodes/story_packs/scifi_fable2/scifi_fable2_v1.json`,
  `nodes/_otr_structured_call.py`,
  `docs/2026-07-10-scifi-fable2-architecture.md` (the ratified plan).

## Context

S1b = the fable2 lane's one-pitch/one-draft spine, live: P0 dossier -> P1
pitch -> P2b treatment -> P3 whole-play markup -> P6 casting -> P7 python
assembly (proof gates) -> P8 audit-only. Local model Mistral-Nemo 12B via the
structured_call ladder (base -> structural retry on JSON-syntax only -> typed
repair; content failures get base + repair = 2 attempts). Suite is green
(7416+); the 30-word LIVE smoke has failed 11 consecutive rolls, each on a
different defect class, each fixed at the root the same session:

| Roll | Killer | Fix applied |
|---|---|---|
| 1 | treatment read gate rejected 'June' (calendar word) | calendar stopwords |
| 2 | jinja TemplateError: two consecutive user msgs in P3 reroll | fold reroll into ONE user msg |
| 3 | P3 markdown-bold labels, parens (all rungs) | seam HARD BANS extended |
| 4 | P3 parens persist | contrastive WRONG/RIGHT pair in seam |
| 5 | dossier gate rejected "Amsterdam's canals" vs "canals of Amsterdam" | token-level entity presence |
| 6 | read gate rejected 'Dr' (honorific) | honorific stopwords |
| 7 | read gate rejected possessive "NASA's"; honest kill on numeral '10' | possessive strip |
| 8 | read gate rejected 'Paris' (story's own setting; 12B dossier missed it) | legality corpus widened to the SOURCE DIGEST |
| 9 | model put its fictional heroine 'Lia' in the REAL-news read | explicit cast-name-in-read gate w/ teaching error |
| 10 | P3 markdown returned + parens through few-shot; nested pitches.0.hook string_too_long skipped by top-level-only clamp | assistant-turn few-shot example; parser delete-only normalization (markdown + paren delivery tags on speaker lines, flagged, legacy stage_dir_stripped precedent); nested-path clamp in _otr_structured_call |
| 11 | P3 died on ONE defect: missing closing MUSIC line (attempt 3) | 4th ladder rung @0.30 + skeleton recap in reroll (in tree, roll running) |

## Addendum -- rolls 12-17 (post kibitz-r2 folding)

| Roll | Killer | Fix applied |
|---|---|---|
| 12 | mid-play ANNOUNCER bridge -> skeleton cascade; multiword "DR. VERONICA VOSS" | announcer-scope recency line; one-word CastShape schema law (kibitz S1) |
| 13 | treatment repair could not converge on the read (fictional 'RAPHAEL') | READ-SPLIT: P2c news_read pass (kibitz r2 Q1 pulled forward), seam + registry row + own validator |
| 14 | read gate outlawed 'JIM' -- the drama named its character after the REAL Jim Ross in the story | source legality beats the fictional ban |
| 15 | otherwise-perfect draft died on CODA terminal '.' vs ':' | pivot colon = STRUCTURAL seam marker; normalizes in the shared pre-lex (flagged); inner sentence break stays the defect |
| 16 | story said "seven days"/"Eighth day"; dossier extracted 7/8 as digits; verbatim gate killed it | spelled-number equivalence (cardinals+ordinals 0-100) |
| 17 | WORD_BUDGET exhaustion: 54 vs 24-36 -- the +/-20% band is 12 words wide at 30w | absolute band slack floor +/-25 words (_word_band); proportional band governs >=125w |

## Addendum 2 -- rolls 18-22 (post kibitz-r3 folding)

| Roll | Killer | Fix applied |
|---|---|---|
| 18 | casting JSON truncated at 1000 tokens -> extractor salvaged inner object -> schema fail | casting budget 1400 + wrapper-tolerant before-validator |
| 19 | 'DR. HARRIS' cast name; repair would not converge | deterministic label normalization: strip honorific tokens, keep surname |
| 20 | 107 character words vs target 30 through every numeric hint | MICRO-EPISODE structural line cap (<=target//7 lines, <10 words each) in prompt + reroll |
| 21 | audit finding omitted 'speaker' -> schema abort of a complete episode | AuditFinding speaker/scene defaults (reporting pass tolerance) |
| 22 | SPINE FULLY GREEN (all 8 passes; canon title via override); THEN freeze cascade Phase 10: a c01 postamble row arrived speaker_role='character' + skip=True + tts_skip_reason null (no compose-flag breadcrumb -> unsanctioned cast-keyed mutator downstream) | fable2 announcer rows now ALL carry sentinel char_id="announcer" (exempt from every cast-keyed path by design). OPEN QUESTION for the panel: which freeze-cascade path mutates a cast-char_id announcer row to character+skip without stamping tts_skip_reason? (reviewer=improved, Stage-7 critic flagged 3 lines, reroll errored on the missing line_composer_system seam and 'kept the original') |

Also noted roll 22: the episode's cast came out VERA/DOKU -- the SCRIPT seam
example's names -- despite the imitation guard and despite the treatment never
seeing that seam. Name-leak vector unknown; flagged for the operator eyeball.

## Current gate design (as in the tree NOW)

- P3 markup ladder: 4 falling-temp rungs (0.75/0.5/0.3/0.3), defect-quoting
  reroll folded into ONE user message, +25% truncation retry once, budget
  reroll max 2 w/ numeric hint, assistant-turn few-shot of the seam example.
- Parser: delete-only normalization (markdown emphasis anywhere; short
  paren/bracket delivery groups on SPEAKER lines only), every strip collected
  in ParsedScript.normalizations, stamped to meta + logged; all other defect
  classes reroll-strict; proof artifact = the same-normalized draft.
- Treatment news_close_read subset gates: numerals must be in allowed_numbers
  OR the python-capped source digest; proper nouns (minus calendar/honorific
  stopwords, possessives stripped, word-boundary) must be in the WHOLE dossier
  + provenance + source digest; fictional cast names hard-banned with a
  teaching error message.
- structured_call content failures still cap at 2 attempts (base + typed
  repair) -- the shared helper's structural rung is JSON-syntax-only by the 2B
  principle.

## Questions (answer with VERDICT + MUST-FIX/SHOULD-FIX, grounded)

1. Is the news_close_read gate set now sound, or is asking one 12B treatment
   call to satisfy title+cast+registers+turn+priced_ending AND a
   subset-law-compliant factual read structurally overloaded? (S2 could split
   the read into its own technical pass -- doc change.)
2. Is the parser's delete-only normalization legal under the operator law
   "Python judges; the LLM writes -- never rewrites a spoken word"? (Legacy
   precedent: compose_line's stage_dir_stripped flag.) Any hole in the
   normalize/proof-artifact coupling (normalize_fable2_markup_text vs
   _prove_constituents)?
3. Content failures get 2 structured_call attempts; the P3 ladder self-manages
   4. Right shape, or should content failures get a bounded extra rung
   (helper-wide or fable2-local)?
4. Anything structurally missed that will bite the NEXT roll (P6 casting, P7
   assembly, P8 triage have barely been exercised live)?

Invariants the fix must respect: no fallback to legacy_many_pass; fail loud;
science_news byte-identical; the ratified architecture doc governs unless live
evidence justifies a documented deviation; UTF-8 no BOM; SFW.
