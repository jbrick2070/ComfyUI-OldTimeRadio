# PLAN -- shield scoping (the banana route's quote shield, narrowed to card text)

**Date:** 2026-08-07. **HEAD:** `22dd4f57` on `v2.0-alpha`.
**Status: SHIPPED 2026-08-07.** Full r1-r4 `kibitz-plugin:kibitz` arc
complete (cold Fable on r1 + Codex `gpt-5.6-sol` high + Antigravity every
round, `--driver claude`); build order
`kibitz-runs/2026-08-07-shield-scoping/r4/final.md`, judgments per round
(LOCAL ONLY, gitignored).

**What shipped.** `apply()` takes `shield_quoted_card_text` (keyword-only,
default True); the still dispatcher passes
`(source == "still_word")`, the video funnel passes False explicitly.
`TABLE_VERSION` "2" -> "3", which versions the WHOLE transform algorithm
(table AND shield scope) -- append-only: historical rows keep "2" and are
never rewritten, and the version is not a cache input, so the bump alone
re-mints nothing.

**KNOWN AND DELIBERATELY NOT FIXED:** `_compose_mesh_fodder_prompt` splices
`get_era_tail` unfolded (`otr_meta_brief_image_prompt.py:1595-1597`) where the
card composer folds. Scoping already ends the under-fire there (mesh rows are
not `still_word`, so nothing shields their quote pair); folding it would move
mesh prompt hashes for zero functional gain, and `_fold_inner_dquotes` is
documented as a still_word-CARD invariant. Cosmetic only: a doubled quote may
appear in a mesh prompt.

---

**Historical -- r1 completion note.** This document was the r1 input; the
CONVERGED shape lives in `kibitz-runs/2026-08-07-shield-scoping/r1/final.md`
(LOCAL ONLY, gitignored) and supersedes the FIX DIRECTION sketch below where
they differ. Headline rulings: per-call-site switch keyed on the object's
`source == "still_word"` stamp (`kind` proven inherited, `role` proven
drifting); video funnel shield-OFF wholesale; default TRUE (fail toward benign
under-firing, never toward transforming card script); MUSIC CARDS STAY
SHIELDED (the credits roll renders the episode title and the announcer speaks
it -- live-proven in the leg log); TABLE_VERSION bumps to "3"; argument named
`shield_quoted_card_text`. Bonus verified finding: the mesh-fodder composer
splices the era tail unfolded (`:1595-1597`) -- scoping defuses the shield
leak there, and r2 decides the one-line hygiene fold. r2 (coding) is next;
nothing is built.
**Parent:** the banana route (`bc8a1bde`, CLOSED) deferred exactly this chunk at
its r4 -- see `docs/2026-08-06-BUILD-SPEC-banana-route.md` build-state note and
GO_FORWARD 0-QUATER OPEN 1.
**Operator ruling (2026-08-07, this chunk's charter):** *"if dialogue that's
fine but not for visuals"* -- quoted CARD/DIALOGUE text stays shielded; quotes
in ordinary VISUAL prompts stop being shielded.

---

## THE DEFECT (grounded at the banana r4, driver-verified by execution)

The banana route substitutes weapon nouns in visual prompts, and shields quoted
spans so still_word cards -- whose visual prompt QUOTES the spoken line as text
to render in the picture -- pass script through untransformed. That shield is
BLANKET: `_shielded_spans` (`nodes/_otr_banana_route.py:243-264`) protects
EVERY same-style quote pair in EVERY prompt at both funnels.

The leak: `_clean_llm_prompt` (`nodes/otr_meta_brief_image_prompt.py:1630-1638`)
strips only LEADING/TRAILING quotes from writer-LLM output, so INNER quotes
survive into `obj["prompt"]` on the portrait and character-scene paths. An
LLM-styled `a man carrying a "revolver"` therefore reaches the still funnel
with a quote pair, the shield protects it, and the revolver ships to the image
engine untransformed. EXECUTED: 0 substitutions on that input. The same class
exists on the video funnel (an M4 creative prompt can carry quotes).

Severity: FALSE NEGATIVE -- the route under-fires; no crash, no ledger fault,
receipts honestly record 0 substitutions. The era-tail instance of this class
(LLM-authored atmosphere quotes shielding a weapon INSIDE a card) is already
CLOSED by `bc8a1bde` (the era tail is folded before composition).

## THE LEGITIMATE SHIELD CASE (must stay byte-identical)

still_word cards, two shapes (`nodes/otr_meta_brief_image_prompt.py`,
`compose_still_word_prompt`):
* WORD mode: `a title card displaying the words "<spoken line>"` -- the quoted
  span IS script rendered as picture text (operator ruling: script, not
  picture).
* MUSIC mode: `an abstract picture evoking "<episode title>"`.
After `bc8a1bde`, a composed card carries EXACTLY ONE double-quote pair (the
template boundary): inner quotes are folded to `'`, backslashes scrubbed, the
era tail folded. The card's NON-quoted pieces (backdrop, lettering, era, grade)
describe PICTURE and must keep transforming (a "smoky saloon with a rifle rack"
backdrop should banana).

## FIX DIRECTION (operator-endorsed; r1 pressure-tests the SHAPE)

Scope the shield to card text only:
* `apply()` grows a switch (working name `shield_quotes`, default TBD) -- ON
  only where the caller is composing/handling a still_word CARD prompt; OFF for
  portraits, scene stills, and ALL video prompts.
* The still funnel (`otr_image_gen_dispatcher.py:1019-1025`) applies the
  transform per object row and must learn WHICH rows are cards -- candidate
  discriminators: the object's `kind` / `role` / `source` fields stamped by
  `derive_image_prompts` (still_word roles vs portrait/scene kinds). Exact
  field choice is r2 work with the code open.
* The video funnel (`render_driver.py:2902-2965`) composes no cards -- shield
  presumptively OFF wholesale there. r1 should challenge: is there ANY
  legitimate quoted-card content on a video prompt? (Cards are stills-only
  today; the ia2v/brief+beat prompts quote nothing by construction.)
* `_shielded_spans` / `_is_escaped` themselves DO NOT CHANGE (QA ruling 9 --
  same-style pairing, odd/even parity -- stands). The mechanism is right; only
  its APPLICATION is scoped.

## CONSEQUENCES TO DESIGN FOR

* Prompt hashes MOVE for exactly the prompts that carried decorative quotes
  around mapped terms (they now transform). Default-ON lane hash movement is
  already accepted (banana ruling 6); the set is tiny. OFF-lane and fidelity
  prompts stay byte-identical (route off = no transform either way).
* The operator's shield ruling gets RESTATED precisely in the module docstring:
  shielded = card-rendered script text, not all quotes.
* Receipt schema UNCHANGED (six keys, contract-frozen). `banana_substitutions`
  may rise on affected prompts -- that is the fix working.
* Cast-time preflight runs the video funnel per beat (`otr_shot_lock.py:1001`)
  -- any new parameter must default such that preflight behaviour is identical
  to render behaviour.

## WHAT r1 IS ASKED (arc / shape -- not line edits)

1. Is the per-call-site switch the right SHAPE, or is there a cleaner seam --
   e.g. the card composer MARKS its own quoted span (position, not pattern) and
   `apply()` shields only marked spans; or shield only the two known template
   patterns; or strip decorative quotes at `_clean_llm_prompt` instead so
   non-card prompts simply carry no quotes to shield?
   Each alternative's failure mode should be named (pattern-matching rots;
   quote-stripping changes prompts wholesale and moves EVERY hash; markers
   thread state).
2. Which surface OWNS the decision -- the composer (knows it made a card), the
   dispatcher (knows the object kind), or the banana module (knows nothing)?
3. Default ON or OFF for unknown/direct callers, and what the conservative
   choice does to the ~30 existing `dispatch_images` harness calls.
4. Does the video funnel go shield-OFF wholesale, and is anything lost?
5. Is any OTHER surface rendering text-in-picture (credits roll, procgen
   overlays) reachable by either funnel and owed the card treatment?
6. Test shape: cards byte-identical; quoted-weapon portrait/scene/video prompts
   transform; a card whose BACKDROP carries a mapped term still transforms
   outside the quoted span; era-tail regression stays green.

## GATES (unchanged discipline)

Full four-round arc (r1 this doc -> r2 coding -> r3 wiring -> r4 convergence,
Codex + Antigravity, `--driver claude`) -> code -> focused + full suite vs the
9067/111/1 receipt -> Bug Bible -> AST/BOM -> Sonnet QA -> Fable gate -> ONE
pathspec commit -> push -> `HEAD == origin`. NO `workflows/` change (the gate
is env + ledger, unchanged). These are pre-production findings -- no PBUG, no
Bible entries.
