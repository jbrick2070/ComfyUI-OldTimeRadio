# r3 JUDGMENT -- CONVERGED. Build authorized.

Driver: Claude/Cowork. Panel: Antigravity (Gemini 3.8 Flash) + **Sonnet, substituted
for Codex**, which hit its GPT-5.3-Codex-Spark usage limit mid-round (retry ~21:43).
Roster stated honestly: Codex did NOT participate in r3.

## BOTH of the driver's clauses were broken, independently, by both lanes

**Clause (A) -- the head-of-family token-count guard -- is DEAD.**
Sonnet swept it: **274 false declines** (188 where the honorific AGREES with the
row's own gender, plus 86 via the ungendered `DR` prefix) against 242 genuine
fixes. The driver verified six of them by hand on the real sidecars:

    DR. ROYLOTT      -> male    ('DR. GRIMESBY ROYLOTT',)   (A) kills it; DR is ungendered so (B) cannot recover it
    DR. KELL         -> female  ('DR. LIRA KELL',)          same
    DR. FRANKENSTEIN -> male    ('VICTOR FRANKENSTEIN',)    same
    MR. DARCY        -> male    ('FITZWILLIAM DARCY',)      (A) severs the citation
    MISS EYRE        -> female  ('JANE EYRE',)              same
    MR. SCROOGE      -> male    ('EBENEZER SCROOGE',)       same

(A) also breaks the regression test THIS CAMPAIGN just added
(`test_a_given_name_alias_is_not_a_join_key`, which pins `MR. DARCY` at
`short_form`) -- the driver's own new guard caught the driver's own next proposal.

**Clause (B) alone is a NO-OP on the headline bug.** Sonnet's proof: the join
never abstains for `MR BENNET` (it answers `female/short_form`), so a rung placed
"after abstention" never runs. (B) needs the join made honest first.

## WHAT SHIPS -- the gender-contradiction guard + the title rung, together

Both lanes reached this independently.

1. **`_GENDERED_HONORIFICS`** -- one dict derived from `_HONORIFICS`, the single
   source of truth for both mechanisms. Antigravity: the driver's 9-token list
   omitted 25+ gendered titles already in `_HONORIFICS` (KING/QUEEN, DUKE/DUCHESS,
   COUNT/COUNTESS, MONSIEUR/MADAME, HERR/FRAU...). Do not re-declare a subset.
2. **CONTRADICTION GUARD, per-row, inside the join tiers.** Refuse a candidate ONLY
   when the slot's leading honorific is GENDERED and CONTRADICTS that row's recorded
   gender. Token counts are irrelevant. Sonnet's sweep: **0 regressions** on 266
   rows x 6 honorifics.
3. **TITLE RUNG, strictly LAST**, only after every roster tier abstains.
   Antigravity found 26 rows already resolved by the `pronouns` rung with real
   textual evidence (Mr./Mrs. Otis, Father Brown, Lord John Roxton, Mrs. Rachel
   Lynde). Running the rung EARLIER would discard sourced evidence for a generic
   title guess at the same value -- pure information loss.
4. **A BONUS THE ROW DID NOT ASK FOR, and it is real.** Three sibling-surname
   families today abstain via `ambiguous_join` and get nothing. The guard resolves
   them correctly because the honorific disambiguates:
   `MISS CUTHBERT -> Marilla`, `MR CUTHBERT -> Matthew`, `MRS CHALLENGER -> Mrs. Challenger`.

## THE ONE DISAGREEMENT, and the ruling

Antigravity: reuse `gender_source="title"` (already maps to `"known"`).
Sonnet: use a NEW `gender_source="title_honorific"`, `gender_confidence="stated"`.

**SONNET WINS.** `"title"` is the STAMPER's source for a vendor-confirmed fact
recorded from the source text (`Aunt Em`, `Mrs. Dorman` carry it today). Reusing it
would make a render-time read of a slot's spelling INDISTINGUISHABLE from a
confirmed source fact -- the exact citation dishonesty this whole round exists to
remove. Sonnet verified nothing breaks: `confidence_for_source` is a `.get(..., "")`,
the stamper's table is a separate closed set the render path never reaches, and the
only persisting consumer (`_otr_casting.py`) stores free text with no enum check.
`"stated"` is unused in the repo (grepped, zero hits). Add the key EXPLICITLY to
`_CONFIDENCE_FOR_SOURCE` rather than hand-stamping it per call site.

## Accepted costs and carried risks

* `MRS BENNET` still cites Elizabeth (female honorific, female row -- no
  contradiction). RIGHT GENDER, wrong citation: the cosmetic class, not the
  wrong-gender class. Left alone deliberately.
* The module docstring says "never guesses from a name" (:19) and must be edited in
  the SAME change. Sonnet's attack is accepted: a courtesy title is a surface
  feature of the slot string, and the slot is written by the adaptation LLM. It is
  still categorically stronger than a name-frequency guess, and it sits BELOW every
  confirmed record -- but the carve-out gets named in the docstring, not implied.
* LATENT, not live: Shakespeare cross-dressing lanes (Twelfth Night, As You Like
  It) could address a disguised character by an opposite-gender title. Swept: ZERO
  live rows contradict today, and no shipped Shakespeare row carries an honorific.
  Recorded, not guarded.
* Housekeeping flagged by Sonnet, NOT touched here: a stray worktree
  `.claude/worktrees/awesome-brahmagupta-a509b4/` holds an older copy of this
  module; two Twelfth Night sidecars carry `aliases: None` rather than `[]`
  (harmless -- `_candidate_names` isinstance-guards it).
