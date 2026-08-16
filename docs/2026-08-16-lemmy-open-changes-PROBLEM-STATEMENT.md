# PROBLEM STATEMENT -- the OPEN, not-yet-coded LEMMY work

**Written 2026-08-16 by a parked consult window. No code, no edits to `nodes/`,
`tests/`, workflow JSON, or any existing doc** -- another coder window is live in
the tree. This file is new and self-contained.

Every claim below is marked **CONFIRMED** (re-read against the tree today) or
**RECALLED** (from an earlier window's history, not re-verified). Where I do not
know, it says **UNKNOWN** rather than guessing.

---

## 0. Tree state I verified before writing

* **CONFIRMED** -- `nodes/_otr_scifi_codex.py` and `nodes/_otr_scifi_fable2.py`
  are GONE; `nodes/_otr_scifi_news_pro.py` exists. The rip and the rename
  happened.
* **CONFIRMED** -- `docs/2026-08-16-lemmy-chunkB-BUILD-CONTRACT.md` exists and is
  the named authority for the cameo roll.
* **CONFIRMED** -- the runway row under scrutiny is GO_FORWARD line 946, row 2:
  *"Give LEMMY a fighting chance: complete Phases 2-4 and its three live PBUGs"*.

---

## 1. PHASES 2-4 -- the phases are NOT DEFINED ANYWHERE IN THE REPO

**CONFIRMED.** `grep` for `Phase 2` / `Phase 3` / `Phase 4` across
`docs/GO_FORWARD_PLAN.md` and `docs/PROD_BUG_LOG.md` returns **only the row-2
exit condition itself and one back-reference at line 2674**
(*"Re-ground Phases 2-4 and PBUG-20260811-01/-02/-03"*). No document in `docs/`
defines what any phase contains. Searching the Lemmy docs for a phase definition
returns nothing.

**CONFIRMED -- and this is the reason.** The phase numbering almost certainly
originates in `kibitz-runs/2026-08-08-lemmy-cockney/r1/`. That directory **exists
on this box** but `kibitz-runs/` is **gitignored** (`.gitignore:251`). So the
plan the exit condition points at is invisible to a fresh clone and to every doc
search. This is the same trap GO_FORWARD's own grounding rule warns about.

**RECALLED -- Phase 1 only.** Phase 1 shipped as `bec0ca79`: `accent: "cockney"`
in `config/cast_pools.py`, plus `dialogue_orthography`, `speech_signature`, and
`nodes/_otr_dialogue_policy.py`. It deliberately shipped `LEMMY_VOICE_POLICY`
**defined-but-unwired**, which is why later work could land at zero behaviour
risk.

**What is in Phases 2, 3 and 4: UNKNOWN.** I will not reconstruct them from the
row-2 exit clauses and present the result as the original plan -- that would
manufacture a spec and give it borrowed authority. **The row-2 exit condition is
the only surviving statement of intent that a fresh clone can read**, and it
lists six clauses:

1. preserve the Cockney floor with **one upstream engine-policy authority**
   wired through the canonical workflow, CastLock and renderer;
2. qualify real routes by **operator-audition receipts**;
3. close the **six-engine gender-only pin gap**;
4. **restore or explicitly decline** `scifi_news` cameo policy;
5. resolve the **fable2 BAD_LINE interaction**;
6. **re-observe the missing closing** before diagnosing.

**Clause 4 is now partly moot: CONFIRMED that `scifi_news` was ripped entirely.**
"Restore or explicitly decline" cannot apply to a bank that no longer ships. What
survives of it is the same question aimed at `scifi_news_pro`.

**Clause 5 is CLOSED -- see section 6.**

Whoever owns the plan should either recover the phase definitions from the
gitignored r1 run and commit them, or retire the phase numbering and let the six
clauses stand as the definition. **Asking a window to "complete Phases 2-4"
against an undefined phase list is not an actionable exit condition.**

---

## 2. THE COCKNEY FLOOR -- partially decided, and the undecided half is the hard half

**CONFIRMED** -- the row-2 exit condition contains the words **"No silent
substitute and no defined-but-unwired policy."**

That settles ONE of the three options you list: **shipping a generic voice
silently is ruled out.** A receipt-bearing substitution is not obviously ruled
out by that phrase -- "silent" is doing the work.

**Which of the remaining two was chosen -- refuse the engine for his row, or
substitute a qualified engine for him alone -- is UNKNOWN, and I could not find
it written down anywhere.** I did not find a document that decides it.

The two options are not equal in cost, and the difference is worth stating:

* **Refuse** is fail-closed and consistent with the qualification contract, but
  it means an operator who selects `chatterbox` gets a hard failure caused by a
  cameo character, on an episode that has nothing else wrong with it.
* **Substitute for his row alone** means one character renders on a different
  engine from the rest of the cast. That is a *mixed-engine episode*, which has
  a downstream consequence nobody has costed: **UNKNOWN** whether the per-line
  render path, the 48 kHz bus resampling and the per-line receipts tolerate two
  character engines in one episode. That should be checked before the option is
  chosen, not after.

**This is the decision I would put to the operator first**, because clauses 1, 2
and 3 are all downstream of it -- the authority in clause 1 cannot be designed
until it is known what the authority is allowed to DO.

---

## 3. THE QUALIFICATION BAR -- and the human critical path, separated

**CONFIRMED, engine coverage, measured today.** Your figure is right; I checked
rather than agreed. `config/voice_reference_bank.json` contains **exactly one**
voice row whose id mentions Lemmy: `idx_lemmy_algenib_cockney_v1`, engine
`indextts2`. The bank's engines are `chatterbox, dia, elevenlabs, google_tts,
indextts2, kokoro` -- **six** -- and **bark is absent from the bank entirely**
because it is preset-based rather than reference-based, which is why
`v2/en_speaker_8` lives hardcoded in `LEMMY_PROFILE` instead. So:

**Lemmy has a real identity on 2 of 7 character voice engines (bark by preset,
indextts2 by qualified route). The other five have ZERO reference rows and cast
him by gender alone -- a different man each episode, no Cockney.** CONFIRMED.

**Do the other five need the same bar? UNKNOWN as a decision** -- I found no
ruling. What is CONFIRMED is what the *contract* demands if they are to be
qualified the same way: `QUALIFICATION_RECEIPT_REQUIRED_FIELDS` includes
`operator_verdict`, and the validator additionally requires a rights block with
a real `decided_at`, an `audition_manifest` with a 64-hex sha, and runtime
identity. **`operator_verdict` cannot be produced by code. That is the critical
path.**

Separated, per your request:

**HUMAN, and only human -- per engine:**
1. Decide the reference material for that engine (a clone reference clip for the
   cloning engines; a provider voice id for `elevenlabs` / `google_tts`, which
   are a different KIND of route and may not need a WAV at all).
2. Listen to a blinded A/B/C audition and score it.
3. Sign the verdict -- state in writing that it clears the gravelly / Cockney /
   intelligibility floor.
4. Make the rights call where the reference is generated or third-party, and
   date it.

**CODE, and reusable across all five:**
* the audition harness already exists in the shape used for indextts2
  (`scripts/otr_g1_lemmy_audition.py`, **RECALLED**) -- preflight the frozen
  reference hashes, render the arms, shuffle the labels, seal a key;
* installing the accepted reference, adding the bank row, writing the policy
  record, and deriving runtime identity;
* the validator, the re-pin and the receipts, all of which already exist and are
  engine-agnostic (**RECALLED**).

**The honest shape of the estimate: five auditions of operator listening time is
the schedule, and no amount of coding shortens it.** A window that plans "close
the six-engine gap" as a coding task has mis-scoped it. Two questions worth
settling before booking that time: whether all five are wanted at all (the pack
may only ever ship two or three char-voice engines in practice), and whether
`elevenlabs` / `google_tts` should be qualified as `provider_voice` routes rather
than local clones -- the route contract already distinguishes the two kinds
(**RECALLED**).

---

## 4. THE ENGINE-POLICY AUTHORITY -- no shape was ever designed

**UNKNOWN / never designed.** I could not find a designed shape for "one upstream
engine-policy authority", and I am not going to claim I intended one -- in the
consult that raised this I said the same thing, and it is still true.

What is **CONFIRMED** is that a structure already exists which is the obvious
candidate, and which is *already* keyed the right way:
`LEMMY_VOICE_POLICY["approved_native_routes"]` is a **dict keyed by engine**, and
it currently holds exactly one key -- `indextts2`. It was built to hold one
qualified route per engine, each carrying its own qualification record.

So the gap is not a missing authority so much as a missing **decision layer above
it**. The dict answers *"is there a qualified route for engine X?"*. It does not
answer *"and what should happen when there isn't?"* -- which is section 2's
undecided question. An authority worth the name owns exactly that:

* the per-engine qualification lookup (exists today);
* the **floor policy** when the lookup misses (refuse / substitute-with-receipt);
* and a single truthful answer that CastLock and the renderer both read, so they
  cannot disagree.

**Where it would live: UNKNOWN and genuinely arguable.** `config/cast_pools.py`
holds the policy data; `nodes/_otr_voice_route.py` holds the validation and is
stdlib-only and cold-import clean, which makes it the least disruptive host for a
pure decision function. **I am not proposing either** -- the point of recording
this is that row 2 asks for the authority to be *"wired through the canonical
workflow, CastLock and renderer"*, and **nothing in that sentence has a design
behind it yet.** Note also that "wired through the canonical workflow" implies a
widget or a graph change, which by the standing rule must land in
`workflows/otr_canonical.json` in the SAME change as the code.

---

## 5. ALREADY TRIED AND REJECTED -- do not re-propose

* **Routing content-owned lanes back through `lock_cast()`.** REJECTED.
  **RECALLED** cause: that block deliberately withholds `cast_seed`, because
  claiming one on a lane-owned cast detonated CastLock's replay
  (`num_characters must be 1-6, got 0`). The repair belongs in each lane runner.
* **Post-script injection of the cameo.** REJECTED, and now on the record
  **CONFIRMED** in the PBUG-01 closure: *"a cameo must be OFFERED to the lane's
  own casting/script passes BEFORE the script is written, because fable2's gate
  (b) (speaker set == cast rows) and codex's cast_coverage gate reject
  post-script injection by construction."* Adding a cast row to a finished script
  breaks the speaker-set equality gate by construction.
* **"The pre-locked LEMMY row is what the scifi_fable2 script pass cannot
  satisfy."** **WITHDRAWN** -- see section 6.
* **Inferring an accent violation from an asset id or a timbre tag.** REJECTED
  (**RECALLED**): the `_indian` filename and the `warm` tag were used to argue
  the incumbent broke the Cockney floor, and that was correctly rejected as
  unsupported by bank metadata. The finding was later established the legitimate
  way -- a blinded listener identified it by ear, without the label.
* **Loosening a confidence margin to raise coverage.** REJECTED in the adjacent
  gender work (**RECALLED**), and the reasoning transfers: a looser threshold
  buys a confident lie in the same shape as the bug, not coverage.

---

## 6. THE THREE LIVE PBUGs -- status as of today

**CONFIRMED** that row 2's three are `PBUG-20260811-01 / -02 / -03`
(GO_FORWARD:2674 names them).

**PBUG-20260811-01 -- CLOSED 2026-08-16, MIS-ATTRIBUTED. This is the most
important change since the sprint was scoped.** CONFIRMED by reading the closure.
The attribution was disproven two ways: (a) at the repro commit `baf338ee` the
runner dispatch returns at `OTR_LedgerScriptWriter.py:4032` while `lemmy_force`
is first computed at `:4415` -- *after* the return -- so **the widget was INERT
on that lane and there was never a pre-locked LEMMY row at any altitude**; the
forced-fails/natural-passes matrix was three stochastic draws of a widget that
did nothing. (b) Both surviving leg logs contain **zero** occurrences of "lemmy";
the real defects were ordinary markup non-compliance (`BAD_LINE_SHAPE` on prose
stage directions, invented speakers, `SKELETON_BREAK`) from Mistral-Nemo at
temp 0.85. The quoted `- BAD_LINE` was a truncation of `BAD_LINE_SHAPE`; no
defect named `BAD_LINE` existed at that commit.

**Consequence for the open work, and it is a big one:** the headline reason
chunk B was considered dangerous **does not exist**. Row 2's clause 5 ("resolve
the fable2 BAD_LINE interaction") is asking for the resolution of a bug that has
been closed as never having been about the cameo. What survives is architectural,
not a landmine: offer the cameo before the script pass. **Any plan still carrying
"forcing the cameo kills the writer" as a constraint is working from a withdrawn
premise.**

**PBUG-20260811-02 -- OPEN.** `scifi_news_pro` dying at video render with no
materialized still for beat `music_closing_001`. CONFIRMED status line reads
*"OPEN. Root cause established and the repair is written, but ..."* (I did not
read the full entry; the remainder is **UNKNOWN** to me). Row 2's clause 6 says
**re-observe before diagnosing** -- consistent with it having been seen once.

**PBUG-20260811-03 -- OPEN**, with an EXTENDED entry
(*"it is BOTH content-owned lanes"*). CONFIRMED. **Its scope has moved again
since that extension was written:** one of the two lanes it names,
`scifi_news`, has been ripped, so the live surface is `scifi_news_pro` alone.
**RECALLED** that chunk A shipped a truthful `cast_contract` on the content-owned
runners, which closes the *silent* half of -03; **UNKNOWN** to me whether the
entry has been updated to reflect either the rip or chunk A.

---

## What I would build first, and why

**I would not build anything until the Cockney-floor decision in section 2 is
made, because it is the only unresolved question that changes the shape of
everything downstream of it** -- the engine-policy authority in clause 1 cannot
be designed before it is known whether that authority may refuse an engine or
substitute one, and the five-audition schedule in clause 3 is only worth booking
if the answer is "substitute", since a policy that simply refuses unqualified
engines needs no new references at all. That decision costs the operator a
sentence and unblocks three clauses. After it, the first thing I would actually
build is the **engine-policy lookup as a pure, unwired function** -- one place
that answers "does Lemmy have a qualified identity on engine X, and what is the
floor policy if not", tested against the real bank, wired to nothing. It is the
piece every other clause depends on, it cannot regress a render because nothing
calls it yet, and it makes the six-engine gap *visible as data* rather than as a
paragraph -- which is the same move that turned the gender work from an argument
into a measurement. The cameo roll (chunk B) I would take second and strictly to
its own build contract, now that its supposed blocker has been withdrawn; and I
would leave PBUG-02 alone until it is re-observed, exactly as clause 6 says,
because a defect seen once with a written repair is a repair waiting for a
reproduction, not a bug waiting for a fix.
