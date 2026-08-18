# r1 ADDENDUM -- the Fable cold lane REVERSES the round's central decision

Fable reviewed r1 **cold** (no driver anchor, per the standing rule that it gets
the first opinion before the anchor frames it) and its review landed after the
antigravity judgment was written. It contradicts that judgment on the main design
question, and **it wins**, because its argument rests on four facts the other lane
did not have. All four were verified by the driver against the real files before
this reversal was accepted.

## THE REVERSAL

**r1 judgment item A2 accepted antigravity's MUST-FIX 2: "drop the runtime
`cast_pools` import -- it creates a reverse dependency," replaced by a blanket
non-empty refusal plus `--resume`. THAT IS NOW REJECTED.**

### Why -- the four verified facts

1. **There is no new coupling to create. The dependency already exists.**
   `scripts/otr_lemmy_cross_engine_audition.py:279` -- inside `render()`, the very
   function that writes -- reads `from config.cast_pools import
   LEMMY_AUDITION_LINES as LINES`. Antigravity's premise ("creates a reverse
   dependency where rendering tools depend on policy") is **factually false**.
   The writer already depends on the ledger.

2. **The campaign directory is a legitimate SHARED WORKSPACE, so sealing it is
   semantically wrong, not merely inconvenient.**
   `scripts/otr_lemmy_listen_page.py:374-375` opens
   `<campaign>/LISTEN.html` and writes it **unconditionally on every run**. A
   guard shaped "this directory is sealed" fights a sibling instrument that is
   supposed to write there. **The unit of immutability is the cited bytes, not
   the directory.**

3. **A ledger walk is schema-agnostic, which kills the driver's own objection to
   cite-awareness.** The driver rejected cite-awareness because `cast_pools.py`
   carries four different citation field shapes and a fifth would be missed.
   Fable's answer: do not parse the schema at all -- walk the **live**
   `LEMMY_VOICE_POLICY` dict recursively and collect every value matching 64 hex
   characters. That covers approved, provisional and superseded tiers alike, plus
   whatever tier is invented next, and commented-out hashes cannot leak in
   because it reads the loaded object rather than the source text.

4. **G1 is a deprecated requalification instrument, which changes what it is
   worth spending on it.** `config/cast_pools.py:800-807`: re-running
   `otr_g1_lemmy_audition.py` produced clips **byte-identical** to 2026-08-10
   because it renders `emo_vector=None` at a hardcoded seed and so "bypasses both
   halves of the fix". Hardening it toward full production parity is investment
   in a tool the project has already routed around.

**Also verified:** `KEY.json` is cited by hash **nowhere** in `config/` or
`tests/` (grep returns nothing), and the blinding map is reconstructible from
`SHUFFLE_SEED`. So the unguarded `_KEY` directory the driver anchor called
alarming is genuinely lower-stakes than the anchor claimed. That is a correction
to the driver, not to antigravity.

## THE FINDING THAT OUTRANKS EVERY GUARD -- verified, and it is cheap

**Detection exists where the stakes are lowest and is absent where they are
highest.**

* The **provisional** routes (pending-listen, lowest stakes) have a byte-level
  on-disk alarm: `tests/test_lemmy_provisional_tier.py:758-796` re-hashes the
  manifest and all six clips. It passes today.
* The **QUALIFIED** route's manifest -- `344ccdf8...`, the evidence behind the
  route production actually selects -- has **no on-disk check anywhere**. Driver
  confirmed: `grep -rn "344ccdf8\|lemmy_production_audition_ceiling" tests/`
  returns **nothing**.
* The **SUPERSEDED** G1 record gets only a config-literal assertion
  (`tests/test_voice_identity_fix.py:759` checks the number is still *in
  cast_pools*) while its own docstring claims the manifest "still hashes to the
  value it claims" -- a comment doing a test's job. Nothing computes that hash
  from disk.

And a hole in the one verifier that does exist: `_output_root`
(`tests/test_lemmy_provisional_tier.py:727-747`) returns `None` both when there
is no ComfyUI output root on this box **and** when the root exists but the
artifacts were deleted -- and the caller `pytest.skip`s on `None`. **Deleting the
entire archive today produces a silent skip, not a failure.**

This is the cheapest, highest-value change in the whole item: it costs no GPU, it
covers all three instruments at once, and it catches destruction paths **no
writer-side guard can see** -- manual deletion, a disk move, a different script
writing there.

## THE CONCRETE FAILURE CASE THAT SETTLES THE SHAPE

`--render --engine bark`. Bark has **no route row at all**
(`config/cast_pools.py:1074`, "bark gets NO row on purpose"), so re-cutting its
comparison clips looks like the one guaranteed-harmless act available. But the
manifest is **shared**: bark's row and `generated_utc` are rewritten into the same
`MANIFEST.json`, so `audition_manifest_sha256` dangles on all three receipts
**while all six clip hashes still verify**. Partial rot, and maximally confusing
to whoever audits it later. This is why the manifest must be the most-protected
file, and why per-clip protection alone is insufficient.

## WHERE THE DRIVER STILL DISAGREES WITH FABLE

Fable's scope call is "fix the cross-engine write path only; leave G1 alone;
unify at the verification layer." The driver accepts the verification-layer half
in full and **partially rejects the G1 half**.

Fable's own objection to G1's `--overwrite` is that its message asks the operator
to be *"certain this one is not cited"* -- **"asking a human to hold a fact the
program could check."** That objection is correct and it does not stop being
correct because G1 is deprecated. Once the citation-guard helper exists for the
cross-engine script, pointing G1 at the same helper is a few lines and it deletes
a standing human-judgment hazard.

**So the driver's call, threading both needles:** G1 gets the shared citation
guard, and `--overwrite` is **narrowed** so that it can still permit overwriting
an *uncited* manifest but can **never** override a citation. What G1 does **not**
get is the full production-shape directory-emptiness rewrite -- that is the churn
on a frozen instrument Fable rightly warns against, and it would break G1's own
legitimate re-run into a partial directory.

## WHAT THIS REVERSAL DELETES FROM THE PLAN

The idempotent-`--resume` machinery (r1 final D1) is **withdrawn as unnecessary**.
A cite-aware guard preserves resumability exactly up to the moment of citation and
forbids it after, so no skip-if-complete mechanism is needed for safety. This also
removes a hazard the driver had dispatched a reviewer to hunt: a resume that skips
re-rendering would report success against clips made by an OLD build, which is
12.111's own "byte-identical is not confirmation" trap. **The simpler design does
not have that failure mode at all.**

`render()` still re-renders completed engines rather than skipping them
(`:288-297`), which wastes GPU on a bare re-run into an *uncited* directory. That
is an ergonomics defect, not an evidence-safety one. **Noted, deliberately not
built** -- it is outside the queue item, and building it would reintroduce the
stale-clip hazard just deleted.

## ROSTER NOTE

Antigravity covered **r1 only** and then hit `RESOURCE_EXHAUSTED (429)` on r2 --
a confirmed provider quota block, not the print-timeout failure mode. Codex is not
installed on this box. Both r2 seats were therefore filled with Cowork subagents
per the 2026-08-17 substitution directive. This campaign will be reported with its
real roster and round count, never as a full two-external-lane arc.
