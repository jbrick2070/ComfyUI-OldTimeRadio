# DRIVER ANCHOR -- should the voice fingerprint hash bytes or code?

Written by the driver (Claude, Opus 5) at HEAD `a6302fa2`, **before reading
either panel lane**. Fable was launched COLD first, deliberately unanchored;
Codex was launched grounded. I remain the judge.

## THE OPERATOR'S FRAMING IS THE PRIMARY INPUT, NOT CONTEXT

> *"has code raw bytes that a programmer thing can you ask fable and codex the
> best option for me a vibe coder and scalability this is fun app not meant to
> be mission critical thing i dunno"*

Three constraints, all binding:
* **He is not going to own this call.** An answer that leaves him adjudicating
  a code-hashing policy has failed regardless of its technical merit.
* **Scalability is explicitly asked about.**
* **"Fun app, not mission critical" is a design input.** It licenses a weaker
  guarantee if the weaker guarantee costs him less attention.

## WHAT I FOUND THAT CHANGES THE QUESTION

The qualification record does **not** rest on the code fingerprint alone. Read
off the live `LEMMY_VOICE_POLICY` record `prod-audition-2026-08-18`, it pins
**three independent runtime axes**:

| axis | field | what it hashes |
|---|---|---|
| model weights | `runtime.weight_revision` = `6238972345f704ef` | the actual weights |
| reference audio | `reference.source_ref_sha256` / `bank_ref_sha256` | the actual WAV bytes |
| **code** | `runtime.engine_impl_version` = `9bee950a7920fd00` | **whole bytes of 4 .py files** |

Plus `audition_manifest.sha256` pinning the evidence itself.

**This reframes the whole fork.** Two of the three axes already hash exactly the
thing that can change the output -- weights hash weights, audio hashes audio.
Only the code axis hashes something COARSER than the thing that matters: it
hashes the file's prose along with its logic.

So option B is not "weakening the guarantee to be convenient". **It is bringing
the code axis into line with how the other two axes already work.** That is the
strongest argument available here and it is structural, not a matter of taste.

## MY POSITION GOING IN: B, and I hold it loosely on the HOW

**B -- hash normalised code, not raw bytes.**

What B actually misses versus bytes, stated honestly rather than minimised:
* comments and blank lines -- **intended**, that is the entire point
* formatting / reflowing -- intended
* docstrings -- **only if explicitly stripped.** `ast.dump` KEEPS docstrings
  (they are `Expr(Constant)` nodes), so a naive AST hash still trips on a
  docstring edit. Whoever implements this must decide docstrings deliberately;
  the repo's own handoff log records that "docstrings move the fingerprint too",
  so this is a known sharp edge.
* **Nothing that changes behaviour.** String literals, numeric constants,
  imports, control flow are all in the AST. A changed constant still trips it.

What NEITHER option catches, and this is the real hole in the mechanism: the
fingerprint covers **four .py files**. It does not cover the torch/CUDA version,
the Python version, the tokenizer, or any transitive dependency. `weight_revision`
covers the weights; nothing covers the stack around them. So the guarantee is
**already partial**. Making the code axis byte-exact on comments is precision in
the one place precision is cheapest and least useful -- it is not the difference
between a sound guarantee and an unsound one.

## THE SCALABILITY ANSWER I EXPECT TO GIVE HIM

Today **1 of 5 engines** has a recipe. If he ever wants approved voices on
bark/kokoro/chatterbox/dia, each needs its own hand-authored file list -- and
each new list is a new set of landmines, on files nobody has flagged. Under A
that chore grows linearly AND the trip rate grows with it. Under B the recipes
still need authoring, but they stop firing on prose. **A scales badly in the
axis he actually feels: interruptions.**

## WHERE I AM GENUINELY UNSURE, AND WHAT I WANT THE PANEL FOR

1. **Is C (drop it entirely) actually the right answer for a hobby project?**
   I lean no -- he chose this voice by ear and cares about it, and the whole
   voice-identity arc exists because voices drifting is a thing he noticed and
   disliked. But "not mission critical" is his own framing and I may be
   over-valuing a mechanism because it is interesting.
2. **Is B worth the implementation risk at all?** Changing how the fingerprint
   is computed invalidates the CURRENT stored value, so shipping B costs exactly
   one re-audition -- the thing we are trying to avoid. It pays for itself only
   if edits keep coming, which the 19-edits-in-60-days figure says they will.
3. **Docstrings: in or out?** In = still trips on prose. Out = a docstring that
   documents a behavioural contract stops being pinned. I do not have a
   confident answer.
4. **The cheapest visibility fix,** given a warning comment inside the four
   files would itself move the hash under A. Under B a comment is free, so B
   also unblocks putting the sign at the point of contact -- which may be the
   strongest practical argument for it and I want that pressure-tested.

## WHAT WOULD CHANGE MY MIND

A concrete failure mode where a comment-only edit genuinely does change
rendering behaviour. I cannot construct one for Python source in this codebase.
If a lane produces a real one, B is wrong and A stands.
