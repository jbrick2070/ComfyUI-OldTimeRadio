# Production Sprint Lessons

## Mandatory use

Before an implementation, live diagnosis, source-bank change, or workflow edit,
read this document and the relevant entry in `PROD_BUG_LOG.md`. Then consult the
portable counterpart in the shared Bug Bible at
`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\BUG_BIBLE.yaml`.
This document retains the OTR-specific causal history; the Bible carries the
generalized rule and its executable guard. Record a newly discovered production
failure in the log first, and promote it to the Bible when it is repeatable and
generalizable. Do not admit an invented fixture, code-review observation, or
static-audit finding as a new production bug; those may only verify a failure
already observed in a live run, smoke, soak, or published artifact.

Use this guide when planning any OTR coding sprint, including story banks,
dynamic visual direction, audio/SFX layers, model integrations, and workflow
changes. The project rules in `AGENTS.md` and `CLAUDE.md` remain authoritative.

## 1. Define ownership before implementation

For every artifact and field, name:

- the one authoritative writer;
- every consumer;
- whether it is authored, derived, or measured;
- its lifecycle boundary (draft, accepted, frozen, or post-freeze extension);
- its durable storage and replay receipt.

If an input list becomes output rows, state the mapping exactly: one row per
input item, one singular owned value per row, exact reference coverage, and
which collections live at the top level. List the exact fields allowed in each
nested row. Do not assume a model infers ownership or nesting from a JSON
schema.

## 2. Keep five representations in lockstep

Every structured LLM pass has five representations that must agree:

1. base prompt;
2. typed schema;
3. worked fixture/example;
4. parser and deterministic validator;
5. repair prompt.

A change to one requires an audit of the other four. Explicitly forbid common
pseudo-shapes: numbered fields, `_secondary`/`_tertiary` variants, schema-path
keys, singular-vs-list aliases, and valid collections nested at the wrong
depth.

## 3. Separate authored decisions from mechanical repair

Python may safely repair only facts already determined by accepted artifacts:
IDs, ordering, references, exact enums, hashes, routing metadata, duplicate
references with an unambiguous first owner, and relocation of values whose
destination is uniquely declared.

Python must not invent story ownership, grounding, or authoritative narrative
decisions: motives, clues, causality, visual taste, sound-design decisions, or
source claims. If placement or meaning is ambiguous, return the defect to the
owning model and fail closed when the bounded ladder exhausts.

The one narrow exception is a terminal spoken-surface hygiene floor after every
bounded model repair rung has exhausted on a non-safety craft defect. It may
strip action/markup, normalize speech-only tokens, trim a clause, or substitute
a small generic SFW utterance whose value is delivery rather than story
authority. It must not alter immutable evidence, grounding, graph ownership, or
factual suffixes; it must validate the exact TTS projection, stamp the gate and
resolving rung, and isolate an empty result to that row. Content safety and
ambiguous structural defects remain fail-closed.

A deterministic repair must validate the complete downstream contract before
returning. A partially valid schema object can consume the repair rung and
prevent the intended LLM fallback.

## 4. Design retry ladders by failure class

- JSON syntax failure: same prompt at lower temperature may help.
- Typed/schema failure: send the failed artifact, exact error, schema, and
  explicit correction rules to typed repair.
- Semantic/graph failure: name the objective invariant and the owning item;
  do not ask a lower-temperature retry to guess.
- Incomplete repair JSON: retry the exact cached repair prompt only when the
  configured budget permits it.

Log which rung ran and why. Never silently truncate, silently coerce authored
content, or fall back to a canned artifact.

## 5. Size context from the real artifact

Do not derive output or context budgets from `target_words` alone. Include the
actual size drivers: accepted line count, evidence rows, graph width, schema
overhead, prior artifact size, and repair envelope size. The repair prompt is
often the largest call.

Resolve the true context cap for the selected model and fail loudly when a
provenance-sensitive prompt cannot fit. A model's native context and the
project's configured safe context are separate facts; measure both.

## 6. Test model diversity, not just repeatability

A prompt that works on one local LLM is not production-qualified. Different
families fail differently: integer IDs, extra fields, renamed forbidden keys,
wrong nesting, copied request envelopes, enum synonyms, or over-wide output.

Qualification ladder for model-sensitive work:

1. unit fixtures and full regression gates;
2. canonical 30-word end-to-end smoke on at least two different local model
   families and one configured cloud/frontier creative lane;
3. the same pairings at 120 words;
4. only then, 720-word qualification or bakeoff.

Record concrete model labels, slot assignments, prompt IDs, repair counts,
ledger paths, and final asset paths for every leg.

## 7. Prove the real workflow end to end

Always load `workflows/otr_canonical.json`. Code that is not wired into that
file is dead. Any node, input, widget, or link change updates and validates the
canonical JSON in the same commit.

An API `SUCCESS`, idle VRAM, or a resident server is not final proof. Verify:

- the saved ledger and required receipts;
- the canonical episode asset under `otr/episodes/<ep>/`;
- `obs_publish OK`;
- the final file exists under `otr/obs/`.

Conversely, resident VRAM after `Prompt executed` is not evidence that a run is
still alive. Check the queue, runner exit code, history result, log, and file.

## 8. Keep GPU experiments clean and comparable

Before every headless run, selectively stop only ComfyUI and its harnesses,
clear port 8000, and confirm baseline VRAM. Never blanket-kill Python. Boot with
UTF-8 through the canonical launcher. Save one server log and one leg log per
run, and use the watchdog for long renders.

Change one meaningful variable per comparison. Do not mix a model change, a
prompt change, a quant change, and a context change into one unexplained result.

## 9. Treat live failures as reusable project knowledge

Every bug that actually fails a live smoke, soak, or published episode gets an
append-only entry in `docs/PROD_BUG_LOG.md` with symptom, root cause, fix,
verification idea, and Bug Bible candidacy. Dev-only catches are fixed and
tested but do not enter the production log.

Promote recurring, machine-checkable failure classes to the Bug Bible later.
Also update the relevant planning guide and preflight so the next sprint avoids
the bug before implementation.

## 10. Preserve concurrency boundaries

One window owns code and the canonical workflow. Parallel windows may perform
read-only investigation or docs-only scoping, but they do not edit active code,
tests, registries, or workflow JSON. Preserve unrelated dirty-tree files and
stage only the owned change.

After every code change: focused tests, full Windows suite, Bug Bible, AST/JSON
checks, commit, push to `v2.0-alpha`, and verify `HEAD == origin` before live
qualification.

## 11. Prove semantic provenance, not just structural references

An ID can remain valid while its meaning disappears. If artifact A carries a
typed clue and artifact B carries only that clue's ID, a later model can emit a
self-consistent but unrelated story while every graph/reference validator
passes.

For content-owned pipelines, build a small immutable grounding contract from
the accepted source/draw. Carry exact positive anchors through every authoring
and retake seam, and verify their spoken evidence at the coordinates that own
them. Structural coverage, an LLM audit, a frozen ledger, and a published video
cannot override a deterministic semantic-fidelity failure.

## 12. Test channel isolation by changing one channel

Comments saying “visual style does not affect story” are not proof. Run the
same mocked story pipeline twice while changing only the visual-style value and
assert every captured story-model message is byte-identical. Apply the same
method to SFX, voice, render, and other downstream channels.

The story may feed visual direction; visual direction must not feed back into
story authorship unless a separately named, explicitly designed mode owns that
decision. Store selectors in separate ledger namespaces so a forensic audit can
distinguish correlation from causation.

## 13. Persist enough accepted artifacts to locate first drift

Response hashes prove identity only when the response is still available. They
cannot reveal where meaning first changed after a production run. Persist a
bounded receipt of accepted typed artifacts and their grounding evidence:
selected input, truth/plan, score/manifest, final authored output, audit, and
line/span coordinates.

Do not retain raw prompts, rejected outputs, or unlimited retry prose by
default. The goal is replayable accepted-state forensics, not an unbounded model
transcript.

## 14. Close every nested object, not only the artifact root

Listing required schema paths tells a model what must exist; it does not tell
the model which plausible bookkeeping fields must not exist. For every strict
structured artifact, state the exact key set at each nested ownership boundary,
especially music, media, provenance, delivery, and file/path-shaped objects.

Pair that prompt contract with a narrow structural normalizer only where Python
can prove the removed fields are non-authoritative metadata. Such a normalizer
must preserve all required authored values byte-for-byte and rerun the complete
strict schema plus semantic/graph validators. Never project an arbitrary model
object onto a schema when an unknown key could carry story meaning.

## 15. Make ordered graph topology executable

An array can contain every required typed row and still violate the graph's
meaning through order alone. State sequence rules with a concrete valid and
invalid example (for example A, A, B versus A, B, A), identify which manifest
owns the canonical group order, and enforce the invariant in Python.

Never reorder authored chronological rows as a structural repair: sequence is
story, even when every row object remains byte-identical. Repair only mechanical
identifier topology when it is provably unambiguous -- for example, split a
reopened A/B/A shot into A/B/A-return by cloning the shot metadata and retagging
the later run -- while holding the authored row sequence fixed. Then rerun the
complete graph, semantic, grounding, and landmark validators. Reject or fall
through to typed repair whenever the identifier split is ambiguous.

## 16. Run deterministic projections at every attempt boundary

A typed-repair prompt factory sees the failed base response, but it does not
necessarily see or normalize the model's typed-repair response. If a safe
projection exists only inside that factory, a model can repeat the identical
mechanical defect on the final repair attempt and bypass the projection.

Hash the actual raw model response first, then apply the same narrow projection
at the slot-output boundary for every base, syntax-retry, and typed-repair
response. Return the projected artifact only when its strict schema and complete
semantic/grounding validators pass; otherwise retain the raw response and let
the ordinary ladder report or repair the real remaining defect.

## 17. Put typed structural repair at the accepted-object boundary

Raw-string cleanup is appropriate for unambiguous JSON-shape defects, such as
lifting a declared collection from the wrong nesting level. It is not enough
for a projection whose safety depends on the fully parsed artifact and its
complete graph/grounding validator. Apply that kind of repair in the
schema-validated post-check over the same typed object that will be returned to
the caller.

Test the boundary itself: deliberately disable or bypass the earlier raw
normalizer, then prove both a base response and a typed-repair response still
accept only after the typed projection preserves authored values and clears all
validators. This prevents a duplicated pre-parse helper from becoming a
false-green proxy for the production acceptance path.

## 18. Group safe graph ownership repairs at one typed boundary

For a schema-valid score, related exact invariants such as contiguous shot
ownership and one-to-one clue ownership should share the same typed
post-validation boundary. The model owns the first authored placement and all
meaningful prose; Python may only derive a later mechanical identity or remove
an exact repeated reference when the complete typed graph proves that no story
meaning is lost.

Keep raw cleanup limited to defects that prevent parsing at all. Test each
typed repair with that raw path disabled, including a typed-repair response,
so a pre-parse helper cannot falsely appear to qualify the production guard.

## 19. Compose independent safe projections before global validation

Do not make each narrow mechanical repair demand that every other invariant is
already clean. A score can contain two independently provable defects, such as
a reopened shot run and a duplicate clue reference. Factor a projector from its
full-validation wrapper, apply only a small declared set of disjoint projectors
in deterministic order, and then run the complete graph, grounding, and safety
checks once over their shared result.

The composition must be bounded, must preserve authored content and chronology,
and must remain fail-closed for an unknown, missing, ambiguous, or still-invalid
condition. Add a regression where the base and typed-repair responses contain
the complete defect combination; testing each repair in isolation is not
evidence that the production boundary can combine them.

## 20. Repair localized semantic omissions with bounded typed patches

Do not resend an entire accepted artifact merely because one LLM-owned leaf
misses an immutable semantic anchor. Whole-document regeneration increases
context pressure, expands the failure surface, and can force the model to
recreate already-valid structure instead of correcting the actual omission.

Define a minimal patch schema that names the allowed targets and fields. Python
may derive those targets from the immutable contract and must verify exact
one-for-one coverage, literal anchor inclusion, and no changes outside the
declared patch scope. The model still authors the replacement prose. After the
merge, rerun the complete artifact, grounding, and authored-surface validators;
an unknown or broader defect must remain fail-closed rather than being squeezed
through the narrow tool.

## 21. Preserve accepted invariants in every replaced patch field

A patch that replaces an entire leaf value can remove a different invariant that
was already correct in that same value. Include both the newly required facts
and every immutable fact currently present in a selected target. This matters
when one beat carries multiple contracts, such as a reveal or closure beat that
also owns a clue.

Validate the *merged canonical artifact* in the structured-call post-validator,
not only the patch's local key set. A local patch can be schema-valid and carry
all newly requested literals while still breaking a graph, grounding, safety, or
authorship invariant elsewhere in the target field. Return the precise merged
error to the typed repair ladder and remain fail-closed if it cannot be cleared.

## 22. Apply localized semantic repair at every artifact boundary

Do not stop the bounded-patch pattern at a planning artifact. A complete
performance script can be just as expensive and unreliable to regenerate for a
single missing immutable phrase as a complete score. Keep the authoring model
responsible for replacement prose, but create a separate typed patch seam for
each artifact whose individual leaves have clear ownership and a full validator.

For spoken-script patches, target line IDs only; retain every valid literal
already spoken by that line, require every newly required literal, and rerun the
complete script graph, safety, and grounding checks after merge. The tool must
not become a broad retake in disguise: no title, roster, ordering, or unplanned
line edits are permitted.

## 23. Put the guarded artifact boundary behind every reauthoring route

An initial authoring pass is not the only place an artifact can regress. Blind
listener retakes, optional polish retakes, and final-audit retakes all create a
fresh complete artifact and can reintroduce the same localized defect. Do not
wire a bounded repair at only the first call site.

Factor one guarded authoring helper that first accepts structural/safety-valid
output, invokes the narrow patch only for a localized full-contract failure,
then validates the merged canonical artifact. Declare the repair seam on every
pipeline pass that can traverse this helper and use pass-specific journal IDs so
production receipts show where the correction occurred.

## 24. Restoring an input is not authoring -- the "lost anchor" class

(2026-07-13: this class cost twelve live rolls before it was named. Causal
record: PBUG-20260713-15..18 in `PROD_BUG_LOG.md`; moved here from
GO_FORWARD_PLAN in the 2026-07-15 baseline.)

A pass hands an LLM an IMMUTABLE string Python already owns -- a constraint-draw
field, a dealt card, a locked speaker, a coordinate from an accepted artifact --
and asks for it back verbatim. The model paraphrases. Python compares exactly
and kills the episode over a copy of its own input.

Restore when the correction is FORCED (exactly one value possible); return it to
the model when it is not. Three further laws proven live: a repair prompt that
does not fit is worse than no repair (PROMPT_GUARD truncates the contract
silently); a bounded repair must ask for the unit the model can deliver (batch a
patch and a partial success becomes a total failure); and "it is broken" is not
a repair prompt -- name the missing object, the unassigned clue, the exact
string.

## 25. A bank lives in ~10 wired surfaces -- rip by the Teardown protocol

(2026-07-18: the Sonnet bake-off retired 4 banks; the hard part was rediscovering where a bank is
wired, not the decision. Runbook: the Teardown protocol in `SOURCE_BANK_PREFLIGHT.md`; proven by the
`499386aa` roster trim and `docs/2026-07-18-rip-4-banks-plan.md`. 2026-07-31: the runner table left
the writer for `nodes/_otr_lane_specs.py` -- same surface, new address.)

A source bank is not just its `banks.json` row -- it is wired across ~10 surfaces: the row, the pack
dir, the `story_rules` file, the runner entry in `_otr_lane_specs.LANE_SPECS`, the pipeline object in
`pipelines.json` (a SECOND registry -- the one hand-removal always forgets), the lane runner module, a
possible `if base == "<family>"` route, the runnable<->executable registry law in `_otr_story_routing`,
the roster/bijection + bank-enumerating guard tests, and any PBUG the bank's live failures earned. Two
rules that end the re-derivation: (1) removal DEPTH is set by whether a sibling version survives -- a
variant rip keeps the shared lane module, a full-family rip (the only version of its lane) deletes it;
(2) retiring a bank that carried a live failure is a legitimate fix, but RECORD the PBUG -- ripping the
lane must not rip the causal record. Gate on a green suite + retired-id absence, never a predicted count.

## 26. Spoken craft gates repair and ship; they do not own episode liveness

A craft or delivery-hygiene finding says that wording needs repair; it does not
prove the ledger is structurally unrenderable. Spend the existing authored
repair, then a lower-temperature gate-specific repair, then the alternate
writer slot. If the model remains stubborn, apply a bounded deterministic SFW
floor, re-run every applicable detector, and stamp the gate plus resolving rung.

Validate the exact surface the voice engine will consume. Number and
abbreviation expansion can make a canonical line fail one-breath or spoken-form
checks only after delivery normalization. Content-owned lanes must repair that
projection before rebuilding hashes, proofs, and seals; shared-ledger lanes need
a final row scour before freeze and another guard after readiness normalization.

Never route craft exhaustion to an episode terminal disposition. A deterministic
floor that still yields empty or punctuation-only text creates a row-local
mechanical failure. Genuine graph ambiguity and the G9 SFW/content-safety gate
remain fail-closed.

## 27. Classify cast identity, not lexical parts

A cast label is an identity declaration, not a bag of forbidden words. Generic
roles such as `THE TRAVELER`, `First Witch`, or `MEDICAL DOCTOR` must remain
usable as visual nouns, while a personal label such as `Doctor Aris Thorne`
must still keep its meaningful name components out of an anonymous visual
brief. Articles, ordinals, and honorifics are grammar, not identities.

Use two complementary projections from one bounded role classifier. Input
anonymization maps a generic full label and its meaningful role noun to one
stable placeholder, but never maps an article. Output validation permits those
generic role forms. For personal labels, protect the full surface and meaningful
personal components while excluding titles and articles as standalone tokens.
Keep the external reason code stable, but carry the exact matched surface
privately into typed repair so the model can correct the actual defect instead
of guessing. If all bounded repairs genuinely fail, retain the explicit failed
sentinel and deterministic non-authoring consumer defaults; do not silently
invent episode content.

## 28. Capacity is part of a bounded artifact contract

An output-token request is normally a ceiling, so reducing it to the room left
by the measured prompt is honest for an artifact that can be shorter. It is
not honest for a closed patch or complete typed artifact whose required rows
must all arrive. Mark that call explicitly as requiring the full requested
output, capture the marker before any message normalization, and reject before
model generation or network I/O when either the real context window or a
provider output cap is smaller.

Do not solve a row-local quality finding by re-emitting a complete script.
Derive a closed target set, give the writer all voiced rows as read-only
context, request only target-line replacements, merge only the owned text
field, and run the complete artifact validator. A null reviewer coordinate may
widen to all voiced rows; an invented non-null coordinate is noise and must not
widen ownership.

Quality work never owns episode liveness. Rejudge every successful patch, but
after both available writer slots fail, keep the best already-valid script and
stop that loop. Never restore an unchanged artifact and immediately ask the
same judge for the same repair again. Build hashes, seals, ledger rows,
readiness state, and media/publication pointers only after the final accepted or
floored artifact so no downstream consumer can inherit a stale identity.

## 29. Reconcile rendered manifests before timeline ownership

A bank may author cues in its own compact vocabulary, but every renderer and
consumer needs one durable ledger contract. Translate producer-local IDs and
placements at the producer boundary. Legacy banks that author only sentinel
rows still need real `music[]` rows once a renderer has produced a manifest;
materialize those rows before the first timeline writer, not in a later
assembler that can no longer recover an interstitial's position.

Treat the manifest as rendered evidence, not permission to overwrite authored
intent. Recompute cue-spec identity from the ledger's authored prompt,
duration, placement, and anchor; only then may manifest paths and downstream
timing attach. A mismatch must stop that join loudly. For synthesized cues,
derive the same deterministic identity from the manifest-owned spec and bind
available ordered sentinels so the new row is equally auditable.

Anchors describe insertion order, not line type. Insert an interstitial before
either a dedicated music sentinel or an ordinary dialogue row, then continue
normal dialogue dispatch so voice-bus counts remain exact. Let the final audio
assembler alone mint visual mirror lines from canonical placement. At the
post-audio wire join, prove episode identity again, merge render-owned music
fields only on cue identity, and forward only mirrors whose cue passed that
join. Music may remain an optional creative bus, but when it renders its ledger
accounting cannot be optional.

## 30. Move serialized identity with a renamed artifact tree

Renaming a directory moves bytes, not the absolute paths serialized inside its
ledger. Treat a pending-to-final rename as one transaction: move the directory,
canonicalize the ledger filename, recursively rebase only absolute string
values contained by the old root, atomically persist the durable ledger, and
only then advance in-memory identity. Use path-component containment rather than
text prefixing, and leave external models, shared caches, source media, and OBS
destinations untouched. Make the transformation idempotent so a retry can
resume after the directory moved but before the ledger save completed.

A keyed merge does not preserve producer-owned rows that exist only on disk.
Add a narrow row join for each such producer instead of copying arbitrary stale
content. Prove the immutable run receipt first; then prove the parent object,
recompute its authored hash on both sides, validate producer-specific timing
and role fields, and reject duplicates. If either side has a freeze receipt,
both must carry the same value; episode-name fallback belongs only to two truly
legacy records with no receipt.

Post-rename consumers must join through the active durable owner, never through
the newest sibling by modification time. Where a stale pre-rename value is
already on a wire, add an explicit graph dependency on rename completion and
let same-receipt durable identity replace stale episode/path fields. Consumer
rescues should validate the durable episode against its directory and use the
freeze receipt whenever the caller carries it. A render is not qualified until
the final ledger can locate every asset without referring to the retired root.

## 31. Derived ledger metrics need one lexical owner and one final boundary

Word count is not `len(text.split())` once authored dialogue contains em dashes,
smart apostrophes, or punctuation glued for performance. Pick one explicit
lexical contract and put it in a dependency-free leaf that every producer,
repair pass, editor, readiness normalizer, aggregate writer, and freeze auditor
can import. A text mutation is one atomic operation over `text`, `char_count`,
and `word_count`; never rely on a later caller remembering the other two.

The durable save is still the ownership backstop. Re-derive row metrics from
canonical text before rolling them into cast, scene, root, and role-specific
meta totals, and reset empty aggregates instead of leaving values from a prior
composition. Preserve the incoming audit before self-healing so production
diagnostics still identify the offending producer. After the last legal text
mutator, refresh derived metrics once more before the final freeze audit.

Counts are derived state, not authorship. A count-only refresh must not rewrite
canonical text, delivery projections, accepted-line hashes, content-authorship
receipts, or seals. Enforce the boundary mechanically: scan production modules
for direct ledger-text assignments so a new sibling writer cannot silently
reintroduce split ownership.

## 32. Transport only model-owned fields across a bounded LLM context

An LLM should not reserialize a compiler-owned graph merely to author its leaf
text. Every repeated ID, parent, speaker, boundary, cue, neutral default, and
scene field spends output tokens, creates a fresh drift opportunity, and is
usually echoed again beside the failed artifact during typed repair. A response
can therefore be semantically small yet impossible to complete inside the
model's real context and output limits.

Define the wire artifact at the authoring boundary, not at the downstream
storage boundary. Give the model a closed list of stable IDs and only the prose
fields it owns. Require exact unique ID coverage, map by ID rather than response
order or fuzzy matching, and compile every mechanically derivable field from
the already accepted graph. Then run the complete downstream validator against
the compiled artifact; a compact transport must not become a validation bypass.

Repairs need the same discipline. Reinject a complete failed compact draft only
when it parsed, plus the smallest trusted authority needed to correct it. Drop a
malformed prefix instead of paying context to repeat truncation. Require both
the prompt and the full output reservation to fit before inference. A liveness
ladder should be finite and flat: a bounded same-slot ladder, then at most one
fresh independent slot, followed by an explicit deterministic floor only for
advisory craft defects. Record every attempt and terminal exhaustion truthfully.
Prove the maximum supported input and repair envelopes with the exact production
tokenizer, not a character estimate or a smaller surrogate prompt.

## 33. Separate render work from positioned timeline ownership

Full per-shot frame requests describe render work, not necessarily final video
duration. Once a post-audio ledger positions rows, opening, interstitial, and
closing cues may intentionally overlap dialogue at crossfades. Summing those
requests counts the overlap twice and creates body drift that a strict terminal
mux should reject.

Carry two explicit quantities: the sum of full render requests for workload and
the authoritative positioned timeline boundary for output. Quantize the accepted
audio-ledger duration upward to a complete CFR frame. In the planner, sort rows
stably by position and give a row only the visible interval before the next row
owns the boundary. End that interval at the earliest of its requested end, the
next start, or the timeline end. This trims overlap without stretching a short
clip across a genuine gap; equal-start collisions must be loud.

Quality telemetry must compare rendered frames with the planned visible slot,
not blindly with the original render request. Preserve requested, rendered,
visible, and overlap-trimmed counts so intentional edits do not masquerade as
engine underruns. A filesystem master probe is a cross-check and fallback: it
may shrink or grow a positioned timeline, while an unpositioned legacy sequence
keeps its historical grow-only behavior. Never widen mux tolerance or charge
body drift to a valid credits declaration.

## 34. Explicit delivery length is a producer contract, not a quality opinion

A requested word count can allow an inclusive tolerance without becoming
optional. Express the tolerance once as integer bounds, persist the target,
bounds, owner, canonical count, and exact character-text hash, and require every
producer family to finish inside that same contract before its final artifact is
sealed. A 180-word request accepts 163..200 words; the same law accepts 289..356
for 320. Floating ratios are planning aids, not the final delivery authority.

Length repair must be small, fresh, and progressive. Target one owned spoken row
at a time, alternate creative and technical writer slots, validate the complete
candidate, and retain only a strict move toward the band. A malformed call, an
unchanged valid response, or one bad sibling row must not discard prior accepted
progress or consume the remaining dynamic budget. Reject padding, repetition,
fake commercials or products, invented numeric claims, markup, stage directions,
and new visual/canon claims. Subjective taste remains fail-open; measurable
delivery exhaustion raises a typed error before hashes, readiness, or media.

**Amended 2026-07-30 (A-4).** "Exhaustion raises a typed error" is the end of
the ladder, not the first thing that goes wrong on it. A capacity failure now
carries a PHASE and only one of the two is terminal: a PRE-CALL refusal, where
the measured prompt leaves no room for the artifact, is deterministic and stays
loud on the spot; a call that RAN and used its whole output allowance without
stopping is stochastic and advances the ladder instead. The typed error is what
the ladder raises when it has actually spent its attempts.

Place each adapter at its last safe authoring boundary. Content-owned lanes fit
and rebuild their own proofs and hashes before assembly. Shared inline lanes fit
after story QA and spoken hygiene, re-scour the accepted rows, then stamp the
actual receipt and build reflections from those exact final rows. Preserve older
pass receipts as history instead of relabeling a later repair as an earlier pass.
After the final readiness normalization, freeze performs a read-only hash-bound
recount before video readiness. A miss becomes `needs_full_rerun`; freeze never
authors prose or mutates a content-owned seal.

## 35. Candidate exhaustion is not episode exhaustion

A bounded model call is healthy; a bounded episode-wide output ladder is not.
Keep each repair attempt and each producer candidate finite, but separate that
local liveness from the episode's acceptance state. Four consecutive calls that
make no strict distance progress retire the current candidate. They do not make
the requested episode impossible, and an LLM verdict never acquires that power.

Escalation is explicit: repair one owned row, try the alternate writer slot, then
discard the whole candidate and ask one producer to author a fresh complete
candidate. Alternate the producer priority on successive rerolls.
Freshness must be model-observable: when two logical slots resolve to one seeded
backend, vary a producer-owned prompt nonce or another validated generation input
so Candidate N cannot replay Candidate N-2 byte for byte forever.
Preserve only candidate-local diagnostics from discarded work; never let its text,
seal, hash,
readiness, or subjective score become authoritative. There is no fixed outer
model-output ceiling. Temporary provider failure stays pending, retryable, and
non-ready until a legal candidate arrives or the operator cancels.

Only deterministic impossibility may fail loud: invalid configuration, a graph
whose declared capacity cannot reach the band, corrupt schema ownership, or a
mechanical safety violation.

**Amended 2026-07-30 (A-4): this doctrine was right and the code did not
implement it.** The rule "only deterministic impossibility may fail loud" now
has a name in the transport -- a capacity failure carries a PHASE, and
`prompt_no_room` (the pre-call refusal, deterministic arithmetic) is the only
one that is terminal. `output_limit` -- the call ran and used its whole output
allowance without reaching a stop condition -- is the stochastic case this
lesson always meant to protect, and it advances the ladder. Before A-4 it
escaped as an unhandled error on attempt 1 of 3, which is how a three-call
budget spent one call and killed three legs of the live 45-word campaign. Two
things keep it honest: the phase lives in the module that owns capacity
arithmetic, so the transport and the retry policy cannot hold two opinions
about one failure; and being CATCHABLE is not being RETRYABLE -- both phases
are caught at the attempt boundary and the deterministic one is re-raised
untouched.

The final in-band ledger recount and text hash are
the acceptance judge. Subjective quality remains fail-open, and audio, video,
captions, credits, mux, and publication stay downstream of the hard final stamp.
Never manufacture prose, facts, products, advertisements, or numeric claims to
satisfy the counter.

## 36. A rejected fiction candidate is not a damaged episode ledger

**Operator ruling 2026-07-30.** On the canonical Sci-Fi route, a recoverable
JSON, schema, content-validation, safety, or output-limit defect exhausts and
retires only that finite candidate ladder. The producer then requests a fresh
complete model-authored candidate, with cancellation as the operator-controlled
stop. There is no fixed outer model-output ceiling and no deterministic canned,
summary, drop, or patch-in-place story floor. This is the controlling amendment
to conflicting readings of Lessons 3, 11, 26, 34, and 35.

The source article is evidence and inspiration, not a fictional continuity
contract. Accepted characters, events, dialogue, and plot may diverge completely
from the article or any discarded draft. Claims presented as factual and the
factual coda still resolve to validated source evidence. The accepted canonical
spoken artifact is the only prose admitted to the ledger; rejected prose, hashes,
seals, and readiness state never become authoritative.

Complete source access is a coverage problem, not a clipping permission. Preserve
the exact selected body and its route/index/count/UTF-8 byte count/hash receipt.
When it exceeds one P0 context, use overlapping windows whose union covers the
complete normalized source, validate locally, rebase exact offsets, merge
deterministically, and validate again against the complete A0.

Assemble and stamp the production ledger once, from the accepted canonical
artifact. The final line identity, graph, safety, recount, authorship, freeze, and
hash checks remain strict corruption and downstream-consumer guards; they do not
prove fidelity to an abandoned work of fiction. Cancellation and deterministic
configuration, source/security, provider, I/O, compiler, ownership, graph,
freeze, and proof failures remain loud. A raw unsafe candidate is retryable;
unsafe text that somehow remains after acceptance is an invariant failure.


## Sprint receipt

Record this at the end of every production sprint:

```text
SPRINT RECEIPT: PASS | FAIL
scope:
authoritative_writers:
durable_artifacts:
canonical_workflow_hash:
focused_tests:
full_suite:
bug_bible:
model_pairings:
30_word_receipts:
120_word_receipts:
720_word_receipts:
live_ledgers:
published_assets:
prod_bug_entries:
head:
origin:
remaining_risks:
```
