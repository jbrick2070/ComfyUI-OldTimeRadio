# Driver anchor -- written before Opus/Cursor fan-out

VERDICT: implemented EOS repair is ready for independent review; live cure unproven.
CONFIRMED: effective decoder and tokenizer EOS disagree on this installed snapshot;
LMFE supports multipleEOS but its integration builder originally copied tokenizer
only. No explicit native generateEOS was supplied. The new shared resolver/4native
callers/completion classification/LMFE cache use one primitive set and preserve
model/tokenizer owners. Actual EOS criterion and parser tests pass. The two source
correction bodies parse as complete JSON before trailing newline degeneration.

CONFIRMED: P1also loops inside a still-open string; EOS alignment cannot end an
incomplete JSON object and is not a cure for that observation. Template thinking
is disabled. The existing liveness guard ultimately exhausts the3attempt main
ladder; source correction stays within2. No fourth/fifth attempt occurred.

MUST-FIX: any verified remaining termination owner mismatch. No such missing owner
found after current diff and independent Einstein/Dewey review. Full/Bible checks
and requested external consensus remain pending.
SHOULD-FIX/VERIFY: inspect whether grammar sampling's cutoff suppresses closing
quotes in these actual outputs. Mechanism is documented in Bible12.100, but a
single run does not prove changing min_p/top_p is the right next patch; do not
introduce a prose-size bound. Preserve the failed attempt and qualify canonical
publication/source facts separately. No new hardware proof from component tests.
