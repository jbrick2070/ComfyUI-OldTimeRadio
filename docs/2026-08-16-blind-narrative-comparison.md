# Blind narrative comparison: July bake-off vs the August pipeline

**Operator question (2026-08-16):** do today's episodes tell better stories
*from a human narrative perspective* than the last historical bake-off --
not better structure, better STORIES?

**Method.** Ten spoken-word transcripts, five per era, shuffled and labelled
A-J with the era mapping withheld from the judge. July side = the mid-July
bake-off arms INCLUDING the cloud writers (`scifi_sonnet`, `scifi_gemini`,
`original_codex56sol`); August side = the 08-15 full-length legs and the
08-16 six-bank sweep. Lengths overlapped deliberately (13-37 recent vs 6-16
bake-off) so the judge could not sort by size. Judge: Fable, narrative
craft only, told nothing about eras or dates. Regenerate with
`scratchpad/make_transcripts.py` (seeded shuffle, 20260816).

**Result: the split was recovered 10/10, blind.** Judge's "stronger" group
= B/D/E/G/H = every August episode; "weaker" = A/C/F/I/J = every July
episode. Ranking put August at #1 and #2 and July at the bottom three.
Confidence self-reported at ~75%; the actual answer was perfect.

**The mechanism, and this is the load-bearing part.** In the judge's own
framing: *"The strong group fails at drama. The weak group fails at being
finished."* August episodes are complete broadcasts whose defects are
AUTHORED (a debate where a scene belonged; a fumbled final line). July
episodes show PIPELINE BREAKAGE on air: an abandoned setup fragment, a
climax skipped between two lines, spoken camera directions, and -- worst
-- an episode broadcasting its own validator log as dialogue
("Resolved advisory defect").

**So the floors rose; the ceilings barely moved.** Best July episode 6/10
(`original_radio_v2`, a LOCAL lane), best August 7/10. Worst July 1/10 vs
worst August 3.5/10. The August sprint bought RELIABILITY, not brilliance
-- which is exactly what the 2026-08-04 "story quality is done" ruling
predicted and priced.

**The cloud arms did not win, and the reason matters.** `scifi_sonnet`
ranked LAST (1/10) -- not for its prose but because the pipeline printed QA
output into its script, and `scifi_gemini` ranked 9th as an abandoned
fragment. A frontier writer inside a leaky pipeline loses to a local writer
inside a sound one. This is evidence FOR the local-default ruling, not
against it.

**Two findings the judge surfaced unprompted, both corroborating other
work this session:**
1. **Name staleness is audible.** Three of ten episodes feature an "Elias",
   twice in the identical Elias/Sarah pairing. Independent confirmation of
   the five-Elias tally in the 2026-08-16 handoff;
   `scripts/otr_name_randomness_lab.py` still has never been run.
2. **A live character-attribution defect** in the Moby-Dick episode
   (`signal_lost_the_price_of_a_soul_20260815_132024`): Stubb speaks
   Ahab's soul-binding rhetoric and is called "master of this vessel",
   while Ahab argues the skeptic's side. That is a CORRECTNESS defect
   (identity/fidelity), carved out of the story-quality freeze, not taste.
   It is also exactly the `JUDGE_ATTRIBUTION` class that ships disabled
   after measuring unstable -- so this is a live instance for that record,
   not a new proposal to re-enable it.

**The one weakness spanning BOTH eras** -- the judge's answer to "what is
still worth fixing": *no episode trusts its characters to land the ending*.
Every one of the ten either stops dead, jump-cuts the climax, or hands the
resolution to the ANNOUNCER to interpret ("that's the detail that
lingers"). The two best moments in all ten were both character-carried
exits. Highest-leverage change if the operator ever reopens craft: the last
dramatic beat belongs to a character; the announcer may FRAME but never
INTERPRET. **Recorded as an observation, not scheduled** -- reopening it is
the operator's call under the 2026-08-04 ruling.
