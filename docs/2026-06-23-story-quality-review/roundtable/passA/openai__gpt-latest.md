<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: Mostly non-drama: conflict-shaped command fragments without stable characters, causality, or playable choices. Single biggest lever: replace the freeform beat planner with a deterministic “one room / one objective / one irreversible choice / named roles” beat template that forces continuity before line generation.

TOP FAILURE MODES:

1. Characters have no stable identity, role, or desire.
Quotes:
- EP1: “c03: I'll have my droids' code when they prove they can teach NC students like I can.” / “c03: I won't compromise my AI's integrity...” / “c03: I must insist, AI moderators, seize Charlie's droid.” / “c03: AI moderators, stand down.” / “c03: This is my prototype!”  
- EP13: “c02: I'm afraid I can't stay, Rufus.” / “c02: Folks, I've seen something alarming.” / “c02: Nonsense, John. My daughter's wedding is this weekend.” / “c02: Frank, it's Rainn.” / “c02: Rainn, I saw you on the phone.”
- EP4: “c02: ...Fingers dancing on the controls, I've got her, Merideth.” / “c02: ...The crew's on the move, Niran.” / “c02: Houston, this is Starfall. We're going manual.”
Why it fails:
The same speaker label repeatedly speaks as different people, holds contradictory authority, and changes relationship mid-scene. In audio drama, the listener tracks story through voices and wants. Here “c02/c03/c04” are not characters; they are line slots. That destroys stakes because nobody can choose, betray, concede, or change.
Symptom-or-root-cause:
Root cause. The pipeline composes one line per beat without a binding character-state ledger. The line composer apparently receives insufficient hard constraints about who each speaker is and what they want.

2. Every premise collapses into the same generic emergency-console standoff.
Quotes:
- EP2: “kill the lockout and give me manual control,” “bypass the safety locks,” “remote purge on the power core,” “manual lever.”
- EP5: “hit the hard-lock on that containment field,” “dump the core,” “manual bypass,” “The purge is the only way.”
- EP8: “server keys,” “servers are scrubbing themselves,” “those keys,” “let me have that drive.”
- EP12: “Override failed! Taking manual control,” “broadcast button,” “Signal's flipped back to me,” “The purge hits with the next signal flip.”
- EP15: “unverified broadcast,” “servers purge the stack,” “drives hit the table,” “the knob wide.”
- EP17: “wipe the buffer,” “manual override,” “automated scrub,” “scorched-earth switch.”
Why it fails:
The news briefs differ — classroom AI, fossils, spiders, El Niño, coal law, archaeology, satellites — but the produced scenes are almost all people fighting over keys, drives, levers, purge codes, vents, overrides, lockdowns, countdowns, and manual control. This is not adaptation; it is a crisis-word substitution engine. The dramatic mechanism is identical: someone threatens to press a button unless someone else hands over a code.
Symptom-or-root-cause:
Root cause. The beat planner is overfitted to “high stakes” as physical sabotage in a control room. It has no genre-specific conflict palette, so all briefs are laundered into the same mechanical jeopardy.

3. Causality is incoherent: actions do not follow from prior actions, and consequences do not land.
Quotes:
- EP3: “My hand's on the abort lever now. You initial that waiver or I decide what leaves the tanks next.” Then: “Broadcasting Krit's threat, implicating him to the board.” Then: “See how this key breaks, Terwilliger.” Then: “I light the fuse, Allan.”  
- EP9: “No code. Lockdown's on the console now.” Immediately followed by: “Code accepted.” Then: “Initiating atmo vent in five... four... three...” Then: “I'm overriding detonator safety.”
- EP6: “I'll bring this cave down on us all, Peter. You know I will.” Then: “You heard the shot.” Then: “what if we took just one more bone, a small sample?” Then: “On second thought, Peter, let's take a bit more.”
- EP14: “it won't let me send the strike command anymore.” Then: “give me the clearance code or I'll have to let the pulse hit both our heads.” Then: “the core is finally waking up to your commands.” Then: “I'm sorry, Charlie; it's time to clear the cache.”
Why it fails:
The scene does not build. It jumps from threat to counter-threat to unrelated technical event to arbitrary ending. No action visibly changes the situation. Characters announce buttons, codes, or danger states, but the story does not track: who controls what, what was attempted, what succeeded, what failed, what changed because of it.
Symptom-or-root-cause:
Root cause. The outline likely lists escalating beats, but no state machine governs the scene. Weak models cannot maintain hidden continuity over 14-18 independent line generations.

4. Dialogue is exposition and threat-noise, not playable human conflict.
Quotes:
- EP11: “Ned, stop! Youâ€™re forcing a mutation we canâ€™t contain, but if you think my hesitation is cowardice while this entire shelf suffocates into a graveyard of silt and dead calcium, then you're already looking at the extinction Iâ€™m trying to prevent.”
- EP14: “You should probably think about what happens to the floor plan if I hit-and then you'll have to explain why your team is currently staring down a scorched terminal instead of a data set.”
- EP17: “The countdown on the console is turning amber... if those telemetry packets hit the public feed in thirty seconds, there's no pulling them back. I need your clearance to wipe the buffer now.”
- EP8: “The humidity in this room is rising, Blake... and I'd hate for the evening news to start reporting on those private holdings in the Caymans before we find a more... cooperative way forward.”
Why it fails:
Characters explain the entire premise, mechanics, stakes, moral position, and threat in the same breath. They do not listen, misdirect, plead, lie, bargain, remember, or wound each other. They mostly narrate dashboard states and issue ultimata. That creates intensity-shaped text, but not drama.
Symptom-or-root-cause:
Symptom of the beat design plus weak line composer. The model is being asked to satisfy “stakes” one isolated line at a time, so every line tries to contain all stakes.

5. Endings are external news-summary resets, not dramatic resolutions.
Quotes:
- EP2: “The iron has latched, securing Swift within the grip of the Link spacecraft while its own internal gears grind into a permanent and silent stillness.”
- EP8: “The gavel has fallen and Australia's export laws are now rewritten, leaving the old coal docks silent as a new era of trade begins.”
- EP10: “The Billion CellÃ—Cell Project, now underway, has rewritten our understanding of cellular communication.”
- EP12: “With the spring trap now recorded, a fenced reserve stands marked across the Australian dunes.”
- EP17: “The Victus Haze Puma now rests in a public shipyard, its secrets buried under a new coat of civilian paint against the gray horizon of the MÄhia Peninsula.”
Why it fails:
The announcer resolves the news item from outside the scene, often contradicting or ignoring what just happened. The characters do not pay the price, make the choice, or experience the consequence. The ending says “the article outcome occurred” rather than completing the dramatic argument.
Symptom-or-root-cause:
Root cause at framing level. The intro/outro are preserving the news brief, while the scene body invents a generic crisis. The two layers do not share a dramatic spine.

6. Mechanical transcript defects break audio credibility.
Quotes:
- EP4: “c03: "Evacuate the site! Imminent debris impact.” / “c03: "Initiating manual override.”
- EP13: “announcer: I'm calling in every able-bodied person, Rainn. We're evacuating Middleton Beach.”
- EP18: “c04: Nia's voice should maintain its warmth and calculation, not shift to a more urgent or aggressive tone.”
- Multiple mojibake examples: EP6 “kÄkÄpÅ”; EP9 “El NiÃ±o”; EP17 “MÄhia Peninsula”; EP11 “Youâ€™re.”
Why it fails:
These are not craft subtleties; they are broken spoken artifacts. The TTS will literally read prompt leakage, malformed quotes, bad character labels, and encoding garbage. That tells the listener the fiction is machine debris.
Symptom-or-root-cause:
Symptom, but high-priority production hygiene. It indicates no deterministic post-generation sanitizer before freezing.

CROSS-EPISODE SAMENESS:

Yes. The episodes feel like one scene with different nouns.

Mechanism:
The pipeline seems to translate “make this dramatic” into a fixed emergency grammar:

- A sealed location: cockpit, lab, boardroom, cave, server room, station.
- A timed danger: countdown, pressure, dead zone, red light, purge, lockdown.
- A contested object: key, code, drive, lever, switch, feed, cable, waiver.
- Repeated imperative moves: override, purge, vent, bypass, lockdown, manual control.
- Final announcer returns to the factual article outcome.

Examples across unrelated premises:
- Classroom AI becomes “AI moderators, seize Charlie's droid” and “secure John Spender's droid.”
- Fossil preservation becomes “Rig the winch” and “Lockdown's triggered.”
- Spider conservation becomes “Override failed! Taking manual control” and “The purge hits with the next signal flip.”
- El Niño public warning becomes “servers purge the stack,” “drives hit the table,” and “the knob wide.”
- Astronomy data becomes “Yank that black cable” and “initiate the purge on the main array.”

The collapse happens because the beat planner is probably abstracting every brief to “race against time / standoff / high stakes,” then the line composer fills with a small set of sci-fi thriller tokens. The news-specific human conflict is lost before dialogue begins.

HIGHEST-LEVERAGE FIXES:

1. Add a deterministic character ledger before beat planning.
Where:
Outline/beat-planner input, not a soft instruction to the line composer.
Change:
For every episode, generate or template exactly three named roles and freeze them:
- Protagonist: wants X because personal reason Y.
- Antagonist: wants incompatible Z because reason W.
- Witness/pressure character: needs one concrete outcome and can force one consequence.
Every beat must name which character acts and whether their leverage increased or decreased.
Why it survives a weak model:
Weak models handle filling short slots better than maintaining implicit identity. Give them fixed names, jobs, wants, forbidden contradictions, and a current leverage value. Then lines become simpler.
Smallest-version:
Prepend a hard cast card to every line-composer call:
“CAST LOCK: NIA = flight controller, wants rescue but refuses to risk crew. DMITRI = mission director, wants satellite saved at any cost. KEVIN = technician, controls manual arm, fears killing oxygen. Do not assign any other role to these names.”
Also ban c02/c03/c04 in generated spoken text; convert to names before composition.

2. Replace generic escalation beats with a four-state causal scene machine.
Where:
Beat planner.
Change:
Use a fixed 12-beat structure:
1. Protagonist states immediate objective.
2. Antagonist blocks it with concrete reason.
3. Witness reveals ticking constraint.
4. Protagonist attempts action A.
5. Action A fails because antagonist did B.
6. Cost becomes personal.
7. Antagonist offers compromise.
8. Protagonist refuses or accepts with sacrifice.
9. Irreversible choice.
10. Immediate consequence.
11. Character reaction.
12. Announcer reports factual aftermath tied to the sacrifice.
Why it survives a weak model:
It removes the need for the model to invent escalation. Each beat has only one job and carries forward a visible state.
Smallest-version:
Add a state line to each beat: “Current control: Nia has code / Dmitri has lever / oxygen 4 minutes / satellite 2 minutes.” Require the next beat to change exactly one state.

3. Use premise-specific conflict palettes instead of the universal “override/purge/lever” palette.
Where:
Beat planner plus post-generation lexical gate.
Change:
Map the news domain to allowed conflict objects:
- Classroom AI: lesson plan, student results, parent board, demo failure, teacher contract.
- Fossils/cave: permit, sample bag, rising water, unstable ceiling, iwi/local authority consent.
- Climate warning: forecast uncertainty, mayor’s evacuation call, public panic, false alarm cost.
- Legal/environment: injunction, leaked memo, minister denial, witness testimony.
- Astronomy: observation time, classification, peer review, instrument failure.
Then disallow the generic crisis words unless domain-appropriate.
Why it survives a weak model:
This is deterministic vocabulary control, not taste. Weak models overuse “override/purge/manual lock” because those tokens are easy. A lexical denylist plus allowed object list constrains them.
Smallest-version:
Post-generation gate rejects/reroutes any episode if more than two of these appear: override, purge, lockdown, manual control, lever, key, drive, code, vent, scrubber, core, countdown. Instead of rerolling blindly, replace the beat’s conflict noun from the domain palette and recompose that one line.

4. Make the line composer write response pairs, not isolated lines.
Where:
Line-composer.
Change:
Instead of one line per beat, compose two linked lines at a time:
- Line A: character attempts pressure.
- Line B: other character responds directly to the last sentence, either “yes but,” “no because,” or “you’re lying because.”
Why it survives a weak model:
Local coherence over two lines is much easier than coherence over 18. It forces listening and prevents pure dashboard monologues.
Smallest-version:
For each line prompt include the previous spoken line and require the first five words to contain either the previous speaker’s name, object, or accusation. This is mechanical and checkable.

5. Shrink the story rather than chasing the 883-word target.
Where:
Length / beat count.
Change:
For 3-4 minute audio, use fewer beats with more coherent turns: 9-12 spoken turns, not 14-18 disconnected beats. The current short outputs are not failing because they are short; they fail because they are discontinuous.
Why it survives a weak model:
Weak models lose thread over longer sequences. A tight 10-turn argument with one irreversible choice is achievable.
Smallest-version:
Cap to 3 characters, 1 location, 1 contested decision, 1 physical object, 1 deadline. No subplot.

6. Add a deterministic transcript sanitizer.
Where:
Post-generation gate, before frozen TTS transcript.
Change:
Reject or repair:
- Prompt leakage: “voice should,” “tone,” “line,” “beat.”
- Bad labels: lowercase “announcer:” inside dialogue.
- Unbalanced quotation marks.
- Mojibake: “Ã,” “Ä,” “Å,” “â€™.”
- Speaker labels speaking as stage directions.
Why it survives a weak model:
No model judgment required. Regex and encoding repair catch most of it.
Smallest-version:
Hard fail any transcript containing “voice should,” “tone,” “Ã,” “Ä,” “Å,” “â€™,” or unmatched opening quotes.

7. Tie the announcer outro to the protagonist’s last choice, not just the article result.
Where:
Announcer outro generation.
Change:
Outro template:
“Because [character] chose [sacrifice/action], [news outcome] now happens, but [personal cost] remains.”
Why it survives a weak model:
It is a fill-in template, not a request for literary closure.
Smallest-version:
Pass the irreversible choice from beat 9 into the outro prompt and require it to be named in the first sentence.

WON'T WORK / AVOID:

1. Do not rely on “be less clichéd,” “make characters distinct,” or “avoid generic sci-fi controls.”
Why:
The evidence shows soft instructions are already being ignored. EP12’s “Override failed! Taking manual control” appears in a spider conservation story. EP15’s climate broadcast becomes “servers purge the stack.” The model needs constrained objects and state, not advice.

2. Do not add a critic/QA gate that merely says “this is incoherent, reroll.”
Why:
A weak model will usually regenerate the same emergency grammar. Without changing the beat structure or vocabulary palette, rerolls will produce another version of “override / purge / key / lever / lockdown.”

3. Do not increase target length to 883 words as the primary fix.
Why:
Longer incoherence is worse. EP11 at 430 words is more textured but still repeats overstuffed threat monologues: “we aren't just losing the sanctuary data, we are going to fry every oxygen scrubber...” The problem is continuity and dramatic choice, not word count.

4. Do not ask the line composer to solve story structure.
Why:
The line composer is downstream and writes one line per beat. It cannot repair broken character identity, missing causality, or premise collapse after the planner has already converted the article into “somebody threatens a purge.”

5. Do not depend on model self-policing for transcript hygiene.
Why:
EP18 contains literal prompt leakage: “Nia's voice should maintain its warmth and calculation...” EP13 has “announcer:” as a character line. These need deterministic sanitation, not another natural-language reminder.

6. Do not switch writer models alone and expect the system to be fixed.
Why:
All three model families exhibit the same structural collapse. Gemma produces somewhat fuller sensory pressure; Mistral produces unstable speaker identity and thin lines; Grok/slot-a produces compressed technical fragments. But all are being driven into the same control-room standoff by the pipeline.

MODEL-vs-SYSTEM:

The larger quality gap is pipeline/prompt design, not writer model alone.

Evidence:
- Mistral failures: extreme identity drift and underfilled scenes. EP1 assigns nearly everything to c03, including both sides of the conflict. EP13 has c02 as multiple townspeople.
- Grok/slot-a failures: fragmentary object-token plotting. EP12 repeats “Signal's flipped back to me” and “thumb... casing” without intelligible stakes. EP15 is knobs, reels, drives, arrays.
- Gemma failures: better surface pressure and longer lines, but same emergency template. EP5, EP8, EP14, and EP17 all rely on keys, drives, purges, vents, switches, servers, cores, and terminal threats.

Gemma is the strongest at producing audio-plausible intensity, but the “strong” episodes are still variations of the same scene. The system is forcing premises into a generic thriller apparatus. Model choice can improve sentence texture; it will not create stable dramatic architecture without a character ledger, causal state tracking, and premise-specific conflict constraints.