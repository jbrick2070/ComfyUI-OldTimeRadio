<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: The craft is abysmal—a disjointed series of contextless, melodramatic threats over control consoles—and the single biggest lever is structurally banning "override/purge/console" tropes in the beat planner to force actual premise-specific dramatization.

TOP FAILURE MODES:
1. THE GENERIC CONSOLE STANDOFF (Root Cause)
- "Give me the authorization now, or I'm hitting the remote purge on the power core and we all go home in pieces." (Ep 2)
- "I need those keys before that mahogany table becomes a funeral pyre." (Ep 8)
- WHY IT FAILS: It replaces the actual premise (teaching AI, corporate law, paleontology) with a generic action-movie countdown. There is no thematic conflict, just a race to pull a lever. 

2. SPOKEN STAGE DIRECTIONS & PROMPT LEAKS (Symptom)
- "...Fingers dancing on the controls, I've got her, Merideth." (Ep 4)
- "Nia's voice should maintain its warmth and calculation, not shift to a more urgent or aggressive tone." (Ep 18)
- WHY IT FAILS: It shatters the audio illusion. The line-composer model doesn't understand the difference between character speech and acting instructions/action tags.

3. UN-EARNED ESCALATION TO LETHAL FORCE (Root Cause)
- "I'll bring this cave down on us all, Peter. You know I will." (Ep 6 - A story about fossils)
- "I’m burning every scrap of data in this core." (Ep 11)
- WHY IT FAILS: Characters jump from zero to murder/suicide in three lines. Without establishing the physical space, the relationship, or the stakes, these threats are just meaningless noise.

CROSS-EPISODE SAMENESS:
Yes, 18 distinct premises completely collapse into the exact same scene: "Three people screaming at each other over a glowing button/lever/drive while a gauge goes into the red." The mechanism of collapse is the beat-planner. Weak models equate "dramatic conflict" with "immediate physical danger." Because they lack the semantic depth to dramatize a legal dispute (Ep 8) or teaching methods (Ep 1), they default to their training data's strongest trope: a sci-fi reactor meltdown. 

HIGHEST-LEVERAGE FIXES:
1. THE BANNED-WORD TETHER (Pipeline: Outline/Beat-Planner)
- THE CHANGE: Inject a hard-coded list of forbidden words into the beat-planner prompt: "DO NOT USE the words: override, console, lever, purge, switch, core, lockdown, detonate." 
- WHY IT SURVIVES A WEAK MODEL: Small LLMs ignore "be subtle," but they are generally obedient to explicit negative token constraints. If they can't use "override," they are forced to find conflict in the actual premise.
- SMALLEST VERSION: Append a strict "BANNED WORDS:" list to the system prompt of the beat planner.

2. SEPARATE ACTION FROM DIALOGUE (Pipeline: Line-Composer)
- THE CHANGE: Force the line composer to output a strict JSON format: `{"internal_action": "...", "spoken_dialogue": "..."}`. Only send `spoken_dialogue` to the TTS engine.
- WHY IT SURVIVES A WEAK MODEL: It gives the model a dedicated trash can (`internal_action`) to dump its stage directions ("Fingers dancing on the keyboard", "Nia's voice should maintain..."). 
- SMALLEST VERSION: Change the composer output schema to two keys and parse out the dialogue.

WON'T WORK / AVOID:
1. QA/CRITIC GATES: Do not use a second pass to say "This is too melodramatic, rewrite it." Weak models cannot self-correct tone; they will just rewrite the exact same console standoff with slightly different synonyms.
2. "SHOW, DON'T TELL" PROMPTS: Soft instructions fail entirely. The model does not know what "show" means in an audio-only medium.

MODEL-vs-SYSTEM:
The system is failing the models. While the models are weak (evidenced by prompt leaks in Grok/Slot-A and mistral-nemo), the pipeline asks them to generate drama without structural guardrails. The beat-planner is allowed to hallucinate generic sci-fi stakes, and the line-composer has no schema to separate action from speech. Gemma-4-12b produces slightly longer, more coherent sentences (Ep 2, Ep 11), but still falls victim to the system's "console standoff" trap. Fix the pipeline constraints, not the models.