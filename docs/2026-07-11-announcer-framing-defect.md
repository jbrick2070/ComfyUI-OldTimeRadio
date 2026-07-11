# 2026-07-11 -- The episode starts a story instead of admitting you into one

Operator note after eyeballing the first published canonical Codex 30w episode
(`signal_lost_navisystem_debate_20260711_084655_..._final.mp4`, commit ccaa869d).

Craft defect, not a crash. Logged so the 720w bake-off does not scale it up.

## The symptom, from the episode's own transcript

```
Dr. Amelia Hart   Our sensor trial was successful. The robot navigated the lab perfectly.
ANNOUNCER         But what about public deployment?
Jason Lee         We should proceed with caution. More testing is needed...
Maya Patel        I disagree. We've tested enough...
```

The ANNOUNCER is taking a DEBATE TURN. It has been written as a fourth character
arguing in the room, not as the voice that puts the listener into the show.

What is missing entirely:
- No billboard / cold open: the listener is never told what series this is, what
  tonight's episode is called, or that they are listening to a radio drama at all.
- No scene-setting: no where, no when, no stakes. We open mid-argument and must
  reverse-engineer the situation from the dialogue.
- No sign-off: the episode simply stops.

An old-time-radio episode ADMITS you into a story. This one just begins one.

## Root cause (structural, not a "bad model" problem)

Nothing in the contract has ever told the writer that an OTR episode HAS a frame.

- `announcer` is just another `char_id` in `CastPlanRowV4` / `ScriptLineV4`
  (alongside c01/c02/c03), and the score is free to assign it to any beat like any
  other speaker. So the writer reasonably treats it as a speaking part.
- No beat is required to be a framing beat. `make_advisory_word_blueprint` +
  `_score_graph_contract` pin word centers and the line manifest, but say nothing
  about the FUNCTION of the first and last beats.
- The pack seams (`codex_play_system`, `codex_coda_contract_system` in
  `nodes/story_packs/scifi_codex/scifi_codex_v1.json`) never describe the opening
  billboard or the sign-off.

The model is doing exactly what it was asked. It was never asked for a frame.

## Fix shape -- Python judges, the LLM writes

This stays inside the LLM-first rule: no Python rewriting of story text. The frame
becomes part of the CONTRACT the LLM writes into, and a validator that fails closed.

1. Pack seam: teach `codex_play_system` what an OTR opening and sign-off ARE --
   series name, episode title, the hook, where/when we are, what is at stake --
   and that the announcer FRAMES, never argues.
2. Score/graph contract: require the FIRST beat to be an announcer framing beat and
   the LAST to be an announcer sign-off beat. The announcer may narrate BETWEEN
   scenes; it may not take a dialogue turn inside a scene's argument.
3. Validator: extend the script post-validator to fail closed when the frame is
   missing or when an announcer line sits inside a scene's dialogue exchange --
   the same fail-closed discipline as the shot_id / boundary graph checks.
4. This is a deterministic, checkable STRUCTURAL contract. The words themselves
   stay the model's.

Applies to every content-owned sci-fi lane (codex, gemini, sonnet, fable2), not
just Codex -- verify each pack's seams and the shared cast/announcer contract.

## Status

OPEN -- operator is planning a Codex revision pass for this. Independent of the
720w context/budget work. Do not let the 720w bake-off ship 6x more of a frameless
episode.
