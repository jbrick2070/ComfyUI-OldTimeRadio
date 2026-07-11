# CODEX -- make the episode ADMIT us into a story (paste this whole file into codex)

This one is a craft problem, not a crash. I want your taste as much as your code.
Prove your mettle: the easy answer here is bad, and I will know.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha  HEAD: 220066ef
You may READ anything. Write your proposal to `codex_framing.md` in the repo root.
Do not edit source, do not git add/commit/push. I run the git.
Label every claim CONFIRMED (you opened the file) or [ASSUMPTION].

## The defect

We just published the first working canonical episode. It is technically perfect
and dramatically inert. Here is its own transcript:

```
Dr. Amelia Hart   Our sensor trial was successful. The robot navigated the lab perfectly.
ANNOUNCER         But what about public deployment?
Jason Lee         We should proceed with caution. More testing is needed...
Maya Patel        I disagree. We've tested enough. It's time to deploy, but with close monitoring.
Dr. Amelia Hart   The robot encountered an unexpected obstacle. It corrected itself...
```

Look at the ANNOUNCER. It is taking a DEBATE TURN. It has been written as a fourth
person arguing in the room. It is not the voice of the show.

What is missing entirely:
- No billboard. The listener is never told what series this is, what tonight's
  episode is called, or even that they are listening to a radio drama.
- No scene-setting. No where, no when, no stakes. We open mid-argument and have to
  reverse-engineer the situation from the dialogue.
- No sign-off. The episode just stops.

The show is called SIGNAL LOST. You would never know.

An old-time-radio episode ADMITS you into a story -- it takes you by the elbow,
tells you where you are standing and why it matters, and then gets out of the way.
This one just begins one. That is the whole defect.

## Why it happens (structural -- the model is doing exactly what we asked)

Nothing in the contract has ever told the writer that an OTR episode HAS a frame.
- `announcer` is just another `char_id` in `CastPlanRowV4` / `ScriptLineV4`
  (nodes/_otr_scifi_codex.py), sitting alongside c01/c02/c03. The score is free to
  assign it to any beat like any other speaker -- so the writer reasonably treats it
  as a speaking part in the argument.
- No beat is required to be a framing beat. `make_advisory_word_blueprint` and
  `_score_graph_contract` pin word centers and the line manifest but say nothing
  about what the first and last beats are FOR.
- The pack seams `codex_play_system` and `codex_coda_contract_system`
  (nodes/story_packs/scifi_codex/scifi_codex_v1.json) never describe an opening or
  a sign-off.

We never asked for a frame. So we did not get one.

## The law you must not break

**Python judges. The LLM writes.** No Python rewriting, templating, or string-
stitching of story text -- no `text = "Tonight's episode: " + title`. That is a
hard operator rule and it is not negotiable. The frame must become part of the
CONTRACT the model writes INTO, plus a validator that FAILS CLOSED when the frame
is absent. The words themselves stay the model's, every time.

Corollary: a hardcoded stock phrase is a FAIL. If every episode opens with the same
canned sentence, you have written a template with extra steps. The frame must be
freshly authored per episode, from that episode's premise and stakes.

## THE HARD PART (this is the mettle test -- do not skip it)

`target_words` ranges 30..900. **At 30 words, a full radio billboard would eat the
entire episode.** "You are listening to Signal Lost. Tonight: NaviSystem Debate. In
a university robotics lab, three scientists decide whether to trust a machine with
the public..." -- that is already 30 words and we have not reached the story.

So the frame CANNOT be a fixed block. It has to scale. Answer this concretely:

1. What is the minimum viable frame at 30 words? What does the listener absolutely
   have to be given, and what can be implied by voice, music, and title card alone?
   (Note the episode already renders a title card and has a music cue system --
   `MusicCueV4`, `music_open` / `music_inter` / `music_close` char_ids. How much
   framing work can the MUSIC and the CARD do, so the words do not have to?)
2. How does the frame grow at 120 / 300 / 720 words? Give me the actual shape at
   each rung -- not "proportionally longer," but what NEW work the frame earns the
   right to do as the budget opens up.
3. Where does the frame's word cost come from? Does it come out of the story's
   budget (making the drama thinner) or is it additive? Look at
   `make_advisory_word_blueprint` + the per-beat word centers and tell me exactly
   how you would account for it, and what that costs the drama.

## The other real tension

The announcer must FRAME, not ARGUE. But the announcer is also our only voice that
can carry exposition between scenes ("Later that evening, in the empty lab...").
Where is the line? Give me a rule sharp enough that a VALIDATOR can enforce it --
i.e. something checkable from the score graph and line metadata, not vibes.
"Announcer may not take a dialogue turn inside a scene's argument" is my first
draft of that rule. Improve it or break it.

## Deliverable (codex_framing.md)

1. THE CRAFT ANSWER. What an OTR frame actually does, and what OUR frame should do
   for a show called SIGNAL LOST. Be opinionated. Cite a real convention if it earns
   its place; do not pad with radio history.
2. THE SEAM TEXT. The ACTUAL PROSE you would put into `codex_play_system` /
   `codex_coda_contract_system` to teach the model the frame. Write it as you would
   ship it -- this is the part I will judge hardest. It must produce a fresh frame
   per episode, never a stock phrase.
3. THE STRUCTURAL CONTRACT. The score/beat rules (first beat, last beat, announcer
   placement) expressed so `_score_graph_contract` / `_validate_radio_score_graph` /
   `_validate_script_post` can enforce them fail-closed. Name the functions and the
   exact checks.
4. THE SCALING TABLE. The frame at 30 / 120 / 300 / 720 words, with the word-budget
   accounting.
5. THE TESTS. What proves the frame exists and that the announcer never argues.
6. BLAST RADIUS. This must work for every content-owned lane (codex, gemini, sonnet,
   fable2), which have different pass ladders -- Gemini drafts per-scene, Sonnet per
   line. What changes per lane, and what is genuinely shared?

Show me something I would not have thought of. That is the assignment.
