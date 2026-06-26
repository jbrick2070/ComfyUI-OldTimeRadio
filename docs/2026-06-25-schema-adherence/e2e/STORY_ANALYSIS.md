# STORY-QUALITY ANALYSIS -- today's scripts (2026-06-25) -- are they getting better?

Read the SHIPPED spoken text (character lines) of every episode rendered today,
across both writer tiers, vs a 2026-06-24 baseline. Honest read below.

## Verdict in one line

**The machinery improved and the FRONTIER lane is a real step up; the LOCAL lane
still hits the model ceiling.** The biggest lever for "better stories" is being
able to run a frontier writer at all -- which the schema-adherence fix unlocked.

## Per-episode

**GPT (frontier) -- "Thumb on the Relay" -- STRONG (best of the batch).**
A coherent transparency-vs-fear drama: a broadcaster (Parry) fights a quarantine
council (Chidi) to unseal the biomedical logs. Real escalation ("sixty seconds to
explain why your council's silence is putting Crew-ten at risk"), distinct voices,
a genuine theme (trust under plague), a clean button ("Burn the truth into the
annals of history"). ZERO artifacts. This reads like an actual radio play.

**Grok (frontier) -- "M82 Request Line" -- MEDIUM-STRONG.**
Coherent, ambitious premise (an overnight jazz host finds an encoded warning in
James Webb/M82 data and fights an embargo). No leak/vocative artifacts. BUT grok
over-writes: long run-on sentences stuffed with jargon ("the encoded warning never
surfaces in M82's disputed signal again", "Cigar Galaxy starburst rates already
demand the coordinate broadcast"). Frontier-level coherence, but dense and
un-speakable in places vs GPT's clean lines.

**Gemma (local) -- "Choking on the Symphony" -- MEDIUM.**
More ambitious + poetic than mistral (a data-stream sabotage standoff; a strong
title-tying button "I'd rather choke on a symphony than breathe in this silence").
But muddled stakes (data / colony / air all at once), the "override / manual
override" console-standoff creeping back, and a malformed unclosed quote artifact
(`"Yuri, we're out of time...`). Better than mistral, below the frontier pair.

**Mistral (local) -- "Time Slipping Away" (LTX) -- MEDIUM-WEAK.**
A NASA-handoff/integrity conflict (signing away NASA's independence under a
deadline) -- actually a more coherent premise than its other run. BUT heavy
NEWS-BLEED: "President Trump's orders are clear", "It's Trump's legacy, your
appointment" (the space-policy news leaked the real name straight into fiction),
plus "Override launch codes." repeated (console standoff).

**Mistral (local) -- "Frozen Awakening" -- WEAK.**
The console-standoff pattern (gloves slipping, manual override, regulator jammed,
keyboard shaking) + the most artifacts of the batch:
- NEWS-BLEED into an incoherent place: "we're losing the funding!" / "the funding
  will dry up before we find anyone else alive" -- a climate/funding news fact
  dropped into a cryogenic-revival survival scene.
- STAGE-DIRECTION LEAK: `Gasping, "We're running out of time..."` -- "Gasping,"
  leaked into the spoken line (the leading-direction scrub missed it: "gasping" is
  not in the `_NARRATION_VERBS` whitelist).
- CAPS-NAME VOCATIVES: "YUKI MARTIN, no!", "MINA ECKELS", "YUKI MARTIN" shouted in
  caps.

**Yesterday baseline -- "Storm's Eye" (2026-06-24, mistral) -- WEAK/GENERIC.**
Pure mission-control survival ("Mission Control, this is Kane... advising an
abort"). NOTE: `story_contract=None` -- yesterday's episode had NO StoryContract at
all.

## What got BETTER (vs yesterday)

1. **Story machinery is live.** EVERY episode today carries a StoryContract
   (style + ending_tag) -- yesterday's "Storm's Eye" had `contract=None`. Today's
   settings are varied and specific (quarantine drama / jazz-host mystery /
   funeral-home / hospital vigil / government handoff) instead of a generic
   mission-control standoff. The style grammar is steering the SETTING.
2. **The frontier lane is unlocked and clearly better.** GPT and Grok produce
   coherent, thematic, artifact-free (GPT) stories. The schema-adherence fix is
   what lets you run them at all -- that is the single biggest quality lever.
3. **sound_palette now populated** (the fix shipped this session): every episode's
   canon now carries its style's audio world (e.g. "smooth jazz, a low mic, a
   coffee pour, a request line, rain on glass").

## What's still WEAK (the honest part)

The LOCAL models (mistral, gemma) get the varied SETTING from the grammar but
their LINE-LEVEL prose still hits the ceiling, and these recurring ARTIFACTS show
up in the SHIPPED text:
- **News-bleed (worst on mistral):** raw news facts (funding, "President Trump")
  drop into the fiction, sometimes incoherently (funding in a cryo-revival).
- **Stage-direction leak:** "Gasping," reached the spoken line (whitelist gap --
  add "gasping" + similar to `_NARRATION_VERBS`, the cheap fix).
- **Caps-name vocatives:** "YUKI MARTIN" shouted (the self-vocative wart).
- **Console-standoff / "override" repetition** still the local default reflex.
- **Grok over-writes** (frontier): dense jargon run-ons, less speakable than GPT.

## Recommendation

For a genuinely better STORY, the path is the frontier writer (GPT cleanest); the
schema fix makes that reliable. For the LOCAL lane, the highest-leverage cheap
fixes are: (a) widen `_NARRATION_VERBS` so leaks like "Gasping," get scrubbed,
(b) a news-bleed guard on the body lines (the raw news noun/name shouldn't enter
dialogue), (c) the caps-vocative scrub. These are content-only, model-agnostic
gates of the kind the story-quality sprints already use.
