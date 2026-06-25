# ANNOUNCER DESIGN -- scene-setting open + dramatic close + news coda (2026-06-24)

Operator product direction: the show TEACHES via story. The drama is the
delivery; the NEWS is the payload. So the announcer has three distinct jobs, and
the real-world news appears EXPLICITLY at the end as the teaching beat -- without
stealing the characters' climax.

This REFRAMES KILL 5 (announcer close): from "suppress the outcome" to "deliver
the real news deliberately, AFTER the characters land the drama."

---

## JOB 1 -- THE OPEN: set the scene, do not give it away
The announcer intro (beat b001) introduces, deterministically from the
outline + StoryContract:
- **TIME** -- era + time of day (`outline.time_of_day`, era/period).
- **PLACE** -- the setting (`outline.setting`).
- **WHO** -- the characters by name + their relationship/roles (the cast roster).
- **WHERE THEY ARE NOW** -- the opening situation / status quo / the tension as
  it stands at the start.
- Anything else that ORIENTS the listener.

HARD CONSTRAINT -- **no spoilers**: the open must NOT reveal the climax, the
outcome, the twist, or how it ends. No "...and they must decide whether to X"
that telegraphs the resolution. Orient + intrigue, then withhold. (Think the
Twilight Zone / OTR cold open: "Tonight: Los Angeles, a hospital where a surgeon
and an ethicist stand on opposite sides of an operating room. What follows is a
question of consent -- and who gets to answer it.")

BUILD: today the open beat intent is HARDCODED -- `_otr_outline._assemble_outline`
~1591 "Open the episode and orient the listener." -> a generic "gather 'round"
every episode. The build: feed the announcer-intro composer the time/place/cast/
opening-situation from the outline + contract, with the explicit no-spoiler
constraint. (Compose path: `_otr_line_composer.compose_announcer_intro`.)
ACCEPTANCE: the open names the setting, the era, and the characters, states the
opening situation, and contains NO climax/outcome/twist words.

## JOB 2 -- THE CHARACTER CLOSE: the drama lands in THEIR voices
The last voiced CHARACTER beat carries the DRAMATIC climax (governed by the
ending_tag / climax class). The announcer does NOT pre-empt it. This already
exists (the climax beat) -- the only requirement is that the announcer beat
after it does not restate the fictional resolution as if IT were the climax.

## JOB 3 -- THE NEWS CODA: the teaching beat, at the very end
After the drama, the announcer delivers the REAL news/fact -- "here's what
actually happened" / "what to know" -- as the educational payload.
- **Every episode, lightly labeled** (operator decision 2026-06-24): consistency
  is what makes it TEACH -- listeners learn the format (drama, then the real
  fact). A clean recognizable coda, not a hidden tag.
- Source the REAL fact, not the fictional outcome: distinguish "the real story
  this is drawn from: [actual news]" from "how the fictional characters' fight
  ended."

GROUNDING WIN: the close ALREADY pulls the real news -- `compose_announcer_outro`
(~2747) is driven by `news_close_brief` (that is WHY a close said "published in
The Lancet"). So the news coda is largely WIRED; the work is to (a) FRAME it as a
deliberate "here's the real story" coda, and (b) ensure it reads as the
real-world fact AFTER the character climax, not as the drama's resolution. The
old KILL-5 "force resolved=False / suppress" plan is REPLACED by this: keep the
news, frame it, protect the character climax.

---

## What this changes in the roadmap
- KILL 5 is REFRAMED: not "suppress the outcome" but "deliberate news coda +
  protect the character climax." (Lower-risk than the suppression branch -- the
  news_close_brief seam already exists.)
- NEW item: **the scene-setting OPEN** (replace the hardcoded generic intro with
  a time/place/character/situation intro + a no-spoiler constraint).
- Sequencing: this pairs naturally with KILL 2 (StoryContract) -- the open should
  draw the same setting/era/style the body uses, so build the announcer pass
  alongside or right after KILL 2.

## Open design questions (operator)
1. News coda: a fixed lead-in phrase (e.g. "The real story:" / "What actually
   happened:") or a varied-but-recognizable sign-off voice? (Recommend a light
   fixed lead-in for teachability.)
2. The open's length: a tight 1-2 sentence cold open, or a fuller scene-set?
   (Recommend tight -- it is a hook, not exposition.)
