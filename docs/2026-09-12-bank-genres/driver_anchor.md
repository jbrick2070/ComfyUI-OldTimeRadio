# Driver anchor -- a genre per source bank, and one editable field

Driver: Claude Opus 5 (Cowork, 5080), sole judge. ONE reviewer (codex), briefed
to REFUTE. READ-ONLY: no git writes, no edits, no GPU -- four canonical legs are
in flight on :8000 and a reviewer touching the tree would corrupt them.

## What shipped, and is under review

Commit `037bf489` on `v2.0-alpha`. Files:
`nodes/_otr_music_palette.py`, `nodes/_otr_music_prompt.py`,
`nodes/stable_audio_theme.py`, `workflows/otr_canonical.json` + 93 generated
variants, and three test files.

## The operator's instruction, verbatim

> "sci-fi news is Detroit techno, media archive will be jazz quartet, original
> will be salsa, public domain Chicago house, people can fill in their own music
> in the widget fields"

then, narrowing it:

> "no, only My Story allows an original music prompt"

and, when asked why the field was framed as an override:

> "ok but these are the new defaults"

## What was built

| bank | palette key | leads with |
|---|---|---|
| scifi_news_pro | detroit_techno | Roland TR-909, sub bass, 128 BPM |
| media_archive | jazz_quartet | brushed drums, walking bass, sax |
| original | salsa_conjunto | congas and timbales, montuno, 100 BPM |
| public_domain | chicago_house | Roland TR-707, rolling bass, 122 BPM |
| shakespeare | early_consort | unchanged |

`Palette` gained `rhythmic: bool`. That one flag switches three things:

1. the rhythm section leads the prompt instead of being pushed behind strings;
2. the cue gets `NEGATIVE_PROMPT_RHYTHMIC`, which still bans noise, hiss,
   distortion, clipping and vocals but NOT loop / ostinato / drum machine /
   metronome / beat -- because those are requirements of the genre, and the
   underscore's negative would be asking for house while forbidding house;
3. a declared genre beats the period band in `story_palette`, because most
   public-domain sources are Victorian and a year-first read would return
   Romantic chamber music and make the instruction inaudible.

`music_style` is a new STRING widget on `OTR_StableAudioTheme`, blank by
default, honoured ONLY when the bank is `my_story`. Typed text is used verbatim
as both instruments and idiom. Whether it is rhythmic is decided by a keyword
regex, biased toward rhythmic on doubt.

## Context measured earlier the same day -- do not re-derive

A cue that asks for a thing while the negative forbids it TEARS. Measured: a
prompt reading "raw distorted TR-909" against a negative banning distortion
produced a broadband burst; removing the word cleaned it on the first re-roll.
Separately, lengthening a negative on a post-trained checkpoint took bursts from
11 to 47 over the same four control pieces. This is why (2) above exists.

## Questions -- answer with file:line

1. `story_palette`: walk it for every bank in the table, with and without a
   year, with junk meta, with `music_style` set and blank, and for a bank not in
   the table. Is there any input where a declared genre is LOST, or where a bank
   without one changes behaviour from before the commit? Is `bank_of` called
   twice where once would do, and can it disagree with itself?
2. `custom_palette` / `_RHYTHM_WORDS`: what does it do with a style that is a
   genre with no rhythm word in it ("shoegaze", "drone metal", "musique
   concrete"), with a style naming an instrument that is percussive but the
   music is not ("timpani roll"), and with unicode or punctuation? Is biasing
   toward rhythmic on doubt actually the safer failure, or does it strip the
   anti-loop negative from cues that needed it?
3. `negative_for`: it reads `getattr(palette, "rhythmic", False)`. Why the
   getattr on a frozen dataclass that always has the field -- is that defensive
   code that hides a real bug, or justified?
4. THE WIDGET WIRING, which is the highest-risk part of this repo. Check that
   `music_style` is last in `INPUT_TYPES`, last in `widgets_values` of the
   canonical graph AND of all 93 variants, and last in the node's `inputs`
   array. Confirm no link's `dst_slot` changed. Confirm the variants were
   GENERATED and not edited.
5. `_render_clips` copies meta with `dict(meta, music_style=...)` only when the
   string is non-blank. Is there any path where the theme node composes a prompt
   WITHOUT going through that copy, so a my_story override would be silently
   dropped? Check both the clip path and the batch path.
6. Argue the other side: the smallest thing here that is WRONG to ship, and
   anything in a comment or a commit message that overstates what was measured.

## Hard constraints

Deterministic and pure in the palette module; no new dependency; the canonical
JSON is the source of truth and variants are generated; the operator's ear is
the verdict on music and nothing here qualifies anything.
