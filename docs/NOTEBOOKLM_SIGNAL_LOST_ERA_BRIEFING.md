# Signal Lost: Alpha, Beta, and the Folder That Looked Like Junk

A source briefing for a NotebookLM explainer. Measured 2026-09-15 from two on-disk archives and matching episode ledgers. This is not a marketing document. It is what the files actually are.

Upload this file into NotebookLM as a source. Ask it for an Audio Overview titled something like: "Was Signal Lost a vibe-coder hero's journey, or a pile of junk?"

---

## The one-sentence verdict

It is a hero's journey that *looks* like junk, because for five months every finished episode was named so badly that a thousand real radio plays truncated to the same unreadable tail. The naming scheme was changed in September 2026 so the watch folder could finally function as a leaderboard. The junk is real, but it is the minority: proofs, still-flat shorts, one story ground 36 times, and a different show (Earthsearch) occupying the live watch directory.

---

## The two archives, and why they disagree

### Live watch folder

Path: `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs`

This is the folder OBS is supposed to watch. It currently holds 1,115 videos and 29.3 GB.

That number is a trap. 1,094 of those files (27.4 GB, 93 percent of the bytes) are Earthsearch Episode 1 masters, inserts, and editorial intermediates -- a different production sitting in the Signal Lost watch directory. Only 18 files are actual Signal Lost broadcasts (about 1.8 GB). Five of those 18 already use the new coded names. One file is a 2.7 MB Vidu smoke test.

If you open live `otr/obs` and ask "is this junk?", you are mostly looking at Earthsearch, not at Signal Lost.

### Ancient / buggy archive

Path: `E:\Old Random Project Folders\Old_time_radio_Los_SIgna-\Ancient_Biuggy_aRhicve`

This is the real Signal Lost leaderboard. 1,480 videos, 95.0 GB, April through September 2026.

| Bucket | Files | Size | What it is |
| --- | --- | --- | --- |
| Legacy broadcasts | 908 | 70.2 GB | Real episodes, old unreadable names |
| Coded broadcasts | 41 | 4.0 GB | Real episodes, new leaderboard names |
| Legacy shorts under 20 MB | 384 | 4.7 GB | One-acts, fails, or stills pretending to be shows |
| Alpha same-story grind | 36 | 13.1 GB | *The Last Frequency*, rendered over and over |
| Still-flat shorts | 32 | 416 MB | Motionless engine proofs, not programs |
| Proofs and tests | 31 | 1.5 GB | Regression, soak, Mac lightning proofs |
| Other / tiny | 48 | 1.0 GB | Leftovers |

By file count, about two-thirds of the ancient archive is a real episode. By bytes, about 78 percent is (74.2 GB of broadcasts). The remaining fifth is the journey's scar tissue: learning to render, proving engines, and refusing to delete the takes.

---

## Why the naming scheme changed

Until early September 2026, every published file looked like this:

`signal_lost_<title>_<timestamp>_silent_procgen_blended_captioned_with_credits_final.mp4`

In a Windows file browser the title dies at the truncation point. Every row ends on the same lie: `..._silent_procgen_blended_captioned_wit...`. "Procgen" is a compositing stage, not the video engine. The operator could not tell cartoon from Shakespeare, Kokoro from Bark, or news from Folger, without opening the ledger.

On 2026-09-07 the published name became a seven-field leaderboard, using four-character codes (five for the writer):

`<title>_<timestamp>__<style>__<video>__<image>__<tts>__<bank>__<writer>__<music>_final.mp4`

Example, sitting in live obs right now:

`the_count_of_three_20260914_103336_silent__vart__stmo__zimg__koko__myst__g412__sa3_final.mp4`

Decoded in order: visual style video_art, video lane still-motion, image Z-Image Turbo, TTS Kokoro, source bank My Story, writer Gemma 4 12B, music Stable Audio 3.

That is why the scheme changed. Not aesthetics. The folder was the success signal, and the old names made success indistinguishable from junk.

A known scar in the new scheme: five My Story dinner-table episodes published on 2026-09-11 carry `__unk__` in the bank slot (`dinner_memories`, `dinner_recollections`, `dinner_table_remembrances`, `the_bay_area_table`, `the_la_table`). The `myst` code was added later. Those names were left as a record of what actually ran.

Combined across both archives, 79 files already carry the coded name. They are the only rows you can rank without opening a ledger.

---

## Era one -- Alpha, April 2026: one story, many bodies

The ancient archive opens on 2026-04-05 with regression tests, then immediately with *The Last Frequency*.

That single title occupies 36 video files and 13.1 GB. Companion WAVs and `_treatment.txt` files sit beside them. Early takes are 30 to 60 MB. Overnight the same story balloons past a gigabyte. The largest *Last Frequency* files in the folder exceed 1.1 GB. This is not a series. It is one radio play being reincarnated until the pipeline stops crashing.

Alpha lesson: the hero's journey starts as "can we emit a file at all?" Volume is not quality. A 1.2 GB mp4 of the same episode is usually a codec or duration accident, not a better story.

April plus May are small (40 then 56 dated files). The factory has not started. The operator is still teaching the machine to finish.

---

## Era two -- Beta, June to August 2026: the factory

Then the counts explode.

- June 2026: 389 dated files
- July 2026: 354
- August 2026: 499 (the peak month)
- September 2026: 140 (and the names finally become readable)

This is Signal Lost as a daily. The show prefix is constant. The titles start to sound like a program: *The Rags of Father's Fury*, *The Spore of St Peter*, *The Coil of the Speckled Band*, *Ink and Inheritance*, *The Weeping Seal*, *The Apprentice's Number*, *The Humidity of History*, *Scour the Wood Until It Bleeds*.

Matching 919 ledgers to published timestamps gives a typical episode of about 198 spoken words. That is a one-act radio play, not a novel. Many coded Shakespeare and archive episodes have eight spoken lines after the announcer and a music cue. The widget allows one to seven acts. The archive mostly delivered one.

Beta lesson: once the machine can finish, it finishes a lot. Story banks mutate in public (`science_news` to `scifi_news` to `scifi_news_pro`; `original_codex56sol`; Fable and Sonnet forks). The ledger, not the filename, is the only honest record in this era. That is the defect the September rename exists to close.

---

## Era three -- Coded names, September 2026: the leaderboard

Seventy-nine coded files, 5.0 GB, are the first ranking you can do by eye.

### Visual styles (the look)

The five house styles are almost tied. This is a roll, not a house look yet.

- video_art: 12 files, 960 MB
- cartoon: 12 files, 691 MB
- visual_storybased: 12 files, 733 MB
- anime: 12 files, 899 MB
- storybook_engraving: 11 files, 820 MB
- paper_origami: 6 files, 224 MB
- archival_documentary: 6 files, 381 MB
- shakespeare_stage_realism: 6 files, 235 MB
- sci_fi_radio: 1 file (a short)
- recur_frac: 1 file, *Magnetic Pulse*, 216 MB -- rare, and large

### Video lanes (how the picture moves)

Here the junk and the show separate cleanly.

- still flat (no motion): 32 files, 416 MB -- proofs. Small, motionless, IndexTTS2 plus Nemo plus Stable Audio 3. These are "did the graph complete?" not "is this tonight's episode?"
- viz camera: 13 files, 1.7 GB -- the heavyweight picture lane among coded names
- haunted AnimateDiff: 9 files, 1.2 GB
- LTX 8GB: 6 files, 426 MB
- still motion: 5 files, 503 MB
- still pan: 5 files, 230 MB -- this is the My Story dinner-table cluster
- LTX 2.5 foley+, mime, video; lightning AnimateDiff; viz MXC CPU; still word: a handful each

If you rank by bytes instead of count, viz camera and haunted AnimateDiff are the actual show. Still-flat wins the file count and loses the argument.

### Image stills

- Z-Image Turbo: 43 files
- no stills / viz-only: 24 files (the camera and haunted lanes often skip a still engine)
- SD 1.5: 6
- Flux2 Klein: 3
- unk: 3 (another honesty scar -- the image slot had no code)

### TTS / voice banks

- Kokoro: 39 files, 4.2 GB -- the voice of the real broadcasts
- IndexTTS2: 38 files, 729 MB -- almost even on count, but those are the small still-flat proofs
- Bark: 2 files, 150 MB -- *Twisted Grain* and *The Warp Near the Spool*

Kokoro is the house announcer. IndexTTS2 looks like a peer until you weigh the files.

### Story banks

Among coded names:

- scifi_news_pro: 17 files, 1.6 GB
- shakespeare: 17 files, 1.0 GB
- original: 15 files, 783 MB
- media_archive: 13 files, 955 MB
- public_domain: 11 files, 404 MB
- My Story before the myst code (`unk`): 5 files, 230 MB
- my_story (`myst`): 1 file, *The Count of Three*, 127 MB

The six banks are real and roughly balanced. News and Shakespeare lead. My Story arrives late and under-counted because of the missing code.

Among all 919 matched ledgers, including the unreadable era, the bank names themselves are a fossil record: `science_news`, `scifi_news`, `scifi_fable2`, `scifi_codex`, `original_radio`, `public_domain_story`, then the v2/v3 suffixes. The product simplified. The archive remembers every name.

### Writers (local LLMs)

- Mistral Nemo: 36 files, 600 MB -- count leader, size loser (it wrote the still-flat proofs)
- Qwen 3.5 4B: 20 files, 2.1 GB -- the workhorse of mid-September dailies
- Gemma 4 12B: 17 files, 1.8 GB -- late, heavier, live-obs default
- Gemma 4 E2B: 5 files
- Gemma 4 E4B: 1 file, *The 522 Field*, 299 MB

### Music

- Stable Audio 3: 54 files, 2.3 GB
- MusicGen: 25 files, 2.7 GB

MusicGen fewer times, more bytes. Stable Audio 3 is the default stamp on the coded era.

---

## Acts

The live widget offers 1 through 7 acts. The archive did not chase length.

Typical matched ledger: about 198 spoken words, often eight dialogue lines plus an announcer open. Shakespeare rows are explicitly "a scene from Act One," not the whole play. My Story *The Count of Three* is act_count 1, 497 words, 47 lines, 187 seconds -- the long end of the one-act habit, not a three-act swing.

Word count was never the gate. The topology is a short radio play with a start, a middle, and an end inside one scene. That is the form Signal Lost actually shipped.

---

## File sizes as a ranking, not a virtue

Gigabyte files in this archive are usually Earthsearch masters or alpha *Last Frequency* accidents. A healthy Signal Lost daily in the coded era is 80 to 160 MB. Still-flat proofs sit at 9 to 20 MB. Legacy shorts under 20 MB are the gray zone: some are honest one-acts, some are failed motion.

The live folder's largest files are Earthsearch EP01 masters at 2.1 to 2.8 GB. They will dominate any naive "biggest episode" leaderboard and they are the wrong show.

Among ancient Signal Lost broadcasts, the heavyweights are beta titles from late August: *The Rags of Father's Fury* (486 MB), *The Summer Frequency Sequence* (447 MB), *The Last Reading* (439 MB), *Bells Beneath Sardis* (424 MB). Those are long or high-bitrate dailies from before the coded names, not a separate premium tier.

---

## The stories (what the machine actually said)

These are representative, not complete. 891 unique titles exist in the matched ledgers. Dumping them all would be a phone book. NotebookLM should treat the following as the flavor of each bank.

### My Story -- *The Count of Three* (2026-09-14)

Bank myst. Style video_art. Lane still motion. Image Z-Image Turbo. Voice Kokoro. Writer Gemma 4 12B. Music Stable Audio 3. 127 MB. One act. 497 words. 187 seconds.

Listener idea, carried almost whole: in a sunlit toy playground, two small orange toy boots named Stomp and Tiptoe play with a tennis ball. Whiskers, a toy cat statue, sits by the fence with painted eyes shut. They do everything on a count of three because it is the only way they agree. Rainbow smoke, three plastic boxes (red, green, blue), a toy hairdryer, a giant candy bar, a humming rainbow candy. They melt chocolate with the hairdryer because the play stove has a painted flame and no heat. They set out one bowl too many and carry it to Whiskers. Behind him, a thin curl of smoke. They decide not to ask.

Cast: ANNOUNCER, Stomp, Tiptoe.

Opening flavor: "Kick it! Kick it now!" / "Wait. We haven't counted yet. One... two... three. Now."

This is the point of My Story: a listener's own idea, not a scraped bank, produced as a broadcast. Five dinner-table My Story episodes from 09-11 (*Dinner Memories*, *The Bay Area Table*, *The LA Table*) are the same lane with the bank code still missing.

### Shakespeare -- *The Trembling Silver Signet* and *The Mask of the Jester*

Folger text, not paraphrased. *Signet* is King Lear Act One Scene One: the map, the divided kingdom, crawl toward death. Cast ANNOUNCER, LEAR, GONERIL. Anime look, viz camera, Kokoro, Qwen 3.5 4B.

*The Mask of the Jester* is Twelfth Night Act One Scene Five. Fool, Olivia, Malvolio. The announcer asks what truth remains hidden behind the mask. This is the fidelity rule in the files: Shakespeare's language is carried as written. There is no violence or swearing filter on the adaptation lane, because that filter had been forbidding blood in Macbeth.

Other Shakespeare titles in the coded set: *Goneril's Exuberant Devotion*, *The Cock's Trumpet*, *The Weight of the Iron Crow*, *The Lantern Burns Bright While Slander...*, *The Weight of the Dead Wife's Phantom*.

### Sci-fi news -- *The Texture of Truth*, *The Apprentice's Number*, *The Quiet Audit*, *Homecoming Orbit*, *Magnetic Pulse*

News-bank episodes dress current science as old-time radio.

*The Texture of Truth*: Elias and Martha in a granary the Historical Society wants torn down. A machine hums. Touch versus sight. Detroit techno bed. Anime. 567 words -- long for this corpus.

*The Apprentice's Number*: Leo, Dr. Voigt, and Chen over a living laboratory colony whose signal is routing to the wrong strain. Tenure, a paper, a broadcast they should not start. Visual storybased, LTX 8GB, SD 1.5.

*The Quiet Audit* sits in live obs (113 MB, cartoon, still motion, Kokoro, Gemma 12B, Stable Audio 3). A news-bank daily that made it into the watch folder after the rename.

*Homecoming Orbit*: 239 MB, anime, viz camera, Qwen, MusicGen. One of the largest coded news files.

*Magnetic Pulse*: the only recur_frac coded episode, 216 MB, viz camera. A style that almost never won the roll, and when it did, it arrived huge.

### Media archive -- films that are about saving film

This bank is the show talking about itself: reels, acetate, heat, catalogs, vaults.

*Snipping the Spool of Time*: Mrs. Gable, Lemmy, Mali Halpert, Som Hayes. Rising heat, a grant, a gate, a fragile spool. Storybook engraving.

*The Shivering Gauge*: processing plant, acetic acid, cooling vats, a pressure gauge that is "behaving with a deliberate, unsettling intent."

*The Humming Reel That Refused to Go Silent*: Eagle Theatre, 250th anniversary screening, suppressed footage, a preservation committee about to seal the archive.

*The Ink of the Archive*: restricted vault, a reel that was never meant to be seen, a requisition slip for acid-free housing.

*The Warp Near the Spool*: Mira and Kent, catalog card marked Fragile Handling Only, Bark voice, cartoon, LTX 8GB, Flux2 Klein. One of only two Bark-coded files.

The media-archive stories rhyme. That is not an accident and it is not quite junk. The source is film-preservation RSS. The machine writes the same attic, over and over, because the attic is the feed.

### Public domain -- *The Papers of the Lonely Desk*

Stephen Leacock's *Nonsense Novels*, recast as a missing prince, a secretary, a prime minister, a Great Detective. Video art, haunted AnimateDiff, Kokoro, Gemma 12B. 95 MB. The announcer asks whether secret notes bring justice or ruin.

Other public-domain coded titles include *Blood in the Gutters* and *Scarecrow's Shadow*.

### Original -- *The Weight of Velvet*, *The Stone That Won't Move*, *Twisted Grain*, *The Silk's Grip*, *The Wax Witness*

No scraped source. The machine invents a closed room and a physical object that will not behave.

*The Weight of Velvet*: grand theater, Ayo Kendall, Martin Cross, Lev Palmer, a curtain, a cold draft in the wings. Shakespeare-stage realism, LTX 2.5 mime.

*The Stone That Won't Move*: frost, a river, Truman Halloway, Edna Reeves, Stone Steele, a stone that keeps the whispering at bay. Anime, LTX 2.5 mime, SD 1.5, Qwen.

*Twisted Grain*: live obs, 85 MB, visual storybased, viz camera, Bark, original, Gemma E2B, MusicGen. One of the few Bark house voices that is not a proof.

*The Wax Witness* and *The Silk's Grip*: live obs coded originals from 2026-09-14, Gemma 12B, Kokoro, Stable Audio 3. Haunted AnimateDiff versus viz camera. Same morning, two looks.

### Beta titles from the unreadable era (legacy names, real shows)

These never got shortcodes, so they do not rank in a file browser. They are most of the 70 GB.

A sample of the heavyweight board: *The Rags of Father's Fury*, *The Summer Frequency Sequence*, *The Last Reading*, *Bells Beneath Sardis*, *The Stone Frequency*, *The Ring of Authority*, *The Spore of St Peter*, *The Coil of the Speckled Band*, *The Autopsy of a Memory*, *Scour the Wood Until It Bleeds*, *The Attribution Decay*, *The Humidity of History*, *Roses of Defiance*, *The Apprentice's Number*, *The Weight of the Brass Key*, *The Weeping Seal*, *The Architect of Misery*, *The Orchard Graveyard*, *Ink and Inheritance*, *The Nursery Feed*, *The Caretaker's Clause*, *The Sanctuary of Silence*, *The Silk Shroud*, *The Weight of Weeping Ink*, *The Soot on the Velvet*, *The Stitch of Truth*.

Live obs still holds a late-August / early-September pocket of these: *The Stroke of the Pen*, *The Tomb of Silence*, *The Ghost Protein*, *The Weight of 1924*, *The Weight of Wax*, *The Furnace Heat*, *The Silk of the Lapel*, *The Key to the Threshold*, *The Weaver's Sovereign Truth*, *Screams in the Ledger*, *Gleaming in Ruby*, *Static Whispers*.

The titles are the show. Objects with weight. Ink, wax, silk, brass, velvet, reels, gauges. A radio program about things that refuse to stay still, named like a pulp catalog.

---

## What is actually junk

Call it junk only when it cannot be a program:

1. Earthsearch sitting in live `otr/obs` (1,094 files, 27.4 GB). Wrong show, right folder. It inflates every naive total.
2. *The Last Frequency* 36 times (13.1 GB). Necessary alpha. Not a season.
3. Still-flat coded shorts (32 files). Graph-completion receipts. IndexTTS2 plus Nemo plus SA3. They pad the TTS and writer leaderboards if you count files instead of bytes.
4. Named proofs: regression tests, Mac lightning proofs, Vidu smoke, bark calibration, music benches, organ suite, bank_genres.
5. The old filename itself. It did not delete the episodes. It hid them.

Do not call the 908 legacy broadcasts junk. They are the show, wearing a bag over its head.

---

## What we learned (the actual hero's journey)

1. A watch folder is a product surface. If every file truncates to the same suffix, the operator cannot see success, and the rule "if I see it in obs, it worked" dies.
2. Codes belong in the filename; full names belong in the ledger. Four characters (five for the writer) bought the title back and made style, lane, voice, bank, writer, and music visible.
3. Count lies, bytes tell. IndexTTS2 and Nemo "win" file count because they rendered proofs. Kokoro, Qwen, Gemma 12B, viz camera, and haunted AnimateDiff win the broadcasts.
4. The form is a one-act of roughly 200 words. Chasing seven acts or word count was never what shipped.
5. Six story banks are a real mix, not a single news bot: Shakespeare in the author's words, film-vault RSS that always returns to acetate, public-domain comic detectives, original object-horror, science news in a radio coat, and finally the listener's own My Story.
6. Alpha is allowed to grind one title. Beta is allowed to emit hundreds. Neither is a reason to flush the folder. The operator's own rule: never move, hide, or clean harness runs out of obs. Seeing them is the point.
7. A second show in the watch directory will drown the first. Earthsearch is 93 percent of live obs. That is an operations problem, not an artistic one.

---

## Suggested NotebookLM prompts

Use this file as the only source, or add a handful of ledgers if you want more dialogue.

- "Explain whether Signal Lost's on-disk archive is a vibe-coder hero's journey or junk. Use the file counts and the naming change."
- "Walk through alpha (The Last Frequency), beta (June to August dailies), and the September leaderboard names."
- "What do Kokoro, IndexTTS2, and Bark actually represent if you weigh bytes instead of file counts?"
- "Read *The Count of Three* and explain what My Story is for."
- "Why do the media-archive episodes all take place in vaults and attics?"
- "List the six story banks and give one example title from each."

---

## Provenance

Inventoried 2026-09-15. Live: 1,115 videos, 29.3 GB. Ancient: 1,480 videos, 95.0 GB. Ledgers matched by timestamp: 1,365 of 1,472 dated videos. Unique matched titles: 891. Coded leaderboard files: 79. This briefing does not claim aesthetic quality. Story quality was declared done by the operator on 2026-08-04. The question here is structural: did the pile become a show, and can you see it?
