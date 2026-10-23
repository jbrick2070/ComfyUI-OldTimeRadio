# ONE FORM, ONE RADIO PLAY
### How to use Old-Time Radio through ComfyUI's App Mode form

**Estimated runtime:** ~4:55 (≈145 wpm, dry narration)
**Visual style:** 3D wireframe / isometric vector motion graphics
**Audio:** dry narration, no music (one non-musical cue — see AUDIO SYSTEM)
**Pipeline:** NotebookLM-compatible scene blocks
**Palette:** locked — deep-violet base (#120A2E), paper-white (#F2EFE8), signal-magenta (#F02B8F), dial-cyan (#3EE9FF)
**Data convention:** SOLID magenta = documented fact (what the form and the README say) · DASHED cyan = the viewer's own choice or our expectation
**Scene count:** 12 (+ end card)
**Narration word count:** ~715

**v2 changes from v1:**
- Scenes 05, 07, 08 and 11 quote the help lines as the form says them after the 2026-10-23 QA fold (05e51de4): Source "Blank, or an id + a named bank"; Visual style "Visualizer frames ignore it"; the still rows "Not every video lane uses it"; the voice rows "kokoro: ja/zh may need extras"; the checklists "… roll only; none = all".
- Narration follows: an id needs a named bank; the credits still name the style; some lanes take no still without saying so; Japanese and Mandarin may need two libraries; an empty checklist is all.
- End card: BANKS.md is the guide to the banks; it does not list source ids.

Facts in this script come from three places only: the pack's README, `config/app_mode.json` (the form's labels and help lines), and the form as it renders in ComfyUI. Every row name and help line below is quoted as the form draws it.

---

## AUDIO SYSTEM

Dry narration over silence. One cue: a tuning dial — a dry, dead-air crackle with a single tick each time a scene lands on a form row, never rhythmic. It does not accelerate; this is a walkthrough, not a chase. The cue cuts to silence at the end card.

---

## SCENE 01 — COLD OPEN: ONE PRESS

**VISUAL:** Isometric deep-violet floor, a thin cyan grid receding to a horizon. A single paper-white panel rises from the floor: a form, forty rows tall, drawn as wireframe rectangles. At its foot one magenta button, solid, labelled RUN. Camera pushes slowly toward the button.

**NARRATION:** This is a form. Forty rows and one button. Press the button, and your own machine writes a radio play, casts it with voices, composes its music, films every line, and hands you a finished video. No account. No key. No cloud unless you ask for one.

**ON-SCREEN TEXT:** OLD-TIME RADIO · COMFYUI APP MODE · ONE PRESS

**TRANSITION:** The camera pulls back; the form shrinks to a card and slides onto a shelf labelled TEMPLATES.

---

## SCENE 02 — INSTALL

**VISUAL:** An isometric ComfyUI window in wireframe. Its menu bar lights in magenta, one item at a time: EXTENSIONS, then NODE MANAGER. A search field types itself: "old time radio". A card appears; its INSTALL button pulses solid magenta. Beside the window, a small wireframe terminal shows one line in cyan.

**NARRATION:** Install it from inside ComfyUI. Open Extensions, then Node Manager. Search for old time radio. Install. Put ffmpeg on your path — on Windows, winget install Gyan dot FFmpeg; on a Mac, brew install ffmpeg. Then restart ComfyUI.

**ON-SCREEN TEXT:** EXTENSIONS → NODE MANAGER → "old time radio" → INSTALL · ffmpeg on PATH · RESTART

**TRANSITION:** The window folds flat and re-rises as a template gallery.

---

## SCENE 03 — OPEN THE APP

**VISUAL:** A wireframe gallery of workflow cards on the violet floor. The breadcrumb lights up: WORKFLOW → BROWSE TEMPLATES → EXTENSIONS → OLD-TIME RADIO. One card, labelled otr_app, lifts and turns to face the camera. It unfolds into the form from scene one, standing on the right of the screen, with a dim wireframe canvas behind it.

**NARRATION:** Open Workflow, Browse Templates, Extensions, Old-Time Radio. Pick otr app. It opens as the app: the form on the right, the node graph out of the way. Every other workflow in that gallery is tuned to a size of machine; they open on the graph, and the App button at the top switches them into this same form.

**ON-SCREEN TEXT:** WORKFLOW → BROWSE TEMPLATES → EXTENSIONS → OLD-TIME RADIO → otr_app · APP MODE

**TRANSITION:** The camera glides up the form to its first row.

---

## SCENE 04 — THE SHOW: LANGUAGE, CAST, ACTS

**VISUAL:** The top three rows of the form, each a wireframe rectangle with a bold label, a smaller help line in paper-white, and a control. They light solid magenta one at a time. Row one: LANGUAGE, "Spoken, written and captioned", English. Row two: CAST SIZE, "Speaking parts, plus announcer". Row three: ACTS, "More acts = longer, slower run". A dashed cyan bracket hovers beside ACTS.

**NARRATION:** The first three rows are the show. Language: the play is spoken, written and captioned in the one you pick. Cast size: the speaking parts, plus the announcer. Acts: more acts make a longer play, and a slower run. One act is a good first try.

**ON-SCREEN TEXT:** LANGUAGE · CAST SIZE · ACTS

**TRANSITION:** The three rows dim; the form scrolls up by one card length.

---

## SCENE 05 — STORY BANK AND STYLE

**VISUAL:** Three rows. STORY BANK, "Pick my_story to tell yours", with its dropdown drawn as a wireframe list fanning out: original, scifi_news_pro, media_archive, shakespeare, public_domain, my_story, and a top entry "roll (any eligible bank)" in solid magenta. SOURCE TO ADAPT, "Blank, or an id + a named bank". VISUAL STYLE, "Visualizer frames ignore it", with its own "roll (any style)" entry lit.

**NARRATION:** Story bank decides what kind of show it is. Leave it on roll and the pack picks one: an original drama, science fiction from tonight's science news, a story from the media archives, a scene of Shakespeare, a public-domain classic. Pick my story to tell your own. Source to adapt stays blank unless you have a source id, and an id needs a bank picked by name, not rolled. Visual style restyles the pictures — but note the help line: the visualizer lanes the form ships with ignore it in their frames. The credits still name it.

**ON-SCREEN TEXT:** STORY BANK · roll · my_story · SOURCE TO ADAPT · VISUAL STYLE

**TRANSITION:** The dropdown list folds back into its row; scroll.

---

## SCENE 06 — THE THREE SWITCHES

**VISUAL:** Three toggle rows, drawn as wireframe switches, off. RANDOMIZE VIDEO (16 GB+), "Overrides video pickers below". RANDOMIZE STILLS (16 GB+), "Overrides still pickers below". RANDOMIZE AUDIO (16 GB+), "Overrides voices and composer". Each help line draws a solid magenta arrow pointing down at the rows beneath it.

**NARRATION:** Three switches. Each one, when on, draws one local engine at random and uses it for the whole episode — and it overrides the pickers under it. They are suggested for cards with sixteen gigabytes or more. Leave them off and the pickers below are yours.

**ON-SCREEN TEXT:** RANDOMIZE VIDEO · RANDOMIZE STILLS · RANDOMIZE AUDIO · "Overrides … below"

**TRANSITION:** The arrows extend downward and the camera follows them.

---

## SCENE 07 — THE PICKERS, ROLE BY ROLE

**VISUAL:** Six rows in two columns of the same wireframe: ANNOUNCER - VIDEO, "Films the announcer's lines"; ANNOUNCER - STILL IMAGE, "Not every video lane uses it"; CHARACTERS - VIDEO, "Films the talking shots"; CHARACTERS - STILL IMAGE; MUSIC - VIDEO, "Films the music beats"; MUSIC - STILL IMAGE. The video rows carry their shipped value in cyan: "viz_… (audio-reactive, no scene image)". A dashed cyan line links each still row to the words "no scene image" in the video row above it.

**NARRATION:** The pickers come in roles: the announcer, the characters, the music. Each role has a video lane and a still image model. The lanes the form ships with are visualizers — they draw from the sound and take no still. So the still rows are unused until you pick a video lane that starts from a picture — and a few lanes whose names do not say so take no still either. The help line says it plainly: not every video lane uses it.

**ON-SCREEN TEXT:** ANNOUNCER · CHARACTERS · MUSIC · VIDEO / STILL IMAGE · "Not every video lane uses it"

**TRANSITION:** The six rows compress into a stack; scroll.

---

## SCENE 08 — VOICES, MUSIC, WRITERS

**VISUAL:** Five rows. ANNOUNCER - VOICE and CHARACTERS - VOICES, both "kokoro: ja/zh may need extras", value kokoro. MUSIC - COMPOSER, "sonilo = paid; google = key". WRITER - STORY, "Writes outline and dialogue". WRITER - STRUCTURE, "Checks and shapes the script". The two writer rows share one solid magenta bracket.

**NARRATION:** Voices: kokoro ships with the pack and speaks every language the form offers; for Japanese and Mandarin it may need two extra libraries, and the help line says so. Music composer: the shipped engine is local; sonilo is paid, and the google engine needs your key. Two writers: one writes the outline and the dialogue, the other checks and shapes the script. Both run on your machine by default.

**ON-SCREEN TEXT:** VOICES · kokoro · MUSIC - COMPOSER · WRITER - STORY · WRITER - STRUCTURE

**TRANSITION:** A dashed cyan curtain drops over the next block.

---

## SCENE 09 — THE CLOUD SLOTS (OPTIONAL)

**VISUAL:** Six rows under the dashed curtain, dimmer: CLOUD WRITER - OPENROUTER A and B, CLOUD WRITER - COMFY CREDITS A and B, CLOUD WRITER - GOOGLE A and B, each with the help line "Also set a Writer to this slot" and a value in parentheses such as "(enable OpenRouter)". A dashed cyan arrow runs from these rows back up to the WRITER rows of the previous scene.

**NARRATION:** Six cloud writer slots, all off. They are optional, they are paid or keyed, and a slot does nothing on its own: you also set one of the Writer rows to that slot. If you never touch these, nothing leaves your machine.

**ON-SCREEN TEXT:** CLOUD WRITER SLOTS · OFF BY DEFAULT · "Also set a Writer to this slot"

**TRANSITION:** The curtain lifts; two single rows slide in.

---

## SCENE 10 — UPSCALER AND LEMMY

**VISUAL:** UPSCALER, "Sharpens video; slows the run", value off. LEMMY CAMEO, "A Cockney officer drops in", value "roll (~11% chance)". A small wireframe figure in a peaked cap tips it at the camera.

**NARRATION:** Upscaler: off by default; on, it sharpens the video and slows the run. And Lemmy — a genial Cockney communications officer who sometimes drops into the show. By default he rolls the dice himself.

**ON-SCREEN TEXT:** UPSCALER · LEMMY CAMEO

**TRANSITION:** The figure steps aside; a taller block of text boxes rises.

---

## SCENE 11 — MY STORY

**VISUAL:** The My Story block: MY STORY - WHO IS IN IT, "Set Story bank to my_story"; CHARACTER PHOTOS (NAME = FILE), "Needs still: flux3_klein_ref"; MY STORY - WHAT HAPPENS, "Events, conflict, ending"; MY STORY - WHERE AND WHEN, "Place, era, mood"; MY STORY - BY, "Your name, or a pen name"; MY STORY - MUSIC IN WORDS, "e.g. surf rock, solo cello". A solid magenta line runs from the first row's help text up the form to STORY BANK. Below, SPACE SAVER, "Finished video is always kept", and the five roll checklists, drawn as collapsed wireframe chevrons, each with its help line "… roll only; none = all".

**NARRATION:** My Story is where your own play goes: who is in it, what happens, where and when, who it is by, and your music, in words. The first help line is the one to read: set Story bank to my story, or these boxes are not read. Character photos paint a character from a picture, and need the flux3 klein ref still model. Space saver tidies the episode's folder; the finished video is always kept. The checklists at the foot narrow a roll; each is read only by its own roll, and an empty checklist means all of them.

**ON-SCREEN TEXT:** MY STORY · "Set Story bank to my_story" · SPACE SAVER · CHECKLISTS

**TRANSITION:** The camera drops to the foot of the form and the RUN button fills the frame.

---

## SCENE 12 — RUN, AND WHERE IT LANDS

**VISUAL:** RUN pulses solid magenta and depresses. The form dims. Along the bottom of the screen a wireframe tray slides out: the outputs pane. Beside it, an isometric folder tree draws itself in cyan: output → otr → obs, and inside, one file icon, solid magenta, with a play triangle. A dashed cyan clock beside the tray.

**NARRATION:** Press Run. The first run downloads the models it needs, several gigabytes, so it takes longer. Your outputs appear at the bottom. The finished episode is an mp4 in your ComfyUI output folder, under otr, obs. Press it again and the next play is different.

**ON-SCREEN TEXT:** RUN · first run downloads models · output/otr/obs/*.mp4

**TRANSITION:** The folder icon brightens, then everything but it cuts to black.

---

## END CARD / SOURCES

**VISUAL:** Black. Citation list scrolls in dial-cyan monospace.
**NARRATION:** (silent — 3 seconds of black before end card)
**ON-SCREEN TEXT:**
SOURCES
• README.md, "Install and run it" — github.com/jbrick2070/ComfyUI-OldTimeRadio
• config/app_mode.json — the form's rows, labels and help lines (2026-10-23)
• apple/BANKS.md — the story banks
• docs.comfy.org/interface/app-mode — App Mode
• Registry: comfyui-old-time-radio (publisher fluxus)

---

**Runtime estimate:** ~4:55 at ≈145 wpm
**Scene count:** 12 (+ end card)
**Narration word count:** ~715

**Production notes:**
- Scene 7 is the one that saves a stranger from a silent failure (stills unused under a visualizer lane); hold the dashed link between "no scene image" and the still rows a beat longer than feels natural (and the narration's aside that some lanes take no still without saying so).
- Scene 11's magenta line from "Set Story bank to my_story" up to the STORY BANK row is the second such save; let it draw slowly.
- If runtime is tight, scenes 9 and 10 collapse into one (cloud slots, upscaler and Lemmy as three quick rows) and lose about twenty seconds.
- Every quoted help line is the form's own text at `config/app_mode.json`; if a line changes there, re-quote it here before recording.
- The dial cue should be barely there; the narration carries the walkthrough.
