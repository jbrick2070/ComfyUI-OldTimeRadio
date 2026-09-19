# Signal Lost: the back alley of the dropdown

A source briefing for a NotebookLM audio overview. Measured 2026-09-18 from
the twenty-one shipping graphs in `SHIPPING_SET` against the live dropdown
roster in `docs/DROPDOWN_MATRIX.md`. This is not a build log and not a
manual. It is what the show *sounds and looks like* when someone leaves the
front door and walks the rest of the menu.

Upload this file into NotebookLM. Pair it with
`NOTEBOOKLM_SIGNAL_LOST_ERA_BRIEFING.md` and
`NOTEBOOKLM_COMFY_CLOUD_COMBOS_BRIEFING.md` if you want the junk-folder era
and the Cloud costume in the same episode. Ask for an Audio Overview.
Keep the hosts off process. Stay on throats, rooms, turntables, and silent
beats.

---

## The one-sentence verdict

Most of the dropdown is a back alley. The saved graphs pin a small front
room. Everything else is listed on purpose -- the menu never hides an
engine -- and listing it is not a recommendation.

---

## The front room (so the alley has a wall)

Twenty-one graphs ship. They do not all wear the same coat, but they agree
on a few things.

**Voices, at home.** Every local shipping graph that names a voice engine
names **Kokoro**. Twenty-eight English presets. The only throat with a
published episode on NVIDIA, on a Mac, and on a box with no GPU at all.
Character slots that the profile does not bother to name still inherit that
same house voice from the canonical graph. If you open a saved JSON and do
not touch Cast Lock, the people sound like Kokoro.

**Voices, in the rented shop.** All five Comfy Cloud graphs pin
**ElevenLabs** on both the cast and the announcer. That is a costume, not
the station.

**Music.** Fifteen local GPU graphs pin **Stable Audio 3** -- a theme
composed for that episode, not a library cue. The CPU graph is the one
exception on the local side: **MusicGen**, because Stable Audio 3 will not
run with no GPU, and MusicGen's weights are non-commercial. All five Cloud
graphs pin **Sonilo**. Sonilo is the Cloud music default. It is the
exception the operator named. It is not alley.

**Pictures the house already chose.**

| What you opened | What the picture is |
|---|---|
| low / first run | scopes and cameras: `viz_camera`, `viz_green`, `viz_mxc_cpu`, sometimes the mandala |
| still | a card that breathes: `still_motion` |
| 8 GB video | the small LTX walk |
| Mac video | LTX 0.9.8 |
| 16 GB video | LTX 2.5 picture, no extra bed |
| 16 GB foley | LTX 2.5 picture **plus** the model's own room tone, mixed under the voices |
| 16 GB mime | the same picture, but those beats throw the voices and music away |
| haunted AnimateDiff | SD 1.5 ghosts on 8 GB and 16 GB; lightning AnimateDiff on the Mac |
| cheap Cloud | Vidu walk, Luma still |
| deluxe Cloud Foley | hosted LTX 2.5 with a rented bed |
| deluxe Cloud audio-in | hosted LTX 2.5 whose mouth is driven by the mix |

The 16 GB foley and mime graphs **are** saved defaults. They are still
alley chapters, because they are mix decisions wearing a picture dropdown.
The rest of this file is everything the roster offers that **no shipping
graph pins**.

---

## Chapter: IndexTTS2, the clone that will not announce

`indextts2` sits in the voice menu. Zero shipping graphs pin it. Lab
profiles used to. The saved ones do not.

It copies a voice from a recording you supply. It can do the cast. It
cannot read the announcer. It wants its own Windows installer, about
11 GiB, and a 16 GB NVIDIA card. There is no Mac path and no Linux
installer. The pack ships no reference wavs. Without a recording it
refuses by name.

The hang people remember is a hang, not a prompt problem. Kokoro is why
every published local episode has a throat. IndexTTS is the alley clone
with feeling controls -- if you have already built the sidecar and you
know you are not asking it to be the narrator.

---

## Chapter: Dia and Chatterbox, the other two clones

`dia` and `chatterbox` are the same kind of animal as IndexTTS and they
are also never a shipping default.

Both copy a voice from a wav you own. Both need a Windows installer of
their own. Both want 16 GB NVIDIA. Chatterbox is the smaller download
(about 3 GiB). Dia is about 6. Neither is offered on 8 GB, on a Mac, or
on CPU.

If you pick the `default` bank, the caster will try to hand these engines
a reference. If you pick `default_clean`, IndexTTS is left out on
purpose -- its model licence is non-commercial -- and the two
permissive cloners remain. The announcer can use Chatterbox or Dia. The
announcer still cannot use IndexTTS.

A lab rotation once put Chatterbox on the cast and Kokoro on the
announcer, and another put Dia on the cast. Those are not graphs a
stranger is meant to open. They are proof that the alley was walked.

---

## Chapter: Bark, the theatrical one that can take a Mac down

`bark` is in the menu. No shipping graph pins it. It has ten speaker
presets and a more theatrical, less predictable read. The weights fetch
themselves.

On a Mac mini M4 with 16 GB it was measured at about twelve times slower
than realtime, and the process grew an 18 GB footprint on a 16 GB
machine. On Apple Silicon, running out of memory reboots the box. There
is no failed render to read. The machine just starts over. Leave Bark
alone on a Mac. On CPU it is a billion-parameter crawl. On NVIDIA it is
the alley throat if you want something wilder than Kokoro and you do not
want to clone.

---

## Chapter: Foley and mime, the two mix decisions

`ltx25_high_foley_plus` and `ltx25_high_mime` (the profiles also say
`ltx25_foley_plus` and `ltx25_mime`) draw the same LTX 2.5 picture as
the 16 GB video graph. They keep the audio the model invented next to
that picture.

**Foley** pours that invented room into the episode master at an even
split. You still hear Kokoro. You also hear the model's chairs, cloth,
and air. The Shivering Gauge episode that opens the README ran a foley
lane: the lab's own sound, not a library.

**Mime** replaces the episode audio on those beats. The voices and the
theme are still generated, then thrown away. Every beat of that role
becomes a silent performance. That is a radio play deciding not to be
radio for a while.

Two Cloud cousins exist as saved deluxe graphs: hosted Foley and hosted
audio-in. Those are costumes. The local foley and mime rows are the
alley of the 16 GB card -- gated weights, a licence click, about 22 GiB
placed by hand, Q3 GGUF because the official safetensors do not fit
16 GB.

---

## Chapter: Mesh stage, a person who becomes a sculpture

`mesh_stage` is never a shipping default. It takes a character portrait,
builds a 3D mesh, and spins a turntable of it in a headless Blender.
Manual weights, a portable Blender, a community licence. It looks like
someone walked out of the radio play and into a museum plinth. The rest
of the episode is still a broadcast. This one role is a statue that
turns.

---

## Chapter: HuMo, Wan, H3 -- the big local walks

None of these are a shipping default. All of them are in the video menu.

**HuMo** (1.7B and 1.4B, portrait and wide) is audio-in: the beat's own
sound moves the mouth. Large. 16 GB territory. Proven there. Out of
memory on 8 GB.

**Wan 2.2** (`wan22_high_video`, `wan22_high_fast`) is local diffusion
with a still pinned as the first frame. The high lane is proven on
16 GB. The fast lane fits and has no published episode. Both refuse an
8 GB card.

**MiniMax H3** is the largest download on the board -- low-40s of
gigabytes -- and the slowest local walk. `h3_low_video` is picture only.
`h3_low_audio_in` takes a portrait plus the beat audio. Neither emits
sound of its own. Proven on 16 GB. Not a house default, because a first
episode should not fetch a third of a disk.

**LTX 2.3** is the honest trap. `ltx23_high_video` runs out of memory on
8 GB and on 16 GB. It stays in the list because the list is the
registry, not a recommendation. `ltx23_low_audio_in` can fit 16 GB and
is still nobody's saved graph.

---

## Chapter: the stills that are not the breathing card

The house still is `still_motion` -- a card that moves a little.
`still_flat`, `still_pan`, and `still_word` are in the menu and pinned
on zero shipping graphs.

Flat is a picture that does not pretend to walk. Pan is a slide across a
card. Word is the script's own line burned onto the frame -- that is
where a typography engine earns its keep. The house never starts there,
because a first listener should see a face breathe, not a slogan.

---

## Chapter: Flux the elder, Ideogram the sign-painter

`flux_gen1` is the first picture engine this pack ever had. Thirteen
gigabytes, non-commercial licence, no shipping pin. `ideogram4_local`
is a typography specialist for the word card, about 95 seconds a still
against `z_image_turbo`'s 12, also non-commercial, also never saved.
`z_image_turbo` itself is the AMD still default. Lumina is the 16 GB
still. SD 1.5 is the 8 GB and Mac still. Those three are front room.
Flux and local Ideogram are memory.

---

## Chapter: Google, the other rented shop

No shipping graph pins a Google engine. The dropdown still offers
`google_veo_video`, `google_omni_video`, `google_image`, `google_tts`,
and `google_lyria`.

They take your own Google key, not Comfy Credits. Veo and Omni walk.
Image stills. TTS is thirty Gemini throats. Lyria is preview music.
Without a key the new queue-time slug check refuses before a credit
moves -- the same courtesy the Comfy lanes already had. A dead model
name should fail in the doorway, not after the writer has already
spent.

Google is not the Cloud costume chapter. Cloud is Vidu, Luma,
ElevenLabs, Sonilo, Flux Pro, hosted LTX. Google is the alley next to
that shop: same radio play, different landlord.

---

## Chapter: Cloud leftovers

Cheap Cloud pins Vidu and Luma. Deluxe pins hosted LTX and Flux Pro.
The menu still holds Kling Avatar, Seedance 2, Wan image-to-video, Wan
audio-in, Nano Banana, Seedream, Krea, and Ideogram-on-credits (`ideo`).

Those are partner costumes nobody saved as a house graph. Pick is the
enable. Credits still spend. A dead slug should now refuse at the
validator the way a dead Comfy writer slug already did.

---

## Chapter: the gated bed, and the upscaler nobody turns on

`stable_audio_music` needs a Hugging Face licence click and a token. No
shipping graph pins it. Stable Audio 3 is the house bed on every local
GPU graph.

`spandrel_esrgan` is the local 2x upscaler. Every shipping composite
leaves upscale **off** and lets ffmpeg Lanczos take the finished frame
to 1080. The upscaler is in the menu for people who want a second
picture pass. It is not how Standard OTR gets to 1080.

---

## Chapter: writers who never got the chair

The house writers are Qwen 3.5 4B on the small cards, Gemma 4 12B on
16 GB, and Comfy Credits slots (Sonnet and Luna) on Cloud and on the
CPU graph. The menu still lists Gemma 2 2B, Gemma 4 E2B, Llama 3.2 3B,
Gemma 4 E4B, and Mistral Nemo. None of those five is a shipping
default. They are leftover chairs in a room that already picked a
playwright.

---

## AnimateDiff's other ghost

Haunted AnimateDiff is a saved graph on 8 GB and 16 GB. Lightning
AnimateDiff is the Mac one. `animatediff15_v3_stillin_lab_video` is a
third cousin -- still-in, lab look -- and no shipping graph pins it.
Same family, never the front door.

---

## How to hear this

The front room is a radio play you can queue without reading a menu:
Kokoro, a composed theme, a picture lane the graph already chose.
The back alley is where someone clones a throat, hangs a mesh in
Blender, lets the video invent the room, or throws the voices away for
a mime beat.

Fans already know the watch folder. Tell them the alley is optional
magic that can empty a disk or reboot a Mac. Curious Comfy people know
partner demos. Tell them the same play can put on HuMo's mouth, Bark's
theater, or a turning statue, and still has to land one file in
`otr/obs` or it did not happen.

Do not tell either audience about sprints. Tell them which door was
the house, and which doors were left unlocked on purpose.

---

## Suggested NotebookLM prompts

- "Walk a fan down the Signal Lost back alley. Start at Kokoro. End at mime."
- "Why is IndexTTS2 in the menu if no saved graph uses it?"
- "Explain Foley versus mime as two ways of treating the model's own sound."
- "What happens if someone picks Bark on a 16 GB Mac?"
- "Describe mesh_stage as if a character stepped out of the radio and onto a plinth."
- "Which Cloud dropdowns are costumes the house already wears, and which are leftovers?"
- "Two hosts: one who only opens saved graphs, one who lives in the dropdown. They argue what 'the show' is."
- "Is Sonilo alley or house? Use only the Cloud graphs."

---

## Provenance

Written 2026-09-18 from `scripts/build_variants.py` `SHIPPING_SET` (21
graphs), each profile's `role_overrides` and `slot_overrides`, and the
dropdown roster in `docs/DROPDOWN_MATRIX.md`. A row is "alley" when no
shipping graph pins it, except Foley and mime, which are saved specialty
graphs and still mix decisions. Sonilo is the Cloud music default --
named exception, not alley. Kokoro is the local voice default. Story
quality is closed. The question is which doors the house left unlocked.
