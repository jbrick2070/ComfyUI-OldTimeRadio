# Music -- the theme and the cues between scenes

Every episode opens on a theme and closes on one. That music is composed for
that episode by a text-to-music model, not pulled from a library, and the show
decides what to ask for: the **genre comes from the source bank**, the feeling
comes from the story's own mood words, and a short phrase of the scene the cue
sits beside rides along with it.

Two widgets control it, both on **OTR_StableAudioTheme**: `engine` and
`music_style`. Neither needs touching for a first episode.

---

## The cues

| Cue | Length | Where it sits |
|---|---|---|
| **opening** | 12 seconds | Before the drama, over a held black frame |
| **closing** | 8 seconds | After it, same |
| **interstitial** | 4 seconds | Between acts |

You get an opening and a closing on every episode. **Interstitials only appear
on the two banks whose writer plans them** -- **My Story** and **Sci-Fi News
Pro** -- and there is one per act break, so `act_count` 1 gives none and
`act_count` 3 gives two. On the other four banks there are exactly two cues.

---

## Which engine

| Engine | What it needs | Where it runs |
|---|---|---|
| **`stable_audio_3`** *(the default)* | Nothing. Two files, about 3.2 GiB, fetched on the first run that needs them | NVIDIA and Apple Silicon. **Not CPU.** |
| **`musicgen`** | Nothing. 2.2 GiB fetched on first use | NVIDIA, Apple Silicon and CPU |
| **`stable_audio_music`** | **A Hugging Face licence acceptance and a token** | NVIDIA only |
| **`sonilo`** | A Comfy account and credits | Hosted -- no GPU of yours |
| **`google_lyria`** | A Google API key. Paid, and preview-only | Hosted -- no GPU of yours |

**Stable Audio 3 is what twenty-one of the twenty-two shipped graphs use**, and it
is the right answer on any machine with a GPU: it downloads itself, and its
licence permits commercial use.

**The CPU preset is the seventeenth**, and it is on MusicGen -- Stable Audio 3
declares CUDA and Metal only, so a machine with no GPU has to use something
else. Worth knowing if you go that way: **MusicGen's weights are
non-commercial** (CC-BY-NC). Fine for listening; not for anything you sell.

The last three are opt-in and none of them is automatic. Two of them cost money
per cue. Picking one that is not set up does not quietly fall back to another --
the run stops and names the engine and the reason.

---

## What `music_style` does

It is your own music, in your own words: `gamelan orchestra`, `surf rock`,
`solo cello`. It is used exactly as typed.

**It is honoured on the My Story bank and nowhere else.** Every other bank has a
fixed musical identity and keeps it whatever you type -- that is the point of
having one. A Shakespeare episode scored as surf rock is not a feature, and the
sci-fi news lane being Detroit techno every week is what makes it recognisable.

Leave it blank and My Story gets the house sound, a 1940s radio drama orchestra.

### What each bank plays

| Bank | The music |
|---|---|
| `scifi_news_pro` | Detroit techno at 128 BPM, hypnotic machine funk |
| `media_archive` | Small-group jazz quartet, relaxed swing |
| `public_domain` | Chicago house at 122 BPM, soulful and steady |
| `original` | Salsa conjunto at 100 BPM, clave-driven |
| `shakespeare` | Elizabethan consort music -- viols, recorders, lute |
| `my_story` | **Yours**, from `music_style`. Blank means the house orchestra. |

Episode language does not change this table. If a multilingual test pins
`source_bank=original` for every language, every episode correctly asks for
salsa conjunto. To test varied stories and scores, vary the source bank among
the rows that language admits. [MULTILINGUAL.md](MULTILINGUAL.md) lists the
non-English set and explains how the two source-faithful adaptation banks
perform in another language (a vendored translator's text where the corpus
holds the scene, the writer's own translation otherwise).

Shakespeare is the one chosen by the source's date rather than declared
outright, so a very old or very new source shifts the ensemble -- baroque
chamber, Romantic chamber, the radio orchestra, a 1960s instrumental combo.
In practice the Folger texts land in the first band, which is why it is
Elizabethan.

---

## You do not write the cue prompts, and that is deliberate

There is no box to type a cue description into. The pack builds each one from
what it already knows about the episode. A public-domain episode whose brief
reads *ominous* and whose scene is a harbour asks the model for:

> ominous Chicago house at 122 BPM, soulful and steady, in a fog bound harbour,
> a straight-in open, instrumental, no vocals

Mood word, genre, setting, which end of the story this is. That is the whole
request.

**Notice what is not in it: instruments.** The pack knows the Detroit techno
palette runs on a TR-909, and it deliberately does not say so. Naming
instruments to a model that already knows the genre spends the request
repeating itself, and risks it pushing one instrument forward as a solo. The
longer instrument-list version of this prompt existed and was removed in
September 2026.

The same reasoning is worth borrowing when you type into `music_style`. A genre
(`surf rock`, `gamelan orchestra`) lands better than a shopping list, and a
plucked or struck instrument asked for first -- a lute, a harpsichord, any drum
-- in a window this short is what the model turns into a two-bar loop rather
than a theme.

The cue also carries a short list of what **not** to play: hiss, static,
crackle, distortion, anyone singing. On the banks whose genre has a beat the
anti-rhythm half of that list is dropped, because asking for a drum machine and
banning drums in one breath is a request that tears.

---

## If the music does not suit the episode

In this order.

1. **Check which bank the episode used.** The genre follows the bank, and the
   shipped graphs roll the bank. If your archive episode came out as jazz, that
   is the archive bank working correctly -- pin `source_bank` to the one you
   want. See [BANKS.md](BANKS.md).
2. **If you want to choose the music yourself, that is the My Story bank.** Set
   `source_bank` to `my_story` and type into `music_style`. It is the
   bring-your-own lane, so it is the one that takes a bring-your-own score.
3. **Two runs of the same bank sound different, and should.** The mood words
   come from the story, so a melancholy episode gets a melancholy cue on the
   same genre. If you are comparing anything, pin the bank first or you are
   hearing two different shows.

## If something goes wrong

**The run stops naming the music engine.** You picked one that is not set up on
this machine. `stable_audio_music` needs a licence accepted on Hugging Face and
a token; `sonilo` needs a Comfy login; `google_lyria` needs a Google key. It
will not silently substitute another engine. Go back to `stable_audio_3`.

**The run stops on the music step on a machine with no GPU.** Stable Audio 3 is
CUDA and Metal only. Use `musicgen`, which is what the CPU preset ships with.

**The cues sound like a tape loop, or carry a short burst of noise.** Look in
your ComfyUI `models/checkpoints` folder. Stable Audio 3 comes in two files, and
this pack wants **`stable_audio_3_small_music_base.safetensors`** -- the one
ending in `_base`. The other file ignores the "do not play hiss, do not loop"
instruction entirely. A fresh install downloads the right one; a machine that
acquired the other one by hand, earlier, will keep using it.

**The music is there but you cannot hear it.** The opening and closing play over
black frames before and after the drama, so they are at the very start and the
very end of the file, not under the dialogue.
