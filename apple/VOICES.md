# Voices -- who speaks, and what speaks them

Two separate things decide how your cast sounds. The **engine** is the software
that turns written lines into speech. The **bank** is the pool of voices that
engine casts from. Both are set on the node titled **3 - Cast Lock**, and the
engine is named a second time on **4a - Character Voices** and **4b - Announcer
Voice**.

You do not have to choose any of it. Every graph that ships -- the canonical and
all twenty-one saved variants -- arrives set to **Kokoro** on every voice slot, with
the **`kokoro_builtin`** bank behind it. That is the combination every published
episode used.

---

Multilingual episodes use those same controls. Kokoro now carries eight
language rows; [MULTILINGUAL.md](MULTILINGUAL.md) names the voices, source-bank
limits, captions and Python-version boundary.

## Bank versus engine

A **bank** is a set of voices: Kokoro's fifty-four built-in voices across eight
languages,
Bark's ten speaker presets, a shelf of reference recordings for the cloning
engines, or a hosted provider's catalogue.

An **engine** is what performs them.

The two are not independent. Each engine declares which banks it can read, so
picking a bank narrows the engines and picking an engine narrows the banks. Get
the pair wrong and the render stops at Cast Lock with a message naming both --
it does not quietly substitute something that works.

---

## The engines

`char_voice_engine` (the cast) and `announcer_voice_engine` (the narrator) are
separate dropdowns and can hold different values.

| Engine | What it is | How you get it | Size | Where it runs |
|---|---|---|---|---|
| **`kokoro`** | Fifty-four preset voices: 28 English and 26 across Spanish, Portuguese, Italian, French, Hindi, Japanese and Mandarin. The shipped default. | **Automatic** | 0.3 GiB | NVIDIA, Apple Silicon, or CPU-only machines; non-English needs Python 3.10-3.12 |
| **`bark`** | Ten preset speaker voices, more theatrical and less predictable | **Automatic** | 4.2 GiB | NVIDIA. **Read the Mac warning below.** |
| **`chatterbox`** | Clones a voice from a reference recording you supply | Its own Windows installer | 3.0 GiB | 16 GB+ NVIDIA, Windows |
| **`dia`** | Clones a voice from a reference recording you supply | Its own Windows installer | 6.0 GiB | 16 GB+ NVIDIA, Windows |
| **`indextts2`** | Clones a voice, with emotion control. Characters only -- it cannot read the announcer. | Its own Windows installer | 11.1 GiB | 16 GB+ NVIDIA, Windows |
| **`elevenlabs`** | Hosted. Twenty-one library voices. | Comfy account and credits | -- | anywhere |
| **`google_tts`** | Hosted. Thirty prebuilt Gemini voices. | Your own Google API key | -- | anywhere |

**Automatic** means the weights arrive on their own the first time they are
needed and you do nothing. Kokoro is fetched earlier still -- at ComfyUI
*startup*, before you ever press Queue -- which is why the default voice is
always ready.

**Kokoro is the default for a reason**, not because it was first. It is the only
voice engine with a published episode behind it on NVIDIA, on Apple Silicon and
on a machine with no GPU at all, and the only one that is a single click
everywhere. On a CPU-only machine it is still much faster than realtime, though
the exact number depends on which Python build ComfyUI is running: about 6x
on ComfyUI Desktop and the portable build (Python 3.13, Kokoro's `kokoro-onnx`
backend) and about 8x on a from-source install (Python 3.12, Kokoro's `torch`
backend). Either way, voices are not what makes those runs long.

The Python 3.13 ONNX backend is the English path. Non-English rows use the torch
Kokoro pipeline and therefore require Python 3.10 through 3.12. Japanese and
Mandarin also check their `misaki[ja]` / `misaki[zh]` readiness extras when
selected. See [MULTILINGUAL.md](MULTILINGUAL.md).

### The three cloning engines need two things, not one

`chatterbox`, `dia` and `indextts2` do not have voices of their own. They copy a
voice out of a recording, so each needs:

1. **An installer run by hand.** Each ships a PowerShell script in `scripts/`
   that builds an isolated Python environment and fetches its weights. There is
   no shell version, so on macOS and Linux there is no install path today. That
   is packaging, not hardware.
2. **Reference recordings you supply.** One `.wav` per cast voice, under
   `<your ComfyUI models folder>/TTS/refs/<engine>/`. **The pack ships none of
   them.** Without a reference the engine refuses by name and there is no
   fallback.

If either is missing, the error says so and names the script to run.

### Bark on a Mac will not just fail -- it can take the machine down

Bark is offered on Apple Silicon and it does produce good audio there. It was
measured end to end on a Mac mini M4 with 16 GB on 2026-09-09 and the numbers
are the problem:

- **11.7 times slower than realtime** -- 134.6 seconds of work for 11.5 seconds
  of speech. An episode's worth of dialogue is hours in the voice stage alone.
- It drove the process to an **18.0 GB memory footprint on a 16 GB machine**.
  The weights are not the cause; the allocator's reserved pool grew to 16.63 GB
  during generation and never gave it back.

**On a Mac, running out of memory reboots the machine.** Unified memory has no
separate pool to exhaust, so there is no failed render to read afterwards --
the box simply restarts. Leave Bark alone on Apple Silicon. Kokoro is why all
the Mac episodes are Mac episodes.

On a CPU-only machine Bark is offered in principle and is not worth using --
it is a billion-parameter model generating one stage at a time.

---

## The banks

`voice_bank` picks the pool. Each bank works with the engines listed beside it,
and only those.

| Bank | The voices | Character engines | Announcer engines |
|---|---|---|---|
| **`kokoro_builtin`** *(shipped)* | Kokoro's 54 voices, filtered by episode language before casting | `kokoro` | `kokoro` |
| **`bark_legacy`** | Bark's 10 speaker presets | `bark` | `bark` |
| **`default`** | The reference recordings for the cloning engines | `indextts2`, `chatterbox`, `dia` | `chatterbox`, `dia` |
| **`default_clean`** | The same recordings, minus IndexTTS2 | `chatterbox`, `dia` | `dia` |
| **`elevenlabs_cloud`** | 21 ElevenLabs library voices | `elevenlabs` | `elevenlabs` |
| **`google_tts`** | 30 prebuilt Gemini voices | `google_tts` | `google_tts` |

`default_clean` exists for one reason: IndexTTS2's model licence is
non-commercial, so that bank routes the cast to the two permissively licensed
cloners instead. If you are only making episodes for yourself, `default` is the
one with the better voices.

### What the caster actually does with a bank

Characters are cast one at a time, language-filtered first and gender-matched
second. **But gender is not a guarantee.** When a gender's column in the
eligible language pool runs out of untaken voices, the caster does not stop the
render -- it falls back to another voice from that same language, any gender,
and keeps going. It never crosses into English to fill a thin French, Italian
or other non-English pool. `google_tts` is the one engine that refuses instead
of using the gender-blind fallback.

`allow_voice_reuse` (on by default) controls something narrower: whether two
characters can share an already-used, gender-matching voice before the
gender-blind fallback above is reached. Turning it off removes that sharing
step, but it does **not** make the render stop when a gender's voices are
genuinely gone -- the render still reaches the same gender-blind fallback,
just without the reuse step first.

So the language pool's size is the real backstop, not a code guarantee. This matters most on
`bark_legacy`, which has ten presets -- six male, four female -- so a cast
with more than four women will draw at least one male-column voice for a
female character. Kokoro's twenty-eight English voices (thirteen male, fifteen
female) do not run out in practice. French has one admitted voice and Italian
has two, so reuse inside those languages is expected.

The **announcer is one voice for the whole episode**, drawn by the episode's
own seed. English draws from a curated four-voice British pool: `bm_george`,
`bm_fable`, `bf_emma`, `bf_lily`. Two are male and two female, so the
narrator's gender lands roughly evenly across English episodes and never
changes mid-show. Non-English episodes draw an announcer only from their
selected language row.

---

## Changing the engine

Supported, and genuinely the point of the dropdowns being there. Two things to
know before you do it.

**First: check the engine against your machine.** Section 2 of
[MACHINES.md](MACHINES.md) has the full grid -- what runs where, what it costs
to download, and whether it needs an installer.

**Second: the engine is named in three places and all three must agree.**

| Node | Widget |
|---|---|
| **3 - Cast Lock** | `char_voice_engine` |
| **4a - Character Voices** | `engine` |
| **3 - Cast Lock** | `announcer_voice_engine` |
| **4b - Announcer Voice** | `engine` |

Change one and not the other and the render stops with *"the two
character-engine controls disagree"* (or the announcer twin of it), naming both
values. This guard exists because without it the ledger and the credits said one
engine while a different one was actually speaking.

For a non-English episode, all four values in the table must be `kokoro`.
Cast Lock rejects every other day-one combination before speech begins.

Note the asymmetry in `auto`:

- On the **character** side, `auto` means "follow the bank" and is never treated
  as a disagreement.
- On the **announcer** side, `auto` resolves to **Kokoro**, immediately. So if
  you set the Announcer Voice node to anything else, you must set
  `announcer_voice_engine` to match it -- leaving Cast Lock on `auto` is a
  disagreement, not a wildcard.

---

## `voice_device`

Which processor speaks the lines. The shipped canonical uses **`default`**,
which asks ComfyUI what this machine has and then records what it chose. The
saved variants pin it: `cuda` on the NVIDIA graphs, `mps` on the Mac graphs,
`cpu` on the CPU graph.

An explicit device is never second-guessed. If you name one the machine does
not have, the render says so rather than quietly dropping to something ten times
slower and letting you wonder why the run took all night.

---

## When something goes wrong

**"The two character-engine controls disagree."** You changed the engine in one
place. Set the Cast Lock widget and the voice node's own `engine` widget to the
same value -- see the table above.

**"voice_bank ... is not allowed for char_voice_engine ..."** The bank and the
engine are not a legal pair. The bank table above says which go together.

**"... has no reference entries in the active voice bank."** You picked a
cloning engine against a preset bank. `indextts2`, `chatterbox` and `dia` read
the `default` or `default_clean` banks, never `kokoro_builtin` or `bark_legacy`.

**"Path B not installed."** A cloning engine whose installer has not been run.
The message names the script. On macOS or Linux there is no installer to run --
pick a different engine.

**A missing `.wav`, or a voice that cannot be resolved.** A cloning engine with
no reference recordings on disk. See "The three cloning engines need two
things" above.

**A named Kokoro voice file is missing.** The startup fetch did not run or could
not reach the network. The error prints the exact `huggingface-cli download`
command, and it is deliberately never run during a render -- a mid-render fetch
once threw away a finished episode.

**Source bank does not pick the TTS engine.** My Story, Sci-Fi News, Shakespeare,
and the rest can all use Kokoro, Bark, Google TTS, or any other voice engine
the Cast Lock + 4a/4b widgets name. Set those five widgets to the SAME engine
or the agreement guard stops the render. `voice_bank` on Cast Lock is the
voice *pool* (kokoro_builtin / bark_legacy / google_tts), not the writer's
`source_bank`.

**The voices are fine but the run took hours.** Check which engine and which
device you are on. Bark on Apple Silicon and Bark on CPU are both far slower
than realtime; Kokoro is faster than realtime even with no GPU.

---

Adding a voice engine of your own is a supported path with its own checklist --
see [EXTENDING.md](EXTENDING.md). Mac specifics are in [MAC.md](MAC.md), and
what arrives on a first run is in [INSTALL.md](INSTALL.md).
