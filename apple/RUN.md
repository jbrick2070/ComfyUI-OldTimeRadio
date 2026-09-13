# Your first episode

The short version: open the canonical workflow, press **Queue**, wait. You do not
have to choose anything. Every dropdown already has a working value, and the ones
set to *roll* pick for themselves, so two runs in a row give you two different
shows.

---

## 1. Open the workflow

**Workflow → Browse Templates → EXTENSIONS → comfyui-old-time-radio.** There is
exactly one entry, **`otr_canonical`**. That is the authored graph, and it is the
one to start with on any machine: it names no vendor anywhere and resolves your
GPU at run time, so the same file is correct on NVIDIA, on Apple Silicon and on
CPU. (Dragging `workflows/otr_canonical.json` onto the canvas loads the same
file.)

If you would rather have a graph pre-set for your hardware, drag one of the saved
variants in `workflows/variants/` onto the canvas instead —
[MACHINES.md](MACHINES.md) section 1 names the file for your machine. Those do
not appear under Browse Templates; they are files you drag.

## 2. Press Queue

The first run downloads about 12 GB (see [INSTALL.md](INSTALL.md) section 6) and
then writes, casts, performs, scores and cuts an episode. Later runs skip the
download.

Expect a first episode to take a while — the writing pass alone is a local
language model producing a full script before a single frame is drawn. On a
16 GB NVIDIA card a short episode is minutes, not seconds; on CPU it is a long
wait, and that is the model, not a hang.

## 3. Find it

```
<your ComfyUI output folder>/otr/obs/
```

That is where a finished episode lands — the `.mp4`, with its audio mixed,
captions burned and credits rolled. The working files for each run live one level
up in `otr/episodes/<episode>/` if you want the stems.

**If nothing is in `otr/obs/`, the run did not finish**, however green the console
looked. That folder is the finish line.

---

## The widgets worth touching

Everything else has a considered default. These are the ones that change the
show, all on the **OTR_LedgerScriptWriter** node unless noted.

| Widget | What it does |
|---|---|
| `episode_title` | Leave it blank and the show titles itself. Whatever you type becomes the title card. |
| `num_characters` | How many speaking parts. Ships at 2. |
| `act_count` | 1 for a short one, 3 for a full show with act breaks. Ships at 1. |
| `custom_premise` | A sentence or two of your own. Blank means the source bank decides. |
| `source_bank` | Where the story comes from — ships on *roll*, which picks any eligible bank. |
| `visual_style` | How it looks — ships on *roll*, which picks any of the nine. |
| `creativity` | `balanced` by default. |
| `seed` (on the sampler nodes) | Fix it to reproduce a run; leave it random for variety. |

Two of those ship on **roll** on purpose: the canonical is meant to hand you a
different show each time you press Queue. Pin them when you want to compare two
runs, because a rolled bank and a rolled style change more than anything else you
could adjust.

## Writing your own story

Set `source_bank` to **My Story** and put your premise in `custom_premise`. That
bank exists to take your idea and produce it, rather than adapting something.

## Changing what renders it

The engine dropdowns live on **OTR_VideoDirector** (video and images),
**OTR_CastLock** and the voice nodes (speech), and **OTR_StableAudioTheme**
(music). The canonical ships:

- **Video:** three procedural lanes that draw their own frames — nothing to
  download, and they work on every machine including CPU.
- **Images:** Z-Image-Turbo, sitting **dormant**, because those three video lanes
  consume no still. Switch one to a still-consuming lane and the image weights
  download then.
- **Voices:** Kokoro, on every slot. It is the only one-click voice on every
  platform, which is why it is the default.
- **Music:** Stable Audio 3.

Before you change one, check it against your machine in
[MACHINES.md](MACHINES.md) section 2 — it says what runs where, how big the
download is, and whether it needs another node pack.

---

## When something goes wrong

**The render stops naming a missing class.** The lane you picked needs a
third-party node pack. The error names the pack and its URL; install it into
`custom_nodes/`, restart, try again. [MACHINES.md](MACHINES.md) section 1 lists
which lanes these are.

**The render stops naming a missing file.** The lane you picked needs weights
that do not download themselves. [MACHINES.md](MACHINES.md) section 3 says which
repository it comes from and which folder it goes in. The error names the file;
it never quietly substitutes a different one.

**It refuses before downloading, saying the writer will not fit.** You picked a
language model bigger than your card's ceiling. Pick a smaller one — the writer
column in [MACHINES.md](MACHINES.md) section 2 shows which fit your machine.

**No mp4, and the log mentions ffmpeg or ffprobe.** See
[INSTALL.md](INSTALL.md) section 3 — you need both binaries, and the bundled
wheel supplies only one of them.

**Captions are blank boxes on Linux.** Install a monospace TTF.

**It finished but nothing is in `otr/obs/`.** Read the console from the end
backwards for the first error. A run that ends without publishing did not
succeed.
