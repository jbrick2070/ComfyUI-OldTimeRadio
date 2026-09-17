# The writer models -- what actually writes your script

Every episode starts as a script, and a language model running on your own
machine writes it. Two dropdowns on the **OTR_LedgerScriptWriter** node choose
which one: `creative_writing_model` and `technical_model`.

This is the largest download in the pack and usually the longest part of a run.
It is also the one choice where picking a model your machine cannot hold costs
you the whole episode.

You do not have to choose -- every pre-set graph already ships a writer that
fits its own tier. Which one depends on which graph you opened:

| Graph | Both slots ship |
|---|---|
| 8 GB NVIDIA, AMD, CPU | **Qwen3.5-4B NF4** (`Qwen/Qwen3.5-4B:nf4`) |
| 16 GB Mac, canonical | **Qwen3.5-4B full** (`Qwen/Qwen3.5-4B`, Quant `none`) |
| 16 GB+ NVIDIA (`otr_16gb_*`) | **gemma-4-12b-it** (23.9 GB) + Quant `bnb_nf4` |
| Comfy Cloud cheap (`otr_cloud_low*`) | creative **Sonnet 5** (`anthropic/claude-sonnet-5`), tech **GPT 5.6 Luna** (`openai/gpt-5.6-luna`) |
| Comfy Cloud deluxe (`otr_cloud_deluxe_3act`) | creative **GPT 5.6 Sol** (`openai/gpt-5.6-sol`), tech **GPT 5.6 Luna** (`openai/gpt-5.6-luna`) |

The Comfy Cloud graphs split the two slots on purpose -- cheap SKUs use
Sonnet 5 to write and Luna for JSON; deluxe uses Sol to write and the same
Luna for JSON -- and bill Credits rather than VRAM. Leaving those two
dropdowns alone is still the right answer there.

**Qwen3.5-4B** is the only writer here with a finished episode to its name on
all three of an 8 GB NVIDIA card, a 16 GB NVIDIA card and a 16 GB Mac, and it
downloads itself. It is also what the AMD and CPU graphs ship, though nobody has
published an episode from either of those yet -- [MACHINES.md](MACHINES.md)
marks both cells unmeasured rather than proven, and that is the honest word for
them. Leaving both slots alone is a good answer on every one of those machines.
The 16 GB+
NVIDIA graphs -- the pack's flagship tier -- ship the bigger `gemma-4-12b-it`
instead; leaving those two slots alone is still a good answer there, but it
means a roughly 24 GB download the first time you Queue, not the 8.7 GB one
described below. That 12B identity is NF4 baked into `google/gemma-4-12b-it`
-- there is no other 12B variant and no second Quant knob. It is the only
Gemma 4 12B in the catalog and in every shipping graph that uses it. Mac and
AMD stay on Qwen.

---

## Two slots, one job each

| Widget | Which passes it runs |
|---|---|
| `creative_writing_model` | The writing -- outline, cast, dialogue, polish, title |
| `technical_model` | The bookkeeping -- JSON the pipeline has to parse, reviewer verdicts, cast contract checks, format normalization |

**Set them to the same model unless you have a reason not to.** When both slots
name the same model, it loads once and stays loaded. When they differ, the pack
tears one down and loads the other every time the script crosses from a writing
pass to a structured one, and back. That is deliberate and it works -- it is how
you run a big writer on a card that can only hold one at a time -- but you pay
for it in minutes.

The reason to split them is memory, not taste: a smaller `technical_model` frees
headroom for a larger creative one.

---

## The list you see

This is the advertised list, exactly as the dropdown spells it. You can still
pick any of them -- the two Qwen identities are what the pack *ships* on 8 GB,
and `google/gemma-4-12b-it` is what the 16 GB NVIDIA graphs *ship*.

| What the dropdown says | Download | Licence | Worth knowing |
|---|---|---|---|
| `Qwen/Qwen3.5-4B:nf4 (8.7 GB, nv8 nv16 nv24)` | 8.7 GB | Apache 2.0 | **The NF4 pick.** Same weights as the row below. 8 GB NVIDIA graphs and a freshly dropped writer node save this with Quant `bnb_nf4`. No Mac tag -- there is no Metal NF4. |
| `Qwen/Qwen3.5-4B (8.7 GB, mac16-tight nv16 nv24)` | 8.7 GB | Apache 2.0 | **The full pick.** Canonical and the Mac graphs save this with Quant `none`. No `nv8` -- unquantized it does not fit an 8 GB card. |
| `unsloth/Llama-3.2-3B-Instruct (6.4 GB, mac16 nv8 nv16 nv24)` | 6.4 GB | Llama 3.2 Community | **The no-quantization row.** It is the one to pick if your machine has no `bitsandbytes` -- AMD above all. Nobody has published an episode with it yet. |
| `mistralai/Mistral-Nemo-Instruct-2407 (24.0 GB, nv16 nv24)` | 24.0 GB | Apache 2.0 | Not what the 16 GB NVIDIA graphs ship (that is `gemma-4-12b-it`, below) -- an earlier writer, still proven on 16 GB+ NVIDIA, and the pack's audio regression baseline. |
| `google/gemma-4-E2B-it (6.0 GB, nv8 nv16 nv24)` | 6.0 GB | Apache 2.0 | Compact technical-slot option. Loads the native text decoder (PBUG-20260906-07). OOM on a 16 GB Mac -- do not pick it there. |
| `google/gemma-4-E4B-it (9.0 GB, mac16-tight nv8 nv16 nv24)` | 9.0 GB | Apache 2.0 | Same family, a size up. Proven on 16 GB NVIDIA. |
| `google/gemma-4-12b-it (23.9 GB, nv16 nv24)` | 23.9 GB | Apache 2.0 | What the ordinary 16 GB NVIDIA graphs ship with. NF4 is baked into the pick -- there is no other 12B variant, so you do not also change Quant. Canonical stays Qwen; switch this row and it loads NF4 even if Quant still says `none`. Too big for an 8 GB card -- it is refused at the gate there, not at the crash. |
| `google/gemma-2-2b-it (5.2 GB, gated mac16 nv8 nv16 nv24)` | 5.2 GB | Gemma Terms of Use | The smallest of all, and **the only one that needs a Hugging Face login**. Intended as a `technical_model`, not a creative one. |

The local rows download themselves except the gated Gemma 2 pick, which still needs a Hugging Face login.

If your dropdown also shows `openrouter:`, `comfy:` or `google_api:` entries,
those are the optional paid cloud writers -- see [CLOUD.md](CLOUD.md). They are
absent unless you set a key yourself.

---

## Reading the label

The size and the machine hints in the label are not decoration. Take
`Qwen/Qwen3.5-4B (8.7 GB, mac16-tight nv16 nv24)` apart:

| Piece | Means |
|---|---|
| `8.7 GB` | What it downloads, once. Not what it occupies while running. |
| `nv8` `nv16` `nv24` | Fits an NVIDIA card of that many GB **at the Quant this pick owns**. The NF4 Qwen row has `nv8`. The full Qwen row does not. |
| `mac16` | Fits a 16 GB Apple Silicon machine. |
| `-tight` | Fits with nothing to spare. Close everything else. |
| `gated` | Needs a Hugging Face account and an accepted licence before it will download. |
| A missing tag | Do not pick it on that machine. `(24.0 GB, nv16 nv24)` has no `nv8`, so an 8 GB card should leave it alone. |

**The download figure is honest everywhere; the fit tags are the part that
changes by machine.** On NVIDIA the pack loads these models 4-bit, so a 24 GB
download runs in roughly 12 GB. Apple Silicon has no 4-bit path at all, so the
same model there costs its full size. That is the whole reason the label carries
tags instead of one number.

### The Mac exception, and it is the expensive one

The recorded result of running `google/gemma-4-E2B-it` on a 16 GB Mac is out of
memory. On a Mac that is a hard reboot, not a failed render. Leave it off a
Mac graph even though the dropdown still lists it.

**On a Mac, use the default.** `Qwen/Qwen3.5-4B` is what the Mac graphs ship
with and what every episode published on an M4 used.

---

## What downloads, and when

Nothing downloads when you pick a model. It downloads the first time you press
Queue with that model selected, and never again. The console says so:

```
[OTR] Downloading Qwen/Qwen3.5-4B -- 8.7 GB -> ... (first run only)
```

A first Queue that appears to sit still for a long time is usually this. Let it
run.

The download needs the model's size **plus 5 GB of headroom** on the drive
holding your Hugging Face cache, or it stops before starting and tells you the
numbers.

---

## The sampling knobs

Three widgets on the same node adjust how the model picks its words. They ship
tuned, and the honest advice is to leave all three alone.

| Widget | Ships at | Range | What it does |
|---|---|---|---|
| `min_p` | 0.05 | 0.0 - 0.5 | Cuts the tail of unlikely words -- the occasional off-key word in an otherwise good line. 0.0 turns it off. 0.10 is aggressive. |
| `repetition_penalty` | 1.03 | 1.0 - 1.2 | Stops a small model looping on a character's name. 1.0 turns it off. Above 1.08 commonly makes short lines worse, not better. |
| `max_new_tokens_cap` | 200 | 40 - 400 | How much the model may write per line before it is cut off. A budget, not a length target -- raising it does not make the episode longer. |

**If you want to change how the show reads, use `creativity` instead.** It is
the dial that was built for this, and it moves temperature and top_p together
to settings that are known to hold together:

| `creativity` | Result |
|---|---|
| `safe & tight` | Predictable, on the nose |
| `balanced` | The default |
| `wild & rough` | Looser |
| `maximum chaos` | As loose as it goes without the script falling apart |

The cap on `maximum chaos` is not shyness. Past it the model stops producing
usable script format at all.

---

## The two GGUF widgets, and why they do nothing

You will see `gguf_n_ctx` and `gguf_quant` further down the node. They belong to
a GGUF writer lane, and **this pack ships no GGUF writer**. There is no GGUF
entry in either model dropdown, so neither widget has anything to act on. They
are inert, and that is the correct state.

This is deliberate and it is not a gap waiting to be filled. A GGUF writer row
has no automatic download -- the file has to be fetched and placed by hand --
so the entry would have sat in the picker looking like every other one-click
choice and then failed for anyone who had not built the lane themselves. The
models it used to offer were repointed to their ordinary twins, which download
themselves, and measured faster than the GGUF lane had been anyway.

If you were sent here looking for a way to turn GGUF back on: there is not one,
and you do not need it.

Video and image GGUF files are a different dropdown and they stay. Foley, mime,
Klein, LTX, and Wan still load their `.gguf` UNets / encoders through those
lanes. That is not a writer.

---

## When it goes wrong

In the order these actually happen.

**"VRAM-fit estimate FAIL ... estimated N GB peak resident vs M GB ceiling".**
A recommendation in the log, not a hard refusal. The load still attempts; a
real OOM is the authority. On NVIDIA, an oversized NF4 pick may retry with
CPU overflow after that runtime failure.
Pick a model whose label carries your machine's tag. The ceiling itself is the
`llm_vram_ceiling_gb` widget, and raising it does not create memory; it only
moves where the failure happens.

**"GatedModelError: ... requires HuggingFace authentication".** You picked
`google/gemma-2-2b-it`, the one gated row. Either pick another one, or make a
free Hugging Face account, accept the licence on that model's page, and set
`HF_TOKEN` in the environment before ComfyUI starts. The message walks through
it. After that, the download fires on the next Queue by itself.

**"requires bitsandbytes, which is not importable on this host".** Your machine
has no 4-bit support -- most often AMD.

**Change `llm_quant_policy` to `none`. That is the fix, and picking a smaller
model instead will not work.** The check fires on the POLICY alone, before any
model is considered, so leaving the policy at `bnb_nf4` raises the same error no
matter which row you choose. Once the policy is `none` the model loads at full
size, so pick one your machine can hold unquantized -- roughly four times its
4-bit size.

The pre-set AMD, Mac and CPU graphs already ship with that policy set to `none`,
so this only comes up if you changed it or built a graph yourself.

**"InsufficientDiskSpaceError".** Exactly what it says, with the arithmetic
shown. Free up space and Queue again.

**The first run sits for a long time with no error.** It is downloading. Read
the console for the `[OTR] Downloading` line and its size.

**It finished, but it took far longer than you expected.** Check whether your
two model slots name different models. Two different writers means the pack
swaps them in and out of memory throughout the run. Set them the same.

**Nothing substitutes itself, ever.** If the model you chose cannot load, the
run stops and says so. It will never quietly write your episode with a different
model than the one the graph names.

---

Which writers have actually been proven on which hardware -- as opposed to
which ones fit on paper -- is in [MACHINES.md](MACHINES.md) section 2.

---

## A model that is not in this list

The rows above are the curated advertised set. You can run a different Hugging Face
causal LM without editing this pack: put a complete snapshot in the cache,
restart ComfyUI, pick it. That is allowed. It is not the same as what the pack
ships. Shipping a new row in the dropdown is a catalog change and wants the
git clone.

Both paths, and the checks that decide whether it will actually write an
episode, are [LLM_PREFLIGHT.md](LLM_PREFLIGHT.md).
