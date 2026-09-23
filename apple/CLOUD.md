# Optional: a cloud writer

**How to store a key is the heading of the README.** Google and OpenRouter
use two files each. Comfy Cloud: sign into the app. This page is what
those lanes turn on.

**You do not need this.** Everything here is off by default, costs money, and
replaces one step — the script writing — with a hosted model. The local writer is
the supported path and the one every published episode used.

Read this only if you want a bigger writer than your card can hold, or you want
to write scripts on a machine with no usable GPU at all.

For the **writer**, nothing else changes: voices, music, images and video stay
local.

**One exception worth knowing before you set a key.** `OTR_GOOGLE_API_KEY` is
not writer-only. If a graph selects the `google_image` engine, stills are minted
through that same key and billed the same way. Check your image dropdowns before
setting it.

---

## The lanes

| Lane | Who bills you | How it turns on |
|---|---|---|
| **OpenRouter** | OpenRouter, per token | `openrouter.secret`, or a path in `openrouter_api_key.location`, or `OPENROUTER_API_KEY`. |
| **Google** | Google, per token / per image / per second of video | `google.secret`, or a path in `google_api_key.location`, or `OTR_GOOGLE_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY`. Writer, voices, music and stills have generous limits; **Veo allows only 2 requests a minute and 10 a day per model on paid Tier 1**, against ~16 clips per episode, so the `google_still_*` presets composite stills instead. Switch to `google_veo_low_*` on a higher tier. |
| **Comfy Credits** | Your Comfy account's credits | Sign into the Comfy app -- that sign-in is the only credential the pack reads (no key file, no server-side env var). Headless only: `OTR_COMFY_API_KEY` on the *submitting* machine, sent by `scripts/otr_api.py` as `extra_data.api_key_comfy_org`. There is no enable flag: the pick plus the key is the whole switch. |

They are not symmetric, and older notes in `docs/` claim they "differ only in who
pays" — they do not. **OpenRouter and Google are gated purely on a key being
present**: set the key and the lane is reachable. **Comfy Credits needs an
explicit opt-in flag** as well as an account.

There is no `OTR_ENABLE_OPENROUTER` flag. It existed once, it was removed, and
setting it does nothing.

## Turning one on

The two files live in the pack folder and do not depend on Desktop inheriting
your user environment. Environment variables still work: set them **before**
ComfyUI starts, in the shell or session that launches it. Setting a variable
in a different terminal after the server is up does nothing -- the server
reads its own environment.

```bash
# Linux / macOS
export OPENROUTER_API_KEY=sk-or-...

# Windows PowerShell, permanently for your user
[Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-or-...", "User")
```

Then restart ComfyUI.

**ComfyUI Desktop does not inherit user-scope Windows variables.** The
`SetEnvironmentVariable` line above is enough for a terminal you launch
`python main.py` from, and is *not* enough for Desktop. Set the key in the
environment of whatever actually starts ComfyUI.

**Never put a key in a workflow widget.** No node here asks for one. A key saved
into a graph travels with the graph to anyone you send it to.

## Picking the model

On **OTR_LedgerScriptWriter** there are paired dropdowns per lane —
`openrouter_slot_a_model`, `comfy_slot_a_model`, `google_api_slot_a_model` and
their `_b` partners. Until the lane's gate is satisfied they read
`(enable OpenRouter)` / `(enable Comfy Credits)` / `(select Google API model)`,
which is the dropdown telling you the lane is off.

Two slots exist so the creative pass and the technical pass can use different
models — a larger one to write, a cheaper one for structural work.

**The shipping cheap Comfy Cloud graphs, the deluxe graphs, and the CPU
graph pin Sonnet 5 on creative and GPT 5.6 Luna on tech.** Measured on the
saved widgets in `otr_cloud_low.json`, `otr_cloud_deluxe_3act.json`,
`otr_cloud_deluxe_audio_in_3act.json`. The Credits
combo uses the OpenRouter ids `anthropic/claude-sonnet-5` and
`openai/gpt-5.6-luna`. Luna is sent with `reasoning_effort=none` on the
Comfy OpenRouter proxy (the Credits node widget spells that `off`).
Sonnet 5 cannot turn reasoning fully off, so the creative slot sends
`low`. `openai/gpt-5.6-sol`, Terra and the `-pro` twins stay on the
dropdown; they are not the saved default.

## Refreshing the OpenRouter model list

The catalogue changes often. `scripts/otr_openrouter_refresh.py` pulls the
current list. It is one of the few scripts that **does** ship in a registry
install, specifically so this works there:

```bash
python scripts/otr_openrouter_refresh.py
```

Run it with ComfyUI's own interpreter, **from the pack's own folder**. That
folder is `comfyui-old-time-radio` in a registry install and
`ComfyUI-OldTimeRadio` in a git clone, so a hard-coded path works for one and
not the other. Restart afterwards so the dropdowns rebuild.

## What it changes

It lets you write with a model far bigger than your VRAM, and it takes the
writer out of your machine's memory budget entirely — on an 8 GB card, that is
the single largest local model in the graph. Everything else still renders
locally.

Whether a bigger model writes a better episode is yours to judge. The local
default is what the project develops against.

## Turning it off

Unset the variable and restart. The dropdowns go back to their `(enable ...)`
placeholders and the local writer takes over again. Nothing else in the graph
needs changing.
