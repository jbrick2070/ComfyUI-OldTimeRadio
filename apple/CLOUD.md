# Optional: a cloud writer

**You do not need this.** Everything here is off by default, costs money, and
replaces one step — the script writing — with a hosted model. The local writer is
the supported path and the one every published episode used.

Read this only if you want a bigger writer than your card can hold, or you want
to write scripts on a machine with no usable GPU at all.

Nothing else in the pipeline changes. Voices, music, images and video stay local.

---

## The three lanes

| Lane | Who bills you | How it turns on |
|---|---|---|
| **OpenRouter** | OpenRouter, per token | Set `OPENROUTER_API_KEY`. That is the whole gate. |
| **Google** | Google, per token | Set `OTR_GOOGLE_API_KEY` (or `GEMINI_API_KEY`, or `GOOGLE_API_KEY`). |
| **Comfy Credits** | Your Comfy account's credits | Set `OTR_ENABLE_COMFY_CREDITS=1` **and** be logged into Comfy. |

They are not symmetric, and older notes in `docs/` claim they "differ only in who
pays" — they do not. **OpenRouter and Google are gated purely on a key being
present**: set the key and the lane is reachable. **Comfy Credits needs an
explicit opt-in flag** as well as an account.

There is no `OTR_ENABLE_OPENROUTER` flag. It existed once, it was removed, and
setting it does nothing.

## Turning one on

Set the environment variable **before** ComfyUI starts, in the shell or session
that launches it. Setting it in a different terminal after the server is up does
nothing — the server reads its own environment.

```bash
# Linux / macOS
export OPENROUTER_API_KEY=sk-or-...

# Windows PowerShell, permanently for your user
[Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-or-...", "User")
```

Then restart ComfyUI.

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

## Refreshing the OpenRouter model list

The catalogue changes often. `scripts/otr_openrouter_refresh.py` pulls the
current list. It is one of the few scripts that **does** ship in a registry
install, specifically so this works there:

```bash
python custom_nodes/comfyui-old-time-radio/scripts/otr_openrouter_refresh.py
```

Run it with ComfyUI's own interpreter, from your ComfyUI root. Restart afterwards
so the dropdowns rebuild.

## What it will and will not fix

**It will** let you write with a model far bigger than your VRAM, and it removes
the writer from your machine's memory budget entirely — which on an 8 GB card is
the single largest local model in the graph.

**It will not** make episodes better in a way the project chases. Script quality
is settled and is not something a paid model is adopted to raise. Use a cloud
writer because your hardware can't hold a local one, not because you expect
better prose.

## Turning it off

Unset the variable and restart. The dropdowns go back to their `(enable ...)`
placeholders and the local writer takes over again. Nothing else in the graph
needs changing.
