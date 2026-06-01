# Comfy Credits remote LLM — setup

The **Comfy Credits** lane lets the writer's creative and/or technical slot
run on a frontier model billed to your **ComfyUI account credits** instead of
your own API key. It is the sibling of the [own-key OpenRouter lane](openrouter-setup.md):
ComfyUI's credit-billed text path *is* the OpenRouter partner node, so both
lanes expose the same frontier catalog — they differ only in **who pays**.

- **OpenRouter lane** → billed to your `OPENROUTER_API_KEY`.
- **Comfy Credits lane** → billed to your logged-in Comfy account's prepaid credits.

The lane is **opt-in and default-off**. With it disabled, nothing changes: the
dropdowns, the offline baseline, and the byte-identical audio path are untouched.

## Enable it

1. **Log in to a Comfy account with credits.** In ComfyUI: `Settings → User`
   to log in, `Settings → Credits` to top up (prepaid — no surprise charges).
   API access requires `127.0.0.1` / `localhost` (or a Comfy API key on a
   non-whitelisted host). See ComfyUI's *Partner Nodes Overview*.
2. **Set the OTR opt-in flag**, then restart ComfyUI in a fresh terminal so the
   process sees it:

   ```
   setx OTR_ENABLE_COMFY_CREDITS 1
   ```

3. On the **1. Story Writer** node, two pickers appear:
   `comfy_slot_a_model` (creative) and `comfy_slot_b_model` (technical). Pick a
   model in each. Then set `creative_writing_model` to **`comfy:slot-a`** and/or
   `technical_model` to **`comfy:slot-b`** to route that slot through Comfy
   Credits. Leaving the selector on a local model id keeps that slot local.

When the lane is disabled the pickers show **`(enable Comfy Credits)`** and the
`comfy:slot-a/b` handles are absent from the selectors.

## Recommended defaults

| Slot | Default slug | Why |
|------|--------------|-----|
| creative (`comfy_slot_a_model`) | `anthropic/claude-opus-4.7` | strongest Anthropic on Comfy's curated catalog (the own-key lane uses 4.8) |
| technical (`comfy_slot_b_model`) | `deepseek/deepseek-v4-pro` | cheap, stable structured output |

Override per slot without changing the pick via
`OTR_COMFY_SLOT_A_DEFAULT` / `OTR_COMFY_SLOT_B_DEFAULT`. The full pinned catalog
lives in `nodes/_otr_comfy_backend.py` (`COMFY_LLM_MODELS`) — bump it when
ComfyUI's partner catalog changes.

## Knowing which model ran

The resolved slug is surfaced three ways: the widget tooltip names it, a
`[ComfyCredits] … → <slug>` line is logged at resolution, and the resolved
public slug is stamped into run meta (the auth token is never logged or
stamped).

## Cost guards

Belt-and-suspenders on top of prepaid credits:

- `OTR_COMFY_MAX_TOKENS_PER_CALL` (default 32768) — per-call ceiling, enforced
  **before** the network call.
- `OTR_COMFY_MAX_TOKENS_PER_RUN` (default 300000) — per-episode ceiling, reset
  by the writer at the top of every run.
- `OTR_COMFY_A_MAXTOK` / `OTR_COMFY_B_MAXTOK` — per-slot output caps.

A failed call **aborts the run** with a clear error — there is no mid-episode
fall-back to a local model and no silent remote→remote swap.

## First-run endpoint check (operator)

The Comfy proxy surface is env-overridable so you can confirm the exact,
version-correct endpoint at your first credit-billed run without a code change:

- `OTR_COMFY_API_BASE` (default `https://api.comfy.org`)
- `OTR_COMFY_CHAT_PATH` (default `/proxy/openrouter/v1/chat/completions`)

If the first live run fails with a clear "confirm OTR_COMFY_API_BASE /
OTR_COMFY_CHAT_PATH" error, point these at the live Comfy proxy and re-run. The
lane is isolated behind the provider seam, so a mismatch degrades to that error
— it never crashes the writer.
