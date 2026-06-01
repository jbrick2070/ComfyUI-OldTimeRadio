# Using OpenRouter (optional, experimental) — get a key & turn it on

OldTimeRadio runs **100% local with zero API keys by default**, and that never changes. This page is only for users who want to *experiment* with running the writer on a hosted frontier model (Claude, GPT, etc.) through [OpenRouter](https://openrouter.ai) to see if it makes a better episode.

> **Status:** opt-in feature on the `v2.0-alpha` branch. It is **off** unless you set an API key *and* an enable flag (below). With them unset, nothing remote ever runs and the local pipeline is byte-for-byte unchanged.

## Will it make a better story?

**Unknown — that's the point of trying it.** The local default (Mistral-Nemo) is the soak-tested baseline. A bigger hosted model *may* write a stronger script, but it also costs money and sends your prompt to a third party. Treat this as an A/B experiment: generate one episode local, one with OpenRouter on the **creative** slot, and compare. Don't assume remote is better until you've heard both.

## What it costs

- **Free to try:** new accounts get a small free allowance, and OpenRouter has many **free models** (append `:free` to the slug, e.g. a `...:free` model). Free models are rate-limited (~20 requests/min; 50/day until you've purchased ≥10 credits, then 1000/day) and aren't guaranteed to support structured output (see the technical-slot note).
- **Paid models:** frontier models bill against prepaid **credits** at pass-through provider pricing (no markup). OpenRouter adds a 5.5% fee — $0.80 minimum — when you *buy* credits. You only spend while remote is enabled and selected.
- **Built-in guard:** OTR enforces a hard per-run spend/token ceiling and aborts before exceeding it, so a runaway loop can't quietly burn credits.

## Step 1 — Create an OpenRouter account

1. Go to [openrouter.ai](https://openrouter.ai) and click **Sign In**.
2. Sign up with Google or email. **No credit card needed** for the free tier.

## Step 2 — Create an API key

1. Click your **profile icon** (top right) → **Keys** (or go straight to [openrouter.ai/keys](https://openrouter.ai/keys)).
2. Click **Create Key**, give it a name like `OldTimeRadio`, and copy it. It looks like `sk-or-v1-…`.
3. Copy it now — OpenRouter only shows the full key once.

*(Optional, only for paid models: add a few dollars of credits on the [Credits page](https://openrouter.ai/settings/credits). Skip this if you'll use `:free` models.)*

## Step 3 — Save the key on your PC (Windows)

Open **Command Prompt** (not PowerShell — its console history is written to disk; cmd's isn't) and run, with your real key in place of the placeholder:

```
setx OPENROUTER_API_KEY "sk-or-v1-PASTE-YOUR-REAL-KEY-HERE"
```

No quotes are required (the key has no spaces), but they're safe — `setx` strips them, so the stored value stays clean. This saves to your User environment permanently. **Never paste your real key into a chat, a commit, or a screenshot.**

## Step 4 — Turn it on and pick your models

Two named remote slots, **A** and **B**, each point at a real model slug of your choice. Set them once:

```
setx OTR_ENABLE_OPENROUTER 1
setx OPENROUTER_MODEL_A "anthropic/claude-3.5-sonnet"
setx OPENROUTER_MODEL_B "openai/gpt-4o"
```

Browse and confirm exact, current slugs at [openrouter.ai/models](https://openrouter.ai/models) — they version over time. To go fully free, point A/B at a `:free` model slug.

> **Since the 2026-06-01 four-dropdown router, `OPENROUTER_MODEL_A`/`_B` are a *fallback*, not the primary control.** In the ComfyUI UI you now pick the actual slug per-workflow from the **slot-model dropdowns** (Step 5). The env vars are used when a slot's picker is left unset (headless runs, or a workflow saved while remote was off). Resolution order per slot: the slot-picker widget value → `OTR_OPENROUTER_SLOT_A_DEFAULT`/`_B_DEFAULT` → `OPENROUTER_MODEL_A`/`_B` → the built-in recommended default.

**Then restart ComfyUI in a fresh terminal** so it reads the new variables.

## Step 4b — (Optional) Route for speed or cost

One model slug is usually served by several providers, and OpenRouter picks one per call. Because the writer makes **many** LLM calls per episode (news, cast, outline, each dialogue line, critic, title), biasing that choice can make a real difference to wall-clock time or spend. Two ways to set it, simplest first:

**On the slug** — append `:nitro` (fastest provider) or `:floor` (cheapest provider) right on the model id:

```
setx OPENROUTER_MODEL_A "anthropic/claude-3.5-sonnet:nitro"
```

**Or with an env knob** — per slot, or globally for both:

```
setx OPENROUTER_A_ROUTE nitro     :: just slot A   (nitro | floor | throughput | price | latency)
setx OPENROUTER_SORT throughput   :: both slots, unless a slot/slug overrides it
```

- **`nitro` / `throughput`** — route to the fastest provider. Best when you want each episode generated as quickly as possible.
- **`floor` / `price`** — route to the cheapest provider. Best for keeping spend down on long runs.
- **`latency`** — lowest time-to-first-token.
- **Unset (default)** — OpenRouter's normal load-balancing; a good neutral choice.

Precedence is most-specific-first: a `:nitro`/`:floor` on the slug wins, then `OPENROUTER_A_ROUTE`/`OPENROUTER_B_ROUTE`, then `OPENROUTER_SORT`. The hard cost ceiling still applies either way — a faster provider is never an uncapped one. The route you used is recorded in the episode's run meta so a run is reproducible.

## Step 5 — Use it in ComfyUI (the four-dropdown router)

Since 2026-06-01 the `OTR_LedgerScriptWriter` node has **four** model dropdowns in two layers:

**Layer 1 — where each pass runs (the routing selectors):**

- **`creative_writing_model`** — the narrative passes (outline, cast, dialogue, polish). Choices are your **local** models plus **`openrouter:slot-a`** and **`openrouter:slot-b`**. This is the best slot to try remote.
- **`technical_model`** — the structured/JSON passes. Same choices. See the technical-slot note below before sending it remote.

**Layer 2 — which real OpenRouter model each slot is (the slug pickers):**

- **`openrouter_slot_a_model`** — the actual slug behind `openrouter:slot-a`.
- **`openrouter_slot_b_model`** — the actual slug behind `openrouter:slot-b`.

So to run the creative passes on a hosted model: set **`creative_writing_model` → `openrouter:slot-a`**, then pick the model you want in **`openrouter_slot_a_model`** (e.g. `anthropic/claude-3.5-sonnet`). Leave `technical_model` on the local default. Queue as usual.

The slug pickers are **passive**: choosing a slug in `openrouter_slot_a_model` does nothing on its own — it only takes effect when a selector points at `openrouter:slot-a`. So you can pre-set both slots and flip between local and remote just by changing the selector.

**Conditional default (fresh node only):** when remote is enabled, a freshly-dropped writer node defaults `creative_writing_model` to `openrouter:slot-a` (so the feature is one click away); `technical_model` stays local and is never auto-flipped. A **saved** workflow always keeps its own values — defaults apply only to new nodes.

## Refresh the model list (populating the slug pickers)

The slug dropdowns are built from an on-disk cache, never a live network call (so opening the node menu is always instant and offline-safe). Until you refresh it, the pickers show only a recommended default plus an `(enable OpenRouter)` / `(no OpenRouter models cached …)` sentinel. Populate it with the refresh script:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_openrouter_refresh.py
```

It fetches the live [openrouter.ai/models](https://openrouter.ai/models) list and writes `models/openrouter_models.json` (per-machine, git-ignored). Re-run it whenever you want to see newly added models. A failed/offline run keeps your existing cache and never crashes. Reload the node (or restart ComfyUI) to see the refreshed list.

## Narrowing the slug list (optional filters)

The slug pickers can get long. These env vars trim them (filters only — they never hide your local models or the slot handles):

```
setx OTR_OPENROUTER_PROVIDER_FILTER "anthropic,openai"   :: only these providers
setx OTR_OPENROUTER_MODEL_ALLOWLIST "anthropic/claude-3.5-sonnet,openai/gpt-4o"  :: only these slugs
setx OTR_OPENROUTER_MODEL_DENYLIST "some/model"          :: hide these slugs
setx OTR_OPENROUTER_SLOT_B_REQUIRE_JSON 1                :: slot B: only structured-output models
```

`REQUIRE_JSON` is **per slot** (default off). It's meant for slot B when you route the technical slot remote — it never narrows slot A, so a creative-only model is never hidden from A because B needs JSON. You can also pin each slot's default with `OTR_OPENROUTER_SLOT_A_DEFAULT` / `OTR_OPENROUTER_SLOT_B_DEFAULT`.

## Your saved model is never silently swapped

A workflow saved with a specific slug keeps it. If you reload that workflow and your local cache is stale or cold (so the slug isn't in the current dropdown list), OTR **preserves your saved slug, logs a warning, and still uses it** — a stale cache is not a missing model, and OTR never quietly substitutes a different one. If a remote call genuinely fails, the run aborts with a clear error (fail-loud) rather than swapping to another model mid-episode. Re-run the refresh script to bring the slug back into the visible list.

## Technical slot — read before using remote there

The technical passes must emit strictly valid JSON. Locally that's guaranteed by grammar-constrained decoding. A remote model can only approximate it via "structured outputs," which **not every model (especially free ones) supports**. OTR is **fail-closed**: if a remote technical reply can't be validated, the run aborts with a clear error rather than writing bad data.

**Recommendation:** keep `technical_model` on the **local default** and only put OpenRouter on `creative_writing_model`. If you do want remote technical, choose a model that explicitly supports structured outputs.

## Privacy

OpenRouter does not log prompts/completions by default, and routes around providers that do unless you opt in. Your script prompts (news summaries, character/dialogue text) will leave your machine when remote is enabled. If that matters to you, keep it off — the local pipeline never sends anything anywhere.

## Turn it off

Set the gate to `0` (or clear it) and restart ComfyUI:

```
setx OTR_ENABLE_OPENROUTER 0
```

The OpenRouter A/B options disappear from the dropdowns and the pipeline is back to 100% local. Your saved key stays put for next time; to remove it entirely, delete the `OPENROUTER_API_KEY` user variable.

## Troubleshooting

- **No OpenRouter A/B in the dropdown** → `OPENROUTER_API_KEY` or `OTR_ENABLE_OPENROUTER=1` isn't set, or ComfyUI wasn't restarted in a new terminal after `setx`.
- **The slug picker (`openrouter_slot_a/b_model`) only shows a recommended default + `(no OpenRouter models cached …)`** → the catalog cache is empty. Run `scripts\otr_openrouter_refresh.py`, then reload the node. (You can still type/keep any valid slug — a saved one is preserved even when out of the visible list.)
- **Run aborts on a technical pass** → the remote model didn't return valid JSON (fail-closed). Switch that slot to local or to a structured-output-capable model.
- **"Insufficient credits" / rate-limit errors** → you're on a paid or rate-limited model; add credits or switch to a `:free` slug.
- **Cost ceiling abort** → expected guard; raise the limit deliberately or use a cheaper/free model.
- **Episodes feel slow** → try `OPENROUTER_SORT throughput` (or `:nitro` on the slug) to route every call to the fastest provider; use `:floor` / `price` if you'd rather cut cost than time.
