# Problem statement — what LLM should OTR default to so it "just works" out of the box for most people

**Date:** 2026-06-01 · **Branch:** v2.0-alpha · **For:** thinking / round-robin review

This is written to be self-contained: a reviewer with no prior session context should be able to reason about it. It frames the question and the options — it does **not** pick an answer.

---

## TL;DR

OTR's writer can now run its LLM calls three ways: **local** (Mistral-Nemo, free/offline), **OpenRouter via the user's own key** (paid, their account), or **OpenRouter via Comfy credits** (paid, billed through the ComfyUI account). As of 2026-06-01 the old "100% local, no-paid" rule is **relaxed from a commandment to an option** — so a paid model is now allowed to be the default if it produces better stories.

The question: **which default makes the largest share of fresh installs produce a great story with the least setup?** "Best quality" and "works for everyone with zero setup" point at *different* defaults, and which one wins depends on a fact we don't yet have — how many OTR installers already have Comfy credits.

---

## The question (one line)

Maximize `P(fresh install → great story, with minimal setup steps)` across OTR's real installer population — and decide whether that default is a paid Comfy-credit LLM, local, the user's own key, or something adaptive.

## What changed (why this is newly open)

- **Local-first demoted to optional (2026-06-01).** Paid/remote is now an allowed default. So "default = Comfy-credit frontier LLM" is on the table for the first time.
- The **4-dropdown router** (creative/technical pick slot handles; slot A/B pick the actual slug) is already specced. This problem is *not* the router mechanics — it's **which default the router points at out of the box.**

## Who actually installs OTR (segments + zero-setup reach)

| Seg | Who | What runs with ZERO extra setup |
|-----|-----|---------------------------------|
| A | ComfyUI user, no Comfy credits, no OpenRouter key (plausibly the majority) | **Local only** |
| B | ComfyUI user already logged into Comfy **with credits** (API-node adopters) | **Comfy-credits** (frontier) — and local |
| C | User with their own OpenRouter key | Own-key remote — and local |
| D | Offline / privacy / free-only | **Local only** |
| E | Low-VRAM (<16 GB) | Local works but may be slow/limited; remote helps |

**The decisive unknown: how big is segment B vs A?** If most installers already have Comfy credits, a Comfy-credit default reaches most people *and* gives best quality (Jeffrey's instinct is right). If few do, a Comfy-credit default errors for the majority out of the box.

## The core tension

- **Highest story quality** → points at a remote frontier model (Comfy-credits or own-key).
- **Highest reach / zero-setup / free / offline** → points at **local** (works for ~everyone, but sub-frontier quality).

These point at *different* defaults. "Best out of the box for most people" only resolves once we know whether "most people" can actually run the paid path.

## Hard realities any answer must respect (mechanics, not philosophy)

1. A **paid default cannot run** for someone with no credits / not logged in / offline → it **errors out of the box** for those users. (Even with the local-first *rule* gone, this physics remains.)
2. **Comfy-credits path** specifics: localhost/`127.0.0.1` only, Comfy login required, prepaid credits, **per-call billing** (the writer makes *dozens* of LLM calls per episode), and it rides ComfyUI's **internal, fast-moving partner API** (maintenance/fragility risk). It also does **not** yet support bring-your-own-key.
3. **Comfy-credits proxies OpenRouter → same models** as the own-key path. So choosing it is an **onboarding/payment-rail** decision, not a quality one (quality comes from the model, e.g. Opus, either way).
4. **Local** needs the model downloaded + enough VRAM; quality is good but not frontier.

## Candidate defaults (to evaluate — none chosen)

1. **Local default.** Max reach, free, offline, no account; sub-frontier quality. Safe, lowest ceiling.
2. **Comfy-credit default.** Frontier quality; zero-setup *only* for segment B; **errors for A/D**; per-call cost; internal-API fragility.
3. **Own-OpenRouter-key default.** Smallest zero-setup reach (few have a key).
4. **Adaptive "best available."** Detect what will actually run (Comfy login+credits? own key? else local) and auto-select the best. Everyone gets a working default; best-quality for those already set up. Cost: detection logic, and doing it **offline-safe at load** is non-trivial.
5. **Guided first-run chooser.** First use asks "best quality (needs credits, ~$X/episode) vs free local," links setup. Everyone served, best informed-consent on cost; adds one step.

## What the code actually supports today (grounding, read 2026-06-01)

- **Routing is data-driven.** A slot goes remote **iff** the selected model id's catalog row has `provider == "openrouter"` (`OTR_LedgerScriptWriter.py` L622 → `make_openrouter_generate_fn`); otherwise local. So "default to remote" literally means **shipping the widget defaulting to `openrouter:slot-a`** — no other machinery needed.
- **The default widget value is `DEFAULT_LLM`** (local) today, set in `INPUT_TYPES` and re-applied in `_resolve` (L1191/L1245).
- **The only offline-safe remote signal is `openrouter_enabled()`** = `OPENROUTER_API_KEY` present **AND** `OTR_ENABLE_OPENROUTER==1` (pure env, no network) — and it is **own-key only**.
- **No Comfy login/credits detection exists anywhere in the code**, and Comfy-credits is **not a wired backend** (only own-key OpenRouter is). Adding it = a new, *execution-time*, internal-API integration (ComfyUI hidden-auth + `comfy_api` client). Critically, **credit/login availability cannot be checked at `INPUT_TYPES`/load**, so it **cannot** drive a load-time default the way `openrouter_enabled()` can.

## Best 1–2 options, grounded in the code (the ask)

**Option A — Flag-gated conditional default (fits the code as-is; lowest risk).**
Set the creative default conditionally on the existing `openrouter_enabled()`: enabled → `openrouter:slot-a`; else `DEFAULT_LLM`. Technical stays local. This is the v5 design with **zero new infra** — it keys only on the env gate that already exists and is offline-safe. *Out-of-box reach then reduces to making those two env vars trivial to set* (a one-step onboarding/setup helper). Comfy-credits can't participate here. **This is the realistic "ship it" default.**

**Option B — Setup-time capability probe + a `comfy_credits` lane (reaches the no-accounts crowd; real work + fragility).**
The only way a *remote* default reaches a fresh user who has set no env vars. Requires: (1) a new `comfy_credits` provider lane routed through the same L622 provider switch, using ComfyUI's hidden-auth + `comfy_api` client (execution-time; couples the writer's core path to Comfy's fast-moving internal API); (2) an explicit **setup / first-run probe** (a script or node action, **not** `INPUT_TYPES`) that detects what's actually available and writes the enable flag + slot binding; (3) Option A's load-time conditional default then takes over. Bigger surface, higher maintenance — but it's what "Comfy-credits out of the box" actually costs.

**My lean:** ship **A** now (it's already ~90% the v5 design and works offline for everyone), and treat **B** as a follow-on *only if* segment B (Comfy users with credits) proves large and you accept the internal-API coupling. A also de-risks B: once A's conditional default + onboarding exist, B becomes "add a lane + a probe," not a redesign.

## How to judge an option (decision criteria)

- **Reach** — share of installers for whom the default *just works* with zero setup.
- **Quality** — story quality of the default path.
- **Setup friction** — steps to a working state, and to the *best* state.
- **Cost transparency** — no surprise charges; informed consent before spend.
- **Robustness** — graceful behavior offline / no-credits / low-VRAM.
- **Maintenance risk** — coupling to internal/fast-moving APIs (Comfy-credits).
- **Reproducibility** — does the default keep saved workflows stable.

## Key unknowns to resolve before deciding

1. **% of installers in segment B** (Comfy logged-in + credits). ← most decisive; currently a guess.
2. Can OTR **detect at load, offline-safe**, whether Comfy credits / an OpenRouter key are available, to drive an adaptive default — without a network call in `INPUT_TYPES`?
3. **Real per-episode credit cost** at smoke (30/100w) and full length, creative-only vs both slots.
4. **Stability** of the Comfy-credits internal path across ComfyUI updates.

## Candidate north-star (a hypothesis to pressure-test, not the decision)

> "Default to the best model that will **actually run** for *this* user with zero setup, and make upgrading to frontier quality **one obvious step**."

That phrasing favors option 4/5 (adaptive + guided), because it satisfies both poles of the tension — but it assumes we can detect availability cheaply and that a guided upgrade is acceptable UX. **The code answers the detection half:** load-time detection exists *only* for the own-key flag (`openrouter_enabled()`); Comfy-credits availability can't be seen at load — so the "adaptive" pole is really **Option B's setup-time probe**, not anything `INPUT_TYPES` can do. If segment B turns out to be the majority, a simpler fixed default may win instead.

## Non-goals (for this document)

- Choosing the answer (this is framing only).
- The 4-dropdown router mechanics (already specced separately).
- Whether to support remote at all (decided: yes).
- The "personalized stories / local profile" feature (separate problem).

---

*Problem statement — uncommitted, v2.0-alpha. v1.1 adds two code-grounded options (A flag-gated conditional default; B setup-probe + comfy_credits lane) read from the live writer/backend. Pairs with the 4-dropdown router sprint plan (`2026-06-01-openrouter-dynamic-model-list__sprint-plan.md`).*
