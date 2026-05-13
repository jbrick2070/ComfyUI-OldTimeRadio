# OTR Pipeline — Public-Facing Polish Notes

**Written:** 2026-05-13
**Purpose:** what to add/fix to make OTR something strangers can actually use to generate cool new episodes off the news feed.
**Scope:** post-cleanbreak. Don't start any of this until S15.5 → S23 is closed. The cleanbreak IS the prerequisite to being shareable.

---

## 1. The 90-second test

Most people decide whether to install in 90 seconds. They don't read the README. They look for:

- a sample of the actual output (audio + video clip)
- a one-sentence pitch they can repeat to themselves
- a sense of "will this work on my hardware"

If those three aren't visible at the top of the repo / Space page, you lose 80% of curious clickers before they ever try anything.

**Action items:**
- One 60–90s sample episode (MP3 + MP4) committed to `samples/` at the repo root
- One-line pitch in the README's first paragraph: *"Generate a 1940s-style radio drama from today's headlines. Local, news-fed, daily-fresh."*
- A hardware tier table near the top: "Works on 8GB / Works on 16GB / Recommended"

---

## 2. The install cliff

ComfyUI custom nodes are a real install barrier. Even with your survival guide, the path is:

1. Install Python 3.11
2. Install ComfyUI
3. Install custom nodes
4. Download models (multi-GB)
5. Open workflow JSON
6. Hope it loads

That's a five-step cliff before anyone sees an episode. Half the people don't have a GPU. Of the ones who do, half are on Apple Silicon. Of the ones on NVIDIA, most are on 8–12GB.

**Action items:**
- **HuggingFace Space wrapping the `8gb_safe` preset.** This is the single highest-impact thing you can do. Free-tier GPU, zero-install, one-click "generate today's episode" button. Every person who can't install ComfyUI now becomes a possible user.
- **Colab notebook fallback** for people who want to use their own Drive for model caching.
- **Pinokio script** for one-click local install for the ComfyUI-curious. Pinokio handles the Python/ComfyUI/node-install dance.
- **macOS / Apple Silicon path** documented, even if it's "use the HuggingFace Space, MPS isn't supported for X/Y." Don't leave Mac users guessing.

---

## 3. The first-run experience

The first thing a new user does after install matters more than any feature. Right now they have to:

- Pick a workflow JSON
- Pick a model
- Pick a news source
- Wait
- Hope

**Action items:**
- A `make-an-episode.bat` / `make-an-episode.sh` that picks sensible defaults and runs the whole pipeline end-to-end with one command. Pure pass-through to ComfyUI's headless CLI, but it hides the dropdown chooser.
- A "today's headlines preview" step before generation kicks off, so the user sees what the spine will be and can re-roll if they don't like it.
- A clear progress display: "Phase 1/7: Writing script (Gemma 4B)... 47s elapsed, ~3 min remaining." Hide the ComfyUI internals.
- A bundled `episode-archive/` folder where outputs auto-save with date + headline-slug filenames, not generic `output_00001.wav`.

---

## 4. The news-feed hook is the moat — make it visible

The single most interesting thing about this pipeline is "today's episode is different from yesterday's because the world is different." Most generative tools produce variations on a static prompt. Yours produces real daily-fresh content.

Right now this hook is invisible in the codebase — it's just one of many input options.

**Action items:**
- **Front-load the news feed in the README.** First demo is "run this, get an episode about whatever happened this morning." Not "set up these eight parameters."
- **Curated default feed set** (BBC, NPR, Reuters, ArXiv top-1, Nature top-1) so the out-of-box first run produces something coherent.
- **`feeds.yaml` config file** so users can add their own without editing Python.
- **Per-episode "this episode was generated from these headlines" credit screen** — both as a closing-card image and as text in the episode metadata. Makes the news-as-spine concept legible.

---

## 5. Failure modes need to be loud and useful

S17.2 already addresses Directive 1 in code. But for a public user, the error message matters as much as the fact of erroring.

Compare:
- **Bad:** `RuntimeError: AudioGen ImportError`
- **Good:** `AudioGen optional dependency missing. Run:  pip install -r requirements-audio.txt  — then re-run this workflow.`

**Action items:**
- Audit every `raise RuntimeError(...)` in the consumer nodes. Each one should tell the user (a) what failed, (b) why it matters, (c) what to do next.
- A `docs/troubleshooting.md` keyed by error message. User pastes the error, finds the fix.
- A pre-flight check script: `python -m otr.preflight` runs through "Do you have CUDA? How much VRAM? Is FLUX downloaded? Is Bark downloaded? Are RSS feeds reachable?" before the first generation. Saves people from 4-minute renders that fail at minute 4.

---

## 6. The community / showcase loop

Once shareable, the thing that turns one-time visitors into recurring users is seeing what other people made.

**Action items:**
- An `episodes/` or `gallery/` folder in-repo where users PR their own generated episodes. Audio + video + the headline that seeded it.
- A simple GitHub Action that auto-builds an `episodes/INDEX.md` from the contents so the gallery scales.
- A weekly "best episode of the week" pinned issue or Discord post — pulls people back to see what's new.
- Hashtag convention so people posting episodes on Mastodon / Bluesky / YouTube can tag them and find each other.

---

## 7. Documentation that respects the user

Right now the survival guide is for contributors. Public users need different docs:

- **Quickstart** — 15-minute path from clone to first episode. One page.
- **Hardware tiers** — which preset for which GPU, plainly stated.
- **Model swapping guide** — the dropdown stays open by design, so document how to flip Mistral → Gemma → Qwen and what each tradeoff costs.
- **News feed configuration** — how to add custom feeds, blocklist topics, weight sources.
- **"What to do when..."** — silent audio, drifted lip-sync, OOM, slow generation. User-facing answers, not stack traces.

Three documents that DON'T need to exist:
- "Architecture overview" with FreezeCascade diagrams (that's the survival guide's job)
- "Why we deleted the LLMDirector" (forensic; contributor-relevant only)
- "Standing directives audit" (internal QA discipline)

Keep contributor docs and user docs separate. The contributor docs you have are good. The user docs barely exist yet.

---

## 8. The license + expectation conversation

Public release means strangers using it for purposes you didn't predict. Worth thinking about now:

- License pick that allows derivative use but protects you from liability ("episode generated something I didn't like"). MIT or Apache 2.0 are the usual; add an explicit "AI-generated content, no warranty about output suitability" line in the README.
- Content policy: the prompt template already has safety filters. Document them so users know what the pipeline will and won't generate.
- Model licensing: every dropdown model has its own license. Bundle a `LICENSES-OF-INCLUDED-MODELS.md` so users don't accidentally violate Mistral's or Google's terms.

---

## 9. What to ship first (S24 sequencing)

If S15.5 → S23 closes the cleanbreak, S24 is the public-polish sprint. Order matters:

1. **S24.1** — One sample episode in `samples/` + README rewrite + hardware tier table. (4 hours)
2. **S24.2** — Failure-mode audit: every raise gets a useful message. (3 hours)
3. **S24.3** — Pre-flight check script. (2 hours)
4. **S24.4** — HuggingFace Space wrapping the `8gb_safe` preset. (1 full day — Space setup, model caching, frontend tuning)
5. **S24.5** — `make-an-episode.sh` one-command runner. (2 hours)
6. **S24.6** — News-feed front-loading: README rewrite around the daily-fresh hook + curated default feeds + `feeds.yaml`. (3 hours)
7. **S24.7** — User docs (quickstart, hardware tiers, model swapping, troubleshooting). (4 hours)
8. **S24.8** — `gallery/` folder + auto-INDEX action + first announcement post. (3 hours + announcement-day time)

Total estimate: 3–4 focused days of work, post-cleanbreak.

---

## 10. The honest cut

The pipeline is real and the news-fed daily-fresh hook is genuinely interesting. What gates public reach isn't whether the code works — it's whether a stranger can get to "first episode" in under 15 minutes without help. Everything in this doc is in service of that single metric.

If you can't shorten the path to a first episode, the most interesting AI radio drama tool in the world still gets 12 users. If you can, you have something people actually share.

S24 isn't extra polish. It's the difference between a portfolio piece and a thing people use.

---

## 11. Re-read criteria

Re-read this when:
- Cleanbreak (S15.5 → S23) is done and you're deciding what to do next
- You're tempted to add a new feature instead of polishing what exists
- A user tries the pipeline and bounces — the post-mortem belongs against this checklist
