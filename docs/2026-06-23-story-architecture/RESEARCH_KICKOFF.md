# OTR Story Architecture — Research Kickoff (fresh window)

**Mission:** redesign the OTR story engine from a pipeline that *subtracts badness* into one
that *creates and selects for brilliance*. The schema/plumbing is done and good. The lever now
is **structural variety + scene-level craft + taste**, not more gates.

Open a fresh window on this file and run it through `/roundtable` (grounded against the real
files below). Produce a hardened spec at `docs/2026-06-23-story-architecture/SPEC.md`, then a
coder kickoff. **Do not write production code in the research window** — plan + roundtable only.

---

## 0. The core reframe (the whole point)

Every layer OTR has added — line hygiene, craft floor, Script Doctor, stage-direction scrubs,
the freeze cascade, `grade_story` — *removes flaws*. A story can pass all of them and still be a
forgettable C+. Removing badness never adds greatness. Tonight's soak confirms it: passes grade
**42–72, almost all below B(75)**, and revision is non-monotonic (a live gemma episode went
42 → 72 → drift back to 65). The ceiling is **structural variety + craft + model**, not the schema.

Two independent external reviews AND Jeffrey's own multi-model roundtables (2026-06-22/23) land on
the **same root cause: the beat planner / structural sameness** ("every premise collapses into a
console standoff; climax off-stage; announcer narrates the outcome"). That triangulation is the
strongest signal in this whole effort — act on it.

---

## 1. What OTR ALREADY has (do NOT rebuild these)

Ground every recommendation against these real files (Windows repo
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`):

- `nodes/news_interpreter.py` — news seed → dramatic spine (a **plot** engine: "extrapolate to extremes").
- `nodes/_otr_outline.py` — grammar-validated `Beat[]` hierarchy, **no dialogue yet** (Path C: macro
  premise/setting → phase/speaker → beat intent/mood). This IS Dramatron's hierarchy already.
- `nodes/_otr_line_composer.py` — **one plain-prose line per beat**, Python attaches the `[VOICE:]`
  tag (creative step already kept JSON-free). This is the line-at-a-time surface to replace.
- `nodes/_otr_ledger_reviewer.py` — cast audit → Script Doctor diagnosis → targeted edits.
- `nodes/OTR_LedgerScriptWriter.py` — the conductor; the `_refine_loop` (v1 iterative revision,
  **keep-best** across up to 5 passes, ships the highest-grade pass) lives here.
- `nodes/_otr_story_select.py` — `score_outline` (best-of-N structural selector) + `grade_story`
  (0-100 grade of composed spoken text) + `resolve_refine_passes`.
- `nodes/_otr_story_quality_l12.py` — conflict palette / beat_role dramatic-function sequence (L1/L2).
- Plus: `craft_floor`, `creative_qa`, `line_hygiene` (cliché / on-the-nose / thesis-close flags),
  `dramatic_state`, series bible in `project_state.py`, constrained generation
  (`_otr_constrained_generate`), the freeze cascade, SDH captions.

**So:** the hierarchy, the reviewer, the QA gates, the schema, the diverge-by-metric selector, and
iterative revision all exist. The gaps below are genuinely missing.

## 2. The genuinely-missing moves (the grade lever)

1. **Diverge, then select — at the *creative* level, not the metric level.** OTR's best-of-N selects
   outlines by structural score; the refine loop selects revisions by grade. Neither pitches *radically
   different episodes* and picks on taste. Add a **pitch room**: one news seed → 3 wildly different
   takes (noir / tender / strange), a **showrunner taste pass** kills all but one. Greatness lives in
   variance; a real room pitches ten and shoots one.
2. **Write at the SCENE level as free prose, transcribe into the ledger after.** The line-at-a-time
   composer with "last N lines" context produces locally-coherent ping-pong with no scene shape (no
   build, no turn, no subtext). Let the model write a **whole scene as a playwright** — no schema, no
   per-line gate — until it's good prose; then a mechanical pass splits it into beats/lines. **The
   ledger becomes a transcription target, not the writing surface.** This also lets a lot of per-line
   repair machinery be deleted.
3. **Theme & ending first; review as a listener, not an auditor.** `news_interpreter` is plot-first;
   A+ radio is about a person, a fear, a longing — the news is the vehicle. Lock the **human truth in
   one sentence + write the final 20 seconds FIRST**, then write toward it. Add a **taste reviewer**
   ("where did I lean in, where did I check out, quote the best line, rewrite only the 3 flattest")
   alongside the defect-naming Script Doctor.

## 3. Candidate "clean room" (bold = new; rest exists) — with prompt seeds

1. Assignment desk (`news_interpreter`) → **THREE candidate seeds, not one** ("surface the 3 headlines
   that could become a great half-hour drama; the real fact + the human question it forces").
2. **Pitch room — DIVERGE** ("pitch 3 completely different episodes from this seed — different genre,
   protagonist, emotional core; 4 sentences each; make them fight for the slot").
3. **Greenlight — SELECT** ("showrunner with taste and no budget: pick the ONE that makes a listener
   sit in their parked car; one line why; kill the rest").
4. **Theme & ending** ("what is this really about, one human sentence? now write the last 20 seconds").
5. Character interviews (augments casting) ("you ARE this character — 3 lines in your speaking voice:
   what you want tonight, what you fear, what you'd never admit").
6. Beat the story (`_otr_outline`, keep grammar validation) — **require every beat to change the
   temperature** (someone gains/loses power); a beat that doesn't turn gets cut.
7. **Playwright pass — write SCENES as prose** (replaces line-at-a-time) ("write this scene as a
   playwright; one wants what the other won't give; they talk around it; all dialogue + what we hear,
   no stage directions; write till it's good, ignore length").
8. **Table read — taste reviewer** (augments Script Doctor) ("where did you lean in / check out; quote
   the best line; rewrite only the 3 flattest, sharper, more under the surface").
9. Transcribe to ledger (the existing stamp step) — mechanical, constrained, **the only JSON in the flow.**

## 4. External repos to MINE (prompts/patterns, not whole clones)

- **`johnnie193/Open-Theatre`** (interactive drama) — director/actor agent modes, character memory,
  plot-chain tracking, real prompt files (`prompt_drama_v1_eng.md`, `prompt_drama_v2_eng.md`, director
  reflection, player/character prompts). **The highest-value lift** for the beat-planner upgrade +
  "who acts next" scene-state. Adapt the prompts; don't import the framework.
- **`XucroYuri/how-to-make-script`** (MIT) — rubric / route→generate→review→self-check + sub-skills.
  Runtime not implemented, so treat as a **rubric library** for upgrading the thin `grade_story`
  critique into structured revision guidance.
- **`google-deepmind/dramatron`** (Apache-2.0) — the canonical hierarchy (logline→characters→plot→
  locations→dialogue). OTR already has this; use only as a reference for the character/location layer.
- `stefanfrench/radio-drama-generator` (Apache-2.0) — closest name, **least useful** (POC behind OTR);
  steal only the doc→transcript→TTS wiring ideas. License caveat: OuteTTS v1 CC-BY-4.0, newer
  CC-BY-NC-4.0 — matters for commercial-clean.
- `mozilla-ai/document-to-podcast`, `lfnovo/podcast-creator`/`open-notebook` — explainer-audio, not
  drama-first; only the transcript/TTS plumbing is relevant (OTR already exceeds it).

## 5. Hard constraints + honest caveats (the new window must respect these)

- **Local-model ceiling is real.** gemma-12b / mistral-nemo top out ~B even when unconstrained
  (tonight's soak). Diverge-select raises quality by *selection* — the pool must contain a good one,
  so pair scene-prose with the **frontier OpenRouter lane** (already built, opt-in, cost-guarded) OR
  accept B as the local ceiling. Don't assume freeing the model yields A+.
- **Scene-prose is the highest-risk change.** It rewrites the compose surface AND the ledger flow; the
  **prose→beats transcription parser (speaker attribution from free prose) is the hard part**, and the
  entire downstream (freeze cascade, audio per-line voices, video shot roles) depends on clean per-beat
  speaker tags. Sequence it LAST, behind a flag, with the line-at-a-time path intact until proven.
- **No-fallbacks pipeline + the working A/V are sacred.** The render (ltx_av bookends + visualizer
  beats, HuMo-free as of `267a53e`) and the audio voice bank ship today — don't destabilize them for a
  story change. Story work is upstream of the ledger; keep it there.
- **Ledger schema** = `nodes/production_ledger.py` (`Ledger` class, `new_ledger()`); version source in
  `_otr_ledger.py`; section schemas in `_otr_{image,video}_engines/schemas.py`. The transcription pass
  writes into this; don't fork it.
- UTF-8 no BOM, SFW, commercial-clean licensing, default-OFF/byte-identical until proven (every story
  change ships dark behind a flag, same as the refine loop did).

## 6. Deliverable

Run `/roundtable` (GPT + Gemini + DeepSeek/Grok, Claude grounded judge) over this brief, **grounded
against §1's real files** so the panel can't propose rebuilding what exists. Output:
`docs/2026-06-23-story-architecture/SPEC.md` (the hardened clean-room + the 2-3 highest-leverage
sprints, sequenced by risk: pitch-room diverge/select + taste-review FIRST (cheap, big lever),
scene-prose LAST (risky)), then a coder kickoff. Confirm the local-ceiling decision (frontier lane vs
accept-B) with the operator before the scene-prose sprint.
