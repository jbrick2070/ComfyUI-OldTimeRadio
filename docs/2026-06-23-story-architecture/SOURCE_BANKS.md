# OTR Source Banks -- modular, toggleable story/asset feeds (design capture)

Status: BRAINSTORM captured 2026-06-23 night (operator-directed). A NEXT-sprint item, SEPARATE from the
Increment-1 story-architecture sprint (`SUBAGENT_SPRINT.md`). Not built yet. The point: stop hardcoding
the news feed; turn the front-end into a registry of toggleable **source banks**, where a bank is a
**feed + an interpreter + a mode**, not just a feed.

## The core insight

`nodes/news_interpreter.py` is already the "source brain" (one control-plane LLM call -> 4 briefs:
casting_brief, script_brief, news_close_brief, key_terms). Generalize it: each enabled bank produces the
same `script_brief` contract; everything downstream (outline -> cast -> voice -> audio -> video -> OBS)
is unchanged and bank-agnostic. **Different sources want different front-end logic**, so a bank carries
an INTERPRETER, not only a URL.

```
SourceBank = { id, enabled:bool, feeds:[url...], interpreter:news|archive_seed|pd_adapt,
               mode:seed|adapt, weight:float, rights_status_field }
conductor: pick an ENABLED bank per episode (weighted/random) -> interpreter -> script_brief -> (rest unchanged)
```

## The bigger picture: an OPEN WRITER-ENGINE platform (mirror the video platform)

This is the real shape, and OTR already built the reference implementation -- on the VIDEO side. The
Open Video Model Platform (`_otr_video_engines` registry + role_compat + `OTR_VideoDirector` selecting an
engine per role, every engine producing the SAME thing: a clip for a shot) is exactly the pattern the
WRITER front-end should take. Don't invent an architecture -- mirror the proven one onto the other half.

- **Universal seam = the Outline (`Beat[]`).** A video engine's contract is "produce a clip for a shot";
  a WRITER engine's contract is "produce a validated `Beat[]` outline (+ cast brief), by whatever internal
  logic." That single boundary is what every writer engine meets; the entire production back-half
  (compose -> cast -> voice -> audio -> video -> OBS) is the uniform consumer, unchanged.
- **A writer-engine registry + a StoryDirector** (mirror `_otr_video_engines` + `OTR_VideoDirector`):
  each engine declares its inputs (feed item | source text | human outline | brief) + capabilities + a
  local/frontier tier; the director selects one per episode; validation/gating mirrors role_compat.
- **The engines (each is just "an Outline producer"):**
  - `news` / `archive_seed` -- reach the Outline via the SHARED planner (seed -> generate_outline).
  - `pd_adapt` -- reaches the Outline FROM the source text (extract -> Beat[] directly), skipping the
    planner -- which is WHY it escapes the beat-planner sameness.
  - `frontier_writer` -- the same interface pointed at the OpenRouter lane; the local-ceiling answer
    becomes a per-engine config, exactly like VRAM tiers gate video engines.
  - `human_outline` -- you hand it a Beat[] / brief; it validates + passes through.
- **Two axes, one architecture:** which BANK (the input, above) x which ENGINE (the logic). The source
  banks are the INPUTS to these engines.

**Realism:** a real multi-step project, the same size the video platform was -- but the template
de-risks it (copy the shape). Incremental: define the `Outline`-producer interface, wrap the EXISTING
news path as engine #1 (proves the seam, byte-identical), THEN add engines. The tiers below are the
first three engines.

## The three interpreter tiers (ordered cheapest -> richest; build in this order)

### Tier 1 -- `news` (EXISTING, keep)
Science/news RSS -> "extract real fact + human question -> INVENT a premise" (the current
`news_interpreter` "extrapolate to dramatic extremes" logic). Subject to the local-model ceiling +
the beat-planner sameness -- the known weakness.

### Tier 2 -- `archive_seed` -- the FRANKENSTEIN HYBRID (~free; do FIRST as an experiment)
- **Build = a feed swap.** Point the EXISTING interpreter at the media-archive RSS (LOC / NFPF / ACE
  below). The logic is feed-agnostic -- it only needs an item with a fact + a human angle, and
  "silent-era print resurfaces" / "rediscovered training film" IS that.
- **Why first:** near-zero effort, and it answers "how far does the current engine get on RICHER,
  on-brand seeds?" before investing in tier 3. The output flavor shifts to archive/film/preservation
  themes (on-brand for an OTR project + for Jeffrey's LA-screening/restoration interests) and should
  collapse to a console-standoff far less often than generic news.
- **Still origination** -> still under the ceiling. Better fuel, same engine. Measure grade + sameness
  vs the news bank on a small N to decide whether tier 3 is worth it.

### Tier 3 -- `pd_adapt` -- the CEILING-BUSTER (real new LLM logic; the big payoff)
- The item is an actual public-domain **story TEXT**, not an RSS blurb. The interpreter ADAPTS it:
  ingest full text -> extract premise / characters / arc / **ending** -> CONDENSE into the beat ledger.
- **Why it fixes both problems:** the source hands over the arc + the ending, so the model stops
  *inventing* (sidesteps the origination ceiling -- the story is already A-grade) AND the beat planner
  no longer manufactures the structure (sidesteps the sameness -- the shape comes from the source). The
  model's job drops from "be a great screenwriter" to "condense a great story for audio."
- **New chain (this is the "real new logic"):** PD stories run 2k-10k words; gemma ctx is 8k, so it is
  NOT a single call:
  1. INGEST + segment the source text (chunk to fit context).
  2. EXTRACT structure -> a known-good arc: protagonist, want, antagonist/pressure, turn, climax,
     ending (map onto the existing `BEAT_ROLE` sequence -- the source supplies the irreversible-choice
     beat instead of the planner guessing).
  3. CONDENSE to ~N voiced beats at the target runtime (faithful, not embellished).
  4. DRAMATIZE for radio: prose narration -> announcer lines + character dialogue + "what we hear"
     (this is where the existing composer / `use_exchange` / voice bank take over -- unchanged downstream).
- **Acceptance angle:** grade should land HIGHER + steadier than tier 1/2 because the arc is given;
  the new risk is FIDELITY (does it preserve the source's actual story) -> add a "faithful to source"
  check, not just the dramatic-quality grade.

## Feeds (from the operator's research)

**Story SEED banks (tier 2, archive_seed):**
- LOC "Now See Hear!" -- `https://blogs.loc.gov/now-see-hear/feed/` (best single seed feed)
- NFPF Access Alley -- `https://www.filmpreservation.org/blog.atom` (restoration hooks)
- ACE (Assoc. des Cinematheques Europeennes) -- `https://ace-film.eu/feed` (European archive flavor)

**Story SOURCE banks (tier 3, pd_adapt -- TEXT, not news):** Project Gutenberg, Wikisource, archive.org
text collections, the LibriVox SOURCE texts. (Curate to short stories / fables / myths that fit a
~3-8 min radio runtime; Poe, Chekhov, O. Henry, Saki, Aesop, fairy tales, early Sherlock Holmes.)

**VISUAL ASSET bank (NOT a story feed -- injects at the image/video director for B-roll texture):**
- Internet Archive / Prelinger -- `https://archive.org/services/collection-rss.php?collection=prelinger`
- Gate the broad IA feeds (noisy: random uploads, dupes, bad metadata). Use ONLY for footage texture,
  queried when needed -- never as the story brain.

## Rights guard (commercial-clean discipline)

Every item carries a `rights_status` field; the interpreter NEVER claims a source is public-domain
unless the feed explicitly says so. Archive *news* describing a film is NOT a PD grant for that film;
PD *texts* (Gutenberg) are clean. Keep the asset-rights and the story-rights separate.

## Suggested experiment order

1. Tier 2 feed-swap (free): run 1-2 episodes off LOC/NFPF/ACE with the EXISTING logic; eyeball + grade
   vs news. Decide if richer seeds alone move the needle.
2. If not enough -> build the Tier 3 `pd_adapt` interpreter (the new chain above) as a parallel
   front-end feeding the same downstream; gate behind its own bank flag; add a source-fidelity check.
3. Wire the bank registry + on/off toggles last (once >=2 interpreters exist to toggle between).

## Scope note

Front-end only -- the entire back half (outline/compose/cast/voice/audio/video/OBS) is reused unchanged.
Independent of, and lower-risk than, the Increment-1 sprint; pairs naturally with the pitch room (seed
banks can still run pitch-room divergence; adapt banks skip invention and feed the real spine).
