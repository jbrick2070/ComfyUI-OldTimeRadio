# Media Archive RSS Integration Plan

Date: 2026-07-07
Status: scoped from the live repo, hardened by one live Antigravity/Gemini
review pass, implementation in progress.

## Goal

Make `source_bank=media_archive` a real RSS-backed story lane that creates
fictional old-time-radio stories from media-history, preservation, restoration,
lost-media, and archive-culture feed items.

Core architecture rule:

```text
media_archive =
  existing science legacy_many_pass spine
  + media archive RSS fetcher
  + media archive interpreter/source brain
  + optimistic archive mystery/adventure prompt profile
```

Do not fork the architecture unless the existing spine proves impossible to
reuse. The media lane should stay as close to the current sci-fi lane as
practical; the swapped parts are the source brain, sources, story rules,
curated theme pool, and prompt pack.

Tone target: optimistic curious radio adventures about discovering, restoring,
researching, or preserving media history. Think film archives, forgotten
recordings, lost broadcasts, restoration projects, librarians, historians,
projectionists, collectors, and archivists. The inspiration set is National
Treasure / Nancy Drew / archive documentaries / public broadcasting mysteries:
curious preservation mysteries, not crime thrillers.

Keep the other lanes untouched:

- `science_news` remains byte-identical unless a test deliberately proves a
  no-op refactor.
- `public_domain_story` stays non-runnable and out of scope.
- `custom_source_bank` stays non-runnable and out of scope.
- The canonical workflow keeps its current defaults until a dedicated media
  smoke changes the real JSON on purpose.

## Grounding

Current production state:

- `nodes/story_packs/banks.json` registers `media_archive`, but it has empty
  `fetcher` and `interpreter` ids and `runnable:false`.
- `nodes/story_packs/media_archive/media_restoration_adventure.json` only
  authors `line_composer_system` and `coda_system`.
- `nodes/_otr_source_payload.py` already makes bank `fetcher` and `interpreter`
  ids live routing coordinates behind a fail-loud exact contract.
- The real workflow already has the source-bank widget wired. Node 1 widget
  slot 23 is `science_news`; slot 24 is `sci_fi_radio`.
- `nodes/visual_styles/archival_documentary.json` is already authored and can
  pair with a media archive run later.
- `nodes/news_interpreter.py` remains science/news-specific. Its
  `FORBIDDEN_ERA_TERMS` validator is not a blocker because the media lane gets
  its own interpreter and must not route through `news_interpreter.py`.

Current blockers before `media_archive` can become runnable:

1. No media archive RSS fetcher is registered.
2. No media archive interpreter/source brain is registered.
3. No `nodes/story_rules/media_archive.json` exists.
4. The media archive story pack lacks the active production prompts now routed
   through packs: `outline_macro_system`, `outline_phase_system`,
   `outline_beat_system`, `exchange_system`, `announcer_intro_system`,
   `announcer_intro_safe_system`, and `announcer_outro_system`.
5. The media archive close still has to satisfy downstream compatibility that
   reads `meta["news"]["news_close_brief"]`; that key can remain as a mirror
   while conceptually meaning "archive close/source note".
6. The style/theme selector currently draws from the broad story-style catalog.
   The media lane needs a curated non-sci-fi subset so archive runs do not
   randomly become space, crime, horror, or emergency stories.

## Review Gate

User requested a real second opinion before coding.

- `kibitz` r1 with Antigravity/Gemini completed successfully under
  `kibitz-runs/2026-07-06-media-archive-rss-lane-agy/r1/`.
- The Claude Code reviewer lane was unavailable: out of extra usage until
  2026-07-10 06:00 America/Los_Angeles.
- Antigravity follow-up r2 hit quota/backend exhaustion after the successful
  r1 pass. The successful r1 review is the accepted second-opinion gate.

Accepted review findings:

- Use `feedparser`; it is already in `requirements.txt` and production already
  uses it for feeds.
- The media interpreter result must include `attempts:int` because
  `_otr_source_payload.validate_interpreter_result` requires it.
- Curated non-sci-fi theme selection needs an explicit implementation hook in
  `_otr_style_catalog.py`.

Rejected review finding:

- Do not modify `news_interpreter.py` era-term validation for media archive.
  That is science-lane behavior. The media lane uses a separate interpreter.

## Source Feeds

Use story-seed feeds, not broad asset feeds:

- LOC Now See Hear: `https://blogs.loc.gov/now-see-hear/feed/`
- NFPF Access Alley: `https://www.filmpreservation.org/blog.atom`
- ACE: `https://ace-film.eu/feed`

Probe results from this machine on 2026-07-07:

- LOC returned HTTP 200 RSS.
- NFPF returned HTTP 200 Atom.
- ACE returned HTTP 403 to PowerShell, even with a browser-like user agent.
  Treat ACE as optional or fixture-only until a reliable access path is proven.

Do not use Internet Archive / Prelinger as the first story brain. The old
source-bank design correctly treats broad IA/Prelinger feeds as visual texture
or asset search material, not the source of the story premise.

## Target Shape

The media archive lane should use the existing `legacy_many_pass` adapter
surface:

```text
RSS item
-> normalized source payload
   {headline, summary, full_text, source, date, link, seed_text}
-> media archive interpreter
-> briefs-like object
   casting_brief, script_brief, news_close_brief, key_terms, attempts
-> existing writer stages
```

The payload key names stay as-is because `_otr_source_payload.py` deliberately
owns an exact article-adapter contract. Media-specific provenance can be
included in the text values and later stamped only where a consumer exists.

## Build Slices

### Slice A: Feed Normalizer, No Routing Change

Add `nodes/_otr_media_archive_sources.py`.

Responsibilities:

- Fetch one or more configured feeds.
- Parse RSS 2.0 and Atom with `feedparser`.
- Normalize each item to the exact source payload keys.
- Compute a deterministic source hash from title, link, date, and body.
- Filter empty or unusable items fail-loud.
- Prefer LOC and NFPF by default; keep ACE disabled or fixture-only until its
  403 is solved.
- Provide a fixture loader path for tests so unit tests do not require network.

Tests:

- RSS fixture parses.
- Atom fixture parses.
- Missing title/link/body fails.
- Payload validates through `validate_source_payload`.
- The module never imports the writer or `news_interpreter`.

### Slice B: Media Archive Interpreter, Still Non-Runnable

Add `nodes/_otr_media_archive_interpreter.py`.

Responsibilities:

- Use the technical LLM function to turn one normalized archive payload into:
  `casting_brief`, `script_brief`, `news_close_brief`, `key_terms`, and
  `attempts`.
- Keep the prompt explicitly archive/media-history oriented:
  preservation, damaged media, provenance, access, labels, missing context,
  recovery, broadcast history, and human meaning.
- Forbid sci-fi/news drift in the instruction:
  no science-fiction anthology, spaceship, mission control, lab containment,
  generic experiment emergency, or real-news-report close.
- Make clear the story is fictional but inspired by the archive/media item.
- Keep `news_close_brief` as the compatibility key, but write an archive source
  note into it.

Register in `nodes/_otr_source_payload.py`:

- `media_archive_rss` fetcher id.
- `media_archive_interpreter` interpreter id.

Update `nodes/story_packs/banks.json` for `media_archive`:

- `fetcher: "media_archive_rss"`
- `interpreter: "media_archive_interpreter"`
- leave `runnable:false`.

Tests:

- `media_archive` resolves both ids while still not runnable.
- `media_archive` never calls `_fetch_rss_seed_or_die`.
- The interpreter result passes `validate_interpreter_result`.
- `NewsInterpreterError` handling for `science_news` remains unchanged.
- Science wrapper forwarding tests remain byte-identical.

### Slice C: Story Rules Pack

Add `nodes/story_rules/media_archive.json`.

Keep it conservative. Start from the science schema shape, not the science
content:

- Keep useful general bans like on-the-nose emotion and thesis statements.
- Remove science/lab/mission-specific assumptions.
- Add media-archive drift bans:
  "spaceship", "mission control", "laboratory containment", "alien signal",
  "generic experiment emergency", "ancient secrets", "haunting", "body count",
  and violent thriller shortcuts.
- Add replacements for archive-specific cliches only when they are real.

Tests:

- `resolve_story_rules("media_archive")` loads.
- Missing rules still fails for non-authored runnable banks.
- Runnable bank sweep catches a missing rules pack.

### Slice D: Curated Non-Sci-Fi Theme Pool

Update `nodes/_otr_style_catalog.py`.

Responsibilities:

- Add a curated media archive style pool of roughly 20-30 existing style slugs.
- Keep the style contract object and the existing style grammar unchanged.
- When `meta["source_bank"] == "media_archive"`, `select_style` draws only from
  that curated pool.
- Keep all other banks byte-identical by using the existing emergency vs
  non-emergency pool logic.

The initial pool should favor archive-adventure, public-broadcast, media,
family-memory, theater, museum, lost-recording, and preservation-adjacent
styles. Exclude obvious sci-fi, space, lab, crime-thriller, murder, ransom,
weapons, horror, plague, and disaster styles.

Tests:

- Media archive style selection always returns a slug from the curated pool.
- Science/default style selection remains byte-identical for pinned seeds.
- The curated pool slugs all exist in `STYLE_CATALOG`.

### Slice E: Complete The Active Pack Prompts

Expand `nodes/story_packs/media_archive/media_restoration_adventure.json` to
the active production prompt set:

- `outline_macro_system`
- `outline_phase_system`
- `outline_beat_system`
- `line_composer_system`
- `exchange_system`
- `coda_system`
- `announcer_intro_system`
- `announcer_intro_safe_system`
- `announcer_outro_system`

Do not paste the old lab-only keys like `outline_system`, `pitch_room_system`,
or `title_system` into production until those consumers exist.

Update `banks.json` `required_seams` for `media_archive` to match the active
production prompts that must exist before runnable.

Tests:

- Exact media pack key set.
- Prompt leakage test: media archive pack contains no forbidden sci-fi/news
  terms except inside `forbidden_leakage_terms`.
- `resolve_creative_system_prompt(..., source_bank_id="media_archive")`
  resolves outline, exchange, coda, and announcer prompts.
- Science pack extracted prompt byte-identity tests still pass.

### Slice F: Registry Wiring And Runnable Flip

Only after slices A-E pass:

- Update `nodes/story_packs/banks.json` for `media_archive`:
  - `fetcher: "media_archive_rss"`
  - `interpreter: "media_archive_interpreter"`
  - `runnable:true`
- Set `media_archive.runnable:true`.
- Keep the canonical workflow defaults as `science_news` / `sci_fi_radio`
  unless the purpose of the commit is a media-archive smoke.
- This makes the lane runnable during development once its required registry
  contracts exist. Downstream story quality can still be rough; tests and
  release gating determine production readiness.
- Run targeted tests for:
  source payload, story routing, story rules, outline seams, exchange seam,
  source-bank widget, style selection, and pack leakage.
- Then run full regression + Bug Bible before commit/push.

Do not weaken the registry to force `runnable:true`.

### Slice G: Fixture Tests, Then Live RSS Tests

Fixture tests come first:

- Fetcher RSS fixture.
- Fetcher Atom fixture.
- Interpreter deterministic JSON fixture.
- Media story rules load.
- Media theme pool.
- Active prompt seam resolution.
- `media_archive` no-fallback guards for science RSS and news interpreter.

Only after fixture tests pass:

- Run a live LOC/NFPF RSS test.
- Keep ACE optional while it returns 403 from this machine.

### Slice H: Real Workflow Media Smoke

For a headless/API media run, obey the workflow source-of-truth rule:

- Load `workflows/otr_scifi_16gb_full.json`.
- If the smoke is meant to run media by default, change the real JSON in the
  same commit:
  - node 1 widget slot 23: `media_archive`
  - node 1 widget slot 24: `archival_documentary`
- Re-run `OTR_WorkflowValidator`, JSON round-trip, link/widget audit, and
  widget-count audit.
- Reset ComfyUI selectively before the render.
- Confirm final assets land under `otr/episodes/<ep>/` and `otr/obs/`.

If the production default should remain science, do not permanently change
those widget values. Use the existing UI/widget path for interactive media
episodes, or make a deliberately scoped smoke commit and revert it after the
proof run.

## Non-Goals

- Do not build `public_domain_story` in this slice.
- Do not build the simple 4-prompt runner.
- Do not add a new workflow node.
- Do not add new writer widgets.
- Do not use a hidden fallback to science RSS.
- Do not route media archive through `news_interpreter.py`.
- Do not use broad Internet Archive asset RSS as the story source.
- Do not implement public-domain source provenance fields in this slice.
- Do not add source provenance fields until a concrete consumer needs them.

## Acceptance

The first real media archive implementation is acceptable when:

- `source_bank=media_archive` fetches a LOC/NFPF fixture or live item through
  its own fetcher.
- Its interpreter produces archive-specific briefs under the existing
  compatibility surface.
- Its pack owns every active production prompt the lane reaches.
- Its story rules pack loads.
- A selected media archive run cannot call science RSS or the science news
  interpreter.
- Science tests stay byte-identical.
- The lane becomes runnable when the required fetcher and interpreter IDs are
  registered and the registry can load without weakening any gate.
