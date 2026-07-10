# Sci-Fi (science_news) + Media Archive story lanes -- Story-Quality QA

- Date: 2026-07-10
- Status: ANALYSIS ONLY -- no code changed. This doc is the input for a later coding sprint.
- Scope: the two "media pack" story lanes (LLM prompts + Python pipeline) judged for STORY QUALITY:
  will they produce compelling, varied radio dramas on the production writer (Mistral-Nemo class 12B)?
- Method: Claude anchor review (grounded, file-level) + two Fable subagent fan-out reviews (one per
  lane) + judge pass grounding every claim against the real Windows files. Claims that did not
  survive grounding are listed in section 7. Every file:line below was verified 2026-07-10.

## 1. Verdict

Both lanes are mechanically excellent and creatively mid-ceiling. The plumbing -- fail-loud
registry, pack-routed seams, structured-call ladders, per-bank regex QA, reroll ladders -- is near
best-in-class for a local-model pipeline. The story gap is ARCHITECTURAL and shared: all the real
conflict engineering runs downstream of the outline, the one divergence pass (pitch room) ships
dark, and several authored craft surfaces are dormant or contradicted by shared science-flavored
rider text. The media_archive lane additionally underpowers the two places a listener actually
lives: per-line dialogue craft (its line seam dropped the science lane's craft block) and
episode-to-episode variety (a payload-free 15-title seed deck with ~3 effective clusters).

Single highest-leverage change, both lanes: light the pitch room (F1) and put the conflict spine
upstream of the outline (F2). Single cheapest big win, media lane: port the craft block into its
line_composer_system seam (F5) -- pure pack-JSON edit.

## 2. Production reality (what actually runs -- verified)

Canonical workflow `workflows/otr_canonical.json`, node 1 (OTR_LedgerScriptWriter), 28 widgets,
all converted to inputs (positional map verified by probe):

| Widget | Value | Meaning |
|---|---|---|
| [03]/[04] creative/technical model | Mistral-Nemo-Instruct-2407 | judge prompts for 12B class |
| [13] lemmy_cameo | roll (~11%) | |
| [14] use_exchange | **True** | grouped-exchange pre-pass IS live in production |
| [15] enable_production_stage3_validators | **True** | stage-3 validators IS live in production |
| [16] news_briefs_required | True | interpreter failure halts (no legacy drift) |
| [23] source_bank | science_news | media_archive selectable via widget |
| [24] visual_style | sci_fi_radio | |

Levers (code defaults):

- Style grammar ON by default (`nodes/_otr_config.py:61` STYLE_GRAMMAR_DEFAULT=True). Writer gate
  `OTR_LedgerScriptWriter.py:3922` fires on `story_quality_l12_enabled() or _style_grammar_on`,
  so conflict_object / beat_role / ending_mode flow on default renders.
- Pitch room OFF by default (`nodes/_otr_pitch_room.py:70-76`, env `OTR_ENABLE_PITCH_ROOM`,
  no workflow widget exists).
- All 10 creative seams pack-route through `nodes/_otr_creative_prompt_router.py` (QA F1
  2026-07-09). Period overlay PREPENDS to outline stage seams (`_otr_outline.py:1863-1866`),
  it does not replace them.

Note the two stale-comment traps for later coders: `OTR_LedgerScriptWriter.py:4899-4901` and
`_otr_line_composer.py:~1447` claim conflict fields are "Empty unless OTR_STORY_QUALITY_L12 is
on" -- false since 2026-06-24 (style grammar bundles the L1/L2 path ON).

## 3. Shared architecture findings (both lanes)

### F1 (P1) -- The primary story-architecture lever ships dark
`nodes/_otr_pitch_room.py` module doc: "THE primary story-architecture lever". It generates 3
forcibly-divergent premises, bans the "console standoff" cliche by name (:187-190), and
greenlights on "a clear human want, a conflict that plays ON-STAGE ... LOWEST 'console standoff'
risk" (:225-229). Default OFF; env-only; never runs in production. With it dark, the entire
premise instruction is one line of the macro seam ("one sentence that extrapolates dramatically").
Fix: flip default ON or add a workflow widget (widget change goes in otr_canonical.json IN THE
SAME CHANGE, appended at END of widgets -- positional law). Consider widening `_GENRES` (only 4
at :155). Fail-soft already built ("never breaks the writer"). Cost: +2 LLM calls/episode.
Goldens that pin script_brief-derived output will change.

### F2 (P1) -- Conflict spine is derived AFTER the skeleton it should shape
`generate_outline` runs at `OTR_LedgerScriptWriter.py:3756`/`:3814`; `derive_news_dramatic_state`
(opposed wants, dramatic question) runs at `:4196`/`:4204` -- after. The outline plans beats from
a <=350-char brief with no stated conflict. Worse: the macro seam asks the LLM for
`central_tension` ("the single dramatic question the episode answers") and the value is never
rendered into ANY downstream prompt -- consumers are arc-ref metadata only
(`_otr_outline.py:1556`, `:1714-1723`; zero hits in writer/composer prompt builders).
Fix: derive dramatic state BEFORE the outline; render "Central conflict: A wants X; B wants Y"
+ "Dramatic question:" into the phase and beat user prompts; render central_tension into the beat
prompt or stop asking a 12B to write a field nobody reads.

### F3 (P1) -- Shared composer riders speak science to every lane and license banned vocab
`nodes/_otr_line_composer.py` `_build_user_prompt`, else-branches (fire on every beat WITHOUT a
conflict_object, in all lanes):
- :1342-1345: 'Generic roles ("the tech", "the lab", "mission control") are fine.' -- media
  story_rules BAN "mission control" (banned_phrases) and the media macro seam forbids it. The
  prompt primes exactly what QA rerolls, and the reroll re-presents the same license.
- :1631-1636: "Ground this line in the news facts and this scene's premise" -- the media pack's
  forbidden_leakage_terms lists "news facts" (dormant field, but authored intent). The fix
  plumbing already exists unconsumed: `banks.json` media defaults carry
  `source_grounding_label: "archive material"`.
Additionally science-side: the science pack line_composer_system seam ITSELF carries the generic-
roles license, while the KILL-1 rider (conflict_object present -- most beats, style grammar ON)
forbids it -- a live system-vs-user contradiction a 12B resolves unpredictably.
Fix: thread bank defaults (source_grounding_label + a per-lane generic-role example list) into
both rider branches with science-default fallback (science must stay byte-identical); soften the
science seam license to "prefer NAMED ENTITIES; generic role only when nothing fits"; fix the two
stale comments (section 2).

### F4 (P2) -- Escalate-vs-resolve contradiction on resolution beats
Beat user prompt appends unconditionally (`_otr_outline.py:1341-1349`): "RAISE THE STAKE: this
beat's pressure must be higher than the previous beat's -- escalate, never tread water." The
resolution arc-phase focus says (`_otr_episode_budget.py:180`): "Close out the arc. Show
consequence. Do not introduce new conflict." Contradictory orders on the same beat; also blocks
the style catalog's quiet/unresolved ending_modes from landing.
Fix: condition the clause on arc phase -- resolution gets "RELEASE the pressure: show the cost
and what changed; do not introduce new conflict."

### F10 (P3) -- Dormant pack fields (the authored-but-unrouted pattern)
`examples`, `tone_guardrails`, `forbidden_plot_patterns`, `forbidden_leakage_terms` are parsed
into StoryPack (`_otr_story_pack.py:85-87`, `:158-159`) and consumed by NOTHING. Few-shot
rendering infra is deliberately pinned at 0 production callers (router doc, D2c). The media pack
authored real content into all four -- including the lane's single best story example ("A brittle
transcription disc forces ... choose between a public anniversary broadcast and patient
restoration") -- all invisible to every model call. Science pack carries them empty.
Fix (pick one, per no-dead-config law): wire them (example + tone_guardrails into the macro USER
prompt; forbidden_leakage_terms into per-lane QA regexes) or delete the fields and content.

### F11 (P3) -- required_seams weaker than production
Canonical workflow runs use_exchange=True, but neither science_news nor media_archive lists
`exchange_system` in banks.json required_seams (public_domain_story and shakespeare do). Runtime
get_pack_prompt still fails loud, but the registry sweep would not catch seam deletion pre-run.
Fix: add `exchange_system` to both banks' required_seams.

### F13 (P3) -- Coda example template monoculture (both packs)
Each pack's coda seam ships 3 examples in one syntactic shape ("Beyond tonight's X:" / "Past
tonight's Y:" / ...). 3 same-shaped examples reliably template-collapse a 12B into one formula
every episode. Fix: vary one example's structure per pack.

## 4. Science lane (science_news) findings

### F8 (P2) -- QA regex deck misses the actual 12B sci-fi register; some replacements de-dramatize
`nodes/story_rules/science_news.json`. Missing high-frequency offenders: "playing God", "what
have we done", "God help us (all)", "point of no return", "on the brink", bare "it's too late"
(only "before it's too late" is caught), "for the (greater) good of humanity/mankind", and the
number-one small-model tic class, as-you-know exposition ("as you (well) know", "let me explain").
On-the-nose alternation omits "frightened" -- the period-correct synonym a 1940s-register model
actually emits. banned_phrases bans bare "absolutely" (kills the legitimate period line
"Absolutely not."). Replacements like "leaving nothing to chance" -> "checking every detail" and
"this changes everything" -> "this changes our plan" trade cliche for bureaucratic flatness and
become house tics themselves (same canned substitute every time).
Fix: add the missing patterns as cliche_patterns (reroll class), NOT banned_phrases (hard ban);
add "frightened"; scope "absolutely"; review the two flattest replacements toward
reroll-not-replace or higher-voltage substitutes.

### F12 (P3) -- Style picker is article-blind; header over-promises
`nodes/_otr_style_catalog.py` header (:13-16): "The picker SELECTS the best-fit style for the
article". Implementation (`select_style` :746+): deterministic seeded pick from the 90
non-emergency styles; the ONLY article awareness is a disaster-keyword gate. A coral-reef article
can draw train_car_murder_mystery; fusing an unrelated style grammar with a 350-char brief on a
12B is an uncontrolled coin flip (delightful or incoherent).
Fix: score styles by tag/keyword match against key_terms+premise, take top ~12, seeded pick
within the subset (fall back to full pool on zero scores). Deterministic, no LLM, unit-testable.

### F14 (P3) -- Unsatisfiable/misparse-prone instructions
- Beat target_words allows up to 80 (`_otr_outline.py:107-112` le=80) while the unconditional
  composer rider demands "one breath, concrete, no nested clauses" -- a 60-80-word one-breath
  line is not writable; the model must violate one instruction.
- Science pack line seam: "Short and charged beats long and explanatory." -- at 12B "beats" reads
  as the noun. Rewrite: "Prefer short and charged over long and explanatory."

### F15a (P3) -- Interpreter briefs: journalistic, not dramatic
`nodes/news_interpreter.py`: script_brief cap 350 chars (:91), casting_brief 200 (:90) -- arc +
central tension + beat hooks in ~55 words forces generic compression. The prompt requests stakes
but never want/obstacle/antagonist/who-pays. Validators V0-V3 check GROUNDING (terms in source,
era leaks) -- none checks dramatic adequacy, so a limp brief passes.
Fix: add "WHO wants WHAT / WHAT stands in the way" to the script_brief spec; consider a cheap
dramatic-adequacy check; keep grounding validators as-is (they are excellent).

Variety note: science has NO seed-deck equivalent; with the pitch room dark, premise variety
rides on the article + the article-blind style pick + arc-shape/casting RNG. The KILL-1 comments
document the observed result ("mission control / console sameness"). F1+F12 are the fix.

## 5. Media archive lane findings

### F5 (P2, cheapest big win) -- Line seam dropped the craft block; nothing restores it
`media_restoration_adventure.json` line_composer_system is one prose paragraph. Gone vs the
science seam: register matching ("Match the speaker's stated speech register (the 'speaks:'
note) exactly ... never blur two characters into the same voice"), "Imply more than you state",
"Inhabit the mood without naming it", the +/-30% word band. Cast cards still print `speaks:`
clauses -- no instruction tells the model to use them, so voices blur into one archivist voice;
with mood-restraint gone while the tail prints "Mood: worried.", the model names the mood and the
lane's own on_the_nose_patterns then burn reroll attempts on a failure the missing craft line
would have prevented. The shared tail restores output MECHANICS only (single line, no stage
directions, word target).
Fix: append an archive-flavored CRAFT block to the seam (register matching, imply-more,
mood-without-naming, pressure-through-the-object-in-hand). Pure pack JSON edit, near-zero risk.

### F6 (P2) -- Seed deck is a variety mechanism that does not vary anything
`drama_seeds.json`: 15 bare noun-phrase titles; ~7 are the same story ("a media item was
missing": Lost Reel / Hidden Cut / Unseen Frame / Forgotten Broadcast / Returned Tape / Buried
Interview / Clue in the Film) -- effective spread ~3 clusters. The interpreter injects exactly
one line ("Dramatic seed lens: {seed}"); a 12B cannot unfold "The Locked Archive" into a want,
an obstacle, or an ending. Deterministic pick per source (`select_drama_seed`) is the RIGHT call
(reproducibility) -- but it is also the lane's only variance mechanism, at interpreter temp 0.45.
Fix: seed deck v2 -- payload per seed: {title, want, obstacle, final_choice}; schema bump to
media_archive_drama_seeds_v2; render all fields in the lens block; keep deterministic selection.
Content pass at the same time: "The Vanishing Witness" primes the crime framing the prompts then
forbid; "The China Girl Mystery" rides niche film-lab jargon a 12B may literalize into a
nationality-based character (story-quality + sensitivity risk) -- retitle or carry a payload
that defuses it.

### F7 (P2) -- No one pays for the ending
Macro seam mandates resolution "through human care, interpretation, courage, or generosity" --
no seam requires the choice to COST anything, and the interpreter never asks briefs for a price.
A 12B under an optimism mandate with no cost instruction writes win-win endings: sweet but
weightless. The pack's own dormant example models the correct priced-dilemma shape.
Fix: one macro rule: "The final choice must cost the chooser something concrete they wanted --
time, credit, a keepsake, a first screening -- given up gladly. Optimistic endings are paid for,
never free." (personal_cost_patterns already police the canned phrasings.)

### F9 (P2) -- Rules collide with the lane's own domain vocabulary
`story_rules/media_archive.json` banned_phrases: "ghost", "phantom", "specter" -- but "ghost
image", "ghosting", "phantom signal" are period-correct film/broadcast restoration terminology,
exactly this lane's register; legitimate technical dialogue gets flagged and rerolled. Same class,
milder: "emergency broadcast" in lost-broadcast stories.
Fix: scope the patterns to supernatural usage (e.g. negative lookahead for image/signal/print),
or move them to cliche_patterns with replacements.

### F15b (P3) -- Interpreter: curatorial mush risk + budget squeeze
`_otr_media_archive_interpreter.py`: script_brief spec asks only for a "fictional radio story
premise" (no want/obstacle/turn); casting_brief "likely human roles and voices" invites the same
archivist/librarian/projectionist trio with no friction; base_temperature=0.45 (right for JSON
validity, wrong as the only variety engine over tame curatorial feeds); max_new_tokens=520
squeezes the ~2600-char combined field budget.
Fix: spine requirement in the spec ("WHO wants WHAT around the archive object, WHAT stands in
the way, and the final choice the episode forces"); one point of friendly friction in
casting_brief; temp 0.45 -> 0.6; tokens 520 -> 640. The structured-call ladder absorbs the retry
cost.

### F16 (P3) -- Minor seam hygiene
- outline_phase_system carries "Favor archive work ... over danger" guidance, but its output
  schema is a speaker list only -- dead tokens; the guidance already lives in the beat seam.
- Line seam paragraph repeats "archive object" / "missing context" / "non-violent"; "archive
  object" risks leaking into spoken dialogue as curator-speak (no regex guards it).

## 6. Prioritized coding plan (for the later sprint)

P1 -- architecture (biggest story-quality lift, both lanes):
1. F1 pitch room: default ON or widget (+ otr_canonical.json in same change; append-at-END law).
2. F2 conflict spine: dramatic state before outline; thread conflict + central_tension into
   phase/beat prompts.
3. F3 rider parameterization: bank-threaded grounding label + role examples; science seam license
   softened; stale comments fixed. Science lane MUST stay byte-identical under defaults -- the
   test_story_pack_stage1 byte-identity pins and PD1 contract gate this; re-pin deliberately.

P2 -- lane craft:
4. F5 media craft block (pack JSON only -- do this first, it is free).
5. F6 seed deck v2 + content pass. 6. F7 priced endings rule. 7. F8 science QA retune.
8. F4 escalation conditioning. 9. F9 jargon-collision scoping.

P3 -- hygiene/polish: 10. F10 dormant fields (wire or delete). 11. F11 required_seams +
exchange_system. 12. F12 article-aware style subset. 13. F13 coda example variety.
14. F14 misparse/limit fixes. 15. F15 interpreter spine specs (both lanes). 16. F16 seam hygiene.

Every chunk: regression suite + Bug Bible after the change; pack edits re-validate via
OTR_WorkflowValidator + JSON round-trip; commit AND push to v2.0-alpha per green chunk.

## 7. Claims discarded or corrected during grounding (audit trail)

- "The exchange_system seam never runs in production (use_exchange OFF default)" -- WRONG. Code
  default is False, but the canonical workflow sets use_exchange=True (widget [14]); the
  grouped-exchange pre-pass IS live. Surviving remnant: the required_seams gap (F11).
- "Stage-3 validators are dark" -- WRONG for production: canonical workflow sets
  enable_production_stage3_validators=True (widget [15]).
- Fable line-number citations spot-checked; all substantive quotes verified verbatim against the
  files listed above.

## 8. Provenance

Anchor review: Claude (Cowork), direct file reads. Fan-out: two Fable subagents (science lane;
media_archive lane), read-only, Windows paths. Judge: Claude -- every surviving claim re-verified
against the repo at HEAD on 2026-07-10; canonical-workflow widget map extracted by a temp probe
script (deleted after run). No code, pack, rules, or workflow files were modified.
