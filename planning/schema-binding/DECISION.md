# Should the other local JSON passes bind their schema?

Written 2026-11-21, after PBUG-20261121-01: a Japanese Shakespeare episode on
the Mac died because the verbatim-translation reply was malformed JSON six times
in a row. Binding that pass's schema at token selection fixed it (2.3.62, live
on the Mac the same night: 12 entries translated in 3 calls, every batch first
try). This page asks whether the rest of the local JSON passes should get the
same treatment. **Decision: D, evidence-led -- approved by the operator
2026-11-21. Step 1 (re-wrapping forwarders) is built; no pass beyond the
four that bound before is bound.**

The first draft recommended C. An Opus contrarian refuted it with
measurements, and each claim was re-checked before this revision; the
corrections are marked.

## What binding is

A local writer slot exposes `_otr_bind_schema`. A pass that calls
`bind(Schema)` gets a generate function that runs lm-format-enforcer (LMFE) on
every token, so a reply that does not parse cannot be sampled. `structured_call`
never binds by itself; each pass opts in.

## Facts, measured

**Who binds today.** Of 34 `structured_call` sites, 33 are JSON passes and 1 is
a text pass. Four bind: verbatim translation, the My Story treatment, the source
rewrite in `_otr_story_source`, and `ledger_clean`'s scope authorization.

**Unbound passes that write episode-language text inside JSON** (the shape that
broke): SciFi Pitch, Treatment, cast_aliases and native_names; My Story
ActScript (dialogue) and StoryFrame (announcer); the character description;
`ledger_clean` f2_fix and line_repair; the episode title. More are unclear
(outline beat flesh-out, news_read, the title gloss, the brief interpreters).

**The wrappers drop the hook -- and more.** The SciFi lane's `_counting` (eight
call sites) and `_counting_slot_fn` (defined in four modules: news,
media_archive, public_domain, shakespeare -- the first draft said two) do not pass
`_otr_bind_schema` through. They also drop `_otr_supports_json_object`,
`_otr_openrouter` and `_otr_inspect_fit`, so on OpenRouter the SciFi and
brief-interpreter passes never get json_object mode today. Copying the bind
hook would also be wrong: `bind(schema)` returns the raw scheduler closure, so
the counter would stop counting. Forwarding must re-wrap.

**Coverage.** Every schema compiles in LMFE with the Gemma 4 tokenizer (262,144
tokens) and the Qwen 3.5 tokenizer (248,077). That proves less than it sounds:
the bound My Story treatment compiled fine and still killed a Japanese episode
on all three attempts (PBUG-20260929-05, an LMFE alphabet trap, since fixed).

**Cost (corrected).** Two parts, per generated token, measured on the desktop
CPU:

| Part | Gemma 4 vocab | Qwen 3.5 vocab |
|---|---|---|
| LMFE prefix function (mean) | 6-7.5 ms | 4-5 ms |
| transformers' mask step when the allowed set is the whole vocabulary (inside any free string) | ~15 ms | ~14 ms |
| **Total, inside strings** | **~20 ms** | **~19 ms** |

The first draft measured only the first row. Set against each writer's own time
per token, a bound pass runs:

| Writer | Machine | Model time per token | Bound pass |
|---|---|---|---|
| Gemma 4 12B native | 5080 | ~24 ms | estimated ~2x; **measured ~8%** (below) |
| Gemma 4 E2B native | 5080 | ~8 ms | estimated ~3-4x; not measured |
| Gemma 4 E2B native | 4060 | ~16 ms | estimated ~2x; not measured |
| Qwen 3.5 4B | Mac M6 | ~120 ms | ~10-15% slower |

**Measured on the 5080, 2026-11-21, and it overturns the NVIDIA estimate.** A
Japanese Shakespeare episode on 2.3.63 (`otr_16gb_low`, native Gemma 4 12B,
`assassin_corpse_ja_20261121_080500`, published in 10:47) ran its bound passes
at a median 35.5 tok/s against 38.7 for its unbound writer passes in the same
run -- about 8% slower, not 2x. ComfyUI's native adapter applies the allowed set
far more cheaply than transformers' `PrefixConstrainedLogitsProcessor`, which is
what the CPU estimate modelled. The E2B rows are still estimates and are likely
far lower too. This weakens the COST argument against binding on NVIDIA; it does
not touch the reason D stands -- binding trades loud, repairable failures for
silent truncation and buried punctuation unless each pass gets its own checks.

The Mac's own log fits the corrected figure: bound passes ran at a median 7.5
tok/s against 8.3 unbound, about 10% (the first draft wrongly called that 4%).
LMFE also keeps a state per generated token for the whole call (about 380 MiB
measured for a 4K prompt and 2K output), which matters on a 16 GB Mac.

**Binding changes how a bad reply fails.** Unbound, a reply that breaks a rule
fails validation loudly and the repair ladder fixes it. Bound, two kinds pass
silently instead:
- JSON punctuation inside a finished string (`]}`), which is why translation
  needed its own residue check (`58fe2a1f`).
- A string cut off at its `max_length`: LMFE forces the closing quote there and
  counts a `\uXXXX` escape as six characters. Today that includes `_Fixed.text`
  (2,000, a repaired dialogue line), `episode_title` (120) and `_Gloss.gloss`
  (40).

**Failures seen.** One unbound malformed-JSON death (translation, Qwen 4B,
Japanese -- now bound and fixed) against one bound death (the treatment,
PBUG-20260929-05 -- fixed). The unbound episode-language passes have produced
Hindi, Mandarin and Japanese episodes with no recorded death.

**A near miss, the night this page was written.** The overnight Mac run
(2.3.62, `otr_mac16_low`, Shakespeare in Hindi, prompt `a0148d26`):
`OTR_Outline.macro` -- unbound -- came back malformed on attempt 1 ("Expecting
property name enclosed in double quotes") and attempt 2 ("Unterminated
string"), and passed only on attempt 3 of 3, the typed repair at temperature
0.1. One more bad reply and the episode dies. Under D this pass is the first
candidate for step 2; check its schema's `max_length` fields before binding it,
since binding would turn an over-long field into silent truncation.

## The options

**A. Bind every JSON pass inside `structured_call`.** Costs 2-4x in the LLM
phase on NVIDIA, changes every English episode, and turns loud, repairable
failures into silent truncation across ~29 unchecked passes. Rejected.

**B. Bind the episode-language passes everywhere.** The same problems on a
narrower set, which still includes My Story's dialogue, most of that lane's
generation. Rejected.

**C. Bind by default on the transformers lane (the first draft's pick).** The
lane is chosen by provider, not machine, so it is not "Mac and AMD only": a
transformers writer picked on NVIDIA (the catalog records Qwen 3.5 4B episodes on
8 and 16 GB NVIDIA) would bind too. It binds ~29 passes at once with no residue
or length check on any of them, for ~10-15% on the Mac. Rejected.

**D. Evidence-led (recommended).**
1. Make every slot wrapper forward the bind hook by RE-WRAPPING (bind the inner
   slot, wrap the result again), so a pass that binds behind a wrapper is not a
   silent no-op and the counters keep counting. Behaviour-neutral today: no pass
   binds through them yet.
2. Bind a pass the first time a live run shows it failing, the same rule that
   admits a bug to the production log, and add that pass's own residue and
   max_length handling in the same change.
3. Treat the dropped json_object markers on OpenRouter as their own question:
   forwarding them changes cloud-lane behaviour, so it needs its own decision and
   a live cloud leg.

## Why D

The record is one death on each side, both fixed in their own pass. Binding is
not a free robustness upgrade: it costs 2-4x on NVIDIA, and it trades loud,
repairable failures for silent ones unless each pass gets its own checks. Doing
that per pass, when a pass earns it, keeps every change small and checked.

## If approved, done looks like

- Re-wrapping forwarders for `_counting` and the four `_counting_slot_fn`
  copies, with tests that fail if a wrapper drops the hook or a bound call skips
  the counter.
- No pass bound beyond the four that bind today.
- Suite green, Sonnet QA on the diff, pushed. No release needed on its own
  (behaviour-neutral); it rides the next one.
