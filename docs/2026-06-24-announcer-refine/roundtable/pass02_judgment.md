# R2 JUDGMENT -- implementability (Claude, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.0953. Convergence:
HIGH -- all three demanded concrete signatures + caught the same data gaps; two
caught real Python/validator bugs.

## ACCEPTED (CONFIRMED vs code) -> folded into pass02
- Exact dataclasses + backward-compatible defaulted keyword-only signatures for
  StoryContract / SafeOpenBrief / build_story_contract / compose_announcer_intro
  / compose_announcer_outro (all 3). pass02 §0/§2/§3.
- Negative-slice bug in the KILL-4 truncation (Gemini#2): `s[:-5]` does NOT clamp
  -> `max(0, reserve)`. Plus the exact reserve formula (GPT#14, DeepSeek#6). §4.
- Double-lead-in stutter (Gemini#4): generate body without a lead-in ("start with
  the facts, no intro phrase"), PREPEND the deterministic lead-in post-generation,
  validator asserts no lead-in in the body (GPT#8, DeepSeek#8). §3.
- Separate `validate_news_coda_line` with a word band (existing validator is
  char-based) + lead-in/fact checks (GPT#6, DeepSeek#3). §3.
- Token-overlap gate must strip stopwords + cast names + setting/time, require
  >=2 content tokens len>=4 (GPT#5, Gemini#3, DeepSeek#9); REUSE the KILL-1
  helpers. §2.
- meta.story_contract as a plain dict, not the frozen dataclass (GPT#13,
  DeepSeek#4). §0.
- OutlineRequest fields named `story_engine`/`ending_mode`, distinct from `style`
  (GPT#12, DeepSeek#5). §0.
- Split into per-area commits (GPT#7). §6. Byte-identity asdict/snapshot caveat
  (GPT#8). §7.

## GROUNDING RESOLVED THIS PASS (panel said "verify"; I verified)
- `select_style` has EXACTLY ONE caller (:3224) -> safe to move/delete (resolves
  Gemini CUT + DeepSeek SHOULD#1). 
- `_build_macro/phase/beat_user_prompt` all exist (:1133/:1187/:1236); phase/beat
  take `macro` -> threading caveat noted (DeepSeek ASSUMPTION).
- Ledger lines DO carry `beat_id` (`_otr_ledger.py:96`) -> climax-line lookup
  feasible (resolves GPT#10/Gemini#1/DeepSeek#10).
- KILL-4 role constants exist (l12:55-72) -> real-constant map (GPT#15).
- cast_seed (:2878) + script_brief (:2785) precede OutlineRequest (:3032) ->
  pre-outline contract feasible (resolves the cyclic-dependency worry).

## JUDGE CALLS (where the panel split)
- `opening_status_quo` source: DeepSeek#1 wanted a NEW outline LLM field; GPT#3
  offered "first setup beat intent OR premise stripped". DECISION: derive
  deterministically from the FIRST character (setup) beat intent -- no new LLM
  field, outcome-free by construction, smaller surface. (Rejects the LLM-field
  primary; an LLM "opening situation" field could itself spoil.)
- Gemini MUST#1 "cyclic dependency": PARTIALLY a misread -- `select_style`'s
  cast_seed hash does NOT need the premise; only `premise_wants_emergency` reads
  text. So feeding `script_brief or news_seed` pre-outline works (no cycle). Folded
  as the documented contained change, not a blocker. CONFIRMED vs :718-733.
- Spoiler BELT: DeepSeek CUT#2 (drop it; rely on starvation) vs GPT (keep,
  precise). DECISION: input-starvation is the deterministic GUARANTEE; keep the
  belt as a CHEAP reuse-the-KILL-1-helpers extra, explicitly the deferrable piece
  if build pressure. Honors both.
- "F2" jargon (DeepSeek#2): removed; replaced with the run()/outline path.

## DEFERRED / CUT (accepted)
- sound_world -> visualizer/LTX routing (GPT CUT#1; call sites unknown) -> first
  build = outline injection + metadata only.
- lead-in variant set -> start with ONE fixed lead-in (GPT CUT#2, DeepSeek).
- premise-specific conflict objects -> already deferred (DeepSeek CUT#1).

## CONVERGENCE CALL
R2 hardened the build to concrete signatures + fixed two real bugs; no
unbuildable step remains. Material wiring/sequencing risks remain (prompt-builder
threading, byte-identity snapshots, the in-run() ordering) -> proceed to R3
(wiring / integration / sequencing).
