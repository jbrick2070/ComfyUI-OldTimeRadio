# ROUND 3 -- WIRING / INTEGRATION / SEQUENCING

You are reviewing THREE ALREADY-HARDENED, ALREADY-REVIEWED plans.
r1 (arc) and r2 (coding plan) are ON RECORD and CLOSED. **Do not re-derive the
designs and do not re-open story/prose quality -- it is closed by operator
directive.** Your ONLY job this round is WIRING, INTEGRATION and SEQUENCING
against the CURRENT code in this repo.

Read the driver anchor first. It records what the driver already verified --
89 of 89 line anchors confirmed, plus behavioural probes. Do not spend your
review re-confirming those. Attack the SEAMS: where a step's described change
meets code that does not match its description, where two steps collide, and
where the prescribed ship order creates a hazard.

Ground every claim in `file:line` from the REAL repo. A claim I cannot check
is discarded, not weighed.

---

# Driver anchor -- continuity-correctness r3 (wiring) + r4 (convergence)

**Driver:** Claude (Opus 5), Cowork, sole judge.
**Date:** 2026-08-05. **Repo:** ComfyUI-OldTimeRadio @ `v2.0-alpha`, HEAD `889448ac`.
**Panel:** Codex `gpt-5.6-sol` high + Antigravity `Gemini 3.6 Flash (High)`.
No `claude -p` lane -- the driver's own family is excluded.
**Profile:** `.kibitz/comfyui.local.md`.

**What is under review:** the three hardened plans in
`kibitz-runs/2026-08-04-continuity-ultracode/opus_hardened_plans.json` and the
cross-track critic `opus_critic.md`. r1 (arc) and r2 (coding) are already on
record from the ultracode campaign (3 Fable designs, 9 Fable adversarial lenses
producing 41 fatal-flaw findings, 3 Opus hardens, 1 Opus critic). **This campaign
is r3 and r4 only.** Do not re-derive the designs; do not re-open story quality
(closed by operator directive 2026-08-04). All three tracks are CORRECTNESS
defects.

---

## 0. What I verified myself, before writing this

Everything below was measured against the REAL Windows files via the ComfyUI venv
(`C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`), not the Linux
sandbox mount. Probe scripts were run and discarded; the numbers are mine, not the
plans'.

### 0.1 Line-anchor audit -- 89 of 89 claims land

I extracted every `file:line` the three plans lean on -- 89 distinct anchors
across 30 files -- and asserted the real line content at each. **All 89 resolve
to what the plan says is there.** The tree has not moved since the plans were
built. A panel claim of the form "that line number is stale" starts from a false
premise unless it names a specific anchor and shows the actual line.

Spot anchors worth knowing, all confirmed:

| anchor | actual line content |
|---|---|
| `nodes/_otr_voice_bank.py:43` | `CASTING_POLICY_VERSION = "2"` |
| `nodes/_otr_voice_bank.py:432` | `if pool:` (the tier-of-one accept) |
| `nodes/_otr_voice_bank.py:702` | `return sorted(pool, key=_key)[0]` |
| `nodes/_otr_voice_bank.py:717` | `if engine in ("google_tts", "chatterbox", "dia"):` |
| `nodes/_otr_casting.py:615` | `rng.shuffle(genders)` -- the allocator's ONLY rng use |
| `nodes/_otr_casting.py:1635` | `"gender": response.gender,` |
| `nodes/cast_lock.py:563` | `gender = str(entry.get("gender") or "").strip().lower()` |
| `nodes/cast_lock.py:616` | `except VoiceCastingError as exc:` |
| `nodes/otr_image_gen_dispatcher.py:159` | `base = int(cfg.get("request_seed") or 0)` |
| `nodes/otr_image_gen_dispatcher.py:160` | the mode gate |
| `config/audio_engine_profiles.yaml:154` | `allowed_voice_banks: [kokoro_builtin]` |
| `tests/test_voice_bank.py:226` | `assert announcer_voice_ref("kokoro").voice_ref_id == ...` |
| `tests/test_cast_lock.py:252` / `:286` | `assert cast["a1"]["voice_ref_id"] == "bm_george"` |

### 0.2 Behavioural probes -- what reproduces

| plan claim | my measurement | verdict |
|---|---|---|
| kokoro announcer pinned to `bm_george` every seed | 64/64 seeds -> `bm_george` | CONFIRMED |
| chatterbox/dia/google_tts have exactly 2 announcer refs (so a seeded draw over a 1-list is identity) | 2 distinct each over 12 seeds | CONFIRMED |
| exactly 3 size-1 first-accepted tiers | exactly 3: `(indextts2,female,warm)` `[1,1,1,23]`, `(kokoro,male,warm)` `[1,1,1,13]`, `(kokoro,male,deep)` `[1,1,1,13]` | CONFIRMED, incl. pool sizes |
| timbre-honouring combos: floor1 8/24, floor2 5/24, floor3 2/24 | 8/24, 5/24, 2/24 | CONFIRMED exactly |
| floor=2 and floor=3 both drive size-1 tiers to 0 | 0 and 0 | CONFIRMED |
| bank is 179 rows: indextts2 40 / kokoro 28 / chatterbox 42 / dia 42 / elevenlabs 21 / google_tts 6 | identical | CONFIRMED |
| `_plan_gender_distribution(1,['male'])` -> female 400/400 | 400/400 female | CONFIRMED -- **the hard gate is real** |
| `_plan_gender_distribution(1,[])` -> male 400/400 | 400/400 male | CONFIRMED |
| `request_cache_key(...)` HEAD literal `e8e5a5dc...9f17d` | byte-identical | CONFIRMED |
| `resolve_object_seed({0,request_hash},'c01','pp_portrait') == 1692108723` | 1692108723, and `== int(sha256(b'0:c01:pp_portrait')[:8],16)` | CONFIRMED |
| scene stills draw unrelated seeds per beat | still_b002/4/6 -> 2664385734 / 3602429248 / 343226934 | CONFIRMED (defect is real) |
| 11 registered image engines, ALL `accepts_reference_image` absent, ALL `required_inputs=('text_prompt',)` | identical, all 11 named | CONFIRMED |
| 14 shakespeare sidecars, all carrying `characters` | 14/14, 85 char rows, `{male:30, female:23, unknown:32}` | CONFIRMED incl. the "32 unknown" figure |
| 42 cast_hints (NOT 39) across `curated_scenes.sample.json` | exactly 42 over 14 scenes | CONFIRMED |
| canonical node 88 `OTR_ImageDirector` is exactly 8 widget slots | `['per_object','per_object','per_object',15,'request_hash',42,'{}','fp8_ok']` | CONFIRMED |
| canonical node 80 = `['default','auto_registry',True,'indextts2','kokoro','cuda']` | identical | CONFIRMED |
| canonical `num_characters` = 2 (node 1 widget[2]) | 2 | CONFIRMED |
| node 87 reads `z_image_turbo` in all three IMAGE slots and viz_* in the video slots | confirmed -- canonical as saved mints zero stills | CONFIRMED |

**The headline defect, re-measured independently.** Walking every ledger under the
REAL output base (`C:\Users\jeffr\Documents\ComfyUI\output\otr`, 4,183 json files
-- note this is NOT `<repo>/otr`, see CLAUDE.md section 6), joining each
non-announcer cast row to the sidecar roster through the plan's own tier ladder:

```
adaptation ledgers : 88   (plan said 94)
non-announcer rows : 176  (plan said 188)
CONTRADICTING ROWS : 44   (plan said 44)   <-- exact
rate               : 25.0% (plan said 23%)
```

All eight characters the plan names by hand are confirmed contradicted:
MALVOLIO 3, MARIA 3, ROMEO 3, JULIET 2, MIRANDA 3, CELIA 4, ROSALIND 3, LEAR 3
-- plus MACBETH 3, BENEDICK 3, VIOLA 2, MARCELLUS 2, CORDELIA 2, FERDINAND 2,
TITANIA, BANQUO, HERO, PROSPERO. Concrete instances:
`PROSPERO ledger=female sidecar=male`, `MIRANDA ledger=male sidecar=female`,
`LEAR ledger=female sidecar=male`, `CORDELIA ledger=male sidecar=female`.

**The defect count is exact. The denominator is not.** My ledger/row totals run
~6% under the plan's because my scan requires `meta.source_bank` and a non-empty
row `name`. This does not weaken anything -- but the plan should say 44 of ~176-188,
and no downstream text should quote "23%" as a measured constant.

### 0.3 Baseline suite -- green

```
271 passed in 5.99s
```
across `test_cast_voice_replay_parity, test_otr_casting, test_character_roster,
test_shakespeare_sources, test_public_domain_sources, test_cast_llm_naming,
test_cast_voice_slots, test_voice_bank, test_voice_bank_coverage,
test_kokoro_char_voices, test_announcer_voice, test_cast_lock,
test_hybrid_voice_fit`.

### 0.4 The one thing that did NOT reproduce

**Track 1, `discarded_objections` D01** rejects the reviewers' proposed
`count=len(unpinned)` fix partly on this measurement:

> "probed, the call consumes 0 rng calls at count 0 and 1, 2 at count 2 and 3,
> 4 at count 4 and 5"

`_plan_gender_distribution` has exactly ONE rng use -- `rng.shuffle(genders)` at
`_otr_casting.py:615`. Counting `getrandbits` consumption:

```
count=0 -> 0     count=1 -> 0     count=2 -> 3
count=3 -> 3     count=4 -> 9     count=5 -> 11    count=6 -> 11
```

The pairing the objection asserts holds for (0,1) and (2,3) but **breaks at (4,5)**
(9 vs 11), and none of the absolute figures match. The post-call stream tells the
load-bearing story cleanly: count=0 and count=1 leave the rng in an **identical**
state; every count >= 2 leaves it in a **different** state from count=0.

**The objection's conclusion survives and is in fact better supported than the
number it cites** -- re-calling the allocator with a reduced count does move the
stream, so the reviewers' fix does re-break replay parity. But the cited tally is
wrong, and this is exactly the class of thing r4 is chartered to find. It is a
documentation defect in the plan, not a design defect. **Correct the sentence; do
not reopen the design.**

---

## 1. r3 -- WIRING / INTEGRATION / SEQUENCING (what I want attacked)

The designs are settled. r3's job is the seams: does each step's described change
meet code that actually looks like that, in the order the critic prescribes?

**The ship order under review** (from `opus_critic.md`, and the operator has
adopted it):

1. Track 3 Step 1 -- announcer unpin
2. Track 1 Steps 1-4 -- roster module -> source_meta plumbing -> curated supplement -> pin
3. Track 1 Step 6 merged with Track 3 Step 7 (the `gender='other'` unservable path)
4. Track 3 Steps 2 + 4 (speaker_id, tier floor = 2)
5. Track 2 Steps 1-4 -- HEAD baselines captured AFTER Track 1 lands
6. Track 2 Steps 5-7 + 9
7. Track 2 Step 8 (klein) only if Step 9's A/B fails

**CUT and not up for re-litigation** (operator-confirmed): Track 3 Step 5 (kokoro
char_voice -- unreachable, proven at `audio_engine_profiles.yaml:154`), Track 3
Step 8 (timbre synonyms), Track 3 Step 6 (drift guard), Track 2 Step 8 unless the
A/B fails. Track 1 Step 5 (replay parity) is **downgraded to a `PROD_BUG_LOG`
entry** with reproducing seed 424242.

**Operator hard gates, not negotiable:**
- The portrait-seed fix must NOT land on adaptation lanes before the gender pin.
  Alone it converts three ambiguous faces into ONE confident wrong-sex face.
- Do NOT feed pinned genders into `prior_genders`. Verified above: that turns a
  coin flip into a guaranteed error. `_plan_gender_distribution` stays untouched;
  override at pinned indices only.
- Gender feeds the description LLM, the outline prompt, the dialogue cast block
  and the image gender anchor, so the fix CHANGES SCRIPTS. That is a downstream
  consequence of a correctness fix, and the commit must say so.

### r3 questions I actually want answered

**Q1 -- Ordering hazard between chunk 2 and chunk 4.** Track 1 Step 4 pins
adaptation casts to source gender; Shakespeare is male-heavy and indextts2 is
male-LIGHT (17 male / 23 female, re-confirmed). Track 3 Step 4's floor=2 was
measured under TODAY's demand curve. Does landing Track 1 BEFORE Track 3 Step 4
(as the critic orders) invalidate the floor measurement, or is the tier-size
invariant demand-independent? My read: the invariant is over the BANK, not over
demand, so it holds -- but a male-shifted cast makes the male tiers the binding
constraint at cast time, and nobody has measured collision rate under the pinned
distribution. Is that a must-fix or a note?

**Q2 -- Track 1 Step 4(i) dataclass field placement.** `EnsembleSlot` gains
`source_owned: bool = False` AFTER `age_band`. `tests/test_cast_llm_naming.py:141`
builds `EnsembleSlot("c02","ALICE STONE","fem...` with FIVE positional args
(confirmed at that line). Is trailing-append genuinely sufficient for BOTH
`EnsembleSlot` and `CastSlot`, and are there construction sites outside `nodes/`
and `tests/` (scripts/, tmp/ excluded) that would shift?

**Q3 -- Track 1 Step 2 keyword-only plumbing.** `source_meta_from_scene` gains
`*, text_path: Path | None = None`. Existing positional callers confirmed at
`tests/test_shakespeare_sources.py:177` and `tests/test_public_domain_sources.py:129`.
Are there OTHER callers of `source_meta_from_scene` / `source_meta_from_unit`
anywhere in the tree -- including `scripts/` and the bench runners -- that the
plan does not name?

**Q4 -- The sidecar path the plan asserts.** Step 1 defines
`sidecar_path_for_text(text_path) = text_path.parent / (stem + '.provenance.json')`
and says it matches the writer at `scripts/otr_fetch_public_domain.py:244-246`.
The sidecars actually live in `config/source_banks/shakespeare/sources/`, one
directory below where Step 3 wants to commit
`roster_gender_supplement.json` (`.../shakespeare/`). Confirm the join resolves
for the REAL manifest `text_path` values, not just in principle.

**Q5 -- Track 3 Step 1's two-copy hazard.** The plan hoists
`_SEEDED_ANNOUNCER_ENGINES` and imports it into `cast_lock.py` to replace the
duplicated literal at `:527` (confirmed present). Does that import create a cycle
(`cast_lock` -> `_otr_voice_bank`), and is `:527`'s tuple the ONLY duplicate?

**Q6 -- Track 2 Step 2 placement.** The branch must land AFTER the mode gate at
`:160-161`, not after the 4242 pins at `:152-157`, because `base` is assigned at
`:159`. I confirmed `base` is at `:159` and the mode gate at `:160`. Verify the
claim that mis-placement raises `UnboundLocalError` in a function called at `:998`
outside any try/except -- i.e. that `dispatch_images` really does die outright.

**Q7 -- Track 2 Step 6 cache-key ordering.** The reference must resolve BEFORE the
key at `:1000`, not at `:1103`, because the cache-HIT branch `continue`s at `:1047`.
All three anchors confirmed. Is resolving at `:997-998` safe with respect to
everything else those lines already compute (`engine_id` resolved at `:939`)?

**Q8 -- Chunk 3's merge.** Track 1 Step 6 and Track 3 Step 7 are the same defect
with two owners. Merged, CastLock stamps the fallback ref, which makes
`_resolve_clone_ref_path`'s `vrid` lookup hit first -- so Track 3's fallback test
would exercise dead code unless it is rewritten. Does the merged step leave the
anyref path still reachable and still tested, or does it silently orphan it?

---

## 2. r4 -- CONVERGENCE, and the DISCARDED-OBJECTIONS AUDIT

**This is the part nobody has done.** Each hardened plan carries a
`discarded_objections` list: fatal-flaw findings from the 9 Fable adversarial
lenses that the Opus harden REJECTED, each with a stated reason. A wrongly-rejected
fatal flaw is a must-fix hiding behind a plausible reason. **19 objections were
discarded across the three plans.** Audit every one.

For each, the panel should answer three questions and nothing else:
1. Is the stated `why_rejected` **factually true against the current code**?
2. If true, does it actually **dispose of the objection**, or only part of it?
3. If the rejection is wrong, what is the **minimum change** that restores it?

I have already audited D01 myself (section 0.4): factually wrong on the cited
number, right in its conclusion -- fix the sentence, keep the design.

### The 19 discarded objections

**Track 1 (gender-voice)**
- **D01** reviewers' `count=len(unpinned)` fix. *Driver finding: the cited rng tally is wrong; the conclusion holds. Verify my correction.*
- **D02** address-form tier as same-line-only. Rejected as a STEP on proportionality (+0 pins at num_characters=2) and on "the whole-play text is not on disk". *Check both: is `config/source_banks/_corpus/` really 64 prose bodies + 1 sidecar and zero Folger plays? Is `body` really rebound at `scripts/otr_fetch_public_domain.py:325` one line after the roster parse at `:324`?*
- **D03** HELENA/HERMIA/LYSANDER/DEMETRIUS empty descriptions = a vendor-time parser defect. Rejected as "not cheaper". *Check the sidecar: do all four really carry a populated `roster_name`?*
- **D04** forbid `other` on source-owned slots. Rejected because override-in-place leaves 'other' incidence byte-identical. *This one is load-bearing for the operator's hard gate -- verify the 400-seed byte-identity claim at num=3.*
- **D05** pass `source_bank_id` into the replay. Action accepted, characterization rejected. **Note: Track 1 Step 5 is now CUT/downgraded, so D05 is moot -- confirm it carries no residue into the merged chunk 3.**
- **D06** rename Puck->Robin, Don Pedro->Prince. Rejected: zero gender payoff + broadcast-facing. *Verify ROBIN `roster_name=''`/`absent_from_roster` and PRINCE `unknown`.*

**Track 3 (voice-variety)**
- **D07** a third announcer-pool copy at `story_orchestrator.py:861`. Rejected as a BARK list, not kokoro. *Verify the actual values at those lines.*
- **D08** floor=3 dominates floor=2. Rejected on the 8/5/2-of-24 measurement. **Driver re-measured: 8/24, 5/24, 2/24 -- exact. Rejection stands.** Confirm only that floor=2 is what the operator gets by default and `OTR_CAST_MIN_TIER_POOL` really makes 3 a one-run A/B.
- **D09** scope step 2 to two pairs, drop mirror-stamping. Rejected: 12 rows, not 6, because same-human pairs have DIFFERENT ref_paths within one engine. *Verify the 40 shared ref_paths and the two genuine pairs.*
- **D10** drop the `not allow_voice_reuse` clause at `cast_lock.py:535-537`. Rejected as unreachable in production + it re-bases chatterbox/dia. *Verify target_engine == announcer_engine is really the only trigger.*
- **D11** re-derive the concentration thresholds. Rejected in favour of a structural invariant. *Driver agrees on method; confirm the invariant is measurable without a simulation.*
- **D12** steps floor+synonyms must land in one commit. Rejected: the asymmetry is real (synonyms alone 3->5 pins; floor alone 3->0). **Since synonyms are now CUT entirely, confirm D12 leaves no residue.**

**Track 2 (portrait)**
- **D13** edit canonical node 87. Rejected in favour of naming a profile. **This is the most consequential rejection in the set** -- it was UNANIMOUS across all three reviews and overruled by one probe. *Verify: does `config/profiles/widget_mapping.json` really map `role_overrides.character_visual` -> `[OTR_VideoDirector, character_video_model]`? Do ~40 shipped profiles really set it? Does `scripts/otr_canonical_api_run.py --profile` really load the real canonical JSON?* If this rejection is wrong, the whole Track 2 live proof is unrunnable as written.
- **D14** schema additions to CanonicalImage/ImageRequest. Rejected as dead. *Verify the grep: constructed nowhere but schemas.py and one test?*
- **D15** name the field plural. Rejected. Low stakes.
- **D16** crop='center' to output dims. Rejected in favour of width=0/height=768/crop='disabled'. *Verify ImageScale really derives the missing side (`nodes.py:1885-1889`) and that VAE.encode crops internally.*
- **D17** GiB/GB unit confusion. Rejected as a reviewer error. Low stakes, easily checked on disk.
- **D18** params-gate `_node_candidates`. Rejected: the singleton cache would ship a latent episode killer. *This is a sharp claim -- verify `self._classes` caching at `z_image_turbo.py:318-320` and the registry singleton.*
- **D19** extend the identity-seed pin to jump segments. Rejected as "half right" -- jump segments get the anchor and the reference but NOT the pinned seed. *Verify `tests/test_multiclip_jump_stills.py:234-237` really asserts three DISTINCT seeds.*

### r4 convergence question

Beyond the audit: **is there any remaining must-fix that would break the build or
the ledger?** Specifically, per the standing operator rule that a ripped or
changed pass must leave NO ledger field unowned -- Track 1 adds
`cast_source_contract`, Track 2 adds `derived_from_portrait_hash` and
`portrait_anchor_mode`, chunk 3 adds `voice_cast_fallback`. Each must have exactly
one owner and a defined value on every path, including the cache-hit path and the
unpinned/invention lanes. Name any field that does not.

---

## 3. Rules for this panel

- **Ground every claim in the real files.** A claim without a `file:line` that I
  can check will be discarded, not weighed.
- **Do not propose story/prose improvements.** Story quality is CLOSED. A finding
  that a script reads differently after the gender fix is EXPECTED and is not a
  defect.
- **Do not re-open the cut items** (kokoro char_voice, timbre synonyms, drift
  guard, speculative klein wiring) except to say a cut breaks something else.
- **Do not re-derive the designs.** r1 and r2 are on record.
- The driver is the sole judge. The panel proposes; every surviving claim is
  re-checked against the Windows files before it is folded in.


---

# THE CROSS-TRACK CRITIC (ship order + cuts)

## (1) WHAT IS MISSING

**A. The gender fix is not a voice fix — it is a SCRIPT fix, and no track says so.** `slot.gender` is handed to `_build_user_prompt(... gender=slot.gender ...)` at `nodes/_otr_casting.py:777-785`, which is the LLM that writes `character_description`. The row gender then also goes into the OUTLINE prompt (`nodes/OTR_LedgerScriptWriter.py:4144`, `cast_descriptions` tuple) and into the dialogue cast block (`nodes/_otr_line_composer.py:446`). So fixing MALVOLIO's gender changes the outline, the dialogue, and the physical description — not just the voice. Track 1's summary ("the row gender IS the audible voice") is true but describes ~20% of the blast radius. This must be stated up front or a reviewer will read the resulting script diff as quality work.

**B. Track 1 does not know the hybrid LLM voice-fit exists.** `nodes/cast_lock.py:575-595` honours the hybrid proposal and `continue`s BEFORE `assign_voice_for_slot` at :603, and `hybrid_voice_fit_enabled()` defaults ON (`nodes/_otr_casting.py:848-851`). Track 1's entire mechanism narrative describes the *fallback* path. The fix still works — `build_voice_cards` hard-filters `e.gender == gnorm` (`nodes/_otr_voice_bank.py:505-507`) and `validate_voice_proposal(accepted, engine, gender, ...)` re-checks at lock time — but only Track 3 Step 7 found this seam, and Track 1 never cites it.

**C. Does the gender fix change which voice pool is drawn from? YES, through three independent mechanisms**, all verified: (i) `_LADDER`'s last tier is `frozenset({"gender"})` and every tier includes gender (`_otr_voice_bank.py:62-67`), so the pool is gender-filtered at every rung; (ii) `stable_cast_seed` puts `gender` in the hashed payload (`:283-298`), so even an identical pool draws a *different element*; (iii) `build_voice_cards(engine, gender)` filters the LLM card window by gender.

**D. Does the voice expansion invalidate the gender fix? Not logically — but it invalidates its EVIDENCE three times, and it shifts the demand curve nobody measured.** Track 3 bumps `CASTING_POLICY_VERSION` 2→3 (Step 4), folds `+mtp{floor}` into the seed, then bumps 3→4 (Step 8). Any `voice_ref_id` literal recorded by Track 1's live leg goes stale at each bump. More seriously: **Track 1 pins adaptation casts to source gender, and Shakespeare is male-heavy, while indextts2 is male-LIGHT (measured: 17 male / 23 female).** At `num_characters=2`, male-male pairs go from ~32% (the 40/40 roll) to the norm on the adaptation lane. Track 3 measured its size-1 tiers under *today's* distribution; Track 1 makes the male pool the binding constraint. Neither plan names this. Track 3's floor=2 helps, which is an argument for landing it, not against.

**E. The portrait fix depends on gender being correct first — hard, through three channels.** (i) `_ensure_gender_anchor` (`nodes/otr_meta_brief_image_prompt.py:78-90`) literally prepends `"adult man, "` / `"adult woman, "` read from `char['gender']`, applied to the portrait appearance at :1251 and to EVERY final prompt at :1702. (ii) `character_description` is written by an LLM handed the wrong gender (see A), so the appearance text itself describes the wrong sex. (iii) Track 2 Step 2 derives the scene-still seed from the portrait's `prompt_hash`, which is a hash of that gendered prompt.

**The amplification nobody states: Track 2 shipped WITHOUT Track 1 makes the gender defect worse.** Today a mis-gendered MIRANDA gets three unrelated faces across three beats — at least one may read ambiguously. After Track 2 Steps 2+7 she gets ONE male face, seed-locked and reference-latent-conditioned, held consistently across the whole episode. Track 2 converts an intermittent defect into a confident, permanent one.

**F. A consumer nobody named: `nodes/otr_shot_lock.py`.** It has its OWN `_appearance_for_char(ledger, char_id)` at :116, feeding the shot-direction LLM at :681 (`appearance[:160]` — truncated) and building `text_prompt` from `appearance, setting, ...` at :788-790. Grep confirms **zero gender references in that file** — the video/shot lane carries the appearance with NO gender anchor and a 160-char truncation that can cut the gendered opening. Track 2 Step 4 hardens only the still composer.

**G. Track 1 Step 6 and Track 3 Step 7 are the SAME defect, claimed by two tracks.** Both target `cast_lock.py:616-620` → `_otr_voice_node_common.py:109-127`. Track 1 wants CastLock to stamp the fallback ref; Track 3 wants a comment telling the next reader not to narrow that draw and a test asserting the path still resolves. Landing Track 1 Step 6 makes `_resolve_clone_ref_path`'s `vrid` lookup (`:91-95`) hit first, so Track 3's fallback test would exercise dead code. **Merge them into one step with one owner.**

**H. Cache-miss collision.** Track 1 (gender → anchor → prompt text → `prompt_hash`) and Track 2 Step 4 (appearance prepend → `prompt_hash`) each invalidate the still cache for every character object. Land them in one sprint so the operator pays ONE full re-render, not two.

## (2) SHIP ORDER

1. **Track 3 Step 1** (announcer unpin). Fully independent — touches no character seed. Audible night one. Ship first because it is the cheapest proof the sprint is live.
2. **Track 1 Steps 1-4** (roster module → source_meta plumbing → supplement → pin + no-rename). Nothing downstream is trustworthy until the cast row is right.
3. **Track 1 Step 6 merged with Track 3 Step 7** (the `gender='other'` unservable path — one owner for `voice_ref_id`).
4. **Track 3 Steps 2 + 4** (speaker_id, tier floor = 2). After Track 1, because the floor's payoff is now measured against a male-shifted adaptation demand.
5. **Track 2 Steps 1-4** — and **capture Step 1's HEAD baselines AFTER Track 1 lands**, not before. The synthetic literals survive, but the episode-level "byte-identical to today" claim does not.
6. **Track 2 Steps 5-7 + 9** (capability, cache-key-safe reference resolution, z_image wiring, live A/B).
7. **Track 2 Step 8** (klein) — contingent on Step 9's A/B failing. Do not build it speculatively.

Hard gates: Track 1 Step 4 **before** Track 2 Step 2 (or you lock a wrong face). Track 2 Step 2 **before** Step 4 (Track 2 says this; it is right). Track 2 Step 6 **before** Step 7 (the anchor must be in the cache key before any engine consumes a reference).

## (3) THE SINGLE HIGHEST-VALUE, LOWEST-RISK STEP

**Track 2 Step 2 + Step 3: derive the `scene_character` seed from the character's own portrait draw, and stamp the anchor.**

It is roughly six lines in one pure function plus a row lookup. No engine wiring, no new model, no VRAM cost, no LLM call, no widget, no canonical-JSON edit, no network. It is reversible with one env var. And it converts "this character's face changes every single beat" — the most visually jarring thing in a nightly watch — into one face per character, on **every lane including the invention lanes**, where there is no gender defect to contaminate it.

Ranked against the alternatives: Track 1's gender fix is the more important *correctness* win but needs a new module, a plumbing change in two source modules, a committed data file, and an allocator override before it shows anything, and it only touches adaptation lanes. Track 2 Step 7 (reference latent) is the biggest potential win and the biggest risk — the installed `z_image_turbo_nvfp4` checkpoint may ignore the reference entirely. Track 3 Step 1 is nearly free and audible, but it is one voice at the top of the episode.

Caveat, stated plainly: on the **adaptation** lanes, Step 2 alone locks in whatever gender the roll produced. Ship Track 1 Steps 1-4 in the same sprint, or accept that Shakespeare episodes get one *consistently wrong* face until it lands.

## (4) IS ANY OF THIS SECRETLY STORY-QUALITY WORK?

**Mostly no — with three things to say plainly:**

- **Track 1's script-text change is NOT quality work, but it will look like it.** Correcting MALVOLIO's gender changes the outline prompt, the description prompt, and the dialogue cast block, so the generated prose differs. That is a downstream consequence of a correctness fix, exactly like "Malvolio speaks with a woman's voice." **The plan must say this explicitly**, because the first person to diff a script and see rewritten dialogue will reasonably ask whether the closed directive was violated.
- **Two items ARE over the line and are correctly already dropped** — the `Don Pedro → Prince` and `Puck → Robin` cast_hint renames. Track 1 rejected them for the right reason (zero gender payoff) but they are also broadcast-facing name changes with no defect behind them. Keep them dead.
- **Track 1 open question #2 (expanding `num_characters` past 2 so a 7-speaker scene is not truncated) is scope creep dressed as fidelity.** It is not on the established-defect list, the operator did not ask for it, and it collides with the count-match invariant at `OTR_LedgerScriptWriter.py:4119`. It is also the *only* item here that would change how much story gets told. Leave it in open questions; do not let it become a step.

One genuinely borderline line: Track 2 Step 4's condition (c) — "prepend only when the first 40 characters are not already present, so an LLM paraphrase does not stack two competing subject descriptions." The reasoning is prose hygiene, but the purpose is identity anchoring. Keep it; describe it as de-duplication, not as prompt improvement.

## (5) WHAT CAN BE CUT WITHOUT THE OPERATOR NOTICING

**Cut outright:**

- **Track 3 Step 5 (kokoro lang_code / roles / begin_episode) — the whole step.** `config/audio_engine_profiles.yaml:154` sets `allowed_voice_banks: [kokoro_builtin]` while the shipped workflow runs `voice_bank='default'`, so kokoro char_voice is unreachable. Track 3 admits this and then builds it anyway. The `requested_role` singleton-leak sub-fix is only reachable *through* kokoro char_voice, so it is unreachable too. Zero audible effect. **This is the largest cut available — a whole step, in the file with the most call-site risk.**
- **Track 3 Step 8 (timbre synonyms).** Self-declared optional, and Track 3's own measurement shows the floor alone takes size-1 tiers from 3 to 0. It adds a second vocabulary to maintain and a dataclass field whose entire purpose is to stay off the LLM card path. Nobody hears "the writer asked for deep and got a baritone."
- **Track 3 Step 6 (drift guard).** Green on landing, guards a docstring. If you want it, it is three assertion lines folded into Step 1's commit — not its own step.
- **Track 2 Step 8 (flux2_klein wiring).** Contingent on an A/B that has not run. Build only if z_image fails.
- **Track 1 Step 3's ARIEL and PUCK entries.** 2 of 42 hints, genuinely editorial, no defensible source. Let them roll.

**Downgrade, do not build:**

- **Track 1 Step 5 (replay parity).** Track 1 proves it is latent: `_assign_bark_voices` writes only `row['voice_preset']` (`cast_lock.py:355-361`) and node 80 ships indextts2. Record it as a known-latent ledger-consistency defect with the reproducing seed (424242) in `PROD_BUG_LOG.md` and move on. It becomes real only if the operator ever switches to bark.

**Do NOT cut (the operator watches these nightly):** Track 1 Steps 1-4, Track 2 Steps 2-3, Track 3 Steps 1 and 4.

**One thing that is not a defect fix at all:** Track 3 Step 3 (registering the operator's three recordings). Nothing is broken by them being unregistered — it is a feature. He will notice it, positively, which is a fine reason to keep it, but it should not be counted as closing a defect, and its three unresolved consent questions about the other six on-disk refs should stay blocking.

---

**Bottom line.** The one thing all three plans miss is that gender is not a voice field — it is fed to the description LLM, the outline prompt, the dialogue cast block, and the image prompt's gender anchor, which means Track 1 changes the script and the portrait, and Track 2 shipped first would lock the wrong sex onto a character's face permanently. Ship order is therefore forced: announcer unpin, then Track 1's roster→gender pin, then Track 2's portrait-seed pin, then the tier floor, then the reference-latent work last behind its live A/B. The single best step to build first is Track 2 Step 2 — six lines that turn a face that changes every beat into one face per character on every lane — but it must not land on the adaptation lanes ahead of Track 1. Cut Track 3 Step 5 entirely (kokoro char_voice is unreachable in production, proven at `audio_engine_profiles.yaml:154`), cut the timbre synonyms and the drift guard, and hold the klein wiring until the z_image A/B says it is needed.

---

# THE THREE HARDENED PLANS


## TRACK: gender-voice

**Summary:** There is ONE audible defect, not two. The row gender is decided by a 40/40/20 roll in `precompute_ensemble_slots`, and under the shipped canonical workflow (node 80 = indextts2) that row gender IS the audible voice: cast_lock.py:563 reads `entry.get('gender')` -> `assign_voice_for_slot(gender=...)` -> `stable_cast_seed(..., gender=gender, ...)`. Measured across all 94 published adaptation ledgers (188 non-announcer rows): 44 rows carry a gender that CONTRADICTS the shipped provenance sidecar -- 23% of every adaptation character ever shipped, including MALVOLIO, MARIA, ROMEO, JULIET, MIRANDA, CELIA, ROSALIND and LEAR. The CastLock bark-replay divergence is REAL and reproduces at cast_seed 424242 (writer male/other/female vs replay female/other/male), but `_assign_bark_voices` writes only `row['voice_preset']` (cast_lock.py:355-361) and never gender, so it is a LATENT ledger-consistency defect, inert while node 80 runs indextts2 -- it must not displace the real fix.

The fix has three parts. (1) The roster truth already on disk must reach the writer: 14 provenance sidecars carry a `characters` list, and `source_meta_from_scene` never loads it. (2) 12 of the 42 shipped cast_hints have no gender in the sidecar -- verified, including ANTIPHOLUS and both DROMIOs, which the original plan claimed were settled by agreement and are not; they are all gender='unknown'. Those 12 get a committed, evidence-carrying data supplement rather than a regex or an LLM. (3) The pin is applied WITHOUT touching the allocator.

The core mechanic differs from the original plan AND from all three reviewers. Every reviewer correctly found that feeding pinned genders into `prior_genders` turns a coin flip into a guaranteed error (probe: `_plan_gender_distribution(1,['male'])` -> female 400/400). But their proposed fix -- call it with `count=len(unpinned)` -- still changes the writer's draw count (probe: count 0/1 consume 0 draws, 2/3 consume 2, 4/5 consume 4), which re-breaks replay parity, and makes a lone unpinned slot deterministically male. The correct design is to leave `_plan_gender_distribution` completely alone -- same count, same priors, same draws -- and OVERRIDE the gender at pinned indices. Probed on the real module over 200-400 seeds: pinned slots correct 200/200 where today is wrong 106/200; the UNPINNED slot's distribution is unchanged (HORATIO 200 male / 200 female both ways); the 'other' incidence at num=3 is byte-identical; the rng call count and the post-call stream are identical on every seed; and with `gender_by_name=None` the ensembles and rng state are byte-identical across seeds x counts, which is the C7 guarantee for the invention lanes.

### Steps

#### Step 1 -- New render-safe roster-join module (pure functions, no behavior change)

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_roster_gender.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_roster_gender.py`

**Change:** Add a NEW module `nodes/_otr_roster_gender.py`, stdlib only (`json`, `pathlib`, `dataclasses`), no I/O at import. Do NOT extend `nodes/_otr_character_roster.py`: verified it is imported ONLY by scripts/otr_fetch_public_domain.py:56 and tests/test_character_roster.py:15, and its docstring at :25-26 states 'the render path reads a confirmed record and never infers'. A separate module keeps that contract structural instead of conventional.

Contents:
- `sidecar_path_for_text(text_path: Path) -> Path` = `text_path.parent / (text_path.stem + '.provenance.json')` -- matches the writer at scripts/otr_fetch_public_domain.py:244-246.
- `load_roster_characters(text_path) -> tuple[dict, ...]`: returns `()` on FileNotFoundError / malformed JSON / missing `characters`; otherwise returns PLAIN, json-serializable dicts (NOT MappingProxyType, NOT frozen dataclasses) -- `_copy_sidecar` at nodes/_otr_source_payload.py:195-203 is a shallow `dict(sidecar)`, so these objects alias straight into durable ledger meta and get json-dumped.
- `@dataclass(frozen=True) RosterGenderVerdict: gender, evidence, tier, matched: tuple[str, ...]`.
- `resolve_roster_gender(slot_name, characters) -> RosterGenderVerdict`. Tier ladder, first tier yielding ANY candidate wins the candidate set: T1 exact (UPPER(slot) == `name` or `roster_name`); T2 alias (reuse the honorific-stripping shape at _otr_character_roster.py:176-192); T3 qualified (`name`.upper() startswith slot+' ' -- the ANTIPHOLUS -> 'ANTIPHOLUS OF EPHESUS' case); T4 contains (slot startswith `name`+' '). AGREEMENT RULE: all candidates in the winning tier agree on male|female -> return it; disagree -> ('unknown','ambiguous_join'); all agree on unknown -> ('unknown','roster_unknown').

**Why:** The join is the only thing standing between the sidecar and the writer, and it must abstain honestly. Grounded corpus measurement over all 14 sidecars x curated_scenes.sample.json: 42 cast_hints (NOT 39), tiers exact:38 / qualified:2 / none:2, and only 30 resolve to male|female. The original plan's headline -- 'you do not need to know which Dromio to know he is a man' -- is FALSE: comedy_errors__act3_scene1.provenance.json records ANTIPHOLUS OF EPHESUS, DROMIO OF EPHESUS and DROMIO OF SYRACUSE all as gender='unknown', gender_source='unknown'. The ladder must abstain there, and Step 3 is what actually closes them.

**Proves it worked:** tests/test_roster_gender.py, run against the REAL shipped sidecars: 'MALVOLIO' -> tier=exact gender=male; 'ANTIPHOLUS' -> tier=qualified, one candidate, gender=unknown evidence=roster_unknown; 'DROMIO' -> tier=qualified, TWO candidates, gender=unknown (pins the corrected claim so nobody re-asserts the false one); 'PUCK' and 'DON PEDRO' -> tier=none; a synthetic pair disagreeing male/female -> evidence='ambiguous_join'. Absent this module the tests cannot even import.

#### Step 2 -- Plumb the sidecar roster into render-path source_meta (plumbing only, zero behavior change)

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_shakespeare_sources.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_public_domain_sources.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_shakespeare_sources.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_public_domain_sources.py`

**Change:** Change `source_meta_from_scene` (nodes/_otr_shakespeare_sources.py:428) to `def source_meta_from_scene(resolved, *, text_path: Path | None = None)`. Pass the TEXT PATH, not a directory -- the sidecar name derives from the text stem. `text_path` is already computed at :478; the call site is :487. When supplied, add `"characters": [...]` to the returned dict ONLY when the list is non-empty (an absent key is honest absence; never `[]`). Mirror exactly in `source_meta_from_unit` (nodes/_otr_public_domain_sources.py:540) -- its `text_path` is in scope at :500, call site :518.

Keyword-only with default None keeps the existing positional test calls green (verified: tests/test_shakespeare_sources.py:177 and tests/test_public_domain_sources.py:129 both call with one positional arg). No schema work: `_copy_sidecar` accepts any dict, and `_otr_source_snapshot.py:218-224` re-validates only `isinstance(source_meta, dict)`.

**Why:** source_meta is the ONLY channel that reaches the writer (`meta['source_meta']` stamped at nodes/OTR_LedgerScriptWriter.py:3565) and is copied wholesale into the durable ledger, so the roster becomes both an input and an auditable receipt. Splitting plumbing from decision makes this step provably behavior-free -- nothing reads the key yet.

**Proves it worked:** Extend tests/test_shakespeare_sources.py: `fetch_shakespeare_scene` for folger-twelfth-night:act2-scene5-malvolio-letter returns `source_meta['characters']` containing MALVOLIO with gender='male'; and for a manifest whose text_path resolves under config/source_banks/shakespeare/fixtures/ (verified: that directory contains NO .provenance.json) the 'characters' key is ABSENT, not an empty list. Without the wiring the first assertion KeyErrors.

#### Step 3 -- Committed curated gender supplement for the 12 names no source tier can reach

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\config\source_banks\shakespeare\roster_gender_supplement.json`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_roster_gender.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_roster_gender.py`

**Change:** Add ONE committed data file keyed `{play_code: {SPEECH_PREFIX_UPPER: {"gender": "male"|"female", "evidence": "<quotable reason>"}}}` covering the 12 measured unresolved cast_hints: Antipholus, Dromio, Horatio, Bottom, Lysander, Demetrius, Toby, Don Pedro (Helena, Hermia female; Ariel and Puck are the operator call in open_questions). Add `apply_gender_supplement(characters, play_code, supplement)` to nodes/_otr_roster_gender.py and a cached loader.

MERGE RULE, enforced not documented: a supplement entry may ONLY fill an entry whose sidecar gender is 'unknown'. It may NEVER overwrite a confirmed male/female sidecar fact -- a supplement that tries raises at load. Every entry must carry a non-empty `evidence` string.

This REPLACES the original Steps 3 (address-form regex), 6 (manifest cast_hint renames) and 7 (LLM tiebreak) outright.

**Why:** PROPORTIONALITY CALL, overruling three reviewers who converged on 'specify the address-form tier as same-line-only'. Grounded: (a) reviewer 3 measured that even the corrected tier adds +0 pinned slots at the shipped num_characters=2 (workflows/otr_canonical.json node 1 widget[2] = 2, and all 94 published adaptation ledgers ran num=2); (b) the whole-play text the tier needs is NOT ON DISK -- config/source_banks/_corpus/ holds 64 Gutenberg PROSE bodies and exactly 1 .provenance.json, zero Folger plays, and the 14 vendored Shakespeare .txt files are SLICED SCENES; (c) `body` is rebound to the sliced scene at scripts/otr_fetch_public_domain.py:325, one line after `parse_character_roster(body)` at :324, so the plan's own placement defeated its own rationale; (d) re-vendoring needs the network. Twelve committed lines with quotable evidence are offline, byte-stable, deterministic, reviewable as a data diff, survive a re-vendor (the vendor rewrites sidecars, not this file), and close exactly the published ANTIPHOLUS and DON PEDRO defects the track was opened to fix -- which no heuristic or LLM tier reaches. It also drops the broadcast-facing 'Don Pedro'->'Prince' and 'Puck'->'Robin' renames entirely: both reviewers proved they buy zero gender resolution while changing a name the audience hears.

**Proves it worked:** Corpus test in tests/test_roster_gender.py: for every scene in curated_scenes.sample.json, every cast_hint resolves through the ladder + supplement to gender in {male, female} (today 30/42; the test asserts the full set and fails on any regression). Second test: a synthetic supplement entry that contradicts a CONFIRMED sidecar gender (e.g. ADRIANA -> male) raises. Delete the supplement file and the first test fails on 12 named hints.

#### Step 4 -- Pin the source gender in the ensemble WITHOUT touching the allocator, and exempt source-named slots from the rename

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_casting.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\OTR_LedgerScriptWriter.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_otr_casting.py`

**Change:** (i) `CastSlot` (nodes/_otr_casting.py:1171-1180) gains TRAILING `source_owned: bool = False` -- verified safe, it is only ever constructed with keywords. `EnsembleSlot` (:535-552) gains TRAILING `source_owned: bool = False` AFTER `age_band` -- mandatory, because tests/test_cast_llm_naming.py:141 and :151 build it with FIVE POSITIONAL args and a mid-dataclass insert shifts them. `assemble_pre_locked_rows` sets `source_owned=True` on exactly the slots whose name came off `source_queue.pop(0)` (:1310-1311).

(ii) `precompute_ensemble_slots` (:683) gains `gender_by_name: Mapping[str,str] | None = None`. THE CALL TO `_plan_gender_distribution` IS UNCHANGED -- same `len(open_slots)`, same `prior_genders`, same rng. After the roll, override `genders[i]` for slots where `slot.source_owned` and `gender_by_name.get(slot.name.upper())` is male|female. Keys are UPPER-CASED on both sides: `_adapt_names` are title-case ('Malvolio','Maria' -- confirmed in a published ledger as `_adaptation_character_names: ['Romeo','Juliet']`) while `assemble_pre_locked_rows` uppercases at :1296.

(iii) `_repair_ensemble_names` (:646-680) skips any `source_owned` slot.

(iv) `lock_cast` (:1468) gains `source_character_genders: dict[str,str] | None = None`, forwards it, and stamps ONE public meta key `cast_source_contract` = {character_names, source_bank_id, gender_by_name, evidence: {NAME: {gender, evidence, tier, roster_name}}} beside `cast_voice_slots` (:1780).

(v) In the writer, inside the EXISTING `propagate_adaptation_cast` block at nodes/OTR_LedgerScriptWriter.py:3807-3815 (verified the correct gate: nodes/story_packs/banks.json sets it true for exactly public_domain and shakespeare; `style_pool_class` is the visual axis and stays untouched), build the map by running Step 1's resolver + Step 3's supplement over `(meta.get('source_meta') or {}).get('characters')` for each name in `_adapt_names`, then pass `source_character_genders=` at the lock_cast call beside line 4047.

**Why:** This is the single decision point -- `cast_one_character` sets `gender=slot.gender` (:1150-1157) and `lock_cast` writes `"gender": response.gender` (:1635), so the slot gender IS the row gender IS the audible voice. Overriding in place rather than re-allocating is the whole design. Probed on the real module: leaving `_plan_gender_distribution` alone keeps the rng call count AND the post-call stream IDENTICAL on all 200 seeds, keeps the UNPINNED slot's distribution exactly as today (HORATIO: 200 male / 200 female both ways), and keeps the 'other' incidence at num=3 byte-identical ({female 133, other 136, male 131} both ways). The reviewers' proposed `count=len(unpinned)` fix does none of those three things. Unknowns are simply not pinned, so `CastingResponse._gender_in_set` (:217-225) is never handed a value outside {male,female,other}.

**Proves it worked:** tests/test_otr_casting.py, four assertions, each of which FAILS with this step reverted: (a) 200-seed sweep of `lock_cast(source_character_names=['Malvolio','Maria'], source_bank_id='shakespeare', source_character_genders={'MALVOLIO':'male','MARIA':'female'})` asserting both row genders correct on EVERY seed -- measured, today's code is wrong on 106/200 seeds; (b) a 400-seed sweep with MARCELLUS pinned male and HORATIO unpinned, asserting HORATIO is NOT deterministically one gender -- this is the guard against the forced-opposite regression every reviewer found, and it fails on their proposed fix too; (c) a source_owned slot named 'TOBY' rolled female keeps the name 'TOBY' -- measured, today it is renamed away on 133/400 seeds (to ERIN, MALI, ANYA, MARGOT); (d) C7 pin: `gender_by_name=None` yields byte-identical ensembles AND identical rng state across seeds (1,42,777,100003) x counts (1,3,6), extending test_source_names_none_is_byte_identical_to_pool at :169. Plus tests/golden/cast_pool_baseline.json must reproduce unchanged.

#### Step 5 -- Give the replay the writer's inputs, and replace the tautological invariant with one that fires

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_casting.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\cast_lock.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_cast_voice_replay_parity.py`

**Change:** Add `replay_voice_assignment_detail(*, cast_seed, num_characters, lemmy_hit, source_character_names=None, source_bank_id=None, gender_by_name=None) -> tuple[dict, dict]` returning (voices, reconstructed_gender_by_char_id). `replay_voice_assignment` keeps its EXACT current signature-plus-defaults and its EXACT `{char_id: voice_preset}` return, delegating to the detail function -- mandatory, because four of the five parity tests compare that dict by direct equality (tests/test_cast_voice_replay_parity.py:74, :89, :103, :116). NO out-parameter, NO widened return.

The detail function forwards `source_character_names` + `source_bank_id` into `assemble_pre_locked_rows` (:1035-1037) and `gender_by_name` into `precompute_ensemble_slots` (:1040), so it RECONSTRUCTS the writer's ensemble rather than being handed the answer.

In `CastLock._assign_bark_voices` (nodes/cast_lock.py:304-369) read the PUBLIC `meta['cast_source_contract']` (character_names + gender_by_name) and `meta['source_bank']` (verified stamped at OTR_LedgerScriptWriter.py:3545, e.g. 'shakespeare_v2', normalized by `base_source_bank_id`). Add `_assert_replay_gender_agrees(cast, reconstructed_genders)` beside `_assert_unique_bark_voices` and call it at :368.

THE INVARIANT COMPARES THE REPLAY'S RECONSTRUCTED GENDER TO THE ROW GENDER -- never to `meta['cast_voice_slots'][cid]['gender']`, which `lock_cast` copies verbatim from the row at nodes/_otr_casting.py:1758-1759 and which therefore can never fire. It asserts the picker's gender INPUT, not the resulting preset's gender, because `python_assign_voice_preset:968-972` legitimately falls back to the whole pool when a gender column is exhausted (bark ships only 4 female presets: v2/en_speaker_2/4/7/9).

**Why:** Measured live at cast_seed 424242 with names ['Antipholus','Dromio','Adriana']: the writer's ensemble is (ANTIPHOLUS male, DROMIO other, ADRIANA female) and today's replay is (ERIN MARTIN female, FABER SATO other, KANE SIRIKIT male) -- the writer pops source names for zero rng draws while the replay burns `pick_first_last` draws before the gender shuffle. Probed: forwarding source_character_names + source_bank_id restores the match exactly. Because Step 4 does not change the draw count, a missing gender map now degrades to 'wrong bark preset', never to a desynchronized stream -- which is why this step is separable and does NOT gate the live leg. Reviewers are right that this is LATENT: `_assign_bark_voices` writes only `row['voice_preset']` (cast_lock.py:355-361), and node 80's shipped indextts2 takes the audible ref from the ROW gender at cast_lock.py:563.

**Proves it worked:** New test in tests/test_cast_voice_replay_parity.py: `lock_cast` with source_character_names=['Antipholus','Dromio','Adriana'], source_bank_id='shakespeare', cast_seed=424242, then `CastLock._assign_bark_voices`, asserting the reconstructed per-slot gender equals the writer's row gender. It FAILS on today's code (measured divergence above) and cannot be satisfied by copying the answer, because the reconstruction never reads the row. The five existing parity tests must stay green UNMODIFIED -- name `test_replay_matches_committed_golden_baseline` (:106) plus tests/golden/cast_pool_baseline.json as the real byte-identity anchor; the other three are `stamped == replay` where `stamped` is produced BY the replay and are tautologies.

#### Step 6 -- Separable: stop an uncastable row from silently drawing a gender-blind reference

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\cast_lock.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_cast_lock_voice_ref_completeness.py`

**Change:** When `assign_voice_for_slot` raises for a non-google engine (nodes/cast_lock.py:616-620), the row is reported 'NOT cast' and `continue`d -- no `_stamp`, no `voice_ref_id` -- and `_otr_voice_node_common.py:109-127` then picks ANY ref for the engine keyed on char_id. Make CastLock STAMP that same deterministic gender-agnostic reference itself and record `voice_cast_fallback='gender_unservable'` on the row, so the ledger names the voice that actually renders. Do NOT restrict the allocation to {male,female} and do NOT add a hard refusal.

**Why:** LEDGER COMPLETENESS, not a content gate -- `voice_ref_id` currently has no owner on this path, and downstream consumers read fields, not intentions. Verified live: `assign_voice_for_slot(engine='indextts2', gender='other')` raises VoiceCastingError; the bank has 179 entries and exactly one non-binary. This fires whenever `_plan_gender_distribution` yields 'other', which is EVERY episode at num_characters>=3 (probe: all six permutations at count=3 contain 'other'). It has never fired in production -- 0 of 188 published adaptation rows carry gender='other', because every published adaptation ran num_characters=2, and the 32 rows with no voice_ref_id are all `pending_*` writer-stage ledgers, not this path. I am REJECTING reviewer 3's fix ('forbid other on source-owned slots'): probed, Step 4's design leaves the 'other' incidence byte-identical to today, so this is pre-existing and must not be paid for by changing the roll for unpinned slots.

**Proves it worked:** New test: build a cast with a gender='other' row, run CastLock's auto_registry caster against the real bank with char_voice_engine='indextts2', and assert the row carries a non-empty voice_ref_id plus the fallback marker. It fails today (the row ships with voice_ref_id=None). Add the same assertion -- every non-ANNOUNCER row has a voice_ref_id -- to the Step 4 live leg.

### Risks

- Adaptation episodes stop being byte-comparable to previously published ones, BY DESIGN: `assign_voice_for_slot` folds gender into `stable_cast_seed(..., gender=gender, ...)` (nodes/_otr_voice_bank.py:392-402), so a corrected gender re-draws the IndexTTS2 reference. The invention lanes (original, scifi_news, media_archive, scifi_news_pro) are proven untouched -- probe: identical ensembles AND identical rng state with gender_by_name=None across seeds x counts, plus the committed golden baseline.
- `source_meta` is copied wholesale into durable ledger meta, so adding `characters` (about 7 small dicts per scene) appears in every adaptation ledger diff. `_copy_sidecar` (nodes/_otr_source_payload.py:195-203) is a SHALLOW `dict(sidecar)`, so the objects `load_roster_characters` returns alias straight into meta and are json-dumped -- they MUST be plain dicts, never MappingProxyType or a frozen dataclass.
- Source-snapshot replay: envelopes captured before Step 2 carry no `characters` key (nodes/OTR_LedgerScriptWriter.py:1677-1694 replaces source_meta wholesale), so a replayed bake-off leg falls back to the roll while a live fetch of the same source_ref pins. PARTIALLY MITIGATED by keying the Step 3 supplement on `play_code`, which IS present in pre-change envelopes (53 of 94 published adaptation ledgers carry it). Sidecar-derived pins still degrade. Decide per bake-off whether to re-capture; the degradation is graceful, never an error.
- The public_domain lane gets NOTHING from Steps 1-5 and the code path is inert there. Confirmed: exactly ONE prose sidecar exists (time_machine__arrival) against 64 raw bodies in config/source_banks/_corpus/, it carries zero characters because prose has no dramatis personae, and `PublicDomainBriefs.character_names` (nodes/_otr_public_domain_sources.py:570) is LLM-extracted per run so its cast names are not stable across runs. Claiming otherwise would overstate the fix.
- Bark's female column is only 4 presets (v2/en_speaker_2/4/7/9, config/cast_pools.py:262-269). Pinning source gender makes column exhaustion more likely on female-heavy scenes (much_ado act3 scene1 is BEATRICE/HERO/URSULA/MARGARET). `python_assign_voice_preset:968-972` falls back to the whole pool by design, so Step 5's invariant MUST assert the picker's gender INPUT, not the resulting preset's gender, or it hard-stops legitimate episodes. IndexTTS2 (23 female refs) is unaffected.
- The sidecar records collective and generic speech prefixes ('ALL' in macbeth and midsummer, 'PRINCE', 'BOY', 'LUCE', 'ROBIN', 'TOBY' with gender_source='absent_from_roster') as characters with gender 'unknown'. They inflate the reported unknown count from a real 25 to 32 and can never resolve from the roster. Not a regression, but it caps the resolution rate.
- Anything left unknown after Step 3 falls to today's roll. That is NO WORSE than today by construction -- probed -- but it must not be described as fixed. HELENA/HERMIA/LYSANDER/DEMETRIUS carry roster_name but EMPTY descriptions in midsummer__act3_scene2.provenance.json, so the parser FOUND them and the fetched Folger text simply did not gender them; the supplement is the only offline route to those four.

### Tests

- Baseline before touching anything (established green, 182 passed in 4.22s): pytest -q -p no:cacheprovider tests/test_cast_voice_replay_parity.py tests/test_otr_casting.py tests/test_character_roster.py tests/test_shakespeare_sources.py tests/test_public_domain_sources.py tests/test_cast_llm_naming.py tests/test_cast_voice_slots.py
- tests/test_roster_gender.py (new) -- the ladder against the REAL shipped sidecars: MALVOLIO exact/male; ANTIPHOLUS qualified/1-candidate/UNKNOWN; DROMIO qualified/2-candidates/UNKNOWN (pins the corrected fact so the false 'they agree he is a man' claim cannot return); PUCK and DON PEDRO tier=none; a synthetic disagreeing pair -> ambiguous_join.
- tests/test_roster_gender.py corpus test -- every one of the 42 shipped cast_hints in config/source_banks/shakespeare/curated_scenes.sample.json resolves to male|female after the supplement (30 do today), and a supplement entry that contradicts a confirmed sidecar gender raises. Assert resolved GENDER, never merely a non-'none' tier.
- tests/test_otr_casting.py (extend) -- the four Step 4 assertions, each failing with Step 4 reverted: the 200-seed MALVOLIO/MARIA pin sweep (today wrong on 106/200); the 400-seed HORATIO-unpinned distribution guard (catches the forced-opposite regression in both the original plan and the reviewers' proposed fix); the TOBY no-rename case (today renamed on 133/400 seeds); and the C7 pin extending test_source_names_none_is_byte_identical_to_pool at :169.
- tests/test_cast_voice_replay_parity.py (extend, do not modify the existing five) -- lock_cast at cast_seed 424242 with source names ['Antipholus','Dromio','Adriana'] then CastLock._assign_bark_voices, asserting the RECONSTRUCTED replay gender equals the row gender. Fails on today's code (writer male/other/female vs replay female/other/male, measured).
- tests/golden/cast_pool_baseline.json unchanged + test_replay_matches_committed_golden_baseline (:106) -- the ONE real byte-identity anchor for the invention lanes. The other three parity tests are `stamped == replay` where stamped is produced by the replay and prove only the stamping loop.
- tests/test_shakespeare_sources.py + tests/test_public_domain_sources.py (extend) -- source_meta carries `characters` for a scene with a sidecar, and the key is ABSENT (not []) for the fixtures directory, which genuinely contains no .provenance.json.
- LIVE PROOF LEG (after Step 4; Step 5 does not gate it) -- one shakespeare render through workflows/otr_canonical.json on folger-twelfth-night:act2-scene5-malvolio-letter. On the produced ledger assert: MALVOLIO gender=male with an indextts2 MALE voice_ref_id; MARIA gender=female with a female ref; meta.cast_source_contract.evidence names the sidecar evidence for both; EVERY non-ANNOUNCER row carries a voice_ref_id; RESULT SUCCESS and obs_publish OK, asset present at its canonical otr/episodes path.

### Open questions

- ARIEL and PUCK/ROBIN have no roster gender and no defensible textual tier. Should the Step 3 supplement assign them (Folger's stage directions use 'he' for both), or leave them to the roll and accept 2 of 42 hints unresolved? This is the only genuinely editorial entry in the data pass and it is an operator call.
- The `num_characters` widget still caps the cast at 2, and every one of the 94 published adaptation ledgers ran num_characters_request=2. A 7-speaker scene rendered at 2 drops five of the source's people. Correct gender for two survivors is still a truncated scene -- is expanding the cast to the passage's speakers in this track or a separate one? Note it interacts with Step 6: num_characters>=3 is exactly when the un-castable 'other' gender starts firing.
- Should cast_hints be retired in favour of the sidecar's `speakers` list as the slot-name authority (docs/2026-08-03-fidelity-pass-ownership.md row 4 decided this in principle)? It would make the join EXACT ('ANTIPHOLUS OF EPHESUS' rather than 'ANTIPHOLUS') and delete the whole ambiguity class, but it changes visible character names and collides with the count-match invariant at nodes/OTR_LedgerScriptWriter.py:4119, which hard-raises when locked != requested.
- The public_domain lane gets nothing here. Its one prose sidecar carries zero characters and `PublicDomainBriefs.character_names` is LLM-extracted per run, so its cast names are not even stable across runs. Does that lane need its own gender answer in this track, or does it wait for the prose-form decision the fidelity doc leaves open?
- Should collective and generic speech prefixes ('ALL' in macbeth and midsummer, 'PRINCE', 'BOY') be excluded from the sidecar `characters` list at vendor time, or recorded and filtered at the join? They can never resolve and currently inflate the reported unknown count from a real 25 to 32.
- Bake-off source snapshots captured before Step 2 replay without `characters`. The play_code-keyed supplement still applies, but sidecar-derived pins do not, so a pack bake-off would compare a partially-pinned cast against a fully-pinned one. Re-capture the envelopes, or accept the documented degrade?


## TRACK: voice-variety

**Summary:** Two defects are real and production-proven; everything else in the original design is either scope or a misdiagnosis. (1) The kokoro announcer is pinned to bm_george on every episode -- `announcer_voice_ref` (nodes/_otr_voice_bank.py:717) routes only ("google_tts","chatterbox","dia") through the seeded mixer, probe returns bm_george 40/40 seeds. (2) `_ladder_pick`'s `if pool: return _pick(pool)` (nodes/_otr_voice_bank.py:432-433) accepts a tier of ONE, which is a 100% deterministic pin -- measured first-accepted-tier sizes: (indextts2,female,warm)=1 -> vz_caro_davy, (kokoro,male,warm)=1 -> am_michael, (kokoro,male,deep)=1 -> am_fenrir. My hardening changes the plan in four material ways. FIRST: the announcer fix needs CODE, not tags. I applied the original plan's exact data edit in memory and ran the UNCHANGED selector over 200 seeds: {bm_fable:97, bf_emma:103}, bm_george unreachable -- because `_seeded_preferred_announcer_voice_ref` seeds only the GENDER then takes `sorted(pool,key=_key)[0]` (:702). Replacing that tail with a seeded draw over the _key-sorted pool yields all four ids (~50/54/49/48 over 200 seeds) and leaves chatterbox/dia/google_tts byte-identical (probe: pick sequences equal over 60 seeds, because each has exactly one preferred announcer per gender and choice() over a 1-list is the identity). SECOND, and this is the biggest correction to both the original plan and both prior hardenings: THE SYNONYM TABLE IS NOT NEEDED TO FIX THE DEFECT AND IS NOT PART OF THE MINIMUM FIX. Measured count of size-1 first-accepted tiers across all 24 (engine x gender x writer-word) combos: today 3, synonyms alone 5 (it MANUFACTURES pins: indextts2/male/sharp -> vz_pd_librivox_phil_chenevert, indextts2/male/deep -> vz_bill_boerst, indextts2/male/dry -> vz_peter_yearsley, kokoro/female/sharp -> bf_emma), floor=2 with NO synonyms 0, floor=2 with synonyms 0. So the floor alone is sufficient and the "steps 5+6 must merge" constraint dissolves in one direction only: floor first is safe and independently green; synonyms first is a regression. Synonyms therefore become an OPTIONAL last step, after the floor, on a match-only field. THIRD: floor=2, not 3. Measured combos that still honour timbre (first accepted tier above the gender-only tier): floor1 8/24, floor2 5/24, floor3 2/24. Floor 3 buys its extra spread by deleting the timbre dimension for 22 of 24 combos; floor 2 removes every 100% pin while keeping 5 working timbre tiers. FOURTH: all percentage acceptance thresholds are struck. Two independent panel reconstructions of the original simulation disagreed 2x on the baseline, and the proposed per-slot<20% criterion is unsatisfiable at floor=2 by construction. Acceptance is re-anchored on a structural invariant that is deterministic and fails today on exactly three combos: no first-accepted tier may have size 1. No widget change, no INPUT_TYPES change, one additive JSON-schema field, and workflows/otr_canonical.json is untouched (node 80 ['default','auto_registry',True,'indextts2','kokoro','cuda'], node 81 ['indextts2'], node 82 ['kokoro'] all stay). Baseline confirmed green: 94 passed across test_voice_bank, test_kokoro_char_voices, test_announcer_voice, test_cast_lock, test_voice_bank_coverage, test_cast_voice_replay_parity, test_hybrid_voice_fit.

### Steps

#### Step 1 -- Unpin the announcer -- seeded WITHIN the gender, bm_george stays reachable

**Files:**
- `nodes/_otr_voice_bank.py`
- `config/voice_reference_bank.json`
- `nodes/cast_lock.py`
- `tests/test_voice_bank.py`
- `tests/test_cast_lock.py`

**Change:** CODE FIRST. In `_seeded_preferred_announcer_voice_ref` (nodes/_otr_voice_bank.py:666-702) replace the tail `return sorted(pool, key=_key)[0]` at :702 with `ordered = sorted(pool, key=_key)` then `return random.Random(hashlib.sha1(("%s_announcer_pick:%s" % (engine, episode_seed if episode_seed is not None else "")).encode("utf-8")).hexdigest()).choice(ordered)`. Both `hashlib` and `random` are already imported at nodes/_otr_voice_bank.py:28-32; leave the gender-selection block at :687-694 exactly as it is. WIRING: hoist `_SEEDED_ANNOUNCER_ENGINES = ("google_tts", "chatterbox", "dia", "kokoro")` beside CASTING_POLICY_VERSION (:43), use it at :717, and import it into nodes/cast_lock.py to replace the duplicated literal tuple at :527 (the two must never drift). DATA: in config/voice_reference_bank.json add "announcer_voice" to the `roles` of the kokoro rows bm_fable/bf_emma/bf_lily, and `"style_tags": ["preferred_announcer", "british_leaning"]` to those three plus bm_george -- reproducing ANNOUNCER_VOICE_POOL (nodes/_otr_audio_engines/eng_kokoro.py:26). TESTS in the same commit: rewrite tests/test_voice_bank.py:226 (`announcer_voice_ref("kokoro").voice_ref_id == "bm_george"`) to the in-set form already used for chatterbox/dia at :227-233, and rewrite tests/test_cast_lock.py:252 and :286 -- both hard-assert `cast["a1"]["voice_ref_id"] == "bm_george"` and both go red. DO NOT touch the `not allow_voice_reuse` clause at nodes/cast_lock.py:535-537: it only fires when target_engine == announcer_engine, and the shipped workflow is indextts2 chars + kokoro announcer, so no character can be cast the announcer's voice today. That edit is latent hardening that also re-bases chatterbox/dia casting whenever allow_voice_reuse is True (node 80 ships True) -- out of scope for this track.

**Why:** The loudest 'same voice every night' symptom, proven in production (six `announcer_voice: ... voice_ref_id=bm_george ... engine=kokoro` lines across three legs in tmp/_render_320b_server.log) rather than inferred. First because it needs no CASTING_POLICY_VERSION bump and touches no caster code. The code change is mandatory and the original design's data-only version is provably wrong.

**Proves it worked:** New test in tests/test_voice_bank.py: `{announcer_voice_ref('kokoro', episode_seed=s).voice_ref_id for s in range(64)} == {'bm_george','bm_fable','bf_emma','bf_lily'}`. It fails today (probe: seeds 0-39 all return bm_george) AND fails under the data-only variant the original plan proposed (probe: 200 seeds -> {bm_fable:97, bf_emma:103}, bm_george count 0), so it discriminates the correct fix from the wrong one. Companion assertion in the same test: the pick sequences for chatterbox, dia and google_tts over seeds 0-59 are unchanged (probe: identical), which fails if the seeded draw is applied before the `_key` sort instead of after it.

#### Step 2 -- speaker_id -- two files of ONE human cannot be cast as two characters

**Files:**
- `config/voice_bank_entry_schema.json`
- `nodes/_otr_voice_bank.py`
- `config/voice_reference_bank.json`
- `scripts/otr_ingest_pd_voices.py`
- `tests/test_voice_bank.py`

**Change:** Add an optional `speaker_id` string property to config/voice_bank_entry_schema.json (`additionalProperties: true` confirmed at :7, so every existing row still validates and it is declarative only). Add `speaker_id: str = ""` to the `VoiceBankEntry` dataclass immediately after `provider_voice_id` (nodes/_otr_voice_bank.py:101) -- verified safe: all four construction sites are keyword-based or `VoiceBankEntry(**{**base.__dict__, ...})` (tests/test_voice_bank.py:427), so no positional call breaks. Read it in `_entry_from_dict` (:165-179) as `speaker_id=str(d.get("speaker_id") or "")` and emit `f"speaker:{speaker_id.strip().lower()}"` from `voice_ref_usage_keys` (:333-347) when non-empty. No other call site changes: `_entry_is_used` (:350-351), `validate_voice_proposal` (:583) and both accumulators (nodes/cast_lock.py:532-533 and nodes/_otr_casting.py:1718) all intersect against that key set. SCOPE: a probe found 40 ref_paths already shared by the vz_/cb_/dia_ mirrors of identical files, so mirror-stamping identical-file rows is redundant -- but stamp BOTH mirrors of the two genuinely-different-file same-human pairs, because within one engine those rows collide: `speaker_id: "librivox_mark_f_smith"` on vz_/cb_/dia_pd_librivox_mark_f_smith and vz_/cb_/dia_pd_librivox_mark_f_smith_elder (6 rows; scripts/otr_ingest_pd_voices.py:72-74 says in its own comment 'Same narrator, his RESONANT/grandfatherly delivery'), and `speaker_id: "ljspeech_linda_johnson"` on vz_/cb_/dia_ljspeech and vz_/cb_/dia_pd_ljspeech_linda_johnson (6 rows; scripts/otr_ingest_pd_voices.py:3 states 'LJSpeech is a single narrator'). Twelve rows total. Add `speaker_id` passthrough to `build_entries` (scripts/otr_ingest_pd_voices.py:149-165) so future ingests carry it.

**Why:** Two distinct ref_paths for one narrator can both be cast in one episode today, in every one of the three cloner engines independently. It is also the hard prerequisite for step 3 -- three takes of the operator are three different files of one human, exactly the case the existing ref_path key cannot catch.

**Proves it worked:** New test asserting `voice_ref_usage_keys(smith) & voice_ref_usage_keys(smith_elder)` is non-empty for the indextts2 pair AND for the chatterbox pair (today the intersection is empty -- different ids, different ref_paths, no provider id), plus an end-to-end assertion that `assign_voice_for_slot` with `used_voice_ref_ids=voice_ref_usage_keys(smith)` and `allow_voice_reuse=False` never returns smith_elder. Both fail without the field.

#### Step 3 -- Register the operator's three recordings as cloner refs

**Files:**
- `scripts/otr_ingest_operator_voices.py`
- `config/voice_reference_bank.json`
- `tests/test_voice_bank.py`

**Change:** New script scripts/otr_ingest_operator_voices.py, structurally a copy of scripts/otr_ingest_pd_voices.py (reuse `_sha256` :141-146, `build_entries` :149-165, `merge_into_bank` :168-185, `_CLONER_ENGINES` :47, `_REF_REL_DIR` :48) but with PINNED segment marks instead of the LibriVox skip-25/take-25 heuristic in `_normalize_clip` (:104-112). Source the three files confirmed on disk at C:\Users\jeffr\Downloads\voices\ (mr_jeffrey_uk.m4a, mr_jeffrey_uk_expressive.m4a, mr_jeffrey_usa.m4a). Per voice: `ffmpeg -y -ss <A> -to <B> -i <src.m4a> -ac 1 -ar 24000 -sample_fmt s16 -af highpass=f=80,lowpass=f=12000,loudnorm=I=-23:LRA=7:TP=-2 <dest.wav>` with (A,B) = uk (33.616, 51.369), uk_expressive (14.721, 35.043), usa (6.122, 24.296), pinned as literals so the artifact sha is reproducible without re-running silencedetect across ffmpeg versions. Write to C:\ComfyUI-Models\TTS\refs\indextts2\vz_mr_jeffrey_{uk,uk_expressive,usa}.wav, overwriting the unpinned 2026-06-18 cuts that no bank row references. Merge 9 rows (3 voices x vz_/cb_/dia_): gender male, age_band adult, roles [char_voice], commercial_clean true, `speaker_id: "operator_jeffrey"` on all nine, timbre from the bank's own descriptive vocabulary (warm for uk, bright for uk_expressive, clear for usa -- accent is the identity difference, not timbre). DO NOT bump `voice_bank_id`: `load_voice_bank` (nodes/_otr_voice_bank.py:189-213) reads only `data.get('voices')` and hashes the file TEXT, never that field; grep finds no Python reader; and the ledger's `meta['voice_bank_id']` is a DIFFERENT value (the CastLock widget string, stamped at nodes/cast_lock.py:212 and asserted as 'elevenlabs_cloud' at tests/test_cloud_elevenlabs_cast.py:154). The bump is inert and invites a later 'consistency' edit that breaks that test.

**Why:** The recordings exist, are already mono 24 kHz on disk, and nothing in the repo references them -- confirmed by grep (zero bank rows match 'jeffrey'). The pinned recut replaces an artifact whose provenance cannot be reproduced. `speaker_id` from step 2 keeps all three from landing in one episode.

**Proves it worked:** Two assertions. (a) Ref-integrity: the three new vz_ rows resolve to an existing file through `_resolve_ref_to_disk` (nodes/_otr_voice_node_common.py:39-65, whose candidate ladder includes the C:\ComfyUI-Models root at :60) and their `ref_sha256` matches the bytes on disk -- fails if the script was not run, wrote to the wrong root, or the marks drifted. (b) Card-window byte-identity: `build_voice_cards('indextts2','male')` and `('indextts2','female')` return identical `descriptor` and `voice_ref_id` lists before and after the add (probe today: True, because the 12-card male window is saturated by vz_bill_boerst + eleven vz_donor_* ids and vz_mr_jeffrey_* sorts after all of them). This fails the moment a new row lands inside the alphabetical 12-card slice and moves the live voice-fit prompt.

#### Step 4 -- A tier of one is not a choice -- minimum tier pool, with the floor folded into the seed

**Files:**
- `nodes/_otr_voice_bank.py`
- `tests/test_voice_bank.py`
- `tests/test_voice_variety.py`

**Change:** THE fix for 'same voices every night', and it is self-contained. (a) Add `_MIN_TIER_POOL_DEFAULT = 2` and `def _min_tier_pool(): return max(1, int(os.getenv("OTR_CAST_MIN_TIER_POOL", _MIN_TIER_POOL_DEFAULT)))` beside the existing `OTR_CAST_WEIGHTED` A/B escape (nodes/_otr_voice_bank.py:416). (b) In `_ladder_pick` (:424-434) change `if pool:` to `if pool and (len(pool) >= floor or dims == _LADDER[-1])`, hoisting `floor = _min_tier_pool()` above the loop. `dims == _LADDER[-1]` is a valid frozenset comparison and the gender-only last tier stays always-accepted, so the gender floor and the fail-closed semantics at :441-452 are untouched -- no new VoiceCastingError is reachable. (c) Close the determinism hole: the floor is not in `stable_cast_seed`'s payload (:283-298), so two different byte-outputs could claim one policy string. At the `stable_cast_seed(...)` call inside `assign_voice_for_slot` (:394-402) pass `casting_policy_version=f"{casting_policy_version}+mtp{floor}"`. One line, no signature change, no ledger-stamp change. (d) Bump `CASTING_POLICY_VERSION` (:43) from "2" to "3" -- the deliberate, documented C7 re-baseline -- and update tests/test_voice_bank.py:419. (e) Rewrite the `bank_age` fixture in `test_caster_ladder_drops_age_then_role_then_timbre` (tests/test_voice_bank.py:139-152): its 2-entry warm_elder/bright_adult bank falls through to the gender tier under floor=2 and asserts the old thin-tier behaviour. Grow it past the floor so it still exercises age-drop, and add an explicit new assertion that a size-1 tier now falls through. The two single-entry cases in that same test (`bank_role`, `bank_gender`) still pass via the last-tier escape -- do not touch them.

**Why:** This is the mechanism that made a 40-voice bank sound like three, and the measurement shows the floor alone is sufficient: size-1 first-accepted tiers across all 24 (engine x gender x writer-word) combos are 3 today and 0 at floor=2 with no other change. Floor=2 rather than 3 because floor=3 leaves only 2 of 24 combos honouring timbre at all (vs 5 at floor=2) -- it buys spread by deleting the dimension. Floor 3 stays one env var away for a live A/B.

**Proves it worked:** NEW tests/test_voice_variety.py, threshold-free and deterministic: for every (engine in {indextts2,kokoro}) x (gender in {male,female}) x (word in `_TIMBRE_VOCAB`), the tier `assign_voice_for_slot` accepts against the SHIPPED bank is either the gender-only last tier or holds >= `_min_tier_pool()` candidates. It fails today on exactly three combos -- (indextts2,female,warm)=[1,1,1,23]->vz_caro_davy, (kokoro,male,warm)=[1,1,1,13]->am_michael, (kokoro,male,deep)=[1,1,1,13]->am_fenrir. Companion assertion: a 200-seed sweep over each of those three triples yields more than one distinct voice_ref_id (today: exactly one, 100% of seeds). Monkeypatch `OTR_CAST_MIN_TIER_POOL` and `OTR_CAST_WEIGHTED` so the test never reads whatever the operator last exported.

#### Step 5 -- Kokoro engine correctness: per-lang_code pipelines on a shared model, real role strings, live call site

**Files:**
- `nodes/_otr_audio_engines/eng_kokoro.py`
- `nodes/_otr_voice_node_common.py`
- `tests/test_kokoro_char_voices.py`
- `tests/test_announcer_voice.py`

**Change:** nodes/_otr_audio_engines/eng_kokoro.py:119-123 constructs `KPipeline(lang_code="b", ...)` hardcoded to British English while 20 of the 28 registered kokoro rows are American (af_ 11, am_ 9). Add `_lang_code_for_voice(voice_id)` returning 'a' for af_/am_ else 'b', next to `_kokoro_voice_path` (:51-53); change `self._pipeline` (:84) to `self._pipelines: dict` and add `_pipeline_for(lang_code)`. CONVERT THE CALL SITE: `generate_voice` calls `self.load()` at :184 and invokes `self._pipeline(...)` at :186 -- both must become `pipe = self._pipeline_for(_lang_code_for_voice(voice_id))` then `for _, _, audio_data in pipe(...)`, or the change leaves an AttributeError in the only path that generates audio. Build the second pipeline with a SHARED model: the installed kokoro signature verified in the venv is `KPipeline(self, lang_code, repo_id=None, model: Union[KModel, bool]=True, trf=False, en_callable=None, device=None)`, so pass `model=<first pipeline>.model` -- no second resident copy and the VRAM concern disappears. `unload()` (:125-140) iterates every pipeline and moves each `.model` to cpu. Announcer stays byte-identical: ANNOUNCER_VOICE_POOL (:26) is all b*. SAME COMMIT: fix the role strings -- `generate_voice` raises `EngineUnusable(self.name, "announcer_voice", ...)` at :163-168 and :176-183 even when serving char_voice. Thread the active role by setting `adapter.requested_role = self.ROLE` UNCONDITIONALLY beside `adapter.requested_device` (nodes/_otr_voice_node_common.py:354) -- unconditionally because `register` stores a module-level singleton (`_REGISTRY[inst.name] = inst`, nodes/_otr_audio_engines/registry.py:104) and `get_engine` returns it (:110-113), so a stale attribute from a char render would otherwise survive into the next announcer render in the same ComfyUI process. Finally gate `begin_episode` (:87-104) on the role so a char-only render cannot fail MISSING_MODEL on an announcer voice it will never speak (`_render_per_line` calls begin_episode unconditionally).

**Why:** Correctness, not variety: forcing British phonemization on American voices makes an opened pool sound wrong rather than varied. LABEL IT HONESTLY AS ENABLING WORK -- `char_kokoro_v1.allowed_voice_banks` is [kokoro_builtin] (config/audio_engine_profiles.yaml:154) while the canonical workflow ships voice_bank='default' (node 80) and `_resolve_char_engine` (nodes/cast_lock.py:712-758) filters on that list, so kokoro char_voice is unreachable in production today.

**Proves it worked:** Four assertions in tests/test_kokoro_char_voices.py / tests/test_announcer_voice.py. (i) `_lang_code_for_voice` returns 'a' for every af_/am_ and 'b' for every bf_/bm_ id across all 28 shipped kokoro rows. (ii) With a fake KPipeline injected, `generate_voice(text, 'am_adam', ...)` invokes a pipeline constructed with lang_code='a' -- this is the assertion that goes red if the :184-186 call site is left on `self._pipeline` (AttributeError) or left hardcoded to 'b'. (iii) `begin_episode` with requested_role='char_voice' does not raise MISSING_MODEL when the seeded announcer .pt is absent, and still raises when the role is announcer_voice. (iv) Singleton-leak guard: after a char dispatch sets requested_role='char_voice', an announcer dispatch on the SAME registry instance resets it to 'announcer_voice' -- fails if the attribute is set conditionally or relied on via a getattr default.

#### Step 6 -- Announcer-pool drift guard (replaces the doc-only truth-up)

**Files:**
- `config/cast_pools.py`
- `tests/test_announcer_voice.py`

**Change:** config/cast_pools.py:355 sets `VOICE_REGISTRY["kokoro"]["presets"] = list(ANNOUNCER_PRESETS)` -- 4 announcer voices -- in a module that documents itself as 'the single source of truth' while the real character catalog is 179 bank rows (probe: indextts2 40, kokoro 28, chatterbox 42, dia 42, elevenlabs 21, google_tts 6). Correct the VOICE_REGISTRY docstring and the ANNOUNCER_PRESETS comment at :296-309 to name config/voice_reference_bank.json as the character catalog and this list as the ANNOUNCER pool only. Do NOT touch `VOICE_PROFILES` (:253-287) or `open_voice_pool` (:477-503): they drive `python_assign_voice_preset` and would re-base the SCRIPT-side cast rng for every lane. Then make the comment enforceable: add a drift-guard test asserting `[p for p, _ in cast_pools.ANNOUNCER_PRESETS] == eng_kokoro.ANNOUNCER_VOICE_POOL` and `VOICE_REGISTRY['kokoro']['presets'] == [p for p, _ in ANNOUNCER_PRESETS]`. Verified zero behavioural consumers of VOICE_REGISTRY: only `KNOWN_TTS_MODELS = tuple(VOICE_REGISTRY.keys())` (:362), two comments (nodes/production_ledger.py:1020, nodes/_otr_casting.py:1646), and tests; nodes/_otr_scifi_fable2.py:728-730 reads `_POOLS.VOICE_PROFILES` / `open_voice_pool`, not VOICE_REGISTRY, so the Fable2 menu is unaffected.

**Why:** A reader who follows the 'single source of truth' comment concludes characters have 10 voices, which is how this defect stayed invisible for weeks. A comment alone rots; the drift guard makes the two-copy hazard mechanically detectable. Note that step 1 edits the bank's announcer roles, so the two lists and the bank now describe the same four voices -- exactly the moment to pin the relationship.

**Proves it worked:** The drift-guard test is green on landing (the two lists agree today), so prove it by mutation: temporarily reorder or drop one entry from `ANNOUNCER_PRESETS` in a scratch copy and confirm the test goes red. Its absence is detected the next time someone edits eng_kokoro.ANNOUNCER_VOICE_POOL without config/cast_pools.py -- which is precisely how the pool acquired two copies. Also assert VOICE_REGISTRY's exported values are unchanged by the docstring edit.

#### Step 7 -- Name and pin the two paths steps 1-4 do NOT govern

**Files:**
- `nodes/_otr_voice_node_common.py`
- `nodes/_otr_voice_bank.py`
- `tests/test_voice_variety.py`

**Change:** DOCUMENT + TEST, no behaviour change. (a) GENDER 'other': `_DEFAULT_GENDER_WEIGHTS` is (male .40, female .40, other .20) at nodes/_otr_casting.py:149-153 and the bank has ZERO `gender: other` rows (probe), so `assign_voice_for_slot` RAISES VoiceCastingError for ~20% of slots on both indextts2 and kokoro (probe-confirmed on both). CastLock catches it and logs 'NOT cast' (nodes/cast_lock.py:616-620), and the render then falls to `_resolve_clone_ref_path`'s gender-agnostic `_random.Random(f'{episode_seed}_{char_id}_anyref').choice(cands)` (nodes/_otr_voice_node_common.py:109-127) -- a UNIFORM draw over the engine's role-matching refs, i.e. already the most varied path in the system. Add the explanatory comment at :109-127 so the next reader does not 'fix' it into a narrower draw. (b) HYBRID LLM VOICE-FIT sits IN FRONT of the caster: `hybrid_voice_fit_enabled()` defaults ON (nodes/_otr_casting.py:848-851) and CastLock honours an accepted proposal and `continue`s at nodes/cast_lock.py:595, BEFORE `assign_voice_for_slot` at :603. `build_voice_cards`'s `max_cards=12` (nodes/_otr_voice_bank.py:489) slices the ALPHABETICAL head of 17 male / 23 female indextts2 rows, so 5 male and 11 female refs are unreachable through that path. Give it an owner: hoist `_VOICE_CARD_BUDGET = 12` as a named module constant with a comment stating it is a PROMPT-LENGTH budget, not a variety policy. Do NOT change its value or make the slice a seeded sample in this track -- either rewrites the live voice-fit prompt and re-bases the invention lanes.

**Why:** Both are real, both were absent from the original design, and both bound the claims this track may make. About 20% of characters never reach the caster at all, so no statement about steps 1-4 covers more than 80% of the cast; and a future working LLM proposal would bypass step 4 entirely for any character it lands on.

**Proves it worked:** (i) A test asserting `assign_voice_for_slot(gender='other')` raises VoiceCastingError on BOTH indextts2 and kokoro, and that `_resolve_clone_ref_path` still returns an existing absolute path for such a cast row -- fails if someone narrows the anyref fallback, swallows the raise, or lets the clone engine silently drop to bark. (ii) A CastLock test with the hybrid path ENABLED and a stub generate_fn returning a valid card id: the accepted proposal is stamped and the deterministic caster is not consulted -- fails if the short-circuit at nodes/cast_lock.py:575-595 is removed or reordered, which is the change that would silently undo step 4.

#### Step 8 -- OPTIONAL, AFTER the floor -- one timbre language on a match-only field

**Files:**
- `nodes/_otr_voice_bank.py`
- `tests/test_voice_variety.py`
- `tests/test_hybrid_voice_fit.py`

**Change:** Only after step 4 has landed; NEVER before it. Add `WRITER_TIMBRE_SYNONYMS: dict` (warm <- warm/rich/jovial/relaxed/melodic/soft/alto; sharp <- crisp/clipped/precise/authoritative; deep <- deep/resonant/baritone/bbc; bright <- bright/light/youthful/playful/elegant/tenor; dry <- measured/steady/neutral/calm/refined/narration/smooth/clear) and a NEW dataclass field `timbre_match: Tuple[str, ...] = ()` computed in `_entry_from_dict` (:165-179) as `tuple(dict.fromkeys(list(raw) + sorted({SYN[t] for t in raw if t in SYN})))` -- order-stable and hash-seed independent. ONLY `_score` (:301-311) and `_matches` (:314-323) read `entry.timbre_match`; everything else keeps reading `entry.timbre`. That separation is the whole point: `build_voice_cards` synthesizes the LLM `descriptor` from `[age_band] + timbre + style_tags` (:512-514) -- probe confirms card #1 of the indextts2 male window is 'adult, warm, baritone' -- and that descriptor goes verbatim into the live voice-fit prompt (nodes/_otr_casting.py:864-865) whose result is stored at :1726-1736 and stamped into the ledger, so a naive edit to `entry.timbre` would make it 'adult, warm, baritone, deep' and re-base the invention lanes. Grep `\.timbre` across nodes/ and confirm only `_score`/`_matches` moved. Bump `CASTING_POLICY_VERSION` to "4". Keep the donor refs' existing warm/bright tags -- the original design's 'leave donors untagged rather than fabricate' is a no-op, because all 29 donor rows already carry warm (indextts2 warm n=15) or bright (n=20); there is no evidence those tags are wrong, so keep them and simply drop the fidelity claim built on that framing.

**Why:** The writer stamps one of six words (`_TIMBRE_VOCAB`, nodes/_otr_casting.py:497-504) and the bank speaks ~20 unrelated ones, so 4 of 6 writer words match nothing on indextts2. Fixing that raises how often the caster honours a requested timbre. It is genuinely OPTIONAL and ordered last because the floor already fixed the variety defect without it, and because synonyms shipped ALONE are a regression: measured size-1 first-accepted tiers go 3 -> 5 (new pins: indextts2/male/sharp -> vz_pd_librivox_phil_chenevert, indextts2/male/deep -> vz_bill_boerst, indextts2/male/dry -> vz_peter_yearsley, kokoro/female/sharp -> bf_emma). With step 4 already in place the same synonym table yields 0 size-1 tiers, so the ordering -- not a merged commit -- is what makes it safe.

**Proves it worked:** (i) Card-path byte-identity: `build_voice_cards(engine, gender)` returns identical `descriptor` strings and `timbre` lists for indextts2 and kokoro x male/female before and after -- fails the instant the synonym words leak onto `entry.timbre` and move the voice-fit prompt. (ii) Step 4's no-size-1-tier invariant still holds with synonyms on (measured: 0 at floor 2 + synonyms; 5 with synonyms and no floor) -- this is the assertion that fails if anyone ships this step without step 4 or reverts the floor. (iii) (kokoro, male, 'dry') and (indextts2, female, 'dry') now reach a timbre-matched tier of >= 4 instead of falling to the gender tier -- fails if the synonym map is not consulted by `_matches`.

### Risks

- C7 RE-BASELINE IS INTENTIONAL AND NARROWER THAN THE ORIGINAL PLAN CLAIMED. Steps 1, 3, 4 and 8 change which voice a given (episode_seed, char_id) draws. The SCRIPT side is untouched -- python_assign_voice_preset / open_voice_pool / precompute_ensemble_slots are not modified -- and step 8's synonym words are deliberately kept OFF entry.timbre so build_voice_cards' descriptor, the voice-fit prompt (nodes/_otr_casting.py:864-865) and meta.voice_cast_decision stay byte-identical. The only real re-baseline lever is CASTING_POLICY_VERSION plus the +mtp{n} suffix folded into stable_cast_seed. The JSON voice_bank_id field is NOT a trigger: load_voice_bank (nodes/_otr_voice_bank.py:189-213) reads only data.get('voices') and hashes the file text.
- THE ANNOUNCER CHANGE IS LISTENER-AUDIBLE. Every published episode to date opens with bm_george. After step 1 it is a seeded four-way rotation across bm_george/bm_fable/bf_emma/bf_lily (probe over 200 seeds: 48/49/49/54). That is the intent, but confirm it on the live leg before the track closes -- and note that the ORIGINAL design's data-only version would have retired bm_george entirely, which is the exact failure the new test guards against.
- SIX OTHER PERSONAL RECORDINGS ARE ON DISK AND UNREGISTERED: vz_mr_buck1, vz_mr_derno, vz_mr_good_buck, vz_mr_spacey, vz_ms_dee_2, vz_ms_dee_sci1 in C:\ComfyUI-Models\TTS\refs\indextts2\, all cut 2026-06-18. Provenance and consent are unknown and two of the names read like impressions of real people. Register none of them without the operator naming whose voices they are. This plan registers ONLY the three files the operator recorded of himself.
- ABOUT 20% OF EVERY CAST NEVER REACHES THE CASTER. _DEFAULT_GENDER_WEIGHTS is 40/40/20 including 'other' (nodes/_otr_casting.py:149-153), the bank has zero 'other' rows, and assign_voice_for_slot raises for all of them on both engines (probe-confirmed). They fall to the uniform anyref draw at nodes/_otr_voice_node_common.py:109-127, which has NO used-set -- so two 'other' characters CAN receive the same reference. Every claim about this track's effect describes at most 80% of characters.
- EVERY KOKORO NUMBER DESCRIBES A LANE THE SHIPPED WORKFLOW CANNOT SELECT. char_kokoro_v1.allowed_voice_banks is [kokoro_builtin] (config/audio_engine_profiles.yaml:154) vs char_indextts2_v1's [default] (:89), the canonical workflow ships voice_bank='default' (node 80), and _resolve_char_engine filters on that list. The am_michael / am_fenrir pins are real in the bank but unreachable in production today; step 5 is enabling work, not a shipped win.
- FLOOR=2 STILL ACCEPTS A TWO-CANDIDATE TIER. Measured, three combos land on a size-2 tier after step 4 (indextts2/male/bright, kokoro/female/warm, kokoro/female/bright), i.e. a ~50/50 pick conditional on that gender+timbre. That is a choice, not a pin, and it is the deliberate trade: floor=3 removes them but leaves only 2 of 24 combos honouring timbre at all. OTR_CAST_MIN_TIER_POOL=3 makes it a one-run live A/B, and the floor is folded into the cast seed so the two settings can never both claim policy '3'.
- STEP 4 REWRITES tests/test_voice_bank.py:139-152 and STEP 1 REWRITES tests/test_cast_lock.py:252 and :286. All three encode the OLD behaviour and rewriting them is correct -- the behaviour intentionally changed -- but a reviewer must not read it as tests loosened to make a change pass. Each rewrite must ADD an assertion for the new behaviour, not merely relax the old one.
- COVERAGE GAPS NO RELABEL CAN FIX (compute_bank_coverage, nodes/_otr_voice_bank.py:227-267): every cloner engine is male-light (indextts2 17 male vs 23 female), elder coverage is male-only, `gravelly` (index 5 of _TIMBRE_VOCAB) has zero candidates in every engine even with step 8's synonyms, and there are zero 'other'/androgynous voices anywhere. The age axis is also one word wide: EnsembleSlot.age_band defaults to 'adult' (nodes/_otr_casting.py:552) and precompute_ensemble_slots never sets it (:722-728), so tier 1 is a constant and the 2 elder refs can never win it.
- PER-EPISODE ENGINE ROTATION IS NOT BUILT IN THIS TRACK, and the blockers are recorded so they are not re-derived. pack_audio_batch RAISES ValueError on mixed sample rates (nodes/_otr_audio_engines/base.py:134-140) -- it does not 'mislabel' clips as the original design said; _render_per_line only WARNS at :614-624 before calling it at :626, and resample_audio is never called from the voice node. indextts2 is 22050 Hz (config/audio_engine_profiles.yaml:91) vs kokoro 24000 Hz (:156). generate() resolves ONE adapter and tears it down in finally (:335-379). A 'from_cast' sentinel on build_engine_combo (:209-227) would additionally have to be intercepted BEFORE assert_usable (:346) which fails closed on unregistered names, be scoped to char_voice only because nodes/stable_audio_theme.py:77 calls the same helper for MUSIC, never sort to index 0 because voice_input_types sets default=engines[0] (:272), and it would break tests/test_announcer_voice.py:100 which asserts exact list equality. Node 82 is a third widget it would touch.

### Tests

- Focused suites ONLY (never the full suite), ComfyUI venv `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`, `$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`: tests/test_voice_bank.py, tests/test_voice_bank_coverage.py, tests/test_kokoro_char_voices.py, tests/test_announcer_voice.py, tests/test_cast_lock.py, tests/test_cast_voice_replay_parity.py, tests/test_hybrid_voice_fit.py, tests/test_voice_two_lane.py, tests/test_cast_voice_slots.py, tests/test_engine_profiles.py, tests/test_tts_engine_sidecars.py, tests/test_cloud_elevenlabs_cast.py, tests/test_voice_mixed_rate_resample.py, tests/test_release_gate.py, tests/test_audio_determinism_wrap.py. Baseline measured today: 94 passed across the first seven.
- NEW tests/test_voice_variety.py -- threshold-free, anchored on the deterministic invariant rather than on percentages. Assert (a) for every (engine, gender, writer-timbre) triple the first accepted tier is the gender-only last tier or holds >= _min_tier_pool() candidates, and (b) a 200-seed sweep over each of the three shipped pins -- (indextts2, female, warm), (kokoro, male, warm), (kokoro, male, deep) -- yields more than one distinct voice_ref_id. Both fail today. Monkeypatch OTR_CAST_MIN_TIER_POOL and OTR_CAST_WEIGHTED so the test never reads the operator's shell.
- Do NOT drive any simulation with `_plan_gender_distribution`: its weights are 40/40/20 INCLUDING 'other', and assign_voice_for_slot raises for 'other' on both engines (probe-confirmed). Use an explicit written-down gender shape, and cover 'other' with its own test asserting the raise plus the uniform anyref fallback at nodes/_otr_voice_node_common.py:109-127.
- NEW announcer-mix test mirroring tests/test_voice_bank.py:227-233: across a seed sweep `announcer_voice_ref('kokoro', episode_seed=N)` must return ALL FOUR of bm_george/bm_fable/bf_emma/bf_lily, with an explicit assertion that bm_george is still reachable -- the exact failure the data-only version of the fix produced (probe: {bm_fable:97, bf_emma:103} over 200 seeds). Plus an assertion that chatterbox, dia and google_tts pick sequences are unchanged by the code edit (probe: identical over 60 seeds).
- NEW same-speaker collision test: two entries sharing a speaker_id cannot both be assigned within one cast. Cover both indextts2 AND chatterbox for vz_/cb_pd_librivox_mark_f_smith vs _elder and vz_/cb_ljspeech vs vz_/cb_pd_ljspeech_linda_johnson, and the three vz_mr_jeffrey_* takes.
- NEW Kokoro tests: `_lang_code_for_voice` returns 'a' for every af_/am_ and 'b' for every bf_/bm_ id across all 28 shipped rows; a fake KPipeline proves generate_voice reaches the lang_code-'a' pipeline for an am_* voice (guards the :184-186 call site); the announcer path still resolves 'b' for all four ANNOUNCER_VOICE_POOL voices; begin_episode does not raise MISSING_MODEL for a char_voice render; and requested_role is reset on every dispatch (singleton-leak guard).
- NEW ref-integrity test WITH the sentinel exemptions the original design missed. Measured across the 179 rows: 124 hex shas, 28 'pending' (ALL kokoro), 27 'cloud' (elevenlabs 21 + google_tts 6). Assert every local ref_path resolves on disk and every hex ref_sha256 matches the file, exempting the literals 'cloud' and 'pending' explicitly. Resolve indextts2/chatterbox/dia refs through `_resolve_ref_to_disk` (nodes/_otr_voice_node_common.py:39-65, whose ladder includes the C:\ComfyUI-Models root) and kokoro refs through `_kokoro_voice_path` (eng_kokoro.py:51-53, one root, no ladder) -- that is the resolver each engine actually opens.
- NEW byte-identity guard for the LLM card path (step 8's gate): assert build_voice_cards returns identical descriptor and timbre values before and after the synonym table lands. Probe today gives 'adult, warm, baritone' for indextts2 male card #1; the naive design would have made it 'adult, warm, baritone, deep'.
- Determinism proof for steps 1, 3, 4 and 8: run the announcer sweep and the variety sweep twice in SEPARATE processes (hash randomization differs) and assert byte-identical pick sequences. The step-8 tag merge must go through dict.fromkeys + sorted, never raw set iteration.
- Invention-lane byte-identity check before step 4 lands and again before step 8 lands: freeze one `original` and one `scifi_news` ledger through the writer, apply the change, re-freeze, and diff the WHOLE ledger JSON -- including meta.voice_cast_decision (candidate_ids / proposed_id / accepted_id / fallback_reason), not just voice_preset. That field is the one the original design's synonym placement would have moved.
- LIVE proof, the only thing that closes this track: one headless leg through workflows/otr_canonical.json per the section 4/5 reset discipline, then grep the server log for `[OTR voice P-OBS]`. Baseline to beat is tmp/_render_320b_server.log -- 3 legs, announcer bm_george x3 (lines 515/520/897/902/1471/1476), and only 3 distinct character voices across 6 slots (vz_bill_boerst 21, vz_caro_davy 14, vz_donor_glenn 7). Assert the announcer is not bm_george on every seed and that N characters got N distinct voice_ref_ids.

### Open questions

- Does the operator want his own voice cast as arbitrary characters, or reserved for a recurring role? roles: ['char_voice'] means MALVOLIO can be Jeffrey. Giving the uk take 'announcer_voice' would put a genuinely different voice in the announcer slot, but the announcer engine is kokoro (preset-based) and his refs are cloner WAVs, so that needs announcer_voice_engine=chatterbox or dia.
- Are the three takes three VOICES or one voice in three modes? The plan registers three ids sharing speaker_id 'operator_jeffrey', so the caster can pick any one but never two in one episode. If uk and uk_expressive are the same character voice, collapse them and keep the expressive take as an audition reference only.
- Ship floor=2 (recommended: removes every 100% pin, keeps 5 of 8 timbre-honouring tiers) or floor=3 (removes the three remaining 50/50 tiers but leaves only 2 of 24 combos honouring timbre)? OTR_CAST_MIN_TIER_POOL makes it a one-leg live A/B, but the shipped default needs a decision and the floor is folded into the cast seed either way.
- Is step 8 (timbre synonyms) wanted at all? It is the only piece that adds a second vocabulary to maintain and a dataclass field whose sole purpose is to stay off the LLM card path. The variety defect is closed without it; its benefit is that the writer asking for 'deep' actually gets a baritone. Cheap to defer, cheap to add later.
- The 'other'-gender anyref fallback (nodes/_otr_voice_node_common.py:109-127) has NO used-set, so two 'other' characters can be handed the same reference -- and 'other' is 20% of every cast. Fixing it means threading cross-character state into a render-path resolver that has none today. Fix here, or open it as its own item alongside the Track 1 gender work that would reduce how often 'other' is rolled in the first place?
- Should the announcer keep the curated 4-voice British pool (this plan's choice, preserving the deliberate BBC identity at eng_kokoro.py:23-26) or open to all 28 registered Kokoro voices? Four voices with a seeded within-gender draw fixes the pin; 28 would fix it harder at the cost of that identity.
- Does anything justify widening char_kokoro_v1.allowed_voice_banks (config/audio_engine_profiles.yaml:154) from [kokoro_builtin] to [default, kokoro_builtin]? Until then kokoro char_voice is unreachable and step 5 is enabling work only -- but widening it also means an 'auto' resolve could land on kokoro and change the shipped character default. Recommend keeping kokoro behind an explicit selection.
- Whose voices are vz_mr_buck1 / vz_mr_derno / vz_mr_good_buck / vz_mr_spacey / vz_ms_dee_2 / vz_ms_dee_sci1? Nine unregistered personal refs sit on disk; six are unaccounted for. Nothing gets registered until they are named.


## TRACK: TRACK 2 -- PORTRAIT / FACE CONTINUITY (judged, build-ready)

**Summary:** The defect is real and structural, and I re-verified it on the Windows files rather than trusting the reviews. `resolve_object_seed` (nodes/otr_image_gen_dispatcher.py:134-164) derives every scene still's seed from object_id (`still_<beat_id>`) + a per-beat prompt_hash, so one character cannot hold one face by construction (probe, seed_cfg {request_seed:0,mode:request_hash}: portrait c01 -> 1692108723, still_b002 -> 282026089, still_b004 -> 3107977459, still_b006 -> 1932467677), and no registered image engine consumes a reference (live probe: 11 engines, all required_inputs=('text_prompt',), none carrying accepts_reference_image). Three corrections to the original plan are load-bearing and I fold them in: (1) the seed branch as written raises UnboundLocalError -- `base` is assigned at :159, INSIDE the range it was told to precede -- and must land AFTER the mode gate at :160-161 or it also silently breaks the documented mode='fixed' contract; (2) the reference must be resolved BEFORE the cache key at :1000, not at :1103, because the cache-HIT branch `continue`s at :1047 and the anchor could never enter the key; (3) ImageScale.upscale is (image, upscale_method, width, height, crop) -- all five required -- and wrapper_bridge calls fn(**kwargs) (wrapper_bridge.py:401-408) with no fallback, so the original's two-argument scale node is a dead episode, and FluxKontextImageScale (image only) can never share an ordered candidate tuple with it. I also overrule the panel on its single unanimous "fatal": the canonical workflow as saved really does mint ZERO stills (probe: roles_requiring_stills -> frozenset(), derive_image_prompts -> {'version':1,'objects':[]}), but the prescribed fix -- editing node 87's widget -- is wrong. The prior harden grepped for the WIDGET name and missed that profiles set `role_overrides.character_visual`, which config/profiles/widget_mapping.json maps to [OTR_VideoDirector, character_video_model]; ~40 shipped profiles set it to still-consuming engines and scripts/otr_canonical_api_run.py --profile LOADS the real canonical JSON and applies it. The live leg therefore names a profile (otr_w45_still_flat, the cheapest still-consuming lane) and edits nothing. Build order: pin HEAD baselines (1), pure-CPU identity work (2-4), inert capability + plumbing (5-6), the two engine wirings (7-8), then the live A/B (9). Steps 1-6 are byte-inert on rendered pixels; step 7 is the first pixel change.

### Steps

#### Step 1 -- Pin HEAD baselines BEFORE any source edit (test-only, zero behavior change)

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_still_spine_helpers.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_image_engine_c2.py`

**Change:** Add literal digests captured from HEAD, not from post-change output. (a) request_cache_key('character_video','still_b002','phX',123,'z_image_turbo','1',kind='scene_character',w=1472,h=832) == 'e8e5a5dcacd093e8db11220d218e477a2656763056bb1faca3a61d9f5539f17d' (I verified this literal by probe against otr_image_gen_dispatcher.py:119-131 at HEAD). (b) resolve_object_seed pins for the paths step 2 must not move: under {'request_seed':0,'mode':'request_hash'} object 'c01' with prompt_hash 'pp_portrait' == 1692108723 (== int(sha256(b'0:c01:pp_portrait').hexdigest()[:8],16), probe-confirmed -- this is the identity step 2 reproduces); fixed mode returns request_seed verbatim; kind='scene_open' and object_id='radio_host_portrait' still return 4242. (c) A klein graph snapshot: assert set(_build_klein_graph(params,_W)) == the 12 ids at flux2_klein.py:167-206. tests/ has ZERO references to _build_klein_graph today (grep), so without this there is nothing for step 8 to be byte-identical AGAINST. z_image already has this guard at tests/test_image_engine_c2.py:152-154 (exactly 9 ids).

**Why:** Every later 'byte-identical to today' claim is self-fulfilling if the snapshot is written after the change. _content_hash json-dumps its list (otr_image_gen_dispatcher.py:113-116), so an unconditional append to the cache-key list changes EVERY digest; the append-only-when-truthy construction in step 6 is load-bearing and only a HEAD-captured literal proves it.

**Proves it worked:** Introduce `parts.append(anchor)` unconditionally in step 6 and test_cache_key_head_digest FAILS on the literal. Without step 1 that same mistake ships green, because the only snapshot would have been taken after the append.

#### Step 2 -- Pin the scene_character seed to the character's OWN portrait draw -- correct placement, truthful mode

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\otr_image_gen_dispatcher.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_still_spine_helpers.py`

**Change:** Add `resolve_seed_and_mode(seed_cfg, object_id, prompt_hash, *, kind='', char_id='', portrait_prompt_hash='') -> (int, str)` beside resolve_object_seed (otr_image_gen_dispatcher.py:134); keep resolve_object_seed as a thin wrapper returning element 0 so all existing call sites stay green untouched (tests/test_still_spine_helpers.py:694-721, tests/test_multiclip_jump_stills.py:234-237 and :540-562, tests/test_brief_radio_host.py:457-463 all call it with no char_id). The new branch goes AFTER the mode gate at :160-161, NOT after the 4242 pins at :152-157: `base` is assigned at :159, so the originally-specified placement raises UnboundLocalError inside a function called at :998 outside any try/except (dispatch_images dies outright), and placing it before :160-161 silently breaks the documented mode=='fixed' contract. Branch body: `if str(kind or '') == 'scene_character' and char_id and portrait_prompt_hash and os.environ.get('OTR_PORTRAIT_IDENTITY_SEED','1') != '0': return (int(hashlib.sha256(f"{base}:{char_id}:{portrait_prompt_hash}".encode()).hexdigest()[:8],16), 'seed')`. That expression is EXACTLY the portrait object's own draw at :162-164 (portrait object_id == char_id, otr_meta_brief_image_prompt.py:1765). Everything else returns (seed, ''). At :998 look the portrait row up ONCE (see step 3) and pass its `prompt_hash` -- the DISPATCHER-computed one on the row (:1177), never the payload object's, because the dispatcher recomputes it over the safety-clause-appended prompt at :911-912. Do NOT extend this branch to jump_segment.

**Why:** The original plan's fresh `char:` namespace unified the scene stills with each other but NOT with the canonical portrait, and the portrait is the sole init for the HuMo / 3D lane -- three faces would have become two. Deriving from the portrait's existing draw makes it one at zero cost to portrait reproducibility (the portrait's own seed is untouched and byte-identical). Returning the mode alongside the seed is what lets step 3 stamp what ACTUALLY happened instead of a literal that lies whenever OTR_PORTRAIT_IDENTITY_SEED=0 -- precisely the control arm step 9 depends on. jump_segment is excluded deliberately: tests/test_multiclip_jump_stills.py:234-237 asserts three DISTINCT seeds across a base plus two segments, and a char-keyed pin would collapse a jump CUT into the same frame twice.

**Proves it worked:** New test: resolve_seed_and_mode({'request_seed':0,'mode':'request_hash'},'still_b002','ph2',kind='scene_character',char_id='c01',portrait_prompt_hash='pp_portrait') == (1692108723,'seed') -- the portrait's own seed. Absent the branch it returns (282026089,'') and the test fails. Placement tests: the fixed-mode call with the same kwargs must return (7,'fixed-mode base) -- placed after :157 it raises UnboundLocalError, placed before :161 it returns a hashed seed.

#### Step 3 -- Stamp a truthful derived_from_portrait_hash on BOTH dispatch paths, including jump segments

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\otr_image_gen_dispatcher.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_image_platform_c1.py`

**Change:** Bind `_cp = _coverage_plan_module()` once at the top of dispatch_images (it is currently local to merge_jump_still_requests at :622). Inside the object loop, BEFORE :998, resolve the portrait row ONCE: `portrait_row = next((r for r in reversed(images) if isinstance(r,dict) and r.get('kind')=='portrait' and str(r.get('object_id') or '')==char_id), None)` when `kind in ('scene_character', _cp.JUMP_STILL_KIND)` and char_id is truthy -- reversed so a stale row from a prior image_revision cannot win. Read `prompt_hash` (step 2), `portrait_content_hash` (the anchor) and `pool_path` (step 6) off that ONE row. Stamp `row['derived_from_portrait_hash']` and `row['portrait_anchor_mode']` (the mode string step 2 returned) on the fresh-render row at :1170-1194 AND on the cache-hit `fresh` dict at :1026-1041. When char_id resolves to no portrait row, leave both empty and append a LOUD warning + log.warning naming the object. Amend the comment at :1175 to say plainly that on a scene row `portrait_content_hash` is that row's OWN decoded-pixel hash; do NOT rename it (render_driver and the mesh cache read it). Do NOT touch nodes/_otr_image_engines/schemas.py.

**Why:** Reading the CAST row via _pl.portrait_hash_for_char (portrait_ledger.py:125) would make the anchor vanish on the warm path: stamp_portrait writes entry['portrait_content_hash'] only at portrait_ledger.py:186, and the cache-HIT branch (:1002-1047) materializes a copy and appends a row but NEVER calls stamp_portrait, so the cast lookup returns None exactly where the fix is most needed. The images-list row carries the hash on BOTH paths (`fresh = dict(ref_row)` at :1026 inherits it). Jump segments are included: merge_jump_still_requests clones the scene object keeping char_id and rewrites kind to _cp.JUMP_STILL_KIND ('jump_segment', coverage_plan.py:639) at :716-737, so a multi-clip character beat would otherwise show a different face inside one beat with no anchor field to see it in. The schemas.py edit is DROPPED: grep shows CanonicalImage and ImageRequest are constructed nowhere outside schemas.py and tests/test_image_platform_c1.py:1126-1132, and the dispatcher's rows already carry ~10 keys CanonicalImage(extra='forbid') would reject -- adding fields to a schema nothing validates is decoration.

**Proves it worked:** New dispatch_images test with an injected gen_fn: three scene_character rows plus one jump_segment row sharing char_id 'c01' all carry derived_from_portrait_hash equal to the PORTRAIT ROW's portrait_content_hash, and portrait_anchor_mode == 'seed'. A second case sets OTR_PORTRAIT_IDENTITY_SEED=0 and asserts portrait_anchor_mode == '' -- the mode must never lie. A third pre-seeds cache_index and images (portrait a cache HIT, cast row carrying NO portrait_content_hash) and still resolves the anchor: this one fails outright if the anchor is read from the cast row.

#### Step 4 -- Make the appearance verbatim on the LLM prompt path -- path-guard safe, LOUD on a miss

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\otr_meta_brief_image_prompt.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_still_spine_helpers.py`

**Change:** In `_compose_char_scene_prompt` (otr_meta_brief_image_prompt.py:1339-1409), LLM-SUCCESS path only, between the finish_visual_prompt assignment and the image_grade_tail append: resolve `_app = _appearance_for_char([ce], str(ce.get('char_id') or ''))` -- mirroring `_build_char_scene_request`:1309 exactly, which keys off `char.get('char_id')`, NOT the separately-passed `cid`; the two disagree whenever `ce` lacks a char_id key and the 'same token sequence' premise then fails silently. Then (a) if `_app` is empty append `warnings.append(f"char-scene: no appearance text for {cid}; still has NO identity anchor (LOUD)")` -- `_cast_by_id.get(_cid)` at :2006 can return None, normalises to {} at :1347, and _appearance_for_char([{}],...) returns '' (:52-67 matches char_id only); (b) SANITIZE before prepending: `_app = _app.replace('\\',' ').replace('/',' or ')`; (c) prepend only when the first 40 characters of the sanitized `_app` are not already present in the prompt, so an LLM paraphrase does not stack two competing subject descriptions. Leave the deterministic path (:1377-1393) alone -- compose_still_prompt already leads with the identical key chain at _otr_story_brief_helpers.py:560-567.

**Why:** A shared seed with a re-worded subject is still a different face. Today the LLM gets the appearance only as the `character_appearance:` hint at :1324 and may paraphrase it, while the deterministic fallback leads with it verbatim -- only one of the two paths anchors at all. The sanitize is not cosmetic: path_guard_arm (otr_image_gen_dispatcher.py:219-272) rejects os.altsep and on Windows os.altsep == '/', so raw appearance text newly reaching the FINAL prompt would expose the LLM path to THE BRANCH THAT COST AN EPISODE at :976-997 -- my probe of 'a black/white striped scarf, a stern man' returns arm='alternate_separator', and the caller warns, records skip evidence and `continue`s with NO still generated. This is identity correctness (the same class as a character's voice contradicting the source), not prose quality: it changes WHICH tokens name the subject, not how the scene is written.

**Proves it worked:** New test with an LLM stub that PARAPHRASES the appearance (the existing fixture at tests/test_still_spine_helpers.py:527-560 echoes it verbatim and would never exercise the prepend): the composed prompt starts with the verbatim appearance. Second test: a cast appearance containing '/' yields a prompt for which disp.path_guard_arm(...) returns None and dispatch still mints the still -- without the sanitize this test fails with a skipped still. Third: an empty appearance emits the LOUD warning.

#### Step 5 -- Declare the reference capability and CLOSE the paid-adapter guard gap (zero behavior change)

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_image_engines\registry.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_image_engines\eng_google_image.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_image_engine_c2.py`

**Change:** (a) Document an OPTIONAL class attribute `accepts_reference_image: bool = False` in the ImageEngine Protocol docstring (registry.py:48-84); every reader uses `getattr(eng,'accepts_reference_image',False)` so an un-migrated adapter is a hard no-op. Leave the CAPABILITIES table (registry.py:106-181) untouched -- it describes toolchain and VRAM, not request shape. (b) Add `'reference_image'` to `_reject_reference_inputs` (eng_google_image.py:144-151), which today checks only `init_image` and `reference_images` (PLURAL) -- a singular key sails straight past it into the network call. (c) Tests: every registered engine reports False by default (live probe confirms all 11 do today: cloud_flux_pro, cloud_krea_2_turbo, cloud_luma_photon_flash, cloud_nano_banana_2, cloud_seedream_2, flux2_klein, flux_gen1, google_image, ideo, lumina_image, z_image_turbo), and google_image / ideo / the five cloud_* adapters stay opted out after steps 7-8.

**Why:** One owner per field, declared before anything can pass a reference, so steps 6-8 each ship green alone. The google fix is not hypothetical -- the original plan's own test bullet asserted a rejection the code does not perform. I keep the SINGULAR name rather than renaming to `reference_images` to match the existing guard: a plural name for a single path is a type lie, and extending the guard is one line.

**Proves it worked:** New test: google_image raises GoogleAPIRequestShapeError for a request carrying reference_image, with the HTTP transport monkeypatched to raise if it is ever called. Absent step 5(b) the transport is reached and the test fails on the wrong exception.

#### Step 6 -- Resolve the reference BEFORE the cache key, and append the anchor only when one is actually used

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\otr_image_gen_dispatcher.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_still_spine_helpers.py`

**Change:** (a) Add keyword-only `anchor: str = ''` to request_cache_key (:119-131); build the list exactly as today and append the anchor ONLY when truthy. (b) Resolve the reference between :997 and :998 -- NOT at :1103. Conditions: `kind in ('scene_character', _cp.JUMP_STILL_KIND)`, truthy char_id, a portrait_row from step 3, `getattr(_safe_engine(engine_id),'accepts_reference_image',False)` (engine_id is already resolved at :939; _safe_engine returns None for an unknown id and getattr(None,...) is False), and `os.environ.get('OTR_PORTRAIT_REFERENCE','1') != '0'`. Take `portrait_row['pool_path']`, falling back to `portrait_row['path']` then `_pl.portrait_path_for_hash(anchor_hash, output_dir)` (portrait_ledger.py:104); verify os.path.isfile and on any miss clear the reference LOUD (warning + log.warning) rather than raising. (c) Pass `anchor=<portrait_content_hash>` to request_cache_key at :1000 ONLY when a reference was actually resolved. (d) Add `'reference_image': reference_image` to the request dict at :1103-1115 and set `row['portrait_anchor_mode']='reference_latent'` on both row builds when it is non-empty.

**Why:** The original resolved the reference ~100 lines AFTER the key was built and after the cache-HIT branch already `continue`d at :1047, so the anchor could never enter the key -- the exact stale-cache correctness bug this step exists to prevent, failing silently: a regenerated portrait would keep serving stills conditioned on the OLD face. Appending only when a reference was truly used means step 6 lands byte-inert (no engine opts in until step 7) and an un-anchored mint never perturbs its key. Fail-SOFT is deliberate: an unanchored mint is strictly no worse than today, and raising would invent a refusal class THE LAW forbids for anything but a structural impossibility. The env kill switch avoids a widget: canonical node 88 OTR_ImageDirector widgets_values is exactly ['per_object','per_object','per_object',15,'request_hash',42,'{}','fp8_ok'] (8 slots, probe-verified) and a new widget would force a canonical edit under the append-only rule.

**Proves it worked:** Discriminator test: dispatch the same scene_character object twice against a stub engine with accepts_reference_image=True, changing ONLY the portrait row's portrait_content_hash between runs -- the second dispatch must be a fresh render (made=1), not a cache HIT (reused=1). Without (c) the two keys are identical and the test fails with reused=1, which is exactly the stale-face bug. Plus: request_cache_key with anchor='' still equals step 1's HEAD literal, and anchor='abc' != anchor='def' != anchor=''.

#### Step 7 -- Wire ReferenceLatent into z_image_turbo -- full node signatures, separate class map, symmetric CFG

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_image_engines\z_image_turbo.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_image_engine_c2.py`

**Change:** Set `accepts_reference_image = True` beside required_inputs (z_image_turbo.py:153) and add `'reference_image': str(get('reference_image') or '')` plus `'reference_height': _eint('OTR_PORTRAIT_REF_HEIGHT', 768)` to _zimage_params (:160-206). Do NOT add the ref keys to _node_candidates (:208-228): render_image caches the resolved map on a REGISTRY SINGLETON (`classes = getattr(self,'_classes',None) or resolve_graph_classes(...)`; `self._classes = classes` at :318-320, and EngineRegistry.register stores one instance at _otr_shared/engine_registry_base.py:148-150), so a params-gated candidate dict would be resolved WITHOUT the ref keys on the episode's first portrait and every later referenced mint would die with GraphExecutionError 'node ... class ... unresolved' (wrapper_bridge.py:386-388) -> ImageRenderError NO FALLBACK. Instead add a module-level `_REF_CANDIDATES = {'load_ref':('LoadImage',),'scale_ref':('ImageScale',),'encode_ref':('VAEEncode',),'ref_pos':('ReferenceLatent',),'ref_neg':('ReferenceLatent',)}` resolved into a SEPARATE `self._ref_classes` inside `if params['reference_image']:` and merged into a local `classes` dict for that call only. In _build_zimage_graph (:230-270): empty reference -> TODAY'S GRAPH UNCHANGED. Non-empty -> stage the PNG with wrapper_bridge.stage_into_comfy_input (wrapper_bridge.py:999-1012, returns the basename LoadImage resolves), then `load_ref` -> `scale_ref` with the FULL five-argument set {image: W('load_ref',0), upscale_method: 'lanczos', width: 0, height: params['reference_height'], crop: 'disabled'} -> `encode_ref` {pixels: W('scale_ref',0), vae: W('vae',0)} -> `ref_pos` {conditioning: W('pos',0), latent: W('encode_ref',0)} AND `ref_neg` {conditioning: W('neg',0), latent: W('encode_ref',0)}; rewire ksampler positive -> W('ref_pos',0) and negative -> W('ref_neg',0). Leave latent (EmptySD3LatentImage) and denoise 1.0 untouched.

**Why:** run_graph calls fn(**kwargs) from the graph's own inputs dict (wrapper_bridge.py:401-404) and ImageScale.upscale is (self, image, upscale_method, width, height, crop) with all five required (ComfyUI nodes.py:1883) -- the original's two-argument scale node is a hard episode kill converted to ImageRenderError 'NO FALLBACK' at otr_image_gen_dispatcher.py:1143-1150. width=0 is deliberate over the prior hardening's crop='center' to the output dims: ImageScale derives the missing side from the actual image (nodes.py:1885-1889) so the 832x1216 portrait keeps its aspect and no face is cropped into a 16:9 band, and comfy.sd.VAE.encode calls vae_encode_crop_pixels internally (comfy/sd.py:1253) so a non-multiple-of-8 width is safe; a 768-high reference also costs ~1.5k latent tokens instead of ~4k. Referencing BOTH conditionings is the CFG decision the original left open: z_image runs cfg=2.0 with a live negative (:198) and NextDiT._forward doubles the timesteps only when omni (comfy/ldm/lumina/model.py:825-828), so referencing only the positive would take the CFG delta between structurally different forward passes. Architecture verified on this box: ReferenceLatent at comfy_extras/nodes_edit_model.py:6 (latent optional, append=True, chainable); ZImage(Lumina2) -> Lumina2.extra_conds ref_latents at model_base.py:1520-1525; patchify_and_embed:730-742 embeds the reference through the SAME embed_all. My own safetensors header probe of z_image_turbo_nvfp4.safetensors (4,509,509,600 bytes, 993 tensors): cap_pad_token PRESENT, x_pad_token PRESENT, siglip_embedder ABSENT -- the omni path lands in the non-siglip branch at :676 and needs no extra weights.

**Proves it worked:** Graph tests: with reference_image unset, set(_build_zimage_graph(...)) is still exactly the 9 ids already pinned at tests/test_image_engine_c2.py:152-154. With it set, the graph gains exactly load_ref/scale_ref/encode_ref/ref_pos/ref_neg AND ksampler.inputs['positive'] == W('ref_pos',0) and ['negative'] == W('ref_neg',0) -- assert the REWIRE, because a graph that builds the chain and forgets to consume it passes a node-count check and renders nothing different. Signature-conformance test: set(graph['scale_ref']['inputs']) equals ImageScale's required INPUT_TYPES keys read from the live class (skipped when ComfyUI is not importable) -- this is the test that fails on the original's two-argument node.

#### Step 8 -- Same wiring on flux2_klein -- signature change at both call sites, one pinned scaler

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_image_engines\flux2_klein.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\test_image_engine_c2.py`

**Change:** Set `accepts_reference_image = True` (flux2_klein.py:90 area) and add 'reference_image' + 'reference_height' to _klein_params (:97-138). Change `_node_candidates(self)` (:140) to `_node_candidates(self, params=None)` and update BOTH call sites -- load() at :211 and render_image at :252 (z_image already has this signature at :208; klein does not) -- then use the SAME separate `_ref_classes` map as step 7, for the same singleton-cache reason. Pin ImageScale with the full five arguments; do NOT put FluxKontextImageScale in an ordered candidate tuple with it -- resolve_graph_classes binds the FIRST installed name (wrapper_bridge.py:100-107) and one graph node id carries ONE inputs dict, so klein would always bind FluxKontextImageScale (comfy_extras/nodes_flux.py:126-146, `execute(cls, image)` -- image ONLY) and then receive upscale_method/width/height/crop -> TypeError -> GraphExecutionError -> dead episode. In _build_klein_graph (:159-206) insert ONE ReferenceLatent between guidance and guider: `ref {conditioning: W('guidance',0), latent: W('encode_ref',0)}` and rewire guider.inputs['conditioning'] from W('guidance',0) to W('ref',0). Empty reference -> today's graph byte-for-byte, checked against step 1's 12-id snapshot.

**Why:** model_base.Flux2(Flux):1075 inherits the ref_latents path at :1036-1048, and FLUX.2 [klein] IS a genuinely reference-trained edit / multi-reference family (the adapter's own docstring at flux2_klein.py:4-5 cites 'strong pose + multi-reference character consistency'). Weights are on disk -- flux-2-klein-4b-Q4_K_M.gguf, 2,604,311,104 bytes (2.60 GB decimal / 2.43 GiB), no download. Klein needs only ONE ReferenceLatent because its graph uses BasicGuider (:194-196) with no uncond at all, so the cond/uncond asymmetry that forces symmetric wiring on z_image cannot arise. If step 9's A/B shows the turbo checkpoint ignores its reference, switching character_image_model to flux2_klein is a Director widget pick, not a code change.

**Proves it worked:** With reference_image unset, _build_klein_graph is byte-identical to step 1's 12-id HEAD snapshot. With it set: exactly four new ids and guider.inputs['conditioning'] == W('ref',0). Candidate test: 'FluxKontextImageScale' NOT in _REF_CANDIDATES['scale_ref'] -- absent this, the first referenced klein mint dies with a TypeError inside GraphExecutionError and nothing in the unit suite sees it.

#### Step 9 -- Prove it live under a NAMED profile -- canonical loaded, never edited -- with a real discriminator

**Files:**
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\otr_canonical_api_run.py`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\PROD_BUG_LOG.md`

**Change:** The leg MUST name its profile, because workflows/otr_canonical.json AS SAVED mints zero stills (node 87 selects viz_mxc_cpu / viz_mxc_mandala / viz_camera; probe: still_consumer_capabilities -> all False, roles_requiring_stills -> frozenset(), derive_image_prompts -> {'version':1,'objects':[]}). Reset per CLAUDE.md section 4, boot headless, then run `scripts/otr_canonical_api_run.py --profile otr_w45_still_flat` -- which LOADS the real canonical JSON and applies the profile through the single applier (config/profiles/widget_mapping.json maps role_overrides.character_visual -> [OTR_VideoDirector, character_video_model]; probe: still_flat gives announcer/music/character all True, and it is the cheapest still-consuming lane). Render ONE shakespeare leg with a recurring character speaking on >= 3 beats. ASSERT in this order: (1) the request carried a non-empty EXISTING reference_image path and the submitted graph contained a ReferenceLatent node -- this is THE discriminator; (2) every scene_character AND jump_segment row sharing a char_id carries the same non-empty derived_from_portrait_hash equal to that character's portrait ROW's portrait_content_hash; (3) portrait_anchor_mode reads 'reference_latent' on the treatment arm and '' on the control; (4) per-still latency delta treatment vs control; (5) RESULT SUCCESS + obs_publish OK + Test-Path each still under otr\episodes\<ep>\stills\ per section 6. Re-run with OTR_PORTRAIT_REFERENCE=0 for the A/B pair and record the eyeball verdict. PBUG entry only after the artifacts exist. NO edit to workflows/otr_canonical.json.

**Why:** The panel unanimously prescribed editing node 87's widget. I overrule that: the prior hardening grepped config/profiles/*.json for the WIDGET name `character_video_model` and concluded 'NOTHING sets it, so there is no run-time profile retarget to lean on'. The profile KEY is `role_overrides.character_visual`, and ~40 shipped profiles set it (probe: otr_w45_still_flat -> still_flat, otr_g4_humo -> humo, otr_w45_ltx_video -> ltx_video, all character_video True). Editing the canonical widget would change the operator's production configuration to make a test run, which section 0 does not require and the operator did not ask for. The assertion set is rewritten because the original's two checks could not fail: copying one cast field onto N rows trivially makes them equal, and 'three DIFFERENT content_hash values' is ALREADY TRUE today -- three seeds and three prompts per beat IS the published defect.

**Proves it worked:** Only this step can answer whether the installed z_image_turbo_nvfp4 checkpoint ATTENDS to a prepended reference or silently ignores it -- a green unit suite proves the graph shape and the seed algebra but cannot prove three faces became one. Assertion (1) fails outright if any of steps 5-7 regressed the plumbing; the OTR_PORTRAIT_REFERENCE=1 vs 0 pair is what distinguishes a working reference from a no-op.

### Risks

- z_image_turbo_nvfp4 is the TEXT-TO-IMAGE TURBO checkpoint, not Z-Image Omni. My header probe (993 tensors) shows cap_pad_token and x_pad_token PRESENT, siglip_embedder / siglip_pad_token / siglip_refiner ABSENT -- so embed_all takes the non-siglip branch at comfy/ldm/lumina/model.py:676 and the reference is embedded through the shared x_embedder path with no missing module. It will NOT crash. But nothing guarantees the checkpoint was TRAINED to attend to prepended reference tokens; identity hold could range from strong to a silent no-op. This must never be described as fixed before step 9's A/B. Step 8 (klein, genuinely reference-trained, weights on disk) is the fallback.
- VRAM and latency. Lumina2 sets memory_usage_factor_conds=('ref_latents',) at model_base.py:1489 so ComfyUI accounts for it, and z_image weights are 4.51 GB against a 14.5 GB ceiling. Step 7 references BOTH cond and uncond, so an 8-step turbo mint runs two omni forward passes. Capping the reference at 768px high (~1.5k latent tokens instead of ~4k for the full 832x1216) is the mitigation, but per-still time will still rise and MUST be measured on the live leg -- one scene_character still per character beat plus one per jump segment.
- The anchor is a DECODED-PIXEL hash (portrait_ledger.compute_portrait_hash), and step 6 puts it in the cache key. Any nondeterminism in the nvfp4/fp8 portrait render changes the anchor and rekeys every scene still with no input change. Prove portrait pixel-hash stability across two identical runs before trusting it. Mitigation already in the design: the anchor enters the key ONLY when a reference was actually passed.
- Rendered pixels change on every lane once step 7 lands, invention lanes included. C7 determinism is PRESERVED -- steps 2-4 are pure functions of stable inputs and nothing in the script / casting / text path is touched, so identical inputs still give byte-identical output. The pixel VALUES are deliberately not byte-identical to the pre-fix build, because the pre-fix pixels are the defect, and there is no scoping that fixes the adaptation lanes while leaving the invention lanes' faces broken. OTR_PORTRAIT_IDENTITY_SEED=0 and OTR_PORTRAIT_REFERENCE=0 reproduce the old behavior exactly.
- Step 4 changes scene_character prompt TEXT, therefore prompt_hash, therefore the request_cache_key for those objects -- an episode re-render loses its cache hit for those stills once. It no longer reseeds anything, because step 2 removed prompt_hash from the scene_character seed; still, land step 2 first or the intermediate commit reseeds twice.
- The pre-existing path-guard exposure on the DETERMINISTIC composer is NOT fixed here and is out of scope: compose_still_prompt (_otr_story_brief_helpers.py:560-567) already leads with the raw appearance, so a cast appearance containing '/' already kills that beat's still today (probe: path_guard_arm returns arm='alternate_separator'). Step 4's sanitize covers only the new LLM-path exposure. That separate defect deserves its own change with its own live evidence.
- The portrait -> scene join relies on portraits being dispatched BEFORE scene stills (appended at otr_meta_brief_image_prompt.py:1765-1784 vs :2028-2042; the dispatcher iterates list order at :895). True today but IMPLICIT -- which is why steps 3 and 6 degrade LOUD (warning + log.warning naming the object) rather than assuming, and why the row lookup scans reversed(images) so a stale row from a prior image_revision cannot win.
- stamp_portrait with require_cast_entry=False returns before the cast stamp (portrait_ledger.py:177-185), so the cast row is never written for the synthetic announcer or radio_host_portrait. Steps 3 and 6 sidestep this by reading the IMAGES row, but no code path may assume every char_id resolves to a hash.
- Do NOT reach for VAEEncode img2img as a shortcut. It is structurally available on all four local engines, but it copies COMPOSITION, and a denoise low enough to hold a face also freezes the pose -- destroying the beat-awareness _build_char_scene_request:1290-1301 exists to deliver ('each character beat yields a DISTINCT still'). It trades a continuity bug for a variety bug.

### Tests

- Focused regression after steps 2-4: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_still_spine_helpers.py tests/test_multiclip_jump_stills.py tests/test_image_platform_c1.py tests/test_brief_radio_host.py -- these pin resolve_object_seed and request_cache_key semantics at test_still_spine_helpers.py:690-722, test_multiclip_jump_stills.py:234-237 and :540-562, test_brief_radio_host.py:457-463, and all of them call resolve_object_seed with NO char_id so a keyword-only parameter leaves them untouched.
- HEAD baseline pins (step 1, captured before any source edit): request_cache_key('character_video','still_b002','phX',123,'z_image_turbo','1',kind='scene_character',w=1472,h=832) == 'e8e5a5dcacd093e8db11220d218e477a2656763056bb1faca3a61d9f5539f17d'; resolve_object_seed({'request_seed':0,'mode':'request_hash'},'c01','pp_portrait') == 1692108723; set(_build_klein_graph(params,_W)) == the 12 ids at flux2_klein.py:167-206.
- Placement tests that would have caught the original's UnboundLocalError: resolve_seed_and_mode({'mode':'fixed','request_seed':7},'still_b002','ph',kind='scene_character',char_id='c01',portrait_prompt_hash='pp') == (7, ''); the same call under request_hash mode returns the character's PORTRAIT seed and mode 'seed'; a scene_character with an empty char_id OR an empty portrait_prompt_hash keeps today's request-hash derivation; kind='scene_open' and object_id='radio_host_portrait' still return 4242; jump_segment rows still draw three DISTINCT seeds.
- Ledger tests via dispatch_images with an injected gen_fn: scene_character AND jump_segment rows carry derived_from_portrait_hash equal to the portrait ROW's portrait_content_hash; portrait_anchor_mode reads 'seed' after step 2 and '' when OTR_PORTRAIT_IDENTITY_SEED=0 (the mode must never lie); a CACHE-HIT dispatch (pre-seeded cache_index and images, cast row carrying no portrait_content_hash) still resolves the anchor from the images list; a char_id with no portrait row yields an empty anchor plus a LOUD warning and still renders.
- Cache-key discriminator (step 6): the same scene_character object dispatched twice against a stub engine with accepts_reference_image=True, changing only the portrait row's portrait_content_hash, must re-render (made=1) rather than cache-hit (reused=1); request_cache_key with anchor='' still equals the step-1 literal; anchor='abc' != anchor='def' != anchor=''.
- Prompt tests: an LLM stub that PARAPHRASES the appearance (the existing fixture at tests/test_still_spine_helpers.py:527-560 echoes it verbatim and would never exercise the prepend); a cast appearance containing '/' produces a prompt for which path_guard_arm returns None and the still is still minted; an empty appearance emits the LOUD warning.
- Engine graph tests (tests/test_image_engine_c2.py): both builders byte-identical to the step-1 HEAD snapshots when reference_image is unset; with a reference set, assert the exact new node ids AND the rewires -- ksampler.positive == W('ref_pos',0) and ksampler.negative == W('ref_neg',0) for z_image, guider.conditioning == W('ref',0) for klein; assert scale_ref's input keys match ImageScale's required INPUT_TYPES read from the live class; assert 'FluxKontextImageScale' is not in the scale_ref candidate tuple.
- Capability opt-out (tests/test_image_engine_c2.py, tests/test_google_image_adapter.py, tests/test_cloud_image_adapters.py): every registered engine reports accepts_reference_image False by default; google_image, ideo and the five cloud_* adapters stay False after steps 7-8; google_image raises GoogleAPIRequestShapeError for a request carrying reference_image with the transport monkeypatched to raise if reached.
- Bug Bible: cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide then pytest -q -p no:cacheprovider tests\bug_bible_regression.py (RELATIVE path -- an absolute forward-slash path fails to collect).
- Live proof (step 9): one shakespeare leg via scripts/otr_canonical_api_run.py --profile otr_w45_still_flat with a character on >= 3 beats. Discriminator first (non-empty existing reference_image path plus a ReferenceLatent node in the submitted graph), then the ledger equalities, then per-still latency delta, RESULT SUCCESS + obs_publish OK, Test-Path each still under otr\episodes\<ep>\stills\. Repeat with OTR_PORTRAIT_REFERENCE=0 for the A/B.

### Open questions

- Does the installed z_image_turbo_nvfp4.safetensors actually ATTEND to a prepended reference latent, or is a separate Z-Image Omni release required? The architecture accepts it with no missing weights (header probe: cap_pad_token and x_pad_token present, siglip absent), but only step 9's A/B answers it. If it does not hold, the production answer is flux2_klein via the OTR_VideoDirector character_image_model dropdown -- node 87 currently reads 'z_image_turbo' in all three IMAGE slots, so that is a widget VALUE pick, not code.
- Which profile should the proof leg run? I recommend otr_w45_still_flat -- the cheapest still-consuming lane, all three roles True -- but otr_g4_humo / otr_w45_ltx_video / otr_g4_wan_ti2v also work and exercise a heavier video lane. This is a measurement choice, not a production-config change, so it needs no operator sign-off; it is listed so the recorded PBUG names the configuration it was proved under.
- The constraint says the invention lanes must stay BYTE-IDENTICAL. My ruling for this plan is that C7 determinism is the property at stake -- identical inputs give byte-identical output, which every step preserves -- and that pixel VALUES moving because a face-continuity defect was fixed is not a determinism violation. There is no scoping that fixes the adaptation lanes while leaving the invention lanes' faces broken. Flagged rather than treated as a blocker; OTR_PORTRAIT_IDENTITY_SEED=0 and OTR_PORTRAIT_REFERENCE=0 reproduce the pre-fix build exactly if the operator disagrees.
- One reference or a chain? ReferenceLatent appends (comfy_extras/nodes_edit_model.py:26, append=True) and its description says multiple can be chained if the model supports it, so the previous beat's still could be a second anchor. That would strengthen beat-to-beat coherence but costs sequence length (already doubled by the symmetric cond/uncond wiring) and risks compounding drift away from the canonical portrait. Recommend one until step 9 measures the first.
- Does every speaking character reliably receive a portrait object to anchor on? The payload emits one per cast row gated on _still_required(role) (otr_meta_brief_image_prompt.py:1612-1625), but the still plans declare portrait cardinality per_subject / per_recurring_subject with required='never' (eng_ltx_8gb.py:471-473) -- so the plan LANGUAGE admits a non-recurring character with no portrait. Confirm on a live multi-character cast; steps 3 and 6 degrade LOUD rather than raise if it is not.
- Should steps 7-8 also cover lumina_image? It shares the Lumina2 reference path, but my header probe of lumina_2_model_bf16.safetensors (400 tensors) shows cap_pad_token ABSENT, and embed_cap(None) at comfy/ldm/lumina/model.py:662 dereferences self.cap_pad_token -- which NextDiT.__init__ only creates when pad_tokens_multiple is set, which model_detection only sets when the state dict carries that tensor. A referenced lumina_image mint would raise AttributeError. It is also not a selected engine in the canonical workflow. Defer.


---

# WHAT r3 MUST DELIVER

A VERDICT plus a MUST-FIX list. For each must-fix give:
`file:line` -> what the plan says -> what the code actually says -> the minimum fix.

Answer these eight explicitly (they are the driver's open wiring questions):

Q1. Does landing Track 1 (gender pin) BEFORE Track 3 Step 4 (tier floor=2)
    invalidate the floor measurement? The floor invariant is over the BANK; the
    pin shifts DEMAND male-heavy on adaptation lanes while indextts2 is male-light
    (17 male / 23 female). Must-fix or note?
Q2. Track 1 Step 4(i): are trailing dataclass appends to `EnsembleSlot` and
    `CastSlot` genuinely safe? Find EVERY construction site in the tree
    (exclude tmp/). tests/test_cast_llm_naming.py:141 uses FIVE positional args.
Q3. Track 1 Step 2: find EVERY caller of `source_meta_from_scene` and
    `source_meta_from_unit`, including scripts/ and bench runners. Does the
    keyword-only `text_path` default leave all of them green?
Q4. Does `sidecar_path_for_text` resolve for the REAL manifest text_path values?
    Sidecars live in `config/source_banks/shakespeare/sources/`.
Q5. Does importing `_SEEDED_ANNOUNCER_ENGINES` from `_otr_voice_bank` into
    `cast_lock.py` create an import cycle? Is `cast_lock.py:527` the only duplicate?
Q6. Track 2 Step 2: confirm mis-placing the new branch before the mode gate at
    :160-161 raises UnboundLocalError, and that `dispatch_images` (calling at :998)
    has no try/except that would swallow it.
Q7. Track 2 Step 6: is resolving the reference at :997-998 safe against everything
    those lines already compute?
Q8. Chunk 3 merges Track 1 Step 6 with Track 3 Step 7. Once CastLock stamps the
    fallback ref, does `_resolve_clone_ref_path`'s vrid lookup hit first and
    orphan the anyref path Track 3 Step 7 wanted to test?

LEDGER COMPLETENESS (standing operator rule -- a changed pass may leave NO field
unowned): the new ledger fields are `cast_source_contract` (Track 1),
`derived_from_portrait_hash` + `portrait_anchor_mode` (Track 2), and
`voice_cast_fallback` (chunk 3). Name any that lacks exactly one owner or a
defined value on EVERY path, including the cache-hit path and the invention lanes.
