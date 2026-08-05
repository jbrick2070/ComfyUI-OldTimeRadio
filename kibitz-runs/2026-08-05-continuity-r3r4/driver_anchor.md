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
