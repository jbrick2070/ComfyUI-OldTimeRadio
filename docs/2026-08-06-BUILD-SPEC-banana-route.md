# BUILD SPEC -- the banana route (v2 contract)

**Build state (2026-08-07):** BUILT and SHIPPED. The build landed with six
defects found by a read-only QA pass; all six are repaired, plus three more the
repair arc surfaced (an unsound phrase-retreat loop in the cap, a
malformed-knob warning storm, and an era-tail quote leak that shielded weapons
from the route). Fix plan and arc:
`docs/2026-08-06-PLAN-banana-route-qa-fixes.md`. **One known defect is
deliberately OPEN and deferred to its own chunk:** the quote shield is applied
to EVERY prompt rather than only to still_word card text, so an ordinary
LLM-authored `a man carrying a "revolver"` is shielded and survives
untransformed (`_clean_llm_prompt` strips only leading/trailing quotes). That is
a false negative -- the route under-fires -- with no ledger or render fault.

**Date:** 2026-08-06. **HEAD:** `9c686886` on `v2.0-alpha`.
**Baseline, measured at this HEAD:** suite 8997 passed / 111 skipped /
1 xfailed; Bug Bible 17; `engine_matrix --check` OK.

**Provenance.** Problem statement `docs/2026-08-06-PROBLEM-STATEMENT-banana-route.md`
(committed `9c686886`); full four-round `kibitz-plugin:kibitz` arc (Codex +
Antigravity every round, Fable twice on r1 -- artifacts in
`kibitz-runs/2026-08-06-banana-route/`, LOCAL ONLY, gitignored); a scoped
Codex QA pass on the arc's final plan (`kibitz-runs/2026-08-06-banana-route-qa/`,
scope receipt inside) whose seven rulings were ALL ACCEPTED by the driver; and
four operator rulings (a)-(d) plus the FINAL switch ruling restated in the
2026-08-07 kickoff. This document is the arc's r4 final REWRITTEN as one
self-consistent contract -- the QA pass demanded exactly that rewrite -- with
every line cite re-pinned at `9c686886` (the SFX rip `9eb6ede1` moved
`render_driver.py` since the arc ran). **Do not re-run the arc.**

**DECIDED, DO NOT REOPEN:**
1. **VISUALS ONLY** (operator ruling (a)/7a): the spoken script is untouched.
   The announcer says "he drew his revolver" over a banana; that IS the joke.
2. **OPTION 2, NO WIDGET** (operator ruling 7c + kickoff): global default ON,
   `shakespeare` + `public_domain` default OFF via the `_LEMMY` exclusion
   idiom. NO node widget, NO canonical-workflow change, NO variants regen.
   This SUPERSEDES the arc r4 final's two-widget design (its sections 3/4/6/7
   are void where they touch widgets or `workflows/otr_canonical.json`).
3. **Option A at the two existing funnels** -- the emptied
   `append_visual_safety_clause` seam's route, repaired, not rebuilt.

---

## 1. THE SWITCHES -- two env keys, the bank idiom, one override

Two master switches (operator ruling (b)), one per funnel, as ENV KEYS read at
the transform sites -- the house pattern for opt-outs (`OTR_PORTRAIT_REFERENCE`
class):

| key | default | meaning |
|---|---|---|
| `OTR_BANANA_STILLS` | ON | banana transform at the still funnel |
| `OTR_BANANA_VIDEO` | ON | banana transform at the video funnel |
| `OTR_BANANA_INCLUDE_FIDELITY_BANKS` | OFF | force the route ON for `shakespeare`/`public_domain` (7c: "the operator can still force it on anywhere") |

**Guarded boolean read** (PBUG-20260723-02 posture -- a malformed knob is
IGNORED, never fatal): unset/`1`/`true`/`yes`/`on` -> ON; `0`/`false`/`no`/`off`
-> OFF; anything else -> the default plus ONE warning naming the key.

**The fidelity exclusion** is the blessed idiom, copied not invented
(`nodes/_otr_casting.py:1238` `_LEMMY_EXCLUDED_SOURCE_BANK_IDS` +
`:1241-1249` `_source_bank_excludes_lemmy`, whose docstring states the rule:
"fidelity is a family behaviour, not a per-row opt-in"):

```python
_BANANA_EXCLUDED_SOURCE_BANK_IDS = frozenset({"public_domain", "shakespeare"})

def source_bank_excludes_banana(source_bank_id) -> bool:
    # normalized through _otr_bank_variants.base_source_bank_id so bake-off
    # variants (shakespeare_v2, public_domain_v3) inherit the exclusion
```

COPIED per the operator's ruling (three lines, not new machinery), with the
drift closed by a TEST instead of a shared module: the suite asserts
`_BANANA_EXCLUDED_SOURCE_BANK_IDS == _otr_casting._LEMMY_EXCLUDED_SOURCE_BANK_IDS`
-- one answer to "what is a fidelity lane," enforced without new surface
(Fable cold-r1 concern, resolved inside the ruling).

The bank id comes from `ledger["meta"]["source_bank"]` -- the durable stamp the
writer mints (`OTR_LedgerScriptWriter.py:3564`) and credits already `_require`
(`otr_credits_roll.py:540`). Both funnels hold the ledger (still:
`dispatch_images`; video: the composer reads `(ledger or {}).get("meta")`).
A ledger with NO `meta.source_bank` (hand-built harness requests) counts as
NOT excluded -- the global default applies.

**Effective gate at each funnel:**
`env_switch_on AND (not source_bank_excludes_banana(bank) OR include_fidelity)`.

**Stated plainly, two consequences of no-widget:**
* Env changes need a FRESH server boot -- env cannot reach a resident ComfyUI
  process and there is no widget to re-hash. Same as every other env knob.
* Default ON means the original lane transforms on the first post-land render,
  and still seeds/cache keys MOVE there (the transform runs before the content
  hash, which is what re-mints a cached gun as a banana instead of serving it
  stale). Fidelity lanes and env-disabled runs are byte-identical to today.

## 2. THE MODULE -- `nodes/_otr_banana_route.py`

Pure; imports `re`, `hashlib`, `os`, `dataclasses` (+ the dep-free
`_otr_bank_variants.base_source_bank_id`, imported inside the function if it is
not import-clean -- verify at build). No import-time side effects, no
`NODE_CLASS_MAPPINGS` entry.

```
TABLE_VERSION = "2"
BANANA_TABLE: tuple[tuple[str, str], ...]   # every source form an explicit pair

@dataclasses.dataclass(frozen=True)
class BananaResult:
    text: str
    substitutions: int          # unquoted regex matches replaced
    table_version: str
    sha256_before: str          # raw-UTF-8 hex of the prompt string
    sha256_after: str
    varieties: str              # canonical compact "SIDEARM=banana,LONG=plantain"

def select_varieties(variety_key: str) -> dict[str, int]   # pure; class -> pool INDEX
def apply(text: str, *, variety_key: str = "") -> BananaResult
def source_bank_excludes_banana(source_bank_id) -> bool
def banana_stills_enabled() -> bool          # guarded env reads
def banana_video_enabled() -> bool
def include_fidelity_banks() -> bool
```

* **Matching:** `\b`-anchored, `re.IGNORECASE`, longest-source-first (enforced
  by a test over the table, not by hand-ordering).
* **Case preservation:** lowercase -> canonical replacement as-is; Title-case
  -> first alphabetic char of the WHOLE replacement uppercased (`Gunman` ->
  `Man wielding a banana`); ALL-CAPS -> entire replacement uppercased; mixed ->
  lowercase. Replacement strings carry canonical casing (scaffold lowercase).
* **Hash contract:** `sha256_*` are raw-UTF-8 digests of the prompt string --
  deliberately NOT `_prompt_content_hash` (`otr_image_gen_dispatcher.py:55-63`,
  which json-wraps first). Two hashes, two purposes, never compared. In the
  module docstring.
* **Quote shielding, state machine (QA ruling 9 -- same-style pairing):**
  straight `"` pairs only with straight `"`; curly open only with curly close;
  cross-style pairs are NOT legal; a delimiter is escaped iff its immediately
  preceding backslash run has ODD length; unmatched openers and closers are
  LITERAL text -- the transform resumes, never silently disabling the rest of
  the prompt.
  **WHY the shield exists, so nobody weakens it (Fable cold-r1 catch):** the
  `still_word` title cards quote the SPOKEN LINE verbatim inside the visual
  prompt (`otr_meta_brief_image_prompt.py` composes
  `a title card displaying the words "<spoken line>"`; the music card quotes
  the episode title the same way). Text rendered INSIDE the picture is script,
  not picture -- transforming it would put "HE DREW HIS BANANA" on a card the
  audience READS while the announcer says "revolver", which breaks the
  visuals-only ruling on its one audience-readable surface. The quoted span is
  therefore shielded, and the test suite pins EXACTLY this shape:
  `title card displaying the words "he drew his revolver"` passes through with
  the quoted span byte-identical. Verify the two card-composition sites'
  quoting at build.
* **Variety pick rule** (per class, independent, so adding a class never
  reshuffles): `idx = 0 if not variety_key else
  int(sha256(f"otr-banana-v2:{variety_key}:{cls}".encode()).hexdigest()[:8], 16) % len(pool)`.
  Index 0 of every pool is the exact v1 replacement, so an empty key is
  byte-identical to the base table.
* `apply()` is unconditional and pure; the gate lives at the call sites. The
  module is imported on OFF runs too: the OFF receipt records `TABLE_VERSION`
  and `select_varieties(freeze_timestamp)` (which table and which fruits WOULD
  have applied -- the QA's fix for OFF-run forensics).

## 3. THE TABLE (one contract; the arc's section-1/2/2A contradictions resolved)

Bare-noun replacements so articles and possessives survive. Every plural is its
own explicit pair. Replacements are drawn per-episode from SHAPE-CLASS pools
keyed on `ledger["meta"]["freeze_timestamp"]` (minted at freeze,
`_otr_ledger_freeze.py:950`; rename-proof; never touches prompts or seeds).

**v1 SHIPS TWO POOLS -- SIDEARM + LONG (QA ruling 8). Everything else is
PINNED to its index-0 replacement.** Four episode flavours; TINY / SCATTER /
ANTIQUE / ENERGY pool rungs stay recorded design, not shipped code.

| class | source forms (each an explicit pair, singular AND plural) | pool |
|---|---|---|
| SIDEARM | gun(s), handgun(s), pistol(s), revolver(s), six-shooter(s), firearm(s), weapon(s), blaster(s) | `banana` / `red banana` |
| SIDEARM (phrase) | gunman -> `man wielding a {SIDEARM}`; gunmen -> `men wielding {SIDEARM plural}` | rides SIDEARM |
| LONG | rifle(s), carbine(s), musket(s), assault rifle(s), sniper rifle(s) | `long banana` / `plantain` |
| PINNED | shotgun(s), machine gun(s), tommy gun(s), submachine gun(s) | `bunch of bananas` |
| PINNED | derringer(s) | `banana` |
| PINNED | flintlock(s), blunderbuss(es) | `banana` |
| PINNED | ray gun(s), death ray(s), disintegrator(s) | `banana beam` |
| PINNED | knife/knives, dagger(s), switchblade(s), sword(s), sabre(s), saber(s), rapier(s), cutlass(es), bayonet(s), machete(s), straight razor(s), ice pick(s) | `banana` |
| PINNED | truncheon(s), billy club(s), club(s)? -- NO: bare `club` EXCLUDED (card clubs, night clubs); only `billy club(s)` and `truncheon(s)` are rows | `banana` |
| PINNED | grenade(s), hand grenade(s), bazooka(s) | `banana` |
| PINNED | brass knuckles | `banana peels` |

`gunman` phrase forms: `man wielding a banana` / `man wielding a red banana`
(Title/ALL-CAPS per the case rule). **`wielding`, never `holding`** (operator
ruling (c): the pose is the gag; a man at rest holds a snack). The verb is
FIXED -- the randomized-verb table is cut (QA ruling 8); `gripping`/`hefting`
stay recorded as future-axis notes only.

**EXCLUDED, and why (collision risk in COMPOSED visual prompts -- measured, not
argued; the 7b sweep at `9c686886` found `shot` inside 15+ authored framing
strings: the still-plan `framing_geometry` rows on six engines, the motion
registers at `render_driver.py:1244-1254`, and the meta-brief framing at
`otr_meta_brief_image_prompt.py:146/:263/:1346`):**

| term | why |
|---|---|
| `shot`, `shoot`, `shooting` | camera vocabulary in nearly every composed prompt |
| `tank` | period water/gas tanks |
| `axe`, `hatchet` | tools as often as weapons |
| `blackjack` | the card game |
| `club` (bare) | card suits and night clubs |
| `poison`, `harpoon`, `dynamite` | not a depictable hand-prop silhouette |
| `gunfire`, `gunshot` | events, nothing to draw |
| `at gunpoint` | an idiom, not a prop |
| all verbs, all gore | **bananas replace the instruments, never the stakes** |

**RESERVED** (closure-safe replacements exist; held to keep v1 to unambiguous
hand props): `bomb`, `missile`, `torpedo`, `cannon`. Trap, tested: adding
`cannon -> banana cannon` breaks single-pass closure.

**Closure, enumerated not asserted:** the test enumerates the ACTUAL Cartesian
product of shipped pools (SIDEARM x LONG = 4 episode tables) plus the pinned
rows, and asserts no source regex matches any instantiated replacement string,
and that no multi-word source can assemble across replacement text. Full
replacement lexicon: `banana(s)`, `red`, `long`, `plantain(s)`, `bunch(es)`,
`of`, `beam(s)`, `peels`, `man`, `men`, `wielding`, `a`.

**Idempotence as a property:** `r1 = apply(x); r2 = apply(r1.text)` asserts
`r2.text == r1.text and r2.substitutions == 0`, over every episode table.
Idempotence-by-construction is also what makes the seven historical adapter
re-call sites harmless with NO stamp machinery -- the arc's D2 stamp is not
built (Fable cold-r1 simplification, matching this contract as written).

**ACCEPTED RESIDUAL, stated so 7d has an owner:** verbs, events and aftermath
pass untouched BY DESIGN ("two shots rang out", "standing over the body") --
rewriting them is the ripped guardrail in a funnier hat. Episodes on the
original lane will therefore MIX banana props with straight-played aftermath.
The mitigation is the receipt + the pre-publish eyeball, never a classifier.

**ANTI-ROT PROPERTY TEST (Fable cold-r1, adopted):** the suite asserts that NO
table source term regex-matches any of OTR's own composition constants -- the
motion registers (`render_driver.py:1244-1254`), `_CHAR_FACE_FALLBACK_PROMPT`,
`_INTENT_CLAUSES`/`_ARC_CLAUSES` values, and the still-plan
`framing_geometry`/`style_tail` strings on every registered engine. A future
vocabulary row that collides with camera/framing language fails CI instead of
shipping "cinematic establishing hit with a banana."

**Verify-at-build:** cross-check `_otr_content_safety.EXPLICIT_WEAPON_TERMS`
(`:48-71`, 22 period-appropriate terms, zero camera collisions) against the
table; adopt any missing unambiguous period hand-prop noun (e.g. `luger`)
subject to the same closure + anti-rot gates. Harvest, not import.

Model-prior notes carried from Fable (probe INCLUDED tokens only, QA ruling 8):
before the first live leg, five-prompt probe `red banana` and `plantain` on the
shipped still engine. Excluded-cultivar prose (Manzano/Burro/Lady Finger) is
history in the arc artifacts, not contract.

## 4. THE CHARACTER CAP (QA ruling 3 -- the arc's r2 call reversed)

`finish_visual_prompt()` promises a word-boundary cap on the FINISHED string
(`_otr_story_brief_helpers.py:619-632`, applied at `render_driver.py:2857`
against the 188-char original-branch budget). `gunman -> man wielding a red
banana` is +19 AFTER that cap, which would break an advertised contract.

**Ship a phrase-safe post-transform cap:** after the banana pass, if the text
exceeds the branch budget it is re-trimmed at a word boundary that NEVER splits
a multi-word replacement mid-phrase and preserves the trailing
`no on-screen text` clause; `sha256_after` is computed AFTER capping. The cap
utility lives in the banana module (pure) and is exercised by unit tests with
prompts built to land the boundary inside `man wielding a red banana`.

## 5. STILL LANE (no widget -- the gate is env + ledger)

At `otr_image_gen_dispatcher.py:1000-1001` (transform BEFORE
`_prompt_content_hash`):

```python
prompt = append_visual_safety_clause(str(obj.get("prompt") or ""))
if _banana_gate(ledger_meta):            # env + bank idiom, section 1
    _b = _otr_banana_route.apply(prompt, variety_key=freeze_ts)
    prompt = _b.text
prompt_hash = _prompt_content_hash(prompt)
```

* Receipt keys stamped on EVERY image row in `ledger["images"]["images"]`,
  UNCONDITIONALLY -- cache-HIT path (`:1176-1187`, exactly as
  `derived_from_portrait_hash` is, for the reason its comment gives) AND fresh
  generation (`:1330-1344`).
* Extend the hardcoded `stills_manifest.json` projection (`:1497-1508`) so all
  SIX receipt keys appear on every `"stills"` item.
* One aggregate INFO line after still dispatch (count of substitutions, the
  episode's variety roll), never per-object.

## 6. VIDEO LANE (no widget -- same ambient gate; ordering PINNED, QA ruling 4)

At the funnel where every branch has converged -- as a SIBLING immediately
AFTER `_apply_visual_safety_prompt(req, shot)` at `render_driver.py:2868`
(re-pinned; the arc's `:2874` moved in the SFX rip). The pinned sequence:

1. `_apply_visual_safety_prompt(req, shot)` (NOT modified);
2. banana transform of `req["text_prompt"]` (gate: env + bank idiom; ledger is
   in scope);
3. the phrase-safe cap (section 4);
4. assign the final `text_prompt`;
5. restamp `prompt_sha8`/`prompt_chars` and the banana receipt keys into
   `observability` WITHOUT logging (`_stamp_prompt_meta` at `:1466-1481` logs
   every call -- do not call it; write the keys directly);
6. only then the request-hash/seed derivation below (`req_hash` is the SAMPLER
   SEED, not a cache key -- D3 was withdrawn; holding the video seed equal
   across ON/OFF is desirable and is what makes the A/B the same noise).

Because the gate is ambient (env) + ledger (bank), every
`build_request_from_shot` call sees the same state -- including every prebuilt
multi-segment request through the `functools.partial` at `:3841`. No flag
threading exists to forget (the QA's widget-threading MUST-FIX dissolves under
the no-widget ruling; the test asserts the RESULT instead: every prebuilt
segment request in a multi-clip beat carries the banana receipt when the gate
is on).

`banana_sha256_after` hashes the final `VideoRequest.text_prompt` handed to the
adapter -- NOT any provider payload (adapters may append their own clauses
afterwards; that is theirs).

**Split-switch honesty (Fable cold-r1, folded without reopening ruling (b)):**
stills-ON/video-OFF (or the inverse) is a real capability ONLY on t2v lanes.
On i2v lanes the anchor still carries the look, so a split state invites the
model to morph a banana anchor back toward the gun the text names. One INFO
line (never LOUD -- cut list) when the two effective gates disagree for an
episode, and this sentence in the module docstring. Related fact worth
knowing: the compact talking register often cuts the weapon clause out of the
VIDEO prompt on face beats, so **the still funnel is the load-bearing joke
site; the video funnel is mostly consistency.**

**`OTR_LTX_RADIO_PROMPT`:** the operator override passes through the banana
gate like every other prompt (the env switch is the escape hatch). Amend its
"verbatim, unfinished" log wording to "verbatim except the banana route" so
the log stops lying when the route fires.

**The other visual prompt surfaces, decided (7b):**
* **Motion clause** (`_motion_clause_override`, `render_driver.py:1393-1404`):
  default OFF and leaks nothing today. OUT OF SCOPE **with the in-code note**
  the kickoff demands, at the function: "banana route deliberately does not
  filter this surface because it is default-OFF; if you enable it, route its
  text through `_otr_banana_route.apply` or record why not."
* **Negative prompts + style tails:** swept at `9c686886` against the full
  source-term list -- clean (no weapon nouns; the WAN/razzle negatives carry
  camera vocabulary only). Recorded here; not re-checked at runtime.
* `ray gun` appears as an OTR-authored SUBJECT (`otr_meta_brief_image_prompt.py:802`);
  it is IN the table and transforms on the original lane -- that is the gag
  working, not a leak.

## 7. RECEIPT (six keys, both lanes)

| key | ON | OFF |
|---|---|---|
| `banana_route` | `"on"` | `"off"` |
| `banana_table_version` | `TABLE_VERSION` | `TABLE_VERSION` |
| `banana_substitutions` | int | `0` |
| `banana_sha256_before` / `_after` | pre/post (post-cap) | identical |
| `banana_varieties` | `"SIDEARM=red banana,LONG=plantain"` | same (what WOULD apply) |

Video keys live inside `observability` (`VideoRequest` is `extra="forbid"`);
**extend the fixed copy list at `render_driver.py:3728-3731`** or they never
reach the node-92 `/history` report. The durable video trace is ONE beat-level
receipt; segment coverage is proven by test, not claimed from the trace.

## 8. ACCEPTANCE

**Baseline FIRST:** capture exact still + video prompt STRINGS at `9c686886`
(worktree or pre-edit run). Then:
* **Fidelity lanes, no env:** prompts byte-identical to baseline; receipt says
  `off`; still cache keys unchanged.
* **Original lane, `OTR_BANANA_STILLS=0` + `OTR_BANANA_VIDEO=0`:**
  byte-identical to baseline (the opt-out is the no-op-at-rest proof now that
  the default is ON).

**BYTE-IDENTITY EXCEPTION (QA fix pass, 2026-08-07).** The two byte-identity
claims above mean **banana-transform-identical**, not composer-identical. The
QA pass repaired two defects in the CARD COMPOSER itself, which runs before and
independently of the banana gate, so their inputs cannot be byte-identical even
with the route OFF:
1. a card line or episode title carrying a BACKSLASH is now scrubbed (an odd
   trailing run made the card's closing quote read as escaped, which dropped the
   quote shield and exposed the whole prompt to substitution); and
2. the ERA TAIL is now quote-folded (LLM-authored `atmosphere_line` /
   `visual_palette` could introduce a second double-quote pair, which wrongly
   SHIELDED whatever it wrapped -- a quoted `"revolver"` survived into the
   picture).
Both are deterministic and apply on every lane including OFF and fidelity.
Prompts with neither a backslash nor an inner era quote remain byte-identical;
the two defect classes get the repair. OFF and fidelity fixtures cover both.
* **Original lane, defaults:** mapped terms transform; receipts stamped on
  cache-HIT rows, fresh rows, `stills_manifest.json`, and the video trace;
  VIDEO seed bundles equal across ON/OFF; STILL seeds legitimately move
  (`:1000-1001` hash-after-transform, request_hash mode).
* **Dry step before any GPU:** compose prompts with the gate forced on/off in
  a unit harness, grep for mapped terms, pin bank/source/seed. Probe script
  under `tmp/`, output pasted into the acceptance record, probe deleted before
  commit.

**Live leg (the QA's exact route):** fresh selective reset + UTF-8 boot; load
`workflows/otr_canonical.json`; force all exercised roles to `ltx_audio_in`
via `OTR_FORCE_ENGINE_MAP` (adapter declares
`required_inputs = ("text_prompt", "audio_ref", "init_image")` and
`accepts_still = True` -- verified at `eng_ltx_av.py:1339/:1358` at this HEAD);
fail preflight if the effective adapter differs. The default procedural graph
is INERT for this route (`viz_*` engines ignore `text_prompt` and mint no
stills) -- a user flipping the env on the stock graph and seeing no bananas is
reading it correctly; this goes in the module docstring. Confirm the episode
asset under `otr/episodes/<ep>/` AND `otr/obs/`, eyeball one frame for the
banana, its period fit, and 7d legibility (a MIXED half-banana episode on
partial coverage is expected -- judge before publishing).

**NOT asserted:** that ON/OFF renders "differ only by the banana." Equal noise
does not guarantee that; composition is visual QA.

## 9. TESTS (new file `tests/test_banana_route.py` + integration touches)

Unit: longest-first enforced over the table; closure enumerated over the real
Cartesian product; idempotence property per episode table; camera corpus
unchanged (`cinematic establishing shot`, `cinematic medium shot`, `wide
shot`, `head and shoulders`); case rules incl. proper-noun capitalization;
quote machine (same-style pairing, curly/straight, odd/even backslash runs,
unmatched delimiters literal); whitespace/punctuation preserved around
substitutions; the phrase-safe cap never splits `man wielding a red banana`
and preserves the trailing clause; guarded env reader (garbage value -> default
+ warning, never raise); bank idiom incl. `shakespeare_v2` variant
normalization and the override key; `select_varieties` determinism + index-0
identity; OFF path byte-identity at both hash boundaries.

Integration: receipt on the cache-HIT still row and in `stills_manifest.json`;
every prebuilt multi-segment video request carries the receipt when on;
promptless procedural families unaffected; direct render callers with no
ledger meta take the global default; the video observability copy list carries
the six keys into the trace.

## 10. GATES (the standing pipeline)

Focused tests -> full suite -> Bug Bible -> AST/BOM/zero-byte on touched files
-> **Sonnet QA on the diff -> Fable gate** (standing final-review pipeline) ->
ONE pathspec commit (module + two funnel edits + motion-clause note + tests +
this spec's status flip) -> push -> `HEAD == origin`. **No canonical JSON
change, no variants regen, no re-baseline obligation** -- if the diff touches
`workflows/`, stop: a decision was made against that.

## 11. CUT LIST (cumulative; do not reopen without new evidence)

Spoken-line transformation. Per-bank default keys beyond the exclusion idiom.
The node widgets and every canonical-workflow edit (final operator ruling).
Rotated variety beyond SIDEARM+LONG (TINY/SCATTER/ANTIQUE/ENERGY pools are
recorded rungs). The randomized-verb axis (`wielding` ships). Frequency caps.
Video request-hash mutation. Engine-aware vocabularies. A new ledger
subsystem. LOUD mismatch warnings (operator input is INFO). The end-credits
joke. Per-adapter provider-payload hashing. `at gunpoint` (and `banana-point`
-- not depictable; re-proposed by Fable cold-r1, stays cut). The excluded-
cultivar probes. The unsupported provider-moderation claim. Weapon terms in
NEGATIVE prompts, either direction -- "helping the joke" with `gun, revolver`
negatives is guardrail-shaped and re-arms the seam the 08-05 rip emptied.
Period-styling the banana -- the bright modern banana IS the legibility marker
(Fable position, adopted): it inherits the era tail like every object and
spends zero prompt budget; one eyeball on the first leg.
