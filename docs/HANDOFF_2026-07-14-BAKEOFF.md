# Handoff -- 2026-07-14 morning: the 420 + 720 bakeoff is ready to run

**Branch:** `v2.0-alpha`. **HEAD == origin.** Suite 7901 passed / 31 skipped /
1 xfailed. Bug Bible 17/16/3.

---

## UPDATE 07:20 -- THE SWEEP IS RUNNING; scifi_codex NEEDED ONE MORE FIX

**HEAD is now `80e30c2e`** (suite 7903 / 31 skipped / 1 xfailed; Bible 17/16/3).

**The "third strike" on scifi_codex was a PHANTOM -- do not count it.** The P2
clamp fix was committed at **06:57:36**; the 30w leg that "failed a third time"
ran **06:42-06:45**, BEFORE the fix existed. The chain script fired the re-roll
off a stale HEAD. Two-strikes never tripped. (The live ComfyUI tree at
`C:\Users\jeffr\ComfyUI-Installs\...\custom_nodes\ComfyUI-OldTimeRadio` is a
**junction** back to this repo, so a file-tool edit IS live the moment the server
next boots -- verified, not assumed.)

**The P2 clamp fix is now LIVE-PROVEN.** On the 420w leg (`f6c42c5f`) P0, P1, P2
and P3 all landed clean. P2 never even warned.

**Then P4 killed the leg -- and it was the same bug class.** `StructureReviewV4`
is `verdict` (a Literal), `issues` (<=120-char critic notes) and `rationale` (a
<=240-char critic note). Mistral-Nemo returned `verdict="rewrite"` with a 240+
char rationale, the typed repair wrote a long one again, and the episode died.
**An audit killed a story over the length of its own footnote.** P4 was the ONE
audit pass explicitly opted out of the clamp; its siblings P6 (`ListenerReviewV4`)
and P8 (`FinalAuditV4`) both take the `clamp=True` default and were never exposed.
Fixed in `80e30c2e`, plus executable coverage pinning **all three** audit passes
(P4/P6/P8) against ever being fail-closed on string length again. No ledger hole:
the clamp preserves the field, so `verdict`/`issues`/`rationale` all still land.

THE LAW: *an audit may improve a story, it may never fail one.*

### What is running, unattended, right now

A single serialized GPU chain -- do NOT start another render:

1. `tmp\_go_bake420.ps1` -- 420w, all 10 banks (RSS banks interleaved).
   `scifi_codex` led and is RED (the P4 kill, pre-fix).
2. `tmp\_chain_codex_releg.ps1` -- waits for (1), then re-legs `scifi_codex` at
   420w against the P4 fix. Sweep name `bake420b`.
3. `tmp\_chain_720.ps1` -- waits for (2), then **GATES**: 720 launches only if all
   10 banks are SUCCESS at 420. If any is red it writes
   `tmp\bake720_GATE_BLOCKED.txt` naming them and stops. Nobody skips the ladder.

**Monitor:** `powershell -File tmp\_status_bake.ps1` (all three sweeps at once).
**Metrics:** `tmp\bakeoff_metrics.py` emits the per-leg beats / voiced lines /
words-per-line / `actual_ratio` table the verdict needs. **Run it only when the
GPU is idle** -- reading episode files mid-render collides with the per-beat
ledger saves (WinError 5).

**The goal, in the operator's words:** *"I want the 420 and 720 words to go so we
can see what story models are worth keeping."* The bank is the ONLY variable.

---

## THE PINNED PAIR (identical on every leg -- do not change it mid-event)

| Slot | Model | Notes |
|---|---|---|
| creative | `aion-labs/aion-3.0-mini` via OpenRouter | ctx **131,072**; mandatory-reasoning model |
| technical | `mistralai/Mistral-Nemo-Instruct-2407` (local) | ctx 8,192, deterministic, free |

**THE WIRING GOTCHA THAT WILL BITE YOU.** You CANNOT set
`creative_writing_model=aion-labs/aion-3.0-mini`. That slug is HF-shaped, so
`validate_model_id` accepts it and the loader then tries to pull it from
HuggingFace -- it never reaches OpenRouter. The creative dropdown only offers the
HANDLE. The correct patch set is:

```
creative_writing_model   = openrouter:slot-a
openrouter_slot_a_model  = aion-labs/aion-3.0-mini
technical_model          = mistralai/Mistral-Nemo-Instruct-2407
```

Confirm on every leg that the server log says:

```
[OpenRouter] load slot=A handle=openrouter:slot-a slug=aion-labs/aion-3.0-mini route=default ctx=131072 (remote, 0 VRAM)
```

`ctx=131072` is the proof the context fix is live. If it says `ctx=8192`, stop.

---

## HOW TO RUN (the harness is written and proven)

Everything is in `tmp/` and already works. The env hydration, the widget patches,
and the sequential driver are all handled:

```powershell
# ONE leg
powershell -NoProfile -ExecutionPolicy Bypass -File tmp\_leg.ps1 -Bank <bank> -Words <n>

# A SWEEP (sequential, one GPU, a failure does not stop the sweep)
powershell -NoProfile -ExecutionPolicy Bypass -File tmp\_sweep.ps1 `
    -Banks @("bank1","bank2",...) -Words 420 -SweepName "bake420"
```

- `tmp\_leg.ps1` hydrates `OPENROUTER_API_KEY` from USER scope into the SAME
  shell that launches the server (a stale shell does not inherit it and the
  creative call dies at P1), sets `OPENROUTER_MAX_TOKENS_PER_RUN=1000000` (the
  300k default can abort a long leg 30 minutes in), and applies the four widget
  patches above.
- `-Set` is `[string[]]`: it must be a REAL PowerShell array. Passing
  `"a=1,b=2"` through `powershell -File` binds the whole comma string as ONE
  element and shoves it into `source_bank`. `_leg.ps1` builds the array literal
  inside a `-Command` string; do not "simplify" it.
- Results land in `tmp\<SweepName>_<N>w_summary.txt`, one line per leg with
  EXIT, RESULT, prompt_id and elapsed seconds. `tmp\_status.ps1` prints a
  compact view (edit the sweep name at the top).

**Spacing matters.** Five banks fetch science RSS. Do not run them back to back
if you can interleave -- see the RSS note below.

---

## WHERE THE 30-WORD SMOKE LANDED (all on the bakeoff pair)

| Bank | 30w | Prompt / note |
|---|---|---|
| `original_codex56sol` | GREEN | `411c2f17` -- 65.5 MB, *The Hidden Door* |
| `science_news` | GREEN | `4691fad1` |
| `media_archive` | GREEN | 707s |
| `public_domain_story` | GREEN | 745s |
| `shakespeare` | GREEN | 695s -- also requalifies PBUG-20260713-19 |
| `original_radio` | GREEN | 805s -- proves the LLM-veto rip (`d07e6a75`) |
| `scifi_fable2` | GREEN | 400s |
| `scifi_gemini` | GREEN | 609s -- unblocked by the bs4 fix |
| `scifi_sonnet` | IN FLIGHT | outlet-acronym fix pushed; leg was running at handoff |
| `scifi_codex` | RED | P2 clamp fix pushed but NOT yet live-proven -- see below |

**`scifi_codex` is the one open bank.** Three distinct failures, three root
fixes, each pushed and unit-green; the last one has not had a live leg yet:
1. P3 `patch_json` -- the text-patch budget was a flat 1,024 sized for the
   grammar-constrained LOCAL transport. Now sized from the real targets
   (`994c7d85`).
2. RSS -- environmental, see below.
3. P2 `cast.2.role_in_conflict` string_too_long -- `clamp_overlong_strings` was
   fail-closed on a CastPlanV4 that contains no spoken prose, only tags. Now
   clamps at a word boundary (this commit).

**FIRST ACTION FOR THE NEXT WINDOW:** re-run `scifi_codex` at 30w. If it is
green, run the 420 sweep on all 10. If it fails a THIRD time on the same
problem, the operator's TWO-STRIKES rule applies -- `/kibitz` BEFORE writing
more code.

---

## WHAT WAS FIXED TONIGHT (all pushed)

* **`a3a48290` -- beautifulsoup4 was never installed, and is not in
  requirements.txt.** `_fetch_full_article` does `except ImportError: return ""`,
  so it had been returning empty INSTANTLY, SILENTLY, FOREVER. Every
  science-sourced episode this project has made was written from a ~120-character
  RSS teaser instead of the article body. Measured on the same feed: **0 chars
  before, 2,041 and 6,708 chars after.** A missing package looked exactly like a
  paywall, and the v4 source floor then blamed the FEEDS. Both now fail loud.
  `science_news` never noticed because it runs `require_science_floor=False` and
  silently shipped the teaser; the scifi lanes hard-failed. **The lane that
  looked broken was the honest one.**
* **`32e680b2` -- PBUG-20260713-20:** a remote model's context window was read
  from the STATIC virtual catalog row (8,192). Aion is 131,072; hy3 is 262,144.
  Every remote call ever made was budgeted against a window 16-32x too small. At
  720 words `original_codex56sol` P6 asks for 9,520 output tokens and would have
  been silently clamped -> script cut off mid-JSON -> dead leg blaming the model.
* **`d07e6a75` -- the last LLM veto.** `original_radio` died on
  `epilogue_moralizes` -- an aesthetic opinion about an outro the model had just
  rewritten AT THAT AUDIT'S OWN REQUEST. Now ships over subjective objection.
  THE LAW holds: an audit may improve a story, it may never fail one.
* **`a3a48290` -- two more 720 budget landmines:** scifi_codex's `min(5400, ...)`
  ceiling (needs 6,960 at 720w/24 lines; a TEST asserted the clamped value as
  correct) and `original_codex56sol` P5's flat `tokens=3600` while the prompt
  hands the model `max_beats: 40`.
* **`a3a48290` -- scifi_sonnet** was killed for saying "BBC", the outlet its own
  attestation seam ORDERS it to name. `allowed_spoken_all_caps` read every
  dossier surface except `provenance_note`. `_allowed_numbers` already carried it.

Deliberately NOT changed: gemini's and fable2's output ceilings. Gemini's is
load-bearing (it guards P3's repair envelope on the 8,192 local slot -- live
2026-07-11) and neither binds at 720.

---

## THE TWO THINGS THAT WILL BITE THE BAKEOFF

**1. RSS rate-limiting / source availability.** Five banks (`science_news`,
`scifi_fable2`, `scifi_codex`, `scifi_gemini`, `scifi_sonnet`) fetch the same
science feeds. Running eight legs in an hour got the body-fetch throttled and
three legs died with "No science RSS candidate met the v4 source floor". The
420/720 legs are 15-45 min apart, which is far more cooling, but **interleave the
RSS banks with the non-RSS banks anyway** (`shakespeare`, `public_domain_story`,
`media_archive`, `original_radio`, `original_codex56sol` do not need it).

**2. THE BANKS DO NOT WRITE THE SAME SHAPE OF STORY AT LENGTH.** This is the
finding that most threatens the verdict, and the operator has ACCEPTED it and
ruled: *"target words are not chaseable ever"* -- do NOT normalize structure.
Record it instead.

| Bank | structure at 720w | words/line |
|---|---|---|
| `original_codex56sol` | 40 beats (`max(8, min(40, tw//4))`) -- the ONLY lane that scales | ~18 |
| ledger lanes (science_news, media_archive, public_domain, shakespeare, original_radio) | 14 voiced beats, FIXED | ~51 |
| `scifi_codex` | 12-24 lines (3 scenes x 4 beats x 2), FIXED | 30-60 |
| `scifi_gemini` | 6 beats, hard-coded | one beat carries 180 |
| `scifi_sonnet` | 5 events, hard-coded | ~144 (five monologues) |

At 720 words codex56sol writes a 40-beat conversation and sonnet writes five
speeches. **These are not the same show.** The VERDICT must state this as a LANE
PROPERTY, not a bank quality -- otherwise it ranks "who writes a 144-word
monologue" against "who writes 40 short exchanges" and calls the difference a
bank effect.

**So record per leg:** `prompt_id`, title, asset size, beats, voiced lines,
words/line, and `meta.word_budget.actual_ratio` (the word target is ADVISORY and
WARN-only -- a bank that undershot to 0.71 is not competing at 720).

---

## THE PLAN

1. Re-run `scifi_codex` 30w. Green -> proceed. Third same-problem failure ->
   `/kibitz`.
2. **420 sweep, all 10 banks** (the operator explicitly wants the 420 rung before
   720 -- nobody skips the ladder).
3. **720 sweep, all 10 banks.**
4. **VERDICT** per `docs/2026-07-11-720-bakeoff-kickoff.md` GATE 3: strip lane
   identifiers, randomize as Story A..N, seal the key, TWO independent Fable
   judges score /10 (premise, structure, character distinctness, dialogue, ending
   payoff, source-fact integration, pacing) with a paragraph rationale each;
   anchor judge reconciles on ARGUMENT QUALITY, never by averaging; unseal; write
   `docs/2026-07-14-720-bakeoff/VERDICT.md` with the ranked table, the rationales,
   the word-count/structure table, and one page on what the winning lane got
   right. **The operator's blind listen is the final crown; the doc is advisory.**

Every leg must prove: `RESULT SUCCESS` + `obs_publish OK` + `Test-Path` the asset
under `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\`. API success is not
proof.

## OPERATOR LAW (new, 2026-07-14 -- both now in CLAUDE.md)

* **TWO STRIKES, THEN THE PANEL.** Two solo attempts at a problem; the THIRD must
  begin with `/kibitz` before any more code is written.
* **RIPPING AN LLM IS ALLOWED. A HOLE IN THE LEDGER IS NOT.** If you remove or
  repurpose an LLM pass, enumerate every ledger field it wrote and give each one
  a new owner before you delete the call. Downstream (TTS, slicing, video,
  captions, credits, obs_publish) reads FIELDS, not intentions. A green unit
  suite does not prove the ledger is complete -- a live leg does.
