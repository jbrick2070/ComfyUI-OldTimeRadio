# The Evolution of OTR -- Six Months of Vibe Coding, Read From the Commit Log

**What this is.** A narrative history of ComfyUI-OldTimeRadio ("SIGNAL LOST"), assembled
from the project's own git log: 5,203 commits on `v2.0-alpha` between 2026-04-05 and
2026-09-11, plus the dated operator directives in `CLAUDE.md`, the headings of
`docs/HANDOFF_LOG.md`, `docs/OTR_STANDING_RULINGS.md` and `docs/PRODUCTION_SPRINT_LESSONS.md`,
and the README as it read on day one versus today.

**Who it is for.** Anyone who wants to see what six months of one person building a
real piece of software with AI pair-programmers actually looks like -- not the demo, the
log. It is written to be dropped into NotebookLM or read straight through. Every claim
that matters cites a commit hash you can `git show`.

**Who built OTR.** One operator, Jeffrey Brick, who describes himself as a vibe coder:
*"I can look at colors and words on a screen"* -- he does not read code fluently, but he
follows architecture, tradeoffs and consequences, and he is the one whose eye and ear
judge the output. The code was written by AI assistants (Claude in Cowork as the main
driver, with Codex, Antigravity/Gemini, Cursor, ChatGPT, DeepSeek, Grok and others as
reviewers and panelists at different points). The commits are in his name because the
decisions were his.

**How to read it.** Part I is the story, month by month. Part II is the lessons that
cut across the months -- how the way of working changed, and why. The appendix holds the
numbers, the milestone timeline and a glossary of the project's own words.

---

## The shape of it, in numbers

| | Apr | May | Jun | Jul | Aug | Sep (to the 11th) |
|---|---:|---:|---:|---:|---:|---:|
| Commits | 569 | 947 | 928 | 1,026 | 1,166 | 567 |
| Busiest single day | -- | 05-13: 103 | 06-14: 58 | 07-10: 91 | 08-29: 84 | 09-06: 95 |
| Average commit-subject length (chars) | 73 | 78 | 134 | 104 | 82 | 73 |
| Python files under `nodes/` at month end | 28 | 99 | 195 | 244 | 267 | 284 |
| Test files at month end | 60 | 245 | 408 | 495 | 650 | 759 |
| Files under `docs/` at month end | 64 | 21 | 1,591 | 308 | 610 | 959 |
| Workflow JSONs at month end | 11 | 4 | 8 | 20 | 86 | 1 |

Today the tree holds 3,207 tracked files, 158,503 lines of Python under `nodes/`, 25
public `OTR_` nodes, one canonical workflow, 282 production bug records, 456 saved
review-panel runs, and a registry listing at `2.0.0-alpha.29`.

What the assistants were asked to lean on, by month (count of commit subjects that
mention the word):

| Reviewer or method | Apr | May | Jun | Jul | Aug | Sep |
|---|---:|---:|---:|---:|---:|---:|
| Gemini / ChatGPT "QA" pastes | 6 | 5 | 7 | 18 | 0 | 1 |
| round-robin (cloud consult) | 3 | 32 | 0 | 0 | 0 | 0 |
| roundtable (paid 4-round arc) | 0 | 0 | 73 | 7 | 0 | 0 |
| kibitz (local $0 panel) | 0 | 0 | 20 | 89 | 8 | 0 |
| codex | 0 | 1 | 6 | 116 | 11 | 5 |
| Fable | 0 | 0 | 1 | 61 | 8 | 4 |
| Sonnet / Opus subagents | 0 | 2 | 2 | 33 | 6 | 8 |
| cursor | 0 | 0 | 0 | 0 | 1 | 9 |
| "panel" / "r1".."r4" / "arc" | 2 | 1 | 57 | 147 | 47 | 17 |

And what the work was about, by the same measure:

| Theme | Apr | May | Jun | Jul | Aug | Sep |
|---|---:|---:|---:|---:|---:|---:|
| `BUG-LOCAL-nnn` (bugs found by the developer) | 78 | 207 | 39 | 4 | 1 | 0 |
| `PBUG-...` (bugs found by a live run) | 0 | 0 | 0 | 8 | 91 | 25 |
| "live" / "proven" | 5 | 26 | 86 | 105 | 121 | 17 |
| "rip" / "retire" / "delete" / "remove" | 36 | 77 | 43 | 93 | 72 | 17 |
| "revert" | 7 | 6 | 1 | 1 | 2 | 0 |
| "operator" / "Jeffrey" | 12 | 9 | 63 | 50 | 58 | 23 |
| 4060 / Mac / RunPod | 1 | 1 | 1 | 2 | 69 | 102 |
| registry / pyproject / alpha.N | 0 | 3 | 19 | 15 | 28 | 44 |

Three of those rows are the whole story in miniature. The bug-id row shows the project
stop counting bugs it found by reading code (`BUG-LOCAL`, 207 in May, zero by September)
and start counting only bugs that failed in a live run (`PBUG`, zero until July). The
reviewer rows show the review budget go from paid cloud consults to a $0 local panel
to a single independent reader -- while the rigor went *up*. And the last two rows show
the project leave the machine it was born on.

---

# PART I -- THE STORY

## April 2026 -- "Ship it every day"

**569 commits. Tags: v1.0, v1.1-beta, v1.1, v1.2.0, v1.3, v1.4, v1.5, v1.6.0, v1.7,
v2.0-alpha-1 -- ten tags in eleven days.**

OTR was born finished. The first commit, 2026-04-05, is `03d5e581 v1.0 Release:
Canonical Audio Engine`. The README that day described a complete product: real science
headlines via RSS, a local Gemma 4 writing a multi-act radio drama, a second Gemma pass
acting as Director, Bark TTS voicing every line with emotional bracket tags, procedural
theremin-and-static SFX, a 48 kHz spatial master, and a CRT-aesthetic MP4. The show was
called *"Transmission From Tomorrow."* The pitch: *"Think X Minus One meets procedural
generation."*

The first week was a release a day. v1.1 on the 6th brought "Open-Close + Critique +
Context Engineering" and the first outside review -- `cdf8d3cf Gemini QA fixes` and
`253b1b38 Gemini QA round 2`, with bugs numbered BUG-004 through BUG-010. The pattern
that would define the project is already visible on day two: a model writes the code,
a *different* model is pasted the result and asked to find what is wrong, and the
operator arbitrates.

April's obsession was Lemmy. A British-accented character who turns up in roughly 11%
of episodes, he got his own RNG fix (`6c3f846f Lemmy 11% roll uses SystemRandom to
bypass per-episode seed freeze`), his own QA guide, a three-generation lineage doc
(`e70d8a91 ... Barnet -> Kilmister -> us`), and a 10,000-trial statistical sanity test.
He would still be a live workstream in August. Lemmy is the thread that proves the
operator cares about the *show*, not the pipeline.

Two things happened in April that the project would pay for and learn from all summer.

The first was the positional-widget bug. ComfyUI saves a node's widget values as a bare
list, so inserting a widget mid-list silently shifts every saved value. It bit on
2026-04-08 (`730dd6f0 Fix Positional Widget Shift in workflows ... (Bug Bible 04.01)`)
and again on the 14th, 23rd and 27th. The rule "only ever APPEND a new optional widget
at the END" is in `CLAUDE.md` today because April paid for it four times.

The second was the scope explosion. On 2026-04-12 the project declared `v2.0-alpha`
(`74afd856 ... Add visual sidecar design spec`) and on 2026-04-17 it built an entire
video stack in one sitting: fourteen commits titled `Day 1:` through `Day 14:`, all
dated the same calendar day -- backends harness, FLUX anchor, PuLID portraits, FLUX
keyframe, LTX motion, Wan 2.1 loop, Florence-2 inpaint, VHS postproc, a planner, a cold-
open canary, a 3-minute continuous scene, an SSIM identity gate, a 20-minute dry run,
and a freeze tag. It is the purest artifact of early vibe coding in the log: ambition
measured in "days" that were really hours, and a confidence that it would all wire up.
Most of it was ripped out within six weeks.

Other April firsts that stuck:

* **The soak harness** (2026-04-13). `soak_operator.py` randomized genre, words, length
  and creativity and queued episodes overnight. It also grew a critic haiku, a Rotten
  Tomatoes score, and a pair of balcony hecklers -- Statler and Waldorf, renamed the same
  day to Baba and Booey (`66cd754f`). The discipline of running the real thing overnight
  and reading what came out began here.
* **The production ledger** (2026-04-24, `f7d271b5 feat: production_ledger L1 -- write-
  only JSON record per episode`). A per-episode JSON that every downstream stage reads
  and writes. By the end of the month it had a schema version (`l3-2026-04-28`), a
  sha256 audio-gate chain, and per-line start/duration stamps. Later rules about "the
  ledger must be filled completely" and "ripping an LLM is allowed, a hole in the ledger
  is not" all descend from this file.
* **Round-robin consults** (2026-04-30, `f0421c2e P1 hardening from 2026-04-30 round-
  robin`). A script that sent the same question to OpenAI, Gemini and NVIDIA models
  and collected their answers. The ancestor of every panel that followed.
* **The autonomy contract, v1.** `65d523a5` (2026-04-15) describes a runner that "Never
  autofixes -- design decisions stay with Jeffrey." Hold that sentence; by September
  the rule reads the opposite way.
* **A hardware fact that never changed.** `593e5baf` (2026-04-08): Flash Attention 2
  has no wheel for sm_120 + CUDA 13 + torch 2.10 on Windows. It is still the first
  platform note in `CLAUDE.md`, with the instruction that no future session should
  chase it.

April also had the most reverts of any month (7), a workflow JSON restored from 0 bytes
after a bad upload (`1e104005`), a CLAUDE.md removed from tracking (`f684c6c7`) that
would later be force-tracked "so the rules file itself survives a disk loss", and a
first brush with the Bug Bible -- a sibling repo of portable ComfyUI-node lessons that
OTR would both consume and feed for the rest of the year.

**What the operator was doing in April:** writing the README by hand, choosing the
default writer model almost daily (Gemma 4 -> Nemo -> Mag-Mell -> Captain-Eris -> Nemo
again in a single week of late April), and pasting model output into other models for
review. The instrument was reading.

## May 2026 -- "The sprint machine and the clean break"

**947 commits. The busiest day in the project's history: 2026-05-13, 103 commits.
Tags: pre-bug-117a-cutover, v2.0-alpha-cleanbreak, s29-clean-slate-gate,
v2.0-alpha-stable-20260530.**

If April was a release a day, May was a *sprint* a day, and the commit subjects say so:
`B0: branch cut + S34 P0/P1 hotfix plan landing`, `B1:`, `B2:`, `B-final: Sprint S34
close`. Sprints S21 through S34 ran in the first half of the month with lettered steps,
a QA document filed "for round-robin review" at each close, and a phrase that appears
in the close commits with remarkable honesty: *"runtime NOT PROVEN."* The pipeline was
being restructured faster than it could be run.

May 1 set the tone: `08ecfcd1 BUG_LOG zeroed: all entries through 2026-05-01 promoted
to survival-guide BUG_BIBLE.yaml; fresh slate for v2.0-beta`. Then the local bug counter
ran from BUG-LOCAL-001 back up past 290 by month end -- 207 commit subjects cite one.
This is the month of finding bugs by reading, fixing them, and filing the next.

Three arcs define May.

**The clean break.** On 2026-05-11 a standing directive landed: `3d00c96e
docs(ROADMAP): standing directive -- no legacy back-compat`. What followed was a
sequence of "cleanbreak" sprints (S26, S27, S28, S29) that deleted every fallback,
shim, legacy path and alias the first five weeks had accumulated -- `cleanbreak(s28-p1-
1)` through `(s28-p1-8)` dropping one legacy path from eight files, one commit each.
The busiest day of the whole project, 2026-05-13, is almost entirely this. The lesson
being learned: a young codebase that tolerates old shapes never finds out which shape
is live.

**The 231 saga.** BUG-LOCAL-231 is the best single case study of how the method
matured. On 2026-05-18 FLUX renders were taking 170-188 seconds per step. Over two
days the log records a hypothesis ladder -- alt-a through alt-j -- each *falsified* in
its own commit: audio residue (FALSIFIED), an attention env var (FALSIFIED, though
"env-var removal stays landed ... architecturally correct regardless"), hardware
ceiling (`5ceda3aa ... EMPHATICALLY FALSIFIES alt-h -- identical RTX 5080 ... runs at
0.75 s/it vs OTR's 170-188 s/it = 226-250x regression`). A minimal stock-FLUX workflow
was built to bisect (`749a8f88`), ran at 1.40 s/it, and proved "hardware accounts for
1.87x, OTR graph accounts for 134x." Along the way the bug was *demoted* from FIXED to
PARTIAL because `d349871f ... one-run promotion retroactively violated curation rule`,
and a commit exists solely to `retract "stuck at 0/20" framing` (`c60dcecd`). Another
records `apply Jeffrey 11:10 pushback corrections 1-3, 5-8` -- the operator reading the
diagnosis and sending it back. This is the month the project learned to write down
what it got wrong, in the log, as a first-class event.

**Build it, then rip it.** The most instructive sequence in May is 2026-05-26 to
2026-05-29. On the 26th-28th the project built an entire multi-agent writers' room:
Stage 1 constrained-generation planner, Stage 2 multi-turn roleplay with best-of-N,
Editor and Director agents, a "Story Room loop", a transcript-to-structured extractor,
a ComfyUI node to commit the room's dialogue to the ledger, a fan-out node, a beat
selector -- `Sprint 10B Wave 0`, `Wave 1 Agent A..E`, `Wave 2 Agent F, G`, `Wave 3`.
On the 29th a "lean-down audit" (`b691adda`) inventoried it as cruft, and
`608eb888`, `6c0943a5`, `b0db85be` and `aad4cfbc` removed the multiturn path, the
Story Room cluster, the fan-out cluster and the shadow critic. Three days from build
to delete. The operator's ROADMAP directive that week was "model router first, then
cleanup; spine-first backup." This was not a failure; it was the project discovering
that the cheapest way to learn whether a clever architecture helps is to build it,
run it overnight, listen, and delete it in the morning.

May also:

* Moved output to `output/otr/` with an `obs/` folder for the one final MP4 per
  episode and `episodes/<ep>/` for everything else (`79ce42b5`, 2026-05-02). Every later
  rule about `otr/obs/` being "the success signal" points at this folder.
* Pruned 217 historical docs in one commit (`7896e5d3`, 2026-05-30), then 40 stale
  scripts (`8c67aa4f`), then six defunct workflows down to one (`86867b48`). The
  docs-file count went from 64 to 21.
* Added SDH open captions, on by default, "accessible-by-default" (`70bcc769`).
* Tagged `v2.0-alpha-stable-20260530` -- the first time the operator declared a
  stable point, and the origin of the rule that only his eyeball gates a tag.
* Wired OpenRouter as an optional remote writer on the 31st and immediately ran a
  story-quality baseline: local vs remote parity, then an Opus comparison at 89%
  (`36966271`). The cloud-writer question would be reopened and closed several times
  before August settled it: local stays the default.
* Began a "story spine" -- arc gate, creative QA critic, radio editor, ledger scrub --
  shipped "default-ON out of the box" on the last day of the month (`0d166203`).

**What the operator was doing in May:** reading diagnoses and pushing back line by
line (the 11:10 corrections), revising specs mid-flight (`6ef2452a ... Jeffrey revised
spec`), choosing between "option 3 per Jeffrey" style forks, and starting to say *no*
to complexity. The lean-down was his call.

## June 2026 -- "The platform, the rules, and the first real episode"

**928 commits. Average commit subject: 134 characters -- the longest of any month.
Tags: m0-a-seam-merged, m1-first-episode, cw8-1 .. cw8-5, A-ship, B-ship.**

June is the month commit messages became essays. A typical subject from 2026-06-10 runs
to 600 characters and contains its own root-cause analysis, live evidence, suite counts
and a Bug Bible result. That is not noise. It is what happens when the person reading
the log cannot read the diff: the message has to carry the *meaning* of the change, or
the change is invisible to the one person who decides.

Three arcs define June.

**The model-agnostic video platform, in a weekend.** Between 2026-06-06 and 06-08 the
legacy FLUX/HuMo/LTX batch render path was torn out and replaced with a registry-driven
platform: an "A-Seam" core (`a3581517 ... CW-1`), engine registries for image and video,
role-compat filters, a gpu-residency lease, a portrait ledger, a render-time retry
taxonomy, a silent composite plus a terminal audio mux (`dba788f4 ... OTR_SilentComposite
+ terminal OTR_MasterAudioMux`), eight image engines added as "peers" in a single day
(`950daaf5`, `5588c781`), and the six legacy batch nodes deleted outright with their
seventeen tests (`70d379b7`, `58ef7015`). The tags tell the sequence: `m0-a-seam-merged`
on the 6th, `m1-first-episode` on the 7th, `A-ship` and `B-ship` on the 8th. This is the
architecture OTR still has: any engine, any slot, and a `fallback_engine=None` contract.

**The production restore, and the rules it paid for.** 2026-06-10 is the hinge of the
whole project. Commit `18e3cbdf` reads, in part: *"the writer-LLM resolver called a
make_generate_fn signature that never existed -- EVERY live run silently fell to
template prompts."* Every episode for some unknown stretch had been written from canned
text while the suite stayed green. The same day's handoff (`25ca68cc PRODUCTION RESTORE
session record -- the saved-workflow punch list, the 6 unpushed commits ...`) names the
second wound: six commits that existed only on one disk. Out of that day came the first
hard rule in `CLAUDE.md`: `1351d783 GIT POLICY -- commit+push together on v2.0-alpha,
every green commit, immediately; the eyeball gates tags/promotions, never pushes
(operator directive 2026-06-10: never lose work to local-only commits). CLAUDE.md force-
tracked: the rules file itself must survive a disk loss.`

Three days later came the second wound and the second rule. A node and a new blend
input shipped, tested, and *unwired* -- the code ran dormant in production because the
workflow JSON never referenced it. The operator's phrase, preserved in `CLAUDE.md`
section 0: *"your updates are for naught."* The rule: `workflows/otr_canonical.json` IS
the workflow; any code change is dead unless the same change wires it there; every
headless run loads that file and no other. On 2026-06-14 the `CLAUDE.md` rewrite
`56469bdb docs(CLAUDE.md): true Cowork optimization -- add 'how Cowork actually works
here'` wrote down the two-filesystem model, the PowerShell traps, the 60-second tool
ceiling, the mount lag and the stale-lock recipe -- the operating manual for an AI
working on this particular Windows box, written the day each trap was paid for.

**The roundtable, and then the $0 panel.** June is when "roundtable" appears 73 times.
The format: two or three frontier models per round, four rounds (R1 high-level arc, R2
coding plan, R3 wiring, R4 convergence), Claude as a code-grounded panelist *and* the
sole judge, and -- written into `CLAUDE.md` section 8 on 2026-06-21 (`82fbd4ec`) --
"never dry-run, never pre-compute the cost, just spend." The spends were small and are
recorded in the subjects: "~$0.05 total panel spend," "~$0.09," "~$0.41," "~$2.29" for
a four-round story-engine assumption audit. The panels caught real things: a
"converged plan" for an opener bug that instrumentation then *overturned* (`2570afd1
instrumented findings overturn the converged plan`), and the next panel rejected the
prior panel's fix.

Then on 2026-06-27 the economics changed. `fdbfa977 feat(roundtable): robust headless
Codex runner` and `684a7064 AntiGravity headless SOLVED via FILE-HANDOFF` made two
local CLI agents usable as reviewers with no API spend -- Codex reading the repo in a
sandbox and writing its review to a file, Antigravity doing the same. The first commit
to say "kibitz" is the same day (`5f0bab70 docs(kibitz): HuMo bakeoff r1`). Within a
week the word "roundtable" nearly vanishes from the log and "kibitz" takes its place.
The method was the same four-round arc; the bill went to zero.

June also:

* Flipped story-quality v2 ON by operator directive after a soak "proved the spine
  stable + harmless (15+ clean eps)" though "not a measurable lift on weak local
  writers" (`550679d7`, 2026-06-23). The operator's phrasing on a sister flip the next
  day became a rule: *"if it makes a better story it's the default, not a lever to
  find"* (`57279156`).
* Ran honest bake-offs: HuMo 14B vs 1.7B (`2646688d FINAL VERDICT -- 14B fp8 wins
  100%`), LTX-AV GGUF quants (13 legs, a winner, and a Codex catch of "a real decode-spec
  contradiction").
* Wrote the first version of THE LAW's ancestor: `c8f0156c ... NO-FALLBACKS hard-fail
  directive + rip-out-all-fallbacks production cleanbreak sprint` (2026-06-29), and
  `c6ca5d88 C2: remove requires_flag GATE (registry IS the menu)`.
* Saw the docs folder swell to 1,591 files -- mostly roundtable pass artifacts, and a
  tell that the review machinery was generating more paper than the code.

**What the operator was doing in June:** look-QA. The log records "operator look-QA
round 3," "round 5," "operator eyeball," "operator catch," "the c01 giant-mic catch" (a
negation in a prompt *planted* the microphone it forbade). He stopped soaks he did not
need (`e729f7e9 ... operator stop-soaking`), flipped defaults by directive, and on the
10th wrote the git rule himself. His instrument was now his eyes on the rendered
frames.

## July 2026 -- "Kibitz month, rip month, the Law"

**1,026 commits. 144 saved kibitz runs. Codex named 116 times, Fable 61, the word
"panel" or a round number 147 times.**

July is when the method hardened into law -- and when the project began ripping out
its own safety rails on purpose.

**No fallbacks, anywhere.** 2026-07-02: `d0463b8c directive: NO fallbacks / NO auto-
defaults anywhere -- the shipped workflow JSON dropdown values are the ONLY defaults`.
The next two days executed it: `8de5862d Sprint A E1/E2: rip the render fallback
machinery out`, then ten sites of LLM fallback turned into loud failures -- audio-voice,
scene_sequencer inline Bark, image-lane, dramatic-state, announcer intro/outro/coda,
image-prompt, casting, ShotLock (`822cb0c9` through `26b236e6`). The Fable model was
used as a gate on the borderline cases (`547b5801 R3 borderline resolved via Fable --
rip J1+J3 (canned text), keep J2 (structural)`). The reasoning, which the project
would restate many times: a fallback that fires silently is a wrong render that looks
like a right one.

**The VRAM rip, and a Fable catch.** 2026-07-03, five chunks (`21292478` ..
`4fa85282`) removed every VRAM tier label, ceiling widget, class estimate and runtime
OOM assert. A grounded Fable fan-out on the rip found a chunk-order KeyError that would
have broken every production render and that both Codex and a general-purpose reviewer
had missed. That became `CLAUDE.md` section 9's "reality exception" the same day
(`4ef2ffe7`): Fable is a scalpel, but it is allowed as the final gate on a high-stakes
structural change.

**A four-round arc whose answer was "don't."** 2026-07-04: a plan to transplant the
prompt library into a second repo went through kibitz r1-r4 (`c98a67ab` .. `1747e2dd`)
and was then `1a3571e1 ... PARKED by operator decision -- reject two-repo transplant
(unanimous kibitz)`. A full review arc that concluded the work should not happen is
recorded as a success. It is one of the clearest signs the method had matured: the
panel's job was to break the framing, and it did.

**Source banks and visual styles, in a day.** 2026-07-05, ten commits with subjects
the length of design docs (`4f611cb3`, `2e59d76b`, `c24dc0fa`, `8da76394` ...) built
the "multimodal story schema": source banks as JSON packs (science news first), visual
styles as JSON packs with byte-identical default tails, story-content rule packs, and
two new selector widgets appended at the END of the writer's widget list with every
positional pin updated in the same commit -- the April lesson, finally structural. On
2026-07-08 `497e7b9e Add Shakespeare source bank lane`. By the 12th the README listed
ten runnable banks.

**The first green episode of the new kind, and the production bug log.** 2026-07-10 is
the month's busiest day (91 commits). `ff4c226d scifi_fable2 S1b live-smoke hardening:
FIRST GREEN EPISODE (Einstein's Echo in obs, 570s) -- 25 rolls, every class root-fixed`.
The same day opened `docs/PROD_BUG_LOG.md` (`05088e32`) with its admission rule: Claude
appends autonomously, but *only* for bugs that failed in a live run; review and audit
catches get fixed, never logged. Twenty-six prior production failures were backfilled
from the June-July history. This is the moment the bug system split into "what a
reader found" (not a bug) and "what the pipeline did to a real episode" (a PBUG).

**The bake-offs, and the banks that lost.** Mid-July ran the banks against each other
at 30, 420 and 720 words -- Codex-, Gemini-, Sonnet- and Fable-authored writer lanes
side by side, with Fable doing blind judging (`9d8265c0`). Then the verdicts were
executed: `3312aec7 rip(banks): remove scifi_gemini + original_codex56sol source lanes
(bake-off LEAVE verdict)`, `499386aa rip(roster): trim to 10 independent lanes`, and
on the 18th `c507acff rip 4 source banks (Sonnet bake-off verdict)`. The rip plan for
those four banks went through eight review folds in one day (`2e722fa6` .. `f7dbba69`),
including an *operator* r4 flag that found the plan's "bare-sonnet invariant was FALSE
(2 carve-outs, not 1)" -- the non-coder catching the reviewers.

**THE LAW, and two strikes.** Two operator rulings from July still govern everything:

* 2026-07-14, `bcc6a3de docs: TWO STRIKES, THEN THE PANEL`. Two solo attempts to fix a
  bug; if the third is about to start, the same bug has survived two fixes, the model
  of the problem is wrong, and a panel runs before any more code. The same day's
  `34f24dd3 docs(bakeoff): the P4 audit kill, and the phantom third strike` is the
  incident that earned it.
* 2026-07-22, THE LAW (recorded in `OTR_STANDING_RULINGS.md`): **"AN AUDIT MAY IMPROVE
  A STORY. IT MAY NEVER FAIL ONE FOR LENGTH, LANGUAGE, STYLE, VISUAL VOCABULARY, OR
  QUALITY."** Structural failures (schema, IDs, roster, rights, graph) stay fail-closed
  because they protect a usable ledger. Everything that is a *judgment about prose* is
  telemetry only. The commits leading to it are blunt: `68057d4d rip(all lanes): no LLM
  opinion may end an episode; prove it instead`; `75173fc4 ... second live-smoke catch:
  3 judge hallucinations in one pass`; `aed66c7a feedback: word counts advisory-only,
  no bands/gates -- never cut words at the cost of narrative (operator directive)`.

The project had spent April building guardrails ("pre-flight guardrail sweep:
foolproof episode presets," `11081086`) and July ripping them out, because every gate
that could kill a render had eventually killed a good one.

**The credit ladder.** 2026-07-24, `ed8d5a6d ... add MODEL & CREDIT BUDGET section`. A
seven-rung table from a $0 local Qwen on the second machine, through Antigravity and
Codex on weekly credits, Sonnet for post-code QA, Opus for the actual work, a paid
cloud roundtable for genuine idea rounds only, and Fable for "exactly two uses." Every
window was to state its rung in its first reply. The operator had turned review
routing into a budget line.

**The voice changes.** Around 2026-07-25 the commit subjects stop reading like
changelogs and start reading like sentences a person would say: `the second encoder
proves what it wrote, and the roster gate looks for a clip WRITER instead of a call
spelling` (`27a4f97c`); `the trimmed tail of a conditioning WAV is silence, never the
next beat's speech` (`4cc76806`); `wan_i2v renders a whole beat from ONE UNET load
instead of one per segment` (`3e89d6b2`). From here to the end of the log the subject
line is written for the reader who will not open the diff.

July also built the GGUF writer registry and promoted Qwen3-8B after a live bake-off
(`f58ed6e6`), shipped a 100-style radio-grammar catalog, and on the 31st specified a
"four-arm clamped video bench" whose kibitz r1 "caught two errors of mine"
(`04ae4f0c`).

**What the operator was doing in July:** writing law. THE LAW, two strikes, the credit
ladder, "word counts advisory-only," "make it simple" (`003875cc Rating doc: cut to the
rule (operator: make it simple)`), and catching a reviewer's false invariant on a rip
plan. He also drew a line on what the project is: "Operator rescope: cut the matrices
and the quick-wins block" (`36da1f9f`).

## August 2026 -- "Live proof, the eyeball, and the world outside"

**1,166 commits -- the peak. 220 saved kibitz runs. 91 PBUGs. "live" or "proven" in 121
subjects. The first commits that mention a second GPU, a Mac, or a rented pod.**

August is the month the project stopped arguing about quality and started proving
correctness, then left the building.

**Two directives that closed doors on purpose.** 2026-08-03, `9c4a0e20 Stop the
adaptation lanes fighting their own sources: no rolled arc, no truncated chapter, no
cap, no clause forbidding Macbeth its violence`. The packs had been instructing the
model to avoid "blood, guns, knives and graphic violence" *while adapting Macbeth and
King Lear* -- a fidelity defect dressed as a safety win. The operator's words, now in
`CLAUDE.md`: *"no violence or swearing guardrails, they just cause problems"* and *"I've
given up chasing profanity."* Same day: `a4bc7917 The word count is a request, not a
gate`. Then 2026-08-04/05: `b91a2b4a Story quality is closed by directive`. The
operator: *"I am not chasing story quality anymore. It works. It works. I will publish
it as open source, and if someone else wants to do it, or in six months I wanna chase
it again when I've got better tools, I will."* The scripts were accepted as they were.
What remained open were *correctness* defects -- and August found them in bulk.

**A week of plain-English correctness bugs.** 2026-08-05 alone:

* `bd05f696 MALVOLIO was cast female, because the row gender was a roll and the play
  was never asked`
* `7e4a4c3c The ledger did not name the voice that spoke, and a tier of one was never a
  choice`
* `f5a5d174 Two uncastable characters could draw the same voice, because the shared
  draw never saw the used-set`
* `99349330 The guardrail rip took the clause and left its comma, so four live prompts
  end mid-list`
* `51cf2938 A viewer read stage direction the voice never spoke, because the caption
  burned the raw line`
* `104c3f78 Shakespeare does not know who Folger is, so the announcer stops thanking
  them on air`

Each one is a sentence with a *because*. This is the house style now.

**Shakespeare performs Shakespeare.** `d633bb91 Shakespeare performs real Folger
scenes, and the cast comes from the text that has it` and `c51bbe27 A play episode is
now a real passage of the play, chosen to fit, carried verbatim` (2026-08-03). The next
day found the adaptation lanes "reading the first eight percent of the book and calling
it faithful" (`fde181b3`) and every beat of a Shakespeare scene "quoting the same middle
passage" (`cdf8bffb`). Fidelity to a source became a correctness class of its own.

**The video lane build.** 2026-08-11 and 12: twenty-one video lanes built, smoked live
and closed in about two days -- Wan i2v and ti2v, HuMo 14B and 1.7B in two aspects,
FastWan for 8 GB, three LTX recipes, a mesh stage, three visualizers, four still lanes,
and MiniMax H3 as "THE FIRST NEW ENGINE." Alongside them, `docs/LANE_BUILD_LESSONS.md`
accumulated lessons L1 through L27, with titles that are arguments: *"A gate that reads
a DECLARATION cannot be proof, whatever it is named"*; *"A fix lands on the path you
TESTED, not the path that RUNS"*; *"A correct test can become a bug's bodyguard without
anyone touching it."* Commit `d0536e72 feat(video): lane 5 -- a live bug that three
tests were defending` is the whole of L10 in one line.

**Live legs catch what suites cannot.** The handoff headings from mid-August say it
repeatedly: *"a LIVE leg catching a regression no unit test could"* (08-15); *"THE
REGRESSION, and it was mine"*; *"four bugs found by measuring rather than by
shipping"*; *"the leg meant to prove D2 found a pass that could never have run."* On
2026-08-15 the one sanctioned headless entrypoint turned out to have been dead for a
day (`7218f11b`). On the 17th: `e1c84cf6 The GPU proof passes 4/4, and the operator
heard four bugs the code could not`.

**Three rules in a day.** 2026-08-17 added three sections to `CLAUDE.md`:

* `7f6a6eca match the review to the task -- a full arc is not the answer to
  everything`. The test is one question: is there a design choice with more than one
  defensible answer? Yes -> the four-round arc before code. No -> no arc; one QA pass on
  the finished diff.
* `b45c5577 a missing kibitz reviewer never blocks the arc -- substitute and keep
  going`. Earned that day when Codex was quota-held and the driver tried to wait for it;
  the item shipped with two Antigravity lanes, a Fable gate and a Sonnet pass instead,
  and the panel "caught three driver errors, two of them build-breakers."
* `08232661 otr/obs is the SUCCESS SIGNAL -- always publish, never tidy it`. Written
  because the assistant had moved seventeen harness runs out of the obs folder as
  "pollution." They were the operator's proof that the full path worked. His words: *"if
  I see it in obs then it's somewhat a success ... if I don't see it in obs and it took
  more than 5 minutes, it's a fail."*

The next day added a fourth: `5a02740b Never report a context percentage (operator
directive 2026-08-18)`. The estimates had been wrong "repeatedly and badly" and each
wrong number had wrapped up work early.

**The operator's ear.** 2026-08-19: `6ae86b85 Operator's deletes: four dead mechanisms
removed, all voted by ear or by hand`; `0fc6f9de Loudness answered: target -14 LUFS,
never the peak ceiling - and my A/B was wrong`. 2026-08-20: `d9de9fc5 Voice bank:
operator auditioned all 63 donors -- 21 retired, 1 gender flipped`. 2026-08-21:
`c5a589de Ideogram 4: NO. The filter is in the weights and it refuses Shakespeare.`
The human was now the final instrument for everything perceptual, and the log says so
in each case.

**Self-correction as a commit type.** August normalized a kind of commit the early
months rarely wrote: `f2eeb6fd Retract the missing-file claim: the models root moved,
nothing was ever missing`; `223fa8c9 docs: retract the sibling-control conclusion -- it
came from one sample`; `62d84230 ... correcting myself`; `ccd5a4f5 Correct the record:
the close was half right and the lesson was the wrong one`; `a959fdf5 Handoff
correction: 12.117 WAS promoted; my first bible answer was wrong`. Being wrong in the
log became cheaper than being wrong in the render.

**Going public.** 2026-08-22 is the day OTR became a product other people could
install. `93c2e54f Registry publish prep`, then `2.0.0-alpha.2` through `alpha.6` in a
single day, each one a lesson: `a914cd60 Exclude exec()-flagged spike/smoke scripts`;
`9925513f alpha.3: declare deps to the registry`; `587d2574 alpha.4: static dependency
list -- the registry does not evaluate dynamic resolution` (alpha.3 had shipped with
`dependencies: []` -- a pack that would install with none of its libraries);
`4060a942 alpha.6: point first-time users at the workflow, not just the nodes`. The
same day `388bfaaa CLAUDE.md: v2.0-alpha is now the GitHub default branch` -- `main` had
been serving a 3,900-commit-stale v1 to every fresh clone. The next day's registry
archaeology read Comfy-Org's own backend source and found `89f4a2c1 ... the automatic
backfill scheduler is paused with a leap-day-only cron -- this is why nothing self-
resolves`. Publishing forced a kind of honesty that a private repo never demands.

**Lean and mean.** 2026-08-23: the 3D family retired (906 adapter lines), four public
nodes retired (34 became 30), the `visual/` POC tree retired (679K), every bake-off
harness retired along with the `CLAUDE.md` carve-out that licensed it (`cb91a8a4`;
operator: *"I think I am done with all bakeoffs"*), and `92b9edea Operator ruling:
every video lane is independent -- order 7 (consolidation) is cancelled`. On the 28th a
dead-code campaign removed roughly 2,600 lines across four rounds and a "knob census"
corrected 35 tooltips. Two rules came out of it: `adeca89e Remove 408 lines of plainly-
dead code -- and keep the five unwired symbols that are findings, not debt` (an
unreferenced symbol may be an unwired fix, not dead code), and from the 28th's link-
slot lesson, the three-part recipe for removing a widget that is now in `CLAUDE.md`
section 0 (it corrupted all 63 workflow variants before it was learned).

**The second machine.** 2026-08-25: `6c91109b Add otr_4060_floor: a zero-download
profile for 8 GB cards`. 2026-08-29, 84 commits: a real RTX 4060 laptop was brought
up from a cold install, and every one of its commits carries a *blast radius* clause --
"(blast radius: 4060 evidence only, no code touched)," "(blast radius: my drill log
only)," "(blast radius: 4060 only, no shared code)." The day produced the first episode
published from the 4060 (`a30846dd ... 55 min end to end`), five unlogged defects the
5080 could never have found (a Kokoro repo id, an undeclared `pyloudnorm`, a hardcoded
VRAM cap), a root-caused `llama-cpp-python` regression (`81f376d7 ... 0.3.35 regression,
0.3.33 loads and generates (hash-verified ...)`), and three explicit retractions of the
day's own earlier claims (`8c6d887d RETRACT step 11`, `53935150 CORRECTION`, `210bb06e
CORRECTION: withdraw the 'shakespeare commits zero repairs' claim -- an n=4 artifact,
falsified at n=5 and n=6`). The operator's decision that day: `43a2a8ae two windows
split by area, both push` -- the 4060 owns portability, the 5080 owns shipping, and
the append-only logs get `merge=union` so two writers cannot lose each other's lines.
The rule that came with it: *"a fix for one machine must prove the other machine is
unchanged -- measured, not asserted."*

**The third machine.** 2026-08-30: `dfcb387e RunPod: the pack LOADS on a pod -- 1036 ->
1061 classes, 25 OTR_ nodes, no skips`, and `c0479213 ... the whole install is zero-
terminal, and our own registry is what breaks it`. The rented pod found that `ffmpeg`
was an undeclared hard requirement, that the template lacked `HF_HOME` "which costs 84
GB per pod," and that a BOM in a secret file silently broke it. `f8c4c2e1 Log the
lessons this pod session actually cost, including two self-inflicted`.

**What the operator was doing in August:** listening. Sixty-three voice auditions. Four
bugs heard that the code could not see. Deletions "voted by ear or by hand." He also
killed the context readout, closed the story-quality question by fiat, told the
assistant that tidying the obs folder was destroying his evidence, and gave the
project a second and a third machine to fail on. His instrument was now the finished
episode, and his judgment was *"overall better"* -- not a rubric.

## September 2026 -- "Portability, the contrarian, and Shakespeare's own words"

**567 commits in eleven days. The 4060, the Mac or the pod in 102 subjects; the registry
in 44. Busiest day: 2026-09-06, 95 commits.**

**The great docs deletion.** 2026-09-02, 80 commits. `45895801 docs deletion pass,
batch 1: 34 stale non-dated docs`; `25ed362f ... batch 4: 17 more stale docs (a second
look at the 237 survivors, operator 2026-09-02: 'do we really need 237 docs')`;
`abaa44fe GO_FORWARD: forward-only rewrite in the operator's order -- 1457 lines to
1007, one queue`. The same day: `00d4b72b One JSON for now: the 4060 floor template
leaves the gallery (operator 2026-09-02)` -- the 86 per-machine workflow variants of
August collapsed back to one canonical graph whose dropdowns the user sets. The
project shape was being decided by a non-coder asking whether each file earned its
place.

**The windows talk to each other.** 2026-09-03, `b3765cea CLAUDE.md: the windows talk
to each other, the operator is not the transport`. After a morning of hand-pasting
registry findings between two assistant sessions, the rule: sessions message each
other directly; a peer can hand over a task or correct a fact but cannot approve a
publish. The operator removed himself from the relay path.

**The registry collapse, and a real security fix.** 2026-09-04: `aa5171c8 GO_FORWARD
1A: the registry ban is a real RCE, reproduced against the real modules`; `14c6a6db
SECURITY: replay import trusted a ledger the manifest never verified`; and a rip
sequence that is the lean-and-mean philosophy at full strength -- `47bf95d6 Rip 37
symbols nothing in production calls, and the 220 tests holding them alive`, `e5a9fd0f
Rip round 2: 11 symbols the first rip orphaned`, `e413ac00 Rip 13 methods`. Two hundred
and twenty tests had been keeping dead code alive by calling it directly. On the 5th,
`99b0e02d A calibrated local replica of the registry scan -- iterate to zero for free`
and `bfc8151f Publish 2.0.0-alpha.23 -- the first version at oracle-zero`. The
scanner that had been flagging versions for weeks was reverse-engineered into a local
oracle and driven to zero findings before the next publish.

**The drill.** 2026-09-06 is the busiest day of the month and the strangest to read:
`Record 0302 continuing 4060 prompt generation`, `Record 0309 ongoing 4060 prompt
progress`, `Log 4060 cast lock and one-act outline progress`, `Log 4060 slot contracts
and dialogue composition milestone` -- dozens of commits that are a live log of one
episode rendering on the 8 GB machine, checkpointed to the repo so the other machine
could follow. It ends with `fbd28432 Record the first complete 4060 episode: Qwen one-
act publishes in 32:34` and two rulings the drill earned: `f5f6fcac Remove every VRAM
cap: fit on GPU or fail loudly, never crawl` and `9087708d Retire the GGUF writer
rows; repoint every profile to an auto-downloading twin`. The 4060 had been given
ownership of the workflow files the day before (`f727a5c4`).

**The Mac.** 2026-09-07, `87ea8491 First Mac hardware run: MPS answered, and 16 GB is
the wall`, on a rented M4. Two days of lessons followed -- NF4 quantization is dead on
Metal, `ps rss` lies, an out-of-memory reboots the machine rather than the render, a
Windows path literal was creating an 8.7 GB junk directory (`e389ddee`) -- and on the
9th `41648d9a lightning_mac_proof_3 PUBLISHED -- the last Mac gap is closed`. The
README's Apple Silicon line, which had said the Mac needed a paid image API, was
corrected the same week and says so in place.

**One CLI review, every time.** 2026-09-08, `e46bce63 Complete the NaN guard
(freq/wave), and add the one-CLI-review directive`. Two defects had shipped that day
that no check the author ran could catch: a helper inserted on the first `"\nclass "`
token landed between a decorator and its class, so the decorator bound to a function
and an engine vanished from every dropdown on every platform -- and `ast.parse` was
happy; and a NaN guard whose comment said "five floats" wrapped three. Both were
caught by "a reader with no memory of having written the code." The rule: one local
CLI reader on every coding change before it is pushed. Not a panel. One.

**No fixed round count -- but every round gets a contrarian.** 2026-09-11, `9da69c89`.
Measured that day across eight arcs: 41 of 42 panel claims held against the real files,
seven of eight driver anchors were wrong, and a second round was unnecessary on eight
of eight rows. So depth became the driver's call and the one non-negotiable became the
reader briefed to *refute*. The same day settled the order of everything remaining:
`dec69d7d Arcs first, Shakespeare its own phase, testing absolutely last`, and the bar,
in the operator's words in `GO_FORWARD_PLAN.md`: **"As long as it doesn't crash when
it's not supposed to."** Crash-class and durability-class defects are the work.
Aesthetic drift is closed.

**Shakespeare performs its own words.** The last commits in the log before this
document: `187baff0 Shakespeare performs its own words: the verbatim executor (plan row
3.6)` and `19a0412c Gender the three Athenian mechanicals the verbatim window can now
seat`. The April README promised a show written from headlines. The September pipeline
can also take a Folger text, select a passage that fits the episode, carry the author's
lines verbatim, cast them by the play's own genders, and publish. The last row of the
fidelity lane is *A Midsummer Night's Dream*'s mechanicals getting their voices.

**What the operator was doing in September:** running three machines and a rented
fourth, asking "do we really need 237 docs," deciding that one JSON ships, removing
himself as the message relay between his own assistants, setting the order of the
remaining work, and defining the finish line in one sentence.

---

# PART II -- THE LESSONS, DISTILLED

These are the arcs that run across the months. Each one names where it starts in the
log and where it ends, because the point is the *change*, not the destination.

## 1. "Done" moved five times

* **April:** done = the tests pass. Commit subjects end in "132/132 passed."
* **May:** done = a QA document is filed for round-robin review. Sprint closes say
  "runtime NOT PROVEN" and ship anyway.
* **June:** done = a roundtable converged. Then 06-10 showed every live run had been
  using template prompts for weeks with a green suite, and a converged plan was
  overturned by instrumentation four days later.
* **July:** done = a live smoke on the canonical workflow. "FIRST GREEN EPISODE
  (Einstein's Echo in obs, 570s)."
* **August onward:** done = the artifact is in `otr/obs/`, the ledger is complete, and
  the operator has seen or heard it. "A leg that does not reach obs did not pass,
  however green its logs are." And a corollary the log states more than once: *a test
  that calls a helper directly proves the helper, never the wiring.*

The lesson for a vibe coder: the thing you cannot read (the code) is not the thing you
are shipping. Define done in terms of the thing you *can* judge.

## 2. The review ecosystem: cost went to zero, rigor went up

The sequence is visible in the reviewer table: Gemini and ChatGPT pastes (April) ->
scripted round-robin cloud consults (May, 32 mentions) -> the paid four-round roundtable
with spend recorded per arc, $0.05 to $2.29 (June, 73 mentions) -> the local $0 kibitz
panel of Codex plus Antigravity the moment they could be driven headlessly (June 27,
then 89 mentions in July and 144 saved runs) -> Sonnet for post-code QA and Fable as a
scarce final gate (July-August) -> one independent CLI reader on every change plus a
contrarian at every round (September).

What stayed constant: the driver writes its own code-grounded judgment *first* and
remains the sole judge; every panel claim is checked against the real files before it
is folded in; the roster is stated honestly ("a campaign a reviewer short is described
as a campaign a reviewer short"). What changed: the project learned *which reviewer
catches which defect*. File-grounded readers (Codex, Cursor) catch code-shaped errors --
on 2026-09-11, seven of eight driver anchors were wrong even though 41 of 42 factual
claims held. Fable catches the defect that lives *between* documents -- a ruling
invalidated by a commit made the same day, a plan that says "parked" where the
operator said "rip." And the author, structurally, catches neither in its own work:
"the whole mechanism is a reader with no memory of having written the thing."

## 3. Build it, then rip it -- as a method

The log is full of things built carefully, tested, shipped, and deleted soon after:
the April video stack (six weeks), the Story Room (three days), four source banks after
a bake-off, the SFX bus, the 3D family, the VRAM tiers, the word-count gates, the
bake-off harnesses themselves, 86 workflow variants, 2,600 lines of dead code in one
campaign, 220 tests that existed only to keep dead symbols alive. The rip/retire/delete
row is high in *every* month.

Two rules made ripping safe. First, **ripping an LLM is allowed, a hole in the ledger
is not**: before a pass is removed, every field it wrote gets exactly one new owner,
and the removal is proven on a live leg. Second, **rip fully or wire back, never half**:
a symbol that turns out to be an unwired fix for an open bug is a finding, not debt
(`adeca89e`), and `dead_code_closure.py` cannot tell the two apart.

The vibe-coder lesson: you will not know which clever idea helps until it has run
overnight and you have listened. Build so that deleting is cheap, and then delete
without sentiment.

## 4. Gates -> no gates

April: "pre-flight guardrail sweep: foolproof episode presets," temperature caps,
phantom-act strippers, a "SFW profanity validator in compose_line" (May 27). July: "no
LLM opinion may end an episode; prove it instead," then THE LAW. August: no profanity
or violence filtering at all, no word-count gate, and the observation that the packs
had been forbidding Macbeth his knives. The same month's `STANDING_RULINGS` records
"AN ALL-REFUSED EPISODE STILL PUBLISHES" and "the model floor should REFUSE, not grind."

The mechanism behind the reversal is in the commits: `3 judge hallucinations in one
pass`; `a clean P5 critique may never fail the story`; `two more audits/ceilings that
killed a live 420 leg`. Every judgment gate that could kill a render eventually killed
a good one, and the operator was the one who saw the good one die. The rule that
survived is narrower and sharper: a guard is legitimate only against a *silent wrong
render*; a guard that kills a render is the defect.

## 5. The operator's instruments: eye and ear

The operator's role changed more than the code did. In April he read READMEs and
pasted output between models. By June he was doing look-QA rounds on rendered frames
("the c01 giant-mic catch"). By July he was writing law and catching a reviewer's false
invariant. By August he was the final instrument for everything perceptual: 63 voice
donors auditioned, 21 retired; "the operator heard four bugs the code could not";
deletions "voted by ear or by hand"; "Ideogram 4: NO" by looking at what it refused to
draw. By September he was defining the finish line.

The assistants learned to route to him accordingly. The rule in `STANDING_RULINGS`:
lead with the decision, plain language, effect-on-output not implementation ("this
changes what Lumina paints, not the fourth ternary arm"), an options table at a
genuine fork, one-line bottom line. And the reverse: never auto-flip a voice gender,
never chase flicker by rewording prompts, judge an episode as radio drama -- "overall
better" is the verdict, not a rubric. The measurements are the machine's; the taste is
his.

## 6. Rules get written the day they are paid for

`CLAUDE.md` was committed 39 times, and nearly every commit carries a date and an
incident. The git policy on 06-10 (six unpushed commits). The workflow-source-of-truth
rule on 06-13/14 (an unwired node). The PowerShell quoting rule on 07-02 ("recurring
session trap"). Two strikes on 07-14 ("the phantom third strike"). Three rules on 08-17
(a reviewer treated as a hard gate; seventeen runs moved out of obs). The context
percentage on 08-18. Two-windows-both-push and pull-first on 08-29. The windows talk
directly on 09-03. One CLI review on 09-08 (a decorator that bound to a function). The
contrarian on 09-11.

The file reads like an incident log because it is one. The vibe-coder lesson: you do
not need to anticipate the rules; you need to write each one down the same day, with
the incident attached, so the next session -- which has no memory -- inherits the scar
and not just the scab.

## 7. Self-correction became a commit type

Count the subjects that begin with or contain *retract*, *correction*, *correcting
myself*, *my first answer was wrong*, *my A/B was wrong*, *an n=4 artifact, falsified at
n=5*. They barely exist before May 19 (`c60dcecd retraction of "stuck at 0/20"
framing`) and are routine by August 29. The BUG-LOCAL-231 saga taught the form: state
the hypothesis, test it, write FALSIFIED in the log, move on. By August the assistants
were retracting their own handoff claims within the same day.

This matters for a non-coder operator in a specific way: he cannot verify a claim by
reading the diff, so the only protection against a confident wrong claim is a culture
in which the claim's author is expected to come back and say it was wrong. The log
shows that culture being built.

## 8. Commit messages as narrative

Average subject length: 73 -> 78 -> 134 -> 104 -> 82 -> 73. The June peak is the
essay era, when a subject carried root cause, evidence and suite counts because the
reader could not get them from the code. Then around 07-25 the form changed rather
than the length: the subject became one plain sentence with a *because*. "A comment
describing a fix is not a fix: still_flat failed for being itself." "The authors were
reading the first eight percent of the book and calling it faithful." "A clock hand is
not a person."

This document exists because that choice was made. A commit log written for a
non-coder reader is a history you can hand to anyone.

## 9. Bug discipline: from "I found it" to "it failed live"

The `BUG-LOCAL` counter (developer-found, 207 mentions in May) fades to zero. The
`PBUG-YYYYMMDD-NN` log (live-failure-only, opened 07-10) takes over: 114 in July, 90 in
August, 66 in the first eleven days of September, 282 unique ids today. The admission
rule is strict: a review observation, a static-audit finding, or an invented fixture
may *verify* a known production bug but never creates a new one. Promotion to the
shared Bug Bible -- a portable YAML of ComfyUI-node lessons with executable coverage --
requires the same live evidence, and a coverage index exists so that history is never
re-scraped (the one full scrape cost roughly four million tokens).

The effect: the bug system stopped measuring how much code the assistants read and
started measuring what the pipeline actually did to episodes.

## 10. Ownership before implementation

`PRODUCTION_SPRINT_LESSONS.md` lesson 1 is "Define ownership before implementation,"
and the log earns it over and over. Five representations must stay in lockstep
(INPUT_TYPES, the execute signature, `widgets_values`, the inputs descriptor array,
and every link's slot index). A ledger field has exactly one owner. A helper is wired
in the same change that builds it, or the row says why not -- because on 2026-09-11
*four* correct, tested, green helpers were found with zero production callers. A
bank "lives in ~10 wired surfaces" and is ripped by a Teardown protocol that names
them. The workflow JSON is the source of truth, and variants are generated from it,
never hand-edited.

For a vibe coder the translation is: every piece of the system must be able to answer
"who reads me, and who writes me?" If the answer is "nobody," it does not exist, no
matter how green its tests are.

## 11. Two machines, then three: the thing you cannot see about yourself

Until 2026-08-25 every commit mentions one machine. Then the 4060 arrived and found,
in one day, five defects the 5080 could never have found about itself -- undeclared
dependencies, a hardcoded cap, a model id that only resolved locally. The Mac found a
Windows path literal writing 8.7 GB of junk. The pod found an undeclared `ffmpeg`. The
rule that followed: "the 4060 exists to find what the 5080 cannot see about itself,"
per-machine choices live in profiles never in shared code, and a shared-code change
prints the other machine's number before and after. The "blast radius" clause in
every 08-29 subject is that rule made visible.

## 12. Publishing forces honesty

Nothing tightened the project like shipping to the registry. `alpha.3` shipped with
an empty dependency list because the registry does not evaluate dynamic metadata
(`587d2574`). The `.comfyignore` had to decide what ships. A `scripts/` folder of 135
PowerShell and subprocess files was being bundled (`a02d186f`). A hidden input that
carried a session bearer was flagged as credential access (`90067e04`). The scanner
was lexical and had to be replaced by a local oracle driven to zero. The registry's
own extraction pipeline had stalled in April and its backfill cron ran only on leap
day. Every one of these was invisible in a private repo and unavoidable in a public
one. The operator's "do we really need 237 docs" is the same force applied to the
tree.

## 13. Collaboration hygiene for a non-coder operator

A short list of the rules that are about *talking*, not code, each with its date:

* Never ask him to run a command; run it (section 2 of `CLAUDE.md`, standing).
* Stop only for a real blocking decision, framed as impact he can judge (08-17).
* Every stop ends with one line: what you need, or "nothing -- proceeding" (08-17).
* Be decisive on the obvious; blanket approvals are durable (08-17).
* Never report a context percentage (08-18).
* Report times in his local zone.
* End anything long with a four-sentence bottom line.
* The windows talk to each other; he is not the transport (09-03).

These read as etiquette. They are throughput. Each one removed a context switch from
the person whose attention was the scarcest resource in the project.

## 14. The bar, at the end

April's README promised everything. September's plan says: *"As long as it doesn't
crash when it's not supposed to."* *"I'm not expecting anything exact."* Story quality:
closed. Aesthetic drift: closed. Crash-class and durability-class defects: the work.
Arcs first, then coding, then Shakespeare as its own phase, then testing absolutely
last.

That is not lowered ambition. It is the shape of a project that found out, by running
roughly a thousand episodes through itself, which of its thousand ideas survive
contact with a listener -- and decided to ship those.

---

# APPENDIX

## A. Milestone timeline

| Date | Event | Commit / tag |
|---|---|---|
| 2026-04-05 | v1.0 "Canonical Audio Engine" -- first commit | `03d5e581`, tag v1.0 |
| 2026-04-06 | First external model QA round (Gemini) | `cdf8d3cf` |
| 2026-04-07..15 | v1.1 through v1.7 released | tags v1.1 .. v1.7 |
| 2026-04-12 | v2.0-alpha declared; visual sidecar design | `74afd856` |
| 2026-04-13 | Overnight soak harness with randomization | `1f8da979` |
| 2026-04-17 | Fourteen "Day N" video-stack commits in one day | `e163d517` .. `130db7a6` |
| 2026-04-24 | Production ledger L1 | `f7d271b5` |
| 2026-04-30 | First round-robin consult folded in | `f0421c2e` |
| 2026-05-01 | BUG_LOG zeroed into the Bug Bible | `08ecfcd1` |
| 2026-05-02 | `otr/obs/` and `otr/episodes/` split | `79ce42b5` |
| 2026-05-11 | "No legacy back-compat" standing directive | `3d00c96e` |
| 2026-05-13 | Busiest day ever (103 commits); cleanbreak sprints S26-S29 | `aad568c0` merge |
| 2026-05-18..19 | BUG-LOCAL-231: 134x graph regression bisected | `4811bfca` |
| 2026-05-27..29 | Story Room built, then ripped | `6a41f0cc` .. `6c0943a5` |
| 2026-05-30 | 217 docs pruned; first stable tag | `7896e5d3`, tag v2.0-alpha-stable |
| 2026-05-31 | OpenRouter remote writer; story-quality baseline | `a117efd7`, `36966271` |
| 2026-06-06..08 | Model-agnostic video platform; legacy batch path deleted | tags m0, m1, A-ship, B-ship |
| 2026-06-10 | Production restore; "every live run fell to template prompts"; GIT POLICY | `18e3cbdf`, `1351d783` |
| 2026-06-13..14 | Workflow-source-of-truth rule; Cowork operating manual | `5e4babd1`, `56469bdb` |
| 2026-06-21 | Roundtable defaults written into CLAUDE.md | `82fbd4ec` |
| 2026-06-23 | Story-quality v2 flipped ON by directive | `550679d7` |
| 2026-06-27 | Headless Codex and Antigravity -- the $0 panel; first "kibitz" | `fdbfa977`, `684a7064`, `5f0bab70` |
| 2026-06-29 | "Registry IS the menu"; no-fallbacks directive drafted | `c6ca5d88`, `c8f0156c` |
| 2026-07-02 | NO fallbacks / dropdown values are the only defaults | `d0463b8c` |
| 2026-07-03 | VRAM rip; Fable catches the chunk-order KeyError | `d9a9c085`, `4ef2ffe7` |
| 2026-07-04 | Ten fallback sites -> fail loud; JSON transplant PARKED after full arc | `e346eeb4`, `1a3571e1` |
| 2026-07-05 | Source banks + visual styles as JSON packs | `4f611cb3` .. `8da76394` |
| 2026-07-08 | Shakespeare source bank | `497e7b9e` |
| 2026-07-10 | First green fable2 episode; PROD_BUG_LOG opened | `ff4c226d`, `05088e32` |
| 2026-07-14 | TWO STRIKES directive | `bcc6a3de` |
| 2026-07-15..18 | Bank bake-offs; four banks ripped | `3312aec7`, `c507acff` |
| 2026-07-22 | THE LAW: an audit may never fail a story for quality | STANDING_RULINGS |
| 2026-07-24 | Model & credit budget ladder | `ed8d5a6d` |
| 2026-08-01 | FastWan renders live on the 8 GB lane | `859b1a1c` |
| 2026-08-03 | No content guardrails; Shakespeare performs real Folger scenes; word count is a request | `9c4a0e20`, `d633bb91`, `a4bc7917` |
| 2026-08-04..05 | Story quality CLOSED by directive | `8330805c`, `b91a2b4a` |
| 2026-08-11..12 | 21 video lanes built and closed; LANE_BUILD_LESSONS L1-L27 | `49adc824` .. `be4aadff` |
| 2026-08-14 | The word rip: acts replace words | HANDOFF_LOG |
| 2026-08-17 | Match review to task; substitute reviewers; obs is the success signal | `7f6a6eca`, `b45c5577`, `08232661` |
| 2026-08-18 | Never report a context percentage | `5a02740b` |
| 2026-08-20 | 63 voice donors auditioned by ear; 21 retired | `d9de9fc5` |
| 2026-08-22 | Registry publishing live (alpha.2 .. alpha.6); v2.0-alpha becomes default branch; Ghost Signal lane | `93c2e54f`, `388bfaaa`, `d0b3a65b` |
| 2026-08-23 | Lean-mean rips; every bake-off retired; video lanes declared independent | `cb91a8a4`, `92b9edea` |
| 2026-08-25 | First 8 GB (4060) profile | `6c91109b` |
| 2026-08-28 | Dead-code campaign (~2,600 lines); scripts/ stops shipping | `3f97b7de`, `a02d186f` |
| 2026-08-29 | Two windows both push; first episode from the 4060; blast-radius discipline | `43a2a8ae`, `a30846dd` |
| 2026-08-30 | Pack loads on a RunPod pod | `dfcb387e` |
| 2026-09-02 | Docs deletion ("do we really need 237 docs"); one JSON ships | `25ed362f`, `00d4b72b` |
| 2026-09-03 | The windows talk to each other | `b3765cea` |
| 2026-09-04 | Registry RCE reproduced; security fix; 37+11+13 symbols and 220 tests ripped | `aa5171c8`, `14c6a6db`, `47bf95d6` |
| 2026-09-05 | Registry scan oracle at zero; alpha.23; 4060 owns the workflow files | `99b0e02d`, `bfc8151f`, `f727a5c4` |
| 2026-09-06 | The 4060 drill; first complete 4060 episode (32:34); every VRAM cap removed | `fbd28432`, `f5f6fcac` |
| 2026-09-07 | First Mac hardware run | `87ea8491` |
| 2026-09-08 | One CLI review on every coding change | `e46bce63` |
| 2026-09-09 | Last Mac gap closed | `41648d9a` |
| 2026-09-11 | No fixed round count, every round gets a contrarian; arcs -> coding -> Shakespeare -> testing; the verbatim executor | `9da69c89`, `dec69d7d`, `187baff0` |

## B. Glossary of the project's own words

* **OTR / SIGNAL LOST** -- the node pack and the show. `OTR_` prefixes every public node.
* **The operator** -- Jeffrey. The log uses "operator" 12 times in April and 63 in June.
* **5080 / IDREAM** -- the RTX 5080 laptop (16 GB) the project was built on; owns the
  shipping surface. **4060 / MRKT** -- the RTX 4060 laptop (8 GB) that owns portability.
  **The pod** -- a rented RunPod GPU. **The Mac** -- a rented M4 mini, 16 GB unified.
* **Canonical** -- `workflows/otr_canonical.json`, the one workflow. **Variants** --
  generated per-machine copies (86 in August, 1 in September).
* **Ledger** -- the per-episode JSON every stage reads and writes. A "hole in the
  ledger" is a field nobody owns.
* **obs** -- `output/otr/obs/`, where one finished MP4 per episode lands. Seeing it
  there is the success signal.
* **Leg** -- one headless run of the canonical workflow. **Soak** -- many legs
  overnight. **Smoke** -- a short leg (30 words) to prove the path.
* **Bank / source bank** -- where a story comes from: science news, Shakespeare,
  public domain, media archive, original. **Lane** -- a runnable engine row (video,
  image, audio). **Engine** -- one model adapter behind a lane.
* **Lemmy** -- a recurring British cameo character since 2026-04-06; still a workstream
  in August.
* **Round-robin** -- the April/May scripted multi-model consult. **Roundtable** -- the
  June paid four-round cloud panel. **Kibitz** -- the $0 local panel (Codex +
  Antigravity, later Cursor) in the same four-round shape. **r1..r4** -- arc, coding
  plan, wiring, convergence. **Contrarian** -- the reader briefed to refute (09-11).
  **The driver / anchor** -- the Claude window doing the work and judging the panel.
* **Fable / Opus / Sonnet** -- Claude model tiers used as subagents; Fable is the
  scarce "scalpel."
* **BUG-LOCAL-nnn** -- the developer-found bug counter (retired). **PBUG-YYYYMMDD-NN**
  -- a bug that failed in a live run (the only kind admitted since 07-10). **The
  Bible** -- `BUG_BIBLE.yaml` in the sibling survival-guide repo, portable rules with
  executable coverage.
* **Rip / cleanbreak / lean-mean** -- delete fully, with a grep receipt and no shim.
* **THE LAW** -- 2026-07-22: an audit may improve a story, never fail one for quality.
* **Two strikes** -- two solo fixes, then a panel before the third.
* **Blast radius** -- the clause in a commit naming which machines' behavior changes.
* **Live-proven** -- the claim was tested by a leg that reached obs, not by a suite.

## C. Method

Everything above was derived from `git log --reverse --format='%h|%ad|%s'` on
`v2.0-alpha` at HEAD `19a0412c` (2026-09-11), read in full for April and May and
through a keyword-filtered view (roughly 1,050 of 3,680 subjects) plus first-commit-
of-each-day for June through September; from `git log -- CLAUDE.md` (39 commits); from
the headings of `docs/HANDOFF_LOG.md`, `docs/OTR_STANDING_RULINGS.md`,
`docs/PRODUCTION_SPRINT_LESSONS.md`, `docs/LANE_BUILD_LESSONS.md` and
`docs/MAC_LESSONS_LEARNED.md`; from the operator-voice sections of the standing
rulings; from `git ls-tree` counts at each month boundary; and from the README at
`03d5e581` versus HEAD. Keyword tallies are case-insensitive matches on commit
subjects and undercount anything phrased differently. Where a quotation is attributed
to the operator it is taken from a commit subject or a dated rules file that records
it as his.
