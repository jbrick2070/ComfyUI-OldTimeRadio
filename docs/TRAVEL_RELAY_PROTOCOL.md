# TRAVEL RELAY PROTOCOL -- Claude as senior dev, Codex as coder

**Written 2026-07-27, for a week away from the desk.** The operator cannot open
new remote Cowork windows while travelling. One Claude window has to survive the
whole trip, and its context is the binding constraint -- not the GPU, not Codex,
not the repo. Everything here is built around that.

## The three roles

- **Claude (senior dev / judge).** Scopes the work, decomposes it, writes the
  TICKET, defines what proof of correctness looks like, reads the REPORT, rules
  ACCEPT / FIX / STOP. Holds the baton. **Reads almost nothing.**
- **Codex (implementer).** The only party that reads the repo, writes code and
  tests, runs the suite and the Bug Bible, commits, pushes. Operating rules live
  in `docs/CODEX_BRIEF_TRAVEL.md`.
- **Operator (wire).** Copies text between the two. Rules only on questions
  marked OPERATOR FORK. Does not need to review code from a phone.

**Design rule: the senior dev must work with zero tools.** No repo access, no
bridge, no container. Text in, text out. That way the protocol survives a dropped
device bridge, a reclaimed container, a dead laptop, or a phone-only day.

## The loop

1. Claude emits **TICKET n** (<=30 lines).
2. Operator pastes it into Codex.
3. Codex works, proves, commits, pushes, emits **REPORT n** (<=35 lines).
4. Operator pastes the report -- **and only the report** -- back into Claude.
5. Claude rules:
   - **ACCEPT** -> emits the updated BATON, then TICKET n+1.
   - **FIX n.1** -> a delta ticket, <=15 lines, naming only what changed.
   - **STOP** -> an OPERATOR FORK, stated as a question with a recommended
     default, so it can be answered from a phone in one word.

## Context budget -- the rules that keep the window alive

These are on Claude, and they are strict:

- **Never ask for a file dump, a log, a diff, or `git status`.** Bare
  `git status` in this repo prints 1000+ untracked lines. One paste of it costs
  more than a whole ticket cycle.
- Claude's own turns: TICKET <=30 lines, VERDICT <=10 lines, BATON <=20 lines.
  No essays, no restating the report back, no summaries of what just happened.
- If a REPORT arrives over-long or in the wrong shape, do not read past it --
  reply "re-emit REPORT n in the format" and spend nothing else on it.
- Claude may use the device bridge only when a report is genuinely ambiguous,
  and only for one-line facts: `git rev-parse HEAD`, `git log --oneline -3`,
  `Test-Path <path>`. Never a bare `git status`, never a full file read. When in
  doubt, ask Codex for the fact instead -- its context is free, ours is not.
- To see code: name the file **and a line range**, cap 60 lines, and have Codex
  paste it. Never "show me the file".
- One ticket in flight at a time. Parallel tickets double the paste volume and
  make the baton ambiguous.

## The BATON -- the reason one window failing is survivable

After every ACCEPT, Claude re-emits a <=20-line state block. The operator keeps
the **latest one only**, in a phone note. It carries: HEAD sha, suite numbers,
Bible count, canonical hash, the lane, what just landed, what is next, and any
open operator fork.

If the window dies, gets slow, or hits its context wall: open a spare window,
paste the KICKOFF, then paste the latest BATON. That is a full reboot in two
pastes with no repo access required.

**Open 2-3 spare Cowork windows before leaving and paste the KICKOFF into each,
then leave them idle.** They are the insurance against the one thing that cannot
be fixed from the road.

## TICKET format (Claude -> Codex)

```
TICKET <id>
GOAL: one sentence -- the behaviour that is wrong and what right looks like
BASE: <sha7> on v2.0-alpha
FILES: the likely set (informative, not binding -- Codex may find better)
DO: 3-7 numbered steps
PROOF: the exact commands, and what green looks like (expected suite/Bible)
DO NOT: the specific traps for this change
STOP-IF: conditions that mean report back instead of pressing on
REPORT: per docs/CODEX_BRIEF_TRAVEL.md section 4
```

A ticket that cannot state its PROOF is not ready to send.

## Week queue (as of 2026-07-27)

The CODER A 8 GB block (B1a through B6) is **complete**. Two lanes are open:

**LANE 1 -- WAN recipe freeze. No GPU. This is the default lane.**
Scouted during session 5c, nothing touched. Both WAN adapters carry the whole
pre-B6 defect: `eng_wan_ti2v` reads loader class, tiled-VAE class, all three
weight names, sampler, scheduler, steps, cfg, shift, negative and four VAE-tile
vars straight from the environment; `eng_wan_i2v` reads six INLINE in
`_build_graph` with bare `int()` / `float()` -- no range check, no named
refusal. Neither emits a recipe receipt at all, so a WAN clip stamps
`recipe: None` and there is not even a wrong receipt to catch the drift with.
**B6 is the shipped reference implementation to mirror** -- see
`docs/2026-07-27-b6-qa-findings.md` and `LTX8_RECIPE_V1`. Suite-provable end to
end, no GPU, no operator judgment. Ideal remote work.

**LANE 2 -- prequalify 512x288. GPU, one measurement sweep.**
Boot with `OTR_LTX_8GB_PREQUALIFICATION=1`; measure T5 device on/off and tiled
decode on/off at 512x288. Codex runs the sweep and reports the four numbers;
Claude picks the winner; a follow-up ticket freezes it as recipe v2 by bumping
the version inside the `RECIPE_LTX8_I2V` string (that moves the session identity
for free). Only run this when no other window holds the 5080.

**PARKED until the operator is back at the desk:** 7d, the canonical 237-frame
opening beat. It is the next real milestone and it wants his eyes on it.

## Standing cautions for the week

- Another Claude window may be live on this repo. Lane 1 must be claimed before
  Codex starts it, or the two will collide in the WAN adapters.
- No code lands without suite + Bible green in the same report that claims it.
- A render is not successful until `Test-Path` says the asset exists.
- Any question that needs a production judgment goes back as an OPERATOR FORK
  with a recommended default, phrased so it can be answered in one word.
