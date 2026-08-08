# Next-window kickoff (paste into a fresh CODER window)

Written 2026-08-08, after cloud-audio-cache chunk 2 shipped at `ebe24bd4`. HEAD is now `ebe24bd4` == `origin/v2.0-alpha`; suite 9222/111/1; Bug Bible 17.

---

resume the OTR build -- you are a CODER window taking queue item 8
(system-agnostic multi-GPU upscale). State your MODEL & CREDIT BUDGET
rung first.

WHY this pick and not item 9 chunk 3: item 9 chunk 2 (content-addressed
audio cache) SHIPPED 2026-08-08 as `ebe24bd4`. Chunk 3 is the Macbeth
safety probe per arm, which needs a LIVE cloud leg (Gemini TTS +
Comfy/Veo video), which needs the API keys to reach the run. That is
the same blocker gating item 1 chunk B (OPENROUTER_API_KEY not reaching
the run). If the operator has set the launcher envs, take chunk 3
instead. Otherwise the ordered runway from GO_FORWARD.md and
ROADMAP.md puts item 8 next -- it is a DESIGN item first, then a coder
chunk, so it needs a full arc and a problem-statement doc before code.

HEAD is ebe24bd4 == origin/v2.0-alpha. Repo:
  C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

READ FIRST, IN ORDER:
1. docs/GO_FORWARD_PLAN.md -- queue at top, tombstones (item 9 chunk 2
   shipped yesterday), MODEL & CREDIT BUDGET ladder.
2. ROADMAP.md line 193-196 -- the exact operator constraint on the
   upscale item: "built against the profile and registry contracts,
   NEVER a resurrection of the retired NVIDIA-only node".
3. docs/HANDOFF_LOG.md top entry (2026-08-08 cloud-audio-cache) -- the
   chunk that just shipped, its follow-up chip owed (wrap loop in
   try/finally for stamp persistence), and the four QA gates that ran.
4. CLAUDE.md in the ComfyUI root AND in the repo root -- standing
   operator rules outrank everything.

STANDING DIRECTIVES YOU MUST OBEY THIS SESSION (verbatim from the
operator, do not re-litigate):
- FULL KIBITZ ARC on this coding work (08-04, still standing until
  the operator withdraws it): invoke `kibitz-plugin:kibitz` (the
  plugin skill by name; the anthropic-skills:kibitz duplicate is NOT
  what the operator asked for). Full = the default r1 -> r2 -> r3 ->
  r4 arc, 8 external calls. Panel is DRIVER-AWARE: you drive from
  Cowork, so Codex + Antigravity review; you do NOT launch a second
  claude -p lane against your own family. Use the ComfyUI profile
  overlay at .kibitz/comfyui.local.md.
- TWO STRIKES THEN THE PANEL (07-14 floor, never lapses): third
  attempt on the same bug ALWAYS gets a panel. The 08-04 rule above
  is stricter and requires a panel on the first swing for coding
  work, but two-strikes remains the backstop.
- SONNET 5 for the post-coding QA pass (08-05 standing rule) --
  independent agent reading the final diff. Kibitz still runs on top.
- FABLE on r1 rounds first (08-06 standing rule) -- Fable gets the
  first opinion on r1, cold, BEFORE the driver anchor frames it.
- FINAL REVIEWS AFTER R4 (08-06): r4 is NOT the last gate. Sonnet 5
  QA on the diff -> Fable final gate -> THEN suite/push.
- NO HANDOFF WHILE BACKGROUND TASKS ARE RUNNING (08-06). Finish or
  explicitly report them first.
- STORY QUALITY IS DONE (08-04). Do not open writing-quality work.
  Correctness bugs (gender/voice/face contradictions, ledger holes)
  are still fair game.
- NO GUARDRAILS ON GENERATED EPISODE CONTENT (08-03). Do not add
  profanity or violence filters to the generation path. Authoring
  style stays clean -- no curse words in code/comments/logs, and
  never the name "dummy".
- NEVER CHASE WORD COUNT (08-03). target_words is a REQUEST, never
  a gate.
- RIP AN LLM IS ALLOWED, A HOLE IN THE LEDGER IS NOT (07-14): if
  you remove or repurpose a pass, enumerate every ledger field it
  wrote, give each a new owner (deterministic Python / another
  pass / explicit default), THEN delete the call, and PROVE on a
  live leg. A green unit suite does NOT prove the ledger is complete.
- LOCK FILES, PRESERVE OPERATOR WORK: never `git add .` or
  `git add -A`. Add by pathspec. A blanket add once swept three
  staged deletions to origin.

PRESERVE ALL OTHER WINDOWS' DIRTY PATHS: config/profiles/otr_g4_wan_ti2v.json,
config/profiles/otr_sbcov_*.json, tmp/*.ps1, kibitz/,
config/source_banks/_corpus/, uv.lock. Do not touch them.

BLOCKED, SKIP:
- Item 1 chunk B (OPENROUTER_API_KEY not reaching the run).
- Item 3 reference A/B verdict (operator eyeball).
- Item 4 WAN 8-GB launch contract (operator call + proof leg).
- Item 5 MiniMax H3 dropdown ruling (operator ruling).
- Item 6 video matrix pattern (operator/planner writes content).
- Item 7 the 23 shipped bad-open episodes (operator decides).
- Item 9 chunk 3 (Macbeth probe) unless the operator has set the
  cloud API keys in the launcher.
- The follow-up chip owed for cloud-audio-cache SF#1 (wrap voice-node
  loop in try/finally); file that as its own tiny arc AFTER item 8
  is scoped and staged, not inside this session.

BOX STATE AT HANDOFF: clean. No resident ComfyUI server, port 8000
free, VRAM at desktop baseline. Suite baseline to carry forward:
9222 passed / 111 skipped / 1 xfailed. Bug Bible 17 passed / 24
skipped / 3 xfailed at survival-guide 3759ae5.

FIRST CONCRETE ACTIONS:
1. Read the 4 files above.
2. Read the retired NVIDIA-only upscale node's history: `git log
   --all --grep upscale --oneline | head -20` and the surrounding
   commits, so you know WHY it was retired and WHAT contract the
   new node must NOT violate. Also check `nodes/RETIRED_ENGINE_IDS`
   or equivalent for the retired-id list.
3. Ground the two contract surfaces: `nodes/_otr_engine_profiles.py`
   (the profile Pydantic class) and `nodes/_otr_video_engines/`
   registry (or the equivalent image/upscale registry) to see how a
   NEW system-agnostic upscale engine would fit in.
4. Write a problem statement `docs/2026-08-08-PROBLEM-STATEMENT-multi-gpu-upscale.md`
   before any code -- what "system-agnostic" means concretely (must
   run on the 5080; must not blow up 5080 VRAM budget when the target
   is the 4060; must be selectable via a profile row; must not
   resurrect the retired node's coupling to NVIDIA drivers), and
   what SUCCESS looks like (a still or a small clip upscaled with
   the new engine, running headless, `obs_publish OK`).
5. Author the r1 driver_anchor.md against those real files, THEN
   kick off Fable-cold r1 (per the 08-06 rule), THEN the Codex +
   Antigravity r1 panel.
6. Do NOT touch workflows/otr_canonical.json unless the plan
   requires a widget/link edit; if it does, the edit ships in the
   same change as the code, and re-validation is on you.

TWO STRIKES THEN /kibitz. THE LAW HOLDS.
