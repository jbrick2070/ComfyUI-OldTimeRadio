# Driver anchor -- THE REGISTRY PLAN FROM HERE (2026-09-05)

Written by the driver (Claude, Fable 5.1) BEFORE any panel opinion, from live API
reads and the real files. Every claim below carries its source. Panelists: read the
REAL files at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
(HEAD `02ca6d5a`, branch `v2.0-alpha`); the published artifact under review is
extracted at
`C:\Users\jeffr\AppData\Local\Temp\claude\C--Users-jeffr-Documents-ComfyUI-custom-nodes-ComfyUI-OldTimeRadio\80f0933f-108e-4a99-9b24-8548e3a3b547\scratchpad\a19\tree\`
(alpha.19, built from commit `531ddda4`). Cite `file:line` for everything.

## 1. The question

The pack `comfyui-old-time-radio` (publisher `fluxus`) has SEVEN registry versions
and none is Active. **What is the best plan from here to get an Active version, and
what exactly is on the punch list a manual reviewer will still hit?** The operator
alone posts, bumps `pyproject.toml`, or publishes. The panel decides WHAT and in
WHICH ORDER; it does not act.

## 2. Live registry state (read 2026-09-05, `GET /nodes/comfyui-old-time-radio/versions?include_status_reason=true`)

| version | status | scan | note |
|---|---|---|---|
| alpha.19 | **Flagged** | 12 `info`, 0 critical | published 09-05 01:43Z from `531ddda4`; record BYTE-IDENTICAL to alpha.18's |
| alpha.18 | Flagged | 12 info | 09-04 |
| alpha.17 | Flagged | 158 info | 09-03 |
| alpha.16 | Flagged | 157 | 09-03 |
| alpha.15 | Flagged | 158 | 09-02 |
| alpha.14 | **Banned** | human | `policy-v0.2: RCE (code execution) -- attacker-reachable via unauthenticated /prompt (node widget) or no-auth route; confirmed by code-level verify-deep.` by `drltdata@comfy.org` |
| alpha.13 | **Banned** | same | same |

The 12 remaining findings (all `info`, YARA): 4 `python_environment_manipulation`
(`prestartup_script.py:60`, `nodes/_otr_writer_heartbeat.py:61`,
`nodes/_otr_audio_engines/eng_indextts2.py:176`, `nodes/_otr_shared/env.py:77`),
5 `python_network_operations` (`nodes/_otr_comfy_backend.py:384`,
`nodes/_otr_feed_fetch.py:249`, `nodes/_otr_openrouter_backend.py:1011`,
`nodes/_otr_google_api/client.py:191`, `nodes/_otr_shared/cloud_media_invoke.py:578`),
3 `python_command_injection_risk` (`eng_indextts2.py:214`, `nodes/_otr_shared/proc.py:161`,
`proc.py:168`).

## 3. HOW THE REVIEWER ACTUALLY WORKS (new today; this changes the plan)

Read from the `status_reason` of ~30 other packs' versions (files `pack_*.json`,
`issuepack_*.json` in the scratchpad; verbatim verdicts in `policy_v02_examples.md`):

* **The "manual review" is an automated deep pass**, signed `drltdata@comfy.org`,
  worded `policy-v0.2: <CLASS> -- ... confirmed by code-level verify-deep`. It runs
  on Flagged versions and emits either a BAN with a class, or
  `reviewed SAFE (GOAL2 verify-deep, policy-v0.2)` -> **Active**.
* **It is fast.** Examples: `comfyui-af-find-nodes` 0.3.3/0.3.4 published 09-01 ->
  reviewed SAFE -> Active the SAME DAY. `feihou-toolbox` 2.9.2/.3/.4 published 09-01
  -> BANNED 09-01 (three different classes). `comfy-import-guard` 1.0.0 SAFE 08-12,
  1.1.0 BANNED 08-29. `ausboss-nodes` four versions banned 08-17..08-31.
* **Our alpha.15-.19 (09-02..09-05) have NOT yet been through it** (still the raw
  YARA list, no human record). alpha.13/.14 were.
* **The GitHub issue channel is write-only.** Of ~20 "Manual review request" issues
  on `Comfy-Org/registry-backend` since 08-02 (#189..#222), **zero have a maintainer
  reply**; the only non-author comment is another publisher. #202 was closed
  `not_planned` in one day. Packs that filed issues were then processed by
  verify-deep exactly like packs that did not (banned or approved on the code, not
  the ask). Filing is therefore NOT the mechanism; the code is.
* **The policy has numbered rules, quoted in verdicts.** Rule 1b clause 1: "node-widget
  taint keeps full severity". Rule 11: a free STRING widget that reaches a
  filesystem write with "no realpath+commonpath containment, no '..' rejection" =
  PATH_TRAVERSAL, INSECURE. Rule 12: a free STRING widget that reaches a network
  call "with NO host allowlist" = SSRF (banned even with a `127.0.0.1` default --
  `ausboss-nodes` LM Studio `endpoint`); if the body carries host data = DATA_EXFIL.
  Section 0.8: a default value does not matter, the widget is attacker-overridable
  via unauthenticated `/prompt`. Also seen: UNAUTHENTICATED_SIDE_EFFECT (POST
  routes, CSRF-reachable), unsafe deserialization, ARBITRARY_FILE_LAUNCH
  (os.startfile even after an output/ containment check), arbitrary file read (LFI),
  egress/SSRF.
* **Ban-class tally across the sample:** PATH_TRAVERSAL 10, UNAUTHENTICATED_SIDE_EFFECT
  6, SSRF/exfil 5, RCE 1 (ours), command injection 1, deserialization 1, LFI 1, launch 1.
  **Write-side path traversal is the #1 ban reason by a wide margin.**
* What passes: `comfyui-video-xy-plot` Active with `"subprocess: ffprobe"` noted by a
  human; `comfyui-llmlink` reviewed SAFE despite `os.environ.get("LLMLINK_API_KEY")`;
  `comfyui-frame-interpolation` "scanner/attack-surface flag assessed as false
  positive". **`info` findings do not block approval; a code-verified attacker path
  does.**

## 4. What is CLOSED at HEAD (with commits)

* The widget->argv[0] RCE: `widget_ffmpeg_is_ignored` at the execute method of 12
  node files; resolvers return absolute-or-nothing (`a9e0383e`, `7fc77501`).
* IS_CHANGED ran before the remote-path guard -> `79dc9828`.
* The two POST render routes are behind `OTR_ENABLE_HTTP_RENDER_ROUTES=1`,
  default off (`__init__.py:594`; landed `b198026a`, 09-03).
* Wildcard CORS removed from `GET /otr/latest_ledger` (`__init__.py:449-463`).
* Forged image-cache entries -> arbitrary local file read via `/view`: `9d3f56a7`.
* Pending sweep rmtree of unreadable dirs: `31dc6861`. Replay manifest membership:
  `14c6a6db`.
* No `pickle`/`marshal`/`yaml.load`/`os.startfile`/`webbrowser`/`shell=True`/`eval(`/
  `exec(` in shipped code; the one `torch.load` is `weights_only=True`
  (`nodes/_otr_audio_engines/_kokoro_backends.py:193`); `proc.py:144` refuses
  `shell=True` outright.
* No STRING widget anywhere in `nodes/` whose name contains url/feed/endpoint/host.
  Feed URLs are constants (`story_orchestrator.py:420-442`) or the env
  `OTR_MEDIA_ARCHIVE_FEEDS` (`_otr_media_archive_sources.py:187-195`); fetches go
  through `_otr_feed_fetch` which is https-only (`:374`), resolves to PUBLIC
  addresses only (`:206`), and bounds the read (`:265`). OpenRouter / Google base
  URLs come from env or a constant (`_otr_openrouter_backend.py:149,1038`;
  `_otr_google_api/client.py:16,179`). `cloud_media_invoke.py:611-613` follows a URL
  returned by the provider's own response.

## 5. THE PUBLISHED alpha.19 IS NOT HEAD -- two security fixes post-date it

`git log 531ddda4..HEAD`: `79dc9828` (IS_CHANGED bypass), `9d3f56a7` (arbitrary file
read), `31dc6861`, `14c6a6db` all came AFTER the publish. **The artifact the
reviewer will verify-deep still contains an arbitrary local file read** -- the exact
class (`arbitrary file read (LFI)`) that banned `feihou-toolbox` 2.9.4 on 09-01. A
Sonnet byte-level diff of the CDN zip against `git archive 531ddda4` and `HEAD` is
running; treat its output as the authority on what the artifact lacks.

## 6. THE OPEN PUNCH LIST -- what verify-deep will find in alpha.19 (driver's read; the panel must confirm or refute each)

**P1 -- Rule 11, write-side PATH_TRAVERSAL (the #1 ban class). OPEN.**
Free STRING widgets `output_path` on `OTR_CaptionBurn` (`otr_caption_burn.py:350`,
used `:441`), `OTR_MasterAudioMux` (`otr_master_audio_mux.py:1064`, used `:1325`),
`OTR_SilentComposite` (`otr_silent_composite.py:1536`, used `:1753`). Each is guarded
ONLY by `reject_remote_paths` (UNC/remote refusal) and then `out = output_path.strip()
or self._default_out(...)` -> ffmpeg writes to ANY local absolute path the `/prompt`
caller names. No realpath+commonpath containment, no `..` rejection. This is
word-for-word the SHARPPredict / OCIOWrite verdict. `_otr_paths.py` has no
containment helper (`comfy_output_dir():122`, `otr_episodes_root():343` exist;
nothing confines an arbitrary value to them). Only three files in `nodes/` use
`commonpath`/`is_relative_to` today (`otr_image_gen_dispatcher.py`,
`production_ledger.py`, `_otr_video_engines/render_driver.py`).
The canonical workflow stores NO absolute path in any widget (grep of
`workflows/otr_canonical.json`), and every `_default_out` lands under
`otr/episodes/<ep>/` inside the ComfyUI output tree -- so confinement to the output
tree costs the shipped workflow nothing. The operator's backup roots
(`E:\OTR-BACKUP`, `U:\OTR-BACKUP`) are not written by any node.

**P2 -- read-side / LFI. OPEN, needs a ruling on severity.**
Free STRING path widgets on the READ side: `video_path`, `ledger_path`
(`otr_caption_burn.py:317,346`), `silent_video_path`, `master_audio_path`
(`otr_master_audio_mux.py:1028,1032`), `base_video_path` (`otr_silent_composite.py:1503`),
`workflow_json_path` (`_otr_workflow_validator.py:233`, guarded `:76` remote-only),
`replay_from` (writer). An attacker names any local file; its bytes flow into an
output that ComfyUI serves via `/view` (a WAV muxed into an mp4; a JSON parsed and
echoed). Is that "arbitrary file read" under policy-v0.2 when the read is of a
media/JSON file through ffmpeg? The panel must argue it both ways with the
verdict texts in `policy_v02_examples.md`.

**P3 -- the unauthenticated `GET /otr/latest_ledger`.** Registered on every install
(`__init__.py:465`), no params, returns the freshest ledger plus `fullpath`. No
side effect, no attacker-chosen path. Info-class exposure; is it a ban class? Argue.

**P4 -- ffmpeg filtergraph text.** Ledger content (titles, captions, credits) reaches
`drawtext`/`ass` filter strings. The `ass=` filename escape was fixed; what about
`drawtext=text=`? Filter-graph injection is not RCE, but `movie=`/`amovie=` inside an
injected graph reads arbitrary files. Trace one path and rule.

**P5 -- `os.environ` writes** (`_otr_shared/env.py:77 pin()`, `prestartup_script.py:60`).
Is any `pin()` value derivable from a widget? If yes it is a persistent
environment mutation from `/prompt`; if all callers pass constants/config it is
noise.

**P6 -- the 12 scanner sites themselves.** For each: taint source, sink, verdict.

## 7. The plan candidates the panel must rank (driver's framing, not a decision)

A. **Fix P1 (and whatever of P2-P5 survives), publish alpha.20, and let verify-deep
   run** -- the channel that actually decides is automatic and fast; a GitHub post
   is at best a courtesy note. Risk: another Flagged-then-Banned cycle if a class
   is missed; each ban burns a version string.
B. **Post the SHORT re-review request now for alpha.19** (`docs/2026-09-04-registry-review-request-SHORT.md`),
   fix in parallel. Risk: it invites verify-deep onto an artifact the driver
   believes still carries an LFI and a Rule-11 write; a THIRD ban on a version we
   asked them to look at.
C. **Fix, publish alpha.20, THEN post a short note that names the two banned
   surfaces, the two new classes closed, and asks nothing but "alpha.20 is the
   version to look at."**
D. Something the panel sees that the driver does not.

Constraints that bind every candidate: no content guardrails on generated episodes;
ffmpeg via subprocess IS the render path (zero findings is not a goal); local /
offline-first; the three byte-hashed files (`eng_indextts2.py`,
`_otr_indextts2_worker.py`, `_otr_resolved_request.py`) must not change a byte;
`pyproject.toml` edits auto-publish and are the operator's; env-driven configuration
is operator-trusted under the policy (verdicts distinguish widget taint from
`os.getenv`), so an env allowlist of extra output roots is a legitimate design.

## 8. What "done" looks like for this panel

1. Each P1-P6 item: CONFIRMED (a ban class, with the verdict text it matches),
   REFUTED (why the reviewer will not reach it), or UNSURE (what byte-level fact
   would settle it -- Sonnet will diff it).
2. A ranked plan (A/B/C/D) with the ORDER of operations and what the operator does
   vs the coder.
3. For each confirmed item: the fix SHAPE (where the guard goes -- at the execute
   method, never in shared resolvers or the spawn gateway; see
   `docs/OTR_STANDING_RULINGS.md`), and what it must NOT break (the shipped
   workflow; the 4060 box; the operator's own paths).
