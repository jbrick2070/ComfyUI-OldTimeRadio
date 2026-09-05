# How do we get an APPROVAL on the Comfy Registry? Attack our assumptions.

Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
READ-ONLY. Change nothing. Cite `file:line`; quote what you actually read.
A grep hit is a lead, not a finding. We want CREATIVE, GROUNDED options.

## THE SITUATION, measured today (2026-09-05)

`comfyui-old-time-radio` (publisher `fluxus`). Versions `.13` and `.14` were
BANNED by a human reviewer:

> policy-v0.2: RCE (code execution) -- attacker-reachable via unauthenticated
> /prompt (node widget) or no-auth route; confirmed by code-level verify-deep.

Both banned surfaces are closed. `2.0.0-alpha.20` is published, byte-verified
against its commit, and sits **Flagged** with **12 findings, ALL `severity=info`,
ALL from `scanner: yara_scan`, and ZERO carrying a remediation recommendation**:

| type | count | where |
|---|---|---|
| `python_environment_manipulation` | 4 | `prestartup_script.py:60`, `nodes/_otr_writer_heartbeat.py:61`, `nodes/_otr_audio_engines/eng_indextts2.py:176`, `nodes/_otr_shared/env.py:77` |
| `python_network_operations` | 5 | `nodes/_otr_comfy_backend.py:384`, `nodes/_otr_feed_fetch.py:249`, `nodes/_otr_openrouter_backend.py:1011`, `nodes/_otr_google_api/client.py:191`, `nodes/_otr_shared/cloud_media_invoke.py:578` |
| `python_command_injection_risk` | 3 | `nodes/_otr_audio_engines/eng_indextts2.py:214`, `nodes/_otr_shared/proc.py:161`, `nodes/_otr_shared/proc.py:168` |

Admin tags the reviewer sees: `any-network-requests` 5, `system-modification` 4,
`any-folder-access` 3, `any-code-execute` 3, `credential-access` 2.

**A SURVEY OF 578 PACKS / 3,707 VERSIONS SAYS THE INFO FINDINGS ARE NOT THE
GATE.** Of 102 versions approved under policy-v0.2 between 08-15 and 09-01,
**zero had a clean scan**; 31 carried `python_command_injection_risk`, 72
carried network findings, 39 carried env findings. Approval is a human verdict
(`reviewed SAFE (GOAL2 verify-deep, policy-v0.2)`); bans name a CLASS
(path traversal 35, unauth side-effect ~18, SSRF/egress ~16, arbitrary file
read 10, RCE 203).

## WHAT THE OPERATOR IS ASKING, in his words

1. **"we were wrong about ffmpeg so we need creative thinking on an approach."**
   Our standing claim has been: this pack encodes video, therefore it must shell
   out to ffmpeg, therefore `subprocess` findings are permanent. **ATTACK THAT.**
   Is it actually true for THIS codebase? Enumerate every `subprocess` spawn we
   ship and ask, per call site, whether it is removable, replaceable
   (PyAV/`av`, imageio-ffmpeg as a library, torchaudio, soundfile, Pillow), or
   genuinely irreducible. Note what we already believe and CHECK it: the pack's
   own claim is that this build's PyAV lacks `libass`/`drawtext` and OpenCV here
   cannot write H.264 (see `docs/GO_FORWARD_PLAN.md` item 4 and
   `docs/2026-09-03-registry-alternatives/`). Is that still true, and does it
   cover ALL the spawns or only the caption/burn one? A partial reduction that
   takes `proc.py` from 2 findings to 1 is worth knowing about.
2. **"maybe we test with a new registry entry."** A CONTROL EXPERIMENT: publish
   a deliberately minimal pack under a DIFFERENT node id and see what the
   scanner/reviewer does with it, to isolate which construct actually drives
   Flagged vs Active. Design that experiment: what is the smallest pack that
   still proves something? What would each outcome tell us? Is it worth a
   publisher's reputation to publish a throwaway id? (Note: `(node_id, version)`
   is uniquely indexed; deleting a NODE is a hard delete and frees its versions,
   deleting a VERSION burns that string forever.)
3. **"maybe remove all tooltips, maybe our widgets are odd -- look for something
   we can strip that isn't needed so we can be more compliant. Audit all inputs,
   outputs, etc."**
   **NOTE FOR YOU, and be honest with him about it:** tooltips and widget labels
   appear in NONE of the 12 findings, so stripping them cannot move that number.
   BUT the widget surface is exactly what the BAN classes key on -- every
   policy-v0.2 ban we surveyed starts "a free STRING widget ..." -- so an audit
   of every `INPUT_TYPES` entry and every `RETURN_TYPES`/output across all
   shipped nodes IS worth doing on its own merits. Do that audit: list every
   input that is (a) inert/unused, (b) a free STRING where a COMBO of known
   values would do, (c) a path or URL that could be a dropdown of registered
   files instead, (d) an output nothing consumes. Name the ones we could
   REMOVE or NARROW without losing a feature. Removing an inert widget is
   operator-approved work ("that's being lazy not to remove an inert widget"),
   but note the cost: a MID-LIST widget removal re-indexes `widgets_values`
   AND every later `dst_slot` across all 63 workflows
   (`CLAUDE.md` section 0), so say which are trailing and which are not.

## HARD CONSTRAINTS -- a suggestion that breaks one is invalid

- No content filtering on generated episodes. Story quality is closed work.
- Local/offline-first: no new cloud dependency, no paid service, no telemetry.
- 16 GB VRAM ceiling on the main box; an 8 GB box also runs this.
- Video-engine adapters are deliberately duplicated; do NOT propose consolidating.
- These three files are BYTE-HASHED and must not change:
  `nodes/_otr_audio_engines/eng_indextts2.py`,
  `scripts/_otr_indextts2_worker.py`, `nodes/_otr_resolved_request.py`.
  (Note two of the 12 findings are IN `eng_indextts2.py` -- so those two are
  immovable by construction. Say so plainly.)
- `.comfyignore` already excludes `tests/`, `docs/`, `scripts/*` (minus three
  workers), `tools/`, `viewer/`. Check what ELSE ships that need not.
- `pyproject.toml` edits auto-fire a publish; every publish burns a version.

## DELIVER

1. **THE FFMPEG VERDICT.** Per spawn site: REMOVABLE / REPLACEABLE (with what,
   and what it costs) / IRREDUCIBLE (why). End with the honest floor: what is
   the minimum achievable `python_command_injection_risk` count, and is getting
   there worth it given that 31 approved packs carry the same finding?
2. **THE INPUT/OUTPUT AUDIT.** A table of every shipped node's inputs/outputs
   flagged inert / over-broad / narrowable, with trailing-vs-mid-list noted.
3. **THE CONTROL EXPERIMENT**, designed or argued against.
4. **ANYTHING WE HAVE NOT THOUGHT OF.** This is the part we most want. What
   would YOU do to get this pack approved, given that the reviewer is a human
   reading code for attacker-reachable classes and not a linter counting
   patterns?

Terse, evidence first, markdown.
