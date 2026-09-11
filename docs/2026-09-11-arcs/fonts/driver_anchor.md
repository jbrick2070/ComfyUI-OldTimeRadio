# Driver anchor -- 3.8-fonts: shared torch-free font resolution

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round. The panel
proposes; the driver disposes and verifies every claim against the real Windows files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding.** A no-brainer gets no arc -- this
row is here because grounding found a genuine fork with more than one defensible
answer.

**THE OPERATOR'S BAR:** *"as long as it doesn't crash when it's not supposed to."*
Exactness is NOT the goal -- this is a fun experimental app, and aesthetic drift is
explicitly acceptable. This row is classified **DURABILITY**.

---

## 1. What is TRUE TODAY, grounded against current code

FOUR distinct, independently-maintained font resolvers exist today, not one shared one:

1. nodes/video_engine.py:97-320 -- `_mono_font_families()` / `_find_mono_font_path()` / `_mono_font_path()` / `_load_font()`. Resolves a monospace TTF/TTC by absolute PATH (Consolas/Courier/Lucida on win32 at line ~112; Menlo/Monaco/Andale on darwin at ~124-141, added by f0aa8e6a 2026-09-09; DejaVu/Liberation on Linux at ~148-168), then falls back to PIL's own recursive bare-name search. Override: `OTR_VIDEO_FONT` (path). Used to MEASURE text for the CRT frame chrome, the caption-burn overlay, and -- via injected `measure_text`/`fit_hero` callables (nodes/_otr_title_card.py:17-19,127-134) -- the hero title card's centering arithmetic. This is the resolver f0aa8e6a fixed (added the darwin branch, family-major search order). No-match fallback: `ImageFont.load_default()` at video_engine.py:309, warned ONCE per process via `_WARNED_BITMAP_FALLBACK`/`log.warning` (line ~297-308). Does not crash; silently mis-measures (bitmap face ignores `size`), which is exactly the macOS off-frame-title defect (PBUG-20260909-04).

2. nodes/_otr_captions.py:105-122 -- `mono_font()`/`_resolve_font()`. Resolves a FAMILY NAME (not a path) written into the ASS `Style:` line libass DRAWS captions and the burned hero title with. `_MONO_BY_PLATFORM = {"win32": "Consolas", "darwin": "Menlo"}`, else `_MONO_FALLBACK = "DejaVu Sans Mono"`. Override: `OTR_CAPTION_MONO_FONT` (family name). This is the "declare" half that was already correct for darwin before the Sep-9 fix -- video_engine's "measure" half was the broken one. Critically, this function does NO existence check at all: it just emits a string. Per the module's own docstring (lines 79-84), if the named family is absent, "libass asks fontconfig for a substitute and silently gets a PROPORTIONAL sans" -- with zero warning anywhere in OTR's logs, because nothing here ever tries to open the font.

3. nodes/otr_credits_roll.py:605-646 -- `_load_font()`. A THIRD independent path-based resolver with its own hardcoded candidate list (JetBrainsMono-Bold/consola/cour/lucon on Windows; DejaVu/Liberation on Linux; Menlo/Monaco/Courier-New-Bold on macOS). Override: `OTR_CREDITS_FONT` (a third, distinct env var name). Already carried macOS/Linux candidates from a SEPARATE, earlier commit (6e54f9ae, 2026-09-01) -- unrelated to and predating the Sep-9 title-card bug hunt. Structurally different failure policy: if nothing resolves, it RAISES `CreditsDataError` (line 641-645) rather than degrading to a bitmap font ('no-fallback: a point-size-less bitmap hero is unacceptable'). This is the one resolver of the four that can genuinely CRASH the render on a box whose fonts aren't at one of its hardcoded absolute paths (it has no PIL bare-name recursive fallback tier the way video_engine does).

4. nodes/_otr_shared/scope_draw.py:148-189 -- `_small_font()`. A FOURTH resolver for scope chrome labels (proportional, not monospace): arial/segoeui/tahoma on win32, Helvetica/Arial/Geneva on darwin, DejaVu/Liberation default. NO override env var exists for this one at all (grepped; confirmed absent). Fixed for the same silent-fallback class the same night as video_engine (commit 765b9e9a, 2026-09-09), via `ImageFont.load_default()` fallback + a one-time warning (line 189, warning at 178-188) -- same pattern as #1, independently re-implemented. Shared by two engines (eng_visualizer.py:271, eng_viz_rainbow.py:236) but is its own module, not reused by #1-3. Its own docstring notes nothing measures this text for placement, so a miss just draws chrome too small, not off-frame.

The project's OWN post-mortem (docs/PROD_BUG_LOG.md:13675-13682, PBUG-20260909-04) already reaches this same conclusion under its 'Still open, and it is a DESIGN item' section: 'The root shape is that video_engine resolves a font by FILE PATH while _otr_captions names a FAMILY to libass... A shared torch-free resolver under nodes/_otr_shared/ is the obvious candidate and now has THREE callers arguing for it' (video_engine, otr_credits_roll, scope_draw). It also names the concrete obstacle: '_otr_captions deliberately keeps thin (it must not import torch)', so unifying isn't mechanical.

Cross-repo, a generalized regression DOES now exist for the narrow 'silent load_default() fallback' defect class: comfyui-custom-node-survival-guide/tests/bug_bible_regression.py::TestSilentFontFallback (Bible id 12.159) is a repo-wide AST walk (not per-module) that fails on ANY `ImageFont.load_default()` call with no `.warning/.warn/.error/print` in the surrounding 18-line window, anywhere in the scanned pack. Both video_engine.py:309 and scope_draw.py:189 already satisfy it (each has a nearby warning). This generalizes the DETECTOR, not the resolver -- it would catch a future regression in any of the four modules but does not unify them into one, and it does not apply to otr_credits_roll.py (which never calls load_default -- it raises instead) or to _otr_captions.py (which never opens a font file at all).

tests/test_load_font_measures_with_a_real_face.py (OTR-local) covers only resolver #1 and its agreement with resolver #2's family name (test_this_platform_measures_in_the_family_libass_draws, line 123). Nothing analogous cross-checks resolver #3 or #4 against anything.

**Key files:** `nodes/video_engine.py:97-320`, `nodes/_otr_captions.py:79-122`, `nodes/otr_credits_roll.py:605-646`, `nodes/_otr_shared/scope_draw.py:140-189`, `nodes/_otr_title_card.py:17-19,120-140`, `tests/test_load_font_measures_with_a_real_face.py`, `docs/PROD_BUG_LOG.md:13576-13690 (PBUG-20260909-04)`, `docs/GO_FORWARD_PLAN.md:144`, `C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/BUG_BIBLE.yaml:11184-11226 (id 12.159)`, `C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py:3207-3273 (TestSilentFontFallback)`

**Blast radius:** Zero effect on the current production render fleet as-is: both machines named in CLAUDE.md (5080/IDREAM, 4060/MRKT) are Windows, where all four resolvers already resolve correctly (Consolas/consola.ttf, Arial, arial.ttf/segoeui.ttf all present under C:\Windows\Fonts). The affected population is non-Windows and atypical installs: the macOS dev/QA laptop that produced PBUG-20260909-04 (already fixed for resolvers #1 and #4), any Linux distro outside the hardcoded candidate lists in resolvers #1/#3, and unknown downstream users since the pack is published to the Comfy Registry (OTR CLAUDE.md section 7A) with no control over installer OS. A shared-resolver refactor would touch nodes/video_engine.py, nodes/_otr_captions.py, nodes/otr_credits_roll.py, and nodes/_otr_shared/scope_draw.py simultaneously -- four modules feeding captions, titles, credits, and two video engines (eng_visualizer.py, eng_viz_rainbow.py) -- so per CLAUDE.md 0B it would need before/after proof that neither box's actual render output changes, even though neither box currently exercises the fallback paths at all.

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- GO_FORWARD_PLAN.md:144 still reads 'Its evidence arrives tonight from the Mac's text-rendering leg... Arc it AFTER that leg reports' -- but the leg already reported on 2026-09-09 (PBUG-20260909-04) and two of the four resolvers were already fixed that same night (f0aa8e6a for nodes/video_engine.py, 765b9e9a for nodes/_otr_shared/scope_draw.py), both BEFORE the plan doc's own last edit (commit 19132847, 2026-09-11 10:07:43). The row's gating condition is already satisfied; it is ready to arc now, not waiting on evidence.
- The row's framing 'shared torch-free font-family/file resolution across captions, titles, credits and scopes' describes a resolver that does not exist yet -- it is the DESIRED end state, not current reality. Current code has four independently-maintained resolvers with no shared module.

## 3. The fork -- this is what the round must pressure-test

Build the shared nodes/_otr_shared/ torch-free font resolver the project's own bug log already calls for (one candidate table per platform, one override-env convention, one fallback/fail policy) -- vs. -- formally accept four independently-maintained resolvers as the standing design (three of four are already individually hardened against the specific silent-bitmap-fallback defect and covered by the portable Bible-12.159 AST detector) and instead close the smaller, cheaper gaps: (a) reconcile the fail-policy split -- 3 resolvers silently degrade-and-warn while otr_credits_roll hard-raises on the same class of miss; (b) give scope_draw.py an override env var (currently has none) and otr_credits_roll's Linux path list the same PIL bare-name recursive fallback video_engine has (currently only 2 hardcoded absolute paths, no recursive search); (c) add an existence check to _otr_captions.mono_font() so a missing declared family fails loud instead of letting libass/fontconfig silently substitute with zero log line. A real merge crosses the module boundary _otr_captions deliberately keeps torch-free/thin (PROD_BUG_LOG.md:13679-13681) -- the project's own record already calls this "a design choice with more than one defensible answer" wanting an arc, not a solo swing.

## 4. Questions for the reviewer -- answer these; do not restate section 1

1. **Is the mechanism in section 1 complete and correct?** Read the cited files
   yourself. Is there a step the driver missed, or something that already partially
   mitigates this?
2. **Which side of the fork survives contact with the real code?** Name the files and
   functions each choice would actually touch, and say which you would not write.
3. **What is the smallest change that resolves it?** Smallest that is CORRECT -- not
   smallest that compiles.
4. **Blast radius, measured not asserted.** This project runs on a 16 GB 5080 and an
   8 GB 4060 (CLAUDE.md section 0B). If the change touches shared code, what
   measurement proves the machine you are NOT fixing is unchanged?
5. **Is any part render-inert enough to ship before a four-machine test wave, or must
   all of it wait?** Say plainly.
6. **What would make this row WRONG to do at all?** Argue the other side once.

## 5. Hard constraints -- a proposal violating any of these is rejected on sight

- **No new content gates**, story-rejection gates, word/duration/cast-size limits,
  standalone checkers, chunkers, or recursive retry loops.
- **No prompt rewrites.** Prompts are hand-crafted per model and CHARACTER-BUDGETED
  (`motion_registers` 240 chars enforced at load, BUG-LOCAL-112; `_fit_motion_slot`
  truncates to 60). Adding conditional nuance spends budget that does not exist.
- **Story/prose QUALITY work is DONE** by operator directive. Correctness defects are
  still open; better prose is not.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful. The goal is to stop CAUSING avoidable ones, never to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more than
  it saves is a worse fix.
- **One canonical graph.** No second workflow JSON, no new output-path owner, no
  reviving deleted code.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling.
- Style: no curse words, never the name "dummy".
