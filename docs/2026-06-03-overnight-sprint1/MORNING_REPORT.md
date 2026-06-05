# OTR Overnight Session -- Morning Report (2026-06-03)

Branch `v2.0-alpha`. You said "do as much as you can, going to sleep." Here is exactly what happened.

## UPDATE -- live audio soak running (you started ComfyUI ~02:30)

ComfyUI came up (v0.22.3, RTX 5080, 15.8 GB free), so I launched a continuous prune-to-audio soak across the working engines. It runs **detached** (~8h budget) and self-gates. **First results: 3/3 episodes SUCCESS, all four working engines validated live, VRAM peak 13.1-13.3 GB (under the 14.5 ceiling).**

| ep | combo | words | engines | status | wall | vram peak |
|----|-------|-------|---------|--------|------|-----------|
| 1 (smoke) | mn_sa3 | 30 | bark / kokoro / SA3 | SUCCESS | 327s | 12.3 |
| 2 | mn_sa3 | 340 | bark / kokoro / SA3 | SUCCESS | 728s | 13.3 |
| 3 | mn_musicgen | 340 | bark / kokoro / musicgen | SUCCESS | 705s | 13.1 |
| 4 | g2_sa3 (gemma-2-2b) | 340 | bark / kokoro / SA3 | running | - | - |

- Engines proven live: **bark** (char voice), **kokoro** (announcer), **stable_audio_3** + **musicgen** (music). Both writer LLMs (mistral-nemo, gemma-2-2b) load + run.
- NOT in the soak: indextts2 / chatterbox / stable_audio_open (opt-in or uninstalled; indextts2 needs the Path B worker). Each 340w episode ~12 min -> expect ~35-40 by morning.
- **No bugs so far.**

Control / inspection:
- Live snapshot: `python scripts\_otr_soak_poll.py`
- Full morning digest: `python scripts\_otr_soak_summary.py`
- Stop early: create the file `_otr_soak_STOP` in the repo root.
- Raw record: `_otr_allnight_results.jsonl` (1 line/episode); console: `_otr_allnight.log`.

---

## TL;DR
- **Sprint 1 (registry spine) BUILT + GREEN.** Full suite **3694 passed / 12 skipped / 0 failed**; Bug Bible **23 passed / 1 skipped / 2 xfailed** (expected).
- **NOT committed** -- deliberate. Your tree is mid-overhaul (20 modified + ~37 untracked, incl. untracked `eng_stable_audio_3.py`). An isolated Sprint 1 commit would fail to import on a clean checkout. Commit recipe below; it is yours to carve up.
- **indextts2 "should work now" -> it does not yet (not installed).** A safe `--no-deps` probe proved the `indextts 2.0.0` wheel is pure-Python and builds **without touching torch**, then I uninstalled it. The real path is your plan's **Path B** (isolated uv venv + subprocess worker), because a full install downgrades the whole Blackwell stack.
- **ComfyUI was DOWN on :8000 all session**, so no live render / end-to-end ladder was possible. Sprints 2/3/6 live parts are deferred to you.

## What I changed (Sprint 1 -- all additive)
1. `nodes/_otr_engine_profiles.py`
   - `EngineProfile` + fields: `rank`, `is_default`, `runtime` (in_graph|oop_venv), `needs_ref_clip`, `caps`, `license_state` (clean|gated|unknown; blank derives from `commercial_clean`), `warn_text`; a `model_validator` enforcing the runtime + license_state enums.
   - Module helpers: `effective_license_state`, `gate_state` (clean|warn|stop), `engine_warning`, `collect_engine_warnings`.
   - Resolver methods: `rank_chain(role)`, `role_default(role)`, `resolve_role_fallback(role, ...)` (walks the rank chain, skips gated-off / stop / bank-incompatible engines, fail-closed if none qualify).
2. `config/audio_engine_profiles.yaml` -- populated all 8 rows with the new metadata.
   - Ranks: char `indextts2(1) > chatterbox(2) > bark(3)`; announcer `kokoro(1) > chatterbox(2)`; music `stable_audio_3(1) > musicgen(2) > stable_audio_open(3)`.
   - `is_default` = indextts2 (char) / kokoro (announcer) / stable_audio_3 (music). `indextts2.runtime = oop_venv`.
3. `tests/test_engine_profiles_rank_gate.py` (NEW) -- 18 tests: schema presence, rank ordering, fallback (indextts2->chatterbox->bark when opt-ins gated off; indextts2 when enabled), gate-state mapping, warning emission + dedup, validators, fail-closed.

### Guardrails honored
- **PD1 byte-identical:** `legacy_first_engines` stays bark-first; engine-level `default_roles` unchanged; the rank-chain resolver is built + tested but **NOT wired into live dispatch**. Live audio is unchanged.
- No node `INPUT_TYPES` / widget changed -> no workflow-JSON surface change (rule 3 n/a this sprint).
- The three-state gate already existed in `_otr_release_gate.py` (True=clean / False=gated-warn / missing|null=stop). I added the explicit `license_state` mirror + profile-level helpers consistent with it -- no parallel gate.

## Tests (logs are gitignored scratch in repo root)
- Full suite: `_otr_full_test2.txt` -> 3694 / 12 / 0.
- Sprint-1 subset: `_otr_s1_test.txt` -> 67 / 0.
- Bug Bible: `_otr_bb_test.txt` -> 23 / 1 / 2.

## Git -- why no commit + recommended recipe
Both files I edited were already in your uncommitted WIP, and `__init__.py` imports the **untracked** `nodes/_otr_audio_engines/eng_stable_audio_3.py`. So a Sprint-1-only commit would not import on a clean checkout, and I can't cleanly separate my delta from your prior edits in shared files. Per your "only Jeffrey merges" posture I left it.

When you review, the cohesive audio-overhaul set is roughly:
```
git add nodes/_otr_audio_engines/eng_stable_audio_3.py nodes/_otr_audio_engines/__init__.py ^
        nodes/_otr_audio_engines/eng_musicgen.py nodes/_otr_engine_profiles.py ^
        config/audio_engine_profiles.yaml tests/test_engine_profiles_rank_gate.py ^
        tests/test_engine_profiles.py tests/test_audio_engine_adapters.py tests/test_stable_audio_theme.py
```
(plus your other WIP as you see fit). Use your COMMIT_EDITMSG + `-F` flow. I did **NOT** push.

## indextts2 -- detail on your "1 C as a test"
- `pip install indextts` -> **no PyPI distribution**. Real source: GitHub `index-tts/index-tts` (commit `830f6f8`). Import is `indextts.infer_v2` -- the adapter's `from indextts.infer import IndexTTS2` is the **v1** path. **Bug to fix in S2.**
- `pip install git+...index-tts --no-deps` -> built `indextts-2.0.0` (pure-Python, `py3-none-any`, 20 MB), torch/numpy/transformers untouched -> then **uninstalled** (venv pristine, confirmed).
- A full (with-deps) install would force **torch 2.10->2.8, numpy 2.4->1.26, transformers 5.5->4.52** -> bricks ComfyUI + bark/kokoro/musicgen/SA3 + FLUX/HuMo. So **Path B only** -- never install into Comfy's venv. `scripts/otr_audio_dep_pilot.py` already covers the import-safety / banned-dep half of "clone-and-prove."

## License sign-off items (Sprint 6 -- I did NOT flip these unattended)
- **bark**: your plan table says clean (MIT/Suno); code says `commercial_clean=false`. I kept it **gated**. Verify Suno terms.
- **stable_audio_3**: plan says "unknown -> stop until pinned"; code + I keep **clean** (Comfy-Org/stable-audio-3 ungated, the working music default). Pin the exact model/license.
- **stable_audio_open**: plan says gated (revenue threshold); code says `commercial_clean=true`. I kept **clean** (mirrors code). Verify Stability community revenue cap.

## What's left + what it needs
- **Sprint 2** (IndexTTS2): needs ComfyUI up + your box. Live steps: freeze-diff clone-and-prove, then Path B uv venv (Py3.10, torch2.8/cu128, `uv sync --extra core`), `hf download IndexTeam/IndexTTS-2`, the subprocess worker, and fix the `infer_v2` import.
- **Sprint 3 / 6**: need GPU + indextts2 working.
- **Sprint 4**: profile rows + gate states already populated; wiring the warn emission into episode-start dispatch is a small, careful step left to avoid touching the byte-identical path unattended.
- **Sprint 5**: the audio cache largely exists; adding `engine_id` to the cache key + the node-3/24 workflow-JSON reconcile touch the JSON -- left for your review.
- **ComfyUI**: start the Desktop app on :8000 before any live work.

## New tool files I added (untracked, optional keep)
- `scripts/_otr_preflight.py` -- read-only preflight (ComfyUI / Ollama / GPU / git). Reusable as the soak Step-1 gate.
- `scripts/_otr_idx_probe.py`, `scripts/_otr_idx_import_test.py` -- indextts probes.
- `_otr_*.txt` in repo root -- scratch test/probe outputs (gitignored, safe to delete).

Say the word and I'll continue: wire the S4 warn-emission, scaffold the S2 Path B worker, or -- once ComfyUI is up -- run the freeze-diff and the smoke ladder.
