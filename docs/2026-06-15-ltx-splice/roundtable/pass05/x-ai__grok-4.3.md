<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Multiple self-contradictions in PASS-04 folds vs. §§4B/4E plus direct mismatch with grounded eng_ltx_av.py node-gate placement and eng_ltx_video.py current _node_candidates/_build_graph/_sampler_mode structure make the splice unimplementable as written.

MUST-FIX BEFORE BUILD:
1. [PASS-04 fold 6] Node-class gate placed in load() citing "assert_usable runs on CPU box with NO ComfyUI nodes importable" — directly contradicts §4E ("assert_usable gains a node-class gate (mirror eng_ltx_av.py)") and the actual eng_ltx_av.py:assert_usable which performs the resolve_node_class loop (step 4) before any load(). Fix: delete fold 6 sentence; keep gate in assert_usable and update the V-12 comment.
2. [PASS-04 fold 5 + §4B] "Add the new candidates to the BASE _node_candidates() ONLY — _node_candidates_sampling/_i2v INHERIT it" vs. §4B ("in BOTH the base and the i2v candidate dicts"). Fix: delete the "ONLY" clause and the inheritance sentence; follow the explicit "BOTH" wording in §4B.
3. [PASS-04 fold 3] "_distilled_lora_file() returns a (name, path) TUPLE. ... Wire LoraLoaderModelOnly{lora_name:_distilled_lora_name()}" — but eng_ltx_video.py:_distilled_lora_file already returns exactly that tuple and _build_graph already does `lora_name: lora_name`. Fix: remove the split into _name/_path methods and the "fix every existing tuple call site" instruction; the current tuple sites are already compatible.
4. [§4D + PASS-04 fold 7] VRAM keep-set changes from {"checkpoint",...} + results.get("checkpoint") to {"lora", self._TERMINAL} + results.get("lora") — but eng_ltx_video.py:render_clip still does the checkpoint retain and the plan never shows the new _TERMINAL value. Fix: add explicit `self._TERMINAL = "vaedecode"` (or "decode") assignment in the class and update the keep dict + results.get line in one place.
5. [§4F + PASS-04.8] "commercial_clean = True" (Apache GGUF) — but eng_ltx_video.py:244 still has `commercial_clean = False` and no registry.py excerpt is supplied. Fix: change the literal in eng_ltx_video.py and add one-line comment; mark registry.py update as verify: registry.py:CAPABILITIES["ltx_video"].

SHOULD-FIX:
1. [§4A] Hardcoded GGUF filenames lack the env-var pattern used by eng_ltx_av.py:_unet_name etc. Fix: wrap each in a _unet_name() etc. that does os.environ.get("OTR_LTX_VIDEO_UNET", "ltx-2.3-22b-dev-Q4_K_S.gguf").
2. [§6 motion smoke] "frame-diff ≥ 2.0" is stated only in the test description; the plan never shows where otr_ltx_motion_smoke baseline is defined or how the numeric assert is wired. Fix: add the constant + assert to the unit test skeleton.
3. [§5 + §4H] render_driver.py and otr_scifi_16gb_full.json are declared untouched, but no grounding excerpt exists for either. Fix: add "verify: render_driver.py:build_request_from_shot" and "verify: otr_scifi_16gb_full.json nodes".

OPTIONAL / NICE-TO-HAVE:
- Add explicit assert in the §6 graph-shape test that no node key equals "checkpoint" or "encoder" (makes the "Zero W("checkpoint",*) left" claim checkable).
- Keep the CLAUDE.md GPU-reset line (harmless belt-and-suspenders).

CUT THESE (over-engineering):
1. §9 "Gemma device=cpu + lighter projection-ckpt ONLY if full episode shows VRAM pressure" — the splice already pins the mini recipe; the conditional is dead weight until a later ticket.
2. §6 "If a t2v role gates, FILE the per-role-sampler ticket" — already covered by the global distilled invariant in §1; the extra ticket language adds no code.
3. PASS-04.10 "Keep the CLAUDE.md GPU-reset + validator-round-trip" — the plan already says "harmless"; the explicit keep instruction is redundant.