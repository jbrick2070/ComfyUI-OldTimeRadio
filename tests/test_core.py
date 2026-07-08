"""
Regression tests for ComfyUI-OldTimeRadio — "SIGNAL LOST"
Canonical Audio Engine v1.0

Coverage:
  - ScriptParser  (canonical v1.0 format, gender-word fallback, edge cases)
  - CleanTextForBark (token whitelist, structural tag stripping)
  - TokenBudget   (1024 floor logic, chunked threshold)
  - CitationGuard (ArXiv / DOI hallucination detection regex)
  - BarkTTS       (warning filter presence, local_files_only pattern)
  - GemmaOrch     (static code patterns: 1024 floor, citation rule, Lemmy egg)
  - AudioContract (waveform shape/dtype/rate — requires torch)
  - SceneSequencer (clip wiring, resampling, silence trim — requires torch)
  - VintageRadioFilter (all presets — requires torch)
  - WorkflowJSON  (full: integrity, node IDs, links, parity)
  - AudioBatcher  (batch/pad/unpack — requires torch)

Run in ComfyUI venv for full coverage:
  python -m pytest tests/ -v

Run in any environment (torch-dependent tests auto-skip):
  python -m pytest tests/ -v
"""

import ast
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

# 2026-05-10: `parser` fixture removed. The script parser lived on the
# legacy `LegacyLLMScriptWriter` class, which was deleted in the legacy
# cleanup commit alongside `nodes/_otr_legacy_writer.py`. The v2.0 LPL
# writer (OTR_LedgerScriptWriter) does not have a parser surface --
# beats come pre-structured from the outline LLM. The four parser test
# classes that used this fixture (TestScriptParserCanonical,
# TestScriptParserGenderFallback, TestParserV5SFXExtraction,
# TestParserV3V4Patterns) were removed in the same commit.


def _load_workflow(name):
    path = os.path.join(os.path.dirname(__file__), "..", "workflows", name)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEAN TEXT FOR BARK — Token Whitelist Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestCleanTextForBark:
    """_clean_text_for_bark() strips structural tags and enforces 13-token whitelist."""

    @pytest.fixture
    def clean_fn(self):
        from nodes._otr_bark_lib import _clean_text_for_bark
        return _clean_text_for_bark

    def test_strips_voice_tag(self, clean_fn):
        assert "[VOICE:" not in clean_fn("[VOICE: HAYES, male, 40s, calm, low] Hello world.")

    def test_strips_sfx_tag(self, clean_fn):
        assert "[SFX:" not in clean_fn("[SFX: door slam] Speech here.")

    def test_strips_env_tag(self, clean_fn):
        assert "[ENV:" not in clean_fn("[ENV: sterile lab] Speech here.")

    def test_strips_scene_header(self, clean_fn):
        assert "===" not in clean_fn("=== SCENE 1 === Some text.")

    def test_preserves_dialogue(self, clean_fn):
        text = "[VOICE: HAYES, male, 40s, calm, low] They found the switch."
        assert "They found the switch." in clean_fn(text)

    def test_supported_token_laughs_preserved(self, clean_fn):
        text = "[VOICE: HAYES, male, 40s, calm, low] [laughs] That is impossible."
        assert "[laughs]" in clean_fn(text)

    def test_unsupported_whispers_removed(self, clean_fn):
        text = "[VOICE: HAYES, male, 40s, calm, low] [whispers] Careful now."
        assert "[whispers]" not in clean_fn(text)

    def test_unsupported_shouts_removed(self, clean_fn):
        text = "[VOICE: HAYES, male, 40s, calm, low] [shouts] Get out!"
        assert "[shouts]" not in clean_fn(text)

    def test_all_13_bark_tokens_pass_whitelist(self, clean_fn):
        supported = [
            "[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]",
            "[clears throat]", "[coughs]", "[pants]", "[sobs]",
            "[grunts]", "[groans]", "[whistles]", "[sneezes]",
        ]
        for token in supported:
            result = clean_fn(f"[VOICE: X, male, 40s, calm, low] {token} Speech.")
            assert token in result, f"Supported Bark token {token} was stripped incorrectly"

    def test_no_double_spaces(self, clean_fn):
        result = clean_fn("[VOICE: HAYES, male, 40s, calm, low] Hello   world.")
        assert "  " not in result

    def test_empty_string(self, clean_fn):
        assert clean_fn("").strip() == ""

    def test_scene_sequencer_clean_matches_batcher(self):
        """The canonical _otr_bark_lib cleaner and the SceneSequencer copy must
        produce identical output for the same input (regex parity)."""
        from nodes._otr_bark_lib import _clean_text_for_bark as lib_clean
        from nodes.scene_sequencer import _clean_text_for_bark as ss_clean
        probe = "[VOICE: X, male, 40s, calm, low] [whispers] Hello [laughs] world [shouts] stop."
        lib = lib_clean(probe)
        ss = ss_clean(probe)
        assert lib == ss, (
            f"_clean_text_for_bark output differs between modules:\n"
            f"  _otr_bark_lib: {lib!r}\n"
            f"  SceneSequencer:{ss!r}"
        )

    # -- BUG-LOCAL-101 paren-drop assertions ---------------------------------
    # Stage-direction parentheticals previously translated to Bark non-verbal
    # tokens ((panting) -> [pants]) which leaked breath/throat audio before
    # dialogue and confused listeners. They now drop wholesale.

    def test_paren_panting_dropped_no_pants_token(self, clean_fn):
        # The exact Stellar Shadows l004 trigger
        text = "(panting) We've lost too much time already, Stanley!"
        result = clean_fn(text)
        assert "[pants]" not in result, (
            "paren-form (panting) must NOT translate to Bark [pants] token "
            "(BUG-LOCAL-101)"
        )
        assert "(panting)" not in result, "paren content must be stripped"
        assert "We've lost too much time" in result, "dialogue must survive"

    def test_paren_laughs_dropped_no_laughs_token(self, clean_fn):
        text = "(laughs) That is impossible."
        result = clean_fn(text)
        assert "[laughs]" not in result, (
            "paren-form (laughs) must NOT translate to Bark [laughs] token"
        )
        assert "(laughs)" not in result
        assert "That is impossible" in result

    def test_paren_unknown_emotion_dropped(self, clean_fn):
        # Emotions that never had a Bark token mapping (already ""):
        # confirm they still drop cleanly
        for emotion in ["(defensive)", "(skeptical)", "(grinning)", "(insistent)"]:
            text = f"{emotion} The reading is wrong."
            result = clean_fn(text)
            assert emotion not in result, f"{emotion} must be stripped"
            assert "The reading is wrong" in result

    def test_multiple_parens_in_one_line_all_dropped(self, clean_fn):
        text = "(softly) Lev... (whispering) what aren't you telling me?"
        result = clean_fn(text)
        assert "(" not in result and ")" not in result, (
            "all paren content must be stripped; no leftover open/close"
        )
        assert "Lev" in result and "what aren't you telling me" in result

    def test_explicit_bracket_token_still_passes_whitelist(self, clean_fn):
        # Writers who deliberately want a Bark non-verbal token can still
        # write [laughs] inline -- the whitelist preserves it. Only the
        # paren -> token translation went away.
        text = "Well, [laughs] that's a problem."
        result = clean_fn(text)
        assert "[laughs]" in result, (
            "explicit [laughs] in dialogue must survive the bracket whitelist "
            "(only paren-to-token translation is gone post-BUG-101)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOKEN BUDGET — 1024 Floor Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenBudget:
    """Verify max_new_tokens formula without running any model.

    Uses _TOKEN_RATIO_DIALOGUE (1.6) for single-pass and
    _TOKEN_RATIO_ACT_CHUNK (2.0) for chunked per-act generation.
    """
    RATIO_DIALOGUE = 1.6   # must match _TOKEN_RATIO_DIALOGUE
    RATIO_ACT = 2.0        # must match _TOKEN_RATIO_ACT_CHUNK

    def _compute_single(self, target_words):
        tokens = max(int(target_words * self.RATIO_DIALOGUE), 1024)
        return min(tokens, 8192)

    def _compute_act(self, words_per_act):
        return max(1024, min(4096, int(words_per_act * self.RATIO_ACT)))

    def test_350w_hits_floor(self):
        assert self._compute_single(350) == 1024  # 350*1.6=560 < 1024 -> floor

    def test_420w_hits_floor(self):
        assert self._compute_single(420) >= 1024

    def test_500w_hits_floor(self):
        assert self._compute_single(500) == 1024  # 500*1.6=800 < 1024 -> floor

    def test_700w_above_floor(self):
        t = self._compute_single(700)
        assert t == 1120  # 700*1.6=1120

    def test_no_episode_below_1024(self):
        for w in [350, 420, 500, 600, 700]:
            assert self._compute_single(w) >= 1024, f"{w}w budget {self._compute_single(w)} < 1024"

    def test_act_budget_scales(self):
        assert self._compute_act(200) == 1024   # 200*2.0=400 < 1024 -> floor
        assert self._compute_act(600) == 1200   # 600*2.0=1200
        assert self._compute_act(2500) == 4096  # 2500*2.0=5000 > 4096 -> cap

    def test_ceiling_8192(self):
        assert self._compute_single(700) <= 8192


# ─────────────────────────────────────────────────────────────────────────────
# 5. CITATION GUARD — ArXiv / DOI Hallucination Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationGuard:
    ARXIV = re.compile(r'\barXiv:\s*\d{4}\.\d{4,5}\b', re.IGNORECASE)
    DOI   = re.compile(r'\bdoi\.org/10\.\d{4,}/\S+', re.IGNORECASE)

    def test_detects_arxiv_id(self):
        assert self.ARXIV.search("See arXiv:2401.12345 for details.")

    def test_detects_arxiv_case_insensitive(self):
        assert self.ARXIV.search("Published at ARXIV:2401.12345.")

    def test_detects_arxiv_with_space(self):
        assert self.ARXIV.search("See arXiv: 2401.12345 for details.")

    def test_detects_doi(self):
        assert self.DOI.search("See doi.org/10.1038/s41586-024-07100-0.")

    def test_no_match_on_real_headline(self):
        text = "Scientists discover blood pressure compound. Nature, 2026."
        assert not self.ARXIV.search(text)
        assert not self.DOI.search(text)

    def test_guard_flags_hallucinated_id(self):
        hallucinated = ["arXiv:2401.99999"]
        real = "nature blood pressure compound renal salt retention 2026"
        bad = [h for h in hallucinated if h.lower() not in real.lower()]
        assert len(bad) == 1

    def test_guard_passes_real_id_in_source(self):
        real_id = "arXiv:2401.12345"
        real = f"see {real_id.lower()} for methodology"
        bad = [h for h in [real_id] if h.lower() not in real.lower()]
        assert len(bad) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. BARK TTS — Static Code Checks
# ─────────────────────────────────────────────────────────────────────────────

# Voice-path-cleanbreak 2026-05-12 (P3): TestBarkTTSCodePatterns deleted.
# nodes/bark_tts.py (the legacy single-line OTR_BarkTTS node) was deleted
# in lockstep. Audio clean-break (1a): the batch OTR_BatchBarkGenerator node
# was retired too -- bark is now a per_line registry engine
# (nodes/_otr_audio_engines/eng_bark.py), its inference in nodes/_otr_bark_lib.py,
# exercised by tests/test_bark_cast_contract.py + tests/test_batch_character_voices.py.


# ─────────────────────────────────────────────────────────────────────────────
# 7. STORY ORCHESTRATOR — Static Code Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestStoryOrchestratorCodePatterns:

    @pytest.fixture(scope="class")
    def src(self):
        """Combined source of the LLM stack: orchestrator + loader.

        S31 B2 ported `_load_llm`'s ~613 LOC bitsandbytes body OUT of
        `story_orchestrator.py` and INTO `_otr_model_loader.load_llm`.
        The legacy patterns this test class pins
        (`1024` floor on max_new_tokens, `local_files_only` tokenizer
        load path, `max_length=None` generate kwarg) now live in the
        loader file. The orchestrator file is the consumer side
        (`_generate_with_llm`, `_run_with_timeout`, streamer wiring)
        plus the ~10-line `_load_llm` shim (deleted at S31 B4).

        Returning both files concatenated keeps the legacy assertions
        load-bearing across the body-move boundary without splitting
        every test by file. Each pattern is unique enough that the
        union assertion is unambiguous (`max_length=None` only appears
        in the generate path, `1024` `max(` only on the floor, etc).
        """
        nodes_dir = os.path.join(os.path.dirname(__file__), "..", "nodes")
        parts = []
        for fname in ("story_orchestrator.py", "_otr_model_loader.py"):
            with open(os.path.join(nodes_dir, fname), encoding="utf-8") as f:
                parts.append(f.read())
        return "\n".join(parts)

    def test_transformers_import_in_try_block(self, src):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if (isinstance(child, ast.ImportFrom)
                            and child.module == "transformers.generation.streamers"):
                        return
        pytest.fail("BaseStreamer import not inside a try/except block")

    def test_1024_floor_present(self, src):
        assert "1024" in src
        assert "max(" in src

    def test_max_length_none_in_generate(self, src):
        """S31 B4 update: `_generate_with_llm` was deleted from
        `story_orchestrator.py` at S31 B4 (Hard rule #1A non-deferrable).
        The `model.generate(..., max_length=None, ...)` call site that
        this assertion historically pinned lived inside that function's
        body. With the function gone the pattern is structurally
        retired -- the modern generate surface is
        `_otr_model_loader.make_generate_fn` which does NOT pass
        `max_length=None` explicitly (transformers' default is `None`
        already; explicit None was a noisy artefact of the legacy
        wrapper). Converted from a "must exist" assertion to a
        "must NOT exist" guard on the deleted symbol.
        """
        assert "max_length=None" not in src, (
            "`max_length=None` should be absent from the LLM stack "
            "post-S31 B4. The legacy `_generate_with_llm` wrapper (and "
            "its explicit `max_length=None` kwarg) is deleted; modern "
            "callers use `make_generate_fn` which omits the kwarg "
            "(transformers default is already None)."
        )

    # Voice-path-cleanbreak 2026-05-12 (P3): test_gender_words_frozenset,
    # test_voice_tag_example_has_charactername, and test_lemmy_easter_egg
    # were deleted. They pinned source-level strings inside the legacy
    # LLMScriptWriter / Director code paths that were extracted out of
    # story_orchestrator.py during the LPL writer rewrite. The new
    # OTR_LedgerScriptWriter has its own dedicated cast / writer tests;
    # the LEMMY easter-egg framework is tracked in BUG_LOG (orphaned
    # in LPL migration). No replacement assertions needed.
    #
    # Sprint C C2b 2026-05-15: test_citation_rule_present was also
    # deleted. It pinned the substring "CITATION RULE" / "cite ONLY"
    # inside the orchestrator+loader source union. Those substrings
    # lived ONLY inside the SCRIPT_SYSTEM_PROMPT constant, which was
    # itself an orphan from the eec4718 LPL extraction sprint (no live
    # consumer; verified via 8-search audit recorded in `SPRINT.md` §B).
    # SCRIPT_SYSTEM_PROMPT + SCAFFOLDING_PREAMBLE were deleted at C2b
    # per the no-legacy-back-compat standing directive. With the
    # constant gone, the test's contract is vacuously defunct -- a
    # deleted constant has no content to pin. No replacement assertion
    # needed because no live prompt module currently carries an
    # equivalent citation-rule string; if the LPL pipeline needs one,
    # the replacement test should live alongside whichever per-phase
    # prompt module gets the rule (`_otr_outline._SYSTEM_PROMPT`,
    # `_otr_line_composer._SYSTEM_PROMPT`, `_otr_ledger_reviewer`, or
    # `_otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT`), not as a
    # union-of-files substring scan here.

    def test_local_files_only_gemma(self, src):
        assert "local_files_only" in src


# ─────────────────────────────────────────────────────────────────────────────
# 8. WORKFLOW JSON — Full Workflow Integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowJSONFull:

    @pytest.fixture(scope="class")
    def wf(self):
        return _load_workflow("otr_canonical.json")

    def test_required_keys(self, wf):
        for k in ["nodes", "links", "last_node_id", "last_link_id"]:
            assert k in wf

    def test_has_minimum_nodes(self, wf):
        assert len(wf["nodes"]) >= 7

    def test_node_ids_unique(self, wf):
        ids = [n["id"] for n in wf["nodes"]]
        assert len(ids) == len(set(ids))

    def test_link_ids_unique(self, wf):
        ids = [l[0] for l in wf["links"]]
        assert len(ids) == len(set(ids))

    def test_last_node_id_valid(self, wf):
        assert wf["last_node_id"] >= max(n["id"] for n in wf["nodes"])

    def test_last_link_id_valid(self, wf):
        if wf["links"]:
            assert wf["last_link_id"] >= max(l[0] for l in wf["links"])

    def test_links_reference_existing_nodes(self, wf):
        ids = {n["id"] for n in wf["nodes"]}
        for lid, src, ss, dst, ds, dtype in wf["links"]:
            assert src in ids, f"Link {lid}: src {src} missing"
            assert dst in ids, f"Link {lid}: dst {dst} missing"

    def test_no_input_slot_collisions(self, wf):
        seen = {}
        for lid, src, ss, dst, ds, dtype in wf["links"]:
            k = (dst, ds)
            assert k not in seen, f"Slot collision: links {seen[k]} and {lid}"
            seen[k] = lid

    def test_node_types_otr_or_known(self, wf):
        # Stock ComfyUI nodes allowed in the FULL workflow. The image-rendering
        # slice (PASS1/2/3 composites + PNG saves) needs CheckpointLoaderSimple
        # for the FLUX checkpoint and SaveImage for the PASS3 PNG outputs.
        # The HuMo in-graph loader chain (BUG-078) needs UNETLoader / CLIPLoader
        # / VAELoader / AudioEncoderLoader / LoraLoaderModelOnly / ModelSamplingSD3
        # plus the KJNodes SageAttention patcher.
        # Sprint 3 (2026-05-02): the LTX 2B v0.9 stage uses ComfyUI-LTXVideo's
        # LowVRAMCheckpointLoader (a CheckpointLoaderSimple subclass with a
        # `dependencies` input for sequential model load -- ensures HuMo unloads
        # before LTX claims VRAM, satisfying the C2 sequencing intent).
        known = {
            "PreviewAudio", "PreviewImage", "Note",
            "CheckpointLoaderSimple", "SaveImage",
            # HuMo loader chain
            "UNETLoader", "CLIPLoader", "VAELoader",
            "AudioEncoderLoader", "LoraLoaderModelOnly",
            "ModelSamplingSD3", "PathchSageAttentionKJ",
            # LTX loader chain (Sprint 3)
            "LowVRAMCheckpointLoader",
            # LTX 2.3 loader chain (BUG-LOCAL-117 cutover 2026-05-06):
            # Gemma encoder loader from ComfyUI-LTXVideo replaces the prior
            # CLIPLoader+t5xxl chain. Required by OTR_BatchLTXRender when
            # OTR_LTX_ENGINE=v2_3 (default). RES4LYF nodes (ClownSampler_Beta,
            # MultimodalGuider, GuiderParameters, LTXVTiledVAEDecode) are
            # called inside batch_ltx_render.py via _call() and never appear
            # in the workflow JSON node list, so they don't need to be in
            # this whitelist.
            "LTXAVTextEncoderLoader",
        }
        for n in wf["nodes"]:
            assert n["type"].startswith("OTR_") or n["type"] in known

    def test_required_pipeline_nodes(self, wf):
        # OTR_LLMDirector removed from required set in voice-path-cleanbreak
        # Sprint 2 (2026-05-12). Director class + workflow node deleted;
        # video consumers migrated to read meta.visual_plan from L3 ledger.
        types = {n["type"] for n in wf["nodes"]}
        required = {
            "OTR_LedgerScriptWriter",
            # Wave 2b: OTR_BatchCharacterVoices (v2) replaced the legacy
            # OTR_BatchBarkGenerator instance; it delegates to bark by default.
            "OTR_BatchCharacterVoices", "OTR_SceneSequencer",
            "OTR_EpisodeAssembler",
        }
        assert not (required - types), f"Missing: {required - types}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. (Removed) Lite workflow tests - otr_scifi_16gb_lite.json was removed
#    in commit 44cbdec. Tests referencing it are no longer valid.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 10. AUDIO CONTRACT — requires torch
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestAudioContract:

    def _check(self, audio):
        assert isinstance(audio, dict)
        assert "waveform" in audio and "sample_rate" in audio
        wf = audio["waveform"]
        assert isinstance(wf, torch.Tensor)
        assert wf.dim() == 3
        assert wf.dtype == torch.float32
        assert 8000 <= audio["sample_rate"] <= 96000

    # Voice-path-cleanbreak 2026-05-12 (P3): test_sfx_generator and
    # test_sfx_all_types were deleted with the legacy OTR_SFXGenerator
    # node class. The procedural SFX node + library were removed in the
    # 2026-05-29 lean-down.

    def test_episode_assembler(self):
        from nodes.scene_sequencer import EpisodeAssembler
        scene = {"waveform": torch.randn(1, 1, 48000).float(), "sample_rate": 48000}
        # assemble() returns 4 outputs, matching EpisodeAssembler.RETURN_NAMES:
        # episode_audio, output_path, episode_info, audio_done.
        audio, _, _, _ = EpisodeAssembler().assemble(scene, "Test")
        self._check(audio)


# ─────────────────────────────────────────────────────────────────────────────
# 11. SCENE SEQUENCER — Clip Wiring (requires torch)
# ─────────────────────────────────────────────────────────────────────────────

# Voice-path-cleanbreak 2026-05-12 (P3): TestSceneSequencerClipWiring's
# two cases pinned the legacy parser-list shape that
# _otr_ledger_consumers.load_ledger now hard-rejects via ValueError, plus
# the production_plan_json kwarg that the cleanbreak deleted. The new
# L3-native SceneSequencer is exercised by tests/test_sequencer_ledger.py
# (4 cases) and the integration assembly tests. The legacy-shape
# coverage is deliberately retired -- no replacement assertions.


# ─────────────────────────────────────────────────────────────────────────────
# 12. SILENCE TRIMMING (requires torch)
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestSilenceTrimming:

    def test_strips_trailing_zeros(self):
        from nodes.scene_sequencer import _trim_trailing_silence
        audio = np.concatenate([
            np.random.randn(48000).astype(np.float32) * 0.5,
            np.zeros(96000, dtype=np.float32)
        ])
        t = _trim_trailing_silence(audio, threshold=1e-4)
        assert len(t) < 55000
        assert len(t) > 47000


    def test_all_silence_returns_min_samples(self):
        from nodes.scene_sequencer import _trim_trailing_silence
        t = _trim_trailing_silence(np.zeros(48000, dtype=np.float32), threshold=1e-4)
        assert len(t) == 100


# ─────────────────────────────────────────────────────────────────────────────
# 13. VRAM GUARDIAN NODE — v1.5 Phase 1
# ─────────────────────────────────────────────────────────────────────────────

class TestVRAMGuardianNode:
    """Verify OTR_VRAMGuardian node is importable and structurally correct."""

    def test_import(self):
        from nodes.vram_guardian import VRAMGuardian
        assert VRAMGuardian is not None

    def test_category(self):
        from nodes.vram_guardian import VRAMGuardian
        assert VRAMGuardian.CATEGORY == "OldTimeRadio"

    def test_function_name(self):
        from nodes.vram_guardian import VRAMGuardian
        assert VRAMGuardian.FUNCTION == "flush"

    def test_return_types(self):
        from nodes.vram_guardian import VRAMGuardian
        assert VRAMGuardian.RETURN_TYPES == ("STRING",)

    def test_input_types_valid(self):
        from nodes.vram_guardian import VRAMGuardian
        inputs = VRAMGuardian.INPUT_TYPES()
        assert "required" in inputs
        assert "optional" in inputs
        assert "trigger" in inputs["optional"]

    @requires_torch
    def test_passthrough_returns_input(self):
        from nodes.vram_guardian import VRAMGuardian
        node = VRAMGuardian()
        result = node.flush(trigger="test_value")
        assert result == ("test_value",)

    @requires_torch
    def test_passthrough_empty_default(self):
        from nodes.vram_guardian import VRAMGuardian
        node = VRAMGuardian()
        result = node.flush()
        assert result == ("",)

    def test_registered_in_init(self):
        """VRAMGuardian must be in __init__.py NODE_CLASS_MAPPINGS."""
        init_path = os.path.join(os.path.dirname(__file__), "..", "__init__.py")
        with open(init_path, encoding="utf-8") as f:
            src = f.read()
        assert "OTR_VRAMGuardian" in src
        assert "vram_guardian" in src

    def test_node_class_mappings(self):
        from nodes.vram_guardian import NODE_CLASS_MAPPINGS
        assert "OTR_VRAMGuardian" in NODE_CLASS_MAPPINGS


# ─────────────────────────────────────────────────────────────────────────────
# 15. AUDIOGEN CANONICAL SFX CONSUMPTION (v1.5 Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

# Audio clean-break 1c: TestAudioGenCanonicalSFX deleted -- the AudioGen / SFX
# node (batch_audiogen_generator.py) was retired (SFX is out of the v2 audio
# infrastructure), so its source-pattern tests no longer apply.


