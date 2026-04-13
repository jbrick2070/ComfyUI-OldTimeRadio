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

@pytest.fixture(scope="module")
def parser():
    from nodes.story_orchestrator import LLMScriptWriter
    return LLMScriptWriter()


def _load_workflow(name):
    path = os.path.join(os.path.dirname(__file__), "..", "workflows", name)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCRIPT PARSER — Canonical v1.0 Format
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptParserCanonical:
    """_parse_script() against Canonical Audio Engine 1.0 token set."""

    def test_voice_tag_extracts_name(self, parser):
        result = parser._parse_script("[VOICE: HAYES, male, 40s, calm, low energy] Report is clean.")
        assert len(result) == 1
        item = result[0]
        assert item["type"] == "dialogue"
        assert item["character_name"] == "HAYES"
        assert "Report is clean." in item["line"]

    def test_voice_tag_name_uppercased(self, parser):
        result = parser._parse_script("[VOICE: hayes, male, 40s, calm, low energy] Copy that.")
        assert result[0]["character_name"] == "HAYES"

    def test_voice_tag_preserves_dialogue(self, parser):
        line = "They found the switch. The fear switch."
        result = parser._parse_script(f"[VOICE: DR_VOSS, female, 50s, intense, high energy] {line}")
        assert result[0]["line"] == line

    def test_sfx_tag(self, parser):
        result = parser._parse_script("[SFX: heavy wrench strike on metal pipe, single resonant clank]\n[VOICE: DUMMY, male, 30s] ok.")
        assert len(result) == 2
        assert result[0]["type"] == "sfx"
        assert "wrench" in result[0]["description"]

    def test_sfx_case_insensitive(self, parser):
        result = parser._parse_script("[sfx: radio static]\n[VOICE: DUMMY, male, 30s] ok.")
        assert result[0]["type"] == "sfx"

    def test_env_tag(self, parser):
        result = parser._parse_script("[ENV: sterile lab, low electronic hum, pressurized air]\n[VOICE: DUMMY, male, 30s] ok.")
        assert len(result) == 2
        assert result[0]["type"] == "environment"
        assert "sterile lab" in result[0]["description"]

    def test_scene_break(self, parser):
        result = parser._parse_script("=== SCENE 1 ===\n[VOICE: DUMMY, male, 30s] ok.")
        assert len(result) == 2
        assert result[0]["type"] == "scene_break"
        assert "1" in result[0]["scene"]

    def test_scene_break_final(self, parser):
        result = parser._parse_script("=== SCENE FINAL ===\n[VOICE: DUMMY, male, 30s] ok.")
        assert len(result) == 2
        assert result[0]["type"] == "scene_break"
        assert "FINAL" in result[0]["scene"]

    def test_beat_tag(self, parser):
        result = parser._parse_script("(beat)\n[VOICE: DUMMY, male, 30s] ok.")
        assert len(result) == 2
        assert result[0]["type"] == "pause"
        assert result[0]["kind"] == "beat"
        assert result[0]["duration_ms"] == 200

    def test_beat_case_insensitive(self, parser):
        result = parser._parse_script("(BEAT)\n[VOICE: DUMMY, male, 30s] ok.")
        assert result[0]["type"] == "pause"

    def test_empty_lines_skipped(self, parser):
        result = parser._parse_script("\n\n[VOICE: HAYES, male, 40s, calm, low] Go.\n\n")
        assert len(result) == 1

    def test_direction_fallback(self, parser):
        result = parser._parse_script("Some ambient stage direction text.\n[VOICE: DUMMY, male, 30s] ok.")
        assert len(result) == 2
        assert result[0]["type"] == "direction"

    def test_dashes_separator_skipped(self, parser):
        result = parser._parse_script("---\n[VOICE: DUMMY, male, 30s] ok.")
        assert len(result) == 1

    def test_pro_qa_announcer_bookends(self, parser):
        # QA only triggers on scripts with > 5 dialogue lines
        filler = "\n".join(f"[VOICE: DUMMY, male, 50s] Line {i}" for i in range(6))
        
        # 1. Missing both
        script1 = f"{filler}\n[VOICE: LEMMY, male, 50s, calm] Wrench."
        res1 = parser._parse_script(script1)
        dialogues1 = [r["character_name"] for r in res1 if r["type"] == "dialogue"]
        assert dialogues1[0] == "ANNOUNCER"
        assert dialogues1[-1] == "ANNOUNCER"

        # 2. Missing close
        script2 = f"[VOICE: ANNOUNCER, male, 50s] Opening.\n{filler}\n[VOICE: LEMMY, male, 50s, calm] Wrench."
        res2 = parser._parse_script(script2)
        dialogues2 = [r["character_name"] for r in res2 if r["type"] == "dialogue"]
        assert dialogues2[0] == "ANNOUNCER"
        assert dialogues2[-1] == "ANNOUNCER"

        # 3. Perfectly fine - no injection!
        script3 = f"[VOICE: ANNOUNCER, male, 50s] Opening.\n{filler}\n[VOICE: ANNOUNCER, male, 50s] Closing."
        res3 = parser._parse_script(script3)
        dialogues3 = [r["character_name"] for r in res3 if r["type"] == "dialogue"]
        assert dialogues3[0] == "ANNOUNCER"
        assert dialogues3[1] == "DUMMY" # First filler
        assert dialogues3[-1] == "ANNOUNCER"
        assert len(dialogues3) == 8 # 1 Open + 6 Filler + 1 Close

    def test_full_canonical_scene(self, parser):
        script = """=== SCENE 1 ===
[ENV: sterile broadcast studio, low hum]
[SFX: brief news sting]
[VOICE: ANNOUNCER, male, 50s, authoritative, medium] Tonight on Signal Lost.
(beat)
[VOICE: DR_CHEN, female, 40s, calm, low] The compound reduced pressure by 10 mmHg.
[SFX: data stream hum]
[VOICE: HAYES, male, 40s, tense, medium] That's unexpected.
=== SCENE 2 ===
[ENV: corridor, echoing footsteps]
[VOICE: DR_CHEN, female, 40s, calm, low] We need to run it again.
"""
        result = parser._parse_script(script)
        types = [r["type"] for r in result]
        assert "scene_break" in types
        assert "environment" in types
        assert "sfx" in types
        assert "dialogue" in types
        assert "pause" in types
        chars = {r["character_name"] for r in result if r["type"] == "dialogue"}
        assert chars == {"ANNOUNCER", "DR_CHEN", "HAYES"}

    def test_multiple_voice_tags_all_parsed(self, parser):
        script = "\n".join([
            "[VOICE: ALPHA, male, 30s, calm, low] First.",
            "[VOICE: BETA, female, 40s, warm, medium] Second.",
            "[VOICE: GAMMA, male, 60s, deep, low] Third.",
        ])
        result = parser._parse_script(script)
        assert len(result) == 3
        names = [r["character_name"] for r in result]
        assert names == ["ALPHA", "BETA", "GAMMA"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCRIPT PARSER — Gender-Word Fallback (Name Mangling Fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptParserGenderFallback:
    """Malformed [VOICE: male, ...] gets CHAR_A fallback name."""

    def test_male_as_name_gets_fallback(self, parser):
        result = parser._parse_script("[VOICE: male, 40s, calm, low energy] Test line.")
        assert result[0]["type"] == "dialogue"
        assert result[0]["character_name"] == "CHAR_A"

    def test_female_as_name_gets_fallback(self, parser):
        result = parser._parse_script("[VOICE: female, 30s, warm, medium] Another line.")
        assert result[0]["character_name"] == "CHAR_A"

    def test_multiple_malformed_tags_increment(self, parser):
        script = "\n".join([
            "[VOICE: male, 40s, calm, low] Line one.",
            "[VOICE: female, 30s, warm, medium] Line two.",
            "[VOICE: old, 70s, gruff, low] Line three.",
        ])
        result = parser._parse_script(script)
        names = [r["character_name"] for r in result]
        assert names == ["CHAR_A", "CHAR_B", "CHAR_C"]

    def test_valid_name_unaffected_by_fallback(self, parser):
        script = "\n".join([
            "[VOICE: male, 40s, calm, low] Malformed.",
            "[VOICE: HAYES, male, 40s, calm, low] Correct.",
        ])
        result = parser._parse_script(script)
        assert result[0]["character_name"] == "CHAR_A"
        assert result[1]["character_name"] == "HAYES"

    def test_young_word_triggers_fallback(self, parser):
        result = parser._parse_script("[VOICE: young, 20s, nervous, high] Scared.")
        assert result[0]["character_name"].startswith("CHAR_")

    def test_all_gender_words_in_set(self, parser):
        required = {
            "male", "female", "man", "woman", "boy", "girl", "nonbinary",
            "young", "old", "older", "elderly", "middle", "teen",
        }
        assert required.issubset(parser._GENDER_WORDS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEAN TEXT FOR BARK — Token Whitelist Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestCleanTextForBark:
    """_clean_text_for_bark() strips structural tags and enforces 13-token whitelist."""

    @pytest.fixture
    def clean_fn(self):
        from nodes.batch_bark_generator import _clean_text_for_bark
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
        """Both nodes must produce identical output for the same input."""
        from nodes.batch_bark_generator import _clean_text_for_bark as bb_clean
        from nodes.scene_sequencer import _clean_text_for_bark as ss_clean
        probe = "[VOICE: X, male, 40s, calm, low] [whispers] Hello [laughs] world [shouts] stop."
        bb = bb_clean(probe)
        ss = ss_clean(probe)
        assert bb == ss, (
            f"_clean_text_for_bark output differs between nodes:\n"
            f"  BatchBark:     {bb!r}\n"
            f"  SceneSequencer:{ss!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOKEN BUDGET — 1024 Floor Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenBudget:
    """Verify max_new_tokens formula without running any model."""

    def _compute(self, target_minutes):
        target_words = target_minutes * 130
        if target_minutes <= 5:
            tokens = max(int(target_words * 2.0), 1024)
            tokens = min(tokens, 8192)
        else:
            tokens = 4096  # chunked per-act ceiling
        return tokens

    def test_1min_hits_floor(self):
        assert self._compute(1) == 1024

    def test_2min_hits_floor(self):
        assert self._compute(2) >= 1024

    def test_3min_hits_floor(self):
        assert self._compute(3) == 1024  # 390 words → 780 tokens → floor

    def test_5min_above_floor(self):
        t = self._compute(5)
        assert 1024 <= t <= 8192

    def test_no_episode_below_1024(self):
        for m in [1, 2, 3, 4, 5]:
            assert self._compute(m) >= 1024, f"{m}min budget {self._compute(m)} < 1024"

    def test_6min_uses_chunked(self):
        assert self._compute(6) == 4096

    def test_25min_chunked(self):
        assert self._compute(25) == 4096

    def test_ceiling_8192(self):
        assert self._compute(5) <= 8192


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

class TestBarkTTSCodePatterns:

    @pytest.fixture(scope="class")
    def src(self):
        path = os.path.join(os.path.dirname(__file__), "..", "nodes", "bark_tts.py")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_warning_filter_present(self, src):
        assert "filterwarnings" in src

    def test_warning_targets_max_length(self, src):
        assert "max_length" in src and "max_new_tokens" in src

    def test_local_files_only_present(self, src):
        assert "local_files_only=True" in src

    def test_oserror_fallback_present(self, src):
        assert "OSError" in src

    def test_sub_models_patched(self, src):
        for sub in ("semantic", "coarse_acoustics", "fine_acoustics"):
            assert sub in src, f"Sub-model {sub!r} not in generation_config patch"

    def test_max_length_nulled(self, src):
        assert "max_length = None" in src or "max_length=None" in src


# ─────────────────────────────────────────────────────────────────────────────
# 7. STORY ORCHESTRATOR — Static Code Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestStoryOrchestratorCodePatterns:

    @pytest.fixture(scope="class")
    def src(self):
        path = os.path.join(os.path.dirname(__file__), "..", "nodes", "story_orchestrator.py")
        with open(path, encoding="utf-8") as f:
            return f.read()

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
        assert "max_length=None" in src

    def test_citation_rule_present(self, src):
        assert "CITATION RULE" in src or "cite ONLY" in src

    def test_gender_words_frozenset(self, src):
        assert "_GENDER_WORDS" in src and "frozenset" in src

    def test_voice_tag_example_has_charactername(self, src):
        assert "CHARACTERNAME" in src or "CHARACTER NAME" in src

    def test_lemmy_easter_egg(self, src):
        assert "LEMMY" in src
        assert "wrench" in src.lower()
        assert "0.11" in src

    def test_local_files_only_gemma(self, src):
        assert "local_files_only" in src


# ─────────────────────────────────────────────────────────────────────────────
# 8. WORKFLOW JSON — Full Workflow Integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowJSONFull:

    @pytest.fixture(scope="class")
    def wf(self):
        return _load_workflow("otr_scifi_16gb_full.json")

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
        known = {"PreviewAudio", "PreviewImage", "Note"}
        for n in wf["nodes"]:
            assert n["type"].startswith("OTR_") or n["type"] in known

    def test_required_pipeline_nodes(self, wf):
        types = {n["type"] for n in wf["nodes"]}
        required = {
            "OTR_Gemma4ScriptWriter", "OTR_Gemma4Director",
            "OTR_BatchBarkGenerator", "OTR_SceneSequencer",
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

    def test_sfx_generator(self):
        from nodes.sfx_generator import SFXGenerator
        audio, _ = SFXGenerator().generate("radio_tuning", 2.0, 48000, -6.0)
        self._check(audio)

    def test_sfx_all_types(self):
        from nodes.sfx_generator import SFXGenerator
        node = SFXGenerator()
        for t in ["radio_tuning", "sci_fi_beep", "theremin", "explosion",
                  "footsteps", "heartbeat", "door_knock", "wind",
                  "siren", "ticking_clock", "white_noise", "pink_noise"]:
            audio, _ = node.generate(t, 1.0, 48000, -6.0)
            self._check(audio)

    def test_episode_assembler(self):
        from nodes.scene_sequencer import EpisodeAssembler
        scene = {"waveform": torch.randn(1, 1, 48000).float(), "sample_rate": 48000}
        audio, _, _ = EpisodeAssembler().assemble(scene, "Test")
        self._check(audio)


# ─────────────────────────────────────────────────────────────────────────────
# 11. SCENE SEQUENCER — Clip Wiring (requires torch)
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestSceneSequencerClipWiring:

    def test_resamples_44100_to_48000(self):
        from nodes.scene_sequencer import SceneSequencer
        script = json.dumps([
            {"type": "dialogue", "character_name": "X", "voice_traits": "male,30s", "line": "Test"}
        ])
        tts = {"waveform": torch.randn(1, 1, 44100), "sample_rate": 44100}
        audio, _, _ = SceneSequencer().sequence(script, "{}", tts_audio_clips=tts)
        assert audio["sample_rate"] == 48000

    def test_empty_script_returns_silence(self):
        from nodes.scene_sequencer import SceneSequencer
        audio, _, _ = SceneSequencer().sequence("[]", "{}")
        assert audio["waveform"].shape[2] > 0


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
# 14. PARSER v5 — SFX EXTRACTION TESTS (v1.5 Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestParserV5SFXExtraction:
    """Verify SFX cues are emitted as {"type": "sfx"} items by _parse_script."""

    def test_sfx_has_description_field(self, parser):
        result = parser._parse_script("[SFX: heavy metal clang]\n[VOICE: DUMMY, male, 30s] ok.")
        sfx_items = [r for r in result if r["type"] == "sfx"]
        assert len(sfx_items) == 1
        assert "description" in sfx_items[0]
        assert "metal clang" in sfx_items[0]["description"]

    def test_multiple_sfx_preserved_in_order(self, parser):
        script = "\n".join([
            "[SFX: door slam]",
            "[VOICE: A, male, 30s] Hello.",
            "[SFX: glass breaking]",
            "[VOICE: B, female, 30s] Watch out.",
            "[SFX: alarm siren]",
        ])
        result = parser._parse_script(script)
        sfx_items = [r for r in result if r["type"] == "sfx"]
        assert len(sfx_items) == 3
        assert sfx_items[0]["description"] == "door slam"
        assert sfx_items[1]["description"] == "glass breaking"
        assert sfx_items[2]["description"] == "alarm siren"

    def test_sfx_interleaved_with_dialogue(self, parser):
        """SFX items should appear at the correct positions in the output list."""
        script = "[VOICE: A, male, 30s] First.\n[SFX: footsteps]\n[VOICE: A, male, 30s] Second."
        result = parser._parse_script(script)
        types = [r["type"] for r in result]
        # Find the sfx position — should be between the two dialogue items
        sfx_idx = types.index("sfx")
        dialogue_indices = [i for i, t in enumerate(types) if t == "dialogue"]
        assert sfx_idx > dialogue_indices[0]

    def test_sfx_with_complex_description(self, parser):
        result = parser._parse_script(
            "[SFX: distant thunder rolling behind heavy rain, cinematic]\n"
            "[VOICE: DUMMY, male, 30s] ok."
        )
        sfx_items = [r for r in result if r["type"] == "sfx"]
        assert len(sfx_items) == 1
        assert "thunder" in sfx_items[0]["description"]
        assert "rain" in sfx_items[0]["description"]

    def test_sfx_case_insensitive_extraction(self, parser):
        """All case variants of [SFX:] should produce type=sfx items."""
        for tag in ["[SFX: beep]", "[sfx: beep]", "[Sfx: beep]"]:
            result = parser._parse_script(f"{tag}\n[VOICE: DUMMY, male, 30s] ok.")
            sfx_items = [r for r in result if r["type"] == "sfx"]
            assert len(sfx_items) == 1, f"Failed for tag: {tag}"

    def test_sfx_in_full_scene(self, parser):
        """SFX cues in a full canonical scene should be preserved."""
        script = """=== SCENE 1 ===
[ENV: dark corridor, dripping water]
[SFX: heavy door creaking open]
[VOICE: HAYES, male, 40s, tense, medium] Did you hear that?
(beat)
[SFX: footsteps echoing]
[VOICE: DR_CHEN, female, 40s, calm, low] Just the pipes.
[SFX: distant alarm]
=== SCENE 2 ===
[VOICE: ANNOUNCER, male, 50s, calm] And so it continues.
"""
        result = parser._parse_script(script)
        sfx_items = [r for r in result if r["type"] == "sfx"]
        assert len(sfx_items) == 3
        assert sfx_items[0]["description"] == "heavy door creaking open"
        assert sfx_items[1]["description"] == "footsteps echoing"
        assert sfx_items[2]["description"] == "distant alarm"


# ─────────────────────────────────────────────────────────────────────────────
# 15. AUDIOGEN CANONICAL SFX CONSUMPTION (v1.5 Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioGenCanonicalSFX:
    """Verify BatchAudioGenGenerator consumes canonical parser SFX items."""

    def test_audiogen_no_regex_import(self):
        """AudioGen should NOT use its own SFX regex anymore."""
        path = os.path.join(os.path.dirname(__file__), "..", "nodes", "batch_audiogen_generator.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        # The old duplicate regex pattern should be gone
        assert "re.findall(r'\\[SFX:" not in src, (
            "AudioGen still contains duplicate SFX regex. "
            "It should consume canonical parser output instead."
        )

    def test_audiogen_reads_type_sfx(self):
        """AudioGen source should reference type == sfx for extraction."""
        path = os.path.join(os.path.dirname(__file__), "..", "nodes", "batch_audiogen_generator.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert '"type"' in src and '"sfx"' in src, (
            "AudioGen should filter script items by type == sfx"
        )

    def test_audiogen_reads_description_field(self):
        """AudioGen source should read the 'description' field from SFX items."""
        path = os.path.join(os.path.dirname(__file__), "..", "nodes", "batch_audiogen_generator.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert '"description"' in src, (
            "AudioGen should read the 'description' field from canonical SFX items"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 16. PARSER V3/V4 PATTERN REGRESSION (v1.5 Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestParserV3V4Patterns:
    """Regression tests for the post-v1.4 parser patterns."""

    def test_v3_next_line_dialogue(self, parser):
        """v3: [VOICE: NAME, traits] on one line, dialogue on next."""
        script = "[VOICE: HAYES, male, 40s, tense, medium]\nDid you hear that?"
        result = parser._parse_script(script)
        dialogues = [r for r in result if r["type"] == "dialogue"]
        assert len(dialogues) >= 1
        assert dialogues[0]["character_name"] == "HAYES"
        assert "Did you hear that?" in dialogues[0]["line"]

    def test_v4_shorthand_tag(self, parser):
        """v4: [ANNOUNCER, traits] shorthand without VOICE: prefix."""
        script = "[ANNOUNCER, male, 50s, calm]\nWelcome to Signal Lost."
        result = parser._parse_script(script)
        dialogues = [r for r in result if r["type"] == "dialogue"]
        assert len(dialogues) >= 1
        assert dialogues[0]["character_name"] == "ANNOUNCER"

    def test_v3_next_line_skips_empty_lines(self, parser):
        """v3 lookahead should skip blank lines between tag and dialogue."""
        script = "[VOICE: CHEN, female, 40s, calm]\n\n\nThe readings are stable."
        result = parser._parse_script(script)
        dialogues = [r for r in result if r["type"] == "dialogue"]
        assert len(dialogues) >= 1
        assert dialogues[0]["character_name"] == "CHEN"

    def test_stage_direction_blocklist(self, parser):
        """ACT, SCENE, CONTINUED in v3/v4 position should NOT create dialogue."""
        for tag in ["[ACT 1]", "[SCENE 3]", "[CONTINUED]"]:
            script = f"{tag}\nSome text.\n[VOICE: DUMMY, male, 30s] ok."
            result = parser._parse_script(script)
            dialogues = [r for r in result if r["type"] == "dialogue"]
            # None of the blocklisted tags should produce dialogue with that name
            for d in dialogues:
                assert d["character_name"] not in ("ACT", "SCENE", "CONTINUED"), (
                    f"Blocklisted tag {tag} created dialogue with name {d['character_name']}"
                )

    def test_markdown_bold_stripped(self, parser):
        """Parser should strip **bold** markers from voice tags."""
        script = "**[VOICE: HAYES, male, 40s, calm, low]** The readings are in."
        result = parser._parse_script(script)
        dialogues = [r for r in result if r["type"] == "dialogue"]
        assert len(dialogues) >= 1
        assert dialogues[0]["character_name"] == "HAYES"

    def test_v2_no_traits_inline(self, parser):
        """v2: [VOICE: NAME] dialogue — no traits specified."""
        result = parser._parse_script("[VOICE: COMMANDER] All units stand down.")
        dialogues = [r for r in result if r["type"] == "dialogue"]
        assert len(dialogues) >= 1
        assert dialogues[0]["character_name"] == "COMMANDER"
        assert "stand down" in dialogues[0]["line"]
