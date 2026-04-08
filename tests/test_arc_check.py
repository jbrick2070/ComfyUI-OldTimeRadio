"""Unit tests for Arc Enhancer Phase A structural coherence scoring."""
import sys
from pathlib import Path

# Import the Gemma4ScriptWriter to access the scorer
sys.path.insert(0, str(Path(__file__).parent.parent))
from nodes.gemma4_orchestrator import Gemma4ScriptWriter


class TestArcCoherence:
    """Test the _score_arc_coherence method (Phase A)."""

    def test_good_script_all_checks_pass(self):
        """Synthetic good script should score 5/5."""
        writer = Gemma4ScriptWriter()

        # Good script: complete, coherent, strong ending
        opening = """[VOICE: ANNOUNCER, female, 50s, authoritative] Deep within the void, signals echo.
[VOICE: CHEN, male, 40s, tense] The resonance chamber is failing.
[VOICE: VIKA, female, 30s, calm] We have four minutes to stabilize."""

        closing = """[VOICE: CHEN, male, 40s, determined] The resonance chamber held.
[VOICE: VIKA, female, 30s, hopeful] We made it.
[VOICE: VIKA, female, 30s, hopeful] And the chamber stayed locked.
[VOICE: ANNOUNCER, female, 50s, authoritative] And in the void, silence returned."""

        full_script = f"=== SCENE 1 ===\n{opening}\n=== EPILOGUE ===\n{closing}"

        score, checks = writer._score_arc_coherence(opening, closing, full_script)

        # All checks should pass
        assert score == 5, f"Expected score 5, got {score}. Checks: {checks}"
        assert checks['truncation'], "Should pass truncation check (ends with period)"
        assert checks['strong_scene'], "Should pass strong_scene check (3 voices)"
        assert checks['payoff'], "Should pass payoff check (RESONANCE shared)"
        assert checks['echo'], "Should pass echo check (resonance, chamber)"
        assert checks['epilogue'], "Should pass epilogue check (ANNOUNCER present)"

    def test_bad_script_multiple_failures(self):
        """Synthetic bad script should score low (≤2/5)."""
        writer = Gemma4ScriptWriter()

        # Bad script: truncated, weak ending, no epilogue
        opening = """[VOICE: CONTROL, female, 50s, authoritative] Into the depths we go.
[VOICE: REX, male, 40s, urgent] Pressure spiking.
[VOICE: NOVA, female, 30s, calm] Hold steady."""

        closing = """[VOICE: REX, male, 40s, exhausted] It's over.
[VOICE: NOVA, female, 30s, and"""  # Truncated mid-word

        # Make full_script long so closing is in final 500 chars but no ANNOUNCER there
        full_script = f"=== SCENE 1 ===\n{opening}\n" + ("padding\n" * 40) + f"=== SCENE 2 ===\n{closing}"

        score, checks = writer._score_arc_coherence(opening, closing, full_script)

        # Multiple checks should fail
        assert score <= 2, f"Expected score ≤2, got {score}. Checks: {checks}"
        assert not checks['truncation'], "Should fail truncation (ends mid-word)"
        assert not checks['epilogue'], "Should fail epilogue (no ANNOUNCER at end)"

    def test_weak_final_scene(self):
        """Script with only 1-2 dialogue lines in closing should fail strong_scene check."""
        writer = Gemma4ScriptWriter()

        opening = """[VOICE: ANNOUNCER, female, 50s] Beginning.
[VOICE: AGENT, male, 40s] Status check.
[VOICE: COMMAND, female, 30s] Ready."""

        closing = """[VOICE: AGENT, male, 40s] Mission complete."""  # Only 1 voice line

        full_script = f"=== SCENE 1 ===\n{opening}\n=== EPILOGUE ===\n{closing}"

        score, checks = writer._score_arc_coherence(opening, closing, full_script)

        assert not checks['strong_scene'], "Should fail strong_scene (only 1 dialogue line)"
        assert score < 5, f"Score should be <5 due to weak scene, got {score}"

    def test_no_shared_keywords(self):
        """Script with no keyword overlap should fail payoff check."""
        writer = Gemma4ScriptWriter()

        opening = """[VOICE: ANNOUNCER, female, 50s] The journey begins.
[VOICE: ALPHA, male, 40s] Let's go to Mars.
[VOICE: BETA, female, 30s] Understood."""

        closing = """[VOICE: GAMMA, male, 50s] We arrived at Venus.
[VOICE: DELTA, female, 40s] Different planet."""

        full_script = f"=== SCENE 1 ===\n{opening}\n=== EPILOGUE ===\n{closing}"

        score, checks = writer._score_arc_coherence(opening, closing, full_script)

        # No common capitalized words (Mars vs Venus)
        assert not checks['payoff'], "Should fail payoff (no keyword overlap)"

    def test_strong_echo_detection(self):
        """Script with repeated long words should pass echo check."""
        writer = Gemma4ScriptWriter()

        opening = """[VOICE: ANNOUNCER, female, 50s] The frequency keeps spiking.
[VOICE: TECH, male, 40s] frequency analysis shows anomalies.
[VOICE: DOCTOR, female, 30s] Continue monitoring."""

        closing = """[VOICE: TECH, male, 40s] The frequency stabilized.
[VOICE: DOCTOR, female, 30s] Anomalies disappeared.
[VOICE: ANNOUNCER, female, 50s] The void is silent."""

        full_script = f"=== SCENE 1 ===\n{opening}\n=== EPILOGUE ===\n{closing}"

        score, checks = writer._score_arc_coherence(opening, closing, full_script)

        assert checks['echo'], "Should pass echo (frequency, anomalies, monitoring)"

    def test_epilogue_detection_edge_case(self):
        """ANNOUNCER anywhere in final 500 chars should pass epilogue check."""
        writer = Gemma4ScriptWriter()

        opening = """[VOICE: ANNOUNCER, female, 50s] We begin.
[VOICE: CHAR, male, 40s] Let's go."""

        closing = """[VOICE: CHAR, male, 40s] It's done.
[VOICE: ANNOUNCER, female, 50s] And so ends our story."""

        # Make script long enough that closing is within final 500 chars
        full_script = f"=== SCENE 1 ===\n{opening}\n" + ("filler\n" * 50) + f"=== EPILOGUE ===\n{closing}"

        score, checks = writer._score_arc_coherence(opening, closing, full_script)

        assert checks['epilogue'], "Should pass epilogue (ANNOUNCER in final region)"


class TestPlotSpine:
    """Test the _extract_plot_spine method (Plot Spine Injection for Phase B)."""

    def test_extracts_middle_events(self):
        """Plot spine should capture middle-act dialogue and scene markers."""
        writer = Gemma4ScriptWriter()

        opening = "[VOICE: ANNOUNCER, female, 50s] The station is failing."
        closing = "[VOICE: CHEN, male, 40s] We survived."

        full_script = f"""=== SCENE 1 ===
{opening}
[VOICE: CHEN, male, 40s] Reactor check.
=== SCENE 2 ===
[VOICE: VIKA, female, 30s] Coolant system ruptured.
[VOICE: CHEN, male, 40s] Seal it off.
=== SCENE 3 ===
[VOICE: VIKA, female, 30s] Emergency protocol engaged.
{closing}"""

        spine = writer._extract_plot_spine(full_script, opening, closing)

        # Should include scene markers and middle dialogue
        assert "Scene 2" in spine, f"Missing Scene 2 marker: {spine}"
        assert "VIKA" in spine or "vika" in spine.lower(), f"Missing VIKA: {spine}"
        assert "coolant" in spine.lower(), f"Missing coolant detail: {spine}"

    def test_truncates_to_fifty_words(self):
        """Plot spine should truncate to ~50 words to stay under token budget."""
        writer = Gemma4ScriptWriter()

        opening = "[VOICE: ANNOUNCER, female, 50s] Start."
        closing = "[VOICE: CHEN, male, 40s] End."

        # Create very long middle content
        long_middle = "\n".join([
            f"[VOICE: CHAR{i}, male, 40s] This is dialogue line number {i} with many extra words to fill space."
            for i in range(20)
        ])
        full_script = f"=== SCENE 1 ===\n{opening}\n=== SCENE 2 ===\n{long_middle}\n=== SCENE 3 ===\n{closing}"

        spine = writer._extract_plot_spine(full_script, opening, closing)

        word_count = len(spine.split())
        assert word_count <= 55, f"Spine too long: {word_count} words"
        assert "..." in spine, f"Missing truncation indicator: {spine}"

    def test_handles_missing_middle(self):
        """Plot spine should gracefully handle empty or missing middle content."""
        writer = Gemma4ScriptWriter()

        opening = "[VOICE: ANNOUNCER, female, 50s] Start."
        closing = "[VOICE: CHEN, male, 40s] End."

        # No middle content
        full_script = f"{opening}\n{closing}"

        spine = writer._extract_plot_spine(full_script, opening, closing)

        # Should return fallback string, not crash
        assert spine, "Should return non-empty fallback"
        assert "unavailable" in spine.lower() or len(spine.split()) < 5


if __name__ == '__main__':
    test = TestArcCoherence()

    print("Running arc coherence tests...")
    test.test_good_script_all_checks_pass()
    print("✓ test_good_script_all_checks_pass")

    test.test_bad_script_multiple_failures()
    print("✓ test_bad_script_multiple_failures")

    test.test_weak_final_scene()
    print("✓ test_weak_final_scene")

    test.test_no_shared_keywords()
    print("✓ test_no_shared_keywords")

    test.test_strong_echo_detection()
    print("✓ test_strong_echo_detection")

    test.test_epilogue_detection_edge_case()
    print("✓ test_epilogue_detection_edge_case")

    print("\nRunning plot spine tests...")
    spine_test = TestPlotSpine()

    spine_test.test_extracts_middle_events()
    print("✓ test_extracts_middle_events")

    spine_test.test_truncates_to_fifty_words()
    print("✓ test_truncates_to_fifty_words")

    spine_test.test_handles_missing_middle()
    print("✓ test_handles_missing_middle")

    print("\nAll arc coherence + plot spine tests passed!")
