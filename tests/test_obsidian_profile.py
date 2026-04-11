
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.story_orchestrator import LLMScriptWriter

def test_obsidian_profile_overrides_flags():
    """Verify that 'Obsidian' profile forces multi-pass flags to False."""
    writer = LLMScriptWriter()
    
    # Mock _generate_chunked and other internal calls to avoid real LLM load
    with patch.object(LLMScriptWriter, '_generate_chunked', return_value="[VOICE: ANNOUNCER] Test"):
        with patch('nodes.story_orchestrator._generate_with_llm', return_value="[VOICE: ANNOUNCER] Test"):
            with patch.object(LLMScriptWriter, '_parse_script', return_value=[]):
                with patch('nodes.story_orchestrator._unload_llm'):
                    
                    # Run with Obsidian profile
                    # Even if we pass True for flags, the profile should override them
                    writer.write_script(
                        episode_title="Test",
                        genre_flavor="scifi",
                        runtime_preset="🧪 test (1 min)",
                        target_minutes=1,
                        num_characters=2,
                        self_critique=True,  # Should be overridden
                        open_close=True,     # Should be overridden
                        arc_enhancer=True,   # Should be overridden
                        optimization_profile="Obsidian (Low VRAM/Fast)"
                    )
                    
                    # We can't easily check internal local variables of write_script,
                    # but we can check if the methods were NOT called.
                    # Wait, write_script calls self._critique_and_revise if self_critique is True.
                    
    # Let's use a more robust check by mocking the sub-methods
    with patch.object(LLMScriptWriter, '_critique_and_revise') as mock_critique:
        with patch.object(LLMScriptWriter, '_open_close_expansion') as mock_oc:
            with patch.object(LLMScriptWriter, '_execute_arc_enhancer') as mock_arc:
                with patch.object(LLMScriptWriter, '_generate_chunked', return_value="[VOICE: ANNOUNCER] Test"):
                    with patch('nodes.story_orchestrator._generate_with_llm', return_value="[VOICE: ANNOUNCER] Test"):
                        with patch.object(LLMScriptWriter, '_parse_script', return_value=[]):
                            with patch('nodes.story_orchestrator._unload_llm'):
                                
                                writer.write_script(
                                    episode_title="Test",
                                    genre_flavor="scifi",
                                    runtime_preset="🧪 test (1 min)",
                                    target_minutes=1,
                                    num_characters=2,
                                    self_critique=True,
                                    open_close=True,
                                    arc_enhancer=True,
                                    optimization_profile="Obsidian (Low VRAM/Fast)"
                                )
                                
                                # Verification: methods should NOT have been called
                                mock_critique.assert_not_called()
                                mock_oc.assert_not_called()
                                mock_arc.assert_not_called()

def test_pro_profile_respects_flags():
    """Verify that 'Pro' profile respects the widget flags (defaults to True)."""
    writer = LLMScriptWriter()
    
    with patch.object(LLMScriptWriter, '_critique_and_revise', return_value="critiqued") as mock_critique:
        with patch.object(LLMScriptWriter, '_execute_arc_enhancer', return_value="enhanced") as mock_arc:
            with patch('nodes.story_orchestrator._generate_with_llm', return_value="[VOICE: ANNOUNCER] Test"):
                with patch.object(LLMScriptWriter, '_parse_script', return_value=[]):
                    with patch('nodes.story_orchestrator._unload_llm'):
                        
                        writer.write_script(
                            episode_title="Test",
                            genre_flavor="scifi",
                            runtime_preset="🧪 test (1 min)",
                            target_minutes=1,
                            target_length="short (3 acts)",
                            num_characters=2,
                            self_critique=True,
                            open_close=False, # We'll skip OC for this test to keep it simple
                            arc_enhancer=True,
                            optimization_profile="Pro (Ultra Quality)"
                        )
                        
                        # Verification: methods SHOULD have been called
                        mock_critique.assert_called_once()
                        mock_arc.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
