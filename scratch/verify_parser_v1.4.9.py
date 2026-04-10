import os
import sys
import re

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.story_orchestrator import LLMScriptWriter

def test_hardened_parser():
    writer = LLMScriptWriter()
    
    # This matches the halluncination seen in the 19:43 logs
    bolded_script = """
=== SCENE 1 ===
[ENV: Deep Space]
**[VOICE: ANNOUNCER, male, 50s, calm]** This is a test.
**[VOICE: KANE, determined, calm]** **Why. I'm talking into a microphone.**
[SFX: beep]
**[VOICE: ANNOUNCER, male, 50s, calm]** End of test.
"""
    
    print("\n--- TEST: BOLDED SCRIPT PARSE (v1.4.9) ---")
    try:
        lines = writer._parse_script(bolded_script)
        dialogue_count = sum(1 for ln in lines if ln.get("type") == "dialogue")
        
        print(f"Parsed {len(lines)} total tokens.")
        print(f"Detected {dialogue_count} dialogue lines.")
        
        for i, ln in enumerate(lines):
            print(f"Line {i}: {ln['type']} | {ln.get('character_name', '')} | {ln.get('line', ln.get('description', ln.get('text', '')))}")

        if dialogue_count == 3:
            print("\n✅ SUCCESS: Parser successfully stripped markdown bolding and recovered all dialogue!")
        else:
            print(f"\n❌ FAILURE: Expected 3 dialogue lines, but found {dialogue_count}.")
            
    except Exception as e:
        print(f"\n❌ CRASH: Script parser failed: {e}")

if __name__ == "__main__":
    test_hardened_parser()
