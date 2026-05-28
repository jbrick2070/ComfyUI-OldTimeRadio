"""Debug BUG-288 wiring test."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes._otr_stage1_plan import (
    Stage1Arc, Stage1Beat, Stage1CastMember, Stage1Plan,
)
from nodes._otr_stage3_validators import (
    CODE_SFW_VIOLATION, validate_line,
)

try:
    plan = Stage1Plan(
        premise="A two-hander about a publication decision.",
        arc=Stage1Arc(
            setup="The recording matches a living pod.",
            complication="Publication will draw boats.",
            resolution="Suri opts for sound-only release.",
        ),
        cast=[
            Stage1CastMember(
                name="REN BLACK",
                gender="male",
                pronouns="he/him",
                voice_id="v2/en_speaker_6",
                persona="Station operator. Clipped, wary.",
                arc_role="reluctant insider",
            ),
        ],
        beats=[
            Stage1Beat(
                beat_id="b002",
                speaker="REN BLACK",
                intent="Refuse the publication offer concretely.",
                length_target_words=20,
                emotional_register="guarded",
            ),
        ],
        running_facts=[],
    )
    result = validate_line(plan, plan.beats[0], "Damn it, I'm not signing.")
    print(f"findings: {[f.code for f in result.findings]}")
except Exception as e:
    traceback.print_exc()
