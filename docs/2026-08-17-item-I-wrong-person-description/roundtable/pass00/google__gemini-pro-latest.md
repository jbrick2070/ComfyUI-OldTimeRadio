<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

VERDICT: no. The proposed options all treat the symptom (two sets of names colliding) while ignoring that you control the upstream schema that forces the collision.

MUST-FIX BEFORE BUILD:
1. [Section 4 / 5.1] **The continuity mapping is a hallucination.** [ASSUMPTION] You assume `meta.continuity.facts` is a structural mapping.