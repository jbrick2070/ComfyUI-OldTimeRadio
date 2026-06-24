<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

tail = f"on-stage, the decision about {obj} is made now, costing {fc.get('personal_cost', '')}"
            ```
            If we want to enforce the ending template, doing it in `_otr_story_quality_l12.py` is deterministic and avoids touching `_otr_outline.py`'s prompt builder.