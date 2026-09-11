<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

NO NEW MUST-FIX. 

The consolidated R4 proposal correctly utilizes the existing `source_rewrite_instruction` parameter in `_call` without introducing duplicate keyword arguments or Python syntax errors. The validator message aggregation in `_make_treatment_validator` and the scope-bounded preservation language in `_full_artifact_repair` are safely implemented within the existing control flow. The plan strictly adheres to the constraints (no new schemas, gates, or retry loops) and relies on the established `rewrite_story_source` mechanism. Proceed to implementation and live09 qualification.