<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

choices` and `validate_model_id`.
    *   Let's look at `validate_model_id`.
        ```python
        # Path 1: curated
        if normalized in _by_repo_id():
            return normalized
        ```
        If we add `available=False`, we need to change `_by_repo_