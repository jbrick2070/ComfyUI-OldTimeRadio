<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

cmd(blender, "", tmp, 3, 256, 256, 0, mode="selftest")`
            So `--portrait` is empty. `--surface` is omitted, so it defaults to `flat`.
        *   In `main()`, `mode == "selftest"` creates an in-memory image and calls `_project_portrait_onto_meshes`.
        *   The plan says: `main()` render-mode state machine: "selftest mode: UNCHANGED (projects the in-memory portrait...)"
        *   So `main()` must explicitly check `if args.mode == "sel