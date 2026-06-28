## story_quality_scan -- r3-pre (15 legs)

| metric | value | v1 target |
|---|---|---|
| length_ratio mean | 0.9464 | >= 0.85 |
| length_pass_fired | 0/15 | <= 2/12 |
| episode_valid | 15/15 | >= 11/12 |
| outro_hedge_vs_resolved | 0/15 | 0/12 |
| narration_self_address | 0 | 0 |
| arc_shapes seen | betrayal, heist, investigation_without_answer, setup_complication_resolution, slow_dread | not single-valued |

| leg | ratio | valid | hedge | narr | arc_shape | ds_source |
|---|---|---|---|---|---|---|
| signal_lost_bar_chip_ultimatum_20260627_010854_ledger.json | 0.8595 | True | False | 0 | setup_complication_resolution | llm |
| signal_lost_brass_button_decision_20260627_070317_ledger.json | 0.5619 | True | False | 0 | slow_dread | llm |
| signal_lost_brass_hinge_warning_20260627_074738_ledger.json | 0.5667 | True | False | 0 | investigation_without_answer | llm |
| signal_lost_compass_points_true_20260627_012504_ledger.json | 0.5667 | True | False | 0 | investigation_without_answer | llm |
| signal_lost_dialing_shadows_20260627_000451_ledger.json | 2.9 | True | False | 0 | slow_dread | llm |
| signal_lost_frostbite_facility_20260627_080010_ledger.json | 0.55 | True | False | 0 | setup_complication_resolution | fallback |
| signal_lost_heatwave_decryption_20260627_005327_ledger.json | 0.6548 | True | False | 0 | investigation_without_answer | llm |
| signal_lost_links_ascent_lesson_20260627_060216_ledger.json | 0.9119 | True | False | 0 | setup_complication_resolution | llm |
| signal_lost_marked_for_erasure_20260627_013919_ledger.json | 0.6 | True | False | 0 | investigation_without_answer | llm |
| signal_lost_marks_keep_climbing_20260627_073214_ledger.json | 0.9048 | True | False | 0 | betrayal | llm |
| signal_lost_power_play_20260627_063044_ledger.json | 0.6143 | True | False | 0 | heist | llm |
| signal_lost_seal_of_the_compound_20260627_061804_ledger.json | 0.6119 | True | False | 0 | setup_complication_resolution | llm |
| signal_lost_shadows_of_the_past_20260627_093012_ledger.json | 2.6333 | True | False | 0 | betrayal | llm |
| signal_lost_shredded_hope_20260627_071727_ledger.json | 0.4262 | True | False | 0 | slow_dread | llm |
| signal_lost_spindle_turns_again_20260627_064616_ledger.json | 0.8333 | True | False | 0 | investigation_without_answer | llm |

