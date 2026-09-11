# Final code delta and grounded consensus

Review only these residual points after the supplied prior review. Root is sole
judge; this is a bounded repair continuation, not an architecture restart.

Initial full suite:14,399 passed /51 unchanged inherited failures /183 skips /
one xfail, versus14,379/51. Exact failure IDs and normalized payloads match.
Focused362pass; actual disk/wire/reference/credits checks2pass. Final delta below
is now being rerun through focused/full/Bible. Canonical validator/JSON/live
schema/link audit passes23nodes/63links/37writerwidgets; no graph/interface change.

Sonnet's actual completed review found no demonstrated must-fix but could not see
the unchanged CastLock fallback body. It is included below. The earlier large
Sonnet packet produced no review (empty content); that is retained as failure.

Opus returned a partial review, cut off during issue3c. Root disposition:

1. Accepted concrete age-join edge: a legacy row's truthy n/a/whitespace can mask
   an already-known treatment age. Normalize only blank/n/a before choosing, and
   expose absence when both sides unknown. Actual pairlock_02 ledger rows contain
   NO age_band field (production set_cast filters it), so the claim that n/a was
   the normal live row and caused the observed pixels is false. The metadata
   projection improvement remains useful; new controls cover both sides.
2. Accepted distinct fallback reason and stale comment cleanup. Blank source
   gender now stamps gender_unspecified; an unservable explicit gender still
   stamps gender_unservable. The actual source gender is unchanged. Rejected
   the claim that writing presentation_gender, route-tier and warning counts is
   a regression: the real reference was selected, its actual presentation must
   be reported, and this is exactly the missing truthful provenance being fixed.
   The existing fallback already participates in these stamps. Google exceptions
   remain necessary for all other unservable genders; no dead branch removal.
3. Rejected speculative gates: structured_call has six successful exits, all
   validate with the supplied post_validator, including its prebuilt typed repair.
   No success cache/zero-attempt/model_construct bypass exists. The captured model
   is the actual validated object, tested after mutations and failed attempts.
   Current aliased model CastMember has populate_by_name=True and alias tests pass.
   Other current models have no aliases. Unkeyed P2 dialogue is intentionally
   replaced as a list: both speaker/text are required, there are no omitted fields
   to inherit, and no stable identity to justify positional merging. P3 lists are
   strings. Requiring every schema/list to declare an identity or rejecting unused
   mappings would add broken gates to valid current callers; not implemented.

No new checker, model call, retry budget, body-length/timing/cast-count rejection,
prose regex or global structured_call API change. Full raw source remains the
authority; prompt-and-pixel qualification remains OPEN until fresh canonical.

Please return a complete <=600-word final verdict. Any remaining must-fix needs
an actual reachable current code path and reproducer; missing evidence is a
limitation, not a confirmed defect. If the bounded code is sound, say so.

## Final changed code and supporting owners


### nodes/otr_meta_brief_image_prompt.py
```python
1661: def _scene_source_context(meta, cast, lines, target, line, ledger_context=None):
1662:     """Exact raw source and a structural scene join, never a presence classifier."""
1663:     from ._otr_story_source import raw_fields_from_ledger
1664:     ledger = ledger_context or {"meta": meta, "cast": cast, "lines": lines or []}
1665:     raw = raw_fields_from_ledger(ledger)
1666:     if raw is None:
1667:         return None
1668:     bid = str(target.get("beat_id") or "")
1669:     beats = [row for row in ledger.get("beats", []) if isinstance(row, dict)]
1670:     beat = next((row for row in beats if str(row.get("beat_id") or "") == bid
1671:                  or bid in [str(value) for value in row.get("line_ids", [])]), {})
1672:     shot_id = str(beat.get("shot_id") or line.get("shot_id") or "")
1673:     shot = next((row for row in ledger.get("shots", [])
1674:                  if isinstance(row, dict) and str(row.get("shot_id") or "") == shot_id), {})
1675:     scene_id = str(beat.get("scene_id") or shot.get("scene_id") or line.get("scene_id") or "")
1676:     scene = next((row for row in ledger.get("scenes", [])
1677:                   if isinstance(row, dict) and str(row.get("scene_id") or "") == scene_id), {})
1678:     scene_line_ids = {
1679:         str(lid) for row in beats
1680:         if (scene_id and str(row.get("scene_id") or "") == scene_id)
1681:         or (not scene_id and shot_id and str(row.get("shot_id") or "") == shot_id)
1682:         for lid in row.get("line_ids", [])
1683:     }
1684:     semantic_keys = ("line_id", "beat_id", "shot_id", "scene_id", "char_id", "speaker",
1685:                      "speaker_role", "text", "beat_intent", "traits", "arc_phase")
1686: 
1687:     def scene_line(row):
1688:         return {key: row[key] for key in semantic_keys if key in row}
1689: 
1690:     ordered_lines = [scene_line(row) for row in (lines or []) if isinstance(row, dict)
1691:                      and (str(row.get("line_id") or "") in scene_line_ids
1692:                           or (shot_id and str(row.get("shot_id") or "") == shot_id))]
1693:     if not ordered_lines and line:
1694:         ordered_lines = [scene_line(line)]
1695:     speakers = {str(row.get("char_id") or "") for row in ordered_lines}
1696:     cid = str(target.get("char_id") or "")
1697:     treatment = (meta.get("my_story") or {}).get("treatment") or {}
1698:     planned_cast = [row for row in treatment.get("cast", []) if isinstance(row, dict)]
1699: 
1700:     def character_context(row):
1701:         # Ledger rows do not retain all treatment casting fields. Join only a
1702:         # unique exact normalized name; never infer an age or gender from prose.
1703:         name = str(row.get("name") or "")
1704:         key = " ".join(name.split()).casefold()
1705:         matches = [item for item in planned_cast
1706:                    if key and " ".join(str(item.get("name") or "").split()).casefold() == key]
1707:         planned = matches[0] if len(matches) == 1 else {}
1708:         ages = [str(value or "").strip() for value in
1709:                 (row.get("age_band"), planned.get("age_band"))]
1710:         age = next((value for value in ages if value.casefold() not in {"", "n/a"}), "")
1711:         return {"char_id": str(row.get("char_id") or ""), "name": name,
1712:                 "appearance": _appearance_for_char([row], str(row.get("char_id") or "")),
1713:                 "age_band": age,
1714:                 "gender": row.get("gender") or planned.get("gender") or ""}
1715: 
1716:     companions = [
1717:         {**character_context(row),
1718:          "speaks_in_scene": str(row.get("char_id") or "") in speakers}
1719:         for row in cast if isinstance(row, dict) and row.get("char_id")
1720:         and str(row.get("char_id")) != cid
1721:         and str(row.get("name") or "").strip().upper() != "ANNOUNCER"
1722:         and not row.get("_synthetic_announcer")
1723:     ]
1724:     context = {
1725:         "prompt_contract": "my_story.scene_source.v2",
1726:         "scope": "scene_character", "beat_id": bid, "target_char_id": cid,
1727:         "resolved_setting": _read_setting(meta),
1728:         "target_character": next((character_context(row)
1729:             for row in cast if isinstance(row, dict) and str(row.get("char_id") or "") == cid), {}),
1730:         "beat": dict(beat), "current_line": scene_line(line), "shot": dict(shot),
1731:         "scene": dict(scene), "ordered_scene_dialogue": ordered_lines,
1732:         "candidate_companions": companions,
1733:         "working_treatment": treatment,
1734:     }
1735:     # A primitive copy prevents later pipeline mutation from changing the receipt.
1736:     return json.loads(json.dumps({"raw_fields": raw, "scene": context}, ensure_ascii=False))
```

### nodes/cast_lock.py
```python
1628:     def _stamp(entry, ref, *, fallback: str = "") -> None:
1629:         """Stamp the chosen reference onto a cast entry (I-4 / I-9).
1630: 
1631:         ``fallback`` records HOW the reference was chosen. It is written on every
1632:         stamped row, empty string for the ordinary deterministic cast, so a
1633:         downstream reader never has to distinguish "cast normally" from "field
1634:         was never written".
1635:         """
1636:         entry["voice_ref_id"] = ref.voice_ref_id
1637:         entry["voice_engine"] = ref.engine
1638:         entry["commercial_clean"] = _delivered_commercial_clean(entry, ref)
1639:         entry["voice_cast_fallback"] = fallback
1640:         # presentation_gender (item 8 chunk 4, 2026-08-06): the gender the
1641:         # DELIVERED voice presents as, taken from the reference actually chosen
1642:         # rather than from the row's label. Stamped HERE because this is the one
1643:         # place every stamped row passes through -- characters, the announcer,
1644:         # the hybrid voice-fit branch and the gender-agnostic fallback alike.
1645:         #
1646:         # Two rows the label cannot answer for, and this is why the field exists:
1647:         # the ANNOUNCER's reference is drawn from the episode seed and never read
1648:         # its row's gender at all, and an `other` row is served by a draw the bank
1649:         # makes without regard to gender. In both cases the row said one thing and
1650:         # the audience heard another, with nothing in the ledger recording it.
1651:         # Whatever the bank's own vocabulary says wins -- including `neutral`,
1652:         # which is a real reference (el_river), not a bucket to round away.
1653:         entry["presentation_gender"] = str(getattr(ref, "gender", "") or "").strip().lower()
1654:         # C3 (cloud-audio 2026-07-03): carry the provider voice id for cloud
1655:         # (ElevenLabs) casting -- ONLY when present, so local (ref-clip/preset)
1656:         # cast entries stay byte-identical. The durable cast stamp copies the
1657:         # whole cast section (production_ledger.stamp_durable), so this survives
1658:         # to the admission gate + OTR_CreditsRoll.
1659:         pvid = getattr(ref, "provider_voice_id", "") or ""
1660:         if pvid:
1661:             entry["provider_voice_id"] = pvid
```

### nodes/cast_lock.py
```python
1170:             # VoiceCastingError here, was caught below, and took the
1171:             # gender-agnostic draw. (It also fed the hybrid voice-fit branch's
1172:             # validation until that branch was ripped on 2026-08-18; the scorer
1173:             # is now the only consumer.)
1174:             from ._otr_roster_gender import canonical_bank_gender
1175:             gender = canonical_bank_gender(entry.get("gender"))
1176:             if not gender and target_engine == "google_tts":
1177:                 raise VoiceCastingError(
1178:                     f"{char_id}: google_tts character casting needs a cast "
1179:                     f"gender to choose a gender-plausible provider voice. "
1180:                     f"NO FALLBACK.")
1181:             # THE HYBRID LLM VOICE-FIT BRANCH WAS HERE AND IS GONE (2026-08-18).
1182:             # It read meta.voice_cast_decision, re-validated the LLM's proposed
1183:             # voice_ref_id, and on success stamped it and `continue`d -- skipping
1184:             # the deterministic scorer below entirely. That is why the scorer
1185:             # handled only ~4% of production casting.
1186:             #
1187:             # `meta.voice_cast_decision` is still STAMPED (empty) by the writer
1188:             # and still verified downstream, so a legacy ledger carrying real
1189:             # decisions loads without complaint -- its proposals are simply
1190:             # ignored now, and the scorer casts the row. That is the intended
1191:             # behaviour, not a fallback: the LLM had no information the scorer
1192:             # lacks. CastLock itself no longer reads the key at all (the dead
1193:             # local above went 2026-08-28); an earlier version of this comment
1194:             # said it did.
1195: 
1196:             # Prefer the writer's voice-fit slot (timbre/age_band); fall back to
1197:             # any entry-level fields for legacy ledgers without the stamp.
1198:             slot = voice_slots.get(char_id) or {}
1199:             slot_timbre = slot.get("timbre") or entry.get("timbre") or ()
1200:             slot_age = str(slot.get("age_band") or entry.get("age_band") or "")
1201:             try:
1202:                 if not gender:
1203:                     # Cast the same real open-pool reference that the renderer
1204:                     # would select, so the wire, ledger and credits name it.
1205:                     # This does not invent a gender for the character.
1206:                     raise VoiceCastingError(f"{char_id}: source gender unspecified")
1207:                 ref = assign_voice_for_slot(
1208:                     role="char_voice",
1209:                     engine=target_engine,
1210:                     char_id=char_id,
1211:                     gender=gender,
1212:                     timbre=tuple(slot_timbre),
1213:                     age_band=slot_age,
1214:                     episode_seed=episode_seed,
1215:                     casting_policy_version=CASTING_POLICY_VERSION,
1216:                     allow_voice_reuse=allow_voice_reuse,
1217:                     used_voice_ref_ids=used,
1218:                     bank=bank_entries,
1219:                 )
1220:             except VoiceCastingError as exc:
1221:                 if target_engine == "google_tts":
1222:                     raise
1223:                 # The bank cannot serve this row's gender -- 'other' is 20% of
1224:                 # every roll and the bank carries zero rows for it. Previously
1225:                 # the row was reported "NOT cast" and left with NO voice_ref_id,
1226:                 # and the render path then drew a gender-agnostic reference of
1227:                 # its own. The ledger therefore did not name the voice that
1228:                 # actually spoke. Stamp the SAME draw the render will make, so
1229:                 # the ledger is complete and honest. This is a ledger fix, not a
1230:                 # content gate: no refusal, no gender restriction.
1231:                 fallback_ref = gender_agnostic_fallback_ref(
1232:                     bank_entries, engine=target_engine, char_id=char_id,
1233:                     episode_seed=episode_seed, role="char_voice", used=used,
1234:                 )
1235:                 if fallback_ref is None:
1236:                     report.append(f"  {char_id}: NOT cast -- {exc}")
1237:                     continue
1238:                 _stamp_row(entry, fallback_ref, fallback=(
1239:                     "gender_unservable" if gender else "gender_unspecified"))
1240:                 _mark_used(fallback_ref)
1241:                 gated += 0 if _delivered_commercial_clean(
1242:                     entry, fallback_ref) else 1
1243:                 reason = f"gender {gender!r} unservable" if gender else "source gender unspecified"
1244:                 report.append(
1245:                     f"  {char_id}: {fallback_ref.voice_ref_id} "
1246:                     f"({fallback_ref.engine}, {reason} -- "
1247:                     f"gender-agnostic reference)"
1248:                 )
1249:                 continue
1250:             _stamp_row(entry, ref)
1251:             _mark_used(ref)
1252:             drawn_clean = _delivered_commercial_clean(entry, ref)
1253:             gated += 0 if drawn_clean else 1
1254:             report.append(
1255:                 f"  {char_id}: {ref.voice_ref_id} ({ref.engine}, "
1256:                 f"clean={drawn_clean})"
1257:             )
1258: 
1259:         # LEDGER COMPLETENESS FOR THE TIER, and this sweep is why the three fields
1260:         # can be read as an enumeration downstream. The claimed row can leave the
1261:         # loop above by several doors -- the hybrid voice-fit, the gender-agnostic
1262:         # fallback, the ordinary draw -- and a field written at only some of them
1263:         # is worse than no field at all.
1264:         #
1265:         # IT REPORTS ONLY ON ROWS THIS LOCK ACTUALLY RE-CAST, and that condition
1266:         # is the whole correctness of the field. `unrouted` is the honest name for
1267:         # "the ordinary seeded draw chose this voice", which is what every
1268:         # unclaimed row in the tree takes -- but a row the caster never reached
1269:         # (no character engine in this bank, no available references)
1270:         # took no draw at all, and stamping `unrouted` on it would assert a
1271:         # decision that was never made. Such a row keeps exactly what it arrived
1272:         # with, in both modes, and the absence of the field says so.
1273:         #
1274:         # `unrouted` is not an error in production. It IS a sprint failure on an
1275:         # acceptance leg, which is a different question asked by a different
1276:         # reader.
1277:         if tier_character_key:
1278:             for entry in cast:
1279:                 if (not isinstance(entry, dict) or _is_announcer_entry(entry)
1280:                         or id(entry) not in stamped_this_lock
1281:                         or _ROUTE.CAST_ROW_TIER_FIELD in entry
1282:                         or not _ROUTE.cast_row_matches_policy(
1283:                             entry, tier_character_key)):
1284:                     continue
1285:                 _stamp_route_tier(entry, _ROUTE.ROUTE_TIER_UNROUTED,
1286:                                   route_id=provisional_route_id,
1287:                                   reason_code=provisional_reason)
1288:                 report.append(
1289:                     "  %s: no voice route applied -- ordinary draw (%s)"
1290:                     % (entry.get("char_id") or entry.get("name"),
1291:                        provisional_reason or "no reason recorded"))
1292: 
1293:         if gated:
1294:             report.append(
1295:                 f"auto_registry: {gated} assigned voice(s) are known-gated "
1296:                 f"(reference clip and/or model licence is not commercial-clean) "
1297:                 f"-- non-blocking warning (I-8)"
1298:             )
1299: 
1300: 
```

### nodes/_otr_structured_call.py
```python
675:     *,
676:     post_validator: Optional[Callable[[T], Optional[str]]] = None,
677: ) -> T:
678:     """Validate exact structured data, then run an optional structural check.
679: 
680:     The core never truncates or rewrites model-authored strings. Schemas retain
681:     only genuine machine, provenance, graph, cardinality, and nonempty
682:     constraints; validation errors advance the bounded structural repair ladder.
683:     """
684:     instance = schema.model_validate(data)
685:     if post_validator is not None:
686:         content_error = post_validator(instance)
687:         if content_error is not None:
688:             raise PostValidationError(content_error)
689:     return instance
690: 
691: 
692: def parse_validate_tolerant(
693:     raw: str,
694:     schema: type[T],
695:     *,
696:     post_validator: Optional[Callable[[T], Optional[str]]] = None,
697: ) -> T:
698:     """Extract the first JSON object and validate it without prose mutation."""
699:     data = _otr_json.parse_first_json_object(raw or "")
700:     return validate_tolerant_data(
701:         data,
702:         schema,
703:         post_validator=post_validator,
704:     )
705: 
706: 
707: def _raw_head(
708:     raw: "str | None", cap: int = 400, error: "BaseException | None" = None,
709: ) -> str:
710:     """Sanitized head of a failed model output for the ladder WARNING
```

### nodes/_otr_structured_call.py
```python
965:         )
966:         try:
967:             last_raw = ""
968:             last_raw = _invoke_slot(
969:                 slot_fn, base_messages,
970:                 temperature=base_temperature,
971:                 max_new_tokens=max_new_tokens,
972:                 force_json_object=text_parser is None,
973:             )
974:             result = _parse_and_validate(
975:                 last_raw,
976:                 schema,
977:                 post_validator,
978:                 text_parser,
979:             )
980:             notify_attempt(None)
981:             return result
982:         except _ATTEMPT_ERRORS as exc:
983:             if not _attempt_is_retryable(exc):
984:                 notify_attempt(exc)
```

### nodes/_otr_structured_call.py
```python
1020:         log.info(
1021:             "[OTR_StructuredCall] '%s' attempt %d/%d: structural retry at "
1022:             "temperature=%.3f (lowered from %.3f)",
1023:             helper_name, attempts_run, max_attempts,
1024:             structural_retry_temperature, base_temperature,
1025:         )
1026:         try:
1027:             last_raw = ""
1028:             last_raw = _invoke_slot(
1029:                 slot_fn, base_messages,
1030:                 temperature=structural_retry_temperature,
1031:                 max_new_tokens=max_new_tokens,
1032:                 force_json_object=text_parser is None,
1033:             )
1034:             result = _parse_and_validate(
1035:                 last_raw,
1036:                 schema,
1037:                 post_validator,
1038:                 text_parser,
1039:             )
1040:             notify_attempt(None)
1041:             return result
1042:         except _ATTEMPT_ERRORS as exc:
1043:             if not _attempt_is_retryable(exc):
1044:                 notify_attempt(exc)
1045:                 raise
```

### nodes/_otr_structured_call.py
```python
1078:             # a locked-cast member via Levenshtein -- and hand back a
1079:             # finished `schema` instance instead of a repair prompt.
1080:             # Accept it directly: no LLM repair call is made. The
1081:             # instance still passes through post_validator so a
1082:             # deterministic "fix" that is itself content-invalid fails
1083:             # the ladder loudly rather than slipping through.
1084:             if isinstance(repair_prompt, schema):
1085:                 if post_validator is not None:
1086:                     content_error = post_validator(repair_prompt)
1087:                     if content_error is not None:
1088:                         raise PostValidationError(content_error)
1089:                 log.info(
1090:                     "[OTR_StructuredCall] '%s' attempt %d: repair factory "
1091:                     "resolved the failure deterministically; no LLM "
1092:                     "repair call made",
1093:                     helper_name, attempts_run,
1094:                 )
1095:                 # The attempt COMPLETED here, so report it like every other
1096:                 # successful return does. This was the one exit of six that
1097:                 # skipped the hook: `attempts_run` had already been
1098:                 # incremented for this rung, so a caller counting attempts
1099:                 # under-reported by one whenever a typed repair factory
1100:                 # resolved the failure itself. Unreachable from callers that
1101:                 # pass no `deterministic_repair` (the story-brief reflection),
1102:                 # but a lane MAY pass one.
1103:                 # Mutation-checked 2026-08-09: deleting this call turns
1104:                 # test_deterministic_repair_return_still_reports_its_attempt
1105:                 # red, so the guard is real rather than decorative.
1106:                 notify_attempt(None)
1107:                 return repair_prompt
1108:             repair_prompt = _inherit_generation_contract(
1109:                 contract_prompt, repair_prompt,
1110:             )
1111:             repair_messages = _prompt_to_messages(
1112:                 repair_prompt if text_parser is not None
1113:                 else _prompt_with_schema_contract(repair_prompt, schema)
1114:             )
1115:             last_raw = ""
1116:             last_raw = _invoke_slot(
1117:                 slot_fn, repair_messages,
1118:                 temperature=_REPAIR_TEMPERATURE,
1119:                 max_new_tokens=max_new_tokens,
1120:                 force_json_object=text_parser is None,
1121:             )
1122:             result = _parse_and_validate(
1123:                 last_raw,
1124:                 schema,
1125:                 post_validator,
1126:                 text_parser,
1127:             )
1128:             notify_attempt(None)
1129:             return result
1130:         except _ATTEMPT_ERRORS as exc:
1131:             if not _attempt_is_retryable(exc):
1132:                 notify_attempt(exc)
```

### nodes/_otr_structured_call.py
```python
1170:         log.info(
1171:             "[OTR_StructuredCall] '%s' attempt %d/%d: typed repair syntax "
1172:             "retry at temperature=%.3f",
1173:             helper_name, attempts_run, max_attempts, retry_temperature,
1174:         )
1175:         try:
1176:             last_raw = ""
1177:             last_raw = _invoke_slot(
1178:                 slot_fn, repair_messages,
1179:                 temperature=retry_temperature,
1180:                 max_new_tokens=max_new_tokens,
1181:                 force_json_object=text_parser is None,
1182:             )
1183:             result = _parse_and_validate(
1184:                 last_raw,
1185:                 schema,
1186:                 post_validator,
1187:                 text_parser,
1188:             )
1189:             notify_attempt(None)
1190:             return result
1191:         except _ATTEMPT_ERRORS as exc:
1192:             if not _attempt_is_retryable(exc):
1193:                 notify_attempt(exc)
```

### nodes/_otr_structured_call.py
```python
1245:             "nonce=%s at temperature=%.3f",
1246:             helper_name, attempts_run, repair_nonce, _REPAIR_TEMPERATURE,
1247:         )
1248:         try:
1249:             last_raw = ""
1250:             last_raw = _invoke_slot(
1251:                 repair_slot_fn,
1252:                 repair_messages,
1253:                 temperature=_REPAIR_TEMPERATURE,
1254:                 max_new_tokens=max_new_tokens,
1255:                 force_json_object=text_parser is None,
1256:             )
1257:             result = _parse_and_validate(
1258:                 last_raw,
1259:                 schema,
1260:                 post_validator,
1261:                 text_parser,
1262:             )
1263:             notify_attempt(None)
1264:             return result
1265:         except _ATTEMPT_ERRORS as exc:
1266:             if not _attempt_is_retryable(exc):
1267:                 notify_attempt(exc)
```
