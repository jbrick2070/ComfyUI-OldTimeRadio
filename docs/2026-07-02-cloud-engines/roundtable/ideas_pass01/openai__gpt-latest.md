<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

NAME: The Locked Playhouse
PITCH: The audience gasps because every scene, memory, and murder happens inside one persistent 1940s radio theater, with the camera gliding from announcer booth to stage flats to foley pit as if the whole episode is a haunted live broadcast.
CHAIN:
1. Cloud: Recraft / Flux2 Pro / Nano Banana 2 generate the period visual bible: playhouse floor plan, announcer booth, orchestra pit, stage flats, microphones, curtains, character wardrobe sheets.
2. Cloud: Tripo / Rodin / Meshy image-to-3D and text-to-3D build the recurring theater set, microphones, chairs, foley tables, doors, thunder sheets, miniature vehicles.
3. Cloud: Meshy rig/animate creates repeatable mannequin/actor blocking poses for recurring cast silhouettes.
4. Local compositing: assemble the playhouse once in Blender or a simple 2.5D layout, save fixed camera coordinates per show beat: booth, stage, balcony, pit, backstage.
5. Cloud: Wan I2V / Luma Ray / Vidu generate smoky black-and-white motion passes from locked camera plates: curtain ripples, light sweeps, slow dolly pushes, film-grain atmosphere.
6. Cloud: Seedance 2.0 reference-to-video audio-ref identity preservation generates close inserts for character_video beats when the same actor must “perform” at their marked spot.
7. Local compositing: cut to frozen master audio timing, add dust/sprocket overlays, mux the untouched master audio last.
WHY CLOUD-ONLY: The value is not one render; it is repeated 3D asset generation, rigged blocking, and video refinement across many beats, which is impractical on a 16GB GPU without reducing everything to static plates.
IDENTITY/CONTINUITY ANGLE: The playhouse becomes the continuity anchor: every character has a fixed booth/stage/pit coordinate, a recurring costume sheet, and repeatable camera lanes, so visual identity is reinforced spatially even when style shifts.
AUDIO-REACTIVE ANGLE: Dialogue amplitude drives spotlight intensity and camera push-ins; music sections trigger balcony/orchestra views; loud stingers fire practical effects like curtain snaps, lamp flicker, or foley-table close-ups.
RISK: The generated theater assets may not share one clean architectural logic unless the first visual bible is tightly curated.
COST SKETCH: High — multiple 3D generations plus many Wan/Luma/Vidu/Seedance clips per episode, but the set amortizes across future episodes.

NAME: The Shadow Cast
PITCH: The gasp comes when characters never fully appear as “video people” but as perfectly consistent noir shadows, reflections, and projected booth silhouettes that lip-sync and act from the walls of the radio studio.
CHAIN:
1. Cloud: Flux2 Pro / Nano Banana 2 generate canonical character portraits, wardrobe silhouettes, profile views, and 1940s studio lighting references.
2. Cloud: Recraft remove/replace background and vectorize convert each character into clean shadow masks, glass-reflection plates, and posterized noir profile assets.
3. Cloud: Seedance 2.0 reference-to-video with audio-ref identity preservation generates each character’s performance as a controlled talking-head or upper-body source clip from the canonical portrait.
4. Cloud: Kling avatar / lip-sync refines mouth motion for dialogue-heavy close shadow moments where syllable clarity matters.
5. Cloud: Wan I2V / Vidu / Luma Ray transform the source performances into wall shadows, frosted-glass booth reflections, cigarette-smoke projections, and rear-projection stage silhouettes.
6. Local compositing: place those shadows/reflections onto locked radio-studio plates, align to beat timing, keep the frozen master audio untouched, mux last.
WHY CLOUD-ONLY: Consistent reference-to-video identity plus lipsync plus multiple style conversions per character is exactly the sort of multi-model video workload that overwhelms local VRAM and time.
IDENTITY/CONTINUITY ANGLE: Identity is locked by silhouette geometry: hat brim, jawline, glasses, hair shape, shoulder line, and recurring reflection placement become more stable than full photoreal faces.
AUDIO-REACTIVE ANGLE: The frozen episode audio drives mouth motion, shadow scale, reflection shimmer, and spotlight jitter; whispers become small booth reflections, shouting throws huge distorted shadows across the set.
RISK: Style-transfering a performance into a shadow can blur facial identity if the silhouette bible is weak.
COST SKETCH: Medium-high — fewer full-scene generations than a cinematic episode, but every speaking character needs Seedance/Kling passes and stylized video conversions.

NAME: Foley Resurrection
PITCH: The audience gasps because every door slam, footstep, scream, gunshot, thunder roll, and kiss is “performed” on-screen by an impossible 1940s foley pit that moves in sync with the finished radio drama.
CHAIN:
1. Cloud: ElevenLabs voice isolation separates frozen master audio into usable dialogue/music/noise references for timing analysis, without replacing the master.
2. Cloud: Recraft / Flux2 Pro generate period foley-board designs: coconut-shell footsteps, thunder sheet, wind crank, door rig, glass tray, slapstick, rain box, pistol blank, creaking chair.
3. Cloud: Tripo / Meshy / Rodin image-to-3D build recurring foley props and a foley-pit environment.
4. Cloud: Meshy rig/animate creates repeatable mechanical motions for props: door swing, thunder-sheet shake, wind-crank turn, footsteps, bell strike.
5. Local compositing: use extracted audio peaks to trigger simple prop animation timing and camera cuts against the frozen master timeline.
6. Cloud: Wan I2V / Luma Ray / Vidu convert foley-pit plates into smoky, tactile 1940s newsreel-style video clips.
7. Cloud: Seedance 2.0 audio-ref on selected foley-table shots for heightened moments where performer hands should feel driven by the sound. [SPECULATIVE]
8. Local compositing: assemble the foley performance under the existing audio and mux the frozen master last.
WHY CLOUD-ONLY: The hard part is generating a library of coherent period props, animated mechanisms, and refined video inserts fast enough for every episode; local 16GB can keyframe triggers but cannot create and stylize the asset/video volume well.
IDENTITY/CONTINUITY ANGLE: The same foley pit, same props, same labeled drawers, and same hand silhouettes recur episode-to-episode, making sound effects visually recognizable characters in their own right.
AUDIO-REACTIVE ANGLE: Default-on: the master audio’s transients directly trigger cuts, prop hits, lamp pops, camera shakes, and foley close-ups; louder sounds get larger physical gestures.
RISK: Automatic matching of complex mixed audio to the correct prop may need manual rules or simple stem heuristics to avoid nonsense triggers.
COST SKETCH: Medium — initial 3D prop library is costly, but later episodes reuse the pit and spend credits mainly on stylized Wan/Luma/Vidu inserts.

NAME: Memory Backlot
PITCH: The gasp is that every location in the drama is revealed to be part of one endless miniature noir backlot, where a detective’s office, rain alley, hotel lobby, train platform, and villain’s mansion occupy a continuous impossible city.
CHAIN:
1. Cloud: Nano Banana 2 / Flux2 Pro / Recraft create the 1940s backlot map, recurring street signs, storefronts, hotel façade, alley, police station, mansion gate, radio tower, and train platform.
2. Cloud: LTX 2.3 outpaint / object-removal LoRAs expand plates into wide backlot panoramas and remove modern artifacts.
3. Cloud: Tripo / Rodin / Meshy image-to-3D generate façades, vehicles, lampposts, phone booths, fire escapes, signs, and set-dressing as reusable spatial assets.
4. Local compositing: assemble a lightweight persistent map with named coordinates for each beat location; render or export simple camera plates.
5. Cloud: Wan I2V / Luma Ray / Vidu animate the plates into rain, fog, passing train shadows, marquee flicker, police-light sweeps, and slow crane moves.
6. Cloud: Seedance 2.0 reference-to-video places recurring characters in windows, doorways, car interiors, or under streetlamps while preserving identity from portrait references.
7. Local compositing: edit to the locked beat list, keep the episode’s frozen master audio intact, mux last.
WHY CLOUD-ONLY: Persistent-world construction plus generative video for dozens of location variants is too asset-heavy and video-heavy for a 16GB local GPU; cloud nodes make the city expandable instead of hand-built.
IDENTITY/CONTINUITY ANGLE: Continuity is geographic: characters return to the same doorways, windows, cars, and street corners; even when the script jumps locations, the viewer feels one stable world underneath the anthology.
AUDIO-REACTIVE ANGLE: Music swells trigger crane moves over the map; dialogue proximity controls window light and camera distance; gunshots flash alley bulbs; suspense beds increase fog, rain, and neon pulse.
RISK: Different video nodes may reinterpret the same backlot plate unless strict reference images and locked camera plates are reused.
COST SKETCH: High at launch, medium later — city generation and first-pass video tests are expensive, but the backlot becomes a reusable episode engine.