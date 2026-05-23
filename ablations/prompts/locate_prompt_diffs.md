# Locate prompt variants — diff vs full

Each section lists lines present in the FULL prompt but NOT in the LOO variant. Use this to sanity-check that disabling a tool actually removes all references to it (tool description, signal-priority bullets, protocol-step references).

Full prompt: 3930 chars, 46 lines

## no_grid_ref

**Removed from full (4 lines):**
```
   - Clean single confident signal (SITE postcode, grid_ref, intersect) → σ=300-500m, 'high'
   - OS grid_ref (any precision)
- grid_ref(gr) — OS BNG grid reference → coord
You have 6 offline geocoder tools:
```

**Added (not in full, 2 lines):**
```
   - Clean single confident signal (SITE postcode, intersect) → σ=300-500m, 'high'
You have 5 offline geocoder tools:
```

## no_intersect

**Removed from full (3 lines):**
```
   - Clean single confident signal (SITE postcode, grid_ref, intersect) → σ=300-500m, 'high'
- intersect(road_a, road_b, la=None, road_c=None) — geometric junction of 2-3 roads
You have 6 offline geocoder tools:
```

**Added (not in full, 2 lines):**
```
   - Clean single confident signal (SITE postcode, grid_ref) → σ=300-500m, 'high'
You have 5 offline geocoder tools:
```

## no_la_check

**Removed from full (8 lines):**
```
   - LA centroid (last resort)
- la_check(lat, lon, la) — verify coord falls inside LA polygon
3. **LETTERHEAD CHECK postcodes:** for each postcode in pdf_info.postcodes, if it's NOT in site_address, treat as POSSIBLE letterhead. Run la_check to verify it's inside admin_region; if it falls outside admin_region, drop unless no other signal is available.
4. **BUILD POOL via tool calls.** Aim for 2-4 candidates from different signal types. Augment with terms FROM THE MAP IMAGE (don't limit yourself to pdf_info).
5. **CLUSTER & PICK:** 
6. **VALIDATE with la_check.** Final pick should be inside the admin_region polygon. Set verified_inside_admin_region=True if la_check confirms inside; leave at default False when admin_region is unknown or every candidate falls outside.
7. **Emit the LocatePick to terminate.** Once you have your pick, output the LocatePick directly as your final response — do NOT make further tool calls. Pydantic-ai parses your final structured output as the LocatePick schema. **Be meticulous and avoid clerical errors when submitting your final pick.** Copy the lat/lon EXACTLY from your strongest tool result — don't paraphrase, don't round prematurely. UK longitude ranges from about -8.6 (western Scotland) to +1.76 (eastern East Anglia). Around two-thirds of cases are NEGATIVE (most of England, Wales, Scotland) but Norfolk / Suffolk / Lincolnshire / Essex coast / east Kent are LEGITIMATELY POSITIVE (~+0.5 to +1.7) — don't force-flip a positive lon to negative if the tool result was positive. The bugs we see most often: (a) dropping a minus sign that should be there (e.g. -0.14 emitted as 0.14), (b) adding a minus sign that shouldn't be (e.g. +1.4 emitted as -1.4), (c) swapping top_lat and top_lon. Before emitting, verify the sign and order of the values against the tool result you're using. If the coord you're about to emit isn't close to a coord any of your tool calls returned, you've made an entry error — fix it.
You have 6 offline geocoder tools:
```

**Added (not in full, 4 lines):**
```
3. **BUILD POOL via tool calls.** Aim for 2-4 candidates from different signal types. Augment with terms FROM THE MAP IMAGE (don't limit yourself to pdf_info).
4. **CLUSTER & PICK:** 
5. **Emit the LocatePick to terminate.** Once you have your pick, output the LocatePick directly as your final response — do NOT make further tool calls. Pydantic-ai parses your final structured output as the LocatePick schema. **Be meticulous and avoid clerical errors when submitting your final pick.** Copy the lat/lon EXACTLY from your strongest tool result — don't paraphrase, don't round prematurely. UK longitude ranges from about -8.6 (western Scotland) to +1.76 (eastern East Anglia). Around two-thirds of cases are NEGATIVE (most of England, Wales, Scotland) but Norfolk / Suffolk / Lincolnshire / Essex coast / east Kent are LEGITIMATELY POSITIVE (~+0.5 to +1.7) — don't force-flip a positive lon to negative if the tool result was positive. The bugs we see most often: (a) dropping a minus sign that should be there (e.g. -0.14 emitted as 0.14), (b) adding a minus sign that shouldn't be (e.g. +1.4 emitted as -1.4), (c) swapping top_lat and top_lon. Before emitting, verify the sign and order of the values against the tool result you're using. If the coord you're about to emit isn't close to a coord any of your tool calls returned, you've made an entry error — fix it.
You have 5 offline geocoder tools:
```

## no_place

**Removed from full (4 lines):**
```
   - Named place / landmark from pdf_info OR from the map image
   - Parish name
- place(q, la=None) — OS Open Names search (villages, schools, churches, named buildings)
You have 6 offline geocoder tools:
```

**Added (not in full, 1 lines):**
```
You have 5 offline geocoder tools:
```

## no_postcode

**Removed from full (9 lines):**
```
   - Clean single confident signal (SITE postcode, grid_ref, intersect) → σ=300-500m, 'high'
   - Full postcode IN site_address (= SITE postcode, trust)
- postcode(pc) — UK postcode → coord (Code-Point Open, sub-100m)
3. **LETTERHEAD CHECK postcodes:** for each postcode in pdf_info.postcodes, if it's NOT in site_address, treat as POSSIBLE letterhead. Run la_check to verify it's inside admin_region; if it falls outside admin_region, drop unless no other signal is available.
4. **BUILD POOL via tool calls.** Aim for 2-4 candidates from different signal types. Augment with terms FROM THE MAP IMAGE (don't limit yourself to pdf_info).
5. **CLUSTER & PICK:** 
6. **VALIDATE with la_check.** Final pick should be inside the admin_region polygon. Set verified_inside_admin_region=True if la_check confirms inside; leave at default False when admin_region is unknown or every candidate falls outside.
7. **Emit the LocatePick to terminate.** Once you have your pick, output the LocatePick directly as your final response — do NOT make further tool calls. Pydantic-ai parses your final structured output as the LocatePick schema. **Be meticulous and avoid clerical errors when submitting your final pick.** Copy the lat/lon EXACTLY from your strongest tool result — don't paraphrase, don't round prematurely. UK longitude ranges from about -8.6 (western Scotland) to +1.76 (eastern East Anglia). Around two-thirds of cases are NEGATIVE (most of England, Wales, Scotland) but Norfolk / Suffolk / Lincolnshire / Essex coast / east Kent are LEGITIMATELY POSITIVE (~+0.5 to +1.7) — don't force-flip a positive lon to negative if the tool result was positive. The bugs we see most often: (a) dropping a minus sign that should be there (e.g. -0.14 emitted as 0.14), (b) adding a minus sign that shouldn't be (e.g. +1.4 emitted as -1.4), (c) swapping top_lat and top_lon. Before emitting, verify the sign and order of the values against the tool result you're using. If the coord you're about to emit isn't close to a coord any of your tool calls returned, you've made an entry error — fix it.
You have 6 offline geocoder tools:
```

**Added (not in full, 6 lines):**
```
   - Clean single confident signal (grid_ref, intersect) → σ=300-500m, 'high'
3. **BUILD POOL via tool calls.** Aim for 2-4 candidates from different signal types. Augment with terms FROM THE MAP IMAGE (don't limit yourself to pdf_info).
4. **CLUSTER & PICK:** 
5. **VALIDATE with la_check.** Final pick should be inside the admin_region polygon. Set verified_inside_admin_region=True if la_check confirms inside; leave at default False when admin_region is unknown or every candidate falls outside.
6. **Emit the LocatePick to terminate.** Once you have your pick, output the LocatePick directly as your final response — do NOT make further tool calls. Pydantic-ai parses your final structured output as the LocatePick schema. **Be meticulous and avoid clerical errors when submitting your final pick.** Copy the lat/lon EXACTLY from your strongest tool result — don't paraphrase, don't round prematurely. UK longitude ranges from about -8.6 (western Scotland) to +1.76 (eastern East Anglia). Around two-thirds of cases are NEGATIVE (most of England, Wales, Scotland) but Norfolk / Suffolk / Lincolnshire / Essex coast / east Kent are LEGITIMATELY POSITIVE (~+0.5 to +1.7) — don't force-flip a positive lon to negative if the tool result was positive. The bugs we see most often: (a) dropping a minus sign that should be there (e.g. -0.14 emitted as 0.14), (b) adding a minus sign that shouldn't be (e.g. +1.4 emitted as -1.4), (c) swapping top_lat and top_lon. Before emitting, verify the sign and order of the values against the tool result you're using. If the coord you're about to emit isn't close to a coord any of your tool calls returned, you've made an entry error — fix it.
You have 5 offline geocoder tools:
```

## no_road

**Removed from full (3 lines):**
```
   - Road name (when LA-filtered)
- road(q, la=None) — OML road centroid in LA bbox
You have 6 offline geocoder tools:
```

**Added (not in full, 1 lines):**
```
You have 5 offline geocoder tools:
```
