# Locate LOO post-hoc audit — cases to rerun after fixes
Two buckets per config:
- **A**: ``picked_source`` contains ``emergency_la_centroid`` (HTTP error fell back to LA centroid). Fix: HTTP retry + image downscale.
- **B**: most recent la_check coord differs from final pick by >1.0 km. Fix: L2 cross-check validator.

| Config | A (HTTP) | B (sign-flip) | Total to rerun |
|---|---:|---:|---:|
| full | 12 | 5 | 17 |
| no_grid_ref | 12 | 9 | 21 |
| no_intersect | 14 | 7 | 21 |
| no_la_check | 11 | 0 | 11 |
| no_place | 5 | 10 | 15 |
| no_postcode | 8 | 8 | 16 |
| no_road | 7 | 10 | 17 |
| **TOTAL** | **69** | **49** | **118** |
