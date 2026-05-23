# Locate LOO post-hoc audit — cases to rerun after fixes
Two buckets per config:
- **A**: ``picked_source`` contains ``emergency_la_centroid`` (HTTP error fell back to LA centroid). Fix: HTTP retry + image downscale.
- **B**: pick is > 5.0 km from EVERY coord any tool returned. Fix: L2 cross-check validator.

| Config | A (HTTP) | B (sign-flip) | Total to rerun |
|---|---:|---:|---:|
| full | 12 | 1 | 13 |
| no_grid_ref | 12 | 2 | 14 |
| no_intersect | 14 | 1 | 15 |
| no_la_check | 11 | 0 | 11 |
| no_place | 5 | 1 | 6 |
| no_postcode | 8 | 1 | 9 |
| no_road | 7 | 2 | 9 |
| **TOTAL** | **69** | **8** | **77** |
