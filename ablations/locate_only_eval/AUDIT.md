# Locate LOO post-hoc audit — cases to rerun after fixes
Two buckets per config:
- **A**: ``picked_source`` contains ``emergency_la_centroid`` (HTTP error fell back to LA centroid). Fix: HTTP retry + image downscale.
- **B**: pick is > 5.0 km from EVERY coord any tool returned. Fix: L2 cross-check validator.

| Config | A (HTTP) | B (sign-flip) | Total to rerun |
|---|---:|---:|---:|
| full | 0 | 0 | 0 |
| min_1_tool | 0 | 1 | 1 |
| no_grid_ref | 0 | 0 | 0 |
| no_intersect | 0 | 0 | 0 |
| no_la_check | 0 | 0 | 0 |
| no_place | 0 | 0 | 0 |
| no_postcode | 0 | 0 | 0 |
| no_road | 0 | 0 | 0 |
| **TOTAL** | **0** | **1** | **1** |
