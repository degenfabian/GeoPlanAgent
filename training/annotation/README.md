# training/annotation/

Interactive UIs for hand-building the SAM3 + rotation training set.
Three pieces feed into `training/dataset/`:

| Script | Produces | Notes |
|---|---|---|
| `boundary_prerender.py` | `boundary_annotations/<case>/{map.png, initial.json}` | Renders each eval PDF at production DPI; seeds an initial polygon from the cached agent affine when one is available, otherwise centred-and-scaled. Must run before `boundary_annotator.py`. |
| `boundary_annotator.py` + `boundary_annotator_ui.html` | `boundary_annotations/<case>/{edited.json, edited_mask.png}` | Flask + canvas UI. Trace / correct the boundary polygon. State persists per-case; resumable. |
| `rotation_annotator.py` | `training/dataset/rotation_annotations.json` | Flask + tiny UI. Click `0` / `90` / `180` / `270` for the corrective rotation needed to make each `map.png` upright. |

These are dev tools — not part of inference. None of them call
external APIs; all I/O is local files + Flask localhost.

## Workflow

```bash
# One-time, takes a few minutes:
uv run python training/annotation/boundary_prerender.py
# → boundary_annotations/<case>/map.png + initial.json for every eval case

# Multi-session, can quit and resume:
uv run python training/annotation/boundary_annotator.py
# → http://localhost:5000/
# Trace boundaries until every case has edited_mask.png

uv run python training/annotation/rotation_annotator.py
# → http://localhost:5000/   (different process, same port — kill the boundary
#                              annotator first if both are wanted)
# Click rotations until every case has a label

# Then assemble the SAM3 training set:
uv run python training/build_sam3_training_set.py
# → training/dataset/{maps, boundary_masks, fold_assignment.json}
```

## boundary_annotator UI keyboard shortcuts

(from the HTML)

| Key | Action |
|---|---|
| `→ / N` | Save + next case |
| `← / P` | Save + previous case |
| `R` | Reset to the seeded initial polygon |
| `+ / -` | Add / remove a ring from the polygon |
| Click+drag | Move a vertex; click a midpoint to insert a vertex |
| Shift+click | Delete a vertex |
| `S` | Skip without saving (no edited_mask.png written) |

## rotation_annotator UI keyboard shortcuts

| Key | Action |
|---|---|
| `0` / `1` / `2` / `3` | Save rotation 0° / 90° / 180° / 270° + next |
| `S` | Skip (no annotation saved) |
| `← / Backspace` | Previous case |
| `→ / Space` | Next case without annotating |

## Why annotate in the raw-PDF frame (no auto-rotation)?

`boundary_prerender.py` deliberately renders without invoking the
rotation classifier. The reason: if annotation happened on the
auto-rotated frame, the training mask would be coupled to the
rotation classifier's confidence — when the classifier is wrong,
the mask aligns to nothing. By annotating in the raw frame and
applying rotation as a downstream training-time augmentation, the
mask stays valid regardless of which rotation the classifier picks.

## Why annotate in image-pixel space (not WGS84)?

Both polygons (`initial.json`, `edited.json`) and the raster
`edited_mask.png` live in image-pixel coordinates of the rendered
PNG. They are NOT geographic. The agent pipeline projects the SAM3
mask back to WGS84 at inference time using the affine recovered
from MINIMA — so the training mask needs to be in the same frame
as the SAM3 input (the image), not the eventual output (WGS84).
