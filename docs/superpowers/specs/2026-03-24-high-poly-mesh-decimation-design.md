# High-Poly Mesh Decimation Pipeline for P3-SAM

**Date:** 2026-03-24
**Status:** Draft

## Problem

P3-SAM cannot practically process meshes with ~1M faces. The bottlenecks are:

1. `face_adjacency` + `build_adjacent_faces_numba()` — O(n) memory and compute, explodes at 1M faces
2. `fix_label()` — iterates up to 50x over all unlabeled faces using adjacency
3. `point_num` is hardcoded to 100,000 in `mesh_sam()` (line 779), so the model never actually "sees" more than 100k sampled points regardless of face count

Extra faces beyond ~100k are invisible to the model and only add processing overhead.

**Known limitation:** `mesh_sam()` line 779 hardcodes `point_num = 100000` and `prompt_num = 400`, overriding API parameters. Fixing this hardcode is out of scope for this spec but is noted as a future improvement.

## Solution

**Decimate -> Segment -> Label Transfer.** Automatically reduce high-poly meshes before segmentation, then transfer labels back to the original mesh.

## Performance target

A 1M-face mesh that currently fails or takes >10 minutes should complete in <2 minutes with decimation to 100k faces, with segmentation quality comparable to running on a natively low-poly mesh.

## Approach

**PyMeshLab quadric edge collapse decimation**, chosen over:
- trimesh built-in simplification (basic, no face mapping, can produce degenerate faces)
- Open3D (heavy ~200MB dependency, not used elsewhere in this repo)

PyMeshLab is already used by XPart (`requirements.txt`), and conversion utilities (`trimesh2pymeshlab`, `pymeshlab2trimesh`) exist in `XPart/partgen/utils/mesh_utils.py`.

## Architecture

### New module: `P3-SAM/demo/mesh_decimation.py`

Two public functions. The module includes its own copies of `trimesh2pymeshlab` / `pymeshlab2trimesh` conversion helpers (~25 lines each) rather than importing from XPart, to avoid fragile `sys.path` dependencies. Temp files created during conversion are cleaned up in `finally` blocks.

#### `decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh`

1. Convert trimesh to PyMeshLab MeshSet
2. Run `meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)`
3. Convert back to trimesh
4. Returns `decimated_mesh`

#### `transfer_labels(original_mesh: trimesh.Trimesh, decimated_mesh: trimesh.Trimesh, face_ids: np.ndarray) -> np.ndarray`

1. Compute face centroids for both meshes internally
2. Build KD-tree from decimated mesh face centroids
3. For each original face centroid, query nearest decimated face and copy its label
4. Handle sentinel values: faces mapped to a decimated face with label -1 or -2 retain that sentinel value — they are not smoothed in the boundary pass
5. Boundary smoothing pass: for faces where any neighbor has a different **non-negative** label, majority-vote from non-negative neighbor labels only. Uses scipy `cKDTree` with a small radius query on original mesh centroids rather than computing full `face_adjacency`, avoiding the O(n) adjacency bottleneck that motivated this entire spec. Single pass only.
6. Returns `face_ids` array sized to original mesh face count

### Integration: `P3-SAM/demo/p3sam_api.py`

#### New API parameter

```python
max_faces: int = Form(100000, ge=10000, le=1000000,
                      description="Max face count; meshes above this are auto-decimated before segmentation")
```

#### Modified flow in `segment_3d()`

```
1. Load mesh (existing)
2. if mesh.faces.shape[0] > max_faces:
     decimated_mesh = decimate_mesh(mesh, max_faces)
     logger.info(f"Decimated: {mesh.faces.shape[0]} -> {decimated_mesh.faces.shape[0]} faces")

     # IMPORTANT: clean_mesh_flag=False when decimation is applied.
     # Cleaning (merge_vertices + process) can change face count/order,
     # which would break the centroid-based label transfer mapping.
     aabb, face_ids_dec, _ = predict_aabb(decimated_mesh, ..., clean_mesh_flag=False)

     face_ids = transfer_labels(mesh, decimated_mesh, face_ids_dec)
     logger.info(f"Labels transferred back to original mesh ({mesh.faces.shape[0]} faces)")

     # Export uses the ORIGINAL mesh (not final_mesh from predict_aabb).
     # final_mesh is the decimated mesh — we discard it for export.
     # face_ids is now indexed to original mesh, so submesh() works directly.
     export_mesh = mesh
   else:
     aabb, face_ids, final_mesh = predict_aabb(mesh, ...)  # unchanged
     export_mesh = final_mesh
3. Export GLB (existing logic, but uses export_mesh instead of final_mesh)
```

**Key detail on export mesh identity:** When decimation is applied, the export block (lines 477-501 of current `p3sam_api.py`) must operate on the **original `mesh`** object, not on `final_mesh` returned by `predict_aabb`. The variable `export_mesh` makes this explicit. In the textured path, `export_mesh.submesh([face_mask], append=True)` preserves UVs because `export_mesh` is the original mesh with intact UV data. In the non-textured path, `export_mesh.copy()` gets face colors applied from the transferred `face_ids`.

#### Response schema additions

Three new fields on `SegmentResponse`:

| Field | Type | Description |
|-------|------|-------------|
| `decimation_applied` | `bool` | Whether the mesh was decimated before segmentation |
| `original_faces` | `int \| None` | Original face count (null if no decimation) |
| `decimated_faces` | `int \| None` | Face count after decimation (null if no decimation) |

### Texture preservation

When decimation is applied to a textured mesh:
- `clean_mesh_flag` is forced to `False` (decimation already implies the mesh was rebuilt)
- Decimation destroys UVs on the decimated copy — expected and harmless for inference
- Export uses the **original mesh** with transferred labels — UVs intact
- `export_mesh.submesh()` works directly since `face_ids` is indexed to original mesh

### Logging

Using the existing `logger` (TaipeiFormatter / p3sam logger), log:
- Original face count and decimated face count
- Decimation duration
- Label transfer duration
- Boundary smoothing stats (how many faces were re-labeled)

### Dependencies

**`P3-SAM/requirements-api.txt`:** Add `pymeshlab==2023.12.post3`

**`P3-SAM/Dockerfile`:** No changes needed — pymeshlab is a self-contained wheel.

**`docker-compose.yml` / `.env`:** No changes.

## Error handling

| Case | Behavior |
|------|----------|
| Decimation returns fewer faces than requested | Proceed with whatever PyMeshLab returns |
| Decimation fails (non-manifold, degenerate) | Log warning, fall back to running P3-SAM on original mesh |
| Textured mesh + decimation | Inference on decimated copy (clean_mesh_flag=False), export on original (UVs preserved) |
| `max_faces` >= actual face count | Skip decimation, no-op |
| Very low `max_faces` (e.g., 10000) | Caller's choice, may degrade quality — no guard needed |
| Sentinel values (-1, -2) in label transfer | Preserved as-is; boundary smoothing only votes among non-negative labels |

No new HTTP error codes. Decimation failures are transparent fallbacks logged server-side.
