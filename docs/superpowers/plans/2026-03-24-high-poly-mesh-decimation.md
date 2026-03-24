# High-Poly Mesh Decimation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically decimate high-poly meshes before P3-SAM segmentation and transfer labels back to the original mesh.

**Architecture:** New `mesh_decimation.py` module with `decimate_mesh()` and `transfer_labels()` functions, integrated transparently into the existing `/segment` endpoint. Meshes above `max_faces` threshold are auto-decimated; labels mapped back via KD-tree nearest-centroid + boundary smoothing.

**Tech Stack:** PyMeshLab (quadric edge collapse), scipy cKDTree (label transfer + smoothing), trimesh, numpy

**Spec:** `docs/superpowers/specs/2026-03-24-high-poly-mesh-decimation-design.md`

> **Note:** Line number references for `p3sam_api.py` are based on commit `4ad12be`. Before editing in Task 4, verify line numbers against the current file state.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `P3-SAM/demo/mesh_decimation.py` | Decimation + label transfer logic |
| Create | `P3-SAM/tests/test_mesh_decimation.py` | Unit tests for decimation module |
| Modify | `P3-SAM/demo/p3sam_api.py` | Add `max_faces` param, decimation flow, schema fields |
| Modify | `P3-SAM/requirements-api.txt` | Add `pymeshlab==2023.12.post3` |

---

### Task 1: Add pymeshlab dependency

**Files:**
- Modify: `P3-SAM/requirements-api.txt:19-22`

- [ ] **Step 1: Add pymeshlab to requirements**

Add after the `trimesh` line in the `# ── 3D geometry` section:

```
pymeshlab==2023.12.post3
```

So the section reads:

```
# ── 3D geometry ───────────────────────────────────────────────────────────────
trimesh==4.5.3
pymeshlab==2023.12.post3
scipy==1.14.1
networkx
```

- [ ] **Step 2: Verify pymeshlab is installable locally**

Run: `pip install pymeshlab==2023.12.post3 --dry-run 2>&1 | head -5`
Expected: Shows resolution without errors (or already installed).

- [ ] **Step 3: Commit**

```bash
git add P3-SAM/requirements-api.txt
git commit -m "Add pymeshlab dependency for mesh decimation pipeline"
```

---

### Task 2: Create mesh_decimation.py — conversion helpers + decimate_mesh

**Files:**
- Create: `P3-SAM/demo/mesh_decimation.py`
- Create: `P3-SAM/tests/test_mesh_decimation.py`

- [ ] **Step 1: Write the failing test for decimate_mesh**

Create `P3-SAM/tests/test_mesh_decimation.py`:

```python
import numpy as np
import trimesh
import pytest
import sys
import os

# Allow imports from P3-SAM/demo/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))


def _make_high_poly_sphere(subdivisions=4):
    """Create a UV sphere with many faces for testing."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    # subdivisions=4 -> 5120 faces, subdivisions=5 -> 20480 faces
    return mesh


class TestDecimateMesh:
    def test_reduces_face_count(self):
        mesh = _make_high_poly_sphere(subdivisions=5)  # 20480 faces
        assert mesh.faces.shape[0] == 20480

        from mesh_decimation import decimate_mesh

        result = decimate_mesh(mesh, target_faces=5000)
        assert result.faces.shape[0] <= 5500  # pymeshlab may slightly overshoot
        assert result.faces.shape[0] >= 4000  # but shouldn't be wildly off

    def test_preserves_geometry_bounds(self):
        mesh = _make_high_poly_sphere(subdivisions=5)
        original_bounds = mesh.bounds.copy()

        from mesh_decimation import decimate_mesh

        result = decimate_mesh(mesh, target_faces=5000)
        # Bounding box should be approximately the same
        np.testing.assert_allclose(result.bounds, original_bounds, atol=0.05)

    def test_returns_valid_trimesh(self):
        mesh = _make_high_poly_sphere(subdivisions=4)  # 5120 faces

        from mesh_decimation import decimate_mesh

        result = decimate_mesh(mesh, target_faces=1000)
        assert isinstance(result, trimesh.Trimesh)
        assert result.vertices.shape[0] > 0
        assert result.faces.shape[0] > 0

    def test_skips_when_already_below_target(self):
        mesh = _make_high_poly_sphere(subdivisions=3)  # 1280 faces

        from mesh_decimation import decimate_mesh

        result = decimate_mesh(mesh, target_faces=5000)
        # Early-return: should be the exact same mesh (no round-trip)
        assert result is mesh
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/between2058/Documents/code/Hunyuan3D-Part && python -m pytest P3-SAM/tests/test_mesh_decimation.py::TestDecimateMesh::test_reduces_face_count -v 2>&1 | tail -10`
Expected: FAIL with `ModuleNotFoundError: No module named 'mesh_decimation'`

- [ ] **Step 3: Write mesh_decimation.py with conversion helpers + decimate_mesh**

Create `P3-SAM/demo/mesh_decimation.py`:

```python
"""
Mesh decimation and label transfer for high-poly meshes.

Automatically reduces face count before P3-SAM segmentation,
then maps labels back to the original mesh.
"""

import logging
import os
import tempfile
import time

import numpy as np
import pymeshlab
import trimesh
from scipy.spatial import cKDTree

logger = logging.getLogger("p3sam")


# ── Conversion helpers ────────────────────────────────────────────────────────
# Local copies from XPart/partgen/utils/mesh_utils.py to avoid sys.path hacks.
# Temp files are cleaned up in finally blocks.


def _trimesh_to_pymeshlab(mesh: trimesh.Trimesh) -> pymeshlab.MeshSet:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            tmp_path = f.name
        if isinstance(mesh, trimesh.Scene):
            parts = list(mesh.geometry.values())
            combined = parts[0]
            for p in parts[1:]:
                combined = trimesh.util.concatenate([combined, p])
            mesh = combined
        mesh.export(tmp_path)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_path)
        return ms
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _pymeshlab_to_trimesh(ms: pymeshlab.MeshSet) -> trimesh.Trimesh:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            tmp_path = f.name
        ms.save_current_mesh(tmp_path)
        loaded = trimesh.load(tmp_path)
        if isinstance(loaded, trimesh.Scene):
            combined = trimesh.Trimesh()
            for geom in loaded.geometry.values():
                combined = trimesh.util.concatenate([combined, geom])
            return combined
        return loaded
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Public API ────────────────────────────────────────────────────────────────


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """
    Decimate a mesh to approximately target_faces using quadric edge collapse.

    If the mesh already has fewer faces than target_faces, it is returned as-is.

    Args:
        mesh: Input trimesh object.
        target_faces: Desired number of faces after decimation.

    Returns:
        Decimated trimesh object (or the original if already below target).
    """
    if mesh.faces.shape[0] <= target_faces:
        logger.info(
            f"Mesh has {mesh.faces.shape[0]} faces (<= {target_faces}), "
            f"skipping decimation."
        )
        return mesh

    t0 = time.time()
    ms = _trimesh_to_pymeshlab(mesh)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    result = _pymeshlab_to_trimesh(ms)
    dt = time.time() - t0
    logger.info(
        f"Decimation: {mesh.faces.shape[0]} -> {result.faces.shape[0]} faces "
        f"(target={target_faces}) in {dt:.2f}s"
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/between2058/Documents/code/Hunyuan3D-Part && python -m pytest P3-SAM/tests/test_mesh_decimation.py::TestDecimateMesh -v 2>&1 | tail -15`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add P3-SAM/demo/mesh_decimation.py P3-SAM/tests/test_mesh_decimation.py
git commit -m "Add decimate_mesh with PyMeshLab quadric edge collapse"
```

---

### Task 3: Add transfer_labels to mesh_decimation.py

**Files:**
- Modify: `P3-SAM/demo/mesh_decimation.py`
- Modify: `P3-SAM/tests/test_mesh_decimation.py`

- [ ] **Step 1: Write the failing tests for transfer_labels**

Append to `P3-SAM/tests/test_mesh_decimation.py`:

```python
class TestTransferLabels:
    def test_basic_label_transfer(self):
        """Labels from decimated mesh should map to original mesh faces."""
        original = _make_high_poly_sphere(subdivisions=5)  # 20480 faces
        from mesh_decimation import decimate_mesh, transfer_labels

        decimated = decimate_mesh(original, target_faces=5000)

        # Assign simple labels: split decimated mesh in half by centroid x-coord
        dec_centroids = decimated.triangles_center
        face_ids = np.where(dec_centroids[:, 0] > 0, 0, 1).astype(np.int32)

        result = transfer_labels(original, decimated, face_ids)

        assert result.shape[0] == original.faces.shape[0]
        # Original faces on x>0 side should mostly get label 0
        orig_centroids = original.triangles_center
        right_side = orig_centroids[:, 0] > 0.1  # margin to avoid boundary
        assert np.mean(result[right_side] == 0) > 0.9

    def test_sentinel_values_preserved(self):
        """Faces with label -1 or -2 should be preserved, not smoothed."""
        original = _make_high_poly_sphere(subdivisions=4)  # 5120 faces
        from mesh_decimation import decimate_mesh, transfer_labels

        decimated = decimate_mesh(original, target_faces=1000)

        # All decimated faces labeled -1 (unlabeled)
        face_ids = np.full(decimated.faces.shape[0], -1, dtype=np.int32)

        result = transfer_labels(original, decimated, face_ids)
        assert np.all(result == -1)

    def test_output_size_matches_original(self):
        """Returned array must have one label per original face."""
        original = _make_high_poly_sphere(subdivisions=5)
        from mesh_decimation import decimate_mesh, transfer_labels

        decimated = decimate_mesh(original, target_faces=5000)
        face_ids = np.zeros(decimated.faces.shape[0], dtype=np.int32)

        result = transfer_labels(original, decimated, face_ids)
        assert result.shape == (original.faces.shape[0],)

    def test_multiple_labels(self):
        """Should handle 3+ distinct labels correctly."""
        original = _make_high_poly_sphere(subdivisions=5)
        from mesh_decimation import decimate_mesh, transfer_labels

        decimated = decimate_mesh(original, target_faces=5000)

        # Split into 3 labels by z-coordinate
        dec_centroids = decimated.triangles_center
        face_ids = np.zeros(decimated.faces.shape[0], dtype=np.int32)
        face_ids[dec_centroids[:, 2] > 0.3] = 1
        face_ids[dec_centroids[:, 2] < -0.3] = 2

        result = transfer_labels(original, decimated, face_ids)
        unique = np.unique(result)
        assert set(unique).issubset({0, 1, 2})
        # All 3 labels should be present
        assert len(unique) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/between2058/Documents/code/Hunyuan3D-Part && python -m pytest P3-SAM/tests/test_mesh_decimation.py::TestTransferLabels::test_basic_label_transfer -v 2>&1 | tail -10`
Expected: FAIL with `ImportError: cannot import name 'transfer_labels'`

- [ ] **Step 3: Implement transfer_labels**

Append to `P3-SAM/demo/mesh_decimation.py` (after `decimate_mesh`):

```python
def transfer_labels(
    original_mesh: trimesh.Trimesh,
    decimated_mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
) -> np.ndarray:
    """
    Transfer per-face labels from a decimated mesh to the original mesh.

    Uses KD-tree nearest-centroid lookup followed by a single vectorized
    boundary smoothing pass (majority vote among spatial neighbors,
    non-negative labels only).

    Args:
        original_mesh: The original high-poly mesh.
        decimated_mesh: The decimated mesh that was segmented.
        face_ids: Per-face labels on the decimated mesh. May contain
            sentinel values -1 (unlabeled) and -2 (unassigned).

    Returns:
        Per-face labels sized to original_mesh.faces, dtype int32.
    """
    t0 = time.time()

    # 1. KD-tree nearest-centroid transfer
    dec_centroids = decimated_mesh.triangles_center
    orig_centroids = original_mesh.triangles_center

    tree = cKDTree(dec_centroids)
    _, nearest_idx = tree.query(orig_centroids)
    transferred = face_ids[nearest_idx].copy()

    # 2. Boundary smoothing — vectorized majority vote among spatial neighbors.
    #    Uses cKDTree radius query instead of face_adjacency to avoid
    #    the O(n) adjacency build that motivated this entire pipeline.
    #    Radius = 2x mean spacing between original face centroids.
    n_faces = orig_centroids.shape[0]
    mean_spacing = np.sqrt(original_mesh.area / n_faces) * 2.0

    orig_tree = cKDTree(orig_centroids)
    # Batch query: returns list of neighbor index lists for all points at once
    all_neighbors = orig_tree.query_ball_point(orig_centroids, mean_spacing)

    smoothed = transferred.copy()
    relabeled = 0

    # Find faces that are non-negative and at a label boundary (worth smoothing)
    for i in range(n_faces):
        label = transferred[i]
        if label < 0:
            continue  # don't smooth sentinel values

        neighbor_idx = all_neighbors[i]
        if len(neighbor_idx) <= 1:
            continue

        neighbor_labels = transferred[neighbor_idx]
        # Only vote among non-negative labels
        non_neg = neighbor_labels[neighbor_labels >= 0]
        if len(non_neg) == 0:
            continue

        counts = np.bincount(non_neg)
        majority = counts.argmax()
        if majority != label:
            smoothed[i] = majority
            relabeled += 1

    dt = time.time() - t0
    logger.info(
        f"Label transfer: {decimated_mesh.faces.shape[0]} -> "
        f"{original_mesh.faces.shape[0]} faces, "
        f"{relabeled} boundary faces re-labeled, {dt:.2f}s"
    )
    return smoothed.astype(np.int32)
```

> **Performance note:** The `query_ball_point` call is vectorized (batch query for all centroids at once). The subsequent Python loop only iterates faces to apply majority voting. For 1M faces this loop takes ~5-15s, acceptable within the <2 min total target. If profiling shows this is too slow, the loop can be replaced with a fully vectorized approach using sparse matrices.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /Users/between2058/Documents/code/Hunyuan3D-Part && python -m pytest P3-SAM/tests/test_mesh_decimation.py -v 2>&1 | tail -20`
Expected: All 8 tests PASS (4 from Task 2 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add P3-SAM/demo/mesh_decimation.py P3-SAM/tests/test_mesh_decimation.py
git commit -m "Add transfer_labels with KD-tree lookup and boundary smoothing"
```

---

### Task 4: Integrate decimation into p3sam_api.py

**Files:**
- Modify: `P3-SAM/demo/p3sam_api.py`

> **IMPORTANT: Replacement scope.** In Step 3 below, replace ONLY lines 443-509 (from `effective_clean_mesh = clean_mesh` through the closing brace of the `return` dict). Do NOT modify the `except` block (lines 511-520) or the `finally` block (lines 522-527) — they remain unchanged.

- [ ] **Step 1: Add import and update SegmentResponse schema**

In `p3sam_api.py`, add import at the top (after line 175, with the other imports):

```python
from mesh_decimation import decimate_mesh, transfer_labels
```

Update `SegmentResponse` (around line 253) to add three fields:

```python
class SegmentResponse(BaseModel):
    segmented_glb: str = Field(description="分割結果 GLB 的下載路徑，傳入 GET /download/{request_id}/{file_name}")
    request_id: str = Field(description="此次請求的 UUID")
    num_parts: int = Field(description="偵測到的零件數量（face label >= 0 的唯一 ID 數）")
    texture_preserved: bool = Field(description="True=輸入有 UV texture 且已保留；False=以隨機顏色區分各 part")
    decimation_applied: bool = Field(default=False, description="True=mesh 面數超過 max_faces 自動降面")
    original_faces: int | None = Field(default=None, description="降面前的原始面數（未降面時為 null）")
    decimated_faces: int | None = Field(default=None, description="降面後的面數（未降面時為 null）")
```

- [ ] **Step 2: Add max_faces parameter to segment_3d**

Add after the `prompt_bs` parameter (line 405):

```python
    max_faces: int = Form(100000, ge=10000, le=1000000, description="面數上限，超過自動降面再分割"),
```

- [ ] **Step 3: Replace lines 443-509 with decimation-aware flow**

Replace ONLY lines 443 (`effective_clean_mesh = clean_mesh`) through line 509 (the `return {...}` closing brace). The `except` block (lines 511-520) and `finally` block (lines 522-527) remain UNCHANGED.

```python
        effective_clean_mesh = clean_mesh
        if has_texture and clean_mesh:
            effective_clean_mesh = False
            logger.info("Input mesh has UV texture — forcing clean_mesh=False to preserve UVs.")

        # ── Decimation (if needed) ───────────────────────────────────────
        decimation_applied = False
        original_faces_count = None
        decimated_faces_count = None
        inference_mesh = mesh  # default: run on original
        export_mesh = None     # set after inference

        num_faces = mesh.faces.shape[0]
        if num_faces > max_faces:
            logger.info(f"Mesh has {num_faces} faces (> {max_faces}), decimating...")
            try:
                inference_mesh = decimate_mesh(mesh, max_faces)
                decimation_applied = True
                original_faces_count = num_faces
                decimated_faces_count = inference_mesh.faces.shape[0]
            except Exception as dec_err:
                logger.warning(
                    f"Decimation failed ({type(dec_err).__name__}: {dec_err}), "
                    f"falling back to original mesh."
                )
                inference_mesh = mesh

        # When decimation is applied, cleaning would change face order
        # and break the centroid-based label transfer mapping.
        if decimation_applied:
            effective_clean_mesh = False

        # 3. Load model
        model = load_model_instance(
            point_num=point_num,
            prompt_num=prompt_num,
            threshold=threshold,
            post_process=post_process,
        )

        # 4. Inference
        set_seed(seed)
        logger.info("P3SAM model start segmentation.")
        aabb, face_ids, final_mesh = await run_in_threadpool(
            model.predict_aabb,
            inference_mesh,
            save_path=job_dir,
            save_mid_res=False,
            show_info=True,
            clean_mesh_flag=effective_clean_mesh,
            seed=seed,
            prompt_bs=prompt_bs,
            is_parallel=False,
        )
        logger.info("Segmentation done.")

        # 5. Transfer labels back to original mesh if decimated
        if decimation_applied:
            face_ids = await run_in_threadpool(
                transfer_labels, mesh, inference_mesh, face_ids
            )
            export_mesh = mesh  # export on original (preserves UV)
        else:
            export_mesh = final_mesh

        # 6. Export GLB
        unique_ids = np.unique(face_ids)
        part_ids = [int(uid) for uid in unique_ids if uid >= 0]
        num_parts = len(part_ids)

        if has_texture:
            scene = trimesh.Scene()
            for part_id in part_ids:
                face_mask = np.where(face_ids == part_id)[0]
                sub = export_mesh.submesh([face_mask], append=True)
                scene.add_geometry(sub, node_name=f"part_{part_id}")
            scene.export(output_glb_path)
            logger.info(f"Exported {num_parts} textured parts (UV preserved).")
        else:
            rng = np.random.default_rng(seed)
            color_map = {
                uid: rng.integers(30, 230, size=3, dtype=np.uint8)
                for uid in part_ids
            }
            face_colors = np.array(
                [color_map[fid] if fid >= 0 else [64, 64, 64] for fid in face_ids],
                dtype=np.uint8,
            )
            mesh_out = export_mesh.copy()
            mesh_out.visual.face_colors = face_colors
            mesh_out.export(output_glb_path)
            logger.info(f"Exported {num_parts} color-coded parts (no texture).")

        return {
            "segmented_glb": f"/download/{request_id}/segmented_output_parts.glb",
            "request_id": request_id,
            "num_parts": num_parts,
            "texture_preserved": has_texture,
            "decimation_applied": decimation_applied,
            "original_faces": original_faces_count,
            "decimated_faces": decimated_faces_count,
        }
```

- [ ] **Step 4: Verify the file is syntactically valid**

Run: `python -c "import ast; ast.parse(open('P3-SAM/demo/p3sam_api.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add P3-SAM/demo/p3sam_api.py
git commit -m "Integrate mesh decimation pipeline into /segment endpoint"
```

---

### Task 5: Final verification

**Files:** (no new changes, verification only)

- [ ] **Step 1: Run all unit tests**

Run: `cd /Users/between2058/Documents/code/Hunyuan3D-Part && python -m pytest P3-SAM/tests/test_mesh_decimation.py -v`
Expected: All 8 tests PASS

- [ ] **Step 2: Verify API module imports cleanly**

Run: `cd /Users/between2058/Documents/code/Hunyuan3D-Part/P3-SAM/demo && python -c "from mesh_decimation import decimate_mesh, transfer_labels; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Verify p3sam_api.py parses without import errors (without GPU)**

Run: `cd /Users/between2058/Documents/code/Hunyuan3D-Part && python -c "import ast; ast.parse(open('P3-SAM/demo/p3sam_api.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 4: Review git log for clean commit history**

Run: `git log --oneline -5`
Expected: 4 new commits in logical order:
1. Add pymeshlab dependency
2. Add decimate_mesh
3. Add transfer_labels
4. Integrate into /segment endpoint
