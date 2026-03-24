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
            parts = list(loaded.geometry.values())
            combined = parts[0]
            for p in parts[1:]:
                combined = trimesh.util.concatenate([combined, p])
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
    face_ids = np.asarray(face_ids, dtype=np.int32)

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
