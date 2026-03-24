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
