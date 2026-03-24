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
