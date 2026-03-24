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
