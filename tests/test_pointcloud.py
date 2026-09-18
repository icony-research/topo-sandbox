"""高さの計算と点群への変換の検証。GPU は不要。"""

import numpy as np
import pytest

from topo_sandbox.processing.plane import fit_plane
from topo_sandbox.processing.pointcloud import heights_from_depth, heights_to_points


class TestHeightsFromDepth:
    def test_基準面が無ければ中央値を基準にする(self):
        depth = np.array([[1000, 1100, 1200]], dtype=np.float32)
        heights = heights_from_depth(depth)
        assert heights[0, 1] == pytest.approx(0.0)
        assert heights[0, 0] == pytest.approx(100.0)
        assert heights[0, 2] == pytest.approx(-100.0)

    def test_手前ほど高くなる(self):
        depth = np.array([[1000, 1100]], dtype=np.float32)
        heights = heights_from_depth(depth)
        assert heights[0, 0] > heights[0, 1]

    def test_基準面があれば傾きが打ち消される(self):
        height, width = 60, 80
        v, u = np.meshgrid(
            np.linspace(0.0, 1.0, height),
            np.linspace(0.0, 1.0, width),
            indexing="ij",
        )
        tilted = (1000.0 + 80.0 * u - 40.0 * v).astype(np.float32)

        reference = fit_plane(tilted)
        heights = heights_from_depth(tilted, reference)
        assert np.abs(heights).max() < 0.5

    def test_基準面が無いと傾きが残る(self):
        """補正の効果を対照で示す。"""
        height, width = 60, 80
        v, u = np.meshgrid(
            np.linspace(0.0, 1.0, height),
            np.linspace(0.0, 1.0, width),
            indexing="ij",
        )
        tilted = (1000.0 + 80.0 * u).astype(np.float32)

        heights = heights_from_depth(tilted)
        assert np.abs(heights).max() > 30.0


class TestHeightsToPoints:
    def test_点数は画素数と一致する(self):
        heights = np.zeros((24, 32), dtype=np.float32)
        points, _ = heights_to_points(heights, z_scale=1.0)
        assert points.shape == (24 * 32, 3)

    def test_中心は画像の中央(self):
        heights = np.zeros((24, 32), dtype=np.float32)
        _, center = heights_to_points(heights, z_scale=1.0)
        assert center == [16, 12]

    def test_1mmの差が1単位になる(self):
        """z_scale=1 のとき 1mm の高低差が高さ 1 に対応する。"""
        heights = np.array([[0.0, 1.0]], dtype=np.float32)
        points, _ = heights_to_points(heights, z_scale=1.0)
        assert abs((points[1][2] - points[0][2]) - 1.0) < 1e-9

    def test_zスケールが高さに掛かる(self):
        heights = np.array([[0.0, 100.0]], dtype=np.float32)
        low, _ = heights_to_points(heights, z_scale=0.1)
        high, _ = heights_to_points(heights, z_scale=1.0)
        assert (high[1][2] - high[0][2]) == (low[1][2] - low[0][2]) * 10

    def test_XY座標は格子状に並ぶ(self):
        heights = np.zeros((2, 3), dtype=np.float32)
        points, _ = heights_to_points(heights, z_scale=1.0)
        np.testing.assert_array_equal(points[:, 0], [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(points[:, 1], [0, 0, 0, 1, 1, 1])

    def test_異なる高さは異なる点になる(self):
        heights = np.array([[0.0, 256.0]], dtype=np.float32)
        points, _ = heights_to_points(heights, z_scale=1.0)
        assert points[0][2] != points[1][2]
