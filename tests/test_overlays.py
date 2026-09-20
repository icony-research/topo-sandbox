"""DEM・地形（水面と標高帯）・陰影起伏・エッジ・等高線の検証。

いずれも numpy と OpenCV だけで完結するため、Kinect も GPU も要らない。
"""

import cv2
import numpy as np
import pytest

from topo_sandbox import config
from topo_sandbox.processing import overlays


def _gradient(height=60, width=80):
    """上から下へ滑らかに明るくなる深度画像。"""
    column = np.linspace(0, 255, height, dtype=np.uint8)
    return np.repeat(column[:, None], width, axis=1)


class TestDemColor:
    def test_出力は入力と同じ大きさのRGB(self):
        depth = _gradient()
        result = overlays.dem_color(depth)
        assert result.shape == (depth.shape[0], depth.shape[1], 3)
        assert result.dtype == np.uint8

    def test_標高で色が変わる(self):
        depth = _gradient()
        result = overlays.dem_color(depth)
        assert not np.array_equal(result[0, 0], result[-1, 0])

    def test_一様な深度でも落ちない(self):
        """砂場が平らな瞬間や、センサが覆われた場合に起こりうる。

        最大と最小が同じだとゼロ除算になるため、明示的に扱っている。
        """
        depth = np.full((10, 10), 128, dtype=np.uint8)
        result = overlays.dem_color(depth)
        assert result.shape == (10, 10, 3)


class TestEdgeLines:
    def test_出力はRGB(self):
        coloring = np.zeros((20, 30, 3), dtype=np.uint8)
        result = overlays.edge_lines(coloring)
        assert result.shape == (20, 30, 3)

    def test_該当する色相だけが抽出される(self):
        coloring = np.zeros((10, 10, 3), dtype=np.uint8)
        result = overlays.edge_lines(coloring)
        # 真っ黒（彩度・明度 0）はエッジ帯の条件を満たさない
        assert result.max() == 0


class TestDrawContours:
    def test_等高線が描き込まれる(self):
        depth = _gradient()
        canvas = np.full((depth.shape[0], depth.shape[1], 3), 255, dtype=np.uint8)
        result = overlays.draw_contours(depth, canvas.copy())
        assert (result != 255).any()

    @pytest.mark.parametrize("value", [0, 255])
    def test_一様な深度でも落ちない(self, value):
        depth = np.full((20, 20), value, dtype=np.uint8)
        canvas = np.zeros((20, 20, 3), dtype=np.uint8)
        overlays.draw_contours(depth, canvas)


def _dome(size=60, peak_mm=100.0, spread=15.0):
    """中央が盛り上がった山。基準面からの高さ[mm]。"""
    rows, columns = np.mgrid[0:size, 0:size]
    center = size / 2
    squared = (rows - center) ** 2 + (columns - center) ** 2
    return (peak_mm * np.exp(-squared / (2 * spread**2))).astype(np.float32)


class TestHillshade:
    def test_平坦地では明るさが変わらない(self):
        """平坦地を 1.0 に正規化していないと、地形全体が暗く沈む。"""
        height = np.zeros((20, 20), dtype=np.float32)
        np.testing.assert_allclose(overlays.hillshade(height), 1.0, atol=1e-5)

    def test_投影像では左上から光が当たる(self):
        """山の左上が明るく、右下が影になること。

        向きを取り違えて光が下から当たると、人は山と谷を反転して知覚する
        （盛った砂が凹んで見える）。Renderer._to_view の左右反転を含めて
        確かめたいので、ここでも同じ反転を掛けてから比べる。
        """
        size = 60
        shade = overlays.hillshade(_dome(size))
        view = cv2.flip(shade, 1)

        quarter = size // 4
        assert view[quarter, quarter] > view[-quarter, -quarter]

    def test_一様な高さでも落ちない(self):
        height = np.full((20, 20), 50.0, dtype=np.float32)
        overlays.hillshade(height)


class TestTerrainColor:
    def test_水位より下は水になる(self):
        height = np.full((10, 10), config.WATER_LEVEL_MM - 10.0, dtype=np.float32)
        red, green, blue = overlays.terrain_color(height)[5, 5]
        assert blue > red and blue > green

    def test_深いほど水が濃くなる(self):
        shallow = np.full((10, 10), config.WATER_LEVEL_MM - 5.0, dtype=np.float32)
        deep = np.full((10, 10), config.WATER_DEEP_MM - 50.0, dtype=np.float32)

        assert int(overlays.terrain_color(deep)[5, 5].sum()) < int(
            overlays.terrain_color(shallow)[5, 5].sum()
        )

    def test_水面は深さだけで決まる(self):
        """水面に陰影が出ると水に見えないため、周囲の勾配に影響されないこと。"""
        flat = np.full((20, 20), config.WATER_LEVEL_MM - 20.0, dtype=np.float32)

        stepped = flat.copy()
        stepped[:, 10:] -= 30.0

        # 段差のすぐ隣の画素。陰影が掛かっていればここに明暗がつく。
        np.testing.assert_array_equal(
            overlays.terrain_color(stepped)[10, 9], overlays.terrain_color(flat)[10, 9]
        )

    def test_一番上の帯は雪になる(self):
        lower_mm, color = config.LAND_BANDS[-1]
        height = np.full((10, 10), lower_mm + 10.0, dtype=np.float32)

        # 平坦地なので陰影は 1.0。設定した色がそのまま出る。
        np.testing.assert_allclose(overlays.terrain_color(height)[5, 5], color, atol=1)

    def test_標高帯ごとに色が変わる(self):
        colors = set()
        for lower_mm, _ in config.LAND_BANDS:
            height = np.full((10, 10), lower_mm + 1.0, dtype=np.float32)
            colors.add(tuple(overlays.terrain_color(height)[5, 5]))

        assert len(colors) == len(config.LAND_BANDS)

    def test_水位は砂を動かしても変わらない(self):
        """dem_color と違いフレームごとに正規化しないこと。

        正規化してしまうと、掘るたびに水際が動いて演出にならない。
        """
        height = _dome()
        height -= 60.0  # 全体を掘り下げる

        raised = height + 200.0  # 砂を盛った状態
        under_water = overlays.terrain_color(height)
        assert (overlays.terrain_color(raised) != under_water).any()

    def test_形と型(self):
        result = overlays.terrain_color(np.zeros((24, 32), dtype=np.float32))
        assert result.shape == (24, 32, 3)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("value", [-1000.0, 0.0, 1000.0])
    def test_極端な高さでも落ちない(self, value):
        overlays.terrain_color(np.full((20, 20), value, dtype=np.float32))
