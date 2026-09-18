"""DEM・テクスチャ・エッジ・等高線の検証。

いずれも numpy と OpenCV だけで完結するため、Kinect も GPU も要らない。
"""

import numpy as np
import pytest

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


class TestBlendTextures:
    def test_平地と斜面でテクスチャが貼り分けられる(self):
        height, width = 20, 20
        coloring = np.zeros((height, width, 3), dtype=np.uint8)
        coloring[:10] = [255, 0, 0]  # 赤 = 平坦
        coloring[10:] = [0, 0, 255]  # 青 = 斜面

        machi = np.full((height, width, 3), 60, dtype=np.uint8)
        mori = np.full((height, width, 3), 200, dtype=np.uint8)

        result = overlays.blend_textures(coloring, machi, mori)

        assert result[0, 0].max() == 60
        assert result[-1, 0].max() == 200


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
