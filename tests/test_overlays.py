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
        deep = np.full(
            (10, 10), config.WATER_LEVEL_MM - config.WATER_DEEP_SPAN_MM - 50.0, dtype=np.float32
        )

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


class TestWaterWaves:
    def _water(self, depth_below_mm=20.0):
        return np.full((40, 40), config.WATER_LEVEL_MM - depth_below_mm, dtype=np.float32)

    def test_時間が進むと水面の明るさが変わる(self):
        """止まった青一色だと、掘った穴が水たまりに見えない。"""
        still = overlays.terrain_color(self._water(), elapsed_s=0.0)
        later = overlays.terrain_color(self._water(), elapsed_s=0.9)
        assert (still != later).any()

    def test_陸は時間で変わらない(self):
        """さざ波が陸へ漏れると、砂場全体がちらついて見える。"""
        land = np.full((40, 40), 50.0, dtype=np.float32)
        np.testing.assert_array_equal(
            overlays.terrain_color(land, elapsed_s=0.0),
            overlays.terrain_color(land, elapsed_s=0.9),
        )

    def test_振幅を0にすると波が止まる(self, monkeypatch):
        """実演中に目障りだったとき、config だけで止められること。"""
        monkeypatch.setattr(config, "WATER_WAVE_AMPLITUDE", 0.0)
        np.testing.assert_array_equal(
            overlays.terrain_color(self._water(), elapsed_s=0.0),
            overlays.terrain_color(self._water(), elapsed_s=0.9),
        )

    @pytest.mark.parametrize("elapsed_s", [0.0, 0.5, 1.0, 1.7, 2.6, 3.9])
    def test_波は水深の濃淡を消さない(self, elapsed_s):
        """振幅が大きすぎると、浅瀬と深場の区別がつかなくなる。"""
        shallow = self._water(5.0)
        deep = np.full(
            (40, 40), config.WATER_LEVEL_MM - config.WATER_DEEP_SPAN_MM - 50.0, dtype=np.float32
        )

        assert int(overlays.terrain_color(deep, elapsed_s).sum()) < int(
            overlays.terrain_color(shallow, elapsed_s).sum()
        )


class TestWaterLighting:
    def _water(self, depth_below_mm=30.0, size=60):
        return np.full((size, size), config.WATER_LEVEL_MM - depth_below_mm, dtype=np.float32)

    def test_一様な深さでもきらめきが出る(self):
        """深さが同じなら色も同じ、では水に見えない。

        水らしさは、細かい波が光源を映してきらめくところから来る。
        """
        brightness = overlays.terrain_color(self._water(), 0.0).astype(np.int32).sum(axis=2)
        assert brightness.max() - brightness.min() > 150

    def test_きらめきは陸へ漏れない(self):
        """水面の照り返しが陸に出ると、砂場全体がぎらついて読めなくなる。"""
        land = np.full((60, 60), 50.0, dtype=np.float32)
        brightness = overlays.terrain_color(land, 0.0).astype(np.int32).sum(axis=2)
        assert brightness.max() - brightness.min() == 0

    def test_水際に白波が出る(self, monkeypatch):
        shore = self._water(2.0)
        with_surf = overlays.terrain_color(shore, 0.0).astype(np.int32).sum()

        monkeypatch.setattr(config, "WATER_SURF_STRENGTH", 0.0)
        without_surf = overlays.terrain_color(shore, 0.0).astype(np.int32).sum()

        assert with_surf > without_surf

    def test_白波は深場には出ない(self, monkeypatch):
        deep = self._water(200.0)
        with_surf = overlays.terrain_color(deep, 0.0).astype(np.int32).sum()

        monkeypatch.setattr(config, "WATER_SURF_STRENGTH", 0.0)
        without_surf = overlays.terrain_color(deep, 0.0).astype(np.int32).sum()

        assert with_surf == without_surf

    def test_波が1本も無くても落ちない(self, monkeypatch):
        """設定を空にしたときにゼロ除算で止まらないこと。"""
        monkeypatch.setattr(config, "WATER_WAVES", [])
        result = overlays.terrain_color(self._water(), 1.0)
        assert np.isfinite(result.astype(np.float32)).all()

    def test_水が1画素も無くても落ちない(self):
        """設営直後はまだ誰も掘っていない。水の計算へ入らない経路。"""
        land = np.full((60, 60), 10.0, dtype=np.float32)
        result = overlays.terrain_color(land, 1.0)
        assert result.shape == (60, 60, 3)

    def test_全面が水でも落ちない(self):
        result = overlays.terrain_color(self._water(150.0), 1.0)
        assert result.shape == (60, 60, 3)


class TestWaterLevel:
    def test_水位を上げると陸が沈む(self):
        land = np.full((20, 20), 0.0, dtype=np.float32)

        dry = overlays.terrain_color(land, 0.0, water_level_mm=-40.0)
        flooded = overlays.terrain_color(land, 0.0, water_level_mm=20.0)

        # 水は青が勝ち、陸（低地の緑）は赤が青を上回る
        assert flooded[..., 2].mean() > flooded[..., 0].mean()
        assert dry[..., 0].mean() > dry[..., 2].mean()

    def test_水位を下げると水が引く(self):
        hollow = np.full((20, 20), -50.0, dtype=np.float32)

        flooded = overlays.terrain_color(hollow, 0.0, water_level_mm=-40.0)
        dry = overlays.terrain_color(hollow, 0.0, water_level_mm=-100.0)

        # 現れるのは砂浜。淡い色なので青も高いが、赤が上回る
        assert flooded[..., 2].mean() > flooded[..., 0].mean()
        assert dry[..., 0].mean() > dry[..., 2].mean()

    def test_水位を動かしても水の濃さは変わらない(self):
        """濃紺までの幅を水位からの相対にしてあること。

        絶対値のままだと、水位を上げたとき水全体が薄い水色に寝てしまう。
        """
        colors = set()
        for level in (-40.0, 0.0, 60.0):
            water = np.full((20, 20), level - 30.0, dtype=np.float32)
            colors.add(tuple(overlays.terrain_color(water, 0.0, level)[10, 10]))

        assert len(colors) == 1

    def test_標高帯は水位につられて動かない(self):
        """地面そのものは変わらないので、帯は絶対高さのまま。"""
        land = np.full((20, 20), 50.0, dtype=np.float32)

        np.testing.assert_array_equal(
            overlays.terrain_color(land, 0.0, water_level_mm=-40.0),
            overlays.terrain_color(land, 0.0, water_level_mm=-100.0),
        )


class TestVolcano:
    def _peak(self, above_mm=20.0):
        return np.full((20, 20), config.VOLCANO_HEIGHT_MM + above_mm, dtype=np.float32)

    def test_高く盛ると溶岩になる(self):
        lava = overlays.terrain_color(self._peak(), 0.0)[10, 10]
        assert int(lava[0]) - int(lava[2]) > 50

    def test_しきい値未満は雪のまま(self):
        snow = np.full((20, 20), config.VOLCANO_HEIGHT_MM - 10.0, dtype=np.float32)
        result = overlays.terrain_color(snow, 0.0)[10, 10]

        np.testing.assert_allclose(result, config.LAND_BANDS[-1][1], atol=1)

    def test_溶岩は時間で揺らぐ(self):
        peak = self._peak()
        assert (overlays.terrain_color(peak, 0.0) != overlays.terrain_color(peak, 3.0)).any()

    def test_溶岩に陰影は掛からない(self):
        """自分で光っているものに影がつくと、ただの赤い岩に見える。"""
        flat = self._peak()
        stepped = flat.copy()
        stepped[:, 10:] += 40.0

        # 段差のすぐ隣の画素。陰影が掛かっていればここに明暗がつく。
        np.testing.assert_array_equal(
            overlays.terrain_color(stepped, 0.0)[10, 9],
            overlays.terrain_color(flat, 0.0)[10, 9],
        )

    def test_火口の中心ほど明るい(self):
        rim = overlays.terrain_color(self._peak(0.0), 0.0).astype(np.int32)
        core = overlays.terrain_color(self._peak(config.VOLCANO_SPAN_MM), 0.0).astype(np.int32)

        assert core.sum() > rim.sum()

    def test_水位を上げれば火口も沈む(self):
        peak = self._peak()
        level = config.VOLCANO_HEIGHT_MM + 100.0
        flooded = overlays.terrain_color(peak, 0.0, water_level_mm=level)[10, 10]

        assert int(flooded[2]) > int(flooded[0])
