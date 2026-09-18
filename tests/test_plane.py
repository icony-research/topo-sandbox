"""基準面の推定の検証。GPU も Kinect も不要。"""

import numpy as np
import pytest

from topo_sandbox.processing.plane import (
    ReferencePlane,
    average_frames,
    capture,
    fit_plane,
)

SHAPE = (240, 320)


def _tilted_depth(distance_mm=1000.0, slope_u=0.0, slope_v=0.0, shape=SHAPE):
    """傾いた平面の深度画像を作る。

    slope_u / slope_v は画像の端から端までの深度の変化量[mm]。
    """
    height, width = shape
    v, u = np.meshgrid(
        np.linspace(0.0, 1.0, height),
        np.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    return (distance_mm + slope_u * u + slope_v * v).astype(np.float32)


class TestFitPlane:
    def test_水平な面は傾きゼロ(self):
        result = fit_plane(_tilted_depth())
        assert result.tilt_deg == pytest.approx(0.0, abs=0.01)
        assert result.distance_mm == pytest.approx(1000.0, abs=0.1)
        assert result.residual_mm == pytest.approx(0.0, abs=0.01)

    def test_傾いた面の係数を復元できる(self):
        result = fit_plane(_tilted_depth(slope_u=60.0, slope_v=-30.0))
        assert result.a == pytest.approx(60.0, abs=0.1)
        assert result.b == pytest.approx(-30.0, abs=0.1)

    def test_傾きが度で得られる(self):
        """端から端まで 60mm 変化する面の傾きを幾何から検算する。"""
        result = fit_plane(_tilted_depth(distance_mm=1000.0, slope_u=60.0))

        # 実世界での横幅 = 距離 × 画素数 / 焦点距離
        from topo_sandbox import config

        span_mm = result.distance_mm * config.SENSOR_SIZE[0] / config.SENSOR_FOCAL_PX[0]
        expected = np.degrees(np.arctan(60.0 / span_mm))
        assert result.tilt_deg == pytest.approx(expected, abs=0.05)

    def test_欠測は無視される(self):
        depth = _tilted_depth(slope_u=40.0)
        depth[:50, :50] = 0
        result = fit_plane(depth)
        assert result.a == pytest.approx(40.0, abs=0.5)

    def test_外れ値に引きずられない(self):
        """視野に入り込んだ手のような塊があっても基準面はずれない。"""
        depth = _tilted_depth(distance_mm=1000.0)
        depth[100:140, 100:140] -= 300.0  # 手をかざした状態
        result = fit_plane(depth)
        assert result.distance_mm == pytest.approx(1000.0, abs=2.0)

    def test_有効画素が少なすぎると例外(self):
        depth = np.zeros(SHAPE, dtype=np.float32)
        with pytest.raises(ValueError):
            fit_plane(depth)


class TestHeights:
    def test_傾いた面が平らになる(self):
        """傾き補正の本題。斜めに取り付けても平らな砂は平らに出る。"""
        depth = _tilted_depth(slope_u=80.0, slope_v=-40.0)
        heights = fit_plane(depth).heights(depth)
        assert np.abs(heights).max() < 0.5

    def test_盛り上がりは正の高さになる(self):
        depth = _tilted_depth(distance_mm=1000.0)
        reference = fit_plane(depth)

        bumped = depth.copy()
        bumped[100:140, 100:140] -= 50.0  # センサに 50mm 近づく＝盛り上がり

        heights = reference.heights(bumped)
        assert heights[120, 120] == pytest.approx(50.0, abs=0.5)
        assert heights[0, 0] == pytest.approx(0.0, abs=0.5)

    def test_解像度が違っても使える(self):
        """正規化座標で保持しているため、別の解像度へも適用できる。"""
        reference = fit_plane(_tilted_depth(slope_u=60.0, shape=(240, 320)))
        small = _tilted_depth(slope_u=60.0, shape=(120, 160))
        heights = reference.heights(small)
        assert np.abs(heights).max() < 1.0


class TestAverageFrames:
    def test_ちらつきが均される(self):
        base = _tilted_depth()
        frames = [base + offset for offset in (-2.0, 0.0, 2.0)]
        averaged = average_frames(frames)
        np.testing.assert_allclose(averaged, base, atol=0.01)

    def test_欠測は平均に含めない(self):
        base = _tilted_depth()
        noisy = base.copy()
        noisy[0, 0] = 0
        averaged = average_frames([base, noisy, base])
        assert averaged[0, 0] == pytest.approx(base[0, 0], abs=0.01)

    def test_一度も測れない画素は欠測のまま(self):
        base = _tilted_depth()
        base[0, 0] = 0
        averaged = average_frames([base, base])
        assert averaged[0, 0] == 0


class TestCapture:
    def test_フレーム列から基準面が得られる(self):
        frames = [_tilted_depth(slope_u=50.0) for _ in range(5)]
        result = capture(frames)
        assert isinstance(result, ReferencePlane)
        assert result.a == pytest.approx(50.0, abs=0.5)

    def test_要約が読める形で出る(self):
        text = capture([_tilted_depth()]).describe()
        assert "基準面" in text
        assert "傾き" in text
        assert "残差" in text
