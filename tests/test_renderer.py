"""射影変換と、水面のさざ波を進める時計の検証。GPU も Kinect も不要。"""

import cv2
import numpy as np
import pytest

from topo_sandbox import config
from topo_sandbox.processing.plane import ReferencePlane
from topo_sandbox.renderer import (
    MappingMode,
    Renderer,
    RenderSettings,
    ViewMode,
    default_projector_quad,
)


@pytest.fixture
def renderer():
    return Renderer()


def _settings(**overrides):
    settings = RenderSettings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _apply(matrix, point):
    """射影変換を 1 点に適用する。"""
    source = np.array([[point]], dtype=np.float32)
    return cv2.perspectiveTransform(source, matrix)[0][0]


class TestDefaultProjectorQuad:
    def test_既定は画面全体(self):
        width, height = config.VIEW_SIZE
        assert default_projector_quad() == [[0, 0], [0, height], [width, height], [width, 0]]

    def test_呼ぶたびに別の実体を返す(self):
        """使い回すと、片方を編集したつもりが両方変わってしまう。"""
        first = default_projector_quad()
        second = default_projector_quad()
        first[0][0] = 999
        assert second[0][0] == 0


class TestPerspectiveMatrix:
    def test_センサ四隅が投影枠の四隅へ写る(self):
        area = [[100, 50], [100, 400], [500, 400], [500, 50]]
        projector = [[40, 30], [60, 570], [740, 560], [760, 20]]
        settings = _settings(area_positions=area, projector_positions=projector)

        matrix = Renderer._perspective_matrix(None, settings)

        for source, expected in zip(area, projector):
            np.testing.assert_allclose(_apply(matrix, source), expected, atol=1e-3)

    def test_既定の投影枠では画面全体へ引き伸ばされる(self):
        """従来の挙動。投影枠を触らなければ今までと同じ結果になる。"""
        area = [[100, 50], [100, 400], [500, 400], [500, 50]]
        settings = _settings(area_positions=area)

        matrix = Renderer._perspective_matrix(None, settings)
        width, height = config.VIEW_SIZE

        np.testing.assert_allclose(_apply(matrix, area[0]), [0, 0], atol=1e-3)
        np.testing.assert_allclose(_apply(matrix, area[2]), [width, height], atol=1e-3)


class TestWarpGuards:
    def test_投影枠が潰れていても落ちない(self, renderer):
        """角を動かしすぎて一直線になった場合。実演中に停止させない。"""
        settings = _settings(
            area_positions=[[0, 0], [0, 100], [100, 100], [100, 0]],
            projector_positions=[[0, 0], [0, 0], [0, 0], [0, 0]],
        )
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        result = renderer._warp(image, settings)
        assert result.shape == image.shape

    def test_エリア未指定なら射影変換しない(self, renderer):
        settings = _settings(mapping_mode=MappingMode.PERSPECTIVE)
        assert renderer._use_perspective(settings) is False

    def test_エリアがそろえば射影変換する(self, renderer):
        settings = _settings(
            mapping_mode=MappingMode.PERSPECTIVE,
            area_positions=[[0, 0], [0, 100], [100, 100], [100, 0]],
        )
        assert renderer._use_perspective(settings) is True


class TestResetProjectorQuad:
    def test_画面全体へ戻る(self):
        settings = _settings()
        settings.projector_positions[0] = [123, 456]
        settings.reset_projector_quad()
        assert settings.projector_positions == default_projector_quad()


#: 合成フレームの基準面までの距離[mm]
_BASE_MM = 1000.0


class _FakeClock:
    """好きなだけ進められる時計。"""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _frame_with_pool():
    """中央を掘った砂場の深度[mm] (480, 640)。掘った底は水位より低い。"""
    width, height = config.SENSOR_SIZE
    rows, columns = np.mgrid[0:height, 0:width]
    squared = (rows - height / 2) ** 2 + (columns - width / 2) ** 2
    hollow = 120.0 * np.exp(-squared / (2 * 120.0**2))
    return (_BASE_MM + hollow).astype(np.float32)


def _dem_settings():
    plane = ReferencePlane(
        a=0.0,
        b=0.0,
        c=_BASE_MM,
        residual_mm=1.0,
        tilt_deg=0.0,
        distance_mm=_BASE_MM,
        coverage=99.0,
    )
    return _settings(view_mode=ViewMode.DEM, reference_plane=plane)


class TestWaterAnimationClock:
    """水面のさざ波はフレーム数ではなく時計で進む。

    app._tick は処理が間に合わないフレームを捨てるため、フレーム数で
    数えると混雑したときだけ波がゆっくりになってしまう。
    """

    def test_時計が止まっていれば同じ絵になる(self):
        renderer = Renderer(clock=lambda: 5.0)
        frame = _frame_with_pool()
        settings = _dem_settings()

        np.testing.assert_array_equal(
            renderer.render(frame, settings), renderer.render(frame, settings)
        )

    def test_時計が進むと水面が変わる(self):
        clock = _FakeClock()
        renderer = Renderer(clock=clock)
        frame = _frame_with_pool()
        settings = _dem_settings()

        first = renderer.render(frame, settings)
        clock.advance(1.1)
        second = renderer.render(frame, settings)

        assert (first != second).any()

    def test_基準面が無ければ時計に影響されない(self):
        """従来の DEM へ落ちるので、さざ波も出ない。"""
        clock = _FakeClock()
        renderer = Renderer(clock=clock)
        frame = _frame_with_pool()
        settings = _settings(view_mode=ViewMode.DEM)

        first = renderer.render(frame, settings)
        clock.advance(1.1)

        np.testing.assert_array_equal(first, renderer.render(frame, settings))
