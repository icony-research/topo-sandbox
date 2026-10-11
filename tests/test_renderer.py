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
    sensor_area_from_view,
    sensor_point_from_screen,
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


class TestSensorAreaFromView:
    def test_左右反転を戻す(self):
        """表示像は左右反転されているので、u は 1 から引いた値になる。

        符号を揃えると、砂場の反対側で基準面をあてはめることになる。
        """
        width, height = config.VIEW_SIZE
        area = sensor_area_from_view([[0, 0], [width, height]])
        assert area[0] == pytest.approx([1.0, 0.0])
        assert area[1] == pytest.approx([0.0, 1.0])

    def test_正規化座標で返す(self):
        """センサ解像度にも処理解像度にも当てられるよう 0〜1 にする。"""
        width, height = config.VIEW_SIZE
        area = sensor_area_from_view([[width // 4, height // 2]])
        assert area[0] == pytest.approx([0.75, 0.5])


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


def _dem_settings(**overrides):
    plane = ReferencePlane(
        a=0.0,
        b=0.0,
        c=_BASE_MM,
        residual_mm=1.0,
        tilt_deg=0.0,
        distance_mm=_BASE_MM,
        coverage=99.0,
    )
    return _settings(view_mode=ViewMode.DEM, reference_plane=plane, **overrides)


class TestMissingDepthIsNotWater:
    """測れなかった画素が水面にならないこと。

    盛った山の頂上がセンサの最短測定距離より近づくと欠測になる。これを
    視野全体の中央値で埋めていたときは、砂場の外の床に引きずられて
    山の頂上が深い水になっていた（実機で「高さが逆」に見えた原因）。
    """

    def test_山の頂上が測れなくても水にならない(self, renderer):
        width, height = config.SENSOR_SIZE
        rows, columns = np.mgrid[0:height, 0:width]
        inside = (np.abs(rows - height / 2) < height / 4) & (
            np.abs(columns - width / 2) < width / 4
        )

        frame = np.where(inside, _BASE_MM, _BASE_MM + 250.0)  # 砂場の外は遠い床
        squared = (rows - height / 2) ** 2 + (columns - width / 2) ** 2
        frame = frame - 200.0 * np.exp(-squared / (2 * 60.0**2))  # 盛った山
        frame = np.where(frame < 800.0, 0.0, frame)  # 最短測定距離より近い所は欠測

        settings = _dem_settings(mapping_mode=MappingMode.NORMAL)
        image = renderer.render(frame.astype(np.float32), settings)

        # 山の頂上は投影像の中央。水（青）ではなく陸の色になる。
        red, green, blue = image[image.shape[0] // 2, image.shape[1] // 2]
        assert not (blue > red and blue > green), f"頂上が水になっている: {(red, green, blue)}"


class TestReset:
    """初期値へ戻す 2 段階。"""

    def _adjusted(self):
        plane = ReferencePlane(
            a=1.0, b=2.0, c=1000.0, residual_mm=1.0, tilt_deg=1.0, distance_mm=1000.0, coverage=90.0
        )
        return _settings(
            view_mode=ViewMode.DEM,
            mapping_mode=MappingMode.PERSPECTIVE,
            z_scale=0.3,
            color_sensitivity=3.5,
            show_contour=True,
            show_rivers=True,
            river_preset=3,
            water_level_mm=20.0,
            area_positions=[[1, 2], [3, 4], [5, 6], [7, 8]],
            projector_positions=[[9, 9], [9, 90], [90, 90], [90, 9]],
            reference_plane=plane,
            show_flood=True,
            heavy_rain=True,
            spring_position=[0.2, 0.3],
        )

    def test_調整値だけ戻す(self):
        settings = self._adjusted()
        settings.reset_adjustments()

        defaults = RenderSettings()
        assert settings.z_scale == defaults.z_scale
        assert settings.color_sensitivity == defaults.color_sensitivity
        assert settings.water_level_mm == defaults.water_level_mm
        assert settings.river_preset == defaults.river_preset
        assert settings.view_mode is defaults.view_mode
        assert settings.mapping_mode is defaults.mapping_mode
        assert not settings.show_contour and not settings.show_rivers
        assert not settings.show_flood and not settings.heavy_rain

    def test_設営は残す(self):
        """合わせ直すのに時間が掛かる。実演中に消えると立て直せない。"""
        settings = self._adjusted()
        setup = ("area_positions", "projector_positions", "reference_plane", "spring_position")
        before = {name: getattr(settings, name) for name in setup}

        settings.reset_adjustments()

        assert {name: getattr(settings, name) for name in setup} == before

    def test_すべて戻すと設営も消える(self):
        settings = self._adjusted()
        settings.reset_all()

        assert settings.area_positions == []
        assert settings.projector_positions == default_projector_quad()
        assert settings.reference_plane is None
        assert settings.spring_position is None
        assert settings.z_scale == RenderSettings().z_scale


class TestContourStability:
    """センサの揺れで等高線が踊らないこと（実機で出た不具合）。

    深度を時間方向に均すのと、等高線の帯を覚えておくのと、二段構えで抑えている。
    """

    def _noisy_frames(self, count, noise_mm=3.0):
        rng = np.random.default_rng(0)
        base = _frame_with_pool()
        for _ in range(count):
            # Kinect は 2mm 刻みで、静止していてもフレームごとに揺れる。
            noisy = base + rng.normal(0, noise_mm, base.shape)
            yield (np.round(noisy / 2.0) * 2.0).astype(np.float32)

    def _line_swap_ratio(self, renderer, settings):
        previous = None
        swapped = []
        for frame in self._noisy_frames(12):
            image = renderer.render(frame, settings)
            line = image.sum(axis=2) < 60  # 黒い線の画素
            if previous is not None:
                swapped.append((line != previous).sum() / max(line.sum(), 1))
            previous = line
        return float(np.mean(swapped))

    def test_砂を動かさなければ線もほとんど動かない(self):
        settings = _dem_settings(show_contour=True)
        assert self._line_swap_ratio(Renderer(), settings) < 0.05

    def test_抑えを外すと線が踊る(self):
        """抑えが効いていることの裏取り。外すと入れ替わりが桁で増える。"""
        settings = _dem_settings(show_contour=True)

        renderer = Renderer()
        renderer._stabilizer.smoothing = 1.0
        renderer._contours.hysteresis_mm = 0.0

        assert self._line_swap_ratio(renderer, settings) > 0.5


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


class TestRivers:
    def test_切り替えで見た目が変わる(self):
        renderer = Renderer(clock=lambda: 0.0)
        frame = _frame_with_pool()

        without = renderer.render(frame, _dem_settings())
        with_rivers = renderer.render(frame, _dem_settings(show_rivers=True))

        assert (without != with_rivers).any()

    def test_水面より下には描かない(self):
        """湖や海に入った川は見えなくなる。"""
        renderer = Renderer(clock=lambda: 0.0)
        frame = _frame_with_pool()

        # 水位を上げ切ると砂場が全部沈むので、川は 1 本も出ない
        drowned = _dem_settings(show_rivers=True, water_level_mm=config.WATER_LEVEL_MAX_MM)
        flooded = _dem_settings(water_level_mm=config.WATER_LEVEL_MAX_MM)

        np.testing.assert_array_equal(
            renderer.render(frame, drowned), renderer.render(frame, flooded)
        )

    def test_基準面が無くても描ける(self):
        """流れの向きは高さの原点に依らないので、k を押す前でも出せる。"""
        renderer = Renderer(clock=lambda: 0.0)
        frame = _frame_with_pool()

        without = renderer.render(frame, _settings(view_mode=ViewMode.DEM))
        with_rivers = renderer.render(frame, _settings(view_mode=ViewMode.DEM, show_rivers=True))

        assert (without != with_rivers).any()


class TestSensorPointFromScreen:
    """水源を置くためのクリック位置の変換。"""

    def test_左右反転を戻す(self):
        width, height = config.VIEW_SIZE
        point = sensor_point_from_screen((width // 4, height // 2), _settings())
        assert point == pytest.approx([0.75, 0.5])

    def test_射影変換モードでは変換を逆にたどる(self):
        """画面に映っているのは引き伸ばしたあとの像。投影枠の角をクリックすると
        エリアの角を指したことになる。"""
        width, height = config.VIEW_SIZE
        area = [[200, 150], [200, 450], [600, 450], [600, 150]]
        settings = _settings(mapping_mode=MappingMode.PERSPECTIVE, area_positions=area)

        point = sensor_point_from_screen((0, 0), settings)

        assert point == pytest.approx([1.0 - 200 / width, 150 / height], abs=1e-4)

    def test_エリアがそろっていなければ変換しない(self):
        settings = _settings(mapping_mode=MappingMode.PERSPECTIVE, area_positions=[[1, 2]])
        width, height = config.VIEW_SIZE
        assert sensor_point_from_screen((0, 0), settings) == pytest.approx([1.0, 0.0])

    def test_砂場の外ならNone(self):
        area = [[200, 150], [200, 450], [600, 450], [600, 150]]
        settings = _settings(mapping_mode=MappingMode.PERSPECTIVE, area_positions=area)
        # 投影枠を画面の中央だけにすると、画面の隅は砂場の外になる
        settings.projector_positions = [[300, 200], [300, 400], [500, 400], [500, 200]]

        assert sensor_point_from_screen((0, 0), settings) is None

    def test_四隅が潰れていても落ちない(self):
        settings = _settings(mapping_mode=MappingMode.PERSPECTIVE, area_positions=[[100, 100]] * 4)
        assert sensor_point_from_screen((400, 300), settings) == pytest.approx([0.5, 0.5])


def _flood_settings(**overrides):
    """中央の窪みに水源を置いた DEM 設定。海は窪みより低くしておく。"""
    overrides.setdefault("show_flood", True)
    overrides.setdefault("spring_position", [0.5, 0.5])
    overrides.setdefault("water_level_mm", config.WATER_LEVEL_MIN_MM)
    return _dem_settings(**overrides)


class TestFlood:
    def _render_for(self, renderer, clock, settings, seconds):
        frame = _frame_with_pool()
        image = None
        for _ in range(int(seconds * 30)):
            clock.advance(1.0 / 30.0)
            image = renderer.render(frame, settings)
        return image

    def test_水源から水が流れると見た目が変わる(self):
        clock = _FakeClock()
        renderer = Renderer(clock=clock)
        dry = renderer.render(_frame_with_pool(), _flood_settings(show_flood=False))

        wet = self._render_for(renderer, clock, _flood_settings(), 2.0)

        assert (dry != wet).any()
        assert renderer._flood.water.sum() > 0.0

    def test_切ると水が抜ける(self):
        """見えないあいだに水を持ち越さない。"""
        clock = _FakeClock()
        renderer = Renderer(clock=clock)
        self._render_for(renderer, clock, _flood_settings(), 1.0)

        renderer.render(_frame_with_pool(), _flood_settings(show_flood=False))

        assert renderer._flood.water is None

    def test_DEM以外では流さない(self):
        clock = _FakeClock()
        renderer = Renderer(clock=clock)
        self._render_for(renderer, clock, _flood_settings(), 1.0)

        settings = _flood_settings()
        settings.view_mode = ViewMode.DEPTH
        renderer.render(_frame_with_pool(), settings)

        assert renderer._flood.water is None

    def test_基準面が無ければ流さない(self):
        """高さの原点がフレームごとに動き、水が揺すられて勝手に流れ出すため。"""
        clock = _FakeClock()
        renderer = Renderer(clock=clock)
        settings = _settings(view_mode=ViewMode.DEM, show_flood=True, spring_position=[0.5, 0.5])

        self._render_for(renderer, clock, settings, 1.0)

        assert renderer._flood.water is None

    def test_水を抜くと次のフレームで空になる(self):
        clock = _FakeClock()
        renderer = Renderer(clock=clock)
        self._render_for(renderer, clock, _flood_settings(), 1.0)

        renderer.drain_flood()
        renderer.render(_frame_with_pool(), _flood_settings())

        assert renderer._flood.water.sum() == 0.0

    def test_水を流しているあいだは雨の筋を描かない(self):
        """D8 の筋と本物の水が重なると、どちらが水か見分けられない。"""
        frame = _frame_with_pool()
        without = Renderer(clock=lambda: 0.0).render(frame, _flood_settings())
        with_rivers = Renderer(clock=lambda: 0.0).render(frame, _flood_settings(show_rivers=True))

        np.testing.assert_array_equal(without, with_rivers)
