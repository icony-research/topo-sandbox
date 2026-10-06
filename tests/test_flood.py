"""水源から流れる水（堤防の決壊）の検証。GPU も Kinect も不要。

水の見た目は実機で確かめるしかないが、「堤防の内側に留まる」「切れたら外へ出る」
「発散しない」は手で作った地形で確かめられる。ここで固定しておかないと、
係数を触ったときに水が見えなくなったり、実演中に画面が壊れたりしても気づけない。
"""

from functools import cache

import numpy as np
import pytest

from topo_sandbox import config
from topo_sandbox.processing.flood import FloodSimulator

#: 1 フレームの間隔[秒]
_FRAME_S = 1.0 / 30.0

#: 水源の位置。河道の上流の端（左の縁からは離す。縁で水が消えるため）。
_SPRING = (0.06, 0.5)


def _levee(breach=False):
    """:func:`_make_levee` の写し。試験の中で書き換えても他の試験に響かない。"""
    return _make_levee(breach).copy()


@cache
def _make_levee(breach):
    """左から右へ下る河道と、その両側の堤防。堤防の外は川の水面より低い。

    大きさは処理解像度。`FloodSimulator` がここから `config.FLOOD_SIZE` へ縮める。

    Args:
        breach: 上側の堤防を途中で切るか。切ったところは河床と同じ高さ。
    """
    width, height = config.PROC_SIZE
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    slope = 20.0 - columns * 0.075  # 全体が右へ緩く下る
    distance = np.abs(rows - height / 2)

    terrain = np.where(distance >= 20, slope - 25.0, slope)  # 堤防の外（堤内地）
    terrain = np.where(distance < 12, slope - 15.0, terrain)  # 河道
    levee = (distance >= 12) & (distance < 20)
    terrain = np.where(levee, slope + 25.0, terrain)  # 堤防

    if breach:
        gap = levee & (rows < height / 2) & (np.abs(columns - 80) < 8)
        terrain = np.where(gap, slope - 15.0, terrain)
    return terrain.astype(np.float32)


def _outside(water):
    """上側の堤防の外の水。"""
    height = water.shape[0]
    return water[: int(height / 2 - 22)]


def _run(simulator, terrain_at, seconds, **kwargs):
    """一定の間隔でフレームを送り、最後に見せた水深を返す。"""
    visible = None
    for frame in range(int(seconds / _FRAME_S)):
        visible = simulator.update(terrain_at(frame), frame * _FRAME_S, **kwargs)
    return visible


class TestLevee:
    def test_堤防があれば外へ出ない(self):
        visible = _run(FloodSimulator(), lambda _: _levee(), 12.0, spring_uv=_SPRING)

        height = visible.shape[0]
        channel = visible[int(height / 2 - 10) : int(height / 2 + 10)]
        assert (channel > 0).mean() > 0.5, "河道に水が見えていない"
        assert not (_outside(visible) > 0).any()

    def test_堤防を切ると外へ溢れる(self):
        """平常時に流しておき、途中で子どもが堤防を切る。"""
        breach_at = int(8.0 / _FRAME_S)

        visible = _run(
            FloodSimulator(),
            lambda frame: _levee(breach=frame >= breach_at),
            12.0,
            spring_uv=_SPRING,
        )

        assert (_outside(visible) > 0).mean() > 0.01

    def test_大雨では川の水かさが増える(self):
        normal = FloodSimulator()
        _run(normal, lambda _: _levee(), 8.0, spring_uv=_SPRING)
        rain = FloodSimulator()
        _run(rain, lambda _: _levee(), 8.0, spring_uv=_SPRING, heavy_rain=True)

        assert rain.water.max() > normal.water.max() * 1.5

    def test_大雨で決壊すると平常時より広く溢れる(self):
        breach_at = int(8.0 / _FRAME_S)

        def terrain_at(frame):
            return _levee(breach=frame >= breach_at)

        normal = _run(FloodSimulator(), terrain_at, 12.0, spring_uv=_SPRING)
        rain = _run(FloodSimulator(), terrain_at, 12.0, spring_uv=_SPRING, heavy_rain=True)

        assert (_outside(rain) > 0).mean() > (_outside(normal) > 0).mean()


class TestStability:
    """実演中に画面を壊さないこと。"""

    def test_深い水柱が崩れても跳ね上がらない(self):
        """勢いに掛ける水深に上限が無かったとき、200mm の水柱が 390mm まで跳ねた。"""
        width, height = config.PROC_SIZE
        flat = np.zeros((height, width), dtype=np.float32)
        simulator = FloodSimulator()
        simulator.update(flat, 0.0)
        simulator.water[:, :20] = 200.0

        peak = 0.0
        for frame in range(1, 60):
            simulator.update(flat, frame * _FRAME_S)
            peak = max(peak, float(simulator.water.max()))

        assert peak <= 200.0

    def test_溜まった水の水面は平らになる(self):
        width, height = config.PROC_SIZE
        rows, columns = np.mgrid[0:height, 0:width]
        squared = (columns - width / 2) ** 2 + (rows - height / 2) ** 2
        bowl = (-150.0 * np.exp(-squared / (2 * 50.0**2))).astype(np.float32)

        simulator = FloodSimulator()
        _run(simulator, lambda _: bowl, 15.0, spring_uv=(0.5, 0.5), heavy_rain=True)

        pool = simulator.water > 5.0
        surface = (simulator._terrain + simulator.water)[pool]
        assert surface.std() < 1.0

    def test_でこぼこの砂場で時計が飛んでも壊れない(self):
        """揺れのある地形、詰まったフレーム、止まった時計を混ぜても有限で負にならない。"""
        rng = np.random.default_rng(0)
        width, height = config.PROC_SIZE
        rough = rng.normal(0.0, 40.0, (height, width)).astype(np.float32)

        simulator = FloodSimulator()
        now = 0.0
        for frame in range(300):
            now += rng.choice([_FRAME_S, 0.0, 0.5, -0.1])
            noise = rng.normal(0.0, 0.2, rough.shape).astype(np.float32)
            visible = simulator.update(
                rough + noise,
                now,
                spring_uv=(0.3, 0.4),
                heavy_rain=frame % 100 < 50,
                sea_level_mm=-30.0,
            )

        assert np.isfinite(simulator.water).all()
        assert (simulator.water >= 0.0).all()
        assert np.isfinite(visible).all()

    def test_地形にNaNが混じっても水は壊れない(self):
        terrain = _levee()
        terrain[100:110, 50:60] = np.nan

        visible = _run(FloodSimulator(), lambda _: terrain, 3.0, spring_uv=_SPRING)

        assert np.isfinite(visible).all()

    def test_水は湧いた量より増えない(self):
        """持っている以上の水を出さないので、水が勝手に湧くことはない。"""
        width, height = config.PROC_SIZE
        rows, columns = np.mgrid[0:height, 0:width]
        squared = (columns - width / 2) ** 2 + (rows - height / 2) ** 2
        bowl = (-150.0 * np.exp(-squared / (2 * 50.0**2))).astype(np.float32)

        simulator = FloodSimulator()
        seconds = 5.0
        _run(simulator, lambda _: bowl, seconds, spring_uv=(0.5, 0.5))

        assert simulator.water.sum() <= config.FLOOD_SPRING_FLOW * seconds


class TestClock:
    def test_時計が止まっていれば水は流れない(self):
        simulator = FloodSimulator()
        for _ in range(30):
            simulator.update(_levee(), 5.0, spring_uv=_SPRING)

        assert simulator.water.sum() == 0.0

    def test_詰まったあとに一気に進めない(self):
        """10 秒詰まっても、進めるのは 1 フレームの上限ぶんだけ。"""
        stalled = FloodSimulator()
        stalled.update(_levee(), 0.0, spring_uv=_SPRING)
        stalled.update(_levee(), 10.0, spring_uv=_SPRING)

        capped = FloodSimulator()
        capped.update(_levee(), 0.0, spring_uv=_SPRING)
        capped.update(_levee(), config.FLOOD_MAX_FRAME_S, spring_uv=_SPRING)

        assert stalled.water.sum() == pytest.approx(capped.water.sum())


class TestSinks:
    def test_海へ流れ込んだ水は消える(self):
        """海面を砂場全体より上げると、湧いたそばから消える。"""
        visible = _run(
            FloodSimulator(), lambda _: _levee(), 5.0, spring_uv=_SPRING, sea_level_mm=100.0
        )

        assert not (visible > 0).any()

    def test_砂場の外へ出た水は消える(self):
        """水源が砂場の外にあれば、水は残らない。"""
        area = [[0.5, 0.0], [0.5, 1.0], [1.0, 1.0], [1.0, 0.0]]

        visible = _run(FloodSimulator(), lambda _: _levee(), 5.0, spring_uv=_SPRING, area_uv=area)

        assert not (visible > 0).any()

    def test_四隅が潰れていれば視野全体を砂場とみなす(self):
        """面積の無いエリアで全部を外にすると、水が出たそばから消えて故障に見える。"""
        collapsed = [[0.5, 0.5]] * 4

        visible = _run(
            FloodSimulator(), lambda _: _levee(), 5.0, spring_uv=_SPRING, area_uv=collapsed
        )

        assert (visible > 0).any()

    def test_水源が無ければ水は出ない(self):
        visible = _run(FloodSimulator(), lambda _: _levee(), 3.0)

        assert not (visible > 0).any()


class TestDisplace:
    def test_手を入れたところの水は消える(self):
        """手の上に乗った水が流れ落ちて、手を抜いたあとに散らばらないこと。"""
        simulator = FloodSimulator()
        _run(simulator, lambda _: _levee(), 6.0, spring_uv=_SPRING)

        width, height = config.PROC_SIZE
        hand = _levee()
        hand[height // 2 - 12 : height // 2 + 12, 100:140] += 80.0  # 河道に手を入れた
        simulator.update(hand, 6.0, spring_uv=_SPRING)

        rows = slice(height // 4 - 5, height // 4 + 5)  # FLOOD_SIZE は処理解像度の半分
        assert (simulator.water[rows, 52:68] == 0.0).all()

    def test_センサの揺れでは消えない(self):
        clean = FloodSimulator()
        _run(clean, lambda _: _levee(), 6.0, spring_uv=_SPRING)
        shaken = FloodSimulator()
        _run(shaken, lambda _: _levee(), 6.0, spring_uv=_SPRING)

        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 0.2, _levee().shape).astype(np.float32)
        clean.update(_levee(), 6.0, spring_uv=_SPRING)
        shaken.update(_levee() + noise, 6.0, spring_uv=_SPRING)

        assert shaken.water.sum() == pytest.approx(clean.water.sum(), rel=0.01)


class TestReset:
    def test_頼むと次のフレームで抜ける(self):
        simulator = FloodSimulator()
        _run(simulator, lambda _: _levee(), 5.0, spring_uv=_SPRING)
        assert simulator.water.sum() > 0.0

        simulator.request_reset()
        visible = simulator.update(_levee(), 5.0, spring_uv=_SPRING)

        assert simulator.water.sum() == 0.0
        assert not (visible > 0).any()


class TestVisibleHysteresis:
    def test_薄い水は描かない(self):
        simulator = FloodSimulator()
        simulator.update(_levee(), 0.0)
        simulator.water[:] = config.FLOOD_HIDE_MM

        assert not (simulator._visible(config.PROC_SIZE[::-1]) > 0).any()

    def test_一度出た水は少し浅くなっても消えない(self):
        """しきい値ちょうどの水がセンサの揺れで点滅しないこと。"""
        simulator = FloodSimulator()
        simulator.update(_levee(), 0.0)
        shape = config.PROC_SIZE[::-1]

        simulator.water[:] = config.FLOOD_SHOW_MM + 0.1
        assert (simulator._visible(shape) > 0).all()

        simulator.water[:] = (config.FLOOD_SHOW_MM + config.FLOOD_HIDE_MM) / 2
        assert (simulator._visible(shape) > 0).all()

        simulator.water[:] = config.FLOOD_HIDE_MM - 0.1
        assert not (simulator._visible(shape) > 0).any()

    def test_入力と同じ大きさで返す(self):
        visible = FloodSimulator().update(_levee(), 0.0)
        assert visible.shape == _levee().shape
