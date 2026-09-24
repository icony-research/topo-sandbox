"""欠測の穴埋めと表示用階調への変換の検証。GPU も Kinect も不要。"""

import numpy as np
import pytest

from topo_sandbox import config
from topo_sandbox.processing.depth import (
    INVALID_DEPTH,
    TemporalStabilizer,
    fill_invalid,
    preprocess,
    to_display,
)


class _FlatPlane:
    """深度が一定の基準面。:class:`ReferencePlane` の代わりに使う。"""

    def __init__(self, depth_mm):
        self.depth_mm = depth_mm

    def depth_at(self, shape):
        return np.full(shape, self.depth_mm, dtype=np.float32)


class TestFillInvalid:
    def test_欠測が周囲の値で埋まる(self):
        depth = np.array([[1000, 1000, INVALID_DEPTH], [1000, 1200, 1000]], dtype=np.uint16)
        result = fill_invalid(depth)
        assert result[0, 2] == 1000

    def test_欠測は最も近い有効画素で埋まる(self):
        """遠くの値ではなく、すぐ隣の砂面の値を取る。"""
        depth = np.full((8, 8), 1000, dtype=np.uint16)
        depth[3:5, 3:5] = 900  # 盛った山
        depth[4, 4] = INVALID_DEPTH  # その頂上が測れなかった
        result = fill_invalid(depth)
        assert result[4, 4] == 900

    def test_散らばった欠測で高さが偏らない(self):
        """ちらついて飛んだ画素を埋めても、砂面の高さが動かないこと。

        近傍の最小値（最も高い面）で埋めると、飛んだ画素だけが数 mm 持ち上がり
        斑に見える。最も近い 1 画素を使うのはこのため。
        """
        rng = np.random.default_rng(0)
        depth = (1000 + rng.normal(0, 3, (120, 160))).astype(np.float32)
        dropped = rng.random(depth.shape) < 0.02
        depth[dropped] = INVALID_DEPTH

        result = fill_invalid(depth)
        assert abs(float(result[dropped].mean()) - 1000.0) < 1.0

    def test_有効画素はそのまま残る(self):
        depth = np.array([[1000, 1200], [INVALID_DEPTH, 1400]], dtype=np.uint16)
        result = fill_invalid(depth)
        assert result[0, 0] == 1000
        assert result[0, 1] == 1200
        assert result[1, 1] == 1400

    def test_欠測が突起にならない(self):
        """欠測を 0 のまま高さへ換算すると、非常に高い突起として扱われてしまう。"""
        depth = np.full((8, 8), 1000, dtype=np.uint16)
        depth[4, 4] = INVALID_DEPTH
        result = fill_invalid(depth)
        assert result.min() == result.max() == 1000

    def test_盛った山の頂上が測れなくても窪みにならない(self):
        """センサの最短測定距離より近づくと欠測になる。**実機で水面が出た原因。**

        遠くの床まで含めた中央値で埋めていたときは、山の頂上が砂面より
        250mm 低い値になり、DEM で水になっていた。
        """
        depth = np.full((64, 64), 1250, dtype=np.uint16)  # 砂場の外の床
        depth[16:48, 16:48] = 1000  # 砂場の砂面
        depth[28:36, 28:36] = INVALID_DEPTH  # 盛りすぎて測れない山の頂上

        result = fill_invalid(depth)
        assert result[32, 32] <= 1000  # 砂面と同じかそれより高い（近い）

    def test_有効画素が遠すぎる穴は基準面で埋まる(self):
        """近くに何も無いほど大きな穴。基準面があればその高さに合わせる。"""
        size = config.FILL_NEIGHBORHOOD_PX * 3
        depth = np.full((size, size), 1250, dtype=np.uint16)
        depth[: size - 4, : size - 4] = INVALID_DEPTH

        result = fill_invalid(depth, _FlatPlane(1000.0))
        assert result[0, 0] == 1000.0

    def test_基準面が無ければ中央値で埋まる(self):
        """従来の挙動。基準面を取る前でも動かせるようにしてある。"""
        size = config.FILL_NEIGHBORHOOD_PX * 3
        depth = np.full((size, size), 1250, dtype=np.uint16)
        depth[: size - 4, : size - 4] = INVALID_DEPTH

        result = fill_invalid(depth)
        assert result[0, 0] == 1250.0

    def test_全面欠測でも落ちない(self):
        """センサが覆われたときに起こりうる。"""
        depth = np.zeros((4, 4), dtype=np.uint16)
        result = fill_invalid(depth)
        assert result.shape == (4, 4)
        assert np.all(result == 0)


class TestToDisplay:
    def test_出力は8bit(self):
        depth = np.full((4, 4), 1000, dtype=np.float32)
        result = to_display(depth)
        assert result.dtype == np.uint8
        assert result.shape == (4, 4)

    def test_窓の外は折り返さずクリップされる(self):
        """窓を外れた深度は端に張り付く。折り返すと実在しない崖ができる。"""
        span = config.DEPTH_DISPLAY_SPAN_MM
        center = 1000.0
        depth = np.array(
            [[center - span, center, center + span]],
            dtype=np.float32,
        )
        result = to_display(depth, span_mm=span)
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_窓の中では深度の順序が保たれる(self):
        depth = np.array([[1000, 1050, 1100, 1150]], dtype=np.float32)
        result = to_display(depth).astype(np.int16)
        assert np.all(np.diff(result) > 0)

    def test_外れ値があっても階調が潰れない(self):
        """手をかざしたときのような外れ値で全体が暗くならないこと。

        中心を中央値に取っているため、砂場の起伏は階調を保つ。
        """
        depth = np.full((10, 10), 1000, dtype=np.float32)
        depth += np.linspace(0, 100, 100).reshape(10, 10)
        depth[0, 0] = 400  # かざした手

        result = to_display(depth)
        sandbox = result[1:]
        assert sandbox.max() - sandbox.min() > 50


class TestPreprocess:
    def test_処理解像度へ縮小される(self):
        depth = np.full((480, 640), 1000, dtype=np.uint16)
        result = preprocess(depth)
        assert result.shape == (config.PROC_SIZE[1], config.PROC_SIZE[0])

    def test_ミリメートルの値域が保たれる(self):
        """平滑化しても mm のスケールは変わらない。"""
        depth = np.full((480, 640), 1234, dtype=np.uint16)
        result = preprocess(depth)
        assert abs(float(result.mean()) - 1234) < 1.0

    def test_欠測が埋まった状態で返る(self):
        depth = np.full((480, 640), 1000, dtype=np.uint16)
        depth[100:110, 100:110] = INVALID_DEPTH
        result = preprocess(depth)
        assert abs(float(result.min()) - 1000) < 1.0

    def test_基準面を穴埋めへ渡せる(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[:4, :4] = 1250  # ほぼ全面が欠測
        result = preprocess(depth, _FlatPlane(1000.0))
        assert abs(float(result[120, 160]) - 1000.0) < 1.0  # 処理解像度の中央


class TestTemporalStabilizer:
    """静止しているところの揺れだけを時間方向に均す。"""

    def _noisy(self, rng, base=1000.0, noise_mm=1.0, shape=(60, 80)):
        return (base + rng.normal(0, noise_mm, shape)).astype(np.float32)

    def test_静止した砂場の揺れが減る(self):
        rng = np.random.default_rng(0)
        stabilizer = TemporalStabilizer()

        stabilizer.apply(self._noisy(rng))
        raw, smoothed = [], []
        for _ in range(30):
            frame = self._noisy(rng)
            raw.append(frame)
            smoothed.append(stabilizer.apply(frame))

        # 時間方向のばらつき（画素ごとの標準偏差の平均）で比べる
        assert np.std(smoothed, axis=0).mean() < np.std(raw, axis=0).mean() / 2

    def test_大きく動いた画素は遅れない(self):
        """手を入れた・砂を崩したといった変化にはその場で追従すること。"""
        stabilizer = TemporalStabilizer(smoothing=0.3, motion_mm=5.0)
        flat = np.full((10, 10), 1000.0, dtype=np.float32)
        stabilizer.apply(flat)

        moved = flat.copy()
        moved[5, 5] -= 40.0  # 40mm 盛った
        result = stabilizer.apply(moved)

        assert result[5, 5] == pytest.approx(960.0)  # そのまま通る
        assert result[0, 0] == pytest.approx(1000.0)

    def test_最初のフレームはそのまま返る(self):
        stabilizer = TemporalStabilizer()
        frame = np.full((4, 4), 1234.0, dtype=np.float32)
        np.testing.assert_allclose(stabilizer.apply(frame), frame)

    def test_解像度が変わっても落ちない(self):
        """再生モードと実機を切り替えたときなど。"""
        stabilizer = TemporalStabilizer()
        stabilizer.apply(np.full((10, 10), 1000.0, dtype=np.float32))
        result = stabilizer.apply(np.full((20, 20), 1000.0, dtype=np.float32))
        assert result.shape == (20, 20)

    def test_1なら素通しになる(self):
        """揺れが気になるとき以外は切れる、という逃げ道。"""
        stabilizer = TemporalStabilizer(smoothing=1.0)
        stabilizer.apply(np.full((4, 4), 1000.0, dtype=np.float32))
        frame = np.full((4, 4), 900.0, dtype=np.float32)
        np.testing.assert_allclose(stabilizer.apply(frame), frame)

    def test_忘れさせられる(self):
        stabilizer = TemporalStabilizer()
        stabilizer.apply(np.full((4, 4), 1000.0, dtype=np.float32))
        stabilizer.reset()
        frame = np.full((4, 4), 900.0, dtype=np.float32)
        np.testing.assert_allclose(stabilizer.apply(frame), frame)
