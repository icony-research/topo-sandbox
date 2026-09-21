"""流向と流量の検証。GPU も Kinect も不要。

川の見た目は主観だが、流れの計算そのものは手で確かめられる。
ここで固定しておかないと、間違っていても「それらしい模様」が出てしまい
気づけない。
"""

import numpy as np

from topo_sandbox import config
from topo_sandbox.processing import rivers


def _ramp(rows=5, columns=3, step=10.0):
    """行が増えるほど低くなる斜面。水は真下へ流れる。"""
    height = -np.arange(rows, dtype=np.float32)[:, None] * step
    return np.repeat(height, columns, axis=1)


def _valley(depth=60.0):
    """中央の列がいちばん低い V 字の谷。全体が下へ傾いている。

    大きさを処理解像度に合わせてあるのは、`river_strength` がここから
    `config.RIVER_SIZE` へ縮めるため。これより小さい入力を与えると
    拡大になり、同じ高さの平坦なタイルができて水がどこへも流れなくなる。
    """
    width, height = config.PROC_SIZE
    rows, columns = np.mgrid[0:height, 0:width]

    across = np.abs(columns - width / 2) / (width / 2) * depth
    along = -rows.astype(np.float32) * 0.5
    return (across + along).astype(np.float32)


class TestFlowDirections:
    def test_急な下りへ流れる(self):
        height = _ramp()
        columns = height.shape[1]

        directions = rivers.flow_directions(height)

        # (0, 0) の水は真下の (1, 0) へ
        assert directions[0] == 1 * columns + 0

    def test_窪地は自分を指す(self):
        """下る先が無いところで水は溜まる。"""
        height = np.zeros((3, 3), dtype=np.float32)
        height[1, 1] = -50.0

        directions = rivers.flow_directions(height)

        assert directions[1 * 3 + 1] == 1 * 3 + 1

    def test_平坦地はどこへも流れない(self):
        height = np.zeros((4, 4), dtype=np.float32)

        directions = rivers.flow_directions(height)

        np.testing.assert_array_equal(directions, np.arange(16))

    def test_落差が同じなら斜めより真横が選ばれる(self):
        """距離で割らないと、遠い斜めばかりが選ばれて流れが歪む。"""
        height = np.zeros((3, 3), dtype=np.float32)
        height[1, 1] = 10.0

        directions = rivers.flow_directions(height)

        straight = {0 * 3 + 1, 1 * 3 + 0, 1 * 3 + 2, 2 * 3 + 1}
        assert directions[1 * 3 + 1] in straight

    def test_砂場の外へは出ていかない(self):
        """端の画素が場外を指すと、添字エラーで実演中に落ちる。"""
        height = _valley()

        directions = rivers.flow_directions(height)

        assert directions.min() >= 0
        assert directions.max() < height.size


class TestFlowAccumulation:
    def test_下流ほど水が集まる(self):
        height = _ramp()
        directions = rivers.flow_directions(height)

        accumulation = rivers.flow_accumulation(directions, height.size, 60)

        np.testing.assert_array_equal(accumulation.reshape(height.shape)[:, 0], [1, 2, 3, 4, 5])

    def test_出口に全画素ぶんが集まる(self):
        """水が途中で消えたり湧いたりしないこと。"""
        height = _ramp()
        directions = rivers.flow_directions(height)

        accumulation = rivers.flow_accumulation(directions, height.size, 60)

        sinks = directions == np.arange(height.size)
        assert accumulation[sinks].sum() == height.size

    def test_窪地があっても発散しない(self):
        """自分自身を指す画素を数え直すと、量が際限なく増えて画面が壊れる。"""
        height = np.zeros((5, 5), dtype=np.float32)
        height[2, 2] = -50.0
        directions = rivers.flow_directions(height)

        accumulation = rivers.flow_accumulation(directions, height.size, 200)

        assert np.isfinite(accumulation).all()
        assert accumulation.max() <= height.size

    def test_繰り返しを打ち切っても落ちない(self):
        """上限に達したときに例外を出さず、その時点の値を返すこと。"""
        height = _ramp(rows=30, columns=4)
        directions = rivers.flow_directions(height)

        accumulation = rivers.flow_accumulation(directions, height.size, 3)

        assert np.isfinite(accumulation).all()
        assert accumulation.min() >= 1.0

    def test_収束したら打ち切る前に止まる(self):
        """上限まで回さずに済むこと。回数だけ増やしても結果が変わらない。"""
        height = _ramp()
        directions = rivers.flow_directions(height)

        few = rivers.flow_accumulation(directions, height.size, 10)
        many = rivers.flow_accumulation(directions, height.size, 200)

        np.testing.assert_array_equal(few, many)


class TestRiverStrength:
    def test_谷筋が川になる(self):
        strength = rivers.river_strength(_valley())

        middle = strength.shape[1] // 2
        assert strength[-1, middle] > strength[-1, 5]

    def test_平坦地には川が出ない(self):
        """ならしたばかりの砂場。どこへも流れないので川は描かれない。"""
        width, height = config.PROC_SIZE
        strength = rivers.river_strength(np.zeros((height, width), dtype=np.float32))

        assert strength.max() == 0.0

    def test_値は0から1に収まる(self):
        strength = rivers.river_strength(_valley())

        assert strength.min() >= 0.0
        assert strength.max() <= 1.0

    def test_入力と同じ大きさで返る(self):
        height = np.zeros((240, 320), dtype=np.float32)

        assert rivers.river_strength(height).shape == (240, 320)

    def test_砂を動かしても濃さの基準は変わらない(self):
        """フレームごとに最大値で正規化すると、掘るたびに川の濃さが揺れる。"""
        valley = _valley()

        shallow = rivers.river_strength(valley)
        deep = rivers.river_strength(valley * 3.0)

        # 流れの向きは変わらないので、濃さもそのまま
        np.testing.assert_allclose(shallow, deep, atol=1e-6)

    def test_設定のしきい値より下は描かれない(self):
        """細い筋が大量に出ると地形が読めなくなる。"""
        assert config.RIVER_MIN_CELLS > 1.0
