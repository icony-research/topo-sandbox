"""水源から流れ出た水の動き。堤防を越えたり、切れたところから溢れたりする様子を出す。

川（:mod:`.rivers`、D8 法）は「雨が降ったらどこを流れるか」を毎フレーム数え直す
だけで、水の量を持たない。そのため川の水かさが増えて堤防を越える、切れ目から
外へ流れ出す、といった動きは出せない。ここではマスごとの**水深を持ち続け**、
水面（地形 + 水深）の高さの差で隣へ水を押し出す。

計算は仮想パイプ法（浅水流を簡単にしたもの）。隣り合うマスを管でつなぎ、
水面の差で管の流れを加速させる。各マスが持っている以上の水は出さないので、
水深が負になったり、水が湧いて増えたりしない。

numpy と OpenCV だけで完結するので GPU も Kinect も要らない。
"""

import cv2
import numpy as np

from .. import config


class FloodSimulator:
    """水深を覚えておき、フレームごとに水を流す。

    前のフレームの水を持つので、:class:`~topo_sandbox.processing.depth.TemporalStabilizer`
    と同じく `Renderer` が 1 つ抱え、ワーカスレッドから順番に呼ぶ（同時に 2 つ
    走らないことは `app._tick` が保証している）。

    水を流す時間はフレーム数ではなく時計で決める。`app._tick` は処理が
    間に合わないフレームを捨てるため、フレーム数で数えると混雑したときだけ
    水がゆっくり流れる（さざ波と同じ理由）。
    """

    def __init__(self, size=None):
        """
        Args:
            size: 計算する解像度 (幅, 高さ)。省略すると :data:`config.FLOOD_SIZE`。
        """
        self.size = size or config.FLOOD_SIZE
        self._reset_requested = False
        self._area_key = None
        self._area_mask = None
        self.reset()

    def reset(self):
        """水をすべて抜く。次の :meth:`update` で空の砂場から始まる。"""
        self.water = None  #: 水深[mm] (H, W)
        #: 4 方向への流れ[mm/秒] (4, H, W)。0 が右（列 +1）、1 が左、2 が下（行 +1）、3 が上。
        self._flux = None
        self._terrain = None  #: 前のフレームの地形[mm]
        self._shown = None  #: 前のフレームで水として描いたマス
        self._last_s = None
        self._pending_s = 0.0

    def request_reset(self):
        """水を抜くよう頼む。次の :meth:`update` の頭で抜く。

        キー操作はメインスレッド、水の計算はワーカスレッドで動くので、
        メインスレッドから直接 :meth:`reset` を呼ぶと計算の途中で配列が
        消えることがある。印だけ付けて、ワーカ側で抜く。
        """
        self._reset_requested = True

    # ------------------------------------------------------------------
    def update(
        self,
        height_mm,
        elapsed_s,
        spring_uv=None,
        heavy_rain=False,
        sea_level_mm=None,
        area_uv=None,
    ):
        """前のフレームから経った時間ぶん水を流し、見せる水深を返す。

        Args:
            height_mm: 基準面からの高さ[mm] (H, W)。
            elapsed_s: 表示を始めてからの経過秒。
            spring_uv: 水源の位置。センサ画像の正規化座標 (u, v)。None なら水は湧かない。
            heavy_rain: 大雨か。水源の量を :data:`config.FLOOD_RAIN_FACTOR` 倍にする。
            sea_level_mm: 海の高さ[mm]。これより低いマスへ流れ込んだ水は海に消える。
                None なら海は無い。
            area_uv: 砂場の四隅。センサ画像の正規化座標。外へ出た水は消える
                （砂場の縁から床へこぼれた扱い）。None なら視野全体が砂場。

        Returns:
            見せる水深[mm] (H, W) float32。入力と同じ大きさ。水として描かない
            ところは 0。
        """
        if self._reset_requested:
            self._reset_requested = False
            self.reset()

        height = np.asarray(height_mm, dtype=np.float32)
        terrain = cv2.resize(height, dsize=self.size, interpolation=cv2.INTER_AREA)
        # 穴埋めのあとなので本来は入らないが、1 マスでも NaN があると周りへ広がって
        # 画面全体の水が壊れる。実演中に止めないための保険。
        terrain = np.nan_to_num(terrain, nan=0.0, posinf=0.0, neginf=0.0)

        if self.water is None or self.water.shape != terrain.shape:
            self.water = np.zeros_like(terrain)
            self._flux = np.zeros((4,) + terrain.shape, dtype=np.float32)
            self._shown = None
        else:
            self._displace(terrain)
        self._terrain = terrain

        steps = self._steps(elapsed_s)
        if steps:
            sink = self._sink(terrain, sea_level_mm, area_uv)
            source = self._source(terrain.shape, spring_uv, heavy_rain)
            for _ in range(steps):
                self._step(terrain, sink, source)

        if not np.isfinite(self.water).all():
            # 計算上は起きないはずだが、起きたら水だけ捨てて続ける。
            self.reset()
            return np.zeros_like(height)

        return self._visible(height.shape)

    # ------------------------------------------------------------------
    def _steps(self, elapsed_s):
        """今回のフレームで水を何刻み進めるか。

        刻みの幅は :data:`config.FLOOD_STEP_S` に固定する。フレームの間隔で
        刻みを変えると、遅いフレームほど刻みが粗くなって計算が発散しやすい。
        """
        if self._last_s is None:
            self._last_s = elapsed_s
            return 0

        passed = elapsed_s - self._last_s
        self._last_s = elapsed_s
        # 時計が戻ったり、処理が長く詰まったりしたときに一気に進めない。
        passed = min(max(passed, 0.0), config.FLOOD_MAX_FRAME_S)

        self._pending_s += passed
        steps = int(self._pending_s / config.FLOOD_STEP_S)
        self._pending_s -= steps * config.FLOOD_STEP_S
        return steps

    def _displace(self, terrain):
        """急に高くなったマスの水を消す。

        手を入れたところや、砂を川へ落としたところ。水を残しておくと手の上に
        乗った水が流れ落ち、手を抜いたあとに砂場へ水が散らばる。押しのけた水を
        周りへ配るほうが本物らしいが、手が通り過ぎるたびに水が増えて見える。
        """
        previous = self._terrain
        if previous is None or previous.shape != terrain.shape:
            return
        raised = terrain - previous > config.FLOOD_DISPLACE_MM
        if raised.any():
            self.water[raised] = 0.0
            self._flux[:, raised] = 0.0

    def _sink(self, terrain, sea_level_mm, area_uv):
        """水が消えるマス。海・砂場の外・視野の縁。"""
        sink = np.zeros(terrain.shape, dtype=bool)

        # 視野の縁から先は測れていない。水はそこで砂場から出ていったとみなす。
        sink[0, :] = sink[-1, :] = True
        sink[:, 0] = sink[:, -1] = True

        if sea_level_mm is not None:
            sink |= terrain < sea_level_mm

        area = self._area(terrain.shape, area_uv)
        if area is not None:
            sink |= ~area
        return sink

    def _area(self, shape, area_uv):
        """砂場の内側のマス。四隅が変わったときだけ作り直す。"""
        if area_uv is None or len(area_uv) != 4:
            return None

        key = (shape, tuple(tuple(map(float, point)) for point in area_uv))
        if key != self._area_key:
            rows, columns = shape
            corners = np.asarray(
                [[u * columns, v * rows] for u, v in area_uv], dtype=np.float32
            ).round()
            self._area_key = key
            # 四隅が潰れて面積が無いと全部が外になり、水が出たそばから消える。
            # それなら砂場の指定が無いのと同じに扱うほうが分かりやすい。
            if cv2.contourArea(corners) < 1.0:
                self._area_mask = None
            else:
                mask = np.zeros(shape, dtype=np.uint8)
                cv2.fillPoly(mask, [corners.astype(np.int32)], 1)
                self._area_mask = mask.astype(bool)
        return self._area_mask

    def _source(self, shape, spring_uv, heavy_rain):
        """水源から 1 秒あたりに各マスへ湧く水[mm/秒]。"""
        source = np.zeros(shape, dtype=np.float32)
        if spring_uv is None:
            return source

        rows, columns = shape
        u, v = spring_uv
        center_row = v * rows
        center_column = u * columns

        row_index, column_index = np.mgrid[0:rows, 0:columns]
        radius = config.FLOOD_SPRING_RADIUS
        inside = (row_index + 0.5 - center_row) ** 2 + (
            column_index + 0.5 - center_column
        ) ** 2 <= radius**2
        if not inside.any():
            return source

        flow = config.FLOOD_SPRING_FLOW
        if heavy_rain:
            flow *= config.FLOOD_RAIN_FACTOR
        # 湧く量はマス数で割る。半径や解像度を変えても全体の水量が変わらない。
        source[inside] = flow / inside.sum()
        return source

    def _step(self, terrain, sink, source):
        """水を 1 刻み流す。"""
        dt = config.FLOOD_STEP_S
        water = self.water
        flux = self._flux

        water += source * dt

        # 隣との水面の差で流れを加速させる。端の外は自分と同じ高さとみなし、
        # 差が 0 になるので外へは流れない（縁は sink で消える）。
        surface = terrain + water
        padded = np.pad(surface, 1, mode="edge")
        neighbours = (
            padded[1:-1, 2:],  # 右
            padded[1:-1, :-2],  # 左
            padded[2:, 1:-1],  # 下
            padded[:-2, 1:-1],  # 上
        )
        # 勢いには水深を掛ける。浅い水ほど流れにくくしないと、斜面の水が
        # 薄い膜に広がって見えなくなる。深い水では上限で止める（振動を防ぐ）。
        reach = np.minimum(water, config.FLOOD_DEPTH_CAP_MM)
        for direction, neighbour in enumerate(neighbours):
            flow = flux[direction]
            flow *= config.FLOOD_FRICTION
            flow += dt * config.FLOOD_GRAVITY * reach * (surface - neighbour)
            np.maximum(flow, 0.0, out=flow)

        # 持っている水より多くは出さない。これを省くと水深が負になり、
        # 水が湧いて増え続けて最後は発散する。
        outgoing = flux.sum(axis=0) * dt
        scale = np.ones_like(water)
        np.divide(water, outgoing, out=scale, where=outgoing > water)
        flux *= scale

        incoming = np.zeros_like(water)
        incoming[:, 1:] += flux[0][:, :-1]
        incoming[:, :-1] += flux[1][:, 1:]
        incoming[1:, :] += flux[2][:-1, :]
        incoming[:-1, :] += flux[3][1:, :]

        water += dt * (incoming - flux.sum(axis=0))
        water -= config.FLOOD_SOAK_MM_PER_S * dt
        np.maximum(water, 0.0, out=water)

        water[sink] = 0.0
        flux[:, sink] = 0.0

    def _visible(self, shape):
        """水として描くマスの水深を、入力の大きさへ戻して返す。

        出るときと消えるときのしきい値をずらす。1 つのしきい値だと、ちょうど
        その深さの水がセンサの揺れで点滅する（等高線の帯を保つのと同じ考え）。
        """
        water = self.water
        appear = water > config.FLOOD_SHOW_MM
        if self._shown is not None and self._shown.shape == water.shape:
            appear |= self._shown & (water > config.FLOOD_HIDE_MM)
        self._shown = appear

        visible = np.where(appear, water, 0.0).astype(np.float32)
        return cv2.resize(visible, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
