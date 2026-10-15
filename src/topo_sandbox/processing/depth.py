"""深度フレーム（ミリメートル）の前処理。

Kinect が返す深度はミリメートル単位の整数で、測定できなかった画素は 0 になる。
彩色や表示へ渡す前に、ここで欠測を埋めて表示用の階調へ落とす。
"""

import cv2
import numpy as np

from .. import config

#: 測定できなかった画素を表す値
INVALID_DEPTH = 0

#: 距離変換の近似マスクの大きさ。3 が最も軽い近似で、穴埋めの範囲判定には十分。
_DISTANCE_MASK_SIZE = 3


def fill_invalid(depth_mm, reference_plane=None):
    """欠測画素を、いちばん近い有効画素の深度で埋める。

    Kinect は赤外線が届かない箇所（物陰、吸収しやすい素材、測定範囲外）で 0 を
    返す。0 をそのまま高さへ換算すると非常に高い突起として扱われ、周囲まで
    巻き込んで色が乱れる。

    埋める値を**周囲から**取るのが要点。フレーム全体の代表値（中央値など）で
    埋めると、視野に入っている砂場の外の床に引きずられて砂面より遠い値になり、
    **欠測がそのまま窪地になる。DEM ではそこが水面として出る。**
    欠測がまとまって出るのは (1) 盛った山の陰と (2) 山の頂上がセンサの最短測定
    距離（既定 800mm）より近づいたときで、どちらも本当の砂面は周囲と同じか
    それより高いため、周囲の値で埋めれば窪地にはならない。

    最も近い 1 画素を使うので、ちらついて飛んだ画素を埋めても高さが偏らない。
    近傍の最小値（最も高い面）で埋めると、散らばった欠測が数 mm ずつ
    持ち上がって斑に見える。

    :data:`config.FILL_NEIGHBORHOOD_PX` より遠くにしか有効画素が無い大きな穴は、
    基準面（無ければ有効画素の中央値）で埋める。

    Args:
        depth_mm: 深度[mm] (H, W)。0 が欠測。
        reference_plane: :class:`~topo_sandbox.processing.plane.ReferencePlane`。
            大きな穴を埋めるのに使う。省略すると中央値で埋める。

    Returns:
        欠測を埋めた float32 の深度[mm] (H, W)。
        有効画素が 1 つも無ければ全面を 0 で返す。
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    invalid = depth == INVALID_DEPTH

    # 埋めるものが無ければ入力をそのまま返す（呼び出し側は書き換えない）。
    # 以降の処理も欠測画素の数に比例させ、全画素を何度も走査しないように
    # してある。毎フレーム通る道なので、ここは軽さを優先する。
    if not invalid.any():
        return depth
    if invalid.all():
        return np.zeros_like(depth)

    valid = ~invalid

    # 距離変換は「0 でない画素から、最も近い 0 画素まで」の距離を返す。
    # 欠測を 1、有効画素を 0 にして渡すと、欠測ごとに最も近い有効画素までの
    # 距離と、その画素の番号（ラベル）が得られる。
    distance, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8),
        cv2.DIST_L2,
        _DISTANCE_MASK_SIZE,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    # ラベルから深度を引く表を作る。ラベルの振り方に依存しないよう、
    # 有効画素の位置から対応を読み取る。
    table = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    table[labels[valid]] = depth[valid]
    patch = table[labels[invalid]]

    # 近くに有効画素が無い大きな穴。センサが覆われたときや、光を吸収しやすい
    # 物を砂場に置いたときに出る。周囲から埋めようがないので面で埋める。
    far = distance[invalid] > config.FILL_NEIGHBORHOOD_PX
    if far.any():
        if reference_plane is not None:
            base = reference_plane.depth_at(depth.shape)[invalid][far]
        else:
            base = np.median(depth[valid])
        patch[far] = base

    filled = depth.copy()
    filled[invalid] = patch
    return filled


def preprocess(depth_mm, reference_plane=None):
    """欠測を埋め、処理解像度へ縮小して平滑化する。

    縮小と平滑化はミリメートルのまま行う。

    Args:
        depth_mm: センサ解像度の深度[mm] (480, 640)。
        reference_plane: :class:`~topo_sandbox.processing.plane.ReferencePlane`。
            :func:`fill_invalid` へそのまま渡す。

    Returns:
        処理解像度の深度[mm] (240, 320) float32。
    """
    depth = fill_invalid(depth_mm, reference_plane)
    depth = cv2.resize(depth, dsize=config.PROC_SIZE)
    return cv2.blur(depth, config.BLUR_KERNEL)


class TemporalStabilizer:
    """静止しているところの揺れだけを、時間方向に均す。

    Kinect の深度は砂場が静止していてもフレームごとに揺れる。空間方向の
    平滑化（:data:`config.BLUR_KERNEL`）のあとでも 0.2mm ほど残り、緩い斜面では
    これだけで等高線や標高帯の境目が 1 画素ぶん左右に動く。何もしていないのに
    線が細かく踊って見えるのはこのため。

    前のフレームと混ぜれば揺れは減るが、そのまま混ぜると手や崩した砂まで
    尾を引く。**大きく変わった画素は混ぜずに新しい値を採る**ことで、
    止まっているところだけを静める。

    1 つのインスタンスを使い回して状態を持つ。`Renderer` が 1 つ持ち、
    ワーカスレッドから順番に呼ぶ（同時に 2 つ走らないことは `app._tick` が
    保証している）。
    """

    def __init__(self, smoothing=None, motion_mm=None):
        """
        Args:
            smoothing: 新しいフレームの重み。1.0 で均さない。
            motion_mm: これを超えて変わった画素は均さない[mm]。
        """
        self.smoothing = config.DEPTH_SMOOTHING if smoothing is None else smoothing
        self.motion_mm = config.DEPTH_MOTION_MM if motion_mm is None else motion_mm
        self._previous = None

    def reset(self):
        """溜めた状態を捨てる。次のフレームはそのまま通す。"""
        self._previous = None

    def apply(self, depth_mm):
        """深度[mm]を均して返す。

        Args:
            depth_mm: 深度[mm] (H, W)。

        Returns:
            均した深度[mm] (H, W) float32。
        """
        depth = np.asarray(depth_mm, dtype=np.float32)
        previous = self._previous

        # 解像度が変わったら比べようがない。最初のフレームも同じ扱い。
        if self.smoothing >= 1.0 or previous is None or previous.shape != depth.shape:
            self._previous = depth
            return depth

        blended = previous + (depth - previous) * self.smoothing

        # 本物の変化には遅れない。揺れの何倍も大きい変化だけを通す。
        moved = np.abs(depth - previous) > self.motion_mm
        stabilized = np.where(moved, depth, blended).astype(np.float32)

        self._previous = stabilized
        return stabilized


class HandHold:
    """急に高くなったところを、落ち着くまで前の地形のまま保つ。

    砂を動かす手は砂面より高いので、そのまま描くと手の形の山が出る。
    手と盛った砂はどちらも「高くなった」としか見えないが、**手は動き続け、
    砂は置かれたら動かない**。そこで高くなったところは、新しい高さがしばらく
    動かなくなるまで採らずに前の地形を保つ。手の上には下の砂の色が映る。

    低くなる変化（掘った、手を抜いた）は待たずに通す。掘った結果まで
    遅らせる理由は無く、手を抜いたあとに手の跡が残るのも防げる。

    比べる相手は前のフレームではなく**保っている地形**。前のフレームと
    比べると、手を砂へ下ろす動きが「低くなった」に見え、掘ったのと同じく
    待たずに通ってしまう。

    時間はフレーム数ではなく時計で測る（さざ波と同じ理由。`app._tick` は
    間に合わないフレームを捨てる）。

    前のフレームを持つので :class:`TemporalStabilizer` と同じく `Renderer` が
    1 つ抱え、ワーカスレッドから順番に呼ぶ。
    """

    def __init__(self, rise_mm=None, still_mm=None, settle_s=None, margin_px=None):
        """
        Args:
            rise_mm: これを超えて高くなったら手かもしれないとみなす[mm]。
            still_mm: 落ち着いたとみなす動きの幅[mm]。
            settle_s: 落ち着くまで待つ時間[秒]。0 なら待たない（素通し）。
            margin_px: 手とみなした範囲を広げる幅[画素]。
        """
        self.rise_mm = config.HAND_RISE_MM if rise_mm is None else rise_mm
        self.still_mm = config.HAND_STILL_MM if still_mm is None else still_mm
        self.settle_s = config.HAND_SETTLE_S if settle_s is None else settle_s
        self.margin_px = config.HAND_MARGIN_PX if margin_px is None else margin_px
        self.reset()

    def reset(self):
        """溜めた状態を捨てる。次のフレームはそのまま通す。"""
        self._held = None  #: 描いている地形の深度[mm]
        self._candidate = None  #: 落ち着くのを待っている深度[mm]
        self._since = None  #: 待ち始めた時刻[秒]。待っていない画素は NaN

    def apply(self, depth_mm, now_s):
        """手を除いた深度[mm]を返す。

        Args:
            depth_mm: 深度[mm] (H, W)。欠測は埋めてあること。
            now_s: 今の時刻[秒]。

        Returns:
            深度[mm] (H, W) float32。
        """
        depth = np.asarray(depth_mm, dtype=np.float32)
        held = self._held

        if self.settle_s <= 0:
            return depth

        # 最初のフレームと、解像度が変わったとき（再生と実機の切り替え）は比べようがない。
        if held is None or held.shape != depth.shape:
            self._held = depth.copy()
            self._candidate = depth.copy()
            self._since = np.full(depth.shape, np.nan)
            return depth

        # 深度はセンサからの距離なので、高くなると小さくなる。
        rise = held - depth
        suspect = rise > self.rise_mm
        if self.margin_px > 0 and suspect.any():
            # 手の縁は平滑化で砂面へなだらかにつながり、しきい値を下回る。
            # 縁だけ通すと手の輪郭が土手として残るので、範囲を広げて保つ。
            size = 2 * self.margin_px + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            suspect = cv2.dilate(suspect.astype(np.uint8), kernel).astype(bool)
        # 低くなったところは、手の近くでも待たない（掘った結果は遅らせない）。
        hold = suspect & (rise > 0)

        candidate = self._candidate
        since = self._since
        restart = hold & (np.isnan(since) | (np.abs(depth - candidate) > self.still_mm))
        candidate[restart] = depth[restart]
        since[restart] = now_s

        waited = now_s - since
        # 時計が戻ったら待たない。待ち始めた時刻より前に戻ると、いつまでも
        # 落ち着かずに古い地形が残り続ける。
        settled = hold & ~restart & ((waited >= self.settle_s) | (waited < 0))
        accept = ~hold | settled

        since[accept] = np.nan
        self._held = np.where(accept, depth, held).astype(np.float32)
        return self._held


def to_display(depth_mm, span_mm=None):
    """深度[mm]を表示用の 8bit グレースケールへ落とす。

    中央値を中心とした一定幅の窓を 0〜255 へ割り当て、窓の外は端の値へ
    張り付かせる。中心を中央値に取るのは、手をかざしたときのような外れ値で
    階調全体が圧縮されるのを避けるため。

    Args:
        depth_mm: 深度[mm] (H, W)。
        span_mm: 階調に割り当てる深度の幅[mm]。省略時は設定値。

    Returns:
        8bit グレースケール (H, W)。値が大きいほど遠い。
    """
    span = span_mm or config.DEPTH_DISPLAY_SPAN_MM
    depth = np.asarray(depth_mm, dtype=np.float32)

    center = float(np.median(depth))
    near = center - span / 2.0

    scaled = (depth - near) / span * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)
