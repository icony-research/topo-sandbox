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
