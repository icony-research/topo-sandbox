"""深度フレーム（ミリメートル）の前処理。

Kinect が返す深度はミリメートル単位の整数で、測定できなかった画素は 0 になる。
彩色や表示へ渡す前に、ここで欠測を埋めて表示用の階調へ落とす。
"""

import cv2
import numpy as np

from .. import config

#: 測定できなかった画素を表す値
INVALID_DEPTH = 0


def fill_invalid(depth_mm):
    """欠測画素を埋める。

    Kinect は赤外線が届かない箇所（物陰、吸収しやすい素材、範囲外）で 0 を返す。
    0 をそのまま高さへ換算すると非常に高い突起として扱われ、周囲まで巻き込んで
    色が乱れる。ここでは有効画素の中央値で埋め、平らな面として扱う。

    Args:
        depth_mm: 深度[mm] (H, W)。0 が欠測。

    Returns:
        欠測を埋めた float32 の深度[mm] (H, W)。
        有効画素が 1 つも無ければ全面を 0 で返す。
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = depth != INVALID_DEPTH

    if not valid.any():
        return np.zeros_like(depth)

    return np.where(valid, depth, np.median(depth[valid])).astype(np.float32)


def preprocess(depth_mm):
    """欠測を埋め、処理解像度へ縮小して平滑化する。

    縮小と平滑化はミリメートルのまま行う。

    Args:
        depth_mm: センサ解像度の深度[mm] (480, 640)。

    Returns:
        処理解像度の深度[mm] (240, 320) float32。
    """
    depth = fill_invalid(depth_mm)
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
