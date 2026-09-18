"""砂場の基準面の推定。

設営のたびにセンサの取り付け角度は変わる。センサが砂場に対して垂直でないと、
平らにならした砂が一様な斜面として着色され、本来の起伏が狭い色域に押し込まれる。

ならした砂に平面をあてはめて基準面とし、以後の高さをその平面からの距離で
測ることで、取り付け角度の違いを打ち消す。

平面は深度画像の座標系で ``depth = a*u + b*v + c`` として表す
（u, v は画像を 0〜1 に正規化した座標）。正規化しているので、
あてはめた解像度と適用する解像度が違っても使える。
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .. import config

#: 平面あてはめで使うフレーム数。30fps なので約 1 秒ぶん。
DEFAULT_FRAME_COUNT = 30

#: 外れ値を落としながら再あてはめする回数
_REFIT_COUNT = 3

#: 残差がこの倍数×標準偏差を超える画素を外れ値として落とす
_OUTLIER_SIGMA = 2.5


@dataclass(frozen=True)
class ReferencePlane:
    """砂場の基準面。"""

    a: float
    b: float
    c: float

    #: あてはめの残差 RMS[mm]。大きいときは砂がならせていない。
    residual_mm: float

    #: 基準面の傾き[度]。センサの取り付け角度のずれにあたる。
    tilt_deg: float

    #: 画像中心での基準面までの距離[mm]
    distance_mm: float

    #: あてはめに使った画素の割合[%]
    coverage: float

    def depth_at(self, shape):
        """指定した大きさの画像上で、基準面の深度[mm]を返す。"""
        height, width = shape
        v, u = np.meshgrid(
            np.linspace(0.0, 1.0, height),
            np.linspace(0.0, 1.0, width),
            indexing="ij",
        )
        return (self.a * u + self.b * v + self.c).astype(np.float32)

    def heights(self, depth_mm):
        """基準面からの高さ[mm]を返す。正が高い（センサに近い）。"""
        depth = np.asarray(depth_mm, dtype=np.float32)
        return self.depth_at(depth.shape) - depth

    def describe(self):
        """画面に出す 1 行の要約。"""
        return (
            f"基準面 {self.distance_mm:.0f}mm  傾き {self.tilt_deg:.1f}°  "
            f"残差 {self.residual_mm:.1f}mm  有効 {self.coverage:.0f}%"
        )


def average_frames(frames):
    """複数フレームを画素ごとの中央値でまとめる。

    深度は 1 フレームごとにちらつくため、時間方向に均してから平面をあてはめる。
    欠測(0)は平均に含めない。一度も測れなかった画素は 0 のままにする。

    Args:
        frames: 深度[mm]の配列の並び。すべて同じ大きさであること。

    Returns:
        画素ごとの中央値 (H, W) float32。欠測は 0。
    """
    stack = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    stack[stack == 0] = np.nan

    # 一度も測れなかった画素は全フレームが NaN になる。想定内なので警告を抑える。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        averaged = np.nanmedian(stack, axis=0)

    return np.nan_to_num(averaged, nan=0.0).astype(np.float32)


def fit_plane(depth_mm):
    """深度画像に平面をあてはめて基準面を求める。

    砂の細かい凹凸や、視野に入り込んだ手などに引きずられないよう、
    残差の大きい画素を落としながら数回あてはめ直す。

    Args:
        depth_mm: 深度[mm] (H, W)。0 は欠測として無視する。

    Returns:
        :class:`ReferencePlane`。

    Raises:
        ValueError: 有効な画素が足りずにあてはめられないとき。
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    height, width = depth.shape

    v, u = np.meshgrid(
        np.linspace(0.0, 1.0, height),
        np.linspace(0.0, 1.0, width),
        indexing="ij",
    )

    valid = depth > 0
    total = depth.size
    if valid.sum() < 100:
        raise ValueError(
            "基準面を求めるための有効な深度が足りません。"
            "センサが砂場を向いているか、距離が近すぎないかを確認してください。"
        )

    u_flat = u[valid]
    v_flat = v[valid]
    d_flat = depth[valid]

    coefficients = None
    residuals = None
    keep = np.ones(d_flat.shape, dtype=bool)

    for _ in range(_REFIT_COUNT):
        design = np.stack([u_flat[keep], v_flat[keep], np.ones(keep.sum())], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, d_flat[keep], rcond=None)

        predicted = coefficients[0] * u_flat + coefficients[1] * v_flat + coefficients[2]
        residuals = d_flat - predicted

        sigma = float(np.std(residuals[keep]))
        if sigma == 0:
            break
        keep = np.abs(residuals) <= _OUTLIER_SIGMA * sigma

        if keep.sum() < 100:
            raise ValueError("砂の面が平らでないため基準面を求められませんでした。")

    a, b, c = (float(value) for value in coefficients)
    residual_mm = float(np.sqrt(np.mean(residuals[keep] ** 2)))

    center_distance = a * 0.5 + b * 0.5 + c
    return ReferencePlane(
        a=a,
        b=b,
        c=c,
        residual_mm=residual_mm,
        tilt_deg=_tilt_degrees(a, b, center_distance),
        distance_mm=float(center_distance),
        coverage=float(keep.sum() / total * 100.0),
    )


def _tilt_degrees(a, b, distance_mm):
    """基準面の傾きを度で求める。

    a, b は「画像の端から端まででの深度の変化量[mm]」なので、
    同じ範囲が実世界で何 mm あるかと比べれば傾きが出る。
    実世界での視野の幅は、ピンホールモデルで ``距離 × 画素数 / 焦点距離``。

    画素数と焦点距離はどちらも解像度に比例するため比は変わらない。
    焦点距離を実測したときの解像度（:data:`config.SENSOR_SIZE`）で
    計算すれば、あてはめた解像度によらず同じ角度が得られる。
    """
    if distance_mm <= 0:
        return 0.0

    focal_x, focal_y = config.SENSOR_FOCAL_PX
    sensor_width, sensor_height = config.SENSOR_SIZE

    span_x_mm = distance_mm * sensor_width / focal_x
    span_y_mm = distance_mm * sensor_height / focal_y

    slope = math.hypot(a / span_x_mm, b / span_y_mm)
    return math.degrees(math.atan(slope))


def capture(frames):
    """フレームの並びから基準面を求める。

    Args:
        frames: 深度[mm]の配列の並び。

    Returns:
        :class:`ReferencePlane`。
    """
    return fit_plane(average_frames(frames))
