"""高さの計算と、点群への変換。"""

import numpy as np


def heights_from_depth(depth_mm, reference_plane=None):
    """深度画像を「基準面からの高さ[mm]」へ変換する。

    基準面が与えられていればそれを使う。センサの取り付け角度の傾きが
    打ち消され、平らにならした砂が本当に平らとして扱われる。

    基準面が無いときはフレームの中央値を基準にする。彩色に効くのは高さの
    勾配だけなので、基準の取り方で色は変わらない。ただし傾きは補正されない。

    Args:
        depth_mm: 深度[mm] (H, W)。値が大きいほど遠い。
        reference_plane: :class:`~topo_sandbox.processing.plane.ReferencePlane`。

    Returns:
        基準面からの高さ[mm] (H, W) float32。正が高い。
    """
    if reference_plane is not None:
        return reference_plane.heights(depth_mm)

    depth = np.asarray(depth_mm, dtype=np.float32)
    return (np.median(depth) - depth).astype(np.float32)


def heights_to_points(height_mm, z_scale):
    """高さの画像を 3 次元点群へ変換する。

    画素の位置をそのまま XY 座標とし、高さを Z とする格子状の点群を作る。
    カメラの内部パラメータは使っていない。砂場を真上から見下ろす用途では、
    これで十分な形状が得られる。

    Z は ``height_mm * z_scale``。高さはミリメートルなので、``z_scale`` は
    「1mm あたり何単位の高さにするか」を意味する。

    Args:
        height_mm: 基準面からの高さ[mm] (H, W)。
        z_scale: 高さの強調倍率。

    Returns:
        (points, center) のタプル。points は (H*W, 3) の float64 配列、
        center は点群の中心の XY 座標。
    """
    height, width = height_mm.shape

    z = np.asarray(height_mm, dtype=np.float64) * z_scale
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")

    points = np.stack([x, y, z], axis=0)
    points = np.transpose(points, (1, 2, 0))
    points = np.reshape(points, (height * width, 3))

    center = [width // 2, height // 2]
    return points, center
