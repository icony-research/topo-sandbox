"""法線ベクトルから傾斜角を求め、配色する処理。

このプログラムの中核。砂場の起伏を「平坦＝赤、急斜面＝青紫」で塗り分ける。

法線推定は Open3D（CPU）、傾斜角の算出と配色テーブル引きは CuPy（GPU）で
行う。点数は 320x240 = 76,800 点あり、ここを CPU で回すと 30fps に
間に合わないため GPU を使っている。
"""

import numpy as np
import open3d as o3d

from .. import config, palette
from .pointcloud import heights_to_points


def estimate_normals(points, center):
    """点群の法線ベクトルを推定する。

    法線の向きは、点群の中心の上空に置いた仮想カメラへ向けて揃える。
    これをしないと隣り合う点で法線が裏返り、色がまだらになる。
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.NORMAL_SEARCH_RADIUS,
            max_nn=config.NORMAL_MAX_NEIGHBORS,
        ),
        fast_normal_computation=False,
    )
    cloud.orient_normals_towards_camera_location(
        camera_location=np.array([center[0], center[1], config.NORMAL_CAMERA_HEIGHT])
    )
    return cloud.normals


def slope_degrees(normals, sensitivity):
    """法線から水平面に対する傾斜角を求め、感度を掛けて添字にする。

    傾斜角 θ は法線 n と鉛直軸のなす角、すなわち ``arccos(|n_z| / |n|)``。
    平坦なら 0 度、垂直な壁なら 90 度になる。

    Args:
        normals: cupy 配列に変換可能な法線の並び (N, 3)。
        sensitivity: 傾斜角に掛ける倍率。大きいほど色の変化が急になる。

    Returns:
        配色テーブルの添字として使う cupy の int32 配列 (N,)。
    """
    import cupy as cp

    normals = cp.asarray(normals)

    squared_sum = cp.sum(cp.square(normals), axis=1)
    vertical = cp.abs(normals[:, 2])
    cosine = vertical / cp.sqrt(squared_sum)

    degrees = cp.rad2deg(cp.arccos(cosine))
    degrees *= sensitivity
    return degrees.astype(cp.int32)


def colorize_by_slope(height_mm, z_scale, sensitivity):
    """高さの画像を傾斜に応じて着色した RGB 画像へ変換する。

    Args:
        height_mm: 基準面からの高さ[mm] (H, W)。欠測が埋まっていること。
        z_scale: 高さの強調倍率。
        sensitivity: 傾斜の色分け感度。

    Returns:
        入力と同じ大きさの RGB 画像 (H, W, 3) uint8。
    """
    import cupy as cp

    points, center = heights_to_points(height_mm, z_scale)
    normals = estimate_normals(points, center)

    degrees = slope_degrees(normals, sensitivity)
    rgb = palette.as_cupy()[degrees]
    rgb = cp.asnumpy(rgb).astype(np.uint8)

    height, width = height_mm.shape
    return np.reshape(rgb, (height, width, 3))
