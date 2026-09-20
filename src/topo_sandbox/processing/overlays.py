"""彩色以外の表示モードと、重ね描きする要素。

DEM（標高による色分け）、エッジ抽出、等高線。
いずれも numpy と OpenCV だけで完結するため、GPU の無い環境でも検証できる。
"""

import cv2
import numpy as np

from .. import config

#: 最低標高に割り当てる色相。OpenCV の HSV は色相を 0-255 で表すため、
#: 240 度（青）を 255 段階に換算している。
_MIN_ELEVATION_HUE = 240 / 360 * 255

#: エッジとして抽出する色相の帯。平地と斜面の境目付近にあたる。
_EDGE_HUE_RANGE = ((35, 10, 10), (40, 255, 255))


def dem_color(depth_image):
    """標高に応じて色分けした画像を返す（DEM 表示）。

    フレーム内の最低標高を青(色相 240 度)、最高標高を赤(色相 0 度)として
    正規化する。フレームごとに正規化するため、砂を盛ると全体の色が変わる。

    Args:
        depth_image: 8bit の深度画像 (H, W)。

    Returns:
        RGB 画像 (H, W, 3) uint8。
    """
    image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)

    max_value = np.max(depth_image)
    min_value = np.min(depth_image)

    # 全画素が同じ値だとゼロ除算になる
    span = max_value - min_value
    if span == 0:
        return np.zeros_like(image)

    image = (image - min_value) / span * _MIN_ELEVATION_HUE

    # 彩度と明度は最大に固定し、色相だけで標高を表す
    image[:, :, 1] = 255
    image[:, :, 2] = 255

    image = image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB_FULL)


def edge_lines(coloring_image):
    """彩色画像から、傾斜が変化する帯だけを抜き出す（エッジ表示）。

    Args:
        coloring_image: :func:`colorize_by_slope` の出力と同じ RGB 画像。

    Returns:
        RGB 画像 (H, W, 3) uint8。白い部分がエッジ。
    """
    hsv = cv2.cvtColor(coloring_image, cv2.COLOR_RGB2HSV_FULL)
    edges = cv2.inRange(hsv, *_EDGE_HUE_RANGE)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def draw_contours(depth_image, canvas):
    """等高線を描き込む。

    しきい値を変えながら二値化と輪郭抽出を繰り返し、得られた輪郭を
    黒線で重ねる。等高線の間隔は :data:`config.CONTOUR_THRESHOLD_STEP` で決まる。

    Args:
        depth_image: 8bit の深度画像 (H, W)。
        canvas: 描き込み先の画像。

    Returns:
        等高線を描き込んだ画像。
    """
    threshold = config.CONTOUR_THRESHOLD_START

    for _ in range(config.CONTOUR_LEVELS):
        _, binary = cv2.threshold(depth_image, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            points = np.reshape(contour, (len(contour), 2))
            canvas = cv2.polylines(canvas, [points], True, (0, 0, 0))

        threshold += config.CONTOUR_THRESHOLD_STEP

    return canvas
