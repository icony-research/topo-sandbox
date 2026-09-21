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

#: Sobel フィルタ（ksize=3）の出力に掛かっている倍率。真の勾配へ戻すのに使う。
_SOBEL_GAIN = 8.0


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


def _wave_field(shape, elapsed_s):
    """水面のさざ波の明暗を作る。

    波長と向きの違う波を重ねる。1 本だけだと縞模様にしか見えない。

    Args:
        shape: (H, W)。処理解像度の大きさ。
        elapsed_s: 表示を始めてからの経過秒。

    Returns:
        おおむね -1〜1 に収まるさざ波 (H, W) float32。
    """
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    field = np.zeros(shape, dtype=np.float32)

    for length_px, direction_deg, period_s in config.WATER_WAVES:
        direction = np.deg2rad(direction_deg)
        wave_number = 2.0 * np.pi / length_px
        along = columns * np.cos(direction) + rows * np.sin(direction)
        field += np.sin(wave_number * along - 2.0 * np.pi * elapsed_s / period_s)

    return (field / len(config.WATER_WAVES)).astype(np.float32)


def _water_color(height_mm, elapsed_s):
    """水面下の画素を、深さに応じた青へ変換する。

    浅いほど明るい水色、深いほど濃紺。陸のような段彩にせず連続で
    変えているのは、段になっていると水面に見えないため。

    さらにさざ波で明るさを振る。止まった青一色だと、掘った穴が
    水たまりではなく「青く塗られた窪み」に見えてしまう。

    Args:
        height_mm: 基準面からの高さ[mm] (H, W)。
        elapsed_s: 表示を始めてからの経過秒。さざ波を進めるのに使う。

    Returns:
        RGB (H, W, 3) float32。水面より上の画素の値は使われない。
    """
    span = config.WATER_LEVEL_MM - config.WATER_DEEP_MM
    ratio = np.clip((config.WATER_LEVEL_MM - height_mm) / span, 0.0, 1.0)

    shallow = np.asarray(config.WATER_SHALLOW_COLOR, dtype=np.float32)
    deep = np.asarray(config.WATER_DEEP_COLOR, dtype=np.float32)
    color = shallow + (deep - shallow) * ratio[..., None]

    ripple = 1.0 + config.WATER_WAVE_AMPLITUDE * _wave_field(height_mm.shape, elapsed_s)
    return color * ripple[..., None]


def _land_color(height_mm):
    """陸地を標高帯で塗り分ける。

    Args:
        height_mm: 基準面からの高さ[mm] (H, W)。

    Returns:
        RGB (H, W, 3) float32。
    """
    bounds = np.asarray([band[0] for band in config.LAND_BANDS], dtype=np.float32)
    colors = np.asarray([band[1] for band in config.LAND_BANDS], dtype=np.float32)

    # 先頭の帯の下限より低い画素も添字 0 に落ちるが、そこは水面下なので
    # あとで水の色に置き換わる。
    index = np.digitize(height_mm, bounds[1:])
    return colors[index]


def hillshade(height_mm):
    """陰影起伏の明るさ係数を作る。

    地形図の陰影起伏と同じ計算（斜面の向きと光源のなす角）で、盛った砂の
    片側に影を落とす。色分けだけでは平面に見えてしまう起伏が、影がつくと
    山として読めるようになる。

    Args:
        height_mm: 基準面からの高さ[mm] (H, W) float32。

    Returns:
        明るさの倍率 (H, W) float32。1 を中心に、影の側が暗くなる。
    """
    height = np.asarray(height_mm, dtype=np.float32)

    # Sobel の出力は勾配の 8 倍になっているため戻す。
    scale = config.HILLSHADE_Z_FACTOR / _SOBEL_GAIN
    dz_dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3, scale=scale)
    dz_dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3, scale=scale)

    slope = np.arctan(np.hypot(dz_dx, dz_dy))

    # 勾配は配列の (列, 行) 方向で出るが、Renderer._to_view が左右反転してから
    # 投影するため、投影像での東 = 配列の -列方向、北 = 配列の -行方向になる。
    # 斜面の下り方向を投影像の座標で測ると両軸の符号が二重に反転して
    # (dz_dx, dz_dy) そのものに戻る。教科書どおりの式に見えないのはこのため。
    # 向きを取り違えて光が下から当たると、人は山と谷を反転して知覚する。
    aspect = np.arctan2(dz_dy, dz_dx)

    zenith = np.deg2rad(90.0 - config.HILLSHADE_ALTITUDE_DEG)
    azimuth = np.deg2rad(360.0 - config.HILLSHADE_AZIMUTH_DEG + 90.0)

    shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        azimuth - aspect
    )
    shade = np.clip(shade, 0.0, 1.0)

    # 平坦地がそのままの明るさになるよう、平坦時の値で割って正規化する。
    # 正規化しないと全体が cos(天頂角) 倍に暗くなる。
    strength = config.HILLSHADE_STRENGTH
    return (1.0 - strength) + strength * shade / np.cos(zenith)


def terrain_color(height_mm, elapsed_s=0.0):
    """基準面からの絶対高さで、水面と標高帯に塗り分ける（DEM 表示）。

    :func:`dem_color` と違い、フレームごとの正規化をしない。水位を
    絶対値で決めるため、砂を動かしても水際が動かない。基準面が
    取れているときだけ使えることに注意（高さの原点が要る）。

    Args:
        height_mm: 基準面からの高さ[mm] (H, W)。
        elapsed_s: 表示を始めてからの経過秒。水面のさざ波を進めるのに使う。
            省略すると波の止まった状態になる。

    Returns:
        RGB 画像 (H, W, 3) uint8。
    """
    height = np.asarray(height_mm, dtype=np.float32)

    # 陰影は陸だけに掛ける。水面に影が出ると水に見えない。
    color = _land_color(height) * hillshade(height)[..., None]

    under_water = height < config.WATER_LEVEL_MM
    color = np.where(under_water[..., None], _water_color(height, elapsed_s), color)

    return np.clip(color, 0, 255).astype(np.uint8)


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
