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


def _wave_surface(rows, columns, elapsed_s):
    """水面の起伏と傾きを作る。

    波長と向きの違う波を重ねる。1 本だけだと縞模様にしか見えない。
    高さそのものは白波の明滅にしか使わないので、光の当たり方を決める
    傾きのほうを主に計算している。

    Args:
        rows: 対象画素の行番号 (N,)。
        columns: 対象画素の列番号 (N,)。
        elapsed_s: 表示を始めてからの経過秒。

    Returns:
        (elevation, slope_col, slope_row)。いずれも (N,)。elevation は
        -1〜1 の波の高さ、slope_* は水面の傾き（列方向・行方向）。
    """
    elevation = np.zeros(rows.shape, dtype=np.float32)
    slope_col = np.zeros(rows.shape, dtype=np.float32)
    slope_row = np.zeros(rows.shape, dtype=np.float32)

    for length_px, direction_deg, period_s, steepness in config.WATER_WAVES:
        direction = np.deg2rad(direction_deg)
        along_col = np.cos(direction)
        along_row = np.sin(direction)

        wave_number = 2.0 * np.pi / length_px
        phase = wave_number * (columns * along_col + rows * along_row)
        phase -= 2.0 * np.pi * elapsed_s / period_s

        elevation += np.sin(phase)

        # 傾きは進む向きへ射影する。sin の微分なので cos になる。
        tilt = np.cos(phase) * steepness * config.WATER_WAVE_AMPLITUDE
        slope_col += tilt * along_col
        slope_row += tilt * along_row

    # 振幅はここでまとめて掛ける。傾きだけに掛けて高さに掛け忘れると、
    # 振幅 0 にしても白波だけが動き続ける。
    elevation *= config.WATER_WAVE_AMPLITUDE / max(len(config.WATER_WAVES), 1)
    return elevation, slope_col, slope_row


def _water_shading(slope_col, slope_row):
    """水面の傾きから、拡散光と光源の映り込みを求める。

    光源は陸の陰影起伏と同じ向きにしてある。別々の向きにすると、同じ砂場なのに
    山の影と水の照り返しが食い違い、かえって嘘くさく見える。

    法線を (slope_col, slope_row, 1) と置けるのは :func:`hillshade` と同じ理由で、
    ``_to_view`` の左右反転と画像の y の向きが打ち消し合うため。

    Args:
        slope_col: 水面の傾き（列方向）。
        slope_row: 水面の傾き（行方向）。

    Returns:
        (diffuse, specular)。どちらも 0〜1。
    """
    length = np.sqrt(slope_col**2 + slope_row**2 + 1.0)

    azimuth = np.deg2rad(360.0 - config.HILLSHADE_AZIMUTH_DEG + 90.0)
    altitude = np.deg2rad(config.HILLSHADE_ALTITUDE_DEG)
    light = np.array(
        [
            np.cos(altitude) * np.cos(azimuth),
            np.cos(altitude) * np.sin(azimuth),
            np.sin(altitude),
        ],
        dtype=np.float32,
    )

    # 砂場は真上から見るので視線は (0, 0, 1)。光源との中間ベクトルが
    # 法線と一致したところが最も強く光る（Blinn-Phong）。
    halfway = light + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    halfway /= np.linalg.norm(halfway)

    diffuse = (slope_col * light[0] + slope_row * light[1] + light[2]) / length
    mirror = (slope_col * halfway[0] + slope_row * halfway[1] + halfway[2]) / length

    diffuse = np.clip(diffuse, 0.0, 1.0)
    specular = np.clip(mirror, 0.0, 1.0) ** config.WATER_SHININESS
    return diffuse, specular


def _water_color(height_mm, rows, columns, elapsed_s):
    """水面下の画素を、深さと波の当たり方に応じた色へ変換する。

    浅いほど明るい水色、深いほど濃紺。そこへ波の傾きから求めた拡散光と
    光源の映り込みを重ね、水際には白波を置く。

    明るさを一様に振るだけでは「青く塗られた窪み」にしか見えない。
    水らしく見えるのは、細かな波が光源を映してきらめくところと、
    水際で白く砕けるところなので、その 2 つを出している。

    Args:
        height_mm: 水面下の画素の高さ[mm] (N,)。
        rows: その画素の行番号 (N,)。
        columns: その画素の列番号 (N,)。
        elapsed_s: 表示を始めてからの経過秒。

    Returns:
        RGB (N, 3) float32。
    """
    # 念のため 0 で止める。負のまま exp に渡すと発散して警告が出る。
    depth_below = np.clip(config.WATER_LEVEL_MM - height_mm, 0.0, None)

    span = config.WATER_LEVEL_MM - config.WATER_DEEP_MM
    ratio = np.clip(depth_below / span, 0.0, 1.0)

    shallow = np.asarray(config.WATER_SHALLOW_COLOR, dtype=np.float32)
    deep = np.asarray(config.WATER_DEEP_COLOR, dtype=np.float32)
    color = shallow + (deep - shallow) * ratio[..., None]

    elevation, slope_col, slope_row = _wave_surface(rows, columns, elapsed_s)
    diffuse, specular = _water_shading(slope_col, slope_row)

    brightness = config.WATER_AMBIENT + config.WATER_DIFFUSE * diffuse
    color = color * brightness[..., None]

    glint = np.asarray(config.WATER_SPECULAR_COLOR, dtype=np.float32)
    color = color + glint * (config.WATER_SPECULAR * specular)[..., None]

    if config.WATER_SURF_DEPTH_MM > 0.0:
        # 水際ほど強く、波の山が来たときだけ白く砕ける。
        nearness = np.exp(-depth_below / config.WATER_SURF_DEPTH_MM)
        crest = np.clip(elevation, 0.0, 1.0)
        foam = nearness * crest * config.WATER_SURF_STRENGTH
        color = color + (255.0 - color) * foam[..., None]

    return color


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
    if under_water.any():
        # 水面の計算は水のある画素だけで行う。砂場の大半は陸なので、
        # 全画素ぶん計算すると 3 倍以上の時間がかかる。
        rows, columns = np.nonzero(under_water)
        color[under_water] = _water_color(height[under_water], rows, columns, elapsed_s)

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
