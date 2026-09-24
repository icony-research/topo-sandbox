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


def _wave_surface(rows, columns, elapsed_s, waves, amplitude, warp=None):
    """波を重ねて、うねりの高さと傾きを作る。

    水面のさざ波にも、溶岩のうねりにも使う。波長と向きの違う波を重ねるのは、
    1 本だけだと縞模様にしか見えないため。

    Args:
        rows: 対象画素の行番号 (N,)。
        columns: 対象画素の列番号 (N,)。
        elapsed_s: 表示を始めてからの経過秒。
        waves: (波長[画素], 向き[度], 周期[秒], 振れ幅) の並び。
        amplitude: 全体に掛ける強さ。0 で凪ぐ。
        warp: 位相をゆがめる波 (波長[画素], 向き[度], 周期[秒], 強さ)。
            省略すると素直な正弦波の和になる。

    Returns:
        (elevation, slope_col, slope_row)。いずれも (N,)。elevation は
        -amplitude〜amplitude のうねりの高さ、slope_* はその傾き
        （列方向・行方向）。
    """
    elevation = np.zeros(rows.shape, dtype=np.float32)
    slope_col = np.zeros(rows.shape, dtype=np.float32)
    slope_row = np.zeros(rows.shape, dtype=np.float32)
    weight_total = 0.0

    # 位相をゆがめる波。正弦波の和はそのままだと規則正しい格子になるため、
    # ゆっくりした波で位相をずらして崩す。勾配にも効くので微分も持っておく。
    warp_phase = 0.0
    warp_col = 0.0
    warp_row = 0.0
    if warp is not None:
        length_px, direction_deg, period_s, strength = warp
        direction = np.deg2rad(direction_deg)
        wave_number = 2.0 * np.pi / length_px
        phase = wave_number * (columns * np.cos(direction) + rows * np.sin(direction))
        phase -= 2.0 * np.pi * elapsed_s / period_s

        warp_phase = np.sin(phase) * strength
        gradient = np.cos(phase) * strength * wave_number
        warp_col = gradient * np.cos(direction)
        warp_row = gradient * np.sin(direction)

    for length_px, direction_deg, period_s, steepness in waves:
        direction = np.deg2rad(direction_deg)
        along_col = np.cos(direction)
        along_row = np.sin(direction)

        wave_number = 2.0 * np.pi / length_px
        phase = wave_number * (columns * along_col + rows * along_row) + warp_phase
        phase -= 2.0 * np.pi * elapsed_s / period_s

        # 振幅は高さと傾きの両方に掛ける。片方だけに掛けると、
        # 振幅 0 にしても白波だけが動き続ける。
        scaled = steepness * amplitude
        elevation += np.sin(phase) * scaled

        # 傾きは位相の勾配の向きへ向く。sin の微分なので cos が掛かり、
        # ゆがみのぶんだけ向きがずれる（合成関数の微分）。
        tilt = np.cos(phase) * scaled / wave_number
        slope_col += tilt * (wave_number * along_col + warp_col)
        slope_row += tilt * (wave_number * along_row + warp_row)

        weight_total += steepness

    # 設定が空だったり振れ幅がすべて 0 だったりしてもゼロ除算で止まらないこと。
    if weight_total > 0.0:
        elevation /= weight_total

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


def _water_color(height_mm, rows, columns, elapsed_s, water_level_mm):
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
        water_level_mm: 水面の高さ[mm]。実演中に n / m キーで上下する。

    Returns:
        RGB (N, 3) float32。
    """
    # 念のため 0 で止める。負のまま exp に渡すと発散して警告が出る。
    depth_below = np.clip(water_level_mm - height_mm, 0.0, None)

    # 濃紺へ振り切るまでの幅は水位からの相対。水位を上げても見え方が変わらない。
    ratio = np.clip(depth_below / config.WATER_DEEP_SPAN_MM, 0.0, 1.0)

    shallow = np.asarray(config.WATER_SHALLOW_COLOR, dtype=np.float32)
    deep = np.asarray(config.WATER_DEEP_COLOR, dtype=np.float32)
    color = shallow + (deep - shallow) * ratio[..., None]

    elevation, slope_col, slope_row = _wave_surface(
        rows, columns, elapsed_s, config.WATER_WAVES, config.WATER_WAVE_AMPLITUDE
    )
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


def _lava_color(height_mm, rows, columns, elapsed_s):
    """火口の画素を、溶岩の色へ変換する。

    固まった表面に、溶けた中身が覗く割れ目の網を描く。割れ目は遅いうねりが
    0 を横切るところで、時間とともにゆっくり形を変える。火口の中心へ
    近づくほど全体が明るくなる。

    陰影は掛けない。溶岩は自分で光っているものなので、影がつくと
    ただの赤い岩に見えてしまう。

    Args:
        height_mm: 火口の画素の高さ[mm] (N,)。
        rows: その画素の行番号 (N,)。
        columns: その画素の列番号 (N,)。
        elapsed_s: 表示を始めてからの経過秒。

    Returns:
        RGB (N, 3) float32。
    """
    above = height_mm - config.VOLCANO_HEIGHT_MM
    ratio = np.clip(above / config.VOLCANO_SPAN_MM, 0.0, 1.0)

    # うねりが 0 を横切るところを光らせると、固まった表面の割れ目から
    # 溶けた中身が覗いているように見える。うねりの値をそのまま明るさに
    # すると、正弦波の格子がそのまま出てワッフルのようになる。
    #
    # 向きの違う 2 組の割れ目を重ねる（明るいほうを採る）。1 組だと割れ目が
    # 一方向へ揃って櫛の跡のように見え、割れた岩に見えない。
    cracks = np.zeros(rows.shape, dtype=np.float32)
    for waves, warp in config.LAVA_CHURN_LAYERS:
        churn, _, _ = _wave_surface(rows, columns, elapsed_s, waves, 1.0, warp)
        layer = np.clip(1.0 - np.abs(churn) / config.LAVA_CRACK_WIDTH, 0.0, 1.0)
        cracks = np.maximum(cracks, layer)

    heat = np.clip(cracks + ratio * config.LAVA_CORE_GLOW, 0.0, 1.0)

    # 固まった表面 → 赤熱 → 最も熱いところ、の 3 段をたどる。黒から黄へ
    # 直接つなぐと、溶けた岩ではなく電球のように見える。
    stops = np.asarray(
        [config.LAVA_CRUST_COLOR, config.LAVA_GLOW_COLOR, config.LAVA_MOLTEN_COLOR],
        dtype=np.float32,
    )
    position = heat * (len(stops) - 1)
    lower = np.clip(np.floor(position), 0, len(stops) - 2).astype(np.int32)
    blend = (position - lower)[..., None]
    return stops[lower] * (1.0 - blend) + stops[lower + 1] * blend


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


def terrain_color(height_mm, elapsed_s=0.0, water_level_mm=None):
    """基準面からの絶対高さで、水面・標高帯・火山に塗り分ける（DEM 表示）。

    :func:`dem_color` と違い、フレームごとの正規化をしない。高さを絶対値で
    見るため、砂を動かしても水際や雪線が動かない。基準面が取れているときだけ
    使えることに注意（高さの原点が要る）。

    Args:
        height_mm: 基準面からの高さ[mm] (H, W)。
        elapsed_s: 表示を始めてからの経過秒。水面のさざ波と溶岩のうねりを
            進めるのに使う。省略すると止まった状態になる。
        water_level_mm: 水面の高さ[mm]。省略すると設定の初期値。

    Returns:
        RGB 画像 (H, W, 3) uint8。
    """
    height = np.asarray(height_mm, dtype=np.float32)
    if water_level_mm is None:
        water_level_mm = config.WATER_LEVEL_MM

    # 陰影は陸だけに掛ける。水面に影が出ると水に見えず、溶岩に影が出ると
    # ただの赤い岩に見える。
    color = _land_color(height) * hillshade(height)[..., None]

    # 水面と火山の計算は、その画素だけを集めて行う。砂場の大半はどちらでも
    # ないので、全画素ぶん計算すると DEM の 1 フレームが 2 倍以上に伸びる。
    molten = height >= config.VOLCANO_HEIGHT_MM
    if molten.any():
        rows, columns = np.nonzero(molten)
        color[molten] = _lava_color(height[molten], rows, columns, elapsed_s)

    # 水位を火山より上まで上げれば火口も沈む。あとから塗るので水が勝つ。
    under_water = height < water_level_mm
    if under_water.any():
        rows, columns = np.nonzero(under_water)
        color[under_water] = _water_color(
            height[under_water], rows, columns, elapsed_s, water_level_mm
        )

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


def draw_rivers(canvas, strength):
    """川を描き込む。

    等高線のような線ではなく、川らしさで色を混ぜる。上流ほど薄く、
    水が集まるほど濃くなるので、支流から本流へ太くなっていく様子が出る。

    :func:`draw_contours` と同じく canvas を直接書き換える。

    Args:
        canvas: 描き込み先の RGB 画像 (H, W, 3) uint8。
        strength: 川らしさ 0〜1 (H, W) float32。

    Returns:
        川を重ねた RGB 画像 (H, W, 3) uint8。
    """
    alpha = np.clip(strength, 0.0, 1.0)
    wet = alpha > 0.0

    # 砂場のほとんどは川ではない。全画素ぶん混ぜると投影解像度では
    # 17ms かかり、それだけで 1 フレームの予算の半分を使ってしまう。
    if not wet.any():
        return canvas

    river = np.asarray(config.RIVER_COLOR, dtype=np.float32)
    weight = alpha[wet][:, None]

    blended = canvas[wet] * (1.0 - weight) + river * weight
    canvas[wet] = np.clip(blended, 0, 255).astype(np.uint8)
    return canvas


def contour_interval(height_mm, interval_mm=None):
    """実際に使う等高線の間隔[mm]を返す。

    起伏が広すぎて :data:`config.CONTOUR_LEVELS` 本に収まらないときは、間隔を
    整数倍に広げる。本数で打ち切ると、いちばん高いところ（＝子どもが盛り上げた
    山）にだけ線が出なくなってしまうため。

    Args:
        height_mm: 高さ[mm]。
        interval_mm: 等高線の間隔[mm]。省略すると設定値。

    Returns:
        間隔[mm] float。
    """
    interval = float(interval_mm or config.CONTOUR_INTERVAL_MM)

    height = np.asarray(height_mm, dtype=np.float32)
    span = float(height.max()) - float(height.min())
    if not np.isfinite(span) or span <= 0.0 or interval <= 0.0:
        return interval

    # 上限を超えるぶんだけ広げる。3.2 倍必要なら 4 倍にする。
    crowding = int(np.ceil(span / interval / config.CONTOUR_LEVELS))
    return interval * max(crowding, 1)


class ContourBands:
    """高さを等高線の帯へ落とす。前のフレームの帯を覚えてちらつきを抑える。

    等高線は帯の境目なので、緩い斜面では深度のわずかな揺れで線が 1 画素ぶん
    左右に動く。実測では、**砂を動かしていないのに 1 フレームごとに線の画素の
    100% 以上が入れ替わっていた**（センサの揺れ σ=3mm のとき）。境目を少し
    越えるまで前の帯を保つと、入れ替わりは 3% まで下がる。

    砂を実際に動かせば、境目を越えた時点で線も動く。遅れるのは
    :data:`config.CONTOUR_HYSTERESIS_MM` のぶん（既定 0.5mm）だけ。
    """

    def __init__(self, hysteresis_mm=None):
        """
        Args:
            hysteresis_mm: 帯を保つ余裕[mm]。0 にすると毎フレーム引き直す。
        """
        if hysteresis_mm is None:
            hysteresis_mm = config.CONTOUR_HYSTERESIS_MM
        self.hysteresis_mm = hysteresis_mm
        self._bands = None
        self._interval = None

    def reset(self):
        """覚えている帯を捨てる。"""
        self._bands = None
        self._interval = None

    def of(self, height_mm, interval):
        """高さ[mm]を帯の番号へ落とす。

        Args:
            height_mm: 高さ[mm] (H, W)。
            interval: 等高線の間隔[mm]。

        Returns:
            帯の番号 (H, W) float32。
        """
        height = np.asarray(height_mm, dtype=np.float32)
        bands = np.floor(height / interval)

        previous = self._bands
        usable = (
            self.hysteresis_mm > 0.0
            and previous is not None
            and previous.shape == bands.shape
            and self._interval == interval
        )
        if usable:
            # 前の帯の上下に余裕を付けた範囲にいるあいだは、前の帯を保つ。
            lower = previous * interval - self.hysteresis_mm
            upper = (previous + 1.0) * interval + self.hysteresis_mm
            bands = np.where((height >= lower) & (height <= upper), previous, bands)

        self._bands = bands
        self._interval = interval
        return bands

    def draw(self, height_mm, canvas, interval_mm=None):
        """等高線を描き込む。引数と戻り値は :func:`draw_contours` と同じ。"""
        height = np.asarray(height_mm, dtype=np.float32)
        bands = self.of(height, contour_interval(height, interval_mm))

        # 隣と帯の番号が違うところが等高線になる。しきい値ごとに輪郭を
        # 抽出していたときは 1 フレーム 8ms 掛かっていたが、この方法なら
        # 本数によらず一定で済む。
        on_line = np.zeros(bands.shape, dtype=bool)
        on_line[:, 1:] |= bands[:, 1:] != bands[:, :-1]
        on_line[1:, :] |= bands[1:, :] != bands[:-1, :]

        canvas[on_line] = 0
        return canvas


def draw_contours(height_mm, canvas, interval_mm=None):
    """等高線を描き込む（1 フレームだけの版）。

    高さを一定間隔で区切り、区切りが変わる境目を黒い線にする。

    **8bit の深度画像ではなく高さ[mm]を直接見る。** 表示用の 8bit は中央値を
    中心とした幅 256mm の窓へ押し込んであり、窓の外は端の値に張り付く。そこから
    線を引いていたときは **ある高さより上に等高線が出ず**、しかも窓の中心が
    フレームごとに動くため、線の消える高さが砂を動かすたびに変わっていた。

    続けて描くときは :class:`ContourBands` を使い回すこと。1 枚ごとに引き直すと、
    センサの揺れで線が細かく踊る。

    Args:
        height_mm: 投影解像度の高さ[mm] (H, W)。
        canvas: 描き込み先の画像。白黒でも RGB でもよい。
        interval_mm: 等高線の間隔[mm]。省略すると設定値。

    Returns:
        等高線を描き込んだ画像。
    """
    return ContourBands().draw(height_mm, canvas, interval_mm)
