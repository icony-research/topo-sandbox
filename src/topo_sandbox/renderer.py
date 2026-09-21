"""1 フレーム分の描画パイプライン。

深度フレームを受け取り、表示モードに応じた投影用の画像を返す。
GUI からは切り離してあるため、Tk を起動せずに検証できる。
"""

import enum
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from .processing import coloring, depth, overlays, pointcloud, rivers


class ViewMode(enum.Enum):
    """表示モード。`v` キーでこの順に巡回する。"""

    COLORING = 0  #: 傾斜による彩色（本命）
    DEPTH = 1  #: 深度をそのまま白黒で
    DEM = 2  #: 標高による色分けと水面
    EDGE = 3  #: 傾斜の変化線


#: `v` キーで巡回する順序
VIEW_MODE_ORDER = [
    ViewMode.COLORING,
    ViewMode.DEPTH,
    ViewMode.DEM,
    ViewMode.EDGE,
]


def default_projector_quad():
    """投影像側の四隅の既定値。画面全体を砂場とみなす。

    並びは :attr:`RenderSettings.area_positions` と同じく
    左上から反時計回り。
    """
    width, height = config.VIEW_SIZE
    return [[0, 0], [0, height], [width, height], [width, 0]]


class MappingMode(enum.Enum):
    """投影範囲の扱い。`j` キーで切り替える。"""

    NORMAL = 0  #: フレーム全体をそのまま表示
    PERSPECTIVE = 1  #: 指定した四隅を画面全体へ引き伸ばす


@dataclass
class RenderSettings:
    """現場で調整する値。キー操作で書き換わる。"""

    view_mode: ViewMode = ViewMode.COLORING
    mapping_mode: MappingMode = MappingMode.NORMAL
    z_scale: float = config.Z_SCALE_INITIAL
    color_sensitivity: float = config.COLOR_SENSITIVITY_INITIAL
    show_contour: bool = False
    #: 川（流量）を重ねるか。r キーで切り替える。
    show_rivers: bool = False
    #: 水面の高さ[mm]。n / m キーで上下し、Shift + N で初期値へ戻す。
    water_level_mm: float = config.WATER_LEVEL_MM
    #: センサ側の砂場の四隅。左上から反時計回りに 4 点。
    area_positions: list = field(default_factory=list)
    #: 投影像のどこが砂場かを表す四隅。既定は画面全体。
    projector_positions: list = field(default_factory=lambda: default_projector_quad())
    #: 設営時に取得した基準面。未取得なら None。
    reference_plane: object = None

    @property
    def has_area(self):
        return len(self.area_positions) == 4

    def reset_water_level(self):
        self.water_level_mm = config.WATER_LEVEL_MM

    def reset_projector_quad(self):
        self.projector_positions = default_projector_quad()


class Renderer:
    """深度フレームから投影用の画像を作る。"""

    def __init__(self, clock=time.monotonic):
        """
        Args:
            clock: 秒を返す関数。水面のさざ波を進めるのに使う。

        さざ波をフレーム数ではなく時計で進めるのは、負荷で処理が間に合わない
        フレームを `app._tick` が捨てるため。フレーム数で数えると、混雑した
        ときだけ波がゆっくりになる。試験では固定値を返す関数を渡す。
        """
        self._clock = clock
        self._started_at = clock()

    def _elapsed(self):
        """表示を始めてからの経過秒。"""
        return self._clock() - self._started_at

    # ------------------------------------------------------------------
    def render(self, depth_frame, settings):
        """深度フレーム 1 枚を表示用の画像へ変換する。

        Args:
            depth_frame: 深度[mm] (480, 640)。欠測は 0。
            settings: :class:`RenderSettings`。

        Returns:
            表示用の画像。彩色系は RGB (600, 800, 3)、深度表示のみ
            グレースケール (600, 800)。
        """
        depth_mm = depth.preprocess(depth_frame)

        # 基準面からの高さ[mm]。基準面があればセンサの傾きも打ち消される。
        height_mm = pointcloud.heights_from_depth(depth_mm, settings.reference_plane)

        # 画面に出す深度・DEM・等高線は 8bit の階調を使う。
        # 高さの符号を反転して「大きいほど低い」に揃える。
        display = depth.to_display(-height_mm)

        mode = settings.view_mode
        if mode is ViewMode.COLORING:
            return self._render_coloring(height_mm, display, settings)
        if mode is ViewMode.DEM:
            return self._render_dem(height_mm, display, settings)
        if mode is ViewMode.DEPTH:
            return self._render_depth(display, settings)
        if mode is ViewMode.EDGE:
            return self._render_edge(height_mm, settings)
        raise ValueError(f"未知の表示モード: {mode!r}")

    # ------------------------------------------------------------------
    def _to_view(self, image):
        """処理解像度の画像を投影解像度へ拡大し、左右反転する。

        左右反転するのは、プロジェクタが砂場へ投影した像とセンサから見た像が
        鏡像の関係になるため。
        """
        image = cv2.resize(image, dsize=config.VIEW_SIZE)
        return cv2.flip(image, 1)

    def _perspective_matrix(self, settings):
        """センサ側の四隅を、投影像側の四隅へ写す行列を作る。

        行き先を画面全体に固定していると、プロジェクタの投影範囲が砂場と
        ぴったり一致するように物理的に据える必要がある。行き先を可変に
        しておけば、投影像のどこが砂場かを指定するだけで済む。
        """
        source = np.asarray(settings.area_positions, dtype=np.float32)
        target = np.asarray(settings.projector_positions, dtype=np.float32)
        return cv2.getPerspectiveTransform(source, target)

    def _warp(self, image, settings):
        try:
            matrix = self._perspective_matrix(settings)
        except cv2.error:
            # 四隅が一直線に並ぶなどで行列が作れないとき。
            # 実演中に落とさないよう、変換せずそのまま返す。
            return image
        return cv2.warpPerspective(image, matrix, config.VIEW_SIZE)

    def _to_projection(self, image, settings):
        """投影解像度へ拡大し、必要なら射影変換まで済ませる。"""
        image = self._to_view(image)
        if self._use_perspective(settings):
            image = self._warp(image, settings)
        return image

    def _use_perspective(self, settings):
        """射影変換を行うか。四隅が未指定なら行わない。

        四隅が 4 点そろっていない状態で cv2.getPerspectiveTransform を
        呼ぶと例外になる。実演中に落とさないためここで防ぐ。
        """
        return settings.mapping_mode is MappingMode.PERSPECTIVE and settings.has_area

    def _colorize(self, height_mm, settings):
        image = coloring.colorize_by_slope(height_mm, settings.z_scale, settings.color_sensitivity)
        return self._to_view(image)

    # ------------------------------------------------------------------
    def _render_coloring(self, height_mm, display, settings):
        colored = self._colorize(height_mm, settings)
        depth_view = self._to_view(display)

        if self._use_perspective(settings):
            depth_view = self._warp(depth_view, settings)
            colored = self._warp(colored, settings)

        if settings.show_contour:
            colored = overlays.draw_contours(depth_view, colored)
        return colored

    def _render_dem(self, height_mm, display, settings):
        depth_view = self._to_view(display)

        if self._use_perspective(settings):
            depth_view = self._warp(depth_view, settings)

        if settings.reference_plane is None:
            # 基準面が無いと高さの原点がフレームの中央値になり、砂を動かすたびに
            # 水位が漂ってしまう。水面は出さず、従来どおりフレーム内で正規化する。
            dem = overlays.dem_color(depth_view)

            # 流量は絶対的な高さを要らない（下る向きだけで決まる）ので、
            # 基準面が無くても川は出せる。こちらは投影解像度で混ぜるしかない。
            if settings.show_rivers:
                strength = self._to_projection(rivers.river_strength(height_mm), settings)
                dem = overlays.draw_rivers(dem, strength)
        else:
            terrain = overlays.terrain_color(height_mm, self._elapsed(), settings.water_level_mm)

            dry = height_mm >= settings.water_level_mm
            if settings.show_rivers and dry.any():
                # 水面より下の川は描かない。湖や海に入った川は見えなくなる。
                # 全部沈んでいるなら流量の計算そのものが要らない。
                strength = np.where(dry, rivers.river_strength(height_mm), 0.0)

                # 川は拡大する前に混ぜる。投影解像度で混ぜると触る画素が
                # 6 倍になり、それだけで 1 フレームの予算の半分を使う。
                terrain = overlays.draw_rivers(terrain, strength)

            dem = self._to_projection(terrain, settings)

        if settings.show_contour:
            dem = overlays.draw_contours(depth_view, dem)
        return dem

    def _render_depth(self, display, settings):
        depth_view = self._to_view(display)
        depth_view = cv2.bitwise_not(depth_view)

        if self._use_perspective(settings):
            depth_view = self._warp(depth_view, settings)

        if settings.show_contour:
            depth_view = overlays.draw_contours(depth_view, depth_view)
        return depth_view

    def _render_edge(self, height_mm, settings):
        colored = self._colorize(height_mm, settings)

        if self._use_perspective(settings):
            colored = self._warp(colored, settings)

        return overlays.edge_lines(colored)
