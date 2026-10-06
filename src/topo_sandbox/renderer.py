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
from .processing import coloring, depth, flood, overlays, pointcloud, rivers
from .recorder import frame_context


class ViewMode(enum.Enum):
    """表示モード。`v` キーでこの順に巡回する。"""

    COLORING = 0  #: 傾斜による彩色
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


def sensor_area_from_view(positions):
    """表示像の上で指定した砂場の四隅を、センサ画像の正規化座標へ戻す。

    エリアの四隅は投影解像度の画面をクリックして決めるが、その画面は
    `Renderer._to_view` が拡大して**左右反転**したあとの像である。
    センサ画像の座標として使うには反転を戻す必要があり、u を 1 から
    引いているのはそのため。符号を「揃える」と左右が入れ替わり、
    砂場の反対側で基準面をあてはめることになる。

    正規化して返すのは、センサ解像度と処理解像度のどちらへでも
    そのまま当てられるようにするため（基準面と同じ考え方）。

    Args:
        positions: 表示像での四隅 [[x, y], ...]（:data:`config.VIEW_SIZE` の画素）。

    Returns:
        正規化座標 (u, v) の四隅の並び。
    """
    width, height = config.VIEW_SIZE
    return [[1.0 - x / width, y / height] for x, y in positions]


def sensor_point_from_screen(point, settings):
    """画面でクリックした 1 点を、センサ画像の正規化座標へ戻す。

    水源の位置を決めるのに使う。射影変換モードでは画面に映っているのは
    四隅を引き伸ばしたあとの像なので、まず変換を逆にたどって表示像の座標へ
    戻し、そこから :func:`sensor_area_from_view` と同じく左右反転を戻す。
    エリアの四隅と違って**投影された砂場そのもの**をクリックするため、
    モードによらず指した場所に水源が置かれる。

    Args:
        point: 画面上の (x, y)（:data:`config.VIEW_SIZE` の画素）。
        settings: :class:`RenderSettings`。

    Returns:
        正規化座標 (u, v)。画面の外や、砂場として写っていない所なら None。
    """
    x, y = float(point[0]), float(point[1])

    area = np.asarray(settings.area_positions, dtype=np.float32)
    # 四隅が潰れていると cv2 は例外を出さずに意味の無い行列を返し、
    # 水源がでたらめな所に置かれる。そのときは変換しない。
    if (
        settings.mapping_mode is MappingMode.PERSPECTIVE
        and settings.has_area
        and cv2.contourArea(area) >= _MIN_CROP_AREA_PX
    ):
        try:
            # 行き先と元を入れ替えれば逆向きの変換になる。
            matrix = cv2.getPerspectiveTransform(
                np.asarray(settings.projector_positions, dtype=np.float32), area
            )
        except cv2.error:
            pass
        else:
            x, y = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), matrix)[0][0]

    ((u, v),) = sensor_area_from_view([[x, y]])
    # 範囲の比較は NaN でも偽になるので、投影枠が潰れていて変換が壊れても弾ける。
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return [float(u), float(v)]


#: 砂場の四隅がこれより狭い面積しか囲んでいなければ、切り取らない[画素^2]。
#: 4 点が 1 点に集まったり一直線に並んだりすると変換が定まらず、
#: cv2 は例外を出さずに真っ黒な結果を返す。黙って 0mm の平面を録るより断るほうがよい。
_MIN_CROP_AREA_PX = 16.0


def crop_to_area(height_mm, area_positions, size=None):
    """砂場の四隅の内側だけを切り出し、長方形の格子へ直す。

    録画（``--record --record-crop``）で使う。センサの視野には砂場の枠や床、
    まわりに立っている人まで入っており、そのまま点群にすると砂場の何倍もの
    広さに床が窪地として広がり、前を横切った人が高い壁として残る。

    行き先の四隅は :func:`default_projector_quad` と同じ並びなので、**画面で
    最初にクリックした角が左上に来る。** エリアの四隅は左右反転したあとの
    表示像で指定するため、出てくるのは投影像と同じ向きの絵になる（センサから
    見たままの並びではない）。切り取ったものを読む側が取り違えないよう、
    meta.json の ``axes`` にもそう書く。

    x と y は砂場の端から端までを等分したもので、ミリメートルではない。
    四隅の間隔は設営のたびに変わるため、ここで長さを決めようがない。

    Args:
        height_mm: 高さ[mm] (H, W)。
        area_positions: 表示像での砂場の四隅 [[x, y], ...]（左上から反時計回りに 4 点）。
        size: 出力の大きさ (幅, 高さ)。省略すると :data:`config.PROC_SIZE`。

    Returns:
        切り出した高さ[mm] (height, width) float32。四隅が一直線に並ぶなどで
        変換を作れないときは None。
    """
    width, height = size or config.PROC_SIZE

    source_height, source_width = np.asarray(height_mm).shape
    source = np.asarray(
        [[u * source_width, v * source_height] for u, v in sensor_area_from_view(area_positions)],
        dtype=np.float32,
    )
    target = np.asarray([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    if cv2.contourArea(source) < _MIN_CROP_AREA_PX:
        return None
    try:
        matrix = cv2.getPerspectiveTransform(source, target)
    except cv2.error:
        return None

    # 端は外側の値で埋める。既定（0 で埋める）のままだと、砂場のいちばん外側の
    # 1 列が高さ 0mm になり、点群にしたときに縁だけ基準面へ落ちた段差として出る。
    return cv2.warpPerspective(
        np.asarray(height_mm, dtype=np.float32),
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )


class MappingMode(enum.Enum):
    """投影範囲の扱い。`j` キーで切り替える。"""

    NORMAL = 0  #: フレーム全体をそのまま表示
    PERSPECTIVE = 1  #: 指定した四隅を画面全体へ引き伸ばす


#: :meth:`RenderSettings.reset_adjustments` が初期値へ戻す項目。
#: 設営ぶん（area_positions / projector_positions / reference_plane / spring_position）は含めない。
_ADJUSTMENT_FIELDS = (
    "view_mode",
    "mapping_mode",
    "z_scale",
    "color_sensitivity",
    "show_contour",
    "show_rivers",
    "river_preset",
    "water_level_mm",
    "show_flood",
    "heavy_rain",
)


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
    #: 川の出やすさ（config.RIVER_PRESETS の添字）。Shift + R で切り替える。
    river_preset: int = config.RIVER_PRESET_INITIAL
    #: 水面の高さ[mm]。n / m キーで上下し、Shift + N で初期値へ戻す。
    water_level_mm: float = config.WATER_LEVEL_MM
    #: 水源から水を流すか。w キーで切り替える。
    show_flood: bool = False
    #: 大雨か（水源の量が増える）。o キーで切り替える。
    heavy_rain: bool = False
    #: 水源の位置。センサ画像の正規化座標 [u, v]。i キーのあと砂場をクリックして決める。
    #: 設営ぶんなので Ctrl + R では消えない。
    spring_position: list = None
    #: センサ側の砂場の四隅。左上から反時計回りに 4 点。
    area_positions: list = field(default_factory=list)
    #: 投影像のどこが砂場かを表す四隅。既定は画面全体。
    projector_positions: list = field(default_factory=lambda: default_projector_quad())
    #: 設営時に取得した基準面。未取得なら None。
    reference_plane: object = None

    @property
    def has_area(self):
        return len(self.area_positions) == 4

    @property
    def river_setting(self):
        """川の出やすさ (表示名, 描き始める流量, 最も濃くなる流量)。

        添字は剰余で丸める。設定ファイルから読んだ値が範囲外でも、実演中に
        添字エラーで止めないため。
        """
        return config.RIVER_PRESETS[self.river_preset % len(config.RIVER_PRESETS)]

    def cycle_river_preset(self):
        """川の出やすさを次の段階へ送る。"""
        self.river_preset = (self.river_preset + 1) % len(config.RIVER_PRESETS)
        return self.river_setting

    def reset_water_level(self):
        self.water_level_mm = config.WATER_LEVEL_MM

    def reset_projector_quad(self):
        self.projector_positions = default_projector_quad()

    def reset_adjustments(self):
        """実演中に触る調整値だけを初期値へ戻す。

        設営ぶん（エリアの四隅・投影枠・基準面・水源）は残す。合わせ直すのに
        時間が掛かるうえ、実演の最中に消えると立て直せないため。
        """
        defaults = RenderSettings()
        for name in _ADJUSTMENT_FIELDS:
            setattr(self, name, getattr(defaults, name))

    def reset_all(self):
        """設営ぶんも含めて、すべて初期値へ戻す。

        設定ファイルには触らない。書き戻すかどうかは Ctrl + S で選ぶ。
        """
        self.reset_adjustments()
        self.area_positions = []
        self.spring_position = None
        self.projector_positions = default_projector_quad()
        self.reference_plane = None


class Renderer:
    """深度フレームから投影用の画像を作る。"""

    def __init__(self, clock=time.monotonic, recorder=None, record_crop=False):
        """
        Args:
            clock: 秒を返す関数。水面のさざ波を進めるのに使う。
            recorder: :class:`~topo_sandbox.recorder.DemRecorder`。
                与えると DEM 表示のあいだの高さ[mm]を書き出す。
                `--record` で起動したときだけ渡され、普段は None。
            record_crop: 書き出すときに砂場の四隅の内側だけを切り出すか
                （`--record-crop`）。

        さざ波をフレーム数ではなく時計で進めるのは、負荷で処理が間に合わない
        フレームを `app._tick` が捨てるため。フレーム数で数えると、混雑した
        ときだけ波がゆっくりになる。試験では固定値を返す関数を渡す。
        """
        self._clock = clock
        self._started_at = clock()
        self.recorder = recorder
        self.record_crop = record_crop

        # どちらも前のフレームを覚えている。センサの揺れで等高線や標高帯の
        # 境目が踊るのを抑えるため。`app._tick` がワーカを 1 つしか走らせない
        # ので、順番に呼ばれることは保証されている。
        self._stabilizer = depth.TemporalStabilizer()
        self._contours = overlays.ContourBands()

        # 水源から流れた水。これも前のフレームの水深を覚えている。
        self._flood = flood.FloodSimulator()

    def _elapsed(self):
        """表示を始めてからの経過秒。"""
        return self._clock() - self._started_at

    def drain_flood(self):
        """水源から流れた水をすべて抜く（W キー）。

        メインスレッドから呼ばれる。ワーカが計算している最中に配列を
        消さないよう、次のフレームの頭で抜く。
        """
        self._flood.request_reset()

    def close(self):
        """抱えているものを片付ける。録画中なら書き残しを書き切る。

        `app._on_close` から呼ばれる。録画していなければ何もしない。
        """
        if self.recorder is not None:
            self.recorder.close()

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
        # 欠測の穴埋めにも基準面を使う。周囲から埋められない大きな穴を
        # 中央値で埋めると、視野に入った砂場の外に引きずられて窪地になる。
        depth_mm = depth.preprocess(depth_frame, settings.reference_plane)

        # 動いていないところの揺れを均す。砂を動かした画素はそのまま通るので、
        # 追従は遅れない。
        depth_mm = self._stabilizer.apply(depth_mm)

        # 基準面からの高さ[mm]。基準面があればセンサの傾きも打ち消される。
        height_mm = pointcloud.heights_from_depth(depth_mm, settings.reference_plane)

        # 画面に出す深度・DEM・等高線は 8bit の階調を使う。
        # 高さの符号を反転して「大きいほど低い」に揃える。
        display = depth.to_display(-height_mm)

        mode = settings.view_mode
        if not self._flood_active(settings):
            # 見えていないあいだは水を持ち越さない。DEM へ戻ったとき、
            # 砂場を作り変えたあとの地形に前の水が残っていると不自然になる。
            self._flood.reset()

        if mode is ViewMode.COLORING:
            return self._render_coloring(height_mm, settings)
        if mode is ViewMode.DEM:
            return self._render_dem(height_mm, display, settings)
        if mode is ViewMode.DEPTH:
            return self._render_depth(height_mm, display, settings)
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
    def _draw_contours(self, height_mm, canvas, settings):
        """等高線を重ねる。

        等高線だけは拡大したあとに描く。線の太さを投影解像度で決めたいのと、
        処理解像度では線が粗すぎるため。高さ[mm]をそのまま投影側へ運ぶのは、
        8bit の表示画像が窓でクリップされており、高く盛ったところに線が
        出なかったため。

        帯は :class:`~topo_sandbox.processing.overlays.ContourBands` が覚えていて、
        センサの揺れで線が 1 画素ぶん踊るのを抑える。毎フレーム引き直すと、
        砂を動かしていなくても線が細かく動く。
        """
        return self._contours.draw(self._to_projection(height_mm, settings), canvas)

    def _render_coloring(self, height_mm, settings):
        colored = self._colorize(height_mm, settings)

        if self._use_perspective(settings):
            colored = self._warp(colored, settings)

        if settings.show_contour:
            colored = self._draw_contours(height_mm, colored, settings)
        return colored

    @staticmethod
    def _flood_active(settings):
        """水源の水を流すか。

        DEM 表示で基準面があるときだけ。基準面が無いと高さの原点がフレームごとに
        動き、水が揺すられて勝手に流れ出す（水位を出さないのと同じ理由）。
        """
        return (
            settings.show_flood
            and settings.view_mode is ViewMode.DEM
            and settings.reference_plane is not None
        )

    def _render_dem(self, height_mm, display, settings):
        # 点群表示アプリのテスト入力として書き出す（--record のときだけ）。
        # 描く前に積むのは、重いフレームの描画を待って書き出しが遅れないようにするため。
        if self.recorder is not None:
            self._record(height_mm, settings)

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
                _, min_cells, full_cells = settings.river_setting
                strength = self._to_projection(
                    rivers.river_strength(height_mm, min_cells, full_cells), settings
                )
                dem = overlays.draw_rivers(dem, strength)
        else:
            elapsed = self._elapsed()
            terrain = overlays.terrain_color(height_mm, elapsed, settings.water_level_mm)

            dry = height_mm >= settings.water_level_mm
            # 水源の水を流しているあいだは、雨の筋（D8）は描かない。「雨が降ったら
            # どこを流れるか」と「いま流れている水」が重なると、どちらが本物の
            # 水か見分けられない。
            if settings.show_rivers and not settings.show_flood and dry.any():
                # 水面より下の川は描かない。湖や海に入った川は見えなくなる。
                # 全部沈んでいるなら流量の計算そのものが要らない。
                _, min_cells, full_cells = settings.river_setting
                flow = rivers.river_strength(height_mm, min_cells, full_cells)
                strength = np.where(dry, flow, 0.0)

                # 川は拡大する前に混ぜる。投影解像度で混ぜると触る画素が
                # 6 倍になり、それだけで 1 フレームの予算の半分を使う。
                terrain = overlays.draw_rivers(terrain, strength)

            if self._flood_active(settings):
                # 海へ流れ込んだ水はそこで消える。砂場の外へ出た水も消す
                # （エリアは左右反転後の座標なので、センサ側へ戻して渡す）。
                area = None
                if settings.has_area:
                    area = sensor_area_from_view(settings.area_positions)
                water = self._flood.update(
                    height_mm,
                    elapsed,
                    spring_uv=settings.spring_position,
                    heavy_rain=settings.heavy_rain,
                    sea_level_mm=settings.water_level_mm,
                    area_uv=area,
                )
                # 川と同じく、拡大する前に混ぜる。
                terrain = overlays.draw_flood(terrain, water, elapsed)

            dem = self._to_projection(terrain, settings)

        if settings.show_contour:
            dem = self._draw_contours(height_mm, dem, settings)
        return dem

    def _record(self, height_mm, settings):
        """録画へ高さ[mm]を 1 枚渡す。

        書き出すのは投影用の絵ではなく高さ[mm]。穴埋め・平滑化・揺れの抑えと
        基準面による傾き補正が済んでおり、受け取る側がそのまま点群にできる。

        `--record-crop` では砂場の内側だけを切り出す。**エリアが決まっていない
        あいだは書き出さない。** 視野全体のフレームを混ぜて書くと、あとから
        どれが砂場だけなのか見分けられなくなり、録り直すしかなくなるため。
        黙って止まらないよう、理由を画面へ出す（`app._check_recorder`）。
        """
        frame = height_mm
        area = None

        if self.record_crop:
            if not settings.has_area:
                self.recorder.warn(
                    "録画: 砂場の四隅が未指定のため書き出していません（画面を 4 点クリック）"
                )
                return
            frame = crop_to_area(height_mm, settings.area_positions)
            if frame is None:
                self.recorder.warn("録画: 砂場の四隅が潰れていて切り取れません")
                return
            area = settings.area_positions

        self.recorder.write(
            frame,
            self._elapsed(),
            frame_context(settings.reference_plane, settings.water_level_mm, area),
        )

    def _render_depth(self, height_mm, display, settings):
        depth_view = self._to_view(display)
        depth_view = cv2.bitwise_not(depth_view)

        if self._use_perspective(settings):
            depth_view = self._warp(depth_view, settings)

        if settings.show_contour:
            depth_view = self._draw_contours(height_mm, depth_view, settings)
        return depth_view

    def _render_edge(self, height_mm, settings):
        colored = self._colorize(height_mm, settings)

        if self._use_perspective(settings):
            colored = self._warp(colored, settings)

        return overlays.edge_lines(colored)
