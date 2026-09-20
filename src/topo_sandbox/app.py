"""Tkinter のウィンドウとキー操作。

イベント会場ではプロジェクタへ全画面表示し、キーボードだけで操作する。
操作方法は README のキーバインド表と対応している。**現場で体に入っている
手順なので、割り当てを変えるときは必ず README も直すこと。**
"""

import threading
import tkinter as tk
from queue import Queue

from PIL import Image, ImageTk

from . import config
from .processing import plane
from .renderer import VIEW_MODE_ORDER, MappingMode, Renderer, RenderSettings, ViewMode


class SandboxApp:
    """TopoSandboxの画面。"""

    def __init__(self, source, renderer=None):
        """
        Args:
            source: :class:`~topo_sandbox.sensor.base.DepthSource` の実装。
            renderer: 省略時は既定の :class:`~topo_sandbox.renderer.Renderer`。
        """
        self.source = source
        self.renderer = renderer or Renderer()
        self.settings = RenderSettings()

        self.window = None
        self.canvas = None
        self._photo_image = None  # GC されると表示が消えるため保持する
        self._after_job_id = None

        self._queue = Queue()
        self._worker = None

        self._message = ""
        self._message_frames = config.MESSAGE_FRAMES
        self._message_duration = config.MESSAGE_FRAMES

        #: 基準面の取得中に貯めているフレーム。None なら取得していない。
        self._plane_frames = None

        #: 投影枠の編集モードと、選択中の角
        self._projector_edit = False
        self._projector_corner = 0

        self._fullscreen = False
        self._geometry = None

    # ------------------------------------------------------------------
    # 起動と終了
    # ------------------------------------------------------------------
    def run(self):
        width, height = config.VIEW_SIZE

        self.window = tk.Tk()
        self.window.title("TopoSandbox")
        self.window.geometry(f"{width}x{height}")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.window.bind("<F11>", self._on_toggle_fullscreen)
        self.window.bind("<KeyPress>", self._on_key)
        self.window.bind("<Tab>", self._select_next_corner)

        for key, handler in self._area_key_bindings().items():
            self.window.bind(key, handler)

        self.canvas = tk.Canvas(self.window, width=width, height=height)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.place(x=0, y=0)

        print("エリア指定は左上から反時計回りにクリックしてください。")

        self._after_job_id = self.window.after(config.FRAME_INTERVAL_MS, self._tick)
        self.window.mainloop()

    def _on_close(self):
        if self._after_job_id is not None:
            self.window.after_cancel(self._after_job_id)
        self.source.close()
        self.window.destroy()

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------
    def _tick(self):
        frame = self.source.read()

        if frame is not None and self._plane_frames is not None:
            self._collect_plane_frame(frame)

        # 前のフレームの処理が終わっていなければ、今回のフレームは捨てる。
        # 溜め込むと表示が実際の砂場から遅れていくため。
        if frame is not None and (self._worker is None or not self._worker.is_alive()):
            self._worker = threading.Thread(target=self._process, args=(frame,))
            self._worker.start()

        self._draw_latest()
        self._draw_projector_frame()
        self._draw_message()
        self._after_job_id = self.window.after(config.FRAME_INTERVAL_MS, self._tick)

    def _process(self, frame):
        """ワーカスレッドで描画用の画像を作る。"""
        self._queue.put(self.renderer.render(frame, self.settings))

    def _draw_latest(self):
        """キューに結果があれば描画する。古い結果は捨てる。"""
        if self._queue.empty():
            return

        image = self._queue.get()
        self._photo_image = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.create_image(
            self._photo_image.width() // 2,
            self._photo_image.height() // 2,
            image=self._photo_image,
        )
        self._queue = Queue()

    # ------------------------------------------------------------------
    # 画面へのメッセージ
    # ------------------------------------------------------------------
    def _show_message(self, message, duration=None):
        self._message = message
        self._message_frames = 0
        self._message_duration = duration or config.MESSAGE_FRAMES

    def _draw_message(self):
        if self._message_frames < self._message_duration:
            self.canvas.create_text(
                10,
                10,
                text=self._message,
                anchor="nw",
                font=("HG丸ｺﾞｼｯｸM-PRO", 18),
                fill="blue",
            )
            self._message_frames += 1

    # ------------------------------------------------------------------
    # 全画面
    # ------------------------------------------------------------------
    def _on_toggle_fullscreen(self, _event=None):
        if self._fullscreen:
            self._fullscreen = False
            self.window.state("normal")
            self.window.geometry(self._geometry)
            self.window.overrideredirect(False)
        else:
            self._fullscreen = True
            self._geometry = self.window.geometry()
            self.window.overrideredirect(True)  # 枠を消す
            self.window.state("zoomed")

    # ------------------------------------------------------------------
    # 投影エリアの指定
    # ------------------------------------------------------------------
    def _on_click(self, event):
        positions = self.settings.area_positions
        if len(positions) == 4:
            self._show_message("これ以上は追加できません。変更する場合は一度クリアしてください。")
            return
        positions.append([event.x, event.y])
        self._show_message(f"エリア指定座標へ追加しました。 -> ({event.x}, {event.y})")

    def _require_area(self):
        """四隅がそろっているか確認する。

        そろっていない状態で座標を操作すると添字エラーで落ちるため、
        実演中の停止を避けるためにここで止める。
        """
        if self.settings.has_area:
            return True
        self._show_message("先にエリアを 4 点クリックしてください。")
        return False

    def _area_key_bindings(self):
        """Shift / Shift+Ctrl + 方向キーでエリアを変形する。

        Shift が拡大、Shift+Ctrl が縮小。左右は左側 2 点と右側 2 点、
        上下は上側 2 点と下側 2 点をそれぞれ動かす。

        なお下方向の符号は他の方向と揃っていないが、現場の操作感を変えない
        ため元の挙動をそのまま維持している。見直すなら改良作業として行うこと。
        """
        # (キー, 動かす点の添字, 座標軸(0=x,1=y), 移動量, メッセージ, 投影枠編集中の移動方向)
        table = [
            ("<Shift-Left>", (0, 1), 0, -1, "左方向へ引き伸ばし", (-1, 0)),
            ("<Shift-Right>", (2, 3), 0, +1, "右方向へ引き伸ばし", (+1, 0)),
            ("<Shift-Up>", (0, 3), 1, -1, "上方向へ引き伸ばし", (0, -1)),
            ("<Shift-Down>", (1, 2), 1, -1, "下方向へ引き伸ばし", (0, +1)),
            ("<Shift-Control-Left>", (0, 1), 0, +1, "左方向へ縮め", None),
            ("<Shift-Control-Right>", (2, 3), 0, -1, "右方向へ縮め", None),
            ("<Shift-Control-Up>", (0, 3), 1, +1, "上方向へ縮め", None),
            ("<Shift-Control-Down>", (1, 2), 1, +1, "下方向へ縮め", None),
        ]

        bindings = {}
        for key, indices, axis, delta, message, coarse in table:
            bindings[key] = self._make_area_handler(indices, axis, delta, message, coarse)
        return bindings

    def _make_area_handler(self, indices, axis, delta, message, coarse):
        def handler(_event):
            # 投影枠の編集中は、Shift + 方向キーで選択中の角を粗く動かす
            if self._projector_edit:
                if coarse is not None:
                    step = config.PROJECTOR_COARSE_STEP
                    self._move_projector_corner(coarse[0] * step, coarse[1] * step)
                return

            if not self._require_area():
                return
            for index in indices:
                self.settings.area_positions[index][axis] += delta
            self._show_message(message)

        return handler

    def _arrow(self, dx, dy, axis, delta, message):
        """方向キーの行き先を、いまのモードに応じて振り分ける。"""
        if self._projector_edit:
            self._move_projector_corner(dx, dy)
            return
        self._move_area(axis, delta, message)

    def _move_area(self, axis, delta, message):
        if not self._require_area():
            return
        for position in self.settings.area_positions:
            position[axis] += delta
        self._show_message(message)

    # ------------------------------------------------------------------
    # キー操作
    # ------------------------------------------------------------------
    def _on_key(self, event):
        key = event.keysym

        if key == "v":
            self._cycle_view_mode()
        elif key == "t":
            self.settings.show_contour = not self.settings.show_contour
            self._show_message("等高線: %s" % ("表示" if self.settings.show_contour else "非表示"))
        elif key == "j":
            self._toggle_mapping_mode()
        elif key == "k":
            self._start_plane_capture()
        elif key == "K":
            self._clear_plane()
        elif key == "p":
            self._toggle_projector_edit()
        elif key == "P":
            self._reset_projector_quad()
        elif key == "z":
            self._adjust_z_scale(+config.Z_SCALE_DELTA)
        elif key == "x":
            self._adjust_z_scale(-config.Z_SCALE_DELTA)
        elif key == "a":
            self._adjust_sensitivity(+config.COLOR_SENSITIVITY_DELTA)
        elif key == "s":
            self._adjust_sensitivity(-config.COLOR_SENSITIVITY_DELTA)
        # 方向キーは、投影枠の編集中なら選択中の角を 1 画素ずつ動かす。
        # そうでなければエリア全体の平行移動。
        # 左キーで座標を + する（＝画面上では右へ動く）のは元からの挙動。
        elif key == "Left":
            self._arrow(-1, 0, 0, +1, "左移動")
        elif key == "Right":
            self._arrow(+1, 0, 0, -1, "右移動")
        elif key == "Up":
            self._arrow(0, -1, 1, -1, "上移動")
        elif key == "Down":
            self._arrow(0, +1, 1, +1, "下移動")
        # `c` によるエリア座標のクリアは、実演中の誤操作を防ぐため無効にしてある。

    def _cycle_view_mode(self):
        index = VIEW_MODE_ORDER.index(self.settings.view_mode)
        self.settings.view_mode = VIEW_MODE_ORDER[(index + 1) % len(VIEW_MODE_ORDER)]
        self._show_message(f"表示モードの切り替え: {self.settings.view_mode.name}")

        if self.settings.view_mode is ViewMode.DEM and self.settings.reference_plane is None:
            # 水位は基準面からの絶対高さで決めるため、未取得のままでは出せない。
            # 何も言わずに従来の DEM が出ると、故障と区別がつかない。
            self._show_message("DEM: 水面を出すには基準面が要ります。k を押してください。")

    def _toggle_mapping_mode(self):
        if self.settings.mapping_mode is MappingMode.NORMAL:
            self.settings.mapping_mode = MappingMode.PERSPECTIVE
        else:
            self.settings.mapping_mode = MappingMode.NORMAL
        self._show_message(f"マッピングモードの切り替え: {self.settings.mapping_mode.name}")

    def _adjust_z_scale(self, delta):
        self.settings.z_scale += delta
        self._show_message(f"Zスケール: {self.settings.z_scale:.2f}")

    def _adjust_sensitivity(self, delta):
        value = self.settings.color_sensitivity + delta
        # 上限は配色テーブルの長さ、下限は 0。範囲外の添字で落ちるのを防ぐ。
        value = max(0.0, min(config.COLOR_SENSITIVITY_MAX, value))
        self.settings.color_sensitivity = value
        self._show_message(f"カラー感度: {value:.1f}")

    # ------------------------------------------------------------------
    # 基準面の取得
    # ------------------------------------------------------------------
    def _start_plane_capture(self):
        """基準面の取得を始める。

        フレームはメインループで少しずつ貯める。まとめて読むと
        その間ずっと画面が固まるため。
        """
        self._plane_frames = []
        self._show_message(f"基準面を取得中… 0/{config.PLANE_FRAME_COUNT}")

    def _collect_plane_frame(self, frame):
        """基準面用のフレームを 1 枚貯め、そろったら平面をあてはめる。"""
        self._plane_frames.append(frame)
        collected = len(self._plane_frames)

        if collected < config.PLANE_FRAME_COUNT:
            self._show_message(f"基準面を取得中… {collected}/{config.PLANE_FRAME_COUNT}")
            return

        frames = self._plane_frames
        self._plane_frames = None
        self._finish_plane_capture(frames)

    def _finish_plane_capture(self, frames):
        try:
            reference = plane.capture(frames)
        except ValueError as error:
            self._show_message(f"基準面を取得できませんでした: {error}", duration=200)
            return

        self.settings.reference_plane = reference

        message = reference.describe()
        if reference.residual_mm > config.PLANE_RESIDUAL_WARN_MM:
            message += "  ※砂がならせていない可能性があります"
        self._show_message(message, duration=200)

    def _clear_plane(self):
        if self.settings.reference_plane is None:
            self._show_message("基準面は取得されていません。")
            return
        self.settings.reference_plane = None
        self._show_message("基準面を破棄しました。傾き補正は行われません。")

    # ------------------------------------------------------------------
    # 投影枠の合わせ込み
    # ------------------------------------------------------------------
    def _toggle_projector_edit(self):
        """投影枠の編集モードを出入りする。

        投影された枠を砂場の物理的な角へ重ねる作業。センサ画像を見て
        推測するより、投影光を直接見て合わせるほうが確実に決まる。
        """
        self._projector_edit = not self._projector_edit
        if self._projector_edit:
            self._show_message(
                "投影枠の合わせ込み: Tab で角を選択、方向キーで移動、Shift で粗く、p で終了",
                duration=200,
            )
        else:
            self._show_message("投影枠を確定しました。")

    def _select_next_corner(self, _event=None):
        if not self._projector_edit:
            return None
        self._projector_corner = (self._projector_corner + 1) % 4
        self._show_message(f"角 {self._projector_corner + 1}/4 を選択")
        return "break"  # Tab によるフォーカス移動を抑える

    def _move_projector_corner(self, dx, dy):
        corner = self.settings.projector_positions[self._projector_corner]
        corner[0] += dx
        corner[1] += dy
        self._show_message(f"角 {self._projector_corner + 1}/4 -> ({corner[0]}, {corner[1]})")

    def _reset_projector_quad(self):
        self.settings.reset_projector_quad()
        self._projector_corner = 0
        self._show_message("投影枠を画面全体へ戻しました。")

    def _draw_projector_frame(self):
        """投影枠と選択中の角を重ねて描く。

        枠が砂場の縁と一致しているかを、投影像そのもので確認できる。
        """
        if not self._projector_edit:
            return

        positions = self.settings.projector_positions
        flat = [value for position in positions for value in position]
        self.canvas.create_polygon(
            flat,
            outline=config.PROJECTOR_FRAME_COLOR,
            fill="",
            width=2,
        )

        size = config.PROJECTOR_CORNER_SIZE
        for index, (x, y) in enumerate(positions):
            selected = index == self._projector_corner
            color = config.PROJECTOR_CORNER_COLOR if selected else config.PROJECTOR_FRAME_COLOR
            width = 3 if selected else 1
            self.canvas.create_line(x - size, y, x + size, y, fill=color, width=width)
            self.canvas.create_line(x, y - size, x, y + size, fill=color, width=width)


# 表示モードの列挙を外へ出しておく（外部から参照されるため）
__all__ = ["SandboxApp", "ViewMode", "MappingMode"]
