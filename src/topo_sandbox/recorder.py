"""DEM 表示の高さ画像を連番ファイルへ書き出す。

砂場の起伏を点群にして別のディスプレイへ出すアプリを別途作るにあたり、その
**テスト入力**を本機から取り出すための道具である。実演そのものには関与しない。

``--record`` を付けて起動したときだけ動き、**DEM 表示のあいだ**の
``renderer.Renderer`` が作る高さ[mm]（:func:`~topo_sandbox.processing.pointcloud.heights_from_depth`
の出力）を 1 フレーム 1 ファイルで書き出す。深度そのものではなく高さを出すのは、
欠測の穴埋め・平滑化・揺れの抑えと、基準面によるセンサの傾き補正が済んでおり、
受け取る側がそのまま点群にできるため。

``--record-crop`` を足すと、砂場の四隅（エリア指定）の内側だけを長方形へ直して
書き出す。付けないとセンサの視野全体が入り、砂場のまわりの床が一面の窪地として、
前を横切った人が高い壁として残る。どちらで録ったかは ``meta.json`` の
``cropped`` と ``axes`` に出る。

書き出す形
----------

``--record-dir`` の下に ``dem_YYYYmmdd_HHMMSS`` を作り、その中へ次を置く。

- ``frame_000001.npy`` … 高さ[mm] (240, 320) float32。正が高い（盛った砂）。
  :func:`numpy.load` でそのまま読める。
- ``index.csv`` … ``frame,elapsed_s``。書けたフレームの通し番号と、表示を
  始めてからの経過秒。**取りこぼしたフレームは番号を飛ばさず、経過秒のほうに
  間隔として現れる。** 受け取る側はこの秒を見て再生すれば実際の間隔で流せる。
- ``meta.json`` … 大きさ・単位・符号・基準面など。読み込み側の手掛かり。

実演を止めないための作り
------------------------

ファイルへの書き込みは**別スレッド**で行い、描画のワーカスレッドは待たない。
書き込みが間に合わなければ**新しいフレームを捨てる**（溜め込んで描画を
遅らせない）。ディスクが一杯になるなどして書けなくなったら録画だけを止め、
アプリは動かし続ける。:meth:`DemRecorder.write` は例外を外へ出さない。
"""

import contextlib
import csv
import dataclasses
import json
import queue
import threading
from datetime import datetime

import numpy as np

from . import config

#: 書き出す形式の版。読み込む側が形の違いに気付けるようにする。
#: 1 → 2 で ``cropped`` / ``crop_area_positions`` が増えた。版 1 の meta.json には
#: これらが無く、どれも切り取っていない（視野全体の）録画にあたる。
FORMAT_VERSION = 2

#: 連番ファイルの名前
_FRAME_NAME = "frame_{:06d}.npy"

_INDEX_NAME = "index.csv"
_META_NAME = "meta.json"

#: 終了を知らせるためにキューへ入れる印
_SENTINEL = None

#: 書き込みスレッドの終了を待つ上限[秒]。
#: 待ち続けるとアプリが閉じられなくなるため、諦めて先へ進む。
_JOIN_TIMEOUT_S = 5.0


class DemRecorder:
    """DEM の高さ[mm]を連番ファイルへ書き出す。

    1 つのインスタンスを :class:`~topo_sandbox.renderer.Renderer` が持ち、
    描画のワーカスレッドから :meth:`write` を呼ばれる。
    """

    def __init__(self, directory=None, max_frames=None, queue_size=None):
        """
        Args:
            directory: 書き出し先の親。省略すると :data:`config.RECORD_DIR`。
                この下に日時の名前で 1 回ぶんのフォルダを作る。
            max_frames: 書き出す上限の枚数。0 以下で無制限。省略すると
                :data:`config.RECORD_MAX_FRAMES`。
            queue_size: 書き込み待ちに積めるフレーム数。省略すると
                :data:`config.RECORD_QUEUE_SIZE`。
        """
        self.directory = directory or config.RECORD_DIR
        self.max_frames = config.RECORD_MAX_FRAMES if max_frames is None else max_frames
        queue_size = config.RECORD_QUEUE_SIZE if queue_size is None else queue_size

        #: 書き出し先の 1 回ぶんのフォルダ。:meth:`open` で決まる。
        self.path = None

        #: 受け付けた枚数（＝連番の最後）
        self.accepted = 0
        #: 実際にファイルへ書けた枚数
        self.written = 0
        #: 書き込みが間に合わず捨てた枚数
        self.dropped = 0
        #: 書き込みに失敗した回数
        self.errors = 0
        #: 最後の失敗の内容。画面に出して原因を伝えるために残す。
        self.last_error = None

        # 実際に書いた画像の大きさ。切り取りの有無で変わるので、
        # 設定値ではなく書いたものから meta.json に残す。
        self._shape = None

        # 直前に出した「書き出せない理由」。毎フレーム同じ文面を積まないため。
        self._last_warning = None

        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._index_file = None
        self._index_writer = None

        # 上限に達した・書けなくなった・閉じた、のいずれか。立つと以後は受け付けない。
        self._stopped = True

        # 最後に書いたフレームの状況（基準面・水位）。meta.json に残す。
        # 1 枚も書かないまま終わっても形が崩れないよう、空の形で持っておく。
        self._context = {
            "reference_plane": None,
            "water_level_mm": None,
            "crop_area_positions": None,
        }

        # 画面へ 1 度だけ知らせるための印。`app` が読んで下ろす。
        self._notice = None

    # ------------------------------------------------------------------
    @property
    def active(self):
        """まだ受け付けているか。"""
        return not self._stopped

    def warn(self, message):
        """書き出せない理由を知らせる。同じ理由は繰り返さない。

        エリアが決まっていないあいだは毎フレーム同じ理由で弾かれるため、
        そのたびに知らせを積むと画面が同じ文字で埋まる。理由が変わったとき
        （または一度書けたあと再び弾かれたとき）だけ出す。
        """
        if message == self._last_warning:
            return
        self._last_warning = message
        self._notice = message

    def take_notice(self):
        """画面に出すべき知らせがあれば 1 度だけ返す。

        録画が勝手に止まったことに気付けるようにするためのもの。
        メインスレッドから呼ぶ。
        """
        notice, self._notice = self._notice, None
        return notice

    # ------------------------------------------------------------------
    def open(self):
        """書き出し先を作り、書き込みスレッドを起こす。

        Returns:
            作った 1 回ぶんのフォルダの :class:`~pathlib.Path`。

        Raises:
            OSError: フォルダやファイルを作れないとき。起動時に 1 度だけ
                呼ばれるので、ここは黙って失敗させず呼び出し側へ伝える。
        """
        if self._thread is not None:
            return self.path

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.directory / f"dem_{stamp}"
        self.path.mkdir(parents=True, exist_ok=True)

        # 行ごとに書き足すので、強制終了されても途中まで残る。
        self._index_file = (self.path / _INDEX_NAME).open("w", encoding="utf-8", newline="")
        self._index_writer = csv.writer(self._index_file)
        self._index_writer.writerow(["frame", "elapsed_s"])

        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="dem-recorder", daemon=True)
        self._thread.start()

        # 途中で落ちても形が分かるよう、始めた時点で 1 度書いておく。
        self._write_meta()
        return self.path

    def write(self, height_mm, elapsed_s, context=None):
        """高さ[mm]を 1 枚、書き込み待ちへ積む。

        描画のワーカスレッドから呼ばれる。**待たず、例外も出さない。**
        積めなければ捨てる。

        Args:
            height_mm: 基準面からの高さ[mm] (H, W)。正が高い。
            elapsed_s: 表示を始めてからの経過秒。
            context: そのフレームの状況（基準面・水位）。meta.json へ残す。

        Returns:
            積めたら True。捨てた・止まっているなら False。
        """
        if self._stopped:
            return False

        if context:
            self._context = context

        # float32 の連続領域に揃えてから渡す。呼び出し側の配列を書き込み
        # スレッドが持ち続けないよう、ここで写しを作る。
        frame = np.ascontiguousarray(height_mm, dtype=np.float32)

        index = self.accepted + 1
        try:
            self._queue.put_nowait((index, float(elapsed_s), frame))
        except queue.Full:
            # 書き込みが追いつかないとき。溜め込むと描画まで遅れるので捨てる。
            self.dropped += 1
            return False

        self.accepted = index
        self._shape = tuple(frame.shape)
        self._last_warning = None  # 書けたので、次に弾かれたらまた知らせる

        if 0 < self.max_frames <= self.accepted:
            self._notice = f"録画の上限 {self.max_frames} 枚に達しました: {self.path}"
            self._stop()
        return True

    def close(self):
        """書き残しを書き切ってから片付ける。

        何度呼んでも構わない。:meth:`open` していなければ何もしない。
        """
        if self._thread is None:
            return

        self._stop()
        self._thread.join(timeout=_JOIN_TIMEOUT_S)
        self._thread = None

        if self._index_file is not None:
            self._index_file.close()
            self._index_file = None
            self._index_writer = None

        self._write_meta()

    # ------------------------------------------------------------------
    def _stop(self):
        """受け付けをやめ、書き込みスレッドへ終わりを伝える。"""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            # 満杯なら、積まれているぶんを書き終えたあとで入る余地ができる。
            # ただし**待ち切らない**。書き込みスレッドが書けなくなって抜けた直後だと
            # 誰も取り出さず、待ち続けると描画のワーカスレッドごと止まる
            # （＝投影が固まる）。伝えられなくてもスレッドは daemon なので残らない。
            with contextlib.suppress(queue.Full):
                self._queue.put(_SENTINEL, timeout=_JOIN_TIMEOUT_S)

    def _run(self):
        """書き込みスレッドの本体。"""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break

            index, elapsed_s, frame = item
            try:
                np.save(self.path / _FRAME_NAME.format(index), frame)
                self._index_writer.writerow([index, f"{elapsed_s:.4f}"])
                self._index_file.flush()
            except (OSError, ValueError) as error:
                # ディスクが一杯になった、書き込み先が消えた、など。
                # 録画だけを止め、アプリは動かし続ける。
                self.errors += 1
                self.last_error = error
                self._notice = f"録画を中止しました: {error}"
                self._stopped = True
                break
            self.written += 1

    def _write_meta(self):
        """読み込む側の手掛かりを meta.json へ書く。

        書けなくても録画そのものは続けられるので、失敗は数えるだけにする。
        """
        width, height = config.PROC_SIZE
        shape = list(self._shape) if self._shape else [height, width]
        area = self._context.get("crop_area_positions")

        if area:
            axes = (
                "行が y、列が x。砂場の四隅の内側だけを長方形へ直してある。"
                "画面でクリックした 1 点目が左上に来るので、向きは投影像と同じ"
                "（センサから見たままの並びではない）。x と y は砂場の端から端までを"
                "等分したもので、ミリメートルではない"
            )
        else:
            axes = (
                "行が y（センサ画像の上から下）、列が x（センサから見たまま。"
                "投影像は左右反転しているので、投影と向きを揃えるなら列を反転する）。"
                "砂場の外（床やまわりの人）も入っている"
            )

        meta = {
            "version": FORMAT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "topo_sandbox DEM height_mm",
            "description": (
                "基準面からの高さ[mm]。1 フレーム 1 ファイル（frame_NNNNNN.npy）。"
                "numpy.load でそのまま読める。"
            ),
            "dtype": "float32",
            "shape": shape,
            "unit": "mm",
            "sign": "正が高い（盛った砂）",
            "origin": ("基準面" if self._context.get("reference_plane") else "フレームの中央値"),
            "cropped": bool(area),
            "axes": axes,
            "frame_files": _FRAME_NAME,
            "index_file": _INDEX_NAME,
            "frames": self.written,
            "accepted": self.accepted,
            "dropped": self.dropped,
            "errors": self.errors,
        }
        meta.update(self._context)

        try:
            (self.path / _META_NAME).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as error:
            self.errors += 1
            self.last_error = error

    # ------------------------------------------------------------------
    def describe(self):
        """いまの状況を 1 行で表す。画面やコンソールへ出す。"""
        parts = [f"録画: {self.written} 枚"]
        if self.max_frames > 0:
            parts.append(f"/ 上限 {self.max_frames}")
        if self.dropped:
            parts.append(f"（取りこぼし {self.dropped}）")
        if not self.active:
            parts.append("※停止中")
        return " ".join(parts)


def frame_context(reference_plane, water_level_mm, area_positions=None):
    """meta.json へ残す「どういう状況で録ったか」を作る。

    高さの原点が基準面なのかフレームの中央値なのかで、受け取る側の扱いが
    変わる。基準面が無いまま録ったものは、砂を動かすたびに原点が動く
    （AGENTS.md「絶対高さには基準面が要ります」）。

    Args:
        reference_plane: :class:`~topo_sandbox.processing.plane.ReferencePlane`。
            未取得なら None。
        water_level_mm: そのときの水位[mm]。
        area_positions: 切り取りに使った砂場の四隅（表示像の座標）。
            切り取っていなければ None。

    Returns:
        JSON にできる辞書。
    """
    plane = None
    if reference_plane is not None:
        plane = {key: float(value) for key, value in dataclasses.asdict(reference_plane).items()}

    area = None
    if area_positions:
        area = [[int(x), int(y)] for x, y in area_positions]

    return {
        "reference_plane": plane,
        "water_level_mm": float(water_level_mm),
        "crop_area_positions": area,
    }
