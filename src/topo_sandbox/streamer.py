"""DEM の高さ[mm]をライブ表示アプリ（topo-sandbox-live）へ WebSocket で送る。

砂場の起伏を別のディスプレイへ 2.5 次元で映すアプリ（topo-sandbox-live）に、
いま投影している砂場の高さをその場で渡すためのもの。``--stream`` を付けて
起動したときだけ動き、**付けなければ何も起きない。** 実演の起動手順は変わらない。

送るのは録画（:mod:`~topo_sandbox.recorder`）と同じく投影像ではなく高さ[mm]で、
欠測の穴埋め・平滑化・揺れの抑えと、基準面による傾き補正が済んでいる。
受け取る側はそのまま高さの格子として描ける。

送る形
------

1 フレーム 1 メッセージのバイナリで、topo-sandbox-live の配信形式と同じ並び::

    offset 0    uint32 LE   N = JSON ヘッダの長さ[バイト]
    offset 4    N バイト     UTF-8 の JSON ヘッダ（4 + N が 4 の倍数になるよう空白で詰める）
    offset 4+N  rows*cols*4  float32 LE の高さ[mm]。行優先。正が高い

ヘッダの例::

    {"type": "height", "version": 1, "rows": 240, "cols": 320, "index": 12,
     "unit": "mm", "cropped": true, "origin": "reference", "waterLevelMm": -40.0}

受け取る側の定義は topo-sandbox-live の ``backend/app/sources/topo_sandbox.py``。
形を変えるときは両方を揃え、:data:`PROTOCOL_VERSION` を上げる。

実演を止めないための作り
------------------------

送信は**別スレッド**で行い、描画のワーカスレッドは待たない。送るのは常に
**いちばん新しい 1 枚だけ**で、送信が間に合わなければ古いフレームを捨てる
（溜め込むと、ライブ表示が砂場から遅れていく）。相手が起動していない・
途中で切れた、というときは間を置いてつなぎ直し続ける。
:meth:`DemStreamer.send` は例外を外へ出さない。
"""

import contextlib
import json
import select
import struct
import threading

import numpy as np

from . import config

#: 送る形式の版。受け取る側が形の違いに気付けるようにする。
PROTOCOL_VERSION = 1

_HEADER_LEN = struct.Struct("<I")

#: ヘッダの後ろを揃える境界[バイト]。受け取る側（ブラウザを含む）が
#: 高さの部分を写さずに float32 の配列として読めるようにするため。
_ALIGN = 4

_FLOAT32_LE = np.dtype("<f4")

#: 送るものが無いあいだも、この間隔[秒]で相手からの ping に答える。
#: 答えないと、受け取る側のサーバ（uvicorn）が既定 20 秒で接続を切る。
_POLL_S = 0.5

#: 送信スレッドの終了を待つ上限[秒]。待ち続けるとアプリが閉じられなくなるため。
_JOIN_TIMEOUT_S = 5.0


def encode_frame(height_mm, header):
    """高さ[mm]とヘッダを 1 メッセージのバイト列にする。

    Args:
        height_mm: 高さ[mm] (rows, cols)。
        header: JSON にできる辞書。``rows`` と ``cols`` はここで足す。

    Returns:
        送るバイト列。
    """
    heights = np.ascontiguousarray(height_mm, dtype=_FLOAT32_LE)
    rows, cols = heights.shape
    header = dict(header, rows=int(rows), cols=int(cols))

    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-(_HEADER_LEN.size + len(header_bytes))) % _ALIGN)
    return _HEADER_LEN.pack(len(header_bytes)) + header_bytes + heights.tobytes()


class _WebSocketConnection:
    """websocket-client の接続の薄い包み。

    :class:`DemStreamer` が使うのは :meth:`send` / :meth:`poll` / :meth:`close`
    の 3 つだけで、試験ではこれと同じ形の身代わりを渡す。
    """

    def __init__(self, url, timeout_s):
        # 遅延 import。--stream を付けないときは読み込まない（起動を重くしない）。
        import websocket

        self._abnf = websocket.ABNF
        self._ws = websocket.create_connection(url, timeout=timeout_s)

    def send(self, payload):
        self._ws.send_binary(payload)

    def poll(self):
        """届いているもの（主に ping）を処理する。

        websocket-client は読んだときにしか ping へ答えない。こちらは送るだけで
        読まないので、ここで読んで答える。

        Returns:
            まだつながっていれば True。相手が閉じたら False。
        """
        while True:
            readable, _, _ = select.select([self._ws.sock], [], [], 0)
            if not readable:
                return True
            opcode, _ = self._ws.recv_data_frame(True)
            if opcode == self._abnf.OPCODE_CLOSE:
                return False

    def close(self):
        # 閉じるときの失敗は、どれも捨ててよい（どのみち次はつなぎ直す）。
        with contextlib.suppress(Exception):
            self._ws.close(timeout=1)


class DemStreamer:
    """DEM の高さ[mm]をライブ表示アプリへ送り続ける。

    1 つのインスタンスを :class:`~topo_sandbox.renderer.Renderer` が持ち、
    描画のワーカスレッドから :meth:`send` を呼ばれる。
    """

    def __init__(self, url=None, timeout_s=None, retry_s=None, connect=None):
        """
        Args:
            url: 送り先。省略すると :data:`config.STREAM_URL`。
            timeout_s: つなぐとき・送るときの待ち時間の上限[秒]。
                省略すると :data:`config.STREAM_TIMEOUT_S`。
            retry_s: つなぎ直すまでの間隔[秒]。省略すると :data:`config.STREAM_RETRY_S`。
            connect: ``connect(url, timeout_s)`` で接続を返す関数。試験で身代わりを
                渡すためのもので、普段は websocket-client を使う。
        """
        self.url = url or config.STREAM_URL
        self.timeout_s = config.STREAM_TIMEOUT_S if timeout_s is None else timeout_s
        self.retry_s = config.STREAM_RETRY_S if retry_s is None else retry_s
        self._connect = connect or _WebSocketConnection

        #: 送れた枚数
        self.sent = 0
        #: つながっているのに送信が間に合わず、送る前に新しいものへ置き換えた枚数
        self.dropped = 0
        #: いまつながっているか
        self.connected = False
        #: 最後の失敗の内容。画面やコンソールに出して原因を伝えるために残す。
        self.last_error = None

        # 送るのを待っている 1 枚（ヘッダ, 高さ）。新しいものが来たら置き換える。
        self._pending = None
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self._index = 0

        # 画面へ 1 度だけ知らせるための印。`app` が読んで下ろす。
        self._notice = None

    # ------------------------------------------------------------------
    @property
    def active(self):
        """送信スレッドが動いているか。"""
        return self._thread is not None and not self._stop_event.is_set()

    def take_notice(self):
        """画面に出すべき知らせがあれば 1 度だけ返す。メインスレッドから呼ぶ。"""
        notice, self._notice = self._notice, None
        return notice

    def start(self):
        """送信スレッドを起こす。つなぐのはスレッドの中なので、ここでは待たない。"""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="dem-streamer", daemon=True)
        self._thread.start()

    def send(self, height_mm, cropped=False, reference=False, water_level_mm=None):
        """高さ[mm]を 1 枚、送信待ちへ置く。

        描画のワーカスレッドから呼ばれる。**待たず、例外も出さない。**
        前の 1 枚がまだ送れていなければ置き換える。

        Args:
            height_mm: 高さ[mm] (H, W)。正が高い。
            cropped: 砂場の四隅の内側だけを切り取ったものか。
            reference: 高さの原点が基準面か（False ならフレームの中央値）。
            water_level_mm: そのときの水位[mm]。

        Returns:
            置けたら True。止まっているなら False。
        """
        if not self.active:
            return False

        # 呼び出し側の配列を送信スレッドが持ち続けないよう、ここで写しを作る。
        frame = np.array(height_mm, dtype=np.float32)

        self._index += 1
        header = {
            "type": "height",
            "version": PROTOCOL_VERSION,
            "index": self._index,
            "unit": "mm",
            "cropped": bool(cropped),
            "origin": "reference" if reference else "median",
            "waterLevelMm": None if water_level_mm is None else float(water_level_mm),
        }

        with self._lock:
            if self._pending is not None and self.connected:
                self.dropped += 1
            self._pending = (header, frame)
        self._wakeup.set()
        return True

    def close(self):
        """送信をやめて片付ける。何度呼んでも構わない。"""
        if self._thread is None:
            return
        self._stop_event.set()
        self._wakeup.set()
        self._thread.join(timeout=_JOIN_TIMEOUT_S)
        self._thread = None

    def describe(self):
        """いまの状況を 1 行で表す。画面やコンソールへ出す。"""
        state = "送信中" if self.connected else "未接続（つながるまで試し続けます）"
        parts = [f"ライブ配信: {state} {self.url}", f"送信 {self.sent} 枚"]
        if self.dropped:
            parts.append(f"（間に合わず飛ばした {self.dropped}）")
        return " ".join(parts)

    # ------------------------------------------------------------------
    def _take_pending(self):
        with self._lock:
            item, self._pending = self._pending, None
        return item

    def _run(self):
        """送信スレッドの本体。

        **ここから例外を外へ出さない。** スレッドが死ぬと、以後は黙って
        何も送らなくなる。websocket-client はソケットの失敗を
        ``OSError`` と自前の例外の両方で出すため、まとめて受ける。
        """
        connection = None
        # 「つながらない」を 1 度だけ知らせるための印。つながったら下ろす。
        waiting_announced = False

        while not self._stop_event.is_set():
            if connection is None:
                try:
                    connection = self._connect(self.url, self.timeout_s)
                except Exception as error:
                    self.last_error = error
                    if not waiting_announced:
                        self._notice = (
                            f"ライブ配信: {self.url} につながりません。つながるまで試し続けます"
                        )
                        waiting_announced = True
                    self._stop_event.wait(self.retry_s)
                    continue
                self.connected = True
                waiting_announced = False
                self._notice = f"ライブ配信: {self.url} へ送っています"

            self._wakeup.wait(_POLL_S)
            self._wakeup.clear()
            item = self._take_pending()

            try:
                if not connection.poll():
                    raise ConnectionError("相手が接続を閉じました")
                if item is not None:
                    header, frame = item
                    connection.send(encode_frame(frame, header))
                    self.sent += 1
            except Exception as error:
                self.last_error = error
                self.connected = False
                connection.close()
                connection = None
                if not self._stop_event.is_set():
                    self._notice = f"ライブ配信が切れました（{error}）。つなぎ直します"
                    # 続けてつながらなくても、もう 1 度は知らせない。
                    waiting_announced = True
                    self._stop_event.wait(self.retry_s)

        if connection is not None:
            connection.close()
        self.connected = False
