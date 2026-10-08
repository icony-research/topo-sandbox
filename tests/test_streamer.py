"""ライブ表示アプリへの配信（--stream）の検証。GPU も Kinect も通信相手も不要。

受け取る側（topo-sandbox-live）はメッセージの並びとヘッダだけを手掛かりにするので、
ここで形を固定しておく。通信は身代わりの接続で置き換える。
"""

import json
import struct
import threading
import time

import numpy as np
import pytest

from topo_sandbox import config
from topo_sandbox.processing.plane import ReferencePlane
from topo_sandbox.renderer import Renderer, RenderSettings, ViewMode
from topo_sandbox.streamer import PROTOCOL_VERSION, DemStreamer, encode_frame

_BASE_MM = 1000.0

#: 表示像での砂場の四隅（左上から反時計回り）
_VIEW_AREA = [[100, 50], [100, 550], [700, 550], [700, 50]]

#: 身代わりのスレッドが動き終わるのを待つ上限[秒]
_WAIT_S = 2.0


def _decode(message):
    """encode_frame の逆。受け取る側と同じ読み方をする。"""
    (header_len,) = struct.unpack_from("<I", message, 0)
    header = json.loads(message[4 : 4 + header_len].decode("utf-8"))
    heights = np.frombuffer(message, dtype="<f4", offset=4 + header_len)
    return header, heights.reshape(header["rows"], header["cols"]), header_len


def _wait_until(predicate):
    deadline = time.monotonic() + _WAIT_S
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _FakeConnection:
    """送ったものを覚えておく身代わりの接続。"""

    def __init__(self):
        self.sent = []
        self.closed = False
        self.open = True
        self.fail_send = False
        self.block = None  # threading.Event。立つまで send で止まる

    def send(self, payload):
        if self.block is not None:
            self.block.wait(_WAIT_S)
        if self.fail_send:
            raise OSError("送れません")
        self.sent.append(payload)

    def poll(self):
        return self.open

    def close(self):
        self.closed = True


class _FakeServer:
    """connect の身代わり。つないだ接続を順に覚え、つなげない回数も指定できる。"""

    def __init__(self, refuse=0):
        self.connections = []
        self.refuse = refuse
        self.attempts = 0

    def __call__(self, url, timeout_s):
        self.attempts += 1
        if self.attempts <= self.refuse:
            raise ConnectionRefusedError("相手がいません")
        connection = _FakeConnection()
        self.connections.append(connection)
        return connection

    @property
    def latest(self):
        return self.connections[-1]


@pytest.fixture
def server():
    return _FakeServer()


@pytest.fixture
def streamer(server):
    """試験用の配信。終わったら必ず閉じる（スレッドを残さない）。"""
    instance = DemStreamer(url="ws://test/ws/ingest", retry_s=0.01, connect=server)
    instance.start()
    assert _wait_until(lambda: instance.connected)
    yield instance
    instance.close()


def _height(value, shape=(240, 320)):
    return np.full(shape, value, dtype=np.float32)


class TestEncode:
    """受け取る側と取り決めた形。"""

    def test_ヘッダと高さがそのまま読める(self):
        heights = np.arange(12, dtype=np.float32).reshape(3, 4) - 5.5
        header, decoded, _ = _decode(encode_frame(heights, {"type": "height"}))

        assert header == {"type": "height", "rows": 3, "cols": 4}
        np.testing.assert_array_equal(decoded, heights)

    def test_高さの始まりが4バイト境界に揃う(self):
        """ブラウザが写さずに Float32Array にできるようにするため。"""
        for extra in range(8):
            message = encode_frame(_height(1.0, (2, 2)), {"pad": "x" * extra})
            _, _, header_len = _decode(message)
            assert (4 + header_len) % 4 == 0

    def test_float64でもfloat32で送る(self):
        heights = np.full((2, 3), 1.25, dtype=np.float64)
        message = encode_frame(heights, {})
        _, decoded, _ = _decode(message)
        assert decoded.dtype == np.dtype("<f4")
        assert len(message) % 4 == 0


class TestSend:
    def test_送った高さとヘッダが届く(self, streamer, server):
        streamer.send(_height(12.5), cropped=True, reference=True, water_level_mm=-40)
        assert _wait_until(lambda: server.latest.sent)

        header, heights, _ = _decode(server.latest.sent[0])
        assert header["type"] == "height"
        assert header["version"] == PROTOCOL_VERSION
        assert header["index"] == 1
        assert header["unit"] == "mm"
        assert header["cropped"] is True
        assert header["origin"] == "reference"
        assert header["waterLevelMm"] == -40.0
        assert heights.shape == (240, 320)
        assert float(heights[0, 0]) == 12.5

    def test_基準面が無ければ原点が中央値だと分かる(self, streamer, server):
        streamer.send(_height(0.0))
        assert _wait_until(lambda: server.latest.sent)
        header, _, _ = _decode(server.latest.sent[0])
        assert header["origin"] == "median"
        assert header["cropped"] is False

    def test_渡したあとで配列を書き換えても送るものは変わらない(self, server):
        """送信スレッドが呼び出し側の配列を持ち続けないこと。"""
        block = threading.Event()
        instance = DemStreamer(retry_s=0.01, connect=server)
        instance.start()
        try:
            assert _wait_until(lambda: instance.connected)
            server.latest.block = block
            frame = _height(1.0)
            instance.send(frame)
            frame[:] = 99.0
            block.set()
            assert _wait_until(lambda: server.latest.sent)
            _, heights, _ = _decode(server.latest.sent[0])
            assert float(heights[0, 0]) == 1.0
        finally:
            block.set()
            instance.close()

    def test_送信が間に合わなければ新しい1枚だけを送る(self, server):
        """溜め込むとライブ表示が砂場から遅れていく。"""
        block = threading.Event()
        instance = DemStreamer(retry_s=0.01, connect=server)
        instance.start()
        try:
            assert _wait_until(lambda: instance.connected)
            connection = server.latest
            connection.block = block

            instance.send(_height(1.0))
            # 1 枚目が送信中（block で止まっている）になるのを待つ
            assert _wait_until(lambda: instance._pending is None)
            for value in (2.0, 3.0, 4.0):
                assert instance.send(_height(value)) is True
            block.set()

            assert _wait_until(lambda: len(connection.sent) == 2)
            values = [float(_decode(m)[1][0, 0]) for m in connection.sent]
            assert values == [1.0, 4.0]
            assert instance.dropped == 2
        finally:
            block.set()
            instance.close()

    def test_sendは待たない(self, server):
        """描画のワーカスレッドから呼ばれる。待つと投影が固まる。"""
        block = threading.Event()
        instance = DemStreamer(retry_s=0.01, connect=server)
        instance.start()
        try:
            assert _wait_until(lambda: instance.connected)
            server.latest.block = block
            started = time.monotonic()
            for _ in range(20):
                instance.send(_height(1.0))
            assert time.monotonic() - started < 0.5
        finally:
            block.set()
            instance.close()


class TestReconnect:
    def test_相手がいなくてもつながるまで試し続ける(self):
        server = _FakeServer(refuse=3)
        instance = DemStreamer(retry_s=0.01, connect=server)
        instance.start()
        try:
            assert _wait_until(lambda: instance.connected)
            assert server.attempts == 4
        finally:
            instance.close()

    def test_つながらないことは1度だけ知らせる(self):
        """相手が起動するまで毎回知らせると、投影像が文字で埋まる。"""
        server = _FakeServer(refuse=5)
        instance = DemStreamer(retry_s=0.01, connect=server)
        notices = []

        def collect():
            notice = instance.take_notice()
            if notice:
                notices.append(notice)
            return instance.connected

        instance.start()
        try:
            assert _wait_until(collect)
            collect()
        finally:
            instance.close()

        assert sum("つながりません" in n for n in notices) == 1
        assert "送っています" in notices[-1]

    def test_送れなくなったらつなぎ直す(self, streamer, server):
        first = server.latest
        first.fail_send = True
        streamer.send(_height(1.0))
        assert _wait_until(lambda: len(server.connections) == 2 and streamer.connected)
        assert first.closed
        assert isinstance(streamer.last_error, OSError)

        streamer.send(_height(2.0))
        assert _wait_until(lambda: server.latest.sent)

    def test_相手が閉じたらつなぎ直す(self, streamer, server):
        server.latest.open = False
        assert _wait_until(lambda: len(server.connections) == 2 and streamer.connected)

    def test_切れたことを知らせる(self, streamer, server):
        streamer.take_notice()
        server.latest.fail_send = True
        streamer.send(_height(1.0))
        assert _wait_until(lambda: len(server.connections) == 2)
        # 切れた知らせのあと、つながった知らせで上書きされることがある。
        # どちらにせよ黙ってはいない。
        assert streamer.take_notice() is not None


class TestGuards:
    def test_始めていなければ受け付けない(self):
        instance = DemStreamer(connect=_FakeServer())
        assert instance.send(_height(1.0)) is False

    def test_閉じたあとに送っても落ちない(self, server):
        instance = DemStreamer(retry_s=0.01, connect=server)
        instance.start()
        instance.close()
        assert instance.send(_height(1.0)) is False
        assert instance.connected is False

    def test_何度閉じても平気(self, streamer):
        streamer.close()
        streamer.close()

    def test_閉じると接続も閉じる(self, streamer, server):
        streamer.close()
        assert server.latest.closed

    def test_相手がいないままでもすぐ閉じられる(self):
        """つなぎ直しの待ちで、アプリの終了が遅れないこと。"""
        instance = DemStreamer(retry_s=60.0, connect=_FakeServer(refuse=10**9))
        instance.start()
        time.sleep(0.05)
        started = time.monotonic()
        instance.close()
        assert time.monotonic() - started < 1.0

    def test_状況を1行で表せる(self, streamer):
        assert "送信中" in streamer.describe()


# ----------------------------------------------------------------------
# 描画との組み合わせ
# ----------------------------------------------------------------------
class _SpyStreamer:
    """送られ方だけを見る身代わり。"""

    def __init__(self):
        self.calls = []

    def send(self, height_mm, cropped=False, reference=False, water_level_mm=None):
        self.calls.append(
            {
                "height_mm": np.array(height_mm),
                "cropped": cropped,
                "reference": reference,
                "water_level_mm": water_level_mm,
            }
        )
        return True

    def close(self):
        self.calls.append("closed")


def _plane():
    return ReferencePlane(
        a=0.0,
        b=0.0,
        c=_BASE_MM,
        residual_mm=1.0,
        tilt_deg=0.0,
        distance_mm=_BASE_MM,
        coverage=99.0,
    )


def _sensor_frame():
    """左半分（センサから見て）だけ 50mm 盛った砂場の深度[mm] (480, 640)。"""
    width, height = config.SENSOR_SIZE
    frame = np.full((height, width), _BASE_MM, dtype=np.float32)
    frame[:, : width // 2] = _BASE_MM - 50.0
    return frame


def _settings(view_mode=ViewMode.DEM, area=None):
    settings = RenderSettings()
    settings.view_mode = view_mode
    settings.reference_plane = _plane()
    if area:
        settings.area_positions = [list(point) for point in area]
    return settings


class TestRendererIntegration:
    @pytest.mark.parametrize("mode", [ViewMode.DEM, ViewMode.DEPTH])
    def test_表示モードを問わず送る(self, mode):
        """実演中に v で切り替えるたびに、別画面の地形が止まると困る。"""
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _settings(mode))

        assert len(spy.calls) == 1
        assert spy.calls[0]["height_mm"].shape == (config.PROC_SIZE[1], config.PROC_SIZE[0])
        assert spy.calls[0]["reference"] is True

    def test_エリアが無ければ視野全体を投影と同じ向きで送る(self):
        """センサから見たままだと、投影像（左右反転後）と左右が逆になる。"""
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _settings())

        call = spy.calls[0]
        assert call["cropped"] is False
        heights = call["height_mm"]
        width = heights.shape[1]
        # センサの左半分に盛った砂は、投影像では右半分に来る
        assert heights[:, width * 3 // 4].mean() == pytest.approx(50.0, abs=1.0)
        assert heights[:, width // 4].mean() == pytest.approx(0.0, abs=1.0)

    def test_エリアがあれば内側だけを切り取って送る(self):
        spy = _SpyStreamer()
        settings = _settings(area=_VIEW_AREA)
        Renderer(streamer=spy).render(_sensor_frame(), settings)

        call = spy.calls[0]
        assert call["cropped"] is True
        assert call["height_mm"].shape == (config.PROC_SIZE[1], config.PROC_SIZE[0])

    def test_切り取りの有無で左右が入れ替わらない(self):
        """四隅を指定した瞬間にライブ表示の地形が裏返らないこと。"""
        width, height = config.VIEW_SIZE
        whole = [[0, 0], [0, height], [width, height], [width, 0]]

        uncropped, cropped = _SpyStreamer(), _SpyStreamer()
        Renderer(streamer=uncropped).render(_sensor_frame(), _settings())
        Renderer(streamer=cropped).render(_sensor_frame(), _settings(area=whole))

        # 盛った境目は切り取りの補間でなまるので、境目から離れたところを比べる
        for call in (uncropped.calls[0], cropped.calls[0]):
            heights = call["height_mm"]
            width = heights.shape[1]
            assert heights[:, width // 4].mean() == pytest.approx(0.0, abs=1.0)
            assert heights[:, width * 3 // 4].mean() == pytest.approx(50.0, abs=1.0)

    def test_水位も一緒に送る(self):
        spy = _SpyStreamer()
        settings = _settings()
        settings.water_level_mm = -25.0
        Renderer(streamer=spy).render(_sensor_frame(), settings)
        assert spy.calls[0]["water_level_mm"] == -25.0

    def test_配信なしでも描ける(self):
        """普段の起動は streamer が None。ここが落ちると実演が止まる。"""
        image = Renderer().render(_sensor_frame(), _settings())
        assert image.shape[:2] == (config.VIEW_SIZE[1], config.VIEW_SIZE[0])

    def test_閉じると配信も閉じる(self):
        spy = _SpyStreamer()
        Renderer(streamer=spy).close()
        assert spy.calls == ["closed"]
