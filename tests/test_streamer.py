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
from topo_sandbox import renderer as renderer_module
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
    header, heights, _, header_len = _decode_with_flood(message)
    return header, heights, header_len


def _decode_with_flood(message):
    """水深も読む。水深が無ければ None。"""
    header, heights, flood, _, header_len = _decode_layers(message)
    return header, heights, flood, header_len


def _decode_layers(message):
    """高さ・水深・川らしさをすべて読む。無いものは None。"""
    (header_len,) = struct.unpack_from("<I", message, 0)
    header = json.loads(message[4 : 4 + header_len].decode("utf-8"))
    rows, cols = header["rows"], header["cols"]
    names = ["heights"] + [key for key in ("flood", "rivers") if header.get(key)]
    data = np.frombuffer(message, dtype="<f4", offset=4 + header_len)
    assert data.size == len(names) * rows * cols
    grids = dict(zip(names, data.reshape(len(names), rows, cols), strict=True))
    return header, grids["heights"], grids.get("flood"), grids.get("rivers"), header_len


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

    def test_水深は高さの後ろに続ける(self):
        heights = _height(3.0, (2, 3))
        water = np.arange(6, dtype=np.float32).reshape(2, 3)
        header, decoded, flood, _ = _decode_with_flood(encode_frame(heights, {}, water))

        assert header["flood"] is True
        np.testing.assert_array_equal(decoded, heights)
        np.testing.assert_array_equal(flood, water)

    def test_水深が無ければfloodを付けない(self):
        """受け取る側は flood が無ければ高さだけだと読む。送る量も増やさない。"""
        message = encode_frame(_height(1.0, (2, 2)), {})
        header, _, flood, _ = _decode_with_flood(message)
        assert "flood" not in header
        assert flood is None
        assert len(message) == 4 + struct.unpack_from("<I", message, 0)[0] + 2 * 2 * 4

    def test_水深の大きさが高さと違えば断る(self):
        with pytest.raises(ValueError):
            encode_frame(_height(1.0, (2, 2)), {}, _height(0.0, (2, 3)))

    def test_川らしさは水深の後ろに続ける(self):
        heights = _height(3.0, (2, 3))
        water = np.arange(6, dtype=np.float32).reshape(2, 3)
        strength = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(2, 3)
        header, decoded, flood, river, _ = _decode_layers(
            encode_frame(heights, {}, water, strength)
        )

        assert header["flood"] is True and header["rivers"] is True
        np.testing.assert_array_equal(decoded, heights)
        np.testing.assert_array_equal(flood, water)
        np.testing.assert_array_equal(river, strength)

    def test_川らしさだけでも送れる(self):
        """基準面が無いときは水を流さずに川だけを描く。"""
        strength = np.full((2, 3), 0.5, dtype=np.float32)
        header, _, flood, river, _ = _decode_layers(
            encode_frame(_height(1.0, (2, 3)), {}, river=strength)
        )
        assert "flood" not in header and header["rivers"] is True
        assert flood is None
        np.testing.assert_array_equal(river, strength)

    def test_川らしさが無ければriversを付けない(self):
        header, _, _, river, _ = _decode_layers(encode_frame(_height(1.0, (2, 2)), {}))
        assert "rivers" not in header
        assert river is None

    def test_川らしさの大きさが高さと違えば断る(self):
        with pytest.raises(ValueError):
            encode_frame(_height(1.0, (2, 2)), {}, river=_height(0.0, (2, 3)))

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

    def test_水深も届く(self, streamer, server):
        water = _height(4.0)
        streamer.send(_height(1.0), reference=True, flood_mm=water)
        assert _wait_until(lambda: server.latest.sent)

        header, _, flood, _ = _decode_with_flood(server.latest.sent[0])
        assert header["flood"] is True
        np.testing.assert_array_equal(flood, water)

    def test_川らしさも届く(self, streamer, server):
        strength = _height(0.75)
        streamer.send(_height(1.0), reference=True, river=strength)
        assert _wait_until(lambda: server.latest.sent)

        header, _, flood, river, _ = _decode_layers(server.latest.sent[0])
        assert header["rivers"] is True
        assert flood is None
        np.testing.assert_array_equal(river, strength)

    def test_等高線の間隔も届く(self, streamer, server):
        streamer.send(_height(1.0), reference=True, contour_interval_mm=10)
        assert _wait_until(lambda: server.latest.sent)

        header, _, flood, river, _ = _decode_layers(server.latest.sent[0])
        assert header["contourIntervalMm"] == 10.0
        # 線は送らない。受け取る側が高さから引く。
        assert flood is None and river is None

    def test_等高線を引いていなければ間隔はnull(self, streamer, server):
        streamer.send(_height(1.0), reference=True)
        assert _wait_until(lambda: server.latest.sent)
        header, _, _ = _decode(server.latest.sent[0])
        assert header["contourIntervalMm"] is None

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

    def send(
        self,
        height_mm,
        cropped=False,
        reference=False,
        water_level_mm=None,
        flood_mm=None,
        river=None,
        contour_interval_mm=None,
    ):
        self.calls.append(
            {
                "height_mm": np.array(height_mm),
                "cropped": cropped,
                "reference": reference,
                "water_level_mm": water_level_mm,
                "flood_mm": None if flood_mm is None else np.array(flood_mm),
                "river": None if river is None else np.array(river),
                "contour_interval_mm": contour_interval_mm,
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


class _StubFlood:
    """決まった水深を返す水の身代わり。"""

    def __init__(self, water):
        self.water = water

    def update(self, height_mm, elapsed_s, **kwargs):
        return self.water

    def reset(self):
        pass


def _left_half_water():
    """センサから見て左半分に 5mm の水がある水深[mm]（処理解像度）。"""
    width, height = config.PROC_SIZE
    water = np.zeros((height, width), dtype=np.float32)
    water[:, : width // 2] = 5.0
    return water


def _flood_settings(area=None):
    settings = _settings(area=area)
    settings.show_flood = True
    settings.spring_position = [0.5, 0.5]
    return settings


@pytest.fixture
def stub_rivers(monkeypatch):
    """川の計算を「センサから見て左半分だけが川（0.8）」に置き換え、呼ばれ方を覚える。"""
    calls = []

    def fake(height_mm, min_cells=None, full_cells=None):
        calls.append((min_cells, full_cells))
        strength = np.zeros(np.shape(height_mm), dtype=np.float32)
        strength[:, : strength.shape[1] // 2] = 0.8
        return strength

    monkeypatch.setattr(renderer_module.rivers, "river_strength", fake)
    return calls


def _river_settings(area=None, reference=True):
    settings = _settings(area=area)
    settings.show_rivers = True
    if not reference:
        settings.reference_plane = None
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

    def test_水を流していなければ水深は送らない(self):
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _settings())
        assert spy.calls[0]["flood_mm"] is None

    def test_水源の水を投影と同じ向きで送る(self):
        """高さと同じく左右反転して送る。揃えないと水が地形の反対側に乗る。"""
        spy = _SpyStreamer()
        renderer = Renderer(streamer=spy)
        renderer._flood = _StubFlood(_left_half_water())
        renderer.render(_sensor_frame(), _flood_settings())

        flood = spy.calls[0]["flood_mm"]
        assert flood.shape == spy.calls[0]["height_mm"].shape
        width = flood.shape[1]
        # センサの左半分の水は、投影像では右半分に来る（盛った砂と同じ側）
        assert flood[:, width * 3 // 4].mean() == pytest.approx(5.0)
        assert flood[:, width // 4].mean() == pytest.approx(0.0)

    def test_切り取るときは水も同じく切り取る(self):
        width, height = config.VIEW_SIZE
        whole = [[0, 0], [0, height], [width, height], [width, 0]]
        spy = _SpyStreamer()
        renderer = Renderer(streamer=spy)
        renderer._flood = _StubFlood(_left_half_water())
        renderer.render(_sensor_frame(), _flood_settings(area=whole))

        call = spy.calls[0]
        assert call["cropped"] is True
        flood, heights = call["flood_mm"], call["height_mm"]
        assert flood.shape == heights.shape
        columns = flood.shape[1]
        # 水のある側と盛った砂の側が一致する
        assert flood[:, columns * 3 // 4].mean() == pytest.approx(5.0, abs=0.1)
        assert heights[:, columns * 3 // 4].mean() == pytest.approx(50.0, abs=1.0)
        assert flood[:, columns // 4].mean() == pytest.approx(0.0, abs=0.1)

    def test_川を描いていなければ川らしさは送らない(self):
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _settings())
        assert spy.calls[0]["river"] is None

    def test_川を投影と同じ向きで送る(self, stub_rivers):
        """高さと同じく左右反転して送る。揃えないと川が地形の反対側に乗る。"""
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _river_settings())

        river = spy.calls[0]["river"]
        assert river.shape == spy.calls[0]["height_mm"].shape
        width = river.shape[1]
        # センサの左半分の川は、投影像では右半分に来る（盛った砂と同じ側）
        assert river[:, width * 3 // 4].mean() == pytest.approx(0.8)
        assert river[:, width // 4].mean() == pytest.approx(0.0)
        # 投影と配信で同じ川を使う。重いので 1 フレームで 2 度は数えない。
        assert len(stub_rivers) == 1

    def test_川の出やすさを投影と揃える(self, stub_rivers):
        settings = _river_settings()
        settings.cycle_river_preset()
        Renderer(streamer=_SpyStreamer()).render(_sensor_frame(), settings)
        _, min_cells, full_cells = settings.river_setting
        assert stub_rivers == [(min_cells, full_cells)]

    def test_切り取るときは川も同じく切り取る(self, stub_rivers):
        width, height = config.VIEW_SIZE
        whole = [[0, 0], [0, height], [width, height], [width, 0]]
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _river_settings(area=whole))

        call = spy.calls[0]
        assert call["cropped"] is True
        river, heights = call["river"], call["height_mm"]
        assert river.shape == heights.shape
        columns = river.shape[1]
        # 川のある側と盛った砂の側が一致する
        assert river[:, columns * 3 // 4].mean() == pytest.approx(0.8, abs=0.01)
        assert heights[:, columns * 3 // 4].mean() == pytest.approx(50.0, abs=1.0)
        assert river[:, columns // 4].mean() == pytest.approx(0.0, abs=0.01)

    def test_水面より下の川は送らない(self, stub_rivers):
        """投影でも水面より下の川は描かない（湖や海に入った川は見えなくなる）。"""
        settings = _river_settings()
        # 盛った側（+50mm）だけが水面より上に出る
        settings.water_level_mm = 25.0
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), settings)

        river, heights = spy.calls[0]["river"], spy.calls[0]["height_mm"]
        assert river is not None and river.max() > 0.0
        assert np.all(river[heights < settings.water_level_mm] == 0.0)

    def test_水源の水を流しているあいだは川を送らない(self, stub_rivers):
        """投影でも描かない。どちらが本物の水か見分けられなくなるため。"""
        settings = _river_settings()
        settings.show_flood = True
        settings.spring_position = [0.5, 0.5]
        spy = _SpyStreamer()
        renderer = Renderer(streamer=spy)
        renderer._flood = _StubFlood(_left_half_water())
        renderer.render(_sensor_frame(), settings)

        assert spy.calls[0]["river"] is None
        assert spy.calls[0]["flood_mm"] is not None
        assert stub_rivers == []

    def test_基準面が無くても川は送る(self, stub_rivers):
        """流量は下る向きだけで決まるので、投影でも基準面なしで川を描いている。"""
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _river_settings(reference=False))
        assert spy.calls[0]["reference"] is False
        assert spy.calls[0]["river"] is not None

    def test_DEM以外では川を送らない(self, stub_rivers):
        """川を描くのは DEM 表示だけ。投影に無い川を別画面にだけ出さない。"""
        settings = _river_settings()
        settings.view_mode = ViewMode.DEPTH
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), settings)
        assert spy.calls[0]["river"] is None
        assert stub_rivers == []

    def test_等高線を引いていなければ間隔は送らない(self):
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), _settings())
        assert spy.calls[0]["contour_interval_mm"] is None

    @pytest.mark.parametrize("mode", [ViewMode.COLORING, ViewMode.DEM, ViewMode.DEPTH])
    def test_等高線を引いていれば間隔を送る(self, mode):
        settings = _settings(mode)
        settings.show_contour = True
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), settings)
        assert spy.calls[0]["contour_interval_mm"] == config.CONTOUR_INTERVAL_MM

    def test_エッジ表示では等高線の間隔を送らない(self):
        """エッジ表示は等高線を重ねない。投影に無い線を別画面にだけ出さない。"""
        settings = _settings(ViewMode.EDGE)
        settings.show_contour = True
        spy = _SpyStreamer()
        Renderer(streamer=spy).render(_sensor_frame(), settings)
        assert spy.calls[0]["contour_interval_mm"] is None

    def test_起伏が広ければ投影と同じく広げた間隔を送る(self):
        """本数の上限で投影の間隔が広がったとき、別画面だけ細かい線にならないこと。"""
        width, height = config.SENSOR_SIZE
        frame = np.full((height, width), _BASE_MM, dtype=np.float32)
        # 400mm 盛る。5mm 間隔では上限の本数を超える。
        mound_mm = 400.0
        assert mound_mm > config.CONTOUR_INTERVAL_MM * config.CONTOUR_LEVELS
        frame[:, : width // 2] = _BASE_MM - mound_mm
        settings = _settings()
        settings.show_contour = True
        spy = _SpyStreamer()
        renderer = Renderer(streamer=spy)
        renderer.render(frame, settings)

        sent = spy.calls[0]["contour_interval_mm"]
        assert sent > config.CONTOUR_INTERVAL_MM
        assert sent == renderer._contours._interval

    def test_配信なしでも描ける(self):
        """普段の起動は streamer が None。ここが落ちると実演が止まる。"""
        image = Renderer().render(_sensor_frame(), _settings())
        assert image.shape[:2] == (config.VIEW_SIZE[1], config.VIEW_SIZE[0])

    def test_閉じると配信も閉じる(self):
        spy = _SpyStreamer()
        Renderer(streamer=spy).close()
        assert spy.calls == ["closed"]
