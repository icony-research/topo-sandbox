"""DEM の高さ[mm]の書き出しの検証。GPU も Kinect も不要。

書き出したものは別アプリ（点群表示）のテスト入力になる。読み込む側は
``meta.json`` の形と ``index.csv`` の経過秒だけを手掛かりにするので、
ここで形を固定しておく。
"""

import csv
import json

import numpy as np
import pytest

from topo_sandbox import config
from topo_sandbox.processing.plane import ReferencePlane
from topo_sandbox.recorder import DemRecorder, frame_context
from topo_sandbox.renderer import Renderer, RenderSettings, ViewMode, crop_to_area

_BASE_MM = 1000.0

#: 表示像での砂場の四隅（左上から反時計回り）。切り取りの試験で使う。
_VIEW_AREA = [[100, 50], [100, 550], [700, 550], [700, 50]]


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


def _height(value, shape=None):
    """一様な高さ[mm]の画像。"""
    if shape is None:
        width, height = config.PROC_SIZE
        shape = (height, width)
    return np.full(shape, value, dtype=np.float32)


@pytest.fixture
def recorder(tmp_path):
    """試験用の録画。終わったら必ず閉じる（スレッドを残さない）。"""
    instance = DemRecorder(directory=tmp_path, max_frames=0)
    instance.open()
    yield instance
    instance.close()


def _frames(path):
    return sorted(path.glob("frame_*.npy"))


def _index_rows(path):
    with (path / "index.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestRoundTrip:
    """書き出したものが、そのまま高さ[mm]として読み戻せること。"""

    def test_高さがそのまま読める(self, recorder):
        recorder.write(_height(12.5), 0.0, frame_context(_plane(), -40.0))
        recorder.close()

        saved = _frames(recorder.path)
        assert len(saved) == 1

        restored = np.load(saved[0])
        assert restored.dtype == np.float32
        assert restored.shape == (config.PROC_SIZE[1], config.PROC_SIZE[0])
        np.testing.assert_allclose(restored, 12.5)

    def test_負の高さも符号そのまま(self, recorder):
        """掘ったところは負。符号を失うと水面より下が表せない。"""
        recorder.write(_height(-37.0), 0.0)
        recorder.close()

        np.testing.assert_allclose(np.load(_frames(recorder.path)[0]), -37.0)

    def test_連番は1から詰めて振る(self, recorder):
        for index in range(3):
            recorder.write(_height(float(index)), index * 0.1)
        recorder.close()

        names = [path.name for path in _frames(recorder.path)]
        assert names == ["frame_000001.npy", "frame_000002.npy", "frame_000003.npy"]


class TestIndex:
    """経過秒の控え。受け取る側はこれを見て実際の間隔で再生する。"""

    def test_経過秒が記録される(self, recorder):
        recorder.write(_height(0.0), 0.0)
        recorder.write(_height(1.0), 0.25)
        recorder.close()

        rows = _index_rows(recorder.path)
        assert [row["frame"] for row in rows] == ["1", "2"]
        assert [float(row["elapsed_s"]) for row in rows] == [0.0, 0.25]


class TestMeta:
    """読み込む側の手掛かり。"""

    def _meta(self, recorder):
        return json.loads((recorder.path / "meta.json").read_text(encoding="utf-8"))

    def test_形と単位と符号が入る(self, recorder):
        recorder.write(_height(1.0), 0.0, frame_context(_plane(), -40.0))
        recorder.close()

        meta = self._meta(recorder)
        assert meta["shape"] == [config.PROC_SIZE[1], config.PROC_SIZE[0]]
        assert meta["dtype"] == "float32"
        assert meta["unit"] == "mm"
        assert meta["frames"] == 1
        assert meta["cropped"] is False

    def test_切り取って録ったことが分かる(self, recorder):
        """切り取りの有無で向きの決まりが変わる。見分けられないと鏡像になる。"""
        recorder.write(_height(1.0), 0.0, frame_context(_plane(), -40.0, _VIEW_AREA))
        recorder.close()

        meta = self._meta(recorder)
        assert meta["cropped"] is True
        assert meta["crop_area_positions"] == _VIEW_AREA
        assert "長方形" in meta["axes"]

    def test_基準面があれば原点が基準面になる(self, recorder):
        """高さの原点が何かで、受け取る側の扱いが変わる。"""
        recorder.write(_height(1.0), 0.0, frame_context(_plane(), -40.0))
        recorder.close()

        meta = self._meta(recorder)
        assert meta["origin"] == "基準面"
        assert meta["reference_plane"]["c"] == pytest.approx(_BASE_MM)
        assert meta["water_level_mm"] == pytest.approx(-40.0)

    def test_基準面が無ければ中央値が原点だと分かる(self, recorder):
        """基準面なしで録ると、砂を動かすたびに高さの原点が動く。

        あとから見分けられないと、受け取る側が絶対高さとして扱ってしまう。
        """
        recorder.write(_height(1.0), 0.0, frame_context(None, -40.0))
        recorder.close()

        meta = self._meta(recorder)
        assert meta["origin"] == "フレームの中央値"
        assert meta["reference_plane"] is None

    def test_始めた時点でも書かれている(self, recorder):
        """強制終了されても、何の記録かは分かるようにしておく。"""
        assert self._meta(recorder)["version"] >= 1


class TestLimit:
    """止め忘れでディスクを使い切らないこと。"""

    def test_上限で受け付けをやめる(self, tmp_path):
        recorder = DemRecorder(directory=tmp_path, max_frames=2)
        recorder.open()
        try:
            assert recorder.write(_height(0.0), 0.0)
            assert recorder.write(_height(1.0), 0.1)
            assert not recorder.write(_height(2.0), 0.2)
            assert not recorder.active
        finally:
            recorder.close()

        assert len(_frames(recorder.path)) == 2

    def test_上限に達したら知らせが出る(self, tmp_path):
        """黙って止まると、あとでファイルを見るまで気付けない。"""
        recorder = DemRecorder(directory=tmp_path, max_frames=1)
        recorder.open()
        try:
            recorder.write(_height(0.0), 0.0)
            notice = recorder.take_notice()
            assert notice is not None and "上限" in notice
            # 知らせは 1 度だけ。毎フレーム出すと画面が埋まる。
            assert recorder.take_notice() is None
        finally:
            recorder.close()

    def test_0なら無制限(self, recorder):
        for index in range(5):
            assert recorder.write(_height(float(index)), index * 0.1)
        assert recorder.active


class TestGuards:
    """実演中に落とさないための守り。"""

    def test_開いていなければ書かずに済ませる(self, tmp_path):
        recorder = DemRecorder(directory=tmp_path)
        assert not recorder.write(_height(0.0), 0.0)
        assert not recorder.active

    def test_閉じたあとに書いても落ちない(self, tmp_path):
        """ワーカスレッドが 1 フレーム遅れて書きに来ることがある。"""
        recorder = DemRecorder(directory=tmp_path)
        recorder.open()
        recorder.close()
        assert not recorder.write(_height(0.0), 0.0)

    def test_何度閉じても平気(self, tmp_path):
        recorder = DemRecorder(directory=tmp_path)
        recorder.open()
        recorder.close()
        recorder.close()

    def test_開いていなくても閉じられる(self, tmp_path):
        DemRecorder(directory=tmp_path).close()

    def test_積めないフレームは捨てる(self, tmp_path):
        """書き込みが追いつかないときに溜め込むと、描画まで遅れる。

        書き込みスレッドを起こさないまま積むことで、あふれる状態を作る。
        """
        recorder = DemRecorder(directory=tmp_path, max_frames=0, queue_size=2)
        recorder.path = tmp_path
        recorder._stopped = False  # スレッドを起こさずに満杯を作る

        assert recorder.write(_height(0.0), 0.0)
        assert recorder.write(_height(1.0), 0.1)
        assert not recorder.write(_height(2.0), 0.2)
        assert recorder.dropped == 1
        # 捨てたぶんは番号を進めない。ファイルの連番に穴を空けないため。
        assert recorder.accepted == 2


class _SpyRecorder:
    """書き出しの呼ばれ方だけを見る身代わり。"""

    def __init__(self):
        self.calls = []
        self.warnings = []
        self.active = True

    def write(self, height_mm, elapsed_s, context=None):
        self.calls.append((np.array(height_mm), elapsed_s, context))
        return True

    def warn(self, message):
        self.warnings.append(message)

    def close(self):
        self.calls.append("closed")


def _sensor_frame():
    """平らな砂場の深度[mm] (480, 640)。"""
    width, height = config.SENSOR_SIZE
    return np.full((height, width), _BASE_MM, dtype=np.float32)


class TestRendererIntegration:
    """DEM 表示のあいだだけ書き出すこと。"""

    def _settings(self, view_mode):
        settings = RenderSettings()
        settings.view_mode = view_mode
        settings.reference_plane = _plane()
        return settings

    def test_DEMでは書き出す(self):
        spy = _SpyRecorder()
        Renderer(recorder=spy).render(_sensor_frame(), self._settings(ViewMode.DEM))

        assert len(spy.calls) == 1
        height_mm, _, context = spy.calls[0]
        assert height_mm.shape == (config.PROC_SIZE[1], config.PROC_SIZE[0])
        assert context["reference_plane"] is not None

    def test_DEM以外では書き出さない(self):
        """点群にしたいのは DEM の高さだけ。ほかのモードまで録ると場所を食う。"""
        spy = _SpyRecorder()
        Renderer(recorder=spy).render(_sensor_frame(), self._settings(ViewMode.DEPTH))

        assert spy.calls == []

    def test_録画なしでも描ける(self):
        """普段の起動は recorder が None。ここが落ちると実演が止まる。"""
        image = Renderer().render(_sensor_frame(), self._settings(ViewMode.DEM))
        assert image.shape[:2] == (config.VIEW_SIZE[1], config.VIEW_SIZE[0])

    def test_閉じると録画も閉じる(self):
        spy = _SpyRecorder()
        Renderer(recorder=spy).close()
        assert spy.calls == ["closed"]


def _full_view_area():
    """画面全体を砂場とみなす四隅。"""
    width, height = config.VIEW_SIZE
    return [[0, 0], [0, height], [width, height], [width, 0]]


class TestCropToArea:
    """砂場の四隅の内側だけを切り出すこと。"""

    def _marked_left(self):
        """センサ画像の左の 4 分の 1 だけが高い高さ[mm]。"""
        width, height = config.PROC_SIZE
        source = np.zeros((height, width), dtype=np.float32)
        source[:, : width // 4] = 100.0
        return source

    def test_画面全体なら左右が入れ替わる(self):
        """エリアは左右反転したあとの表示像で指定するため、反転が入る。

        符号を「揃える」と砂場の反対側を切り出すことになる。
        """
        cropped = crop_to_area(self._marked_left(), _full_view_area())
        width = cropped.shape[1]

        # センサの左にあった山が、切り出した絵では右へ移っている。
        assert float(cropped[:, -width // 8 :].min()) == pytest.approx(100.0)
        assert float(cropped[:, : width // 8].max()) == pytest.approx(0.0)

    def test_画面の左半分は砂場の右半分になる(self):
        """表示像の左半分は、センサ画像では右半分にあたる。"""
        width, height = config.PROC_SIZE
        source = np.zeros((height, width), dtype=np.float32)
        source[:, width // 2 :] = 100.0  # センサの右半分

        view_width, view_height = config.VIEW_SIZE
        left_half = [[0, 0], [0, view_height], [view_width // 2, view_height], [view_width // 2, 0]]

        cropped = crop_to_area(source, left_half)
        np.testing.assert_allclose(cropped, 100.0)

    def test_端が0mmで埋まらない(self):
        """既定の縁取り（0 で埋める）だと、砂場の外周だけ基準面へ落ちた段差になる。"""
        width, height = config.PROC_SIZE
        source = np.full((height, width), 55.0, dtype=np.float32)

        cropped = crop_to_area(source, _full_view_area())
        np.testing.assert_allclose(cropped, 55.0)

    def test_出力の大きさは処理解像度(self):
        cropped = crop_to_area(self._marked_left(), _VIEW_AREA)
        assert cropped.shape == (config.PROC_SIZE[1], config.PROC_SIZE[0])
        assert cropped.dtype == np.float32

    def test_四隅が潰れていたら切り取らない(self):
        """実演中に落とさないため、例外ではなく None を返す。

        cv2 は 1 点に集まった四隅でも例外を出さず、真っ黒な結果を返す。
        そのまま録ると、高さ 0mm の平面が延々と残る。
        """
        assert crop_to_area(self._marked_left(), [[10, 10]] * 4) is None
        assert crop_to_area(self._marked_left(), [[0, 0], [0, 300], [0, 600], [0, 900]]) is None


class TestRecordCrop:
    """--record-crop の振る舞い。"""

    def _settings(self, area=None):
        settings = RenderSettings()
        settings.view_mode = ViewMode.DEM
        settings.reference_plane = _plane()
        if area:
            settings.area_positions = [list(point) for point in area]
        return settings

    def test_エリアがあれば切り取って書き出す(self):
        spy = _SpyRecorder()
        renderer = Renderer(recorder=spy, record_crop=True)
        renderer.render(_sensor_frame(), self._settings(_VIEW_AREA))

        assert len(spy.calls) == 1
        height_mm, _, context = spy.calls[0]
        assert height_mm.shape == (config.PROC_SIZE[1], config.PROC_SIZE[0])
        assert context["crop_area_positions"] == _VIEW_AREA

    def test_エリアが無ければ書き出さずに知らせる(self):
        """視野全体のフレームを混ぜて書くと、あとから見分けられなくなる。"""
        spy = _SpyRecorder()
        renderer = Renderer(recorder=spy, record_crop=True)
        renderer.render(_sensor_frame(), self._settings())

        assert spy.calls == []
        assert spy.warnings and "四隅" in spy.warnings[0]

    def test_同じ理由を繰り返し知らせない(self, tmp_path):
        """毎フレーム同じ理由で弾かれる。そのたび積むと画面が埋まる。"""
        recorder = DemRecorder(directory=tmp_path, max_frames=0)
        recorder.open()
        try:
            recorder.warn("エリアが未指定です")
            recorder.warn("エリアが未指定です")
            assert recorder.take_notice() == "エリアが未指定です"
            assert recorder.take_notice() is None
        finally:
            recorder.close()

    def test_一度書けたら同じ理由をまた知らせる(self, tmp_path):
        """途中でエリアを消したときに、黙って止まらないようにするため。"""
        recorder = DemRecorder(directory=tmp_path, max_frames=0)
        recorder.open()
        try:
            recorder.warn("エリアが未指定です")
            recorder.take_notice()
            recorder.write(_height(1.0), 0.0)
            recorder.warn("エリアが未指定です")
            assert recorder.take_notice() == "エリアが未指定です"
        finally:
            recorder.close()

    def test_切り取らなければ視野全体のまま(self):
        """既定は従来どおり。付けていないのに形が変わると驚く。"""
        spy = _SpyRecorder()
        Renderer(recorder=spy).render(_sensor_frame(), self._settings(_VIEW_AREA))

        assert spy.calls[0][2]["crop_area_positions"] is None
