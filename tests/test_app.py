"""キー操作のうち、Tk を起動せずに確かめられる範囲の検証。

`SandboxApp.__init__` は値を持つだけでウィジェットを作らないため、
水位や表示の切り替えは画面なしで確かめられる。描画そのものは実機での確認が必要。
"""

import numpy as np
import pytest

from topo_sandbox import config
from topo_sandbox.app import SandboxApp
from topo_sandbox.renderer import MappingMode, ViewMode


def _app():
    return SandboxApp(source=None)


class TestWaterLevelKeys:
    def test_上げ下げできる(self):
        app = _app()
        start = app.settings.water_level_mm

        app._adjust_water_level(+config.WATER_LEVEL_DELTA_MM)
        assert app.settings.water_level_mm == start + config.WATER_LEVEL_DELTA_MM

        app._adjust_water_level(-config.WATER_LEVEL_DELTA_MM)
        assert app.settings.water_level_mm == start

    def test_上限と下限で止まる(self):
        """際限なく動かすと水面が画面から消え、戻し方が分からなくなる。"""
        app = _app()

        for _ in range(200):
            app._adjust_water_level(+config.WATER_LEVEL_DELTA_MM)
        assert app.settings.water_level_mm == config.WATER_LEVEL_MAX_MM

        for _ in range(200):
            app._adjust_water_level(-config.WATER_LEVEL_DELTA_MM)
        assert app.settings.water_level_mm == config.WATER_LEVEL_MIN_MM

    def test_初期値へ戻せる(self):
        app = _app()
        app._adjust_water_level(+config.WATER_LEVEL_DELTA_MM * 5)
        app._reset_water_level()

        assert app.settings.water_level_mm == config.WATER_LEVEL_MM

    def test_DEM以外では見えないと案内する(self):
        """押しても何も起きないように見えるため。"""
        app = _app()
        app.settings.view_mode = ViewMode.COLORING
        app._adjust_water_level(+config.WATER_LEVEL_DELTA_MM)
        assert "DEM" in app._message

        app.settings.view_mode = ViewMode.DEM
        app._adjust_water_level(+config.WATER_LEVEL_DELTA_MM)
        assert "DEM" not in app._message


class TestRiverKey:
    def test_切り替えられる(self):
        app = _app()
        assert app.settings.show_rivers is False

        app._toggle_rivers()
        assert app.settings.show_rivers is True

        app._toggle_rivers()
        assert app.settings.show_rivers is False

    def test_DEM以外では見えないと案内する(self):
        app = _app()
        app.settings.view_mode = ViewMode.COLORING
        app._toggle_rivers()
        assert "DEM" in app._message

        app.settings.view_mode = ViewMode.DEM
        app._toggle_rivers()
        assert "DEM" not in app._message


class _FakeKey:
    """Tk のキーイベントの代わり。keysym と修飾キーの状態だけ持つ。"""

    def __init__(self, keysym, state=0):
        self.keysym = keysym
        self.state = state


class TestKeyDispatch:
    def test_単独のキーは効く(self):
        app = _app()
        before = app.settings.color_sensitivity
        app._on_key(_FakeKey("s"))
        assert app.settings.color_sensitivity != before

    def test_Ctrl併用のキーは無視する(self):
        """Ctrl + S（保存）が、単独の s（カラー感度）として二重に効かないこと。"""
        app = _app()
        before = app.settings.color_sensitivity
        app._on_key(_FakeKey("s", state=SandboxApp._CONTROL_MASK))
        assert app.settings.color_sensitivity == before


class TestSettingsKeys:
    """Ctrl + S で保存し、次の起動で読み戻せること。"""

    def test_保存して読み戻せる(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

        app = _app()
        app.settings.water_level_mm = -15.0
        app.settings.area_positions = [[1, 2], [3, 4], [5, 6], [7, 8]]
        app._save_settings()
        assert "保存しました" in app._message

        restored = _app()
        restored._load_settings()
        assert restored.settings.water_level_mm == -15.0
        assert restored.settings.area_positions == [[1, 2], [3, 4], [5, 6], [7, 8]]

    def test_保存が無ければ既定値のまま(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "無い.json")

        app = _app()
        app._load_settings()
        assert app.settings.water_level_mm == config.WATER_LEVEL_MM

    def test_壊れていても既定値で起動する(self, tmp_path, monkeypatch):
        """設定ファイルの不備で実演が始められない、という事態を避ける。"""
        path = tmp_path / "settings.json"
        path.write_text("こわれている", encoding="utf-8")
        monkeypatch.setattr(config, "SETTINGS_PATH", path)

        app = _app()
        app._load_settings()

        assert app.settings.water_level_mm == config.WATER_LEVEL_MM
        assert "既定値で起動" in app._message


class TestResetKeys:
    def test_調整値を戻しても設営は残る(self):
        app = _app()
        app.settings.area_positions = [[1, 2], [3, 4], [5, 6], [7, 8]]
        app.settings.water_level_mm = 40.0
        app.settings.spring_position = [0.2, 0.3]

        app._reset_adjustments()

        assert app.settings.water_level_mm == config.WATER_LEVEL_MM
        assert app.settings.area_positions == [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert app.settings.spring_position == [0.2, 0.3]
        assert "エリア・投影枠・基準面・水源はそのまま" in app._message

    def test_すべて戻すと設営も消える(self):
        app = _app()
        app.settings.area_positions = [[1, 2], [3, 4], [5, 6], [7, 8]]
        app.settings.spring_position = [0.2, 0.3]
        app._projector_edit = True
        app._spring_edit = True

        app._reset_all()

        assert app.settings.area_positions == []
        assert app.settings.spring_position is None
        assert not app._spring_edit
        assert app.settings.reference_plane is None
        assert not app._projector_edit  # 投影枠の編集モードからも抜ける
        assert "すべて初期値へ戻しました" in app._message

    def test_戻しても保存は消さない(self, tmp_path, monkeypatch):
        """書き戻すかどうかは Ctrl + S で選ぶ。"""
        monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

        app = _app()
        app.settings.water_level_mm = 40.0
        app._save_settings()

        app._reset_all()
        restored = _app()
        restored._load_settings()

        assert restored.settings.water_level_mm == 40.0


class TestRiverPresetKey:
    def test_出やすさを切り替えられる(self):
        app = _app()
        before = app.settings.river_setting

        app._cycle_river_preset()
        assert app.settings.river_setting != before
        assert app.settings.river_setting[0] in app._message

    def test_一周すると元に戻る(self):
        """実演中に押しすぎても、押し続ければ戻ってこられること。"""
        app = _app()
        before = app.settings.river_preset

        for _ in range(len(config.RIVER_PRESETS)):
            app._cycle_river_preset()

        assert app.settings.river_preset == before

    def test_範囲外の添字でも落ちない(self):
        """設定ファイルに古い添字が残っていても実演を止めない。"""
        app = _app()
        app.settings.river_preset = 999
        assert app.settings.river_setting in config.RIVER_PRESETS


class TestPlaneCapture:
    """基準面を砂場の中だけであてはめているかの検証。

    センサ画像の右半分を砂場、左半分を遠い床にしてある。表示像は左右反転
    されるので、砂場は画面の**左半分**に写る。エリアを画面左半分に指定して
    砂場の距離が出れば、反転を正しく戻せている。
    """

    def _frame(self, sand_mm=1000.0, floor_mm=1500.0):
        width, height = config.SENSOR_SIZE
        frame = np.full((height, width), floor_mm, dtype=np.float32)
        frame[:, 300:] = sand_mm  # センサから見て右半分が砂場
        return frame

    def _left_half_of_view(self):
        width, height = config.VIEW_SIZE
        half = width // 2
        return [[0, 0], [0, height], [half, height], [half, 0]]

    def test_エリアの中だけで基準面を求める(self):
        app = _app()
        app.settings.area_positions = self._left_half_of_view()

        app._finish_plane_capture([self._frame()])

        assert app.settings.reference_plane.distance_mm == pytest.approx(1000.0, abs=1.0)

    def test_エリア未指定なら視野全体で求めて注意する(self):
        app = _app()

        app._finish_plane_capture([self._frame()])

        # 砂場と床を混ぜた面になるため、どちらの距離とも一致しない。
        assert app.settings.reference_plane.distance_mm > 1000.0
        assert "エリア未指定" in app._message

    def test_取得できないときは落とさずに知らせる(self):
        """実演中に例外で止めない。"""
        app = _app()
        app.settings.area_positions = self._left_half_of_view()

        app._finish_plane_capture([np.zeros(config.SENSOR_SIZE[::-1], dtype=np.float32)])

        assert app.settings.reference_plane is None
        assert "基準面を取得できませんでした" in app._message


class _FakeClick:
    """Tk のクリックイベントの代わり。"""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeRenderer:
    """水を抜くよう頼まれたかだけを覚える。"""

    recorder = None

    def __init__(self):
        self.drained = 0

    def drain_flood(self):
        self.drained += 1


def _flood_ready_app():
    """DEM 表示・基準面あり・水源ありの、水がすぐ見える状態。"""
    app = SandboxApp(source=None, renderer=_FakeRenderer())
    app.settings.view_mode = ViewMode.DEM
    app.settings.reference_plane = object()
    app.settings.spring_position = [0.5, 0.5]
    return app


class TestFloodKeys:
    def test_wで水を流すかを切り替える(self):
        app = _flood_ready_app()

        app._on_key(_FakeKey("w"))
        assert app.settings.show_flood
        assert app._message == "水を流す: 開始"

        app._on_key(_FakeKey("w"))
        assert not app.settings.show_flood

    def test_Shift_Wで水を抜く(self):
        app = _flood_ready_app()
        app._on_key(_FakeKey("W"))
        assert app.renderer.drained == 1

    def test_oで大雨を切り替える(self):
        app = _flood_ready_app()
        app.settings.show_flood = True

        app._on_key(_FakeKey("o"))
        assert app.settings.heavy_rain
        assert "大雨" in app._message

        app._on_key(_FakeKey("o"))
        assert not app.settings.heavy_rain

    @pytest.mark.parametrize(
        "prepare, hint",
        [
            (lambda s: setattr(s, "view_mode", ViewMode.COLORING), "v で切り替え"),
            (lambda s: setattr(s, "reference_plane", None), "k を押して"),
            (lambda s: setattr(s, "spring_position", None), "i を押して"),
        ],
    )
    def test_水が見えない理由を案内する(self, prepare, hint):
        """条件がそろっていないと何も起きず、壊れているように見える。"""
        app = _flood_ready_app()
        prepare(app.settings)

        app._on_key(_FakeKey("w"))

        assert hint in app._message


class TestSpringPlacement:
    def test_iのあとのクリックで水源を置く(self):
        app = _app()
        width, height = config.VIEW_SIZE

        app._on_key(_FakeKey("i"))
        app._on_click(_FakeClick(width // 4, height // 2))

        assert app.settings.spring_position == pytest.approx([0.75, 0.5])
        assert app.settings.area_positions == []  # エリアの点にはならない
        assert not app._spring_edit  # 1 回置いたら抜ける

    def test_ふだんのクリックはエリアの点になる(self):
        app = _app()
        app._on_click(_FakeClick(10, 20))

        assert app.settings.area_positions == [[10, 20]]
        assert app.settings.spring_position is None

    def test_砂場の外ならモードを抜けずに待つ(self):
        app = _app()
        app.settings.mapping_mode = MappingMode.PERSPECTIVE
        app.settings.area_positions = [[200, 150], [200, 450], [600, 450], [600, 150]]
        app.settings.projector_positions = [[300, 200], [300, 400], [500, 400], [500, 200]]

        app._on_key(_FakeKey("i"))
        app._on_click(_FakeClick(0, 0))

        assert app.settings.spring_position is None
        assert app._spring_edit
        assert "砂場の外" in app._message

    def test_もう一度iで取り消せる(self):
        app = _app()
        app._on_key(_FakeKey("i"))
        app._on_key(_FakeKey("i"))

        app._on_click(_FakeClick(10, 20))

        assert app.settings.spring_position is None
        assert app.settings.area_positions == [[10, 20]]
