"""現場で合わせた値の保存と読み込みの検証。GPU も Kinect も不要。"""

import json

import pytest

from topo_sandbox import settings_store
from topo_sandbox.processing.plane import ReferencePlane
from topo_sandbox.renderer import MappingMode, RenderSettings, ViewMode


def _adjusted():
    """設営で合わせたあとの設定。"""
    settings = RenderSettings()
    settings.z_scale = 0.17
    settings.color_sensitivity = 2.6
    settings.water_level_mm = -25.0
    settings.show_contour = True
    settings.show_rivers = True
    settings.river_preset = 0
    settings.show_flood = True
    settings.heavy_rain = True
    settings.spring_position = [0.25, 0.75]
    settings.view_mode = ViewMode.DEM
    settings.mapping_mode = MappingMode.PERSPECTIVE
    settings.area_positions = [[10, 20], [12, 580], [790, 575], [788, 18]]
    settings.projector_positions = [[5, 6], [7, 594], [795, 590], [793, 4]]
    settings.reference_plane = ReferencePlane(
        a=12.0,
        b=-8.0,
        c=1042.0,
        residual_mm=1.8,
        tilt_deg=3.2,
        distance_mm=1042.0,
        coverage=87.0,
    )
    return settings


class TestRoundTrip:
    def test_合わせた値が次の起動で戻る(self, tmp_path):
        path = tmp_path / "settings.json"
        settings_store.save(_adjusted(), path)

        restored = RenderSettings()
        settings_store.load(restored, path)

        source = _adjusted()
        assert restored.z_scale == source.z_scale
        assert restored.color_sensitivity == source.color_sensitivity
        assert restored.water_level_mm == source.water_level_mm
        assert restored.show_contour and restored.show_rivers
        assert restored.river_preset == source.river_preset
        assert restored.view_mode is ViewMode.DEM
        assert restored.mapping_mode is MappingMode.PERSPECTIVE
        assert restored.area_positions == source.area_positions
        assert restored.projector_positions == source.projector_positions
        assert restored.show_flood
        assert restored.spring_position == source.spring_position

    def test_大雨は持ち越さない(self, tmp_path):
        """次の起動でいきなり大雨から始まると驚く。"""
        path = tmp_path / "settings.json"
        settings_store.save(_adjusted(), path)

        restored = RenderSettings()
        settings_store.load(restored, path)

        assert not restored.heavy_rain

    def test_水源が無ければ書かない(self, tmp_path):
        path = tmp_path / "settings.json"
        settings_store.save(RenderSettings(), path)

        assert "spring_position" not in json.loads(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize(
        "broken", [[0.5], [0.5, 1.5], ["a", 0.5], [True, 0.5], {"u": 0.5}, None]
    )
    def test_壊れた水源は捨てる(self, tmp_path, broken):
        path = tmp_path / "settings.json"
        settings_store.save(_adjusted(), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["spring_position"] = broken
        path.write_text(json.dumps(data), encoding="utf-8")

        restored = RenderSettings()
        settings_store.load(restored, path)

        assert restored.spring_position is None
        assert restored.z_scale == _adjusted().z_scale  # 残りは読める

    def test_水源の無い古いファイルも読める(self, tmp_path):
        """版を上げずに項目を足したので、前の設営で保存したファイルがそのまま使える。"""
        path = tmp_path / "settings.json"
        settings_store.save(_adjusted(), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["spring_position"]
        del data["show_flood"]
        path.write_text(json.dumps(data), encoding="utf-8")

        restored = RenderSettings()
        settings_store.load(restored, path)

        assert restored.spring_position is None
        assert restored.area_positions == _adjusted().area_positions

    def test_基準面も戻る(self, tmp_path):
        """センサを動かしていなければ、取り直さずに済む。"""
        path = tmp_path / "settings.json"
        settings_store.save(_adjusted(), path)

        restored = RenderSettings()
        settings_store.load(restored, path)

        assert restored.reference_plane == _adjusted().reference_plane

    def test_基準面が無ければ書かない(self, tmp_path):
        path = tmp_path / "settings.json"
        settings_store.save(RenderSettings(), path)

        assert "reference_plane" not in json.loads(path.read_text(encoding="utf-8"))

    def test_表示モードは名前で持つ(self, tmp_path):
        """数値だと、並びを変えたときに黙って別のモードになる。"""
        path = tmp_path / "settings.json"
        settings = RenderSettings()
        settings.view_mode = ViewMode.EDGE
        settings_store.save(settings, path)

        assert json.loads(path.read_text(encoding="utf-8"))["view_mode"] == "EDGE"

    def test_人が読める形で書く(self, tmp_path):
        """当日に中身を確かめたり手で直したりできること。"""
        path = tmp_path / "settings.json"
        settings_store.save(_adjusted(), path)
        text = path.read_text(encoding="utf-8")

        assert "\n" in text.strip()  # 1 行に詰め込まない
        assert '"water_level_mm"' in text
        assert "[10, 20]" in text  # 四隅は 1 点 1 行にまとめる


class TestBrokenFile:
    """設定が読めないくらいで実演を止めないこと。"""

    def test_ファイルが無ければ何もしない(self, tmp_path):
        settings = RenderSettings()
        assert settings_store.load(settings, tmp_path / "無い.json") == 0
        assert settings.z_scale == RenderSettings().z_scale

    def test_壊れたファイルは例外にする(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{ こわれている", encoding="utf-8")

        with pytest.raises(ValueError):
            settings_store.load(RenderSettings(), path)

    def test_版が違えば読まない(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"version": 99, "z_scale": 9.9}), encoding="utf-8")

        with pytest.raises(ValueError):
            settings_store.load(RenderSettings(), path)

    def test_一部が壊れていても残りは反映する(self, tmp_path):
        """手で直したときの書き損じで、全部が飛ばないこと。"""
        data = settings_store.to_dict(_adjusted())
        data["area_positions"] = [[1, 2], [3, 4]]  # 4 点そろっていない
        data["view_mode"] = "存在しないモード"

        path = tmp_path / "settings.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        restored = RenderSettings()
        settings_store.load(restored, path)

        assert restored.water_level_mm == -25.0  # 壊れていない項目は入る
        assert restored.area_positions == []  # 壊れた項目は捨てる
        assert restored.view_mode is RenderSettings().view_mode

    def test_知らない項目は無視する(self, tmp_path):
        data = settings_store.to_dict(RenderSettings())
        data["将来の項目"] = 1

        path = tmp_path / "settings.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        assert settings_store.load(RenderSettings(), path) > 0
