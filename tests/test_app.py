"""キー操作のうち、Tk を起動せずに確かめられる範囲の検証。

`SandboxApp.__init__` は値を持つだけでウィジェットを作らないため、
水位や表示の切り替えは画面なしで確かめられる。描画そのものは実機での確認が必要。
"""

from topo_sandbox import config
from topo_sandbox.app import SandboxApp
from topo_sandbox.renderer import ViewMode


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
