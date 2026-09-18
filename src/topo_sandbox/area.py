"""投影エリア（砂場の四隅）の保存。

`w` キーで現在の四隅を :data:`topo_sandbox.config.AREA_FILE` へ書き出す。

**読み込みは未実装**であり、起動のたびに四隅を指定し直す必要がある。
イベント設営の手間に直結するため、改良作業の候補として README の
「既知の制限」に挙げてある。
"""

from . import config


def save_area(positions, path=None):
    """四隅の座標をファイルへ書き出す。

    1 行 1 点で ``[178, 115]`` のように書く。既存の area.txt と同じ形式。

    Args:
        positions: ``[[x, y], ...]`` の座標リスト。
        path: 書き出し先。省略時は :data:`topo_sandbox.config.AREA_FILE`。
    """
    path = path or config.AREA_FILE
    with open(path, "w", encoding="utf-8") as handle:
        for position in positions:
            handle.write(f"{position}\n")
    return path
