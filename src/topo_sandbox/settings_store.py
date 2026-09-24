"""現場で合わせた値をファイルへ保存し、次の起動で読み戻す。

設営のたびに投影エリア・投影枠・Z スケール・カラー感度・水位を合わせ直すのは
手間で、当日の限られた時間では二度目のやり直しが効かない。一度合わせた値を
保存しておけば、次は読み込むだけで同じ状態から始められる。

保存先は JSON。現場で中身を目で確かめたり、手で直したりできるようにするため。
壊れたファイルや、版の違うファイルを読んだときは**既定値のまま起動する**。
設定が読めないくらいで実演を止めないこと。
"""

import json
import re

from . import config
from .processing.plane import ReferencePlane
from .renderer import MappingMode, ViewMode

#: 保存形式の版。読めない版は無視して既定値で起動する。
FORMAT_VERSION = 1

#: そのまま読み書きする値
_PLAIN_FIELDS = (
    "z_scale",
    "color_sensitivity",
    "water_level_mm",
    "show_contour",
    "show_rivers",
    "river_preset",
)

#: 基準面から保存する項目
_PLANE_FIELDS = (
    "a",
    "b",
    "c",
    "residual_mm",
    "tilt_deg",
    "distance_mm",
    "coverage",
)


def to_dict(settings):
    """:class:`~topo_sandbox.renderer.RenderSettings` を保存用の辞書にする。"""
    data = {"version": FORMAT_VERSION}
    data.update({name: getattr(settings, name) for name in _PLAIN_FIELDS})

    # 表示モードは名前で持つ。数値だと、並びを変えたときに黙って別のモードになる。
    data["view_mode"] = settings.view_mode.name
    data["mapping_mode"] = settings.mapping_mode.name

    data["area_positions"] = [list(map(int, position)) for position in settings.area_positions]
    data["projector_positions"] = [
        list(map(int, position)) for position in settings.projector_positions
    ]

    plane = settings.reference_plane
    if plane is not None:
        data["reference_plane"] = {name: float(getattr(plane, name)) for name in _PLANE_FIELDS}

    return data


def apply(data, settings):
    """:func:`to_dict` の辞書を設定へ反映する。

    知らない項目は無視し、壊れている項目はその項目だけ捨てる。設定ファイルを
    手で直したときに、1 か所の書き損じで全部が飛ばないようにするため。

    Args:
        data: 読み込んだ辞書。
        settings: 反映先の :class:`~topo_sandbox.renderer.RenderSettings`。

    Returns:
        反映できた項目の数。
    """
    if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
        raise ValueError(f"設定ファイルの版が違います（対応: {FORMAT_VERSION}）")

    applied = 0

    for name in _PLAIN_FIELDS:
        if name in data:
            setattr(settings, name, data[name])
            applied += 1

    for name, enum_type in (("view_mode", ViewMode), ("mapping_mode", MappingMode)):
        member = enum_type.__members__.get(data.get(name))
        if member is not None:
            setattr(settings, name, member)
            applied += 1

    for name in ("area_positions", "projector_positions"):
        positions = data.get(name)
        if _is_quad(positions):
            setattr(settings, name, [[int(x), int(y)] for x, y in positions])
            applied += 1

    plane = data.get("reference_plane")
    if isinstance(plane, dict) and all(name in plane for name in _PLANE_FIELDS):
        settings.reference_plane = ReferencePlane(
            **{name: float(plane[name]) for name in _PLANE_FIELDS}
        )
        applied += 1

    return applied


def _is_quad(positions):
    """四隅として使える並びか。"""
    return (
        isinstance(positions, list)
        and len(positions) == 4
        and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in positions)
    )


def save(settings, path=None):
    """設定をファイルへ書き出す。

    Args:
        settings: :class:`~topo_sandbox.renderer.RenderSettings`。
        path: 保存先。省略すると :data:`config.SETTINGS_PATH`。

    Returns:
        書き出した先の :class:`~pathlib.Path`。
    """
    path = path or config.SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(to_dict(settings), ensure_ascii=False, indent=2)

    # 四隅の座標は 1 点 1 行にまとめる。json のままだと 1 点が 4 行へ広がり、
    # 当日に目で追えない。中身は変えず、見た目だけを詰める。
    text = re.sub(r"\[\s+(-?\d+),\s+(-?\d+)\s+\]", r"[\1, \2]", text)

    path.write_text(text + "\n", encoding="utf-8")
    return path


def load(settings, path=None):
    """ファイルから設定を読み込んで反映する。

    Args:
        settings: 反映先の :class:`~topo_sandbox.renderer.RenderSettings`。
        path: 読み込み元。省略すると :data:`config.SETTINGS_PATH`。

    Returns:
        反映できた項目の数。ファイルが無ければ 0。

    Raises:
        ValueError: ファイルが壊れている、または版が違うとき。
    """
    path = path or config.SETTINGS_PATH
    if not path.exists():
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"設定ファイルを読めません: {error}") from error

    return apply(data, settings)
