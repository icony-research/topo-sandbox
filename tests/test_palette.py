"""配色テーブルの検証。

見た目を決める値なので、長さと端の色を固定しておく。
"""

import numpy as np

from topo_sandbox import config
from topo_sandbox.palette import SLOPE_COLORS


def test_感度の上限まで添字が届く():
    """テーブルは傾斜 90 度 × 感度上限 を引ける長さが必要。

    ここが足りないと、垂直な砂の壁が現れた瞬間に添字エラーで落ちる。
    """
    max_index = int(90 * config.COLOR_SENSITIVITY_MAX)
    assert len(SLOPE_COLORS) > max_index
    SLOPE_COLORS[max_index]  # 例外にならないこと


def test_テーブルの形と型():
    assert SLOPE_COLORS.shape == (361, 3)
    assert SLOPE_COLORS.dtype == np.uint8


def test_平坦は赤():
    """傾斜 0 度は赤。教材としての約束事。"""
    np.testing.assert_array_equal(SLOPE_COLORS[0], [255, 0, 0])


def test_色相環を一周している():
    """先頭と末尾が同じ色であること（色相 0 度と 360 度）。"""
    np.testing.assert_array_equal(SLOPE_COLORS[0], SLOPE_COLORS[-1])


def test_隣り合う色は滑らかに変化する():
    """階調が飛んでいると投影像に縞が出る。"""
    difference = np.abs(np.diff(SLOPE_COLORS.astype(np.int16), axis=0))
    assert difference.max() <= 5
