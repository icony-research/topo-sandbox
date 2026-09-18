"""Kinect の生データからミリメートルへの変換の検証。"""

import numpy as np
import pytest

from topo_sandbox.sensor.kinect import PLAYER_INDEX_BITS, to_millimeters

#: DepthRange.Default の有効範囲[mm]
MIN_DEPTH = 800
MAX_DEPTH = 4000

#: 深度は 13bit
ALL_DEPTHS = np.arange(0, 1 << 13, dtype=np.uint16)


def _packed(depth_values, player_index=0):
    """深度[mm]を Kinect が返す packed な形式へ組み立てる。"""
    return np.left_shift(depth_values, PLAYER_INDEX_BITS) | player_index


def test_有効範囲の深度はミリメートルのまま通る():
    result = to_millimeters(_packed(ALL_DEPTHS), MIN_DEPTH, MAX_DEPTH)
    valid = (ALL_DEPTHS >= MIN_DEPTH) & (ALL_DEPTHS <= MAX_DEPTH)
    np.testing.assert_array_equal(result[valid], ALL_DEPTHS[valid])


def test_有効範囲外は欠測になる():
    outside = np.array([0, MIN_DEPTH - 1, MAX_DEPTH + 1, 8191], dtype=np.uint16)
    result = to_millimeters(_packed(outside), MIN_DEPTH, MAX_DEPTH)
    assert np.all(result == 0)


@pytest.mark.parametrize("player_index", [0, 1, 3, 7])
def test_プレイヤーインデックスビットは結果に影響しない(player_index):
    actual = to_millimeters(_packed(ALL_DEPTHS, player_index), MIN_DEPTH, MAX_DEPTH)
    expected = to_millimeters(_packed(ALL_DEPTHS), MIN_DEPTH, MAX_DEPTH)
    np.testing.assert_array_equal(actual, expected)


def test_異なる深度は異なる値になる():
    """深度を狭いビット幅へ丸めると衝突する。それが起きていないことの確認。"""
    depths = np.array([1000, 1256, 1512], dtype=np.uint16)
    result = to_millimeters(_packed(depths), MIN_DEPTH, MAX_DEPTH)
    assert len(set(result.tolist())) == len(depths)
    np.testing.assert_array_equal(result, depths)


def test_深度は単調に増加する():
    depths = np.arange(MIN_DEPTH, MAX_DEPTH + 1, dtype=np.uint16)
    result = to_millimeters(_packed(depths), MIN_DEPTH, MAX_DEPTH).astype(np.int32)
    assert np.all(np.diff(result) > 0)


def test_Near_Modeの範囲でも同様に扱える():
    near_min, near_max = 400, 3000
    result = to_millimeters(_packed(ALL_DEPTHS), near_min, near_max)
    valid = (near_min <= ALL_DEPTHS) & (near_max >= ALL_DEPTHS)
    np.testing.assert_array_equal(result[valid], ALL_DEPTHS[valid])
    assert np.all(result[~valid] == 0)
