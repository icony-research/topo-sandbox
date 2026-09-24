"""水の流れの計算。降った雨がどこへ流れ、どこに集まるかを求める。

地形学でいう D8 法。各画素から最も急に下る隣へ水を流し、集まった量を数える。
量の多いところが川になり、川と川のあいだが分水嶺になる。

numpy だけで完結するので GPU も Kinect も要らない。ただし流量の計算は
繰り返しが必要で、他の処理より重い。処理解像度をさらに落としてから行う。
"""

import cv2
import numpy as np

from .. import config

#: 8 近傍の (行のずれ, 列のずれ)
_NEIGHBOURS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def flow_directions(height_mm):
    """各画素の水が流れ込む先を返す。

    8 近傍のうち、最も急に下るものを選ぶ。下る先が無い画素（窪地や平坦地）は
    自分自身を指す。水はそこで溜まる、という扱い。

    Args:
        height_mm: 高さ[mm] (H, W)。値が大きいほど高い。

    Returns:
        流れ込む先の画素番号 (H*W,) int32。1 次元に並べたときの添字。
    """
    height = np.asarray(height_mm, dtype=np.float32)
    rows, columns = height.shape

    row_index = np.arange(rows)[:, None]
    column_index = np.arange(columns)[None, :]

    target = row_index * columns + column_index
    target = np.broadcast_to(target, height.shape).astype(np.int32).copy()

    # 下り勾配が 0 より大きいものだけを選ぶので、初期値は 0 でよい。
    steepest = np.zeros_like(height)

    for row_step, column_step in _NEIGHBOURS:
        # 端からはみ出す先は端の画素自身にする。高さの差が 0 になるため
        # 選ばれることがなく、砂場の外へ水が出ていかない。
        neighbour_row = np.clip(row_index + row_step, 0, rows - 1)
        neighbour_column = np.clip(column_index + column_step, 0, columns - 1)

        # 斜めの隣は遠いぶん勾配が緩い。距離で割らないと斜めばかり選ばれる。
        distance = float(np.hypot(row_step, column_step))
        drop = (height - height[neighbour_row, neighbour_column]) / distance

        better = drop > steepest
        steepest = np.where(better, drop, steepest)
        target = np.where(better, neighbour_row * columns + neighbour_column, target)

    return target.ravel().astype(np.int32)


def flow_accumulation(directions, cell_count, iterations):
    """各画素へ集まる水の量を数える。

    ``量[i] = 1 + (i へ流れ込む画素の量の合計)`` を繰り返す。水は必ず下るので
    流れに循環が無く、繰り返せば必ず収束する。ただし川が長いほど回数が要るため、
    実演中に止まらないよう上限を設けている。打ち切った場合は上流の寄与が
    届ききらず、本流と支流の差が小さめに出る（川が消えるわけではない）。

    Args:
        directions: :func:`flow_directions` の結果 (N,)。
        cell_count: 画素数 N。
        iterations: 繰り返しの上限。

    Returns:
        集まった水の量 (N,) float32。自分の 1 を含むので最小は 1。
    """
    index = np.arange(cell_count, dtype=np.int32)

    # 自分自身を指す画素（窪地）を混ぜると、自分の量を自分に足し続けて発散する。
    sources = np.nonzero(directions != index)[0]
    targets = directions[sources]

    accumulation = np.ones(cell_count, dtype=np.float32)

    for _ in range(iterations):
        incoming = np.bincount(targets, weights=accumulation[sources], minlength=cell_count)
        updated = (1.0 + incoming).astype(np.float32)

        if np.array_equal(updated, accumulation):
            break
        accumulation = updated

    return accumulation


def river_strength(height_mm, min_cells=None, full_cells=None):
    """川らしさを 0〜1 で返す。1 に近いほど水が集まっている。

    流量の計算は繰り返しが重いので、いったん :data:`config.RIVER_SIZE` まで
    落としてから行い、結果を元の大きさへ戻す。川は太めの線として見えれば
    十分なので、粗くしても見た目はほとんど変わらない。

    しきい値は「何画素ぶんの水が集まったら川とみなすか」で決め打ちにしてある。
    フレームごとに最大値で正規化すると、砂をいじるたびに川の濃さが変わって
    落ち着かない。

    Args:
        height_mm: 高さ[mm] (H, W)。
        min_cells: 川として描き始める流量[画素]。省略すると起動時の設定。
        full_cells: 色が最も濃くなる流量[画素]。省略すると起動時の設定。

    Returns:
        入力と同じ大きさの 0〜1 (H, W) float32。
    """
    _, initial_min, initial_full = config.RIVER_PRESETS[config.RIVER_PRESET_INITIAL]
    min_cells = initial_min if min_cells is None else min_cells
    full_cells = initial_full if full_cells is None else full_cells

    height = np.asarray(height_mm, dtype=np.float32)
    coarse = cv2.resize(height, dsize=config.RIVER_SIZE, interpolation=cv2.INTER_AREA)

    directions = flow_directions(coarse)
    accumulation = flow_accumulation(directions, coarse.size, config.RIVER_ITERATIONS)

    # 流量は上流から下流へ一気に増えるので、対数で見ないと本流だけが真っ青になる。
    lower = np.log(min_cells)
    upper = np.log(full_cells)
    strength = (np.log(accumulation) - lower) / (upper - lower)
    strength = np.clip(strength, 0.0, 1.0).reshape(coarse.shape)

    return cv2.resize(strength, dsize=(height.shape[1], height.shape[0]))
