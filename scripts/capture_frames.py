"""再生モード用の深度画像を Kinect から撮り直す。

``--replay`` が読む ``data/test_frames`` の中身を作るための道具で、
アプリ本体からは独立している。実演中には使わない。

使い方::

    python scripts/capture_frames.py                 12 枚を 1 秒間隔で撮る
    python scripts/capture_frames.py --count 20      枚数を指定する
    python scripts/capture_frames.py --interval 2.0  間隔[秒]を指定する
    python scripts/capture_frames.py --near          Near Mode で撮る

撮影中に砂を動かすと、再生したときに起伏の違うフレームが順に出る。
再生は撮った順に巡回するので、**山を盛る／崩すといった変化を
一連の流れで撮っておく**と、Kinect の無い環境でも動きのある確認ができる。

保存形式について
----------------
再生モードは 8bit の画像を ``1 階調 = 1mm`` として読み、
:data:`~topo_sandbox.config.REPLAY_BASE_MM` を足して深度[mm]に戻す
（:mod:`~topo_sandbox.sensor.replay` 参照）。そのためここでも
``深度[mm] - REPLAY_BASE_MM`` を 0〜255 に収めて書き出す。

この形式には次の制約がある。実機の代わりにはならず、
**パイプラインの疎通確認のためのもの**だと承知して使うこと。

- 表せる深度は ``REPLAY_BASE_MM`` から +255mm までで、外側は端に張り付く。
  センサから砂面までの距離がこの範囲に入るよう、事前に確認すること
  （範囲外の画素が多いと撮影時に警告を出す）。
- 欠測（深度 0）は 0 に丸められ、再生時には ``REPLAY_BASE_MM`` の平面として
  読み戻される。欠測そのものは再現できない。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# アプリ本体を import できるようにする。scripts/ から直接叩けるようにするためで、
# run.bat のように PYTHONPATH を整えてから呼ぶ必要をなくしている。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from topo_sandbox import config  # noqa: E402  sys.path を整えてからでないと import できない
from topo_sandbox.sensor.kinect import KinectDepthSource  # noqa: E402

#: 撮影する枚数の既定値。再生は巡回するので、多すぎても間延びする。
DEFAULT_COUNT = 12

#: 撮影の間隔[秒]の既定値。砂を動かしながら撮れる程度に空けている。
DEFAULT_INTERVAL = 1.0

#: フレームが来るのを待つ上限[ms]。
_READ_TIMEOUT_MS = 2000

#: 撮り始める前に捨てるフレーム数。開始直後は深度が安定しない。
_WARMUP_FRAMES = 10

#: 範囲外の画素がこの割合を超えたら警告する。
_OUT_OF_RANGE_WARN_RATIO = 0.10


def _to_replay_image(depth_mm):
    """深度[mm] を再生モードが読む 8bit 画像へ変換する。

    Returns:
        (画像, 範囲外だった画素の割合) の組。
    """
    offset = depth_mm.astype(np.int32) - config.REPLAY_BASE_MM
    out_of_range = np.count_nonzero((offset < 0) | (offset > 255)) / offset.size
    return np.clip(offset, 0, 255).astype(np.uint8), out_of_range


def _read_frame(source):
    """フレームが来るまで待って 1 枚返す。来なければ None。"""
    deadline = time.monotonic() + _READ_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        frame = source.read(timeout_ms=_READ_TIMEOUT_MS)
        if frame is not None:
            return frame
    return None


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="capture_frames",
        description="再生モード用の深度画像を Kinect から撮る。",
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT, help=f"撮る枚数（既定: {DEFAULT_COUNT}）"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"撮影の間隔[秒]（既定: {DEFAULT_INTERVAL}）",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=config.TEST_FRAMES_DIR,
        help="保存先（既定: data/test_frames）",
    )
    parser.add_argument(
        "--near", action="store_true", help="Near Mode で撮る（Kinect for Windows センサのみ）"
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.count < 1:
        print("枚数は 1 以上を指定してください。", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.outdir.glob("*.png"))
    if existing:
        # 古い画像が混ざったまま撮ると、再生時に前後のフレームで起伏が飛ぶ。
        print(f"警告: {args.outdir} に既に {len(existing)} 枚あります。")
        print("      撮り直すなら先に消してください（混在すると再生が飛びます）。")

    source = KinectDepthSource(near_mode=args.near)
    source.open()
    try:
        print(f"深度が安定するまで {_WARMUP_FRAMES} フレーム捨てます…")
        for _ in range(_WARMUP_FRAMES):
            _read_frame(source)

        print(f"{args.count} 枚を {args.interval} 秒間隔で撮ります。砂を動かしながらどうぞ。")
        saved = 0
        for index in range(args.count):
            depth_mm = _read_frame(source)
            if depth_mm is None:
                print(f"  [{index + 1}/{args.count}] フレームが来ませんでした。飛ばします。")
                continue

            image, out_of_range = _to_replay_image(depth_mm)
            path = args.outdir / f"frame_{index + 1:02d}.png"
            cv2.imwrite(str(path), image)
            saved += 1

            note = ""
            if out_of_range > _OUT_OF_RANGE_WARN_RATIO:
                note = (
                    f"  ※ 範囲外 {out_of_range:.0%}。"
                    f"センサと砂面の距離が {config.REPLAY_BASE_MM}〜"
                    f"{config.REPLAY_BASE_MM + 255}mm から外れています"
                )
            print(f"  [{index + 1}/{args.count}] {path.name}{note}")

            if index + 1 < args.count:
                time.sleep(args.interval)
    finally:
        source.close()

    print(f"{saved} 枚を {args.outdir} へ保存しました。")
    print("確認: scripts\\run.bat --replay")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
