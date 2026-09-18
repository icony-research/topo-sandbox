"""コマンドラインからの起動。

python -m topo_sandbox              Kinect で起動（通常）
python -m topo_sandbox --near       Near Mode
python -m topo_sandbox --replay     保存画像で起動（Kinect 不要）
python -m topo_sandbox --bench      フレーム取得性能の実測（GUI 無し）
"""

import argparse
import sys
import time

import numpy as np


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="topo_sandbox",
        description="TopoSandbox。砂場の起伏を傾斜に応じて着色し投影する。",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Kinect を使わず data/test_frames の画像を再生する",
    )
    parser.add_argument(
        "--near",
        action="store_true",
        help="Near Mode で起動する（Kinect for Windows センサのみ対応）",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="GUI を出さずにフレーム取得性能を実測する",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="--bench で計測するフレーム数（既定: 120）",
    )
    return parser


def _make_source(args):
    if args.replay:
        from .sensor.replay import ReplayDepthSource

        return ReplayDepthSource()

    from .sensor.kinect import KinectDepthSource

    return KinectDepthSource(near_mode=args.near)


def _bench(source, frames):
    """フレーム取得から 8bit 変換までの所要時間を測る。"""
    source.open()
    try:
        for _ in range(5):  # ウォームアップ
            source.read(timeout_ms=2000)

        elapsed = 0.0
        acquired = 0
        last = None
        started = time.perf_counter()

        while acquired < frames:
            begin = time.perf_counter()
            frame = source.read(timeout_ms=2000)
            if frame is None:
                continue
            elapsed += time.perf_counter() - begin
            acquired += 1
            last = frame

        wall = time.perf_counter() - started

        print(f"\n--- {acquired} フレーム取得 ---")
        print(f"取得+変換: {elapsed / acquired * 1000:6.2f} ms/frame   ※予算 33.3ms (30fps)")
        print(f"実効フレームレート: {acquired / wall:.1f} fps")

        valid = last[last != 0]
        ratio = valid.size / last.size * 100
        if valid.size:
            print(f"深度[mm]: min={valid.min()} max={valid.max()} 中央値={int(np.median(valid))}")
        print(f"有効画素: {valid.size}/{last.size} ({ratio:.1f}%)  ※欠測は 0")
    finally:
        source.close()
    return 0


def main(argv=None):
    args = _build_parser().parse_args(argv)
    source = _make_source(args)

    if args.bench:
        return _bench(source, args.frames)

    from .app import SandboxApp

    source.open()
    SandboxApp(source).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
