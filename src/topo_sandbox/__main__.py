"""コマンドラインからの起動。

python -m topo_sandbox              Kinect で起動（通常）
python -m topo_sandbox --near       Near Mode
python -m topo_sandbox --replay     保存画像で起動（Kinect 不要）
python -m topo_sandbox --bench      フレーム取得性能の実測（GUI 無し）
python -m topo_sandbox --record     DEM の高さ[mm]を連番ファイルへ書き出す
python -m topo_sandbox --record --record-crop   砂場の内側だけを切り出して書き出す
python -m topo_sandbox --stream     DEM の高さ[mm]をライブ表示アプリへ送る
python -m topo_sandbox --stream ws://192.168.0.10:8000/ws/ingest   送り先を指定する

``--record`` は別アプリ（点群を大型ディスプレイへ出すもの）のテスト入力を
作るためのもの、``--stream`` はその別アプリ（topo-sandbox-live）へいまの砂場を
送るためのもので、どちらも**付けなければ何も起きない**。実演の起動手順は変わらない。
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from . import config

#: Kinect v1 のフレームレート。録画の上限が何秒ぶんかを示すのに使う。
_SENSOR_FPS = 30


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
    parser.add_argument(
        "--record",
        action="store_true",
        help="DEM 表示のあいだの高さ[mm]を連番ファイルへ書き出す（別アプリのテスト入力用）",
    )
    parser.add_argument(
        "--record-crop",
        action="store_true",
        help="--record で、砂場の四隅（エリア指定）の内側だけを切り出して書き出す",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=f"--record の書き出し先の親（既定: {config.RECORD_DIR}）",
    )
    parser.add_argument(
        "--record-frames",
        type=int,
        default=config.RECORD_MAX_FRAMES,
        help=(f"--record で書き出す上限の枚数。0 で無制限（既定: {config.RECORD_MAX_FRAMES}）"),
    )
    parser.add_argument(
        "--stream",
        nargs="?",
        const=config.STREAM_URL,
        default=None,
        metavar="URL",
        help=(
            "高さ[mm]をライブ表示アプリ（topo-sandbox-live）へ WebSocket で送る。"
            f"URL を省略すると {config.STREAM_URL}"
        ),
    )
    return parser


def _make_source(args):
    if args.replay:
        from .sensor.replay import ReplayDepthSource

        return ReplayDepthSource()

    from .sensor.kinect import KinectDepthSource

    return KinectDepthSource(near_mode=args.near)


def _make_recorder(args):
    """``--record`` のときだけ書き出し先を開く。

    開けなくても録画を諦めるだけで、アプリは動かし続ける。録画は実演の
    付帯物であって、これのために起動できないほうが困るため。

    Returns:
        :class:`~topo_sandbox.recorder.DemRecorder`、または None。
    """
    if not args.record:
        return None

    from .recorder import DemRecorder

    recorder = DemRecorder(directory=args.record_dir, max_frames=args.record_frames)
    try:
        path = recorder.open()
    except OSError as error:
        print(f"録画を始められません（録画なしで続けます）: {error}", file=sys.stderr)
        return None

    print(f"録画: {path}")
    print("      DEM 表示のあいだだけ書き出します（v で表示モードを切り替え）。")
    print("      水位を正しく出すには基準面（k）が要ります。高さの原点も基準面になります。")
    if args.record_crop:
        print("      砂場の四隅の内側だけを切り出します。エリアが未指定のあいだは書き出しません。")
    else:
        print("      センサの視野全体を書き出します（砂場の外の床や人も入ります）。")
        print("      砂場だけにするには --record-crop を付けてください。")
    if args.record_frames > 0:
        # 止め忘れでディスクを使い切らないよう上限がある。何秒ぶん・何 MB に
        # なるかは枚数だけでは分かりにくいので、目安を添える。
        width, height = config.PROC_SIZE
        seconds = args.record_frames / _SENSOR_FPS
        megabytes = args.record_frames * width * height * 4 / 1024 / 1024
        print(f"      上限 {args.record_frames} 枚（約 {seconds:.0f} 秒・約 {megabytes:.0f}MB）")
    return recorder


def _make_streamer(args):
    """``--stream`` のときだけ送信を始める。

    送り先がまだ起動していなくても待たずに起動し、つながるまで裏で試し続ける。
    websocket-client が入っていなければ配信を諦めるだけで、アプリは動かし続ける。
    配信は実演の付帯物であって、これのために起動できないほうが困るため。

    Returns:
        :class:`~topo_sandbox.streamer.DemStreamer`、または None。
    """
    if args.stream is None:
        return None

    # 入っているかだけを先に確かめる。読み込むのは送信スレッドがつなぐとき。
    if importlib.util.find_spec("websocket") is None:
        print(
            "ライブ配信を始められません（配信なしで続けます）: websocket-client がありません。"
            " pip install -r requirements.txt を確認してください",
            file=sys.stderr,
        )
        return None

    from .streamer import DemStreamer

    streamer = DemStreamer(url=args.stream)
    streamer.start()
    print(f"ライブ配信: {streamer.url}")
    print("      表示モードを問わず、高さ[mm]を送り続けます。")
    print("      砂場の四隅を指定すると、その内側だけを送ります。")
    print("      高さの原点を砂面に合わせるには基準面（k）が要ります。")
    return streamer


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
    from .renderer import Renderer

    source.open()
    renderer = Renderer(
        recorder=_make_recorder(args),
        record_crop=args.record_crop,
        streamer=_make_streamer(args),
    )
    SandboxApp(source, renderer=renderer).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
