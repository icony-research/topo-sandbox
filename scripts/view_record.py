"""``--record`` で書き出した高さ[mm]（.npy）を画像として見る。

録れているかどうかを、数字ではなく絵で確かめるための道具。アプリ本体からは
独立していて、実演中には使わない。Kinect も GPU も要らない。

使い方::

    python scripts/view_record.py                       いちばん新しい録画を再生
    python scripts/view_record.py data/records/dem_...  録画を指定して再生
    python scripts/view_record.py --mode gray           白黒（高さの階調）で見る
    python scripts/view_record.py --save out            画面を出さずに PNG へ書き出す

再生中のキー操作::

    space       止める／動かす
    a / d       1 フレーム戻る／進む（止めているとき。←／→ も可）
    g           DEM の色分けと白黒を切り替え
    t           等高線の重ね描きを切り替え
    r           先頭へ戻る
    s           いま映っている絵を PNG で保存
    q / Esc     終わる

見える向きについて
------------------

既定では**投影していたときと同じ向き**（左右反転したあと）で出す。砂場を見ながら
録った人が、その場で見ていた絵と見比べられるようにするため。``--sensor`` を付けると
反転せず、.npy に入っているままの並びで出る。点群にしたときの座標と見比べたい
ときはこちら（meta.json の ``axes`` も参照）。

``--record-crop`` で切り取って録ったもの（meta.json の ``cropped`` が真）は、
**すでに投影と同じ向きになっている**ので反転しない。``--sensor`` も効かない。

白黒（``--mode gray``）の階調について
-------------------------------------

高さ 0mm を中央の灰色に置き、``--gray-span`` の幅を 0〜255 へ割り当てる
（既定 256mm なので 1 階調 = 1mm）。**フレームごとの正規化はしない。**
毎フレーム伸縮させると、砂を動かしていないのに明るさが変わり、
どこが高いのかがフレーム間で比べられなくなる。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# アプリ本体を import できるようにする。scripts/ から直接叩けるようにするためで、
# run.bat のように PYTHONPATH を整えてから呼ぶ必要をなくしている。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from topo_sandbox import config  # noqa: E402  sys.path を整えてからでないと import できない
from topo_sandbox.processing import overlays  # noqa: E402

_WINDOW_TITLE = "TopoSandbox record viewer"

#: 画面に出すときの拡大率。処理解像度 320x240 のままでは細かすぎて見えない。
DEFAULT_SCALE = 2

#: 白黒表示で 0〜255 へ割り当てる高さの幅[mm]。
#: :data:`config.DEPTH_DISPLAY_SPAN_MM` と同じにしてあるので 1 階調 = 1mm。
DEFAULT_GRAY_SPAN_MM = config.DEPTH_DISPLAY_SPAN_MM

#: 1 フレームの表示を待つ上限[ms]。index.csv の間隔が開いていても、
#: ここで頭打ちにして「固まった」と見えないようにする。
_MAX_WAIT_MS = 200

#: Kinect v1 のフレームレート。index.csv が読めないときの目安に使う。
_NOMINAL_FPS = 30

#: 間隔が分からないときの 1 フレームぶんの待ち[ms]。
_DEFAULT_WAIT_MS = int(1000 / _NOMINAL_FPS)

#: 止めているあいだにキーを見に行く間隔[ms]。
_PAUSED_WAIT_MS = 50

#: Windows の cv2.waitKeyEx が返す方向キーの値。
_KEY_LEFT = 2424832
_KEY_RIGHT = 2555904

_MODE_DEM = "dem"
_MODE_GRAY = "gray"


# ----------------------------------------------------------------------
# 録画の読み込み
# ----------------------------------------------------------------------
def find_latest(base=None):
    """いちばん新しい録画のフォルダを返す。無ければ None。

    引数なしで叩いたときに、直前に録ったものがそのまま出るようにするため。
    """
    base = base or config.RECORD_DIR
    if not base.is_dir():
        return None
    folders = sorted(path for path in base.iterdir() if path.is_dir())
    return folders[-1] if folders else None


def load_meta(directory):
    """meta.json を読む。無い・壊れているときは空の辞書。

    水位などは meta から取るが、読めなくても再生そのものはできる。
    手で集めた .npy を見たいこともあるので、ここで止めない。
    """
    path = directory / "meta.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_elapsed(directory, frame_count):
    """index.csv から各フレームの経過秒を読む。

    録画は間に合わないフレームを捨てるので、**番号の間隔と時間の間隔は
    一致しない**。実際の間隔で再生するにはこの秒が要る。

    Returns:
        フレームと同じ長さの経過秒の並び。読めなければ None。
    """
    path = directory / "index.csv"
    if not path.exists():
        return None

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        elapsed = [float(row["elapsed_s"]) for row in rows]
    except (OSError, KeyError, ValueError):
        return None

    # 途中で強制終了されていると、行数とファイル数が食い違うことがある。
    return elapsed if len(elapsed) == frame_count else None


def frame_paths(directory):
    """連番の .npy を順に並べて返す。"""
    return sorted(directory.glob("frame_*.npy"))


# ----------------------------------------------------------------------
# 絵にする
# ----------------------------------------------------------------------
def to_gray(height_mm, span_mm=DEFAULT_GRAY_SPAN_MM):
    """高さ[mm]を白黒へ落とす。高さ 0 が中央の灰色。

    基準面の高さを毎回同じ明るさに置くので、フレームをまたいで明るさを
    比べられる。フレームごとに正規化すると、砂を動かしていなくても
    明るさが変わってしまう。
    """
    height = np.asarray(height_mm, dtype=np.float32)
    scaled = (height + span_mm / 2.0) / span_mm * 255.0
    gray = np.clip(scaled, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def to_image(height_mm, mode, water_level_mm, elapsed_s, gray_span_mm):
    """高さ[mm] 1 枚を RGB 画像にする。

    DEM の色分けはアプリ本体と同じ :func:`~topo_sandbox.processing.overlays.terrain_color`
    を呼ぶ。見た目を別に作ると、投影で見ていたものと見比べられなくなる。
    """
    if mode == _MODE_GRAY:
        return to_gray(height_mm, gray_span_mm)
    return overlays.terrain_color(height_mm, elapsed_s, water_level_mm)


def to_view(image, scale, flip):
    """表示用に拡大し、必要なら左右反転して BGR にする。

    反転するのは、投影像がセンサから見た像の鏡像だから（本体の
    `Renderer._to_view` と同じ理由）。.npy はセンサから見たままの並びで
    入っているので、投影で見ていた向きに戻すにはここで反転する。
    """
    if scale != 1:
        image = cv2.resize(
            image,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    if flip:
        image = cv2.flip(image, 1)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def annotate(image, text):
    """左上に文字を重ねる。

    OpenCV の描画は日本語を出せないので、ここだけ英数字にしてある。
    黒で縁取ってから白で書くのは、明るい砂浜の上でも読めるようにするため。
    """
    position = (8, 22)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def status_text(index, total, elapsed_s, height_mm, mode, paused):
    """左上に出す 1 行。高さの範囲を添えて、録れ具合を数字でも見られるようにする。"""
    parts = [
        f"{index + 1}/{total}",
        f"t={elapsed_s:6.2f}s",
        f"h={float(height_mm.min()):+7.1f}..{float(height_mm.max()):+7.1f}mm",
        mode,
    ]
    if paused:
        parts.append("[PAUSED]")
    return "  ".join(parts)


def render_frame(height_mm, options, elapsed_s, bands=None):
    """1 枚ぶんの、画面に出す直前の絵を作る。

    等高線だけは拡大したあとに描く。線の太さを表示解像度で決めたいためで、
    本体の `Renderer._draw_contours` と同じ考え方。

    Args:
        bands: :class:`~topo_sandbox.processing.overlays.ContourBands`。
            連続再生では**使い回すこと**。1 枚ごとに作り直すと、残っている
            センサの揺れで線が細かく踊る（本体が `Renderer` に 1 つ持たせて
            いるのと同じ理由）。
    """
    image = to_image(
        height_mm,
        options.mode,
        options.water_level_mm,
        elapsed_s,
        options.gray_span_mm,
    )
    view = to_view(image, options.scale, options.flip)

    if options.contour:
        scaled = cv2.resize(
            np.asarray(height_mm, dtype=np.float32),
            dsize=(view.shape[1], view.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        if options.flip:
            scaled = cv2.flip(scaled, 1)
        view = (bands or overlays.ContourBands()).draw(scaled, view)
    return view


class _Options:
    """表示のしかた。キー操作で mode だけが変わる。"""

    def __init__(self, args, meta):
        self.mode = args.mode
        self.scale = args.scale
        self.gray_span_mm = args.gray_span
        self.contour = args.contour

        # 切り取って録ったものは、切り取りの時点で投影と同じ向きになっている
        # （renderer.crop_to_area）。ここで反転すると鏡像になってしまう。
        self.cropped = bool(meta.get("cropped"))
        self.flip = not self.cropped and not args.sensor

        # 水位は録ったときの値を使う。見ているあいだに水際が動くと、
        # 録画の中身ではなく見方のほうが変わってしまう。
        if args.water is not None:
            self.water_level_mm = args.water
        else:
            self.water_level_mm = meta.get("water_level_mm")
        if self.water_level_mm is None:
            self.water_level_mm = config.WATER_LEVEL_MM


# ----------------------------------------------------------------------
# 再生と書き出し
# ----------------------------------------------------------------------
def _wait_ms(elapsed, index):
    """次のフレームまでの待ち[ms]。index.csv が無ければ 30fps ぶん。"""
    if elapsed is None or index + 1 >= len(elapsed):
        return _DEFAULT_WAIT_MS
    delta = (elapsed[index + 1] - elapsed[index]) * 1000.0
    return int(max(1, min(_MAX_WAIT_MS, delta)))


def _save(view, directory, index):
    """いま映っている絵を PNG で残す。"""
    out_dir = directory / "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{index + 1:06d}.png"
    cv2.imwrite(str(path), view)
    return path


def play(directory, paths, elapsed, options):
    """ウィンドウに出して再生する。"""
    cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    total = len(paths)
    bands = overlays.ContourBands()

    index = 0
    paused = False
    dirty = True  # 絵を作り直す必要があるか
    height_mm = None
    view = None

    while True:
        if dirty:
            height_mm = np.load(paths[index])
            seconds = elapsed[index] if elapsed else index / _NOMINAL_FPS
            view = render_frame(height_mm, options, seconds, bands)
            annotate(view, status_text(index, total, seconds, height_mm, options.mode, paused))
            cv2.imshow(_WINDOW_TITLE, view)
            dirty = False

        # 止めているあいだも短い待ちで回す。待ち 0（キー待ち）にすると、
        # ウィンドウの×で閉じられたことに気付けず閉じられなくなる。
        # 絵は作り直さないので、回していても負荷にはならない。
        wait = _PAUSED_WAIT_MS if paused else _wait_ms(elapsed, index)
        key = cv2.waitKeyEx(wait)

        # ウィンドウの×で閉じられたら終わる。
        if cv2.getWindowProperty(_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == -1:
            # 何も押されていない。再生中なら次の 1 枚へ。
            if not paused:
                index = (index + 1) % total
                dirty = True
            continue

        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("g"):
            options.mode = _MODE_GRAY if options.mode == _MODE_DEM else _MODE_DEM
        elif key == ord("t"):
            options.contour = not options.contour
        elif key == ord("r"):
            index = 0
        elif key == ord("s"):
            print(f"保存しました: {_save(view, directory, index)}")
            continue  # 絵は変わらない
        elif key in (ord("a"), _KEY_LEFT):
            index = (index - 1) % total
            paused = True
        elif key in (ord("d"), _KEY_RIGHT):
            index = (index + 1) % total
            paused = True
        dirty = True

    cv2.destroyAllWindows()


def save_all(paths, elapsed, options, out_dir):
    """画面を出さずに、全フレームを PNG へ書き出す。

    画面の無いところで中身を確かめたいときや、まとめて別の道具へ渡したいとき用。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = overlays.ContourBands()

    for index, path in enumerate(paths):
        height_mm = np.load(path)
        seconds = elapsed[index] if elapsed else index / _NOMINAL_FPS
        view = render_frame(height_mm, options, seconds, bands)
        cv2.imwrite(str(out_dir / f"frame_{index + 1:06d}.png"), view)

    print(f"{len(paths)} 枚を {out_dir} へ書き出しました。")


# ----------------------------------------------------------------------
def _build_parser():
    parser = argparse.ArgumentParser(
        prog="view_record",
        description="--record で書き出した高さ[mm]（.npy）を画像として見る。",
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        help=f"録画のフォルダ（省略すると {config.RECORD_DIR} のいちばん新しいもの）",
    )
    parser.add_argument(
        "--mode",
        choices=[_MODE_DEM, _MODE_GRAY],
        default=_MODE_DEM,
        help="dem: 投影と同じ色分け / gray: 高さの白黒（既定: dem）",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help=f"表示の拡大率（既定: {DEFAULT_SCALE}。320x240 の何倍か）",
    )
    parser.add_argument(
        "--sensor",
        action="store_true",
        help="左右反転せず、.npy に入っているままの並びで出す（切り取り済みの録画では効かない）",
    )
    parser.add_argument(
        "--contour",
        action="store_true",
        help="等高線を重ねて始める（再生中は t で切り替え）",
    )
    parser.add_argument(
        "--water",
        type=float,
        default=None,
        help="水位[mm]を指定する（既定: 録ったときの値。meta.json から読む）",
    )
    parser.add_argument(
        "--gray-span",
        type=float,
        default=DEFAULT_GRAY_SPAN_MM,
        help=f"白黒で 0〜255 に割り当てる高さの幅[mm]（既定: {DEFAULT_GRAY_SPAN_MM:.0f}）",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="画面を出さずに、全フレームを PNG でここへ書き出す",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    directory = args.directory or find_latest()
    if directory is None:
        print(f"録画が見つかりません: {config.RECORD_DIR}", file=sys.stderr)
        print("  先に scripts\\run.bat --record で録ってください。", file=sys.stderr)
        return 2
    if not directory.is_dir():
        print(f"フォルダがありません: {directory}", file=sys.stderr)
        return 2

    paths = frame_paths(directory)
    if not paths:
        print(f"frame_*.npy がありません: {directory}", file=sys.stderr)
        return 2

    meta = load_meta(directory)
    elapsed = load_elapsed(directory, len(paths))
    options = _Options(args, meta)

    print(f"録画: {directory}")
    print(f"  {len(paths)} 枚  水位 {options.water_level_mm:+.0f}mm", end="")
    if meta.get("origin"):
        print(f"  高さの原点: {meta['origin']}", end="")
    print("  切り取り済み" if options.cropped else "  視野全体（砂場の外も入る）", end="")
    if meta.get("dropped"):
        # 取りこぼしは番号ではなく時間の間隔として現れる。先に断っておく。
        print(f"  ※録画時の取りこぼし {meta['dropped']} 枚", end="")
    print()
    if elapsed is None:
        print("  index.csv が読めないので 30fps で再生します。")

    if meta.get("origin") == "フレームの中央値":
        # 基準面なしで録ったもの。水際や雪線はあてにならない。
        print("  ※基準面なしで録られています。高さの原点がフレームごとに動くため、")
        print("    水面や標高帯の位置は実際の砂場と一致しません。")

    if args.save is not None:
        save_all(paths, elapsed, options, args.save)
        return 0

    print("  space 止める / a d 送る / g 白黒 / t 等高線 / r 先頭 / s 保存 / q 終了")
    play(directory, paths, elapsed, options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
