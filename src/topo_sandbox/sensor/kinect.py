"""Kinect v1 から深度フレームを取得する。

pythonnet 経由で Kinect for Windows SDK 1.8 の ``Microsoft.Kinect.dll`` を
直接呼び出す。SDK は .NET Framework 向けのため、pythonnet が Windows 既定で
使う .NET Framework ランタイムがそのまま使える。

.NET への依存はこのモジュールに閉じている。センサを別機種へ置き換える
場合、影響するのはここだけになる。
"""

import os

import numpy as np

from .base import DepthSource

#: 深度値の下位 3bit はプレイヤーインデックス。深度[mm]を得るには右シフトする。
PLAYER_INDEX_BITS = 3

_DEFAULT_SDK_DIR = r"C:\Program Files\Microsoft SDKs\Kinect\v1.8"


def _load_kinect_assembly():
    """Microsoft.Kinect.dll を読み込み、必要な型を返す。

    import 時ではなく呼び出し時に読み込む。こうしておくと、Kinect SDK の
    無い環境でもこのモジュール自体は import でき、:func:`to_millimeters` の
    ような純粋な関数を単体で検証できる。
    """
    import clr

    sdk_dir = os.environ.get("KINECTSDK10_DIR", _DEFAULT_SDK_DIR)
    clr.AddReference(os.path.join(sdk_dir, "Assemblies", "Microsoft.Kinect.dll"))

    from Microsoft.Kinect import (  # noqa: PLC0415
        DepthImageFormat,
        DepthRange,
        KinectSensor,
        KinectStatus,
    )
    from System import Array, Int16  # noqa: PLC0415

    return {
        "KinectSensor": KinectSensor,
        "KinectStatus": KinectStatus,
        "DepthImageFormat": DepthImageFormat,
        "DepthRange": DepthRange,
        "Array": Array,
        "Int16": Int16,
    }


def to_millimeters(raw, min_depth, max_depth):
    """Kinect の生の深度値をミリメートルへ変換する。

    下位 3bit のプレイヤーインデックスを落とし、有効範囲外を欠測(0)にする。

    Args:
        raw: Kinect が返す packed な深度値 (N,) uint16。
        min_depth: 有効とみなす深度の下限[mm]。
        max_depth: 有効とみなす深度の上限[mm]。

    Returns:
        深度[mm] (N,) uint16。有効範囲外は 0（欠測）。
    """
    depth_mm = np.right_shift(raw, PLAYER_INDEX_BITS)
    valid = (depth_mm >= min_depth) & (depth_mm <= max_depth)
    return np.where(valid, depth_mm, 0).astype(np.uint16)


class KinectDepthSource(DepthSource):
    """Kinect v1 の深度ストリーム。"""

    def __init__(self, near_mode=False):
        self.near_mode = near_mode
        self._kinect = None
        self._sensor = None
        self._buffer = None  # .NET short[]。毎フレーム確保し直さない
        self._width = 0
        self._height = 0

    def open(self):
        self._kinect = _load_kinect_assembly()

        sensors = [
            s
            for s in self._kinect["KinectSensor"].KinectSensors
            if s.Status == self._kinect["KinectStatus"].Connected
        ]
        if not sensors:
            raise RuntimeError(
                "Kinect センサが見つかりません。\n"
                "  - 電源アダプタが接続されているか\n"
                "  - デバイスマネージャーで Kinect for Windows Camera が正常か\n"
                "    （コード 39 の場合は Windows のメモリ整合性を無効にする。"
                "README のトラブルシューティング参照）"
            )

        self._sensor = sensors[0]
        self._sensor.DepthStream.Enable(self._kinect["DepthImageFormat"].Resolution640x480Fps30)

        if self.near_mode:
            try:
                self._sensor.DepthStream.Range = self._kinect["DepthRange"].Near
            except Exception:
                # Kinect for Xbox 360 センサは Near Mode に対応していない
                print("Near Mode を設定できませんでした。通常モードで続行します。")

        self._sensor.Start()

        length = self._sensor.DepthStream.FramePixelDataLength
        self._buffer = self._kinect["Array"].CreateInstance(self._kinect["Int16"], length)
        self._width = self._sensor.DepthStream.FrameWidth
        self._height = self._sensor.DepthStream.FrameHeight

    def read(self, timeout_ms=0):
        """深度フレームを 1 枚読む。

        既定では待ち時間 0 の非ブロッキング取得。GUI のメインループから
        呼ぶため、ここで待つと画面が固まる。

        Returns:
            深度[mm] (480, 640) uint16。欠測は 0。
        """
        frame = self._sensor.DepthStream.OpenNextFrame(timeout_ms)
        if frame is None:
            return None

        try:
            frame.CopyPixelDataTo(self._buffer)
            # 有効レンジは Near/Default の切り替えで変わるため都度取得する
            min_depth = frame.MinDepth
            max_depth = frame.MaxDepth
        finally:
            frame.Dispose()

        raw = np.frombuffer(bytearray(self._buffer), dtype=np.uint16)
        depth_mm = to_millimeters(raw, min_depth, max_depth)
        return np.reshape(depth_mm, (self._height, self._width))

    def close(self):
        if self._sensor is not None:
            self._sensor.Stop()
            self._sensor = None
