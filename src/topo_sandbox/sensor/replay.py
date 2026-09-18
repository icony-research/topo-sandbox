"""保存した深度画像を再生する。Kinect が無い場所での動作確認に使う。"""

import cv2
import numpy as np

from .. import config
from .base import DepthSource


class ReplayDepthSource(DepthSource):
    """`data/test_frames` の画像を深度[mm]として順に返す。

    実機が無い環境でも、彩色から描画までの処理パスを実機と同じように
    通せる。展示前のリハーサルや、GUI を伴う変更の確認に使う。

    .. note::
       保存画像は 8bit なので、深度の分解能は実機に及ばない。
       パイプラインの疎通確認には使えるが、見た目の最終確認は実機で行うこと。
    """

    def __init__(self, directory=None, loop=True):
        self.directory = directory or config.TEST_FRAMES_DIR
        self.loop = loop
        self._frames = []
        self._index = 0

    def open(self):
        paths = sorted(
            p for p in self.directory.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not paths:
            raise RuntimeError(f"再生できる画像がありません: {self.directory}")

        width, height = config.SENSOR_SIZE
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            if (image.shape[1], image.shape[0]) != (width, height):
                image = cv2.resize(image, dsize=(width, height))
            self._frames.append(image)

        if not self._frames:
            raise RuntimeError(f"画像を読み込めませんでした: {self.directory}")

        print(f"再生モード: {len(self._frames)} 枚の画像を読み込みました")

    def read(self, timeout_ms=0):  # noqa: ARG002  再生では待ち時間を使わない
        """深度[mm] (480, 640) uint16 を返す。"""
        if not self._frames:
            return None
        if self._index >= len(self._frames):
            if not self.loop:
                return None
            self._index = 0

        frame = self._frames[self._index]
        self._index += 1
        # 保存画像は 8bit。1 階調 = 1mm として基準面からの深度に読み替える。
        return frame.astype(np.uint16) + config.REPLAY_BASE_MM

    def close(self):
        self._frames = []
        self._index = 0
