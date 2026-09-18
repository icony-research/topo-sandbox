"""深度フレームの供給元の共通インタフェース。"""

import abc


class DepthSource(abc.ABC):
    """8bit の深度フレームを供給するもの。

    実装は :class:`~topo_sandbox.sensor.kinect.KinectDepthSource`（実機）と
    :class:`~topo_sandbox.sensor.replay.ReplayDepthSource`（保存画像）の 2 つ。
    アプリ側はこのインタフェースだけを見るので、センサを別機種へ置き換える
    場合もここを実装すればよい。
    """

    @abc.abstractmethod
    def open(self):
        """フレームの取得を開始する。"""

    @abc.abstractmethod
    def read(self, timeout_ms=0):
        """深度フレームを 1 枚返す。

        Args:
            timeout_ms: フレームを待つ上限[ms]。0 なら待たずに返す。

        Returns:
            深度[mm] の ndarray (480, 640) uint16。欠測は 0。
            まだフレームが来ていなければ None。
        """

    @abc.abstractmethod
    def close(self):
        """フレームの取得を終了する。"""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
