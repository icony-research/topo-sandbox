"""定数とファイルパス。

解像度は各所で連動しているため、ひとつ変えると他も揃えて直す必要がある。
特に :data:`PROC_SIZE` は点群の点数（幅 × 高さ）を決めており、彩色処理の
負荷に直結する。
"""

from pathlib import Path

#: リポジトリのルート（src/topo_sandbox/config.py から 2 つ上）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

#: Kinect 無しで動かすときに読み込むフレーム
TEST_FRAMES_DIR = DATA_DIR / "test_frames"

#: 投影エリアの四隅を保存するファイル
AREA_FILE = PROJECT_ROOT / "area.txt"

#: Kinect の深度フレーム（640x480 固定）
SENSOR_SIZE = (640, 480)

#: 点群化・彩色を行う解像度。この幅 × 高さが点群の点数になる。
PROC_SIZE = (320, 240)

#: 表示・投影の解像度
VIEW_SIZE = (800, 600)

#: 深度画像の平滑化カーネル
BLUR_KERNEL = (7, 7)

#: Z スケール（高さの強調）の初期値と増減幅
Z_SCALE_INITIAL = 0.1
Z_SCALE_DELTA = 0.01

#: 傾斜の色分け感度の初期値・増減幅・上限
COLOR_SENSITIVITY_INITIAL = 2.0
COLOR_SENSITIVITY_DELTA = 0.1
COLOR_SENSITIVITY_MAX = 4.0

#: 法線推定の近傍探索パラメータ
NORMAL_SEARCH_RADIUS = 30
NORMAL_MAX_NEIGHBORS = 10

#: 法線の向きを揃えるための仮想カメラ高さ
NORMAL_CAMERA_HEIGHT = 500

#: 画面左上のメッセージを表示し続けるフレーム数
MESSAGE_FRAMES = 40

#: メインループの間隔[ms]
FRAME_INTERVAL_MS = 10

#: 等高線を引くときの二値化しきい値（開始値・刻み・本数）
CONTOUR_THRESHOLD_START = 0
CONTOUR_THRESHOLD_STEP = 5
CONTOUR_LEVELS = 50

#: 表示用グレースケールに割り当てる深度の幅[mm]。
#: この幅を 256 段階へ写すため、1 階調 = 1mm になる。窓の外はクリップする。
DEPTH_DISPLAY_SPAN_MM = 256

#: 再生モードで読み込む 8bit 画像を深度[mm]へ読み替えるときの基準面。
#: 1 階調 = 1mm として扱う。
REPLAY_BASE_MM = 1000

#: Kinect v1 の焦点距離[画素]（640x480 時）。基準面の傾きを度で表すために使う。
#: チェスボードによるキャリブレーションで実測した値。
SENSOR_FOCAL_PX = (587.90676126853919, 588.12623003510498)

#: 基準面の取得に使うフレーム数。30fps なので約 1 秒ぶん。
PLANE_FRAME_COUNT = 30

#: 残差がこの値[mm]を超えたら「砂がならせていない」と警告する
PLANE_RESIDUAL_WARN_MM = 10.0

#: 投影枠の四隅を粗く動かすときの移動量[画素]（Shift + 方向キー）
PROJECTOR_COARSE_STEP = 10

#: 投影枠の表示色と、選択中の角の表示色
PROJECTOR_FRAME_COLOR = "#00ff00"
PROJECTOR_CORNER_COLOR = "#ffff00"

#: 投影枠の角に描く印の大きさ[画素]
PROJECTOR_CORNER_SIZE = 12
