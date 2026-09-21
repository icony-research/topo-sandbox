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

#: 地形表示（DEM モード）の水面の高さ[mm]。基準面より下がこの値を割ると水になる。
#: 「砂を掘ると地下水面より下がって水が出る」という見立て。
#: 現場の砂の厚みに合わせて調整する値で、実機で追い込むこと。
WATER_LEVEL_MM = -40.0

#: 水の色を濃紺へ振り切る深さ[mm]。水面からここまでを浅瀬→深場で補間する。
WATER_DEEP_MM = -100.0

#: 浅瀬と深場の色（RGB）
WATER_SHALLOW_COLOR = (90, 170, 220)
WATER_DEEP_COLOR = (16, 52, 110)

#: 陸地の標高帯。(下限[mm], RGB) を下限の昇順に並べる。
#: 地形図の標高別色分けにならい、段彩で「どこが高いか」を読めるようにしている。
#: 先頭の下限は WATER_LEVEL_MM と揃えること（水際が砂浜になる）。
LAND_BANDS = [
    (WATER_LEVEL_MM, (222, 210, 165)),  # 砂浜
    (-15.0, (96, 150, 70)),  # 低地
    (15.0, (160, 185, 85)),  # 丘陵
    (40.0, (205, 180, 105)),  # 山地
    (70.0, (160, 115, 75)),  # 高地
    (100.0, (248, 248, 248)),  # 雪
]

#: 陰影起伏の光源。方位角は投影像で見たときの向き（315 度 = 北西）。
HILLSHADE_AZIMUTH_DEG = 315.0
HILLSHADE_ALTITUDE_DEG = 45.0

#: 高さの強調倍率。深度[mm]と画素間隔の比を吸収するための調整値で、
#: 大きくすると陰影が強くなる。実機で見ながら決めること。
HILLSHADE_Z_FACTOR = 0.5

#: 陰影の効き具合。0 で陰影なし、1 で最大。強くしすぎると影の側の
#: 標高帯が黒くつぶれて読めなくなる。
HILLSHADE_STRENGTH = 0.55

#: 水面のさざ波。(波長[画素], 進む向き[度], 周期[秒], 傾きの振幅) を重ねる。
#: 波長は処理解像度 PROC_SIZE の画素数で数える。
#: 傾きの振幅は水面がどれだけ傾くかで、これが光源の映り込む角度を決める。
#: 合計で 0.4 前後になると、きらめきがはっきり出る。
#: 波長の長い波ほど大きく、短い波ほど小さくしてあるのは実際の波と同じ。
#: 波長・周期とも互いに割り切れない値にして、繰り返しを目立たせない。
WATER_WAVES = [
    (34.0, 15.0, 4.1, 0.20),
    (21.0, -50.0, 2.9, 0.14),
    (13.0, 65.0, 1.9, 0.10),
    (7.0, -20.0, 1.3, 0.06),
]

#: さざ波全体の強さ。0 にすると水面が凪いで、きらめきも白波も止まる。
#: 実演中に目障りだったときの逃げ道。
WATER_WAVE_AMPLITUDE = 1.0

#: 水面の環境光と拡散光の配分。合計を 1 に近づけるほど明るい水になる。
WATER_AMBIENT = 0.62
WATER_DIFFUSE = 0.38

#: 光源の映り込み（鏡面反射）の強さと鋭さ。
#: 鋭さを上げるときらめきが小さく硬くなり、下げると水面全体がぼんやり光る。
WATER_SPECULAR = 0.85
WATER_SHININESS = 48.0

#: 映り込みの色。白から少し青を落とすと水らしくなる。
WATER_SPECULAR_COLOR = (255, 250, 232)

#: 波打ち際の白波。水際からこの深さ[mm]で消える。0 にすると白波が出ない。
WATER_SURF_DEPTH_MM = 14.0

#: 白波の濃さ。1 で真っ白。上げすぎると水際の位置が読めなくなる。
WATER_SURF_STRENGTH = 0.5

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
