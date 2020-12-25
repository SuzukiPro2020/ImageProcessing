import numpy as np
import cv2
import time
import sys
REPO_ROOT = '..'
sys.path.append(REPO_ROOT) #新たなパスを追加
#sys.pathにパスを追加したあとでimportを行うと、追加したパスの中のモジュールがインポートできる。
from UDP import cliant


# Need adjustment
lower_light_red = np.array([150, 220, 120])
upper_light_red = np.array([200, 270, 230])


_LOWER_COLOR_d = lower_light_red
_UPPER_COLOR_d = upper_light_red


lower_light_yellow = np.array([0, 150, 160])
upper_light_yellow = np.array([50, 200, 270])

_LOWER_COLOR_h = lower_light_yellow
_UPPER_COLOR_h = upper_light_yellow


class ParticleFilter:
    def __init__(self):
        self.SAMPLEMAX = 1000
        # frame.shape
        self.height, self.width = 720, 1280

    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height
        self.X = np.random.random(self.SAMPLEMAX) * self.width

    # Need adjustment for tracking object velocity
    def modeling(self):
        self.Y += np.random.random(self.SAMPLEMAX) * 200 - 100  # 2:1
        self.X += np.random.random(self.SAMPLEMAX) * 200 - 100

    def normalize(self, weight):
        return weight / np.sum(weight)

    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)
        sample = []

        # choice by weight
        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)
        return sample

    def calcLikelihood(self, image):
        # white space tracking
        mean, std = 250.0, 10.0
        intensity = []

        for i in range(self.SAMPLEMAX):
            y, x = self.Y[i], self.X[i]
            if y >= 0 and y < self.height and x >= 0 and x < self.width:
                intensity.append(image[int(y), int(x)])
            else:
                intensity.append(-1)

        # normal distribution
        weights = 1.0 / np.sqrt(2 * np.pi * std) * \
            np.exp(-(np.array(intensity) - mean)**2 / (2 * std**2))
        weights[intensity == -1] = 0
        weights = self.normalize(weights)
        return weights

    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        self.Y = self.Y[index]
        self.X = self.X[index]

        # return COG
        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))


def video_trim(cap, width, height):
    x_l, y_l = 0, 0
    x_r, y_r = width // 2, 0
    l = cap[y_l:y_l + height, x_l:x_l + (width // 2)]
    r = cap[y_r:y_r + height, x_r:x_r + (width // 2)]
    return r, l

def assert_point_dorone(a, b, c):
    x = ''
    y = ''
    z = ''
    if a >= 0 and a < 500:
        x = '0'
    elif a >= 500 and a < 900:
        x = '1'
    elif a >= 900:
        x = '2'
    if b >= 0 and b < 250:
        y = '2'
    elif b >= 250 and b < 500:
        y = '1'
    elif b >= 500:
        y = '0'
    if c >= 0.0 and c < 4.0:
        z = '0'
    elif c >= 4.0 and c < 5.0:
        z = '1'
    elif c >= 5.0:
        z = '2'
    text = x + z + y

    return text

def assert_point_human(a,c):
    x = ''
    z = ''
    if a >= 0 and a < 500:
        x = '0'
    elif a >= 500 and a < 900:
        x = '1'
    elif a >= 900:
        x = '2'
    if c >= 0.0 and c < 4.0:
        z = '0'
    elif c >= 4.0 and c < 5.0:
        z = '1'
    elif c >= 5.0:
        z = '2'
    text = x + z

    return text

# _filterL_h :
# _filterR :
# _left :
# _right :
# sum_h :
# sum_d :

def tracking(_filterL_h, _filterR_h, _filterL_d, _filterR_d, _left, _right, sum_h, sum_d):
    hsvL = cv2.cvtColor(_left, cv2.COLOR_BGR2HSV)
    hsvR = cv2.cvtColor(_right, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only a color
    mask_hL = cv2.inRange(hsvL, _LOWER_COLOR_h, _UPPER_COLOR_h)  # human
    mask_hR = cv2.inRange(hsvR, _LOWER_COLOR_h, _UPPER_COLOR_h)  # human
    mask_dL = cv2.inRange(hsvL, _LOWER_COLOR_d, _UPPER_COLOR_d)  # dorone
    mask_dR = cv2.inRange(hsvR, _LOWER_COLOR_d, _UPPER_COLOR_d)  # dorone

    # Start Tracking
    #人の追跡
    y_hL, x_hL = _filterL_h.filtering(mask_hL)
    leftImg = cv2.circle(_left, (int(x_hL), int(y_hL)), 5, (255, 0, 0), -1)
    y_hR, x_hR = _filterR_h.filtering(mask_hR)
    rightImg = cv2.circle(_right, (int(x_hR), int(y_hR)), 5, (255, 0, 0), -1)
    #ドローンの追跡
    y_dL, x_dL = _filterL_d.filtering(mask_dL)
    leftImg = cv2.circle(_left, (int(x_dL), int(y_dL)), 5, (0, 0, 0), -1)
    y_dR, x_dR = _filterR_d.filtering(mask_dR)
    rightImg = cv2.circle(_right, (int(x_dR), int(y_dR)), 5, (0, 0, 0), -1)

    b = 60                   #カメラ間距離
    f = 1706.66666666666 * 25.4 / 72  #焦点距離(mm変換)
    #人の追跡点
    mm_hr = int(x_hR) * 25.4 / 72
    mm_hl = int(x_hL) * 25.4 / 72
    d_h = mm_hr - mm_hl          #人の追跡点の視差
    if d_h < 0:                #正の値へ変換
        d_h = abs(d_h)
    #ドローンの追跡点
    mm_dr = int(x_dR) * 25.4 / 72
    mm_dl = int(x_dL) * 25.4 / 72
    d_d = mm_dr - mm_dl          #人の追跡点の視差
    if d_d < 0:                #正の値へ変換
        d_d = abs(d_d)

    h = [x_hL, y_hL]
    d = [x_dL, y_dL]
    try:
        Z_h = f * b / d_h
        Z_h = Z_h / 1000
        Z_d = f * b / d_d
        Z_d = Z_d / 1000
        print(f"[human]距離:{Z_h}m, x:{x_hL}, y:{y_hL}")
        print(f"[dorone]距離:{Z_d}m, x:{x_dL}, y:{y_dL}")
        sum_h = Z_h
        sum_d = Z_d
    except ZeroDivisionError:
        pass

    # origin is upper left
    frame_size = _left.shape

    # for i in range(_filterL_h.SAMPLEMAX):
    #     leftImg = cv2.circle(leftImg, (int(_filterL_h.X[i]), int(
    #         _filterL_h.Y[i])), 2, (0, 0, 255), -1)

    # for i in range(_filterR_h.SAMPLEMAX):
    #     rightImg = cv2.circle(rightImg, (int(_filterR_d.X[i]), int(
    #         _filterR_d.Y[i])), 2, (0, 0, 255), -1)

    cv2.imshow("Left", leftImg)
    cv2.imshow("Right", rightImg)

    time.sleep(1)
    return sum_h, sum_d, h, d

def run():
    cap = cv2.VideoCapture(2)

    filter_l_h = ParticleFilter()
    filter_l_h.initialize()

    filter_l_d = ParticleFilter()
    filter_l_d.initialize()

    filter_r_h = ParticleFilter()
    filter_r_h.initialize()

    filter_r_d = ParticleFilter()
    filter_r_d.initialize()


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(height)


    # 保存しておいたカメラ行列を読み込む
    cameraMatrix1 = np.loadtxt('./camera_para/cameraMatrix1.csv', delimiter=',')
    cameraMatrix2 = np.loadtxt('./camera_para/cameraMatrix2.csv', delimiter=',')
    distCoeffs1 = np.loadtxt('./camera_para/distCoeffs1.csv', delimiter=',')
    distCoeffs2 = np.loadtxt('./camera_para/distCoeffs2.csv', delimiter=',')
    R = np.loadtxt('./camera_para/R.csv', delimiter=',')      # カメラ間回転行列の読みこみ
    T = np.loadtxt('./camera_para/T.csv', delimiter=',')      # カメラ間並進ベクトルの読み込み

    count = 0
    sum_h, sum_d = 0, 0
    while True:
        ret, frame = cap.read()
        right, left = video_trim(frame, width, height)


        # 画像のサイズを取得
        imageWidth  = left.shape[1]
        imageHeight = left.shape[0]
        imageSize = (imageWidth, imageHeight)

        # 平行化変換
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            cameraMatrix1,     # カメラ行列1
            distCoeffs1,       # カメラの歪みパラメータ1
            cameraMatrix2,     # カメラ行列2
            distCoeffs2,       # カメラの歪みパラメータ2
            imageSize,         # ステレオキャリブレーションで使った画像のサイズ
            R,                 # 回転行列
            T,                 # 並進ベクトル
            0,                 # flags
            1,                 # alpha
        )

        mType = cv2.CV_32FC1
        # print(mType)
        # 平行化変換マップを求める
        map1_L, map2_L = cv2.initUndistortRectifyMap(
            cameraMatrix1,      # カメラ行列
            distCoeffs1,        # 歪み係数ベクトル
            R1,                 # オプション、物体空間における平行化変換（3x3 の行列
            P1,                 # newCameraMatrix
            imageSize,          # 歪み補正された画像サイズ
            mType               # 出力マップ型
        )
            # print(map1_L)
            # print(map2_L)

        map1_R, map2_R = cv2.initUndistortRectifyMap(
            cameraMatrix2,      # カメラ行列
            distCoeffs2,        # 歪み係数ベクトル
            R2,                 # オプション、物体空間における平行化変換（3x3 の行列
            P2,                 # newCameraMatrix
            imageSize,          # 歪み補正された画像サイズ
            mType               # 出力マップ型
        )
            # print(map1_R)
            # print(map2_R)

            # ReMapにより平行化を行う
        interpolation = cv2.INTER_LINEAR

        Re_TgtImg_L = cv2.remap(
            left,       # 入力画像
            map1_L,          # 座標点1
            map2_L,          # マップ2
            interpolation    # 補間手法
        )

        Re_TgtImg_R = cv2.remap(
            right,
            map1_R,
            map2_R,
            interpolation
        )

        count = count + 1
        plus_h, plus_d, h, d  = tracking(filter_l_h, filter_r_h, filter_l_d, filter_r_d, Re_TgtImg_L, Re_TgtImg_R, sum_h, sum_d)
        sum_h += plus_h
        sum_d += plus_d
        if count > 5:
            ave_distance_h = sum_h / count
            ave_distance_d = sum_d / count
            sum_h, sum_d, count = 0, 0, 0
            pos_H = assert_point_human(h[0], ave_distance_h)
            pos_D = assert_point_dorone(d[0], d[1], ave_distance_d)
            print(f"Human:{pos_H} Dorone:{pos_D}")
            cliant.Cliant(pos_H, "192.168.1.22")
            cliant.Cliant(pos_D, "192.168.1.22")
            print(f"Dorone Y:{d[1]}")

        #time.sleep(1)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
