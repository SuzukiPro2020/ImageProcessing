import numpy as np
import cv2 as cv
import pylab as plt

# 結果表示
def showResult(im_l, im_r):#, disp):
    graph = plt.figure()
    plt.rcParams["font.size"] = 15

    # 左画像
    plt.subplot(1, 2, 1)
    plt.imshow(im_l)
    plt.title("Left")

    # 右画像
    plt.subplot(1, 2, 2)
    plt.imshow(im_r)
    plt.title("Right")

    # # 左画像
    # plt.subplot(2, 2, 3)
    # plt.imshow(disp)
    # plt.title("Disp")
    #
    # # 右画像
    # plt.subplot(2, 2, 4)
    # plt.imshow(disp, 'gray')
    # plt.title("Gray Disp")

    plt.show()

def run():

    # 保存しておいたカメラ行列を読み込む
    cameraMatrix1 = np.loadtxt('./cameraMatrix1.csv', delimiter=',')
    cameraMatrix2 = np.loadtxt('./cameraMatrix2.csv', delimiter=',')
    distCoeffs1 = np.loadtxt('./distCoeffs1.csv', delimiter=',')
    distCoeffs2 = np.loadtxt('./distCoeffs2.csv', delimiter=',')
    R = np.loadtxt('./R.csv', delimiter=',')      # カメラ間回転行列の読みこみ
    T = np.loadtxt('./T.csv', delimiter=',')      # カメラ間並進ベクトルの読み込み

    # 画像を読み込む
    imgLeft = "Target(left).jpg"
    imgRight = "Target(right).jpg"

    TgtImg_L = cv.imread(imgLeft, 0)
    TgtImg_R = cv.imread(imgRight, 0)

    # 画像のノイズを削除
    imageBlur = 15
    kSize = (imageBlur, imageBlur)
    gTgtImg_L = cv.GaussianBlur(TgtImg_L, kSize, 0)
    gTgtImg_R = cv.GaussianBlur(TgtImg_R, kSize, 0)

    # cv.imshow('Gray Left Target Image', gTgtImg_L)
    # cv.imshow('Gray Right Target Image', gTgtImg_R)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 画像のサイズを取得
    imageWidth  = TgtImg_L.shape[1]
    imageHeight = TgtImg_L.shape[0]
    imageSize = (imageWidth, imageHeight)
    print(imageSize)
    # return

    # 平行化変換
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(
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

    mType = cv.CV_32FC1
    # print(mType)
    # 平行化変換マップを求める
    map1_L, map2_L = cv.initUndistortRectifyMap(
        cameraMatrix1,      # カメラ行列
        distCoeffs1,        # 歪み係数ベクトル
        R1,                 # オプション、物体空間における平行化変換（3x3 の行列
        P1,                 # newCameraMatrix
        imageSize,          # 歪み補正された画像サイズ
        mType               # 出力マップ型
    )
    # print(map1_L)
    # print(map2_L)

    map1_R, map2_R = cv.initUndistortRectifyMap(
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
    interpolation = cv.INTER_LINEAR

    Re_TgtImg_L = cv.remap(
        gTgtImg_L,       # 入力画像
        map1_L,          # 座標点1
        map2_L,          # マップ2
        interpolation    # 補間手法
    )

    Re_TgtImg_R = cv.remap(
        gTgtImg_R,
        map1_R,
        map2_R,
        interpolation
    )


    # -- 視差を求める ---
    window_size       = 3                       # 1ブロック3ピクセルにする
    minDisparity      = 32                      # 最小の視差数
    numDisparities    = 240 - minDisparity      # 最大の視差数(16の倍数推奨)
    blockSize         = 11
    P1                = 8  * 3 * window_size * window_size
    P2                = 32 * 3 * window_size * window_size
    disp12MaxDiff     = 1                       # 視差チェックにおけて許容される最大の差
    preFilterCap      = 0                       # 画像ピクセルを切り捨てる値域
    uniquenessRatio   = 0                       # パーセント単位で表現されるマージン
    speckleWindowSize = 100                     # ノイズ削除ピクセル
    speckleRange      = 1

    stereo = cv.StereoSGBM_create(
        minDisparity      = minDisparity,
        numDisparities    = numDisparities,
        blockSize         = blockSize,
        P1                = P1,
        P2                = P2,
        disp12MaxDiff     = disp12MaxDiff,
        preFilterCap      = preFilterCap,
        uniquenessRatio   = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange      = speckleRange,
        mode              = cv.STEREO_SGBM_MODE_SGBM,
    )

    disp = stereo.compute(Re_TgtImg_L, Re_TgtImg_R).astype(np.float32) / 16.0
    # 視差データを[0.0, 1.0]に正規化
    disp = cv.normalize(disp, disp, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    showResult(Re_TgtImg_L, Re_TgtImg_R)#, disp)

if __name__ == "__main__":
    run()
