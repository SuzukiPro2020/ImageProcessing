import numpy as np
import cv2 as cv
import pylab as plt

def drawlines(_img1, _img2, _lines, _pts1, _pts2):
    # _、列数
    _, col = _img1.shape

    for r, pt1, pt2 in zip(_lines, _pts1, _pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map( int, [ 0, -r[2] / r[1] ] )
        x1, y1 = map( int, [ col, -( r[2] + r[0] * col ) / r[1] ] )

        #print(x0, y0, x1, y1)

        _img1 = cv.line(_img1, (x0, y0), (x1, y1), color, 1)

        _img1 = cv.circle(_img1, tuple(pt1), 5, color, -1)
        _img2 = cv.circle(_img2, tuple(pt2), 5, color, -1)

    return _img1, _img2

def match(from_img, to_img):
    # 各画像の特徴点を取る
    from_key_points, from_descriptions = akaze.detectAndCompute(from_img, None)
    to_key_points, to_descriptions = akaze.detectAndCompute(to_img, None)

    # 2つの特徴点をマッチさせる
    bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
    matches = bf_matcher.match(from_descriptions, to_descriptions)

    # 特徴点を同士をつなぐ
    match_img = cv2.drawMatches(
        from_img, from_key_points, to_img, to_key_points,
        matches,  None, flags=2
    )

    return match_img, (from_key_points, from_descriptions, to_key_points, to_descriptions, matches)

def run():

    # 保存しておいたカメラ行列を読み込む
    cameraMatrix1 = np.loadtxt('./cameraMatrix1.csv', delimiter=',')
    cameraMatrix2 = np.loadtxt('./cameraMatrix2.csv', delimiter=',')
    distCoeffs1 = np.loadtxt('./distCoeffs1.csv', delimiter=',')
    distCoeffs2 = np.loadtxt('./distCoeffs2.csv', delimiter=',')
    R = np.loadtxt('./R.csv', delimiter=',')      # カメラ間回転行列の読みこみ
    T = np.loadtxt('./T.csv', delimiter=',')      # カメラ間並進ベクトルの読み込み

    # 画像を読み込む
    imgLeft = "./Target(left).jpg"
    imgRight = "./Target(right).jpg"

    TgtImg_L = cv.imread(imgLeft)
    TgtImg_R = cv.imread(imgRight)

    # グレースケール
    gTgtImg_L = cv.cvtColor(TgtImg_L, cv.COLOR_BGR2GRAY)
    gTgtImg_R = cv.cvtColor(TgtImg_R, cv.COLOR_BGR2GRAY)

    # 画像のノイズを削除
    imageBlur = 15
    kSize = (imageBlur, imageBlur)
    gTgtImg_L = cv.GaussianBlur(gTgtImg_L, kSize, 0)
    gTgtImg_R = cv.GaussianBlur(gTgtImg_R, kSize, 0)

    # cv.imshow('Gray Left Target Image', gTgtImg_L)
    # cv.imshow('Gray Right Target Image', gTgtImg_R)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 画像のサイズを取得
    imageWidth = TgtImg_L.shape[1]
    imageHeight = TgtImg_L.shape[0]
    imageSize = (imageWidth, imageHeight)
    # print(imageSize)

    newImageSize = (TgtImg_L.shape[1], TgtImg_L.shape[0])
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

    map1_R, map2_R = cv.initUndistortRectifyMap(
        cameraMatrix2,      # カメラ行列
        distCoeffs2,        # 歪み係数ベクトル
        R2,                 # オプション、物体空間における平行化変換（3x3 の行列
        P2,                 # newCameraMatrix
        imageSize,          # 歪み補正された画像サイズ
        mType               # 出力マップ型
    )

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

    # ----- エピポーラ線 -----
    orb = cv.AKAZE_create()

    kp1, des1 = orb.detectAndCompute(Re_TgtImg_L, None)
    kp2, des2 = orb.detectAndCompute(Re_TgtImg_R, None)


    # flaot32に変換
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # 対応点検出
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # 外れ値を取り除く
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(Re_TgtImg_L, Re_TgtImg_R, lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(Re_TgtImg_R, Re_TgtImg_L, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)

    plt.show()


if __name__ == "__main__":
    run()
