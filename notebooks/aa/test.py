#--------------------------------------------------------3.平行化変換
#平行化の作業はえぴぽーら線を求めやすくなる
import numpy
import cv2
from matplotlib import pyplot as plt

TgtImg_L = cv2.imread("Target(left).jpg") #という名前で保存しておく
TgtImg_R = cv2.imread("Target(right).jpg") #     　　　〃

cameraMatrix1 = numpy.loadtxt('cameraMatrix1.csv',delimiter = ',')
cameraMatrix2 = numpy.loadtxt('cameraMatrix2.csv',delimiter = ',')
distCoeffs1 = numpy.loadtxt('distCoeffs1.csv',delimiter = ',')
distCoeffs2 = numpy.loadtxt('distCoeffs2.csv',delimiter = ',')

imageSize = (TgtImg_L.shape[1],TgtImg_L.shape[0]) #shapeは画像の幅、高さを取得してくる
R = numpy.loadtxt('R.csv',delimiter = ',') #カメラ間回転行列の読みこみ
T = numpy.loadtxt('T.csv',delimiter = ',') #カメラ間並進ベクトルの読み込み

# 平行化変換のためのRとPおよび3次元変換行列Qを求める
flags = 0
alpha = 1
newimageSize = (TgtImg_L.shape[1],TgtImg_L.shape[0])
#内部パラメータと平行化行列から歪補正と平行化をあわせたマップを求める(stereoRectifyについてです)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, flags, alpha, newimageSize)

# 平行化変換マップを求める
m1type = cv2.CV_32FC1
map1_L, map2_L = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R1, P1, newimageSize, m1type) #m1type省略不可
map1_R, map2_R = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R2, P2, newimageSize, m1type)

# ReMapにより平行化を行う
interpolation = cv2.INTER_NEAREST # INTER_RINEARはなぜか使えない
Re_TgtImg_L = cv2.remap(TgtImg_L, map1_L, map2_L, interpolation)
Re_TgtImg_R = cv2.remap(TgtImg_R, map1_R, map2_R, interpolation) #interpolation省略不可

cv2.imshow('Rectified Left Target Image', Re_TgtImg_L)
cv2.imshow('Rectified Right Target Image', Re_TgtImg_R)
cv2.waitKey(0)  # なにかキーを押したらウィンドウを閉じる
cv2.destroyAllWindows()
#平行化した画像を保存
cv2.imwrite('RectifiedLeft.png', Re_TgtImg_L)
cv2.imwrite('RectifiedRight.png', Re_TgtImg_R)

#--------------------------------------------------------4.2.2 セミグローバルブロックマッチング(SGBM)
window_size = 25
min_disp = -144
num_disp = 176 - min_disp

#stereo = cv2.StereoSGBM_create(
#    numDisparities = num_disp,
#    blockSize = 17,
#    P1 = 8*3*window_size**2,
#    P2 = 32*3*window_size**2,
#    disp12MaxDiff = 1,
#    uniquenessRatio = 10,
#    speckleWindowSize = 50,
#    speckleRange = 32
#)
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 7,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 0,
    preFilterCap = 10,
    uniquenessRatio = 15,
    speckleWindowSize = 10,
    speckleRange = 2,
    mode=cv2.STEREO_SGBM_MODE_HH
)

#stereo = cv2.StereoBM_create(numDisparities = 256, blockSize = 13)

# 視差を求める
print('computing disparity...')
disp = stereo.compute(Re_TgtImg_L,Re_TgtImg_R)
disp = numpy.int16(disp)
disp = cv2.normalize(disp, disp, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#--------------------------------------------------------5.3次元座標への変換
import pylab as plt
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

text2 = '''

# ply形式の3Dモデルファイルを生成
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = numpy.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        numpy.savetxt(f, verts,"%f %f %f %d %d %d")
'''

def bgr2rbg(im):
    b,g,r = cv2.split(im)
    im = cv2.merge([r,g,b])
    return im

# 結果の表示
def show_result(im_l,im_r,disp):
    graph = plt.figure()
    plt.rcParams["font.size"]=15
    # 左画像
    plt.subplot(2,2,1),plt.imshow(bgr2rbg(im_l))
    plt.title("Left Image")
    # 右画像
    plt.subplot(2,2,2),plt.imshow(bgr2rbg(im_r))
    plt.title("Right Image")
    # 視差画像
    plt.subplot(2,2,3),plt.imshow(disp,"gray")
    plt.title("Disparity")
    plt.show()

# 視差画像からx,y,z座標を取得
#print('generating 3d point cloud...')
#points = cv2.reprojectImageTo3D(disp, Q)
# RGBを取得
#colors = cv2.cvtColor(Re_TgtImg_L,cv2.COLOR_BGR2RGB)
# 最小視差(-16)より大きな値を抽出
#mask = disp > min_disp
#mask = disp > disp.min()
#out_points = points[mask]
#out_colors = colors[mask]
# plyファイルを生成
#write_ply("out.ply", out_points, out_colors)

# 結果表示
show_result(Re_TgtImg_L, Re_TgtImg_R,(disp-min_disp)/(num_disp-min_disp))
print("fin")
