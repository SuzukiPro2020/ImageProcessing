import os
import sys
import cv2
import argparse
import chainer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import datetime
from PIL import Image
import math
import numpy as np

# モジュール検索パスの設定
REPO_ROOT = '..'
sys.path.append(REPO_ROOT) #新たなパスを追加
#sys.pathにパスを追加したあとでimportを行うと、追加したパスの中のモジュールがインポートできる。
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints

chainer.using_config('enable_backprop', False)

###############################################################################
def find_point_pose(pose, p):
    list_p = [int(pose[0][p][0]),int(pose[0][p][1])]
    return list_p

def find_point_hand(pose, p, left_top):
    if pose[p] is not None:
        left, top = left_top
        list_p = [int(pose[p][0] + left),int(pose[p][1] + top)]
        return list_p
    list_p = [0,0]
    return list_p

def angle_calc(p0, p1, p2 ):
    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)

def hide_face(x,y,orig_img):
    img = orig_img.copy()
    cv2.circle(img, (x,y), 50, (0,0,0), -1)
    return img

def ActionJudge(act_list,orig_img):
    img = orig_img.copy()
    if len(act_list) >= 2:
        listLen = len(act_list)
        maxlen = listLen - 1
        max_1len = maxlen -1
        #RightArm right to left
        if (act_list[max_1len][0] == 'RightArmToRight') and (act_list[maxlen][0] == 'RightArmToLeft'):
            print("RightArm right to left")
            overlay = img.copy()
            cv2.rectangle(overlay,(act_list[max_1len][1], act_list[max_1len][2]),(act_list[maxlen][1]+50, act_list[maxlen][2]+50),(255, 241, 0), -1)
            opacity = 0.4
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    return img

###############################################################################

def _draw_orbit(canvas, point_list, orbit_list, i):
    #初期化
    k = bf_x = bf_y = af_x = af_y = 0
    #右手首の座標
    if (point_list[0][4][0] == 0) and (point_list[0][4][1] == 0 and (i != 0)):
        orbit_list.append([orbit_list[i-1][0],orbit_list[i-1][1]])
    else:
        orbit_list.append([point_list[0][4][0],point_list[0][4][1]])
    for j in range(i):
        #右手首
        af_x = orbit_list[k][0]
        af_y = orbit_list[k][1]
        #右手首の軌道上に点を描画する
        cv2.circle(canvas, (math.ceil(af_x), math.ceil(af_y)), 2, (255, 255, 255), -1)
        if (i > 0) and (k > 0):
            bf_x = orbit_list[k-1][0]
            bf_y = orbit_list[k-1][1]
            cv2.line(canvas, (math.ceil(af_x), math.ceil(af_y)), (math.ceil(bf_x), math.ceil(bf_y)), (255, 255, 255), 1)
        k += 1
    return canvas

def PoseJudge(r_s, l_s, r_e, l_e, orig_img, poses, act_list):
    img = orig_img.copy()
    img = hide_face(int(poses[0][0][0]),int(poses[0][0][1]),img)
    #kira
    if(r_s <= 160 and r_s >= 120) and (r_e <= 85 and r_e >= 50) and (l_s <= 25 and l_s >= 0) and (l_e <= 110 and l_e >= 70):
        print("mix!")
        arch_name_hand = 'handnet'
        weight_path_hand = os.path.join(REPO_ROOT, 'models', 'handnet.npz')
        hand_detector = HandDetector(arch_name_hand, weight_path_hand)
        print("Estimating hands keypoints...")
        for pose in poses:
            unit_length = pose_detector.get_unit_length(pose)
            hands = pose_detector.crop_hands(img, pose, unit_length)
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                img = draw_hand_keypoints(img, hand_keypoints, (bbox[0], bbox[1]))
                x,y = find_point_hand(hand_keypoints, 8,(bbox[0], bbox[1]))
        overlay = img.copy()
        cv2.circle(overlay, (x,y), 20, (0, 255, 255), -1)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        return img
    #kamehame
    elif (70 <= r_s <= 125) and (70 <=l_s <= 125) and (100 <= r_e <= 150) and (50 <= l_e <= 80):
        print("kame!")
        x = (find_point_pose(poses, 4)[0] + find_point_pose(poses, 7)[0]) // 2 - 30
        y = (find_point_pose(poses, 4)[1] + find_point_pose(poses, 7)[1]) // 2 + 30
        overlay = img.copy()
        cv2.circle(overlay, (x,y), 50, (255, 241, 0), -1)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        return img
    #ha
    elif (0 <= r_s <= 50) and (170 <=l_s <= 200) and (170 <= r_e <= 190) and (170 <= l_e <= 190):
        print("ha!")
        x = (find_point_pose(poses, 4)[0] + find_point_pose(poses, 7)[0]) // 2 + 50
        y = (find_point_pose(poses, 4)[1] + find_point_pose(poses, 7)[1]) // 2 + 30
        overlay = img.copy()
        cv2.circle(overlay, (x,y), 70, (255, 241, 0), -1)
        cv2.rectangle(overlay,(find_point_pose(poses, 7)[0] + 50,  find_point_pose(poses, 7)[1] + 50),(img.shape[1], find_point_pose(poses, 7)[1] - 10),(255, 241, 0), -1)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        return img
    #RightArm - right To left
    elif (170<= r_s <= 190) and (170 <= r_e <= 190):
        print('rightArm To right stretched!')
        x = (find_point_pose(poses, 4)[0])
        y = (find_point_pose(poses, 4)[1])
        overlay = img.copy()
        cv2.circle(overlay, (x,y), 50, (255, 241, 0), -1)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        if len(act_list) >= 1:
            if not(act_list[len(act_list) - 1][0] =="RightArmToRight"):
                act_list.append(["RightArmToRight",x,y])
        return img
    elif (0<= r_s <= 10) and (170 <= r_e <= 190):
        print('rightArm To left stretched!')
        x = (find_point_pose(poses, 4)[0])
        y = (find_point_pose(poses, 4)[1])
        overlay = img.copy()
        cv2.circle(overlay, (x,y), 50, (255, 241, 0), -1)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        if len(act_list) >= 1:
            if not(act_list[len(act_list) - 1][0] == "RightArmToLeft"):
                act_list.append(["RightArmToLeft",x,y])

        return img

    return img

if __name__ == '__main__': #おまなじない的な感じ
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    #figオブジェクトを作る
    fig = plt.figure()
    #空のリストを作る
    ims = []
    orbits_list = []
    action_list = [['none',0,0]]
    #フォルダの名前
    folderName = "image_folder"
    #繰り返し変数
    count = 0
    frame_count = 0

    # load model
    arch_name = 'posenet'
    weight_path = os.path.join(REPO_ROOT, 'models', 'coco_posenet.npz')
    #posedetectorクラスのインスタンスを作成
    pose_detector = PoseDetector(arch_name, weight_path, device=args.gpu)

    #動画ファイルの読み込み
    cap = cv2.VideoCapture('../data/movie_1.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # get video frame
        ret, img = cap.read()
        if frame_count % 3 == 0:
            if not ret:
                print("Failed to capture image")
                break

            person_pose_array, _ = pose_detector(img)
                #res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

            poses, _ = pose_detector(img)
            img = draw_person_pose(img, person_pose_array)
            ############################################################################################
            r_s = angle_calc(find_point_pose(poses,3), find_point_pose(poses,2), find_point_pose(poses,1))
            l_s = angle_calc(find_point_pose(poses,6), find_point_pose(poses,5), find_point_pose(poses,1))
            r_e = angle_calc(find_point_pose(poses,4), find_point_pose(poses,3), find_point_pose(poses,2))
            l_e = angle_calc(find_point_pose(poses,7), find_point_pose(poses,6), find_point_pose(poses,1))
            print('r_s:',r_s)
            print('r_e:',r_e)
            print('l_s:',l_s)
            print('l_e:',l_e)
            img = PoseJudge(r_s, l_s, r_e, l_e, img, poses, action_list)
            img = ActionJudge(action_list, img)

            ############################################################################################
            #軌道描画
            #_draw_orbit(img, poses, orbits_list, count)
            #表示
            cv2.imshow("result", img)
            # draw and save image
            #現在時刻を取得
            dt_now = datetime.datetime.now()
            img_time = str(dt_now.month) +str(dt_now.day) + str(dt_now.hour) + str(dt_now.minute) + str(dt_now.second) + str(dt_now.microsecond)+ ".png"
            img_name_path = os.path.join(REPO_ROOT,folderName,img_time)
            print('Saving result into ' + str(img_time) + '...')
            cv2.imwrite(img_name_path, img)
            #print(poses)
            count += 1
            cv2.waitKey(30)
        frame_count += 1

    print(action_list)
    #画像ファイルの一覧を取得
    picList = glob.glob("../" + folderName + "/*.png")
    print(len(picList))
    #ファイルの作成日時にソートする
    #python3ではsortの比較するのをcmpではなくkey
    picList.sort(key = lambda x: int(os.path.getctime(x)),reverse = False)
    #画像ファイルを順々に読み込んでいく
    for i in range(len(picList)): #lenはlengthの関数（画像ファイル数の長さ（数））
        #1枚1枚のグラフを描き、appendしていく
        tmp = Image.open(picList[i])
        ims.append(tmp)
    dt_now = datetime.datetime.now()
    gif_name = str(dt_now.month) +str(dt_now.day) + str(dt_now.hour) + str(dt_now.minute)
    ims[0].save(gif_name + 'jpg_to_gif.gif',format='GIF',append_images=ims[1:],save_all=True,duration=200,loop=0)

    print("fin")
