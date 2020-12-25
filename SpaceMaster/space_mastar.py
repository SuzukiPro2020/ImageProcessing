import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import minimum_filter, maximum_filter
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import time

sys.path.append('..') #新たなパスを追加
#sys.pathにパスを追加したあとでimportを行うと、追加したパスの中のモジュールがインポートできる。
from UDP import cliant

#(3dグラフ)表示関数
def show(step, start, goal, cost, barrier):
    #通れる場所の座標、障害物の座標
    x_t, y_t, z_t = np.where(barrier == True) #配列
    x_f, y_f, z_f = np.where(barrier == False) #配列

    print(x_t, y_t, z_t)
    print(x_f, y_f, z_f)

    #3Dでグラフ表示
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    #通れる場所、障害物の表示
    ax.scatter(x_f, y_f, z_f, c='y', s=200) #通れる場所は黄色
    ax.scatter(x_t, y_t, z_t, c='k', s=200) #障害物は黒

    #スタート、ゴールの記入
    ax.text(start[0]+0.1, start[1], start[2], 'S', size = 20, color = 'r')
    ax.text(goal[0]+0.1, goal[1], goal[2], 'G', size = 20, color = 'b')

    #コストの記入
    x, y, z = np.where(cost != 999)
    c = cost[x, y, z]
    for i in range(len(x)):
        ax.text(x[i]-0.1, y[i]+0.1, z[i], c[i], size = 15, color = 'k')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.view_init(25, -75)
    #plt.savefig('save/{}.png'.format(step), bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    print()

#ルート算出関数
def show_route(route, start, goal, barrier):
    #通れる場所の座標、障害物の座標
    x_t, y_t, z_t = np.where(barrier == True)
    x_f, y_f, z_f = np.where(barrier == False)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    #通れる場所、障害物の表示
    ax.scatter(x_f, y_f, z_f, c='y', s=200) #通れる場所は黄色
    ax.scatter(x_t, y_t, z_t, c='k', s=200) #障害物は黒

    #スタート、ゴールの記入
    ax.text(start[0]+0.1, start[1], start[2], 'S', size = 20, color = 'r')
    ax.text(goal[0]+0.1, goal[1], goal[2], 'G', size = 20, color = 'b')

    #コストの記入
    for i in range(1, len(route)):
        ax.plot([route[i - 1][0], route[i][0]], [route[i - 1][1], route[i][1]], [route[i - 1][2], route[i][2]], c='r')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.view_init(25, -75)
    #plt.savefig('save/{}.png'.format(step), bbox_inches='tight', pad_inches=0)
    plt.show(); plt.close()


def read_point(move, h_point, d_point):

    maze = [['---',
             '---',
             '---'],

            ['---',
             '---',
             '---'],

            ['---',
             '---',
             '---']]

    pt0 = int(d_point[0])
    pt1 = int(d_point[1])
    pt2 = int(d_point[2])
    #この位置にSがあるかどうか調べる

    #s決め
    maze[pt0][pt1] = maze[pt0][pt1][:pt2] + 's' + maze[pt0][pt1][pt2+1:]

    #g決め
    if move == 'up':
        print("up")
        if pt2 < 2:
            maze[pt0][pt1] = maze[pt0][pt1][:pt2+1] + 'g' + maze[pt0][pt1][pt2+2:]
    elif move == 'down':
        print("down")
        if pt2 > 0:
            maze[pt0][pt1] = maze[pt0][pt1][:pt2-1] + 'g' + maze[pt0][pt1][int(d_point[2]):]
    elif move = 'left':
        print('left')
        if pt0 > 0:
            maze[pt0][pt1] = maze[pt0][pt1][:pt2-1] + 'g' + maze[pt0][pt1][int(d_point[2]):]


    #人のポジション決め
    maze[int(h_point[0])][int(h_point[1])] = '###'
    try:
        maze[abs(int(h_point[0])+1)][abs(int(h_point[1]))] = '###'
    except:
        print("10")
        pass
    try:
        maze[abs(int(h_point[0]))][abs(int(h_point[1])-1)] = '###'
    except:
        print("0-1")
        pass
    try:
        maze[abs(int(h_point[0])-1)][abs(int(h_point[1]))] = '###'
    except:
        print("-10")
        pass
    try:
        maze[abs(int(h_point[0]))][abs(int(h_point[1])+1)] = '###'
    except:
        print("01")
        pass

    print(maze)

    #たてとよこ
    h, w, l = len(maze), len(maze[0]), len(maze[0][0])
    #コスト
    cost = np.zeros((h, w, l), dtype=int) + 999
    #コストが書き込まれて探索が終了したマス（bool）
    done = np.zeros((h, w, l), dtype=bool)
    #障害物（bool）
    barrier = np.zeros((h, w, l), dtype=bool)

    start = None
    goal = None

    #mazeからスタート位置、ゴール位置、障害物位置を取得
    for i in range(h):
        for j in range(w):
            maze[i] = list(maze[i])
            for k in range(l):
                if maze[i][j][k] == 's':
                    start = (i, j, k)
                    cost[i, j, k] = 0
                    done[i, j, k] = True
                if maze[i][j][k] == 'g':
                    goal = (i, j, k)
                if maze[i][j][k] == '#':
                    barrier[i, j, k] = True

    if start is None:
        start = "null"
    if goal is None:
        goal = "null"


    return start, cost, done, goal, barrier, maze


def calc_route(start, cost, done, goal, barrier, pt0, pt1, pt2):

    #プーリング用のフィルタ
    g = np.array([[[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]],
                  [[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]],
                  [[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]]])

    for i in range(1, 999):
        #print(f"{i}回目")
        #次に進出するマスのbool
        done_next = maximum_filter(done, footprint=g) * ~done
        # print(~done)
        # print(maximum_filter(done, footprint=g))
        #print('done_next\n{}'.format(done_next))

        #次に進出するマスのcost
        cost_next = minimum_filter(cost, footprint=g) * done_next
        cost_next[done_next] += 1
        #print('cost_next\n{}'.format(cost_next))

        #costを更新
        cost[done_next] = cost_next[done_next]
        #ただし障害物のコストは999とする
        cost[barrier] = 999
        #print('cost\n{}'.format(cost))

        #探索終了マスを更新
        done[done_next] = done_next[done_next]
        #ただし障害物は探索終了としない
        done[barrier] = False
        #print('done\n{}'.format(done))

        #(3dグラフ)表示関数
        show(i, start, goal, cost, barrier)

        #終了判定
        if done[goal[0], goal[1], goal[2]] == True:
            break
    # ゴールから逆順でルート計算
    point_now = goal
    cost_now = cost[goal[0], goal[1], goal[2]]
    route = [goal]
    #print('route\n{}'.format(route))
    while cost_now > 0:
        #x-から来た場合
        try:
            if cost[point_now[0] - 1, point_now[1], point_now[2]] == cost_now - 1:
                #更新
                point_now = (point_now[0] - 1, point_now[1], point_now[2])
                cost_now = cost_now - 1
                #記録
                route.append(point_now)
        except: pass
        #x+から来た場合
        try:
            if cost[point_now[0] + 1, point_now[1], point_now[2]] == cost_now - 1:
                #更新
                point_now = (point_now[0] + 1, point_now[1], point_now[2])
                cost_now = cost_now - 1
                #記録
                route.append(point_now)
        except: pass
        #y-から来た場合
        try:
            if cost[point_now[0], point_now[1] - 1, point_now[2]] == cost_now - 1:
                #更新
                point_now = (point_now[0], point_now[1] - 1, point_now[2])
                cost_now = cost_now - 1
                #記録
                route.append(point_now)
        except: pass
        #y+から来た場合
        try:
            if cost[point_now[0], point_now[1] + 1, point_now[2]] == cost_now - 1:
                #更新
                point_now = (point_now[0], point_now[1] + 1, point_now[2])
                cost_now = cost_now - 1
                #記録
                route.append(point_now)
        except: pass
        #z-から来た場合
        try:
            if cost[point_now[0], point_now[1], point_now[2] - 1] == cost_now - 1:
                #更新
                point_now = (point_now[0], point_now[1], point_now[2] - 1)
                cost_now = cost_now - 1
                #記録
                route.append(point_now)
        except: pass
        #z+から来た場合
        try:
            if cost[point_now[0], point_now[1], point_now[2] + 1] == cost_now - 1:
                #更新
                point_now = (point_now[0], point_now[1], point_now[2] + 1)
                cost_now = cost_now - 1
                #記録
                route.append(point_now)
        except: pass

    #ルートを逆順にする
    route = route[::-1]
    print('route\n{}'.format(route))
    try:
        #表示
        show_route(route, start, goal, barrier)
    except:
        print("その場で待機")

    return route


if __name__ == '__main__': #おまなじない的な感じ
    #読み込み
    a_file = open("./action.txt")
    move = a_file.read()
    h_file = open("./human.txt")
    h_point = h_file.read()
    d_file = open("./dorone.txt")
    d_point = d_file.read()
    pt0 = int(d_point[0])
    pt1 = int(d_point[1])
    pt2 = int(d_point[2])

    error_goal = None

    n = 1
    while n > 0:
        #推定
        start, cost, done, goal, barrier, maze = read_point(move, h_point, d_point)

        #不具合対策
        if not(error_goal is None):
            goal = error_goal

        if start == 'null':
            #maze再構築
            h, w, l = len(maze), len(maze[0]), len(maze[0][0])
            dx = [1, 0, -1, 0]
            dy = [0, 1, 0, -1]
            for i in range(4):
                next_d_pos_x = pt0 + dx[i]
                next_d_pos_y = pt1 + dy[i]
                if next_d_pos_x < 0 or next_d_pos_x >= 3 or next_d_pos_y < 0 or next_d_pos_y >= 3:
                    continue
                if not(maze[next_d_pos_x][next_d_pos_y] == '###'):
                    maze[next_d_pos_x][next_d_pos_y] = '-g-'

            maze[pt0][pt1] = maze[pt0][pt1][:pt2] + 's' + maze[pt0][pt1][pt2+1:]
            print(maze)
            #------------------------------------------#
            #もう一度読み込む
            cost = np.zeros((h, w, l), dtype=int) + 999
            done = np.zeros((h, w, l), dtype=bool)
            barrier = np.zeros((h, w, l), dtype=bool)
            for i in range(len(maze)):
                for j in range(w):
                    maze[i] = list(maze[i])
                    for k in range(l):
                        if maze[i][j][k] == 's':
                            start = (i, j, k)
                            cost[i, j, k] = 0
                            done[i, j, k] = True
                        if maze[i][j][k] == 'g':
                            goal = (i, j, k)
                        if maze[i][j][k] == '#':
                            barrier[i, j, k] = True
            #------------------------------------------#
        if goal == "null":
            goal = (pt0,pt1,pt2)

        #経路算出
        route = calc_route(start, cost, done, goal, barrier, pt0, pt1, pt2)
        Route = ''
        for i in range(len(route)):
            Route += str(route[i][0])
            Route += str(route[i][1])
            Route += str(route[i][2])

        cliant.Cliant(Route, "192.168.1.21")

        #UDP再読み込み
        #
        #
        d_file = open("./dorone_goal.txt")
        d_point = d_file.read()
        #
        #
        goal = list(goal)
        d_point = list(d_point)
        d_point = [int(n) for n in d_point]
        print(goal)
        print(d_point)
        if not(goal == d_point):
            error_goal = tuple(goal)
            print("もう一度")
            continue
        n = n - 1
