#coding:utf-8

import socket

def Cliant(text,adress):
    SIZE = 1024
    ADDRESS = adress    #'127.0.0.1'（今はローカル）
    PORT = 5000        # ポート番号
    # サーバのアドレスを用意
    serv_address = (ADDRESS, PORT)
    # ソケット作成
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    msg = text
    # 送信
    sock.sendto(msg.encode('utf-8'), (ADDRESS, PORT))
    sock.close()
