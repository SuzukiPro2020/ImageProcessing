import socket
import time

SIZE = 1024 #最大文字列
ADDRESS = '0.0.0.0'     # 受信側IPアドレス
PORT = 5000        # ポート番号

# ソケット用意
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# バインドしておく
sock.bind((ADDRESS, PORT))

ip = socket.gethostbyname(socket.gethostname())
print(ip)

while True:
    # 受信
    msg, address = sock.recvfrom(8192)
    print(f"message: {msg}\nfrom:{address}")

    if msg == '.':
        break

# ソケットを閉じる
socket.close()
