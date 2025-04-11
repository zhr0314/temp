import base64
from io import BytesIO
from PIL import Image
import numpy as np

# 图像处理相关的函数

def base64_to_2d_list(base64_str):
    try:
        # 解码 Base64 字符串
        img_data = base64.b64decode(base64_str)
        # 从字节数据中打开图像
        img = Image.open(BytesIO(img_data))
        # 将图像转换为灰度图
        img = img.convert('L')
        # 调整图像大小为 28x28
        img = img.resize((28, 28))
        # 获取图像的像素数据
        pixels = list(img.getdata())
        # 将一维像素列表转换为 28x28 的二维列表
        two_d_list = [pixels[i * 28:(i + 1) * 28] for i in range(28)]
        return two_d_list
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None


def two_d_list_to_image(two_d_list, output_path):
    try:
        # 将二维列表转换为一维列表
        pixels = [pixel for row in two_d_list for pixel in row]
        # 创建一个新的 28x28 的灰度图像
        img = Image.new('L', (28, 28))
        # 将像素数据放入图像中
        img.putdata(pixels)
        # 保存图像到指定路径
        img.save(output_path)
        print(f"图像已保存到 {output_path}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


def image_file_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            # 读取图像文件的二进制数据
            img_data = image_file.read()
            # 对二进制数据进行 Base64 编码
            base64_str = base64.b64encode(img_data).decode('utf-8')
            return base64_str
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, dropout_prob = 0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
device='cuda' if torch.cuda.is_available() else 'cpu'
tags=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
model = torch.load("./model.pth", map_location=torch.device(device), weights_only=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def model_predict(image):
    model.eval()
    with torch.no_grad():
        tensor = (transform(np.array(image))+1)*0.5
        tensor = tensor.unsqueeze(0)
        img_tensor = tensor
        output = model(img_tensor)
        output = torch.softmax(output,dim=1)
        return tags[torch.argmax(output).item()]

def check_single_difference(arr1, arr2):
    # 计算两个数组对应元素不同的布尔数组
    diff = arr1 != arr2
    # 统计不同元素的数量
    diff_count = np.count_nonzero(diff)
    return diff_count == 1

def check_abs_difference(arr1, arr2):
    # 计算两个数组对应元素差值的绝对值
    abs_diff = np.abs(arr1 - arr2)
    # 检查差值的绝对值数组中是否只包含 0 和 1
    unique_values = np.unique(abs_diff)
    return all(val in [0, 1] for val in unique_values)

import os
flag=os.getenv("FLAG") # 这里填入 flag

def check1(base64_str):
    data = np.array(base64_to_2d_list(base64_str))
    reference = Image.open('./reference1.png')
    reference = reference.convert('L')
    reference = reference.resize((28, 28))
    reference = np.array(reference)
    if check_single_difference(data, reference):
        if model_predict(data.astype(np.float32)) is not model_predict(reference):
            return True
    return False

def check2(base64_str):
    data = np.array(base64_to_2d_list(base64_str))
    reference = Image.open('./reference2.png')
    reference = reference.convert('L')
    reference = reference.resize((28, 28))
    reference = np.array(reference)
    if check_abs_difference(data, reference):
        if model_predict(data.astype(np.float32)) is not model_predict(reference):
            return True
    return False

import socket
from threading import Thread

FLAG_PART1 = flag[:len(flag)//2]
FLAG_PART2 = flag[len(flag)//2:]

def handle_client(conn, addr):
    try:
        print(f"[+] New connection from {addr}")
        conn.sendall(b"Welcome to the Double Question Challenge!\r\n")
        conn.sendall(b"Answer both questions correctly to get the complete flag.\r\n")
        conn.sendall(b"Each correct answer will give you part of the flag.\r\n\r\n")

        conn.sendall(f"Question 1: Input the Base64 of Picture 1:\r\n".encode())
        answer1 = read_until_enter(conn) 
        if check1(answer1) :
            conn.sendall(b"Correct! Here's the first part of the flag:\r\n")
            conn.sendall(f"{FLAG_PART1}\r\n".encode())
        else:
            conn.sendall(b"Wrong answer! No flag for you.\r\n")
            conn.close()
            return

        conn.sendall(f"\r\nQuestion 2: Input the Base64 of Picture 2:\r\n".encode())
        answer2 = read_until_enter(conn) 
        if check2(answer2) :
            conn.sendall(b"Correct! Here's the second part of the flag:\r\n")
            conn.sendall(f"{FLAG_PART2}\r\n".encode())
        else:
            conn.sendall(b"Wrong answer! You only got half the flag.\r\n")

    except Exception as e:
        print(f"[-] Error handling client: {e}")
    finally:
        conn.close()

def read_until_enter(conn):
    buffer = b""
    while True:
        data = conn.recv(1)  
        if not data or data == b"\n":  
            return buffer.decode().strip()
        buffer += data
def main():
    HOST = '0.0.0.0'
    PORT = 9999
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[*] Server listening on {HOST}:{PORT}")

    try:
        while True:
            conn, addr = server.accept()
            client_thread = Thread(target=handle_client, args=(conn, addr))
            client_thread.start()
    except KeyboardInterrupt:
        print("\r\n[!] Server shutting down...")
    finally:
        server.close()

if __name__ == "__main__":
    main()