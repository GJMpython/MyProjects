import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# 加载模型参数
model = np.load('model.npz')

# 创建神经网络实例并加载参数
nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
nn.W1 = model['W1']
nn.b1 = model['b1']
nn.W2 = model['W2']
nn.b2 = model['b2']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img  # 反转颜色，如果需要的话
    img = img / 255.0  # 归一化
    img = img.flatten()
    return img

# 处理 My_digit 文件夹中的所有图片
import os

folder_path = 'My_digits'
image_files = os.listdir(folder_path)

for image_file in image_files:
    if image_file.endswith('.png') or image_file.endswith('.jpg'):
        image_path = os.path.join(folder_path, image_file)
        hand_img = preprocess_image(image_path)
        hand_img = hand_img.reshape(1, -1)

        # 预测
        predictions = nn.predict(hand_img)

        # 显示图像和预测结果
        img = Image.open(image_path).convert('L')
        plt.imshow(img, cmap='gray')
        plt.title(f'Predicted: {predictions[0]}')
        plt.show()


