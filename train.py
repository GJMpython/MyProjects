import struct
import numpy as np

# 【1】预处理MNIST数据集
def load_mnist_images(filename, num_images):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images[:num_images] / 255.0

def load_mnist_labels(filename, num_labels):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels[:num_labels]

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

train_images = load_mnist_images('train-images.idx3-ubyte', 10000)
train_labels = load_mnist_labels('train-labels.idx1-ubyte', 10000)
train_labels = one_hot_encode(train_labels)

test_images = load_mnist_images('t10k-images.idx3-ubyte', 200)
test_labels = load_mnist_labels('t10k-labels.idx1-ubyte', 200)
test_labels = one_hot_encode(test_labels)

train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# 【2】定义网络
class NeuralNetwork:
    # 网络结构
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    # 激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # 激活函数的导数
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    # 前向传播函数
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    # 后向传播函数
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.hidden_error = np.dot(self.output_delta, self.W2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.A1)

        self.W2 += np.dot(self.A1.T, self.output_delta) * self.learning_rate
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += np.dot(X.T, self.hidden_delta) * self.learning_rate
        self.b1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate
    # 随机梯度下降法训练网络
    def train(self, X, y, epochs=500, batch_size=32, print_every=10):
        num_samples = X.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

            if epoch % print_every == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

input_size = train_images.shape[1]
hidden_size = 128
output_size = 10
learning_rate = 0.1

# 【3】训练网络
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(train_images, train_labels, epochs=500, print_every=10)

# 【4】保存模型参数
np.savez('model.npz', W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)

# 【5】评估网络
predictions = nn.predict(test_images)
accuracy = np.mean(predictions == np.argmax(test_labels, axis=1))
print(f"Accuracy on small test set: {accuracy * 100:.2f}%")
