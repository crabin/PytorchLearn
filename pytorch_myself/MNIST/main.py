
from collections import OrderedDict
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch

# 导入数据
train_datasets = dsets.MNIST('data',
                             train=True,
                             transform=transforms.ToTensor,
                             download=False)
test_datasets = dsets.MNIST('data',
                            train=False,
                            transform=transforms.ToTensor,
                            download=False)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Relu():
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = np.maximum(0, x)
        out = self.x

        return out

    # dout 为上一层传过来的导数
    def backward(self, dout):
        dx = dout
        dx[self.x <= 0] = 0
        return dx


class _sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


# 激活函数softmax的定义
def _softmax(x):
    if x.ndim == 2:
        # 因为x为二维函数，所以shape为(ndim,row,column)
        # axis=1:求各column的最大值
        # axis=2:求各row的最大值
        D = np.max(x, axis=1)
        # 由于是求各列的最大值，所以需要对x进行转置
        x = x.T - D  # 溢出对策
        # np.sum(a, axis=0) 表示的是将二维数组中的各个元素对应相加
        # axis =1 时， 表示的是二维数组中的各自维的列相加
        # axis =2 时， 表示的是二维数组中的各自维的行相加
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    D = np.max(x)
    exp_x = np.exp(x - D)
    return exp_x / np.sum(exp_x)


def cross_entropy_error(p, y):
    delta = 1e-7
    batch_size = p.shape[0]
    return np.sum(-y * np.log(p + delta)) / batch_size


def numberical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 取索引
        idx = it.multi_index
        # 取值
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)
        # 求导数 中心差值公式
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.p = None
        self.y = None

    def forward(self, x, y):
        self.y = y
        self.p = _softmax(x)
        self.loss = cross_entropy_error(self.p, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.p - self.y) / batch_size
        return dx


class TowLayerNet:
    # weight_init_std=0.01防止权重过大
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        # 输入层和隐层之间的权重
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # 隐层和输出层之间的权重
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layer = OrderedDict()
        self.layer['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layer['Relu1'] = Relu()
        self.layer['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layer['Relu2'] = Relu()
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layer.values():
            # 第一个就是Affine1层，调用此层的forward函数
            x = layer.forward(x)
        return x

    # x:输入数据，y：监督数据
    def loss(self, x, y):
        # 得到预测值
        p = self.predict(x)
        #  SoftmaxWithLoss层
        return self.lastLayer.forward(p, y)

    def accuracy(self, x, y):
        p = self.predict(x)
        # 在一维数组中argmax有一个参数axis,默认是0,表示每一列的最大值的索引 axis=1表示每一行的最大值的索引
        # 而在二维数组中默认是求矩阵中的最大值的索引，axis=1表示求每列的最大值的索引，axis=2,表示求每行的最大值索引
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        accuracy = np.sum(p == y) / float(x.shape[0]) * 100
        return accuracy

    def gradient(self, x, y):
        # forward
        self.loss(x, y)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layer.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layer['Affine1'].dW, self.layer['Affine1'].db
        grads['W2'], grads['b2'] = self.layer['Affine2'].dW, self.layer['Affine2'].db

        return grads


x_train = train_datasets.train_data.numpy().reshape(-1, 28 * 28)
# 转换为一列
y_train_tmp = train_datasets.train_labels.reshape(
    train_datasets.train_labels.shape[0], 1)
# 转换为one-hot 编码
y_train = torch.zeros(y_train_tmp.shape[0], 10).scatter_(
    1, y_train_tmp, 1).numpy()
x_test = test_datasets.test_data.numpy().reshape(-1, 28 * 28)
y_test_tmp = test_datasets.test_labels.reshape(
    test_datasets.test_labels.shape[0], 1)
# 转换为one-hot 编码
y_test = torch.zeros(y_test_tmp.shape[0], 10).scatter_(
    1, y_test_tmp, 1).numpy()


train_size = x_train.shape[0]
iters_num = 600
lr = 0.001
epoch = 5
batch_size = 100

network = TowLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in range(epoch):
    print('current epoch is :', i)
    for num in range(iters_num):
        # 在 train_size 范围内随机取 batch_size 个数
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = network.gradient(x_batch, y_batch)

        # 梯度下降
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= lr * grad[key]

        # 记录学习过程
        loss = network.loss(x_batch, y_batch)
        if num % 100 == 0:
            print(loss)
print('accuracy: %f %%' % network.accuracy(x_test, y_test))
