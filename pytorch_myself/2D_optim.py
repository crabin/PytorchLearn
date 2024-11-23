import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def himmelbau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print(x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print(X.shape, Y.shape)
Z = himmelbau([X, Y])

fig = plt.figure('himmelbau')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

x = torch.tensor([0.,0.], requires_grad=True)

optimizer = torch.optim.Adam([x], lr=1e-3)

for i in range(20000):
    gred = himmelbau(x)
    optimizer.zero_grad()
    gred.backward()
    optimizer.step()
    if i % 2000 == 0:
        print(f'x: {x}, gred: {gred}')