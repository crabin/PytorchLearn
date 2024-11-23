import torch
# nn contains all of PyTorch's building blocks for neural networks
from torch import nn

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms


batch_size = 200
learning_rate = 0.01
epochs = 10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size, shuffle=True
)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = MLP().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()


for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28).to(device)
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        test_loss = 0
        correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28).to(device)
        data, target = data.to(device), target.to(device)
        logits = model(data)
        test_loss += criteon(logits, target).item()
        pred = logits.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    