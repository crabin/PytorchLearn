import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lenet5 import LeNet5


def main():
    batchsz = 32

    # 定义数据预处理
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = next(iter(cifar_train))
    print(x.shape, label.shape)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    criten = torch.nn.CrossEntropyLoss()
    model = LeNet5()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模型训练
    for epoch in range(1000):
        for batchidex, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criten(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, batchidex, loss.item())


if __name__ == '__main__':
    main()
