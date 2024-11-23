import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lenet5 import LeNet5
from ResNet import ResNet18

def main():
    # 检查 GPU 是否可用
    print("CUDA available:", torch.cuda.is_available())

    # 创建一个张量并分配到 GPU
    tensor = torch.randn(1000, 1000, device="cuda")
    print("Allocated memory (bytes):", torch.cuda.memory_allocated())

    # 查看显存使用情况
    print("Reserved memory (bytes):", torch.cuda.memory_reserved())
    # 获取当前设备
    print("Current device:", torch.cuda.current_device())

    # 显示 GPU 的名称
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.__version__)  # PyTorch 版本
    print(torch.version.cuda) # PyTorch 使用的 CUDA 版本

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

    criten = torch.nn.CrossEntropyLoss().to(device)
    # model = LeNet5().to(device)
    model = ResNet18().to(device)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模型训练
    for epoch in range(1000):
        model.train()
        for batchidex, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criten(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(epoch, loss.item())

        total_correct = 0
        total_num = 0

        model.eval()
        with torch.no_grad():
            for batchidex, (x, label) in enumerate(cifar_test):
                x, label = x.to(device), label.to(device)
                logits = model(x)
                
                pred = logits.argmax(dim=1)

                total_correct += torch.eq(pred, label).sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, acc)



if __name__ == '__main__':
    main()
