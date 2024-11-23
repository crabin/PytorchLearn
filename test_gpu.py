from torchvision import models
import torch

# 加载 ResNet50 模型
model = models.resnet50().cuda()

# 创建模拟输入
inputs = torch.randn(64, 3, 224, 224, device="cuda")  # Batch size: 64

# 运行推理
outputs = model(inputs)

# 输出显存使用情况
print("Allocated memory (bytes):", torch.cuda.memory_allocated())
print("Reserved memory (bytes):", torch.cuda.memory_reserved())
