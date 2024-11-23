import torch.nn as nn
import torch
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.extra = nn.Sequential()

        if in_channels != out_channels:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
        self.blk1 = ResBlk(16, 16) 
        self.blk2 = ResBlk(16, 32)
     

        
        self.outlayer = nn.Linear(32 * 32 * 32, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
    
        x = x.view(x.size(0), -1)
        x  = self.outlayer(x)
        return x


def main():
    blk = ResBlk(64, 128)
    tmp  = torch.randn(2, 64, 32, 32)
    out = blk(tmp)

    print(out.shape)

    model = ResNet18()
    tmp  = torch.randn(2, 3, 32, 32)
    out = model(tmp)
    print(out.shape)

if __name__ == '__main__':
    main()
