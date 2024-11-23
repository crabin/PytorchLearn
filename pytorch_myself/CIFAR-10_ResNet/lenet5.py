import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # [b,3,32,32] => [b,16,5,5]
        x = self.conv_unit(x)
        # [b,16,5,5] => [b,16*5*5]
        x = x.view(x.size(0), -1)
        # [b,16*5*5] => [b,10]
        logits = self.fc_unit(x)

        # loss = nn.CrossEntropyLoss(logits, y)

        return logits


def main():
    net = LeNet5()
    # [b,3,32,32]
    tmp = torch.randn(2, 3, 32, 32)
    # [b,16,5,5]
    out = net(tmp)
    print(out.parms())


if __name__ == '__main__':
    main()
