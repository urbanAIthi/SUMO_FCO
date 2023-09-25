import torch
from torchvision import models, transforms
from torch import nn


# Define the new ResNet model
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        # Load the pre-trained ResNet50 model
        self.resnet = models.resnet34(pretrained=False)

        # Replace the first convolution layer to accept 400x400 images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the last fully connected layer to output 1 feature
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 64)
        self.add1 = nn.Linear(64+2, 64)
        self.add2 = nn.Linear(64, 1)

    def forward(self, x, v):
        x = self.resnet(x)
        x = torch.relu(x)
        x = torch.cat((x, v), dim=1)
        x = self.add1(x)
        x = torch.relu(x)
        x = self.add2(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    model = CustomResNet()
    out = model.forward(torch.rand(1, 3, 400, 400), torch.rand(1, 2))
    print(out)
