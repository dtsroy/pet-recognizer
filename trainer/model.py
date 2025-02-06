import torch.nn as nn
import torchvision.models as models

class PetRecognizer(nn.Module):
    def __init__(self, input_size=224, hidden_dim=32, types=37):
        super(PetRecognizer, self).__init__()

        self.fc_in_dim = input_size * input_size * hidden_dim * 2 // 16

        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.classify = nn.Sequential(
            nn.Linear(self.fc_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, types),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_in_dim)
        x = self.classify(x)
        return x


class M2(nn.Module):
    def __init__(self, input_size=224, hidden_dim=32, types=37):
        super(M2, self).__init__()

        self.fc_in_dim = input_size * input_size * hidden_dim * 2 // 16

        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.classify = nn.Sequential(
            nn.Linear(self.fc_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, types),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_in_dim)
        x = self.classify(x)
        return x


class M3(nn.Module):
    def __init__(self, num_classes=37, freeze=True):
        super(M3, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class M4(nn.Module):
    def __init__(self, num_classes=37, freeze=True):
        super(M4, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
