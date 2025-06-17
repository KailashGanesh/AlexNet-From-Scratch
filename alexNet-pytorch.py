import numpy as np
import torch
import torch.nn as nn

class AlexNetTorch(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetTorch, self).__init__()

        # Conv Layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0),
            # nn.BatchNorm2d(96), # not including this, because the orginal paper didn't 
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        # Fully connected layers 
        self.FC1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        )
        self.FC3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes)
        )


    def forward(self, x):
        # Conv Pass
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        # Flatten the output for fully connected layers
        out = out.view(out.size(0), -1)

        # Fully connected pass
        out = self.FC1(out)
        out = self.FC2(out)
        out = self.FC3(out)