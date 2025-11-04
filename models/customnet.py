import torch
from torch import nn
import torch.nn.functional as F

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.features = nn.Sequential(
            # B x 3 x 64 x 64
            #Block1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # B x 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # B x 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), #B x 64 x 32 x 32
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #B x 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), #B x 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), ##B x 128 x 16 x 16
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  #B x 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),   #B x 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  #B x 256 x 8 x 8
            nn.MaxPool2d(2, 2), # #B x 256 x 4 x 4
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 200),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits