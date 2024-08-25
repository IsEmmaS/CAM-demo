import torch
import torch.nn as nn
"""
This file contains the implementation of model architecture
and NUB_CLASSES defined in Malaric Datasets

Model on Default device
"""

NUB_CLASSES = 10


def convBlock(in_planes, out_planes):
    """
    Convolutional Block
    :param in_planes: int number of input channels
    :param out_planes: int number of output channels
    :return: block with conv layers
    """
    return nn.Sequential(
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_planes),
        nn.MaxPool2d(2)
    )


class MalariaClassifier(nn.Module):
    """
    Malaria Classifier use Convolutional Block
    """

    def __init__(self):
        super(MalariaClassifier, self).__init__()

        self.model = nn.Sequential(
            convBlock(3, 64),
            convBlock(64, 64),
            convBlock(64, 128),
            convBlock(128, 256),
            convBlock(256, 512),
            convBlock(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUB_CLASSES)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, preds, targets):
        """
        Calculate metrics
        :param preds: tensor output of model
        :param targets: targets from datasets
        :return:
        """
        loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()

        return loss, acc
