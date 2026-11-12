from torch import nn


class LungCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features.flatten(1))


def build_model(architecture, num_classes, pretrained=False):
    if architecture != 'cnn':
        raise ValueError('Only cnn is available in this project')
    return LungCNN(num_classes)


def explanation_layer(model, architecture):
    return model.features[9]
