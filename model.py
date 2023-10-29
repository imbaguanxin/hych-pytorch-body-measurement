import torch
import torch.nn as nn

class ExtractorAlexNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ExtractorAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0),

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class RegressorInPaper(nn.Module):

    def __init__(self, in_channels=256, num_class=16):
        super(RegressorInPaper, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 4096, kernel_size=5, padding=0),
            nn.ReLU(),

            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=0),
            nn.Conv2d(4096, num_class, kernel_size=1, padding=0),
            nn.Flatten(),
        )
    
    def forward(self, x):
        x = self.features(x)
        return x


class RegressorInCode(nn.Module):
    
    def __init__(self, in_channels=256, dropout_rate=0.5, num_class=16):
        super(RegressorInCode, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 4096, kernel_size=5, padding=0),
            nn.Dropout(p=dropout_rate),

            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.Dropout(p=dropout_rate),

            nn.AvgPool2d(kernel_size=3, padding=0),
            nn.Conv2d(4096, num_class, kernel_size=1, padding=0),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Regressor(nn.Module):

    def __init__(self, in_channels=256, dropout_rate=0.5, num_class=16):
        super(Regressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 4096, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.AvgPool2d(kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(4096, num_class, kernel_size=1, padding=0),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class FullModel(nn.Module):

    def __init__(self, in_channels=1, num_class=16):
        super(FullModel, self).__init__()
        self.front_extractor = ExtractorAlexNet(in_channels=in_channels)
        self.side_extractor = ExtractorAlexNet(in_channels=in_channels)
        self.regressor = Regressor(num_class=num_class)

    def forward(self, front, side):
        front = self.front_extractor(front)
        side = self.side_extractor(side)
        x = torch.maximum(front, side)
        x = self.regressor(x)
        return x


class FullModelInPaper(nn.Module):

    def __init__(self, in_channels=1, num_class=16):
        super(FullModelInPaper, self).__init__()
        self.front_extractor = ExtractorAlexNet(in_channels=in_channels)
        self.side_extractor = ExtractorAlexNet(in_channels=in_channels)
        self.regressor = RegressorInPaper(num_class=num_class)

    def forward(self, front, side):
        front = self.front_extractor(front)
        side = self.side_extractor(side)
        x = torch.maximum(front, side)
        x = self.regressor(x)
        return x

class FullModelInCode(nn.Module):

    def __init__(self, in_channels=1, dropout_rate=0.5, num_class=16):
        super(FullModelInCode, self).__init__()
        self.front_extractor = ExtractorAlexNet(in_channels=in_channels)
        self.side_extractor = ExtractorAlexNet(in_channels=in_channels)
        self.regressor = RegressorInCode(dropout_rate=dropout_rate, num_class=num_class)
    
    def forward(self, front, side):
        front = self.front_extractor(front)
        side = self.side_extractor(side)
        x = torch.maximum(front, side)
        x = self.regressor(x)
        return x
    
if __name__ == "__main__":
    # Test the model
    print("Test extractor")
    model = ExtractorAlexNet()
    print(model)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    print(y.shape)

    print("Test regressor")
    regressorC = RegressorInCode()
    print(regressorC)
    yc = regressorC(y)
    print(yc.shape)

    regressorP = RegressorInPaper()
    print(regressorP)
    yp = regressorP(y)
    print(yp.shape)

    print("Test full model")
    fullC = FullModelInCode()
    print(fullC)
    x1 = torch.randn(1, 1, 224, 224)
    x2 = torch.randn(1, 1, 224, 224)
    yfullc = fullC(x1, x2)
    print(yfullc.shape)

    fullP = FullModelInPaper()
    print(fullP)
    yfullp = fullP(x1, x2)
    print(yfullp.shape)


        