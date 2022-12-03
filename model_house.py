from torch import nn, Tensor


class Perceptron(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 41) -> None:
        super(Perceptron, self).__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, num_classes))

    def forward(self, X: Tensor) -> Tensor:
        return self.fc(X)


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 41) -> None:
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.fc(X)
