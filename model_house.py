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
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(),
            nn.Linear(64, num_classes)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.fc(X)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, use_bn:bool =False, dropout:float = 0.5) -> None:
        super().__init__()
        buff = []
        for _ in range(layers):
            buff.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                buff.append(nn.BatchNorm1d(hidden_dim))
            buff.extend([nn.ReLU(), nn.Dropout(dropout)])
        self._fc = nn.Sequential(*buff)

    def forward(self, X: Tensor) -> Tensor:
        Y = self._fc(X)
        return Y + X


class ResMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 41) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(),
            ResBlock(512, 2, use_bn=True, dropout=0.5),
            ResBlock(512, 2, use_bn=True, dropout=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self._net(X)
