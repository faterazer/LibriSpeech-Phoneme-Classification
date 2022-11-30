from typing import Tuple

import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def test(test_loader: DataLoader, model: nn.Module, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    count = 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        count += len(y)
        with torch.no_grad():
            y_hat = model(X)
            loss = criterion(y_hat, y)
            preds = y_hat.argmax(1)
            acc = torch.sum((preds == y).float())

            running_loss += loss.item()
            running_acc += acc.item()
    return running_loss / count, running_acc / count


def train(
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        valid_steps: int,
        trial_name: str,
) -> None:
    writer = SummaryWriter(f"./logs/{trial_name}")
    train_iterator = iter(train_loader)
    best_accuracy = -1.0

    model.train()
    for step in tqdm(range(total_steps)):
        try:
            X, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            X, y = next(train_iterator)
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        preds = y_hat.argmax(1)
        batch_loss = loss.item()
        batch_acc = torch.mean((preds == y).float()).item()
        writer.add_scalars("loss", {"train_loss": batch_loss}, global_step=step)
        writer.add_scalars("acc", {"train_acc": batch_acc}, global_step=step)

        if (step + 1) % valid_steps == 0:
            valid_loss, valid_acc = test(valid_loader, model, criterion)
            model.train()
            writer.add_scalars("loss", {"valid_loss": valid_loss}, global_step=step)
            writer.add_scalars("acc", {"valid_acc": valid_acc}, global_step=step)

            if valid_acc > best_accuracy:
                best_accuracy = valid_acc
                best_state_dict = model.state_dict()
                torch.save(best_state_dict, f"./models/{trial_name}.ckpt")
                print("Saving model with acc {:.3f}".format(valid_acc))


def prediction(test_loader: DataLoader, model: nn.Module) -> None:
    preds = np.array([], dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for features in tqdm(test_loader):
            features = features.to(device)
            outputs = model(features)
            preds = np.concatenate((preds, outputs.argmax(1).cpu().numpy()), axis=0)

    with open("prediction.csv", "w") as fp:
        fp.write("Id,Class\n")
        for i, y in enumerate(preds):
            fp.write("{},{}\n".format(i, y))
