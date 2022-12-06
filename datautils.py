import os
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing_extensions import Protocol


class Scaler(Protocol):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        ...

    def fit(self, X, y=None, sample_weight=None):
        ...

    def transform(self, X, copy=None):
        ...


def concat_nframes_feat(feat: Tensor, nframes: int) -> Tensor:
    """
    对于每一个 frame, 左右更扩展 nframes 个 frame, 共同作为特征表示

    Output shape: (len of feat, 2 * nframes + 1, 39)
    """
    assert nframes >= 0
    # feat: Tensor = F.pad(feat, (0, 0, nframes, nframes))  # 边界缺失的部分补 0

    # 对称补齐
    # 例如数据 [1, 2, 3, ...]，若 nframes = 3，对一个第一个 frame，若使用 zero padding，则对应特征为 [0, 0, 1, 2, 3],
    # 若使用对称补齐，则为 [3, 2, 1, 2, 3]
    feat = torch.cat([feat[0:nframes, :].flip(0), feat, feat[-nframes:, :].flip(0)], dim=0)
    return feat.unfold(0, 2 * nframes + 1, 1).permute(0, 2, 1)


def get_fixed_frame_data(
    nframes: int, feat_dir: str, split_filepath: str, labels_filepath: str = None
) -> Union[Tuple[Tensor, Tensor], Tensor]:
    label_dict = {}
    if labels_filepath:
        with open(labels_filepath, "r") as fp:
            for line in fp.readlines():
                line = line.strip().split()
                label_dict[line[0]] = [int(label) for label in line[1:]]

    with open(split_filepath, "r") as fp:
        datafiles = [line.strip() for line in fp.readlines()]

    features, labels = [], []
    for filename in tqdm(datafiles):
        filepath = os.path.join(feat_dir, f"{filename}.pt")
        try:
            feat = torch.load(filepath)
            features.append(concat_nframes_feat(feat, nframes))
            if label_dict:
                labels.append(torch.LongTensor(label_dict[filename]))
        except ValueError as e:
            print(e)
            print(f"[WARNING]: {filepath}.pt failed to be loaded.")

    if label_dict:
        return torch.cat(features, dim=0), torch.cat(labels)
    else:
        return torch.cat(features, dim=0)


class FixedFramesDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor = None):
        self.X = X
        self.y = None
        if y is not None:
            self.y = y

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

    def __len__(self):
        return len(self.X)


class SeqDataset(Dataset):
    def __init__(self, feat_dir: str, split_filepath: str, labels_filepath: str = None, scaler: Scaler = None) -> None:
        super().__init__()
        with open(split_filepath, "r") as fp:
            datafiles = [line.strip() for line in fp.readlines()]

        self.features = []
        failed_files = set()
        for filename in tqdm(datafiles):
            filepath = os.path.join(feat_dir, f"{filename}.pt")
            try:
                feat = torch.load(filepath).numpy()
                if scaler is not None:
                    feat = scaler.transform(feat.reshape(-1, feat.shape[-1])).reshape(feat.shape)
                self.features.append(torch.FloatTensor(feat))
            except ValueError as e:
                failed_files.add(filename)
                print(e)
                print(f"[WARNING]: {filepath}.pt failed to be loaded.")

        if labels_filepath is None:
            self.labels = None
        else:
            label_dict = {}
            with open(labels_filepath, "r") as fp:
                for line in fp.readlines():
                    line = line.strip().split()
                    label_dict[line[0]] = [int(label) for label in line[1:]]
            self.labels = [
                torch.LongTensor(label_dict[filename]) for filename in tqdm(datafiles) if filename not in failed_files
            ]

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

    def __len__(self):
        return len(self.features)
