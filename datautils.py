import os
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def concat_nframes_feat(feat: Tensor, nframes: int) -> Tensor:
    # Output shape: (len of feat, 2 * nframes + 1, 39)
    assert nframes >= 0
    # feat: Tensor = F.pad(feat, (0, 0, nframes, nframes))
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
