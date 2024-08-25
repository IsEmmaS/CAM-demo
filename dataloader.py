import glob
import random

import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_snippets import fname, parent, read
from typing import Tuple, Any


class MalariaDataset(torch.utils.data.Dataset):
    """
    Dataset for Images of Malaria cells
    """

    def __init__(self, file_path: str, transform=None):
        super(MalariaDataset, self).__init__()

        self.files = file_path
        self.transform = transform
        self.type_2_bool = {
            'Parasitized': 0,
            'Uninfected': 1
        }
        self.type_num = len(self.type_2_bool)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        fpath = self.files[index]
        class_name = fname(parent(fpath))
        img = read(fpath, 1)
        return img, class_name

    def __len__(self):
        return len(self.files)

    def choose(self):
        rand_index = random.randint(len(self.files) - 1)
        return self[rand_index]

    def collate_fn(self, batch):
        _imgs, targets = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img)[None] for img in _imgs]

        targets = [torch.tensor([self.type_2_bool[target]]).to(self.device) for target in targets]

        imgs, targets = [torch.cat(i).to(self.device) for i in [imgs, targets]]

        return imgs, targets, _imgs


train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ColorJitter(brightness=(.95, 1.05),
                  contrast=(.95, 1.05),
                  saturation=(.95, 1.05),
                  hue=0.05),
    T.RandomAffine(degrees=5,
                   translate=(.01, .1)),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5],
                std=[.5, .5, .5])
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


def load_data(file_path: str) -> tuple[DataLoader[Any], DataLoader[Any], MalariaDataset, MalariaDataset]:
    """
    Loads all images under file path
    :param file_path: the path of the images
    :return: Tuple of 2 data loaders and 2 dataset
    """
    all_files = glob.glob(f'{file_path}/*/*.png')
    train_files, val_files = train_test_split(all_files)
    train_ds = MalariaDataset(train_files, train_transform)
    val_ds = MalariaDataset(val_files, val_transform)
    train_ld, val_ld = (DataLoader(train_ds, shuffle=True, collate_fn=train_ds.collate_fn),
                        DataLoader(val_ds, shuffle=True, collate_fn=val_ds.collate_fn))

    return train_ld, val_ld, train_ds, val_ds
