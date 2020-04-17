import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms

class Autoencoder(Dataset):
    """autoencoder dataset."""

    def __init__(self ,train =True , root_dir, transform=None , val_perc):

        self.root_dir = root_dir
        self.transform = transform
        self.frame_list = sorted(os.listdir(root_dir), key = lambda x: int(x.split(".")[0]) )
        limit = int(round(val_perc*len(self.frame_list)))
        if split == "validation":
            self.frame_list = self.frame_list[:limit]
        else :
            self.frame_list = self.frame_list[limit:]
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.frame_list[idx])
        image = io.imread(img_name)
        if self.transform:
            sample = self.transform(image)

        return sample