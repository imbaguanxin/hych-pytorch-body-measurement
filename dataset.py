import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py

class BodyMeasurementDataset(Dataset):

    def __init__(self, h5fname, transform=None):
        self.h5fname = h5fname
        self.transform = transform

        self.h5file = h5py.File(self.h5fname, 'r')
        self.front_images = self.h5file['data_front']
        self.side_images = self.h5file['data_side']
        self.labels = self.h5file['labels']
        self.label_names = self.h5file['label_names']

        self.length = len(self.labels)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        front_image = torch.tensor(self.front_images[idx])
        side_image = torch.tensor(self.side_images[idx])
        front_image = front_image.to(torch.float32) / 255.0
        side_image = side_image.to(torch.float32) / 255.0
        label = self.labels[idx]

        if self.transform:
            front_image = self.transform(front_image)
            side_image = self.transform(side_image)

        return front_image, side_image, label


if __name__ == '__main__':
    dataset = BodyMeasurementDataset('test_female.h5')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for front, side, label in dataloader:
        print(front.dtype)
        print(side.dtype)
        print(label.dtype)
        break