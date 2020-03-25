import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset


from .args import args



# ----- Data to Image Transformer -----

def data2img(arr, font_size=50, resolution=(256, 256), font=cv2.FONT_HERSHEY_SIMPLEX):
    """ Structured Tabular Data to Image with cv2

        NOTE currently supports only iris and wine dataset
    """
    x, y = resolution
    n_colums, n_features = 2, len(arr)
    n_lines = n_features % n_colums + int(n_features / n_colums)
    frame = np.ones((*resolution, 3), np.uint8)*0

    k = 0
    # ----- iris -----
    if args.dataset=='iris':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=1, color=(255, 255, 255), thickness=2)
                    k += 1
                except IndexError:
                    break

    # ----- wine -----
    elif args.dataset=='wine':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.4, color=(255, 255, 255), thickness=1)
                    k += 1
                except IndexError:
                    break

    return np.array(frame, np.uint8)


# ----- Dataset -----

class CustomTensorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        x = self.data[0][index]
        img = data2img(x)
        if self.transform:
            x = self.transform(img)

        y = self.data[1][index]
        return x, y



# ----- Load Data Pipeline -----


def load_data(dataset=args.dataset, batch_size=args.batch_size, val_size=args.val_size, test_size=args.test_size, device='cpu'):
    # load dataset
    if dataset=='iris':
        data = datasets.load_iris()
    elif dataset=='wine':
        data = datasets.load_wine()

    # Split dataset -- Cross Vaidation
    x_train, x_test, y_train, y_test \
        = train_test_split(data.data, data.target, test_size=test_size, random_state=1)

    x_train, x_val, y_train, y_val \
        = train_test_split(x_train, y_train, test_size=val_size, random_state=1)


    # Dataset and Dataloader settings
    kwargs = {} if args.device=='cpu' else {'num_workers': 2, 'pin_memory': True}
    loader_kwargs = {'batch_size':batch_size, **kwargs}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Build Dataset
    train_data = CustomTensorDataset(data=(x_train, y_train), transform=transform)
    val_data   = CustomTensorDataset(data=(x_val, y_val), transform=transform)
    test_data  = CustomTensorDataset(data=(x_test, y_test), transform=transform)

    # Build Dataloader
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_data, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    pass

