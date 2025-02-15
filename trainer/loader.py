import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torchvision.transforms as transformers
import torch.nn.functional as F
from PIL import Image
from io import BytesIO


class PetDataset(Dataset):
    def __init__(self, fp, device, bin=False):
        df = pd.read_parquet(fp)
        self.length = len(df['label'])
        if not bin:
            ts = transformers.Compose([
                transformers.Resize((224, 224)),
                transformers.ToTensor(),
                transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            ts = transformers.Compose([
                transformers.Resize((224, 224)),
                transformers.ToTensor(),
                transformers.Normalize(mean=[0.5], std=[0.5]),
            ])

        self.label = torch.tensor(df['label'], dtype=torch.long).to(device)
        self.imgs = []
        if not bin:
            for k in range(self.length):
                self.imgs.append(
                    ts(
                        Image.open(BytesIO(df.loc[k, 'image']['bytes'])).convert('RGB')
                    ).to(device)
                )
        else:
            for k in range(self.length):
                t = ts(Image.open(BytesIO(df.loc[k, 'image']['bytes'])).convert('L'))
                threshold = 0.5
                self.imgs.append((t>threshold).float().to(device))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.imgs[item], self.label[item]


def get_loader(fp, device, ratio=0.7, batch_size=32, bin=False):
    origin = PetDataset(fp, device, bin)
    train, test = random_split(origin, [k := int(len(origin)*ratio), len(origin)-k])
    trainl = DataLoader(train, shuffle=True, batch_size=batch_size)
    testl = DataLoader(test, shuffle=False, batch_size=batch_size)
    return trainl, testl
