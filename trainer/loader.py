import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torchvision.transforms as transformers
import torch.nn.functional as F
from PIL import Image
from io import BytesIO


class PetDataset(Dataset):
    def __init__(self, fp, device):
        df = pd.read_parquet(fp)
        self.length = len(df['label'])
        ts = transformers.Compose([
            transformers.Resize((224, 224)),
            transformers.ToTensor(),
            transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # lbs = torch.tensor(df['label'], dtype=torch.long)
        # print(lbs)
        # self.label = F.one_hot(lbs, -1).to(device)
        self.label = torch.tensor(df['label'], dtype=torch.long).to(device)
        self.imgs = []
        for k in range(self.length):
            self.imgs.append(
                ts(
                    Image.open(BytesIO(df.loc[k, 'image']['bytes'])).convert('RGB')
                ).to(device)
            )

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.imgs[item], self.label[item]


def get_loader(fp, device, ratio=0.7, batch_size=32):
    origin = PetDataset(fp, device)
    train, test = random_split(origin, [k := int(len(origin)*ratio), len(origin)-k])
    trainl = DataLoader(train, shuffle=True, batch_size=batch_size)
    testl = DataLoader(test, shuffle=False, batch_size=batch_size)
    return trainl, testl
