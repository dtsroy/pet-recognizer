import torch
import torch.optim as optim
import torch.nn as nn
import tqdm

from model import PetRecognizer
from loader import get_loader


class Trainer:
    def __init__(self, device, fp='../data/pro.parquet', mp='../models/'):
        self.mp = mp
        self.model = PetRecognizer().to(device)
        self.train_loader, self.test_loader = get_loader(fp, device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_v = 1e9
        # print('Trainer loaded successfully.')

    def train(self, epoch=30):
        tr = tqdm.tqdm(range(epoch))
        for epoch in tr:
            tmp_loss = 0
            cnt = 0
            for x, y in self.train_loader:
                y_p = self.model(x)
                loss = self.criterion(y_p, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tmp_loss += loss.item()
                cnt += 1
                tr.set_postfix_str('epoch=%d, loss=%.6f, cnt=%d' % (epoch+1, loss.item(), cnt))
            tmp_loss /= cnt
            if tmp_loss < self.loss_v:
                torch.save(self.model.state_dict(), self.mp+f'm_ep{epoch}.pt')
                self.loss_v = tmp_loss
