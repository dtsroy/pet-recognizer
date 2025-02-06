import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt

import model
from loader import get_loader


class Trainer:
    def __init__(self, device, fp='../data/pro.parquet', mp='../models/', using_model='M4'):
        self.mp = mp
        if using_model == 'M3':
            self.model = model.M3(freeze=True)
            self.optimizer = optim.Adam(self.model.resnet.fc.parameters(), lr=1e-3)
        elif using_model == 'M4':
            self.model = model.M4(freeze=True)
            self.optimizer = optim.Adam(self.model.resnet.fc.parameters(), lr=1e-3)
        elif using_model == 'M1':
            self.model = model.PetRecognizer()
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(device)

        self.train_loader, self.test_loader = get_loader(fp, device)

        self.criterion = nn.CrossEntropyLoss()
        self.loss_v = 1e9
        self.acc_v = 0
        self.lacc_v = 0
        # print('Trainer loaded successfully.')

    def train(self, n_epoch=30):
        tr = tqdm.tqdm(range(n_epoch))
        ll, al = [], []
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
                tr.set_postfix_str(
                    'epoch=%d, loss=%.6f, cnt=%d, old_acc=%.5f, max_acc=%.5f' %
                    (epoch+1, loss.item(), cnt, self.lacc_v, self.acc_v)
                )
            tmp_loss /= cnt
            ll.append(tmp_loss)
            # if tmp_loss < self.loss_v:
            #     torch.save(self.model.state_dict(), self.mp+f'm_ep{epoch}.pt')
            #     self.loss_v = tmp_loss

            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    y_p = self.model(x)
                    _, y_p = torch.max(y_p, 1)
                    correct += (y_p == y).sum().item()
                    total += y.size(0)
            self.model.train()

            accuracy = correct / total
            if accuracy > self.acc_v:
                # torch.save(self.model.state_dict(), self.mp + f'm_ep{epoch}.pth')
                torch.save(
                    self.model.module.state_dict()
                    if isinstance(self.model, nn.DataParallel)
                    else self.model.state_dict(),
                    self.mp + f'm_ep{epoch}.pth'
                )
                self.acc_v = accuracy
            self.lacc_v = accuracy
            al.append(accuracy)

        plt.plot(ll, color='blue', label='Loss')
        plt.plot(al, color='red', label='Accuracy')
        plt.savefig('train.png')
        plt.show()


