import torch
import model as m
from loader import get_loader
from tqdm import tqdm


def eval1(fp, mp):
    model = m.PetRecognizer()
    model.load_state_dict(torch.load(mp, map_location=torch.device('cpu')))
    model.eval()
    _, l = get_loader(fp, 'cpu', 0.95, batch_size=1)
    cnt = 0
    cor = 0
    with torch.no_grad():
        for x, y in l:
            y_p = model(x)
            if torch.argmax(y_p[0]) == y[0]:
                cor += 1
            cnt += 1
    print(cor, cnt, cor/cnt)


def evalM3(fp, mp):
    model = m.M3()
    model.load_state_dict(torch.load(mp, map_location=torch.device('cpu')))
    model.eval()
    _, l = get_loader(fp, 'cpu', 0.95, batch_size=1)
    cnt = 0
    cor = 0
    with torch.no_grad():
        for x, y in tqdm(l):
            y_p = model(x)
            if torch.argmax(y_p[0]) == y[0]:
                cor += 1
            cnt += 1
    print(cor, cnt, cor/cnt)


evalM3('../data/pro.parquet', '../m33/m_ep8.pth')
