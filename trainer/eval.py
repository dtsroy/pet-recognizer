import torch
from model import PetRecognizer, M2
from loader import get_loader


def eval(fp, mp):
    model = PetRecognizer()
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

def eval2(fp, mp):
    model = M2()
    model.load_state_dict(torch.load(mp, map_location=torch.device('cpu')))
    model.eval()
    _, l = get_loader(fp, 'cpu', 0.98, batch_size=1)
    cnt = 0
    cor = 0
    with torch.no_grad():
        for x, y in l:
            y_p = model(x)
            if torch.argmax(y_p[0]) == y[0]:
                cor += 1
            cnt += 1
    print(cor, cnt, cor/cnt)


eval2('../data/pro.parquet', '../models/m2_ep27.pt')
