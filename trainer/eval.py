import torch
import model as m
from loader import get_loader
from tqdm import tqdm


def eval_model(using_model, fp, mp):
    if using_model == 'M3':
        model = m.M3(freeze=True)
    elif using_model == 'M4':
        model = m.M4(freeze=True)
    else:
        model = m.PetRecognizer()
    model.load_state_dict(torch.load(mp, map_location=torch.device('cpu')))
    model.eval()
    _, l = get_loader(fp, 'cpu', 0.99, batch_size=1)
    print('successfully loaded.')
    cnt = 0
    cor = 0
    with torch.no_grad():
        for x, y in tqdm(l):
            y_p = model(x)
            print(y_p)
            print(torch.nn.functional.softmax(y_p[0]))
            if torch.argmax(y_p[0]) == y[0]:
                cor += 1
            cnt += 1
    print(cor, cnt, cor/cnt)


eval_model('M4', '../data/pro.parquet', '../m4/m_ep23.pth')
