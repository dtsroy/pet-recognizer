from train import Trainer
import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def main():
    trainer = Trainer(device)
    trainer.train()


if __name__ == '__main__':
    main()
