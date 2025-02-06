from train import Trainer
import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# import torch_directml
# device = torch_directml.device()


def main():
    trainer = Trainer(device, mp='../m4/', using_model='M4')
    trainer.train()


if __name__ == '__main__':
    main()
