import sys
sys.path.append("../")

import argparse
import torch
from train_resnet import train_resnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train ResNet")

    parser.add_argument("-d", "--dataset", type=str,
                        help="dataset to train on: ['X', 'pho_id']", default='X')
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device",
                        default=-1)
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=20)
    parser.add_argument("-l", "--num_layers", type=int, help="total number of layers(blocks)", default=2)
    parser.add_argument("-i", "--hidden_dim", type=int, help="hidden dimensions", default=128)
    parser.add_argument("-r", "--learning_rate", type=float, default=1.5e-5)
    
    args = parser.parse_args()
    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))

    train_resnet(device=device, lr=args.learning_rate, num_layers=args.num_layers, hidden_dim=args.hidden_dim, epochs=args.num_epochs, train_data=args.dataset)
