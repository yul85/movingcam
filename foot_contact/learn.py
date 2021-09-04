import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
import os

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from network import Net

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, required=True)
parser.add_argument('--output_size', type=int, required=True)
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    net = Net(input_size=args.input_size, output_size=args.output_size, is_cuda=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.data_file, 'rb') as f:
        data = pickle.load(f)
    os.system(f'mkdir -p {args.output_dir}')

    input_data = np.asarray([data[i][0] for i in range(len(data))])
    output_data = np.asarray([data[i][1] for i in range(len(data))])

    x_train = torch.from_numpy(input_data.astype(np.float32))
    y_train = torch.from_numpy(output_data.astype(np.float32))

    dataset = TensorDataset(x_train, y_train)
    net = net.to(device)

    lr = 1e-4
    l2_weight_decay = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=l2_weight_decay)

    net.train()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    nb_epochs = 10000
    for epoch in range(nb_epochs + 1):
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss = F.binary_cross_entropy(net(x_train), y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 100 == 0:
                print(f'Epoch {epoch:4d}/{nb_epochs} Batch {batch_idx+1}/{len(dataloader)} Loss: {loss.item():.6f}')
        if epoch % 100 == 0:
            torch.save(net.state_dict(), f'{args.output_dir}/{epoch:05d}.pt')
