import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size=351, output_size=20, is_cuda=True):
        super(Net, self).__init__()
        self.device = torch.device("cuda") if is_cuda else torch.device("cpu")
        self.noise = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(0.005))

        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop_layer = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_size)

        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc5.weight, nonlinearity='relu')

        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()
        self.fc4.bias.data.zero_()
        self.fc5.bias.data.zero_()

    def forward(self, x):
        if self.training:
            x = F.relu(self.bn1(self.fc1(x + self.noise.sample(x.size()).to(self.device))))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop_layer(self.bn2(F.relu(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return torch.sigmoid(x)
