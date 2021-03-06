import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class dev_net(nn.Module):
    def __init__(self, args):
        super(dev_net, self).__init__()
        self.args = args

        self.conv1 = nn.Conv1d(48, args.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels, args.num_channels)     # (512, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc_balance = nn.Linear(1, args.num_channels)
        self.fc_connect = nn.Linear(2*args.num_channels, args.num_channels)

        self.fc3 = nn.Linear(args.num_channels, 2)
        self.fc4 = nn.Linear(args.num_channels, 1)

    def forward(self, s):
        s, balance = s[:, :-1, :], s[:, -1, :]

        # s = F.relu(self.bn1(self.conv1(s)))     # TODO fix bn, it is not accepting 1 feature
        s = F.relu(self.conv1(s))
        s = s.squeeze(-1)

        s = torch.tanh(s)   # tanh might be helping a lot after a conv TODO find out why

        s = F.dropout(self.fc1(s), p=self.args.dropout, training=self.training)

        balance = torch.tanh(self.fc_balance(balance))
        s = self.fc_connect(torch.cat((s, balance), dim=1))

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)