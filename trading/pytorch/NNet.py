import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .TradingNNet import OthelloNNet as onnet
from .TradingNNet import dev_net

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

args = dotdict({
    'lr': 0.001,
    'dropout': 0.4,
    'epochs': 10,
    'batch_size': 8,    # 4
    'cuda': torch.cuda.is_available(),
    'num_channels': 64,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = dev_net(args)
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        # optimizer = optim.SGD(self.nnet.parameters(), lr=args.lr)
        scaler = GradScaler()

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # TODO - add mixed precision
                with autocast(enabled=False):
                
                    # compute output
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # TODO - add mixed precision - GradScaler

                # compute gradient and do SGD step
                optimizer.zero_grad()
                # total_loss.backward()
                # optimizer.step()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        # board = torch.FloatTensor(board.astype(np.float64))
        board = torch.FloatTensor(board).cuda().view(1, board.size, 1)    # board_dev
        # if args.cuda: board = board.contiguous().cuda()     # TODO check if contiguous is needed
        # board = board.view(1, self.board_x, self.board_y)   # [1, 6, 6]
        self.nnet.eval()
        # self.dev_net.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
            # pi2, v2 = self.dev_net(board_dev)     # board_dev

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi.exp().cpu().numpy()[0], v.cpu().numpy()[0]
        # return pi2.exp().cpu().numpy()[0], v2.cpu().numpy()[0]  # board_dev

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
