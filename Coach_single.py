import time
import logging
import coloredlogs

from trading.pytorch.NNet import NNetWrapper as nn
from utils import *

import gym
import gym_trading

import os
import sys
import copy
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena, simple_evaluation
from MCTS_single import MCTS

import concurrent
import multiprocessing
from multiprocessing import Pool

env = gym.make('btc-dev-mcts-v1',
            state_window=48+96,      # TODO check 48+4 might not be working
            history_size=48,
            testing=True,
            columns = ['close'])

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 25,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','checkpoint_7.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


# log = logging.getLogger(__name__)



class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        # board = self.game.getInitBoard()
        canonicalBoard = self.game.reset(key=4)
        # self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            # canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)  # TODO single-player
            temp = int(episodeStep < self.args.tempThreshold)   # Boolean in int

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            trainExamples.append([canonicalBoard, pi])
            # sym = self.game.getSymmetries(canonicalBoard, pi)   # TODO what is this, gets all symmetries but why?
            # for b, p in sym:
            #     trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            canonicalBoard, _, _, _ = self.game.step(action)
            # board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            # r = self.game.getGameEnded(board, self.curPlayer)
            r = self.game.getGameEnded()

            if r != 0:
                return [(x[0], x[1], r) for x in trainExamples]

            # if r != 0:
            #     return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def mp_test2(self, x):
        self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
        # print(x)
        return self.executeEpisode()

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # TODO Multiprocessing
                """
                    - It freezes sometimes
                    - Fix tqdm bar for self learning
                    - 100 mü gelecek 25 mi?
                """
                start = time.time()
                data = Pool().map(self.mp_test2, range(100))

                for item in data:
                    iterationTrainExamples += item

                print(f'it took: {time.time() - start}')
                print('deubg')                

                # for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                #     p1 = multiprocessing.Process(target=self.mp_test2)

                #     p1.start()
                #     p1.join()
                #     # self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                #     # iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # Arena
            # TODO add a simple evaluation -for now- for our single-player env
            simple_evaluation(self.game, nmcts)

            # temp
            log.info('ACCEPTING NEW MODEL')
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            # # Original Arena
            # log.info('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            # pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            # log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            #     log.info('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     log.info('ACCEPTING NEW MODEL')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

def main():
    # log.info('Loading %s...', Game.__name__)
    g = env

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # try:
        
    # except RuntimeError:
    #     pass
    main()