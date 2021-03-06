import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws


# Single-player MCTS
import numpy as np
import matplotlib.pyplot as plt

# Plotting
def plot_end_result(plot_price, plot_action):
    time = np.arange(len(plot_price))
    # colors {0: 'b', 1: 'r'}
    print(plot_action)

    plt.plot(plot_price, zorder=0)
    plt.scatter(x=time,y=plot_price,c=['b' if x == 1 else 'r' for x in plot_action])
    
    plt.show()

    print('debug')

# plot_end_result([4, 10, 40, 20, 18], [1, 1, 0, 0, 1])

def simple_evaluation(game, mcts):
    trainExamples = []
    # board = game.getInitBoard()
    canonicalBoard = game.reset(key=0)
    episodeStep = 0

    plot_price = []
    plot_action = []

    while True:
        # canonicalBoard = game.getCanonicalForm(board, 1)

        pi = mcts.getActionProb(canonicalBoard, temp=0)
        trainExamples.append([canonicalBoard, pi])

        action = np.argmax(pi)

        plot_price.append(game.current_price)
        plot_action.append(action)

        canonicalBoard, _, _, info = game.step(action)

        r = game.getGameEnded()

        if r != 0:
            print(r)
            print(info)
            plot_end_result(plot_price, plot_action)
            return info['achievement']


    for i in range(10000):
        with torch.no_grad():
            _, action, _ = model(state.unsqueeze(1))

        state, reward, done, info = env_evaluate.step(action.item())
        state = torch.from_numpy(state).float().to(device)

        if render:
            env_evaluate.render()

        if done:
            rewards.append(info['reward'])
            achievements.append(info['achievement'])
            break

