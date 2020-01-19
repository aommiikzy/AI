"""
This module contains agents that play reversi.

Version 3.1
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2
import copy
from copy import deepcopy
import itertools
import sys


import math



_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.

        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)
            p = Process(
                target=self.search,
                args=(
                    self._color, board, valid_actions,
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
                self._move = np.array(
                [output_move_row.value, output_move_column.value], dtype = np.int32
                )
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array([output_move_row.value, output_move_column.value], dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.

        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for
            `output_move_row.value` and `output_move_column.value`
            as a way to return.
        """

        raise NotImplementedError('You will have to implement this.')

class FangAgent(ReversiAgent):

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        # super().__init__(color)
        super().__init__(color)
        self._move = None
        self._color = color

        weight = [
            120, -20,  20,   5,   5,  20, -20, 120,
            -20, -40,  -5,  -5,  -5,  -5, -40, -20,
             20,  -5,  15,   3,   3,  15,  -5,  20,
              5,  -5,   3,   3,   3,   3,  -5,   5,
              5,  -5,   3,   3,   3,   3,  -5,   5,
             20,  -5,  15,   3,   3,  15,  -5,  20,
            -20, -40,  -5,  -5,  -5,  -5, -40, -20,
            120, -20,  20,   5,   5,  20, -20, 120,
        ]

        self.evaluate = np.array(weight).reshape(8,8)

    def __index__(self):
        super(FangAgent, self)

    def search(
        self, color, board, valid_actions,
        output_move_row, output_move_column):

        #time.sleep(3)

        try:
            if self._color == 1:
                r, c = self.maximum_value(board, valid_actions, 4, 0, -10000, 10000, True)
            else:
                r, c = self.maximum_value(board, valid_actions, 2, 0, -5000, 5000, True)

            if c is None or valid_actions is None:

                #time.sleep(1)
                print("cannot solve this ja")

                #time.sleep(3)
                randidx = random.randint(0, len(self.evaluate) - 8)
                random_action = valid_actions[randidx]
                output_move_row.value = random_action[0]
                output_move_column.value = random_action[1]
                #time.sleep(3)

            elif c is not None:
                #time.sleep(2)
                output_move_row.value = c[0]
                output_move_column.value = c[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


    def maximum_value(self, board: np.array, valid_actions: np.array, depth, level, alpha, beta, gain):

        if depth == 0:

            # do evaluation

            count_B = 0
            count_W = 0

            copy_weight = copy.deepcopy(self.evaluate)
            # pair two values in i index of array together
            to_evaluate = np.array(list(zip(*board.nonzero())))

            for r in to_evaluate:
                x,y = r[0], r[1]
                if board[r[0]][r[1]] != self._color:
                    count_W += (copy_weight[r[0]][r[1]])
                else:
                    count_B += (copy_weight[r[0]][r[1]])
            return count_B - count_W

        best_move_board = None
        max_alpha_value = alpha
        player = self._color
        MAX = -100000
        MIN = 100000

        for take_turn in valid_actions:
            #print("try" , take_turn)
            r,c = self.create_board(board, take_turn, player)
            # r,c = transition(board, take_turn, player)

            new_turn = self.minimum_value(r, c, depth - 1, level + 1, max_alpha_value, beta, not gain)

            if MAX < new_turn:
                MAX = new_turn

                if level == 0:
                    best_move_board = take_turn

            max_alpha_value = max(max_alpha_value, MAX)

            if beta <= max_alpha_value:
                break

        if level != 0:
            return MAX
        else:
            return MAX, best_move_board

    def minimum_value(self, board: np.array, valid_actions: np.array, depth, level, alpha, beta, gain):

        if depth == 0:

            count_B = 0
            count_W = 0

            copy_weight = copy.deepcopy(self.evaluate)
            # pair two values in i index of array together
            to_evaluate = np.array(list(zip(*board.nonzero())))

            for r in to_evaluate:
                x,y = r[0], r[1]
                if board[r[0]][r[1]] != self._color:
                    count_W += (copy_weight[r[0]][r[1]])
                else:
                    count_B += (copy_weight[r[0]][r[1]])
            return count_B - count_W

        best_move_board = None
        min_beta_value = beta
        player = self._color
        MAX = -100000
        MIN = 100000

        for take_turn in valid_actions:
            # print("Try -> ", take_turn)
            r,c = self.create_board(board, take_turn, player)
            # r,c = transition(board, take_turn, player)

            new_turn = self.maximum_value(r, c, depth - 1, level + 1, alpha, min_beta_value, not gain)

            if MIN > new_turn:
                MIN = new_turn

                if level == 0:
                    best_move_board = take_turn

            min_beta_value = min(min_beta_value, new_turn)

            if min_beta_value <= alpha:
                break

        if level != 0:
            return MIN
        else:
            return MIN, best_move_board


    @staticmethod
    def opponent(player):
        if player == 1:
            return -1
        else:
            return 1

    def create_board(self, board, take_turn, player):
        new_board = transition(board, player, take_turn)
        check_valid = _ENV.get_valid((new_board, self._color))
        check_valid = np.array(list(zip(*check_valid.nonzero())))

        return new_board, check_valid

class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
class Aom(ReversiAgent):
    def __index__(self):
        super(self.minimax, self)

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            # 99999 is the maximum number of state
            max_state = 99999
            # -99999 is minimum number of state
            min_state = -99999
            check = 0
            evaluation, best_state = self.move_max(board, valid_actions, 4, 0, min_state, max_state, True)
            # this condition will check that best state is none or valid_actions equal null will random the action
            if best_state is None or valid_actions is None:
                # random the action
                Action_random = valid_actions[0]
                MoveForColumn = Action_random[1]
                MoveForRow = Action_random[0]
                output_move_column.value = MoveForColumn
                output_move_row.value = MoveForRow
                print(" Aommiizky Selected:" + str(best_state))
                check = 1
            # this loop for valid_actions is not null because it
            if check == 0:
                if best_state is not None:
                    MoveForColumn = best_state[1]
                    MoveForRow = best_state[0]
                    output_move_column.value = MoveForColumn
                    output_move_row.value = MoveForRow
                    print(" Aommiizky Selected:" + str(best_state))
            # if best_state is None:
            #     time.sleep(30)
            # output_move_row.value = best_state[0]
            # output_move_column.value = best_state[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


    # this function will find the max value
    def move_max(self, board: np.array, valid_step, depth, level, alpha, beta,gain: bool):

        if depth == 0:
            return self.evaluate(board)
        min_state = -99999

        alpha_max: int = alpha
        eval_max = min_state
        player: int = self._color
        best_step: np.array = None

        for move in valid_step:
            recent_board, next_step = self.generate_State(board, move, player)
            # set the next move equal the next step by using the move_min function
            next_move = self.move_min(recent_board, next_step, depth - 1, level + 1, alpha_max, beta, not gain)
            # if the max evaluation value is less than the next step or next move will do in here.
            if eval_max < next_move:
                eval_max = next_move
                if level == 0:
                    best_step = move
            # it will set the new value of alpha_max
            alpha_max = max(alpha_max, eval_max)
            # if the Beta less than or equal the value of alpha_max
            if beta < alpha_max or beta == alpha_max:
                break

        # if the level is not equal 0 will return the min evaluation
        if level != 0:
            return eval_max
        else:
            # self.move() = best_state
            # if the level is equal 0 will return the min evaluation  and the best step or best state
            return self.evaluate(board), best_step

    # this function will find the min value
    def move_min(self, board: np.array, valid_step: np.array, depth, level, alpha, beta, gain: bool):
        if depth == 0:
            return self.evaluate(board)
        best_step: np.array = None
        max_state = 99999
        beta_min: int = beta
        min_eval = max_state
        player: int = self.create_rival(self._color)

        for move in valid_step:
            recent_board, next_step = self.generate_State(board, move, player)
            # set the next move equal the next step by using the move_max function
            next_move = self.move_max(recent_board, next_step, depth - 1, level + 1, alpha, beta_min, not gain)
            # if the min evaluation value is more than the next step or next move will do in here.
            if min_eval > next_move:
                min_eval = next_move
                if level == 0:
                    best_step = move
            # it will set the new value of beta
            beta_min = min(beta_min, min_eval)
            if beta_min < alpha or beta_min == alpha:
                break
        # if the level is not equal 0 will return the min evaluation
        if level != 0:
            return min_eval
        # if the level is equal 0 will return the min evaluation  and the best step or best state
        else:
            return self.evaluate(board), best_step

    def evaluate(self, board: np.array):
        countX: int = 0
        countY: int = 0
        eval_board = np.array(list(zip(*board.nonzero())))
        check = 0
        # this while loop use for check that if it runs all of the value in eval_board
        # it will break to count the score
        while check is 0:
            for i in eval_board:
                # the value i is the matrix 2 x2  or 2 dimension
                # it means that it contains of position x and y
                # so we set to positionY and positionX for easy to understand
                positionY = i[0]
                positionX = i[1]

                if board[positionY][positionX] == self._color:
                    countX += 1
                else:
                    countY += 1
            check = check + 1
        FinalScore = countX - countY
        # return the finalscore
        return FinalScore

    @staticmethod
    def create_rival(player: int):
        if player == 1:
            return -1
        else:
            return 1

    # this function will create the state
    def generate_State(self, board: np.array, action: np.array, player: int) -> (np.array, np.array):
        new_State: np.array = transition(board, player, action)
        correct_move = _ENV.get_valid((new_State, self.create_rival(player)))
        correct_move = np.array(list(zip(*correct_move.nonzero())))
        # it will return the new state and the correct that to move
        return new_State, correct_move
# ==================
# Waris     Vorathumdusadee 6088128 Section 3
# Sirichoke Yooyen          6088232 Section 3
# Phummarat Yosamornsuntorn 6088233 Section 3

class NuttyAgent(ReversiAgent):
    def __index__(self):
        super(NuttyAgent, self)

    def search(
            self, color, board, valid_actions, output_move_row,
            output_move_column):
        try:
            evaluation, Opt_state = self.MaxFnd(board, valid_actions, 4, 0, -1900, 1900)
            output_move_row.value = Opt_state[0]
            output_move_column.value = Opt_state[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def MaxFnd(self, board, validactions, depth, level, alpha, beta):

        if (depth == 0):
            return self.evaluation(board)

        OptMove = None
        MaxAlpha = alpha
        MaxEval = -1900
        player = self._color

        for Moves in validactions:
            NewBoard, NewAct = self.createState(board, Moves, player)
            NewMoves = self.MinFnd(NewBoard, NewAct, depth - 1, level + 1, MaxAlpha, beta)

            if (MaxEval < NewMoves):
                MaxEval = NewMoves

                if (level == 0):
                    OptMove = Moves

            MaxAlpha = max(MaxAlpha, MaxEval)

            if (MaxAlpha >= beta):
                break

        if (level != 0):
            return MaxEval
        else:
            return (MaxEval, OptMove)

    def MinFnd(self, board, validactions, depth, level, alpha, beta):

        if (depth == 0):
            return (self.evaluation(board))

        OptMove = None
        MinBeta = beta
        MinEval = 1900
        player: int = self.getOpponent(self._color)

        for Moves in validactions:
            NewBoard, NewAct = self.createState(board, Moves, player)
            NewMoves = self.MaxFnd(NewBoard, NewAct, depth - 1, level + 1, alpha, MinBeta)

            if (MinEval > NewMoves):
                MinEval = NewMoves

                if (level == 0):
                    OptMove = Moves

            MinBeta = min(MinBeta, MinEval)

            if (alpha >= MinBeta):
                break

        if (level != 0):
            return MinEval
        else:
            return (MinEval, OptMove)

    def evaluation(self, board):

        countA: int = 0
        countB: int = 0
        evaluationBoard = np.array(list(zip(*board.nonzero())))

        for i in evaluationBoard:
            if (board[i[0]][i[1]] == self._color):
                countA += 1
            else:
                countB += 1

        return (countA - countB)

    @staticmethod
    def getOpponent(player):

        if (player == 1):
            return (-1)
        else:
            return (1)

    def createState(self, board, action, player):

        newState = transition(board, player, action)
        validMoves = _ENV.get_valid((newState, self.getOpponent(player)))
        validMoves = np.array(list(zip(*validMoves.nonzero())))

        return (newState, validMoves)


class PPAgent(ReversiAgent):
    def __index__(self):
        super(self.alphabeta, self)

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            best_move = self.alphabeta(self._color, board, -10, 10, 4, self.evaluate(board), valid_actions)
            output_move_column.value = best_move[1]
            output_move_row.value = best_move[0]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def alphabeta(self, player, board, alpha, beta, depth, evaluate, validactions):
        if depth == 0:
            return self._move

        def value(board2, alpha, beta):
            return -self.alphabeta(self.getOpponent(player), board2, -beta, -alpha, depth - 1, evaluate, validactions)[
                0]

        self._move = validactions[0]
        for move in validactions:
            if alpha >= beta:
                break
            if board is not None:
                val = value(transition(board, player, move), alpha, beta)
                if val > alpha:
                    alpha = val
                    self._move = move
        return self._move

    def evaluate(self, board: np.array):

        countSelf: int = 0
        countOpponent: int = 0
        evaluationBoard = np.array(list(zip(*board.nonzero())))

        for i in evaluationBoard:
            if (board[i[0]][i[1]] == self._color):
                countSelf += 1
            else:
                countOpponent += 1
        return countSelf - countOpponent

    @staticmethod
    def getOpponent(player):
        if player == 1:
            return -1
        else:
            return 1
