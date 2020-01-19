"""A module for homework 2. Version 3."""
# noqa: D413

import abc
import copy
import time
from collections import defaultdict

from hw1 import EightPuzzleState, EightPuzzleNode


def eightPuzzleH1(state, goal_state):
    """
    Return the number of misplaced tiles including blank.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """

    # TODO 1:
    round = 0
    if state.board == None:
        print("Cannot move")
        return 100
    else:
        for i in range(3):
            for j in range(3):
                if state.board[i][j] != goal_state.board[i][j] and state.board[i][j] != 0:
                    round += 1
                # print(h)
        print("Different = ", round)
    # print("Hash = ",hash(round))
        return round
    # pass


def eightPuzzleH2(state, goal_state):
    """
    Return the total Manhattan distance from goal position of all tiles.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    # TODO 2:
    h = 0
    # print(state.board)
    answer = 0
    term1 = 0
    term2 = 0
    for i in range(3):
        for j in range(3):
            for x in range(3):
                for y in range(3):
                    if state.board[i][j] == goal_state.board[x][y] and state.board[i][j] != 0:
                        h = abs(i - x) + abs(y - j)
                        answer = answer + h
                        # print(h)
                        break
                else:
                    continue
                break
        # print(state.board[i][j])
    print("To minimal s as much as psb = ", answer)
    return answer
    pass


class Frontier(abc.ABC):
    """An abstract class of a frontier."""

    def __init__(self):
        """Create a frontier."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self):
        """Return True if empty."""
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        raise NotImplementedError()


class DFSFrontier(Frontier):
    """An example of how to implement a depth-first frontier (stack)."""

    def __init__(self):
        """Create a frontier."""
        self.stack = []

    def is_empty(self):
        """Return True if empty."""
        return len(self.stack) == 0

    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        for n in self.stack:
            # HERE we assume that state implements __eq__() function.
            # This could be improve further by using a set() datastructure,
            # by implementing __hash__() function.
            if n.state == node.state:
                return None
        self.stack.append(node)

    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        return self.stack.pop()



class GreedyFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state, goal_state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()

        """
        self.h = h_func
        self.goal = goal_state
        self.stack = []
        print("test")
        # TODO: 3
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.
        self.add(self.h)
        print(self.stack[0])


    # def __init__(self):
    #     """Create a frontier."""
    #     self.stack = []

    def is_empty(self):
        """Return True if empty."""
        return len(self.stack) == 0

    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        print("Add already")
        for n in self.stack:
            # HERE we assume that state implements __eq__() function.
            # This could be improve further by using a set() datastructure,
            # by implementing __hash__() function.
            # __eq__()
            # set()
            # __hash__()
            if n.state == node.state:
                return None
        self.stack.append(node)

    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        return self.stack.pop()




class AStarFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()


        """
        self.h = h_func
        self.goal = goal_state
        # TODO: 4
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.



def _parity(board):
    """Return parity of a square matrix."""
    inversions = 0
    nums = []
    for row in board:
        for value in row:
            nums.append(value)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] != 0 and nums[j] != 0 and nums[i] > nums[j]:
                inversions += 1
    return inversions % 2


def _is_reachable(board1, board2):
    """Return True if two N-Puzzle state are reachable to each other."""
    return _parity(board1) == _parity(board2)


def graph_search(init_state, goal_state, frontier):
    if not _is_reachable(init_state.board, goal_state.board):
        return None, 0
    if init_state.is_goal(goal_state.board):
        return [], 0
    num_nodes = 0
    solution = []
    # Perform graph search
    root_node = EightPuzzleNode(init_state, action='INIT')
    # print(root_node.state)
    num_nodes += 1

    # if current_node.state.x != 2:
    #     acts.add('u')
    # if current_node.state.x != 0:
    #     acts.add('d')
    # if current_node.state.y != 2:
    #     acts.add('l')
    # if current_node.state.y != 0:
    #     acts.add('r')
    # aom tryyyyyyy=======================================================

    print("test Out")
    cur_node = root_node
    round = 10
    history = []
    trace = [100,100,100,100]
    actionList = ['u','d','l','r']
    same = 0
    while round != 0:
        for i in range(4):
            print("round i = ",round ,i)
            new_State = cur_node.state.successor(actionList[i])
            if new_State != None and new_State.board not in history:
                same = 0
                print("After Move ",actionList[i])
                new_node = EightPuzzleNode(new_State, cur_node, actionList[i])
                history.append(new_State.board)
                solution.append(actionList[i])
                print(new_State)
                frontier.add(new_node)
                eightPuzzleH1(new_State, goal_state)
        num_nodes += 1
        temp = frontier.next()
        print(temp)
        if type(temp) == tuple:
            next_node = temp[2]
        else:
            next_node = temp
            print("Test")

        check = 0
        for i in range(3):
            for j in range(3):
                if next_node.state.board[i][j] != goal_state.board[i][j]:
                    check = check+1
        i=0
        if check==0:
            temp = next_node.trace()
            for i in range(len(temp)):
                if i != 0:
                    solution.append(temp[i].action)
            break
        else:
            print("\033[H\033[J")
            cur_node = next_node
    return solution, num_nodes


def test_by_hand(verbose=True):
    """Run a graph-search."""
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()
    while not _is_reachable(goal_state.board, init_state.board):
        init_state = EightPuzzleState.initializeState()

    # frontier = GreedyFrontier(eightPuzzleH2,goal_state)  # Change this to your own implementation.
    # frontier = AStarFrontier(eightPuzzleH2,goal_state)
    frontier = DFSFrontier()
    # frontier.stack
    if verbose:
        print(init_state)
    # print(frontier.stack.board[0])
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    # print(frontier.stack[0])
    if verbose:
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose:
        for action in plan:
            print(f'- {action}')
            pass

    # ========================== checking solution paet ==========================
    # print(init_state)
    cur_node = EightPuzzleNode(init_state, action="INIT")
    for i in plan:
        new_state = cur_node.state.successor(i)
        cur_node = EightPuzzleNode(new_state, cur_node, i)
        # print(new_state)
        # print()
        if new_state.is_goal():
            # print('Congratuations!')
            break

    return len(plan), num_nodes


def experiment(n=10000):
    """Run experiments and report number of nodes generated."""
    result = defaultdict(list)
    for __ in range(n):
        d, n = test_by_hand(False)
        result[d].append(n)
    max_d = max(result.keys())
    for i in range(max_d + 1):
        n = result[d]
        if len(n) == 0:
            continue
        print(f'{d}, {len(n)}, {sum(n) / len(n)}')


if __name__ == '__main__':
    __, __ = test_by_hand()
    # experiment()  #  run graph search 10000 times and report result.
