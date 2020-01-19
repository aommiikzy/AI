"""A module for homework 2. Version 3."""
# noqa: D413

import abc
import copy
import queue
import heapq
from collections import defaultdict

from hw1 import EightPuzzleState, EightPuzzleNode
# hash(set)

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
    # TODO 1:
    # print(goal_state)
    # state.goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    # state.board = [[1, 2, 0], [4, 5, 6], [7, 8, 0]]
    # print(type(state.board) == type(None))
    # if(type(state.board) != type(None)):

    if state == None:
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
    # state.goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
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
        """Create a frontier."""
        # print(set(EightPuzzleState.initializeState()))
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
        self.explore = []

    def add(self, node):
        for n in self.explore:
            # HERE we assume that state implements __eq__() function.
            # This could be improve further by using a set() datastructure,
            # by implementing __hash__() function.
            if n.state == node.state:
                return None
        self.explore.append(node)
    def is_empty(self):
        return self.explore.pop()
    def next(self):
        # TODO: 3
        return self.stack.pop()
        # print(self.goal)
    # def find(self):
    #     actionList = ['u', 'd', 'l', 'r']
    #     print("Original State")
    #     print(self.goal)
    #     cur_node = EightPuzzleNode(self.goal, action='INIT')
    #     new_state = cur_node.state.successor('u')
    #     eightPuzzleH1(self.goal,state_goal)
    #     print(new_state)
    #     print(self.goal)



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
        # self.goal
        Frontier.add(1)


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
    """
    Search for a plan to solve problem.

    Parameters
    ----------
    init_state : EightPuzzleState
        an initial state
    goal_state : EightPuzzleState
        a goal state
    frontier : Frontier
        an implementation of a frontier which dictates the order of exploreation.

    Returns
    ----------
    plan : List[string] or None
        A list of actions to reach the goal, None if the search fails.
        Your plan should NOT include 'INIT'.
    num_nodes: int
        A number of nodes generated in the search.

    """

    if not _is_reachable(init_state.board, goal_state.board):
        return None, 0
    if init_state.is_goal(goal_state.board):
        return [], 0
    num_nodes = 0
    solution = []
    # Perform graph search
    # frontier.__hash__()

    # Temp = copy.deepcopy(init_state)
    # root_node = EightPuzzleNode(init_state, action='INIT')
    # num_nodes += 1
    # new = root_node.state.successor('u')
    # print("After U")
    # print(new)
    # eightPuzzleH1(new, goal_state)
    #
    # print("ORIginal = ",Temp)
    #
    # new_state = root_node.state.successor('d')
    # print("After D")
    # print(new_state)
    # eightPuzzleH1(new_state, goal_state)
    round =1
    i =0
    Temp = [copy.deepcopy(init_state), copy.deepcopy(init_state), copy.deepcopy(init_state),
            copy.deepcopy(init_state)]

    VeryTemp = copy.deepcopy(init_state)
    actionList = ['u', 'd', 'l', 'r']
    trace = [100,100,100,100]
    min = 99
    traceFinal = []
    while min != 0:
    # while round < 10:
        for i in range(4):
            Temp = copy.deepcopy(VeryTemp)
            print("Round = ",round/4)
            # print("Min = ",min)
            print(Temp)

            root_node = EightPuzzleNode(Temp, action='INIT')

            # num_nodes += 1

            new = root_node.state.successor(actionList[i])
            if new is not None:
                print("NONEeeeee++===========")
                print("After ", actionList[i])
                print(new)
                trace[i] = eightPuzzleH1(new, goal_state)+round
                eightPuzzleH1(new, goal_state)
            i += 1

            # round += 1
        print("R = ",round)

        roundAction = 0
        r = 0
        print("Min before check = ",min)
        for r in range (4):
            if trace[r]  <= min:
                min = trace[r]
                roundAction = r
        print(trace)
        print("Choose Action  = ",actionList[roundAction])
        # for r in range(3):
        #     print(trace[r])
        print(min,roundAction)

        Temp = copy.deepcopy(VeryTemp)
        root_node = EightPuzzleNode(Temp, action='INIT')
        num_nodes += 1
        new = root_node.state.successor(actionList[roundAction])
        print("Move ==>",actionList[roundAction])
        print(new)
        traceFinal.append(actionList[roundAction])
        solution.append(actionList[roundAction])
        VeryTemp = copy.deepcopy(new)
        Sto = eightPuzzleH1(new, goal_state)
        r = 0
        if Sto == 0:
            print("==Finish==")
            print(new)
            break
        print("Finish")
        print(Temp)
        print(round)
        trace = [100,100,100,100]
        # if
        round += 1
        # round += 1


    # TODO: 5

    return solution, num_nodes


# def test_by_hand(verbose=True):
#     """Run a graph-search."""
#     goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
#     init_state = EightPuzzleState.initializeState()
#     while not _is_reachable(goal_state.board, init_state.board):
#         init_state = EightPuzzleState.initializeState()
#     frontier = GreedyFrontier(eightPuzzleH1(init_state,goal_state),init_state)  # Change this to your own implementation.
#     # frontier = DFSFrontier()
#
#     # eightPuzzleH1(init_state,goal_state)
#     # eightPuzzleH2(init_state, goal_state)
#     # frontier.find()
#     if verbose:
#         print(init_state)
#     plan, num_nodes = graph_search(init_state, goal_state, frontier)
#     if verbose:
#         print(f'A solution is found after generating {num_nodes} nodes.')
#     if verbose:
#         for action in plan:
#             print(f'- {action}')
#     return len(plan), num_nodes

def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()
    while not _is_reachable(goal_state.board, init_state.board):
        print("Cannot")
        init_state = EightPuzzleState.initializeState()
    print(eightPuzzleH1(init_state, goal_state))
    frontier = DFSFrontier()  # Change this to your own implementation.
    if verbose:
        print(init_state)
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    # print(eightPuzzleH1(init_state,goal_state))
    if verbose:
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose:
        for action in plan:
            print(f'- {action}')
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
