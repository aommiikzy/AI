"""A module for homework 1."""
# noqa: D413

import abc
import heapq
from collections import defaultdict

from hw1 import EightPuzzleState, EightPuzzleNode

rount = 0
def eightPuzzleH1(state):
    A = heapq
    # A.
    position = 0
    """
    Return the number of misplaced tiles including blank.

    Parameters
    ----------
    state: EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])

    Returns
    ----------
    int

    """
    # round = 0
    # position = 0
    # j=0
    # i=0
    # find = 0
    # print(state.board)
    # for i in range(3):
    #     for j in range(3):
    #         print("round= ",round)
    #         print(state.board[i][j])
    #
    #         if(state.board[i][j]==0):
    #             print("HERE = ",i,j)
    #             position = i+j
    #             find = 1
    #             print("Find 0 at ",round)
    #             break
    #         round += 1
    #     # if(find > 0):
    #     #     break
    #     # else
    #     else:
    #         continue
    #     break
    #
    #
    #
    # print("position of 0 = ",round)
    round = 0
    # TODO 1:
    state.goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    # state.board = [[1, 2, 0], [4, 5, 6], [7, 8, 0]]
    for i in range(3):
        for j in range(3):
                    if state.board[i][j] != state.goal[i][j]:
                        round +=1
                        # print(h)

    print(round)
    return round


def eightPuzzleH2(state):
    """
    Return the total Manhattan distance from goal position of all tiles.

    Parameters
    ----------
    state: EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])

    Returns
    ----------
    int

    """
    # TODO 2:
    # print(state.board)
    h = 0
    answer = 0
    term1 = 0
    term2 = 0
    state.goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    for i in range(3):
        for j in range(3):
            for x in range(3):
                for y in range(3):
                    if state.board[i][j] == state.goal[x][y] and state.board[i][j] != 0:
                        h = abs(i-x)+abs(y-j)
                        answer = answer+h
                        # print(h)
                        break
                else:
                    continue
                break
        # print(state.board[i][j])
    print("To minimal s as much as psb = ",answer)
    return answer


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
        print("Create frontier")

    def is_empty(self):
        """Return True if empty."""
        print("Test")
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
        print("Add")
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

    def __init__(self, h_func):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state)
            a heuristic function to score a state.


        """
        self.fontier = []
        self.h = h_func
        # TODO: 3
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.


    
class AStarFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state)
            a heuristic function to score a state.


        """
        self.h = h_func

        # TODO: 4
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.
        print(self.h)



def _parity(board):
    """Return parity of a square matrix."""
    inversions = 0
    nums = []
    for row in board:
        for value in row:
            # print(value)
            nums.append(value)
    for i in range(len(nums)):
        for j in range(i+ 1, len(nums)):
            if nums[i] != 0 and nums[j] != 0 and nums[i] > nums[j]:
                inversions += 1
    # print(inversions)
    return inversions % 2


def _is_reachable(board1, board2):
    """Return True if two N-Puzzle state are reachable to each other."""
    # print("Board1 = ",board1)
    # print("Board2 = ", board2)
    # print(_parity(board1),_parity(board2))
    # print(_parity(board1) == _parity(board2))
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
    root_node = EightPuzzleNode(init_state, action='INIT')
    num_nodes += 1

    # TODO: 5

    return solution, num_nodes


def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

    init_state = EightPuzzleState.initializeState()
    # eightPuzzleH2(init_state)
    init_state = EightPuzzleState([[7, 2, 4], [5, 0, 6], [8, 3, 1]])

    # print(goal_state.board)
    # print(init_state.board)
    rount = 0
    while not _is_reachable(goal_state.board, init_state.board):

        # print("Hi")
        # round += 1
        init_state = EightPuzzleState.initializeState()
        # print("hi")
    # if _is_reachable(goal_state.board, init_state.board) == True:
    #     print("yes")
    frontier = DFSFrontier()  # Change this to your own implementation.
    print(frontier.add(1))




    # eightPuzzleH1(init_state)
    if verbose:
        print(init_state)
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    if verbose:
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose:
        for action in plan:
            print("T")
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
        print(f'{d}, {len(n)}, {sum(n)/len(n)}')

if __name__ == '__main__':
    __, __ = test_by_hand()
    # experiment()  #  run graph search 10000 times and report result.