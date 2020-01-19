"""A module for homework 2. Version 11."""
# noqa: D413

import abc
import copy
import heapq
import itertools
from collections import defaultdict
from queue import PriorityQueue

from hw1 import EightPuzzleState, EightPuzzleNode
REMOVED = '<removed-task>'

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
                if state.board[i][j] != goal_state.board[i][j]:
                    round += 1

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
                    if state.board[i][j] == goal_state.board[x][y]:
                        h = abs(i - x) + abs(y - j)
                        answer = answer + h
                        # print(h)
                        break
                else:
                    continue
                break
        # print(state.board[i][j])
    # print("To minimal s as much as psb = ", answer)
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
        # TODO: 3
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.

        self.lists = []  # list of entries arranged in a heap
        self.count = itertools.count()
        self.task = {}

    def add(self, node):
        'This function will add new task and update in the priority'

        if node in self.task:
            'if has this node already, it will delete this node'
            self.remove_node(node)
        count = next(self.count)
        entrance = [self.h(node.state, self.goal), count, node]
        self.task[node] = entrance
        'This function use heap to sort the node in list in frontier'
        heapq.heappush(self.lists, entrance)

    def remove_node(self, node):

        'Use pop to remove node in frontier'
        entry = self.task.pop(node)

        entry[-1] = REMOVED

    def next(self):
        'Return the node which has the lowest priority task'
        checkDuplicate = True
        'Use for check this node has in frontier or not'
        'If not in frontier set checkDuplicate equal false it will add this node  '
        while self.lists:
            count, priority, node = heapq.heappop(self.lists)

            if node is not REMOVED:
                checkDuplicate = False

            if checkDuplicate == False:
                del self.task[node]
                return node

        raise KeyError('Cannot pop from an empty priority queue')


    def is_empty(self):
        'If the length is equal 0 it means empty'
        return len(self.lists) == 0




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
        self.list = []  # list of entries arranged in a heap
        self.count = itertools.count()
        self.task = {}

    def is_empty(self):
        'If the length is equal 0 it means empty'
        return len(self.list) == 0

    def add(self, node):
        if node in self.task:
            self.remove_node(node)
        count = next(self.count)
        entrance = [node.path_cost +     self.h(node.state, self.goal) , count, node]
        self.task[node] = entrance
        heapq.heappush(self.list, entrance)

    def remove_node(self, node):
        'This function will remove the node in frontier'
        entry = self.task.pop(node)
        entry[-1] = REMOVED

    def next(self):
        'Return the node which has the lowest priority task'
        checkDuplicate = True
        'Use for check this node has in frontier or not'
        'If not in frontier set checkDuplicate equal false it will add this node  '
        while self.list:
            count, priority, node = heapq.heappop(self.list)

            if node is not REMOVED:
                checkDuplicate = False

            if checkDuplicate == False:
                del self.task[node]
                return node

        raise KeyError('Cannot pop from an empty priority queue')



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


    # =======================================================

    cur_node = root_node
    round = 10
    history = []
    actionList = ['u','d','l','r']
    move = 10
    frontier.add(root_node)
    while not frontier.is_empty():
        # print("  Round   ", round)

        for i in range(4):
            new_State = cur_node.state.successor(actionList[i])
            # if new_State != None:
            #     for checkDup in range(len(history)):
            #         if new_State == history[checkDup]:
            #             same += 1

            if new_State != None:
                if new_State.board not in history:
                    move = 0

            if move == 0 and new_State!=None:
                new_node = EightPuzzleNode(new_State, cur_node, actionList[i])
                history.append(new_State.board)
                # solution.append(actionList[i])
                frontier.add(new_node)
                move = 10
        num_nodes += 1
        next = frontier.next()
        if type(next) == tuple:
            next_node = next[2]
        else:
            next_node = next
        print(next_node.state)
        if next_node.state.is_goal(goal_state.board)==True:
            check = 0
        else:
            check = 1
        i=0
        if check == 0:

            list = next_node.trace()
            for i in list:
                if i.action != "INIT":
                    solution.append(i.action)
            break
        cur_node = next_node
        round += 1
    return solution, num_nodes


def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()
    while not _is_reachable(goal_state.board, init_state.board):
        init_state = EightPuzzleState.initializeState()

    # frontier = GreedyFrontier(eightPuzzleH2,goal_state)  # Change this to your own implementation.
    frontier = AStarFrontier(eightPuzzleH2,goal_state)
    # frontier = DFSFrontier()
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

    # experiment()
    return len(plan), num_nodes


def experiment(n=10000):
    """Run experiments and report number of nodes generated."""
    result = defaultdict(list)
    for __ in range(n):
        d, n = test_by_hand(False)
        result[d].append(n)
    max_d = max(result.keys())
    for d in range(max_d + 1):
        n = result[d]
        if len(n) == 0:
            continue
        print(f'{d}, {len(n)}, {sum(n) / len(n)}')


if __name__ == '__main__':
    __, __ = test_by_hand()
    # experiment(1000)  #  run graph search 10000 times and report result.
