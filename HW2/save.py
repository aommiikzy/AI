"Name: Nattawat Lumtansawan ID: 6088085 SECTION: 3"
"""A module for homework 2. Version 3."""
# noqa: D413

import abc
import itertools
import heapq

from collections import defaultdict

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
    pass
    "set count = 0 for count the misplaced"
    count = 0
    for i in range(3):
        for j in range(3):
            "check the state that is equal the goal or not"
            if state.board[i][j] != goal_state.board[i][j]:
                "if it not then update count"
                count += 1
    # print("Wrong State: ", count)
    return count


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
    pass
    "manhattan mean the distance from goal position"
    manhattan = 0
    "i for row in state"
    for i in range(3):
        "j for column in goal_state"
        for j in range(3):
            "check the state that are not equal to the goal"
            if state.board[i][j] != goal_state.board[i][j]:
                "a for row in goal_state"
                for a in range(3):
                    "b for column in goal_state"
                    for b in range(3):
                        "find where is the goal state"
                        if state.board[i][j] == goal_state.board[a][b]:
                            "calculate the distance and sum the distance"
                            manhattan += abs(i-a)+abs(j-b)
    # print("Manhattan: ", manhattan)
    return manhattan



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

        "From heapq https://docs.python.org/3.7/library/heapq.html"
        self.pq = []  # list of entries arranged in a heap
        self.counter = itertools.count()
        self.entry_finder = {}

    def add(self, node):
        'Add a new task or update the priority of an existing task'
        if node in self.entry_finder:
            self.remove_node(node)
        count = next(self.counter)
        entry = [self.h(node.state, self.goal), count, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.pq, entry)

    def remove_node(self, node):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(node)
        entry[-1] = REMOVED

    def next(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, node = heapq.heappop(self.pq)
            if node is not REMOVED:
                del self.entry_finder[node]
                return node
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        return len(self.pq) == 0

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
        self.pq = []  # list of entries arranged in a heap
        self.counter = itertools.count()
        self.entry_finder = {}

    def is_empty(self):
        return len(self.pq) == 0

    def add(self, node):
        'Add a new task or update the priority of an existing task'
        if node in self.entry_finder:
            self.remove_node(node)
        count = next(self.counter)
        entry = [self.h(node.state, self.goal) + node.path_cost, count, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.pq, entry)

    def remove_node(self, node):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(node)
        entry[-1] = REMOVED

    def next(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, node = heapq.heappop(self.pq)
            if node is not REMOVED:
                del self.entry_finder[node]
                return node
        raise KeyError('pop from an empty priority queue')


def _parity(board):
    """Return parity of a square matrix."""
    inversions = 0
    nums = []
    for row in board:
        for value in row:
            nums.append(value)
    for i in range(len(nums)):
        for j in range(i+ 1, len(nums)):
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
    root_node = EightPuzzleNode(init_state, action='INIT')
    frontier.add(root_node)
    num_nodes += 1

    # TODO: 5
    passed = set()
    while not frontier.is_empty():
        current_node: EightPuzzleNode = frontier.next()
        if current_node.state.is_goal():
            "set new goal"
            goal_node = current_node
            break
        if current_node.state not in passed:
            passed.add(current_node.state)
            num_nodes += 1
            acts = set()
            if current_node.state.y-1 >= 0:
                acts.add('u')
            if current_node.state.y+1 < 3:
                acts.add('d')
            if current_node.state.x-1 >= 0:
                acts.add('l')
            if current_node.state.x+1 < 3:
                acts.add('r')

            for i in acts:
                new_state = current_node.state.successor(i)
                new_node = EightPuzzleNode(new_state, current_node, i)
                frontier.add(new_node)
        print("\n")
        print(current_node.state)
    path = goal_node.trace()
    path.pop(0)
    for node in path:
        solution.append(node.action)


    return solution, num_nodes

def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()
    while not _is_reachable(goal_state.board, init_state.board):
        init_state = EightPuzzleState.initializeState()
    frontier = AStarFrontier(eightPuzzleH2, goal_state)  # Change this to your own implementation.
    if verbose:
        print(init_state)
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    if verbose:
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose:
        for action in plan:
            print(f'- {action}')
# Test for Todo1 and 2
    # eightPuzzleH1(init_state,goal_state)
    # eightPuzzleH2(init_state,goal_state)
# Test
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
        print(f'{d}, {len(n)}, {sum(n)/len(n)}')

if __name__ == '__main__':
    __, __ = test_by_hand()
    # experiment(1000)  #  run graph search 10000 times and report result.

28
