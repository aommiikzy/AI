"""A module for homework 1."""
import random
import copy


class EightPuzzleState:
    """A class for a state of an 8-puzzle game."""

    def __init__(self, board):
        """Create an 8-puzzle state."""
        self.action_space = {'u', 'd', 'l', 'r'}
        self.board = board
        for i, row in enumerate(self.board):
            for j, v in enumerate(row):
                if v == 0:
                    self.y = i
                    self.x = j

    def __repr__(self):
        """Return a string representation of a board."""
        output = []
        for row in self.board:
            row_string = ' | '.join([str(e) for e in row])
            output.append(row_string)
        return ('\n' + '-' * len(row_string) + '\n').join(output)

    def __str__(self):
        """Return a string representation of a board."""
        return self.__repr__()

    @staticmethod
    def initializeState():
        """
        Create an 8-puzzle state with a SHUFFLED tiles.

        Return
        ----------
        EightPuzzleState
            A state that contain an 8-puzzle board with a type of List[List[int]]:
            a nested list containing integers representing numbers on a board
            e.g., [[0, 1, 2], [3, 4, 5], [6, 7, 8]] where 0 is a blank tile.
        """
        # TODO: 1
        rand_ls = list(random.sample(range(9), 9))
        count = 0
        List = []
        InList = []
        size = 0
        while size < 3:
            InList = []
            for j in range(3):
                InList.append(rand_ls[count])
                count = count + 1
            #     print(count)
            List.append(InList)
            size = size + 1
        temp_state = EightPuzzleState(board=List)
        return temp_state

    def successor(self, action):
        """
        Move a blank tile in the current state, and return a new state.

        Parameters
        ----------
        action:  string
            Either 'u', 'd', 'l', or 'r'.

        Return
        ----------
        EightPuzzleState or None
            A resulting 8-puzzle state after performing `action`.
            If the action is not possible, this method will return None.

        Raises
        ----------
        ValueError
            if the `action` is not in the action space

        """
        # TODO: 2
        # YOU NEED TO COPY A BOARD BEFORE MODIFYING IT
        new_board = copy.deepcopy(self.board)
        rount = 0
        for x in range(3):
            for y in range(3):

                if self.board[x][y] == 0:
                    spaceX = x
                    spaceY = y
                    # rount = rount + 1
        print(spaceX,spaceY)
        # print(spaceX)
        # print(spaceY)
        if action not in self.action_space:
            raise ValueError(f'`action`: {action} is not valid.')
            return None
        if action in self.action_space:
            # print("Yeah")
            if action == 'd' and spaceX != 0:
                self.board[spaceX][spaceY] = new_board[spaceX - 1][spaceY]
                self.board[spaceX - 1][spaceY] = new_board[spaceX][spaceY]
                # self.board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
                return self
            if action == 'u'and spaceX != 2:
                self.board[spaceX + 1][spaceY] = new_board[spaceX][spaceY]
                self.board[spaceX][spaceY] = new_board[spaceX + 1][spaceY]
                return self
            if action == 'r'and spaceY != 0:
                self.board[spaceX][spaceY - 1] = new_board[spaceX][spaceY]
                self.board[spaceX][spaceY] = new_board[spaceX][spaceY - 1]
                return self
            if action == 'l'and spaceY != 2:
                self.board[spaceX][spaceY + 1] = new_board[spaceX][spaceY]
                self.board[spaceX][spaceY] = new_board[spaceX][spaceY + 1]
                return self
        return None


    def is_goal(self, goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
        """
        Return True if the current state is a goal state.

        Parameters
        ----------
        goal_board (optional)
            The desired state of 8-puzzle.

        Return
        ----------
        Boolean
            True if the current state is a goal.

        """
        # TODO: 3
        check = True
        for x in range(3):
            for y in range(3):
                if self.board[x][y] != goal_board[x][y]:
                    check = False
                    break
            else:
                continue
            break
        return check





class EightPuzzleNode:
    """A class for a node in a search tree of 8-puzzle state."""

    def __init__(
            self, state, parent=None, action=None, cost=1):
        """Create a node with a state."""
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        if parent is not None:
            self.path_cost = parent.path_cost + self.cost
        else:
            self.path_cost = 0

    def trace(self):
        # TODO: 4
        # print(self.parent)
        TraceBack = []
        if self.parent == None :
            TraceBack.append(copy.deepcopy(self))
            return TraceBack
        elif self.parent != None :
            CurrentTrace = self.parent.trace()
            CurrentTrace.append(copy.deepcopy(self))
            return CurrentTrace

def test_by_hand():
    """Run a CLI 8-puzzle game."""
    state = EightPuzzleState.initializeState()
    root_node = EightPuzzleNode(state, action='INIT')
    cur_node = root_node
    print(cur_node.state)
    action = input('Please enter the next move (q to quit): ')
    while action != 'q':
        new_state = cur_node.state.successor(action)
        cur_node = EightPuzzleNode(new_state, cur_node, action)
        print(new_state)
        if new_state.is_goal():
            print('Congratuations!')
            break
        action = input('Please enter the next move (q to quit): ')
    print('Your actions are: ')
    for node in cur_node.trace():
        print(f'  - {node.action}')
    print(f'The total path cost is {cur_node.path_cost}')


if __name__ == '__main__':
    test_by_hand()
