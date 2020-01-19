"""A module for homework 1."""
import random
import copy
from platform import node

space =0
spaceX = 9
spaceY = 0
M = [[4, 1, 2], [3, 0, 5], [6, 7, 8]]
goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]
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
        # print(output)
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
        List[List[int]]
            A nested list containing integers representing numbers on a board
            e.g., [[0, 1, 2], [3, 4, 5], [6, 7, 8]] where 0 is a blank tile.
        """

        # list.insert(M,1,0)
        # list.insert(M, 2, 0)
        #
        # for x in range(len(list.size)):
        #     print(list[x])
            # n1 = node(1, state1)
        # self = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # # TODO: 1
        # rand_ls = list(random.sample(range(9), 9))
        # c = 0
        # tiles = []
        # for i in range(3):
        #     temp_tiles = []
        #     for j in range(3):
        #         temp_tiles.append(rand_ls[c])
        #         c += 1
        #     tiles.append(temp_tiles)



        M = [[4, 1, 2], [3, 0, 5], [6, 7, 8]]
        sumorigin = 0
        spaceX = 9
        spaceY = 0
        position = 0
        for x in range(3):
            for y in range(3):
                if M[x][y] == 0:
                    spaceX = x
                    spaceY = y
                    # print("spaceX = ",spaceX)
                    # print("spaceY = ",spaceY)
                if M[x][y] == goal_board[x][y]:
                    sumorigin = sumorigin + 1
                position += 1
    #     for row in M:
    #         for elem in row:
    #         #     print(elem, end=' ')
    #         # print()
    # # return list
            temp_state = EightPuzzleState(board=M)
        return temp_state
    #     # pass



    def successor(self, action):
        if action not in self.action_space:
            raise ValueError(f'`action`: {action} is not valid.')
        # TODO: 2
        # YOU NEED TO COPY A BOARD BEFORE MODIFYING IT
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == 0:
                    spaceX = x
                    spaceY = y

        new_board = copy.deepcopy(self.board)
        if action == 'd':
            print("Hi from d ")
            # print("spaceX = ", spaceX)
            # print("spaceY = ", spaceY)
            self.board[spaceX][spaceY] = new_board[spaceX-1][spaceY]
            self.board[spaceX-1][spaceY] = new_board[spaceX][spaceY]
            # output = []
            # for row in new_board:
            #     row_string = ' | '.join([str(e) for e in row])
            #     output.append(row_string)
            # print(output)
            return self.board
        if action == 'u':
            self.board[spaceX+1][spaceY] = new_board[spaceX][spaceY]
            self.board[spaceX][spaceY] = new_board[spaceX+1][spaceY]
            return new_board
        if action == 'r':
            self.board[spaceX][spaceY-1] = new_board[spaceX][spaceY]
            self.board[spaceX][spaceY] = new_board[spaceX][spaceY-1]
            return  self.board
        if action == 'l':
            self.board[spaceX][spaceY+1] = M[spaceX][spaceY]
            self.board[spaceX][spaceY] = M[spaceX][spaceX+1]
            return self.board

        # return new_board
        pass


    def is_goal(self, goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]):

        check = (self.board == goal_board)
        return check
        # TODO: 3
        # for x in range(3):
        #     for y in range(3):
        #         if self.board[x][y] != goal_board[x][y]:
        #             check = False
        #             break
        #     else:
        #         continue
        #     break
        # return check
        # print("Hello from goal",self.board[0][0])
        # for x in range(3):
        #     for y in range(3):
        #         print(self.board[x][y])
        # return False



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

        self.list = []
        # print(self.state)
        # print(self.parent)
        # print(self.action)
        # print(self.cost)
        # print(self.path_cost)

    def trace(self):
        """

        Return a path from the root to this node.

        Return
        ----------
        List[EightPuzzleNode]
            A list of nodes stating from the root node to the current node.

        """

        # TODO: 4
        # sumOriginal = 0
        # path = ['u', 'd', 'l', 'r']
        # for x in range(3):
        #     for y in range(3):
        #         if self.bor[x][y] == goal_board[x][y]:
        #             sumOriginal = sumOriginal + 1
        # print("sumOriginal",sumOriginal)
        # i = 0
        # sum = 0
        # while i > 4:
        #     # EightPuzzleState.successor(M , path[i])
        #     for x in range(3):
        #         for y in range(3):
        #             if self.[x][y] == goal_board[x][y]:
        #                 sum = sum + 1
        #     if sum > sumOriginal:
        #         return path[i]
        #         break
        #     i += 1
        # pass
        if self.parent == None:
            temp = []
            temp2 = copy.deepcopy(self)
            temp.append(temp2)
            return temp
        elif self.parent != None:
            temp = self.parent.trace()
            temp2 = copy.deepcopy(self)
            temp.append(temp2)
            return temp


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
        print("Finist")
        # if new_state.is_goal():
        #     print('Congratuations!')
        #     break
        action = input('Please enter the next move (q to quit): ')
    print('Your actions are: ')
    # for node in cur_node.trace():
    #     print(f'  - {node.action}')
    # print(f'The total path cost is {cur_node.path_cost}')


if __name__ == '__main__':
    test_by_hand()
