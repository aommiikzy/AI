
import numpy as np
A = [[4, 1, 2], [3, 0, 5], [6, 7, 8]]
CopyA = [[4, 1, 2], [3, 0, 5], [6, 7, 8]]
goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]
B =[[1, 2, 3], [4, 5, 6], [7, 8, 0]]
elt = CopyA[2]
print(elt[0])
# for row in A:
#     for elem in row:
#         print(elem, end=' ')
#     print()
# for inner_l in A:
# #     for item in inner_l:
# #         print(item)
# x = A[0].split()
# print(A[1][1])
# check = True
# print(check)
# for x in range(3):
#     for y in range(3):
#         if A[x][y] != goal_board[x][y]:
#             check = False
#             break
#     else:
#         continue
#     break
# print(check)
        # print(A[x][y])
sumorigin = 0
spaceX = 9
spaceY = 0
position = 0
for x in range(3):
    for y in range(3):
        if A[x][y] == 0:
            spaceX = x
            spaceY = y
        if A[x][y] == goal_board[x][y]:
           sumorigin = sumorigin+1
        position+=1
print("Sum origin = ",sumorigin)
# print(space)

sum = 0
# if space == 0:
#     A[0][1] = CopyA[0][0]
#     A[0][0] = CopyA[0][1]
#     for x in range(3):
#         for y in range(3):
#             if A[x][y] == goal_board[x][y]:
#                 sum = sum + 1
#     if sum > sumorigin:
#         path = 'l'
#     else:
#         A[0][0] = CopyA[1][0]
#         A[1][0] = CopyA[0][0]
#         path = 'u'
# space = 11
# if space == 4:
# A[spaceX][spaceY] = CopyA[spaceX-1][spaceY]
# A[spaceX-1][spaceY] = CopyA[spaceX][spaceY]
#     down

# A[spaceX+1][spaceY] = CopyA[spaceX][spaceY]
# A[spaceX][spaceY] = CopyA[spaceX+1][spaceY]
#     up

# A[spaceX][spaceY-1] = CopyA[spaceX][spaceY]
# A[spaceX][spaceY] = CopyA[spaceX][spaceY-1]
    #     right

# A[spaceX][spaceY+1] = CopyA[spaceX][spaceY]
# A[spaceX][spaceY] = CopyA[spaceX][spaceX+1]
    #     left


for row in A:
    for elem in row:
        print(elem, end=' ')
    print()
print("Sum New = ",sum)
path = ['u', 'd', 'l', 'r']
print("path New = ",path[1])







