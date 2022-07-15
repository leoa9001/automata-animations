import random
import sys
import os

#We'll have two double arrays, board0, board1, and parity tells us which is the current array and the other array is for 

class cellBoard:
	w,h = None, None
	board, boardAux = None, None
	windowAux = None

	def __init__(self, width, height):
		self.w = width
		self.h = height
		self.parity = 0
		self.board = [[0 for x in range(width)] for y in range(height)] 
		self.boardAux = [[-1 for x in range(width)] for y in range(height)] 
		self.windowAux = [[-1 for x in range(3)] for y in range (3)]

	def printBoard(self):
		print("---")
		for x in self.board:
			print(x)
		print("---")

	def setRandom(self):
		for i in range(self.w):
			for j in range(self.h):
				self.board[i][j] = random.randint(0,1)

	#updates board. ends with shifting parity. 
	def updateBoard(self):
		#loop here and update by taking input from board and applying to boardAux
		for i in range(self.w):
			for j in range(self.h):
				self.boardAux[i][j] = self.updateCell(i,j)

		#Swap board data post update.
		t = self.board
		self.board = self.boardAux
		self.boardAux = t

	def updatePrint(self):
		self.updateBoard()
		self.printBoard()

	#updates the cell at i,j by populating windowAux and passing to an update function. Also been tested!
	def updateCell(self,i,j):
		if(self.board[i][j]==-1):
			return -1
		window = self.windowAux
		for k in range(3):
			for l in range(3):
				a = i+k-1
				b = j+l-1
				if(a < 0 or b < 0 or a >= self.w or b >= self.h):
					window[k][l] = -1
				else:
					window[k][l] = self.board[a][b]

		return self.conwayUpdate(window)


	#3x3 double-array of integers required. Outputs what the middle cell turns into after a Conway Game of Life update. 
	#This has been checked in a rudimentary fashion. 
	def conwayUpdate(self, window):
		cc = window[1][1]
		if(cc==-1):
			return -1
		nCount = 0
		for i in range(3):
			for j in range(3):
				if(window[i][j]==1 and (i!= 1 or j != 1)):
					nCount+= 1
		if(cc==0 and nCount==3):
			return 1
		elif (cc==1 and nCount>=2 and nCount <=3):
			return 1
		else:
			return 0
		return -1









cb = cellBoard(5,5)
cb.setRandom()
cb.printBoard()
cb.updateBoard()
cb.printBoard()
cb.updateBoard()
cb.printBoard()
cb.updateBoard()
cb.printBoard()
cb.updateBoard()





# #returns the value on an update at square i,j in the board according to Conway's game of life rule
# def golUpdate(i,j):
# 	nCount =

# printBoard()



