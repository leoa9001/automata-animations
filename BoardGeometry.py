import random
import numpy as np
import exc



#Functionality:
#Initialize a board geometry
#Iterate through cells
#Get a cell's boundary window
#Update a cell's value

#Standard board geometry: a set of 2-d grids
#It is almost state-blind. Only -1 and -2 have any meaning as states passed to this code. Other states have no differentiating handling. 
#-1 corresponds to boundary or outside the grid, -2 generally means an error has happened. -2 shouldn't come up but I'd avoid using it as a state in CA code. 
class BoardGeometry:
	grids = None #note by default, this dictionary has -1st grid with [[-1]] shape. 
	grid_num = None
	stitching = None
	corners = None
	init_data = None

	windowAux = None

	def __init__(self, grid_dims, paste_data):
		self.grids = {-1: [[-1]]}
		self.init_data = (grid_dims,paste_data)
		self.grid_num = len(grid_dims)
		for i in range(len(grid_dims)):
			self.grids[i] = np.zeros((grid_dims[i][0],grid_dims[i][1]))
		self.stitching_from_paste(paste_data)
		self.windowAux = [[0 for x in range(3)] for y in range(3)]


	def setRandom(self, num_states):
		for i in range(self.grid_num):
			grid = self.grids[i]
			for a in range(grid.shape[0]):
				for b in range(grid.shape[1]):
					grid[a][b] = random.randint(0,num_states-1)

	def setGrid(self,i, grid):
		if(i<0 or i>= self.grid_num):
			print("Grid index out of bounds in setGrid")
			return
		d0 = self.grids[i].shape[0]
		d1 = self.grids[i].shape[1]
		if(d0 != grid.shape[0] or d1 != grid.shape[1]):
			print("setGrid shape mismatch")

		self.grids[i] = grid

	def printBoard(self):
		for i in range(self.grid_num):
			print("---Grid "+str(i))
			self.printGrid(self.grids[i])

	def printGrid(self, grid):
		for x in grid:
			print(x)
		print("---")


	def get_isometric(self):
		return BoardGeometry(self.init_data[0],self.init_data[1])

	def getWindow(self,grid_ind,i,j):
		window = self.windowAux
		grid = self.grids[grid_ind]
		d0 = grid.shape[0]
		d1 = grid.shape[1]
		for k in range(3):
			for l in range(3):
				a = i+k-1
				b = j+l-1
				if(a < 0 or b < 0 or a >= d0 or b >= d1):
					window[k][l] = self.getWrapValue(grid_ind,a,b)
				else:
					window[k][l] = grid[a][b]

		return window



	def getWrapValue(self, grid_ind, i,j):
		grid = self.grids[grid_ind]
		d0 = grid.shape[0]
		d1 = grid.shape[1]

		#Checking for corners
		if((i==-1 or i ==d0) and (j==-1 or j==d1)):
			tup = self.corners[(grid_ind,i,j)]
			return self.grids[tup[0]][tup[1]][tup[2]]

		#finding the correct transversal
		new_grid_ind = -2
		xj,yj = -2,-2

		if(j==-1): #side 0
			tv = self.transversal(grid_ind,0,i,0)
			new_grid_ind = tv[0]
			xj = tv[2]
			yj = tv[3]
		elif(i==-1): #side 1
			tv = self.transversal(grid_ind,1,0,j)
			new_grid_ind = tv[0]
			xj = tv[2]
			yj = tv[3]
		elif(j==d1): #side 2
			tv = self.transversal(grid_ind,2,i,d1-1)
			new_grid_ind = tv[0]
			xj = tv[2]
			yj = tv[3]
		elif(i==d0): #side 3
			tv = self.transversal(grid_ind,3,d0-1,j)
			new_grid_ind = tv[0]
			xj = tv[2]
			yj = tv[3]

		if(new_grid_ind==-1):
			return -1

		if(new_grid_ind==-2):
			raise RuntimeError('unwrappable value in get wrap value with coords: ('+ str(grid_ind)+","+str(i)+","+str(j)+")") from exc

		val = self.grids[new_grid_ind][xj][yj]

		return val


	def stitching_from_paste(self,paste_data):
		stitching = {}
		corners = {}
		for tup in paste_data: #tup of form [[i,s],[j,t],sign] -> [i,s]: [j,t,sign] but better!, well but redundant
			if((tup[0] in stitching.values()) or (tup[1] in stitching.values())):
				 raise RuntimeError('over-stitching') from exc
			else: 
				s0 = tup[0]
				s1 = tup[1]
				sign = tup[2]
				stitching[s0] = (s1[0],s1[1],sign)
				stitching[s1] = (s0[0],s0[1],sign)

		self.stitching = stitching
		#corner check
		self.corners = {}
		for i in range(len(self.grids) - 1):
			self.corner_init(i,0,0)
			self.corner_init(i,1,0)
			self.corner_init(i,0,1)
			self.corner_init(i,1,1)

	#returns the transverse of grid 
	# note x is the vertical dimension. also the first dimsnion. 
	#Test: the transversal of a tranvseral should be itself. This has been tested some for an edge on RP2
	#format: (self, grid index, grid side, coord0, coord1) coords are adjacent to edge
	def transversal(self,i,s,x,y):
		if(not(self.verify(i,s,x,y))):
			raise RuntimeError('Transverse error on Grid '+ str(i)+ ", edge "+str(s)+", at ("+str(x)+","+str(y)+").") from exc

		if(not((i,s) in self.stitching.keys())):
			return (-1,-1,-1,-1)

		tranverse_tup = self.stitching[(i,s)] #[j,t,sign]
		j = tranverse_tup[0]
		t = tranverse_tup[1]
		sign = tranverse_tup[2]
		d0 = self.grids[i].shape[0]
		d1 = self.grids[i].shape[1]
		ind = y
		dim = d1
		if(s%2==0):
			ind = x
			dim = d0
		if(sign < 0):
			ind = dim - 1 -ind
		#at this point, we should have ind correspond to the first edge's index transformed to be the next edges's index. now we handle edge cases and return the right one. 
		dj0 = self.grids[j].shape[0]
		dj1 = self.grids[j].shape[1]
		xj,yj = -2,-2

		if(t==0):
			xj = ind
			yj = 0
		elif(t==1):
			xj = 0
			yj = ind
		elif(t==2):
			xj = ind
			yj = dj1 - 1
		elif(t==3):
			xj = dj0-1
			yj = ind

		return (j,t,xj,yj)



	#returns true if (a,b) is on edge s in grid i
	def verify(self, i,s,x,y):
		d0 = self.grids[i].shape[0]
		d1 = self.grids[i].shape[1]
		if(x < 0 or x >= d0 or y < 0 or y>= d1):
			return false

		return (s==0 and y==0) or (s==1 and x==0) or (s==2 and y==d1-1) or (s==3 and x==d0-1)


	#returns tuple of form ((grid_ind, s1), (grid_ind,s2)) of adjacent edges to 
	def adjacent_edges(self, grid_ind, i,j):
		grid = self.grids[grid_ind]
		d0 = grid.shape[0]
		d1 = grid.shape[1]

		if(i==0 and j==0):
			return ((grid_ind,0),(grid_ind,1))
		elif(i==0 and j==d1-1):
			return ((grid_ind,1),(grid_ind,2))
		elif(i== d0-1 and j == d1-1):
			return ((grid_ind,2),(grid_ind,3))
		elif(i== d0-1 and j==0):
			return ((grid_ind,0),(grid_ind,3))
		else:
			# raise RuntimeError("Adjacent edge run on non-corner: ("+str(i)+","+str(j)+")") from exc
			print("ADJACENT EDGE RUN ON NON-CORNER: ("+str(i)+","+str(j)+")")
			print("Dimensions: ("+str(d0)+","+str(d1)+")")

	#on a corner, double transversal through corners and return the result. 
	def corner_double_traversal(self, grid_ind, grid_side,i,j):
		tv1 = self.transversal(grid_ind,grid_side,i,j)

		if(tv1[0]==-1): #unstitched case. 
			return (-1,0,0)

		in_side = (tv1[0],tv1[1])
		corner = (tv1[2],tv1[3])
		sides = self.adjacent_edges(tv1[0], corner[0],corner[1])
		out_side = (-1,-1)
		if(sides[0]==in_side):
			out_side = sides[1]
		elif(sides[1]==in_side):
			out_side = sides[0]
		else:
			print(sides)
			print(in_side)
			raise RuntimeError("Adjacent side error") from exc

		tv2 = self.transversal(out_side[0],out_side[1],corner[0],corner[1])

		return (tv2[0],tv2[2],tv2[3])


	#fills in the corners dictionary with pairs that go from an out of bounds grid to the relevant corner it should go to
	# if a corner's double traversals end in unstitched or inconsistent, it sets the corner destination (-1,-1,-1)
	#(a,b) in {0,1}^2
	def corner_init(self,i,a,b):
		grid = self.grids[i]
		d0 = grid.shape[0]
		d1 = grid.shape[1]
		x = (d0-1)*a
		y = (d1-1)*b
		corners = self.corners
		

		adj = self.adjacent_edges(i,x,y)

		dbl_tv1 = self.corner_double_traversal(i,adj[0][1],x,y)
		dbl_tv2 = self.corner_double_traversal(i,adj[1][1],x,y)

		c0,c1 = -2,-2
		if(a==0):
			c0 = -1
		elif (a==1):
			c0 = d0

		if(b==0):
			c1 = -1
		elif(b==1):
			c1 = d1

		if(c0==-2 or c1==-2):
			raise RuntimeError("invalid corner init input") from exc


		if(dbl_tv1==dbl_tv2):
			self.corners[(i,c0,c1)] = dbl_tv1
		else:
			# print("CORNER TRANSVERAL INCONSISTENCY AT CORNER: ("+ str(i)+","+str(x)+","+str(y)+")")
			self.corners[(i,c0,c1)] = (-1,0,0) #send you to the -1 which only has a -1 entry




#2 dimensional grid geometries to use ~
class GridGeometry(BoardGeometry):
	def __init__(self, d0,d1):
		super().__init__([(d0,d1)], [])


class TorusGeometry(BoardGeometry):
	def __init__(self, d0,d1):
		grid_dims = [(d0,d1)]
		paste_data = [
			[(0,0),(0,2),1],
			[(0,1),(0,3),1]
		]
		super().__init__(grid_dims,paste_data)

#projective plane from gluing together two grids. 
class ProjectivePlaneGeometry(BoardGeometry):
	def __init__(self, d0,d1):
		grid_dims = [(d0,d1),(d1,d0)]
		paste_data = [
						[(0,3),(1,0),-1],
						[(0,2),(1,1),-1],
						[(0,0),(1,3),-1],
						[(0,1),(1,2),1]
						]
		super().__init__(grid_dims, paste_data)

class FourGridSphereGeometry(BoardGeometry):
	pass

class CubeGeometry(BoardGeometry):
	def __init__(self, sidelength):
		cube_dims = [(sidelength,sidelength)]*6
		paste_cube = [
			[(0,0),(5,2),1],
			[(0,1), (1,0),1],
			[(0,2), (2,0),1],
			[(0,3),(3,0),1],
			[(1,1),(5,1),-1],
			[(1,2),(4,1),-1],
			[(1,3),(2,1),1],
			[(2,2),(4,0),1],
			[(2,3),(3,1),1],
			[(3,2),(4,3),1],
			[(3,3),(5,3),-1],
			[(4,2),(5,0),1]
		]
		super().__init__(cube_dims,paste_cube)




