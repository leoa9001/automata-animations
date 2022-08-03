import numpy as np



#Functionality:
#Initialize a board geometry
#Iterate through cells
#Get a cell's boundary window
#Update a cell's value

#Standard board geometry: a set of 2-d grids
class BoardGeometry:
	grids = None
	stitching = None
	corners = None

	windowAux = None

	def __init__(self, grid_dims, paste_data):
		self.grids = {-1: [[-1]]}
		for i in range(len(grid_dims)):
			self.grids[i] = np.zeros((grid_dims[i][0],grid_dims[i][1]))
		self.stitching_from_paste(paste_data)
		self.windowAux = [[0 for x in range(3)] for y in range(3)]



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
			tup = corners[(grid_ind,i,j)]
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

		#corner check
		# for i in len(self.grids):
		# 	self.corner_check(i,0,0)
		# 	self.corner_check(i,1,0)
		# 	self.corner_check(i,0,1)
		# 	self.corner_check(i,1,1)

		self.stitching = stitching
		self.corners = corners

	#returns the transverse of grid 
	# note x is the vertical dimension. also the first dimsnion. 
	#Test: the transversal of a tranvseral should be itself. NOT YET ERROR TESTED. 
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

	# def corner_check(i,a,b):
	# 	grid = self.grids[i]
	# 	corners = self.corners
	# 	if((a,b)==(0,0)):
	# 		j_tup = self.transversal(i,0,0,0)
	# 		if(j_tup[0]==-1):
	# 			corners[[i,-1,-1]] = [-1,-1,-1]
	# 		elif()
	# 	elif((a,b)==(0,1)):





	# 	self.corners = corners

	# # def double_traverse(i,s,j,t):





#2 dimensional grid. Probably only used by RPSBoard
class GridGeometry(BoardGeometry):
	pass


class TorusGeometry(BoardGeometry):
	pass

class ProjectivePlaneGeometry(BoardGeometry):
	pass

class FourGridSphereGeometry(BoardGeometry):
	pass

class CubeGeometry(BoardGeometry):
	pass

