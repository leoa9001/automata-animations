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

	def __init__(self, grid_dims, paste_data):
		self.grids = {-1: [[-1]]}
		for i in len(grid_dims):
			self.grids[i] = np.zeros((grid_dims[i][0],grid_dims[i][1]))
		self.stitching = self.

	def stitching_from_paste(self,paste_date):
		stitching = {}
		corners = {}
		for tup in paste_date: #tup of form [(i,s),(j,t),sign] -> (i,s): (j,t,sign)
			if(tup in stitching.values()):
				 raise RuntimeError('over-stitching') from exc
			else: 
				s0 = tup[0]
				s1 = tup[1]
				sign = tup[2]
				stitching[s0] = [s1[0],s1[1],sign]
				stitching[s1] = [s2[0],s2[1],sign]


		self.stitching = stitching

	#returns the transverse of grid 
	# note x is the vertical dimension. also the first dimsnion. 
	#Test: the transversal of a tranvseral should be itself. NOT YET ERROR TESTED. 
	def transversal(i,s,x,y):
		if(not(verify(i,s,x,y))):
			raise RuntimeError('Transverse error on Grid '+ str(i)+ ", edge "+str(s)+", at ("+str(x)+","+str(y)+").") from exc

		tranverse_tup = stitching[[i,s]] #[j,t,sign]
		j = tranverse_tup[0]
		t = tranverse_tup[1]
		sign = tranverse_tup[2]
		ind = y
		dim = d1
		if(s%2==0):
			ind = x
			dim = d0
		if(sign < 0):
			ind = dim - 1 -ind
		#at this point, we should have ind correspond to the first edge's index transformed to be the next edges's index. now we handle edge cases and return the right one. 
		dj0 = grids[j].shape[0]
		dj1 = grids[j].shape[1]
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

		return [j,t,xj,yj]




	#returns true if (a,b) is on edge s in grid i
	def verify(i,s,x,y):
		d0 = self.grids[i].shape[0]
		d1 = self.grids[i].shape[1]
		if(x < 0 or x >= d0 or y < 0 or y>= d1):
			return false

		return (s==0 and y==0) or (s==1 and x==0) or (s==2 y==d1-1) or (s==3 and x==d0-1)




#2 dimensional grid. Probably only used by RPSBoard
class GridGeometry(BoardGeometry):
	pass




class CubeGeometry(BoardGeometry):
	pass

