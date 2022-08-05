import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import ImageColor

import CellAutomata as ca
import BoardGeometry as bg


#visualizer: Should maintain a cellboard and then be able to
#1 generate frames from a given cellboard/grid
#recolor etc. 
#2 initialize/edit and have the counting boards. 
#this is not the parser 


class BoardVisualizer:
	fps = 15
	board = None
	width,height = None, None
	num_colors = None
	frame_num = 0



	border_color = [0,0,0]
	colors = [
	    np.array([[31,119,180]]),
	    np.array([[255,127,14]]),
	    np.array([[44,160,44]]),
	    np.array([[214,39,40]]),
	    np.array([[148,103,189]]),
	    np.array([[140,86,75]]),
	    np.array([[227,119,194]]),
	    np.array([[127, 127, 127]]),
	    np.array([[188, 189, 34]]),
	    np.array([[23, 190, 207]])
	]



	#defined from a board
	def __init__(self, cb):
		self.board = cb
		gr = self.board.getBoard()
		self.width = gr.shape[0]
		self.height = gr.shape[1]
		self.num_colors = cb.num_states

	#not best practice to use this but you can append frames/framenums this way. 
	def reset_board(self,cb):
		self.board = cb
		gr = self.board.getBoard()
		self.width = gr.shape[0]
		self.height = gr.shape[1]
		self.num_colors = cb.num_states

	#like the end of a coolors.com url hex_string with - separating them
	def set_palette(self, hex_string):
		hs = hex_string.split("-")
		l = len(hs)
		c = np.zeros((l,3))
		for i in range(l):
			a = ImageColor.getcolor("#"+hs[i], "RGB")
			c[i][0] = a[2]
			c[i][1] = a[1]
			c[i][2] = a[0]
		self.colors = c



	#random image processing functions

	def dist_pt(self,p1, p2):
	    return np.sum(np.square(p1 - p2))

	def closest_color_index(self,pix, maxind):
	    index = 0
	    dist = self.dist_pt(pix, self.colors[0])
	    for i in range(1, maxind):
	        dist2 = self.dist_pt(pix, self.colors[i])
	        if dist2 < dist:
	            dist = dist2
	            index = i
	    return index

	def invert_img(self,img,w,h):
	    img2 = np.zeros((w,h,3), dtype = np.uint8)
	    for i in range(w):
	        for j in range(h):
	            img2[i][j] = self.invert_pixel(img[i][j])
	    return img2

	def invert_pixel(self, pix):
	    d = pix
	    for i in range(3):
	        d[i] = (256 - d[i])%256
	    return d

	def avg_pixel(self, img):
		run_sum = np.zeros((3))
		run_sum[0] = np.sum(img[:,:,0])
		run_sum[1] = np.sum(img[:,:,1])
		run_sum[2] = np.sum(img[:,:,2])

		run_sum = (1/(img.shape[0]*img.shape[1]))*run_sum
		for i in range(3):
			run_sum[i] = int(run_sum[i])
		return run_sum


	#cellboards -> frames

	def grid_from_image(self, img, w, h):
	    grid = np.zeros((w,h))
	    for i in range(w):
	        for j in range(h):
	            grid[i][j] = self.closest_color_index(img[i][j],self.num_colors)
	    return grid


	def make_image(self, frame_i, grid, image):
	    for i in range(self.num_colors):
	        mask = (grid == i)
	        image[mask] = self.colors[i]
	    bd_mask = (grid==-1)
	    image[bd_mask] = self.border_color

	    resize_factor = 8
	    out_image = cv2.resize(
	        image,
	        (resize_factor * self.height, resize_factor * self.width),
	        interpolation=cv2.INTER_NEAREST
	    )
	    cv2.imwrite(f'frames/{frame_i:04d}.png', out_image)


	def gen_frames(self, initial_seconds, seconds,fps):
		Path('frames').mkdir(exist_ok=True)
		fn = self.frame_num
		
		image = np.zeros((self.width, self.height, 3), dtype=np.uint8)
		grid = self.board.getBoard()

		initial_frames = initial_seconds * fps
		for frame_i in tqdm(range(fn, fn + initial_frames)):
			self.make_image(frame_i, grid, image)

		subsequent_frames = seconds * fps
		fn += initial_frames
		for frame_i in tqdm(range(fn, fn + subsequent_frames)):
			self.make_image(frame_i, grid, image)
			self.board.updateBoard()
			grid = self.board.getBoard()

		self.frame_num = fn+subsequent_frames


class LayoutVisualizer(BoardVisualizer):
	scale_d0, scale_d1 = None,None
	blank_grid = None
	layout = None


	def __init__(self, cb, d0, d1,layout):
		super().__init__(cb)
		self.scale_d0 = d0
		self.scale_d1 = d1 
		self.layout = layout


	def make_image(self, frame_i, grid, image):
		sp = self.layout.shape
		sd0 = self.scale_d0
		sd1 = self.scale_d1
		out_image = np.zeros((sp[0]*sd0,sp[1]*sd1,3),dtype = np.uint8)
		grids = self.board.board.grids

		
		for i in range(sp[0]):
			for j in range(sp[1]):
				grid = grids[self.layout.form[i][j]]
				if(self.layout.form[i][j]==-1): 
					pass #fill this in with setting the border grid. 
				else:
					out_image[sd0*i:sd0*(i+1),sd1*j:sd1*(j+1)] = self.grid_to_img(grid)
				
				
		cv2.imwrite(f'frames/{frame_i:04d}.png', out_image)

	def grid_to_img(self, grid):
		img = np.zeros((grid.shape[0],grid.shape[1],3),dtype = np.uint8)
		for i in range(grid.shape[0]):
			for j in range(grid.shape[1]):
				col = None
				state = int(grid[i][j])
				if(state==-1):
					col = self.border_color
				else:
					col = self.colors[state]
				img[i][j] = col
		return img



class Layout:
	shape = None
	form = None

	def __init__(self, shape, form):
		self.shape = shape 
		self.form = form

class CubeLayout(Layout):
	def __init__(self):
		super().__init__((3,4),[
								[-1,1,-1,-1],
								[0,2,4,5],
								[-1,3,-1,-1]
								])


class OneFace(Layout):
	def __init__(self,i):
		super().__init__((1,1),[[i]])






class IconVisualizer(BoardVisualizer):
	icons = None
	border_icon = None
	icon_dim = None


	#load square icons
	def __init__(self,board, path_to_icon):
		super().__init__(board)
		num_icons = board.num_states
		self.icons = [0 for x in range(num_icons)]
		for i in range(num_icons):
			self.icons[i] = cv2.imread(path_to_icon+str(i)+".png")
		self.border_icon = cv2.imread(path_to_icon+"border-icon.png")
		self.icon_dim = (self.icons[0]).shape[0]

	def set_palette_from_icons(self):
		col = np.zeros((len(self.icons),3))
		for i in range(len(self.icons)):
			col[i] = self.avg_pixel(self.icons[i])
		self.colors = col 

	def make_image(self, frame_i, grid, image):
		idim = self.icon_dim
		out_image = np.zeros((idim*self.width,idim*self.height,3),dtype = np.uint8)

		for i in range(self.width):
			for j in range(self.height):
				if(int(grid[i][j])>=0):
					out_image[idim*i:idim*(i+1),idim*j:idim*(j+1)] = self.icons[int(grid[i][j])] 
				elif(int(grid[i][j])==-1):
					out_image[idim*i:idim*(i+1),idim*j:idim*(j+1)] = self.border_icon
				else:
					print("ICON-STATE ISSUE")
		cv2.imwrite(f'frames/{frame_i:04d}.png', out_image)



if __name__ == '__main__':
	pass




