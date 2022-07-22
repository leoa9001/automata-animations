import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import ImageColor

import CellAutomata as ca




# make two classes: 
# rps_automata as the updater 
# visualizer: as code for taking grids and generating frames etc. 


#visualizer: Should maintain a cellboard and then be able to
#1 generate frames from a given cellboard/grid
#recolor etc. 
# initialize/edit and have the counting boards. 
#this is not the parser we'll add a main.py


class BoardVisualizer:
	fps = 15
	board = None
	width,height = None, None
	num_colors = None
	frame_num = 0


	border_color = [255,255,255]
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
	def swap_palette(self, hex_string):
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

if __name__ == '__main__':
	pass




