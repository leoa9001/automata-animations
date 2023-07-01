import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import random

import CellAutomata as CA
import BoardGeometry as BG
import BoardVisualizer as BV




def rand_grid(width, height, values):
	L = len(values)
	grid = np.zeros((width,height))
	for i in range(width):
		for j in range(height):
			v= random.randint(0,L-1)
			grid[i][j] = values[v]
	return grid

def set_from_img(bvis, grid_ind, filename):
		img = cv2.imread("img-assets/"+filename)
		grid = bvis.grid_from_image(img,img.shape[0],img.shape[1])
		bvis.board.board.setGrid(grid_ind,grid)


if __name__ == '__main__':

	img = cv2.imread("img-assets/bluepichu.png")
	w,h = img.shape[0],img.shape[1]
	# w,h = 40,40

	automatan = CA.MatchUpBoard(BG.CubeGeometry(w),26,3,"assets/melee-chart.txt")




	# bv = BV.LayoutVisualizer(cb,w,h,BV.CubeLayout())
	bv = BV.IconVisualizer(automatan, w,h, BV.CubeLayout(),"img-assets/melee-icons/")
	bv.set_palette_from_icons()
	# bv.set_palette("A8A77A-EE8130-6390F0-F7D02C-7AC74C-96D9D6-C22E28-A33EA1-E2BF65-A98FF3-F95587-A6B91A-B6A136-735797-6F35FC-705746-B7B7CE-D685AD")

	# for i in range(6):
	# 	set_from_img(bv,i, "bluepichu.png")

	set_from_img(bv,0,"greenfox.png")
	set_from_img(bv,1,"redmarth.png")
	set_from_img(bv,2,"whitefalco.png")
	set_from_img(bv,3,"greenpuff.png")
	set_from_img(bv,4,"redyoshi.png")
	set_from_img(bv,5,"whitesheik.png")


	bv.gen_frames(1,10,10)




	#neon: f8ff30-5afafa-ff30be-ff9b30-25eb50
	#nice one: cae7b9-f3de8a-eb9486-7e7f9a-97a7b3
	#greens: 30ff5d-5bfa7d-30ff9f-4cff30-25eb50
	#pokemon: A8A77A-EE8130-6390F0-F7D02C-7AC74C-96D9D6-C22E28-A33EA1-E2BF65-A98FF3-F95587-A6B91A-B6A136-735797-6F35FC-705746-B7B7CE-D685AD


