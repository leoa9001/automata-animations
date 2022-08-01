import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import random

import CellAutomata as ca
import boardvisualizer as BV




def rand_grid(width, height, values):
	L = len(values)
	grid = np.zeros((width,height))
	for i in range(width):
		for j in range(height):
			v= random.randint(0,L-1)
			grid[i][j] = values[v]
	return grid




if __name__ == '__main__':

	img = cv2.imread("img-assets/bluepichu.png")
	w,h = img.shape[0],img.shape[1]
	# w,h = 40,40


	# cb2 = ca.MatchUpBoard(10,10, 26, 2, "assets/MeleeMUChart.txt")
	cb2 = ca.WeightedMatchUpBoard(w,h,26,3,"assets/MeleeMUChart.txt")

	# cb3 = ca.RPSSpockBoard(w,h,9,3)




	# cb2 = ca.MatchUpBoard(150,150,18,2,"assets/poke-chart.txt")

	cb2.setRandom()


	bv = BV.IconVisualizer(cb2,"img-assets/meleecon/")
	# bv = BV.BoardVisualizer(cb3)

	# bv.set_palette("f8ff30-5afafa-ff30be-ff9b30-25eb50")

	# bv.set_palette("A8A77A-EE8130-6390F0-F7D02C-7AC74C-96D9D6-C22E28-A33EA1-E2BF65-A98FF3-F95587-A6B91A-B6A136-735797-6F35FC-705746-B7B7CE-D685AD")

	bv.set_palette_from_icons()

	# bv.board.setGrid(bv.grid_from_image(img,img.shape[0],img.shape[1]))
	
	# bv.board.setRandom()
	# bv.board.setGrid(rand_grid(w,h, range(0,26)))

	bv.gen_frames(1,10,10)



	#neon: f8ff30-5afafa-ff30be-ff9b30-25eb50
	#nice one: cae7b9-f3de8a-eb9486-7e7f9a-97a7b3
	#greens: 30ff5d-5bfa7d-30ff9f-4cff30-25eb50


