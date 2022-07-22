import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path

import CellAutomata as ca
import boardvisualizer as BV




if __name__ == '__main__':

	img = cv2.imread("img-assets/jo.png")

	print(img.shape)

	cb0 = ca.RPSBoard(img.shape[0],img.shape[1],3,3)
	cb0.setRandom()
	cb1 = ca.RPSBoard(img.shape[0],img.shape[1],9,1)
	cb2 = ca.RPSBoard(img.shape[0],img.shape[1],5,2)

	bv = BV.BoardVisualizer(cb0)
	# bv.board.setGrid(bv.grid_from_image(img,img.shape[0],img.shape[1]))
	# bv.gen_frames(1,5,15)
	# bv.reset_board(cb1)
	# bv.board.setGrid(bv.grid_from_image(img,img.shape[0],img.shape[1]))
	# bv.gen_frames(1,5,15)

	bv.reset_board(cb2)
	bv.swap_palette("30ff5d-5bfa7d-30ff9f-4cff30-25eb50")
	bv.board.setGrid(bv.grid_from_image(img,img.shape[0],img.shape[1]))
	bv.swap_palette("cae7b9-f3de8a-eb9486-7e7f9a-97a7b3")
	bv.gen_frames(1,5,15)


