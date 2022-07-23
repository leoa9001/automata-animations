import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path

import CellAutomata as ca
import boardvisualizer as BV




if __name__ == '__main__':

	img = cv2.imread("img-assets/JigglypuffHeadSSBM.png")

	print(img.shape)

	cb2 = ca.RPSBoard(24,24,3,3)
	# cb2.setRandom()

	bv = BV.IconVisualizer(cb2,"img-assets/icons/",3)
	# bv = BV.IconVisualizer(cb2)
	bv.swap_palette("30ff5d-5bfa7d-30ff9f-4cff30-25eb50")
	bv.board.setGrid(bv.grid_from_image(img,img.shape[0],img.shape[1]))
	bv.swap_palette("cae7b9-f3de8a-eb9486-7e7f9a-97a7b3")
	bv.gen_frames(1,10,15)

	#neon: f8ff30-5afafa-ff30be-ff9b30-25eb50
	#nice one: cae7b9-f3de8a-eb9486-7e7f9a-97a7b3
	#greens: 30ff5d-5bfa7d-30ff9f-4cff30-25eb50


