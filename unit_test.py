## Intended to test functionality in a basic way

import numpy as np
import cv2
import random

import CellAutomata as CA
import BoardGeometry as BG
import BoardVisualizer as BV

pokemon_palette = "A8A77A-EE8130-6390F0-F7D02C-7AC74C-96D9D6-C22E28-A33EA1-E2BF65-A98FF3-F95587-A6B91A-B6A136-735797-6F35FC-705746-B7B7CE-D685AD"

def set_from_img(bvis, grid_ind, filename):
		img = cv2.imread("img-assets/"+filename)
		grid = bvis.grid_from_image(img,img.shape[0],img.shape[1])
		bvis.automatan.board.setGrid(grid_ind,grid)

# Make a random square grid animation
def basic_random_grid(width, height, num_frames):
    pass

def pokemon_random_gen(width,height, initial_seconds,seconds,fps):
    automatan = CA.MatchUpBoard(BG.GridGeometry(width,height),18, 2, "./assets/poke-chart.txt")
    automatan.setRandom()

    bv = BV.BoardVisualizer(automatan)
    bv.set_palette(pokemon_palette)

    return bv

    bv.gen_frames(initial_seconds,seconds, fps)



def elliptic_gen():
    image = cv2.imread('img-assets/ellipticcurvestack.png')
    width,height = image.shape[0],image.shape[1]

    automatan = CA.RPSSpockBoard(BG.TorusGeometry(width,height), 3, 3)
    bv = BV.BoardVisualizer(automatan=automatan)
    bv.set_palette(pokemon_palette)
    grid = bv.grid_from_image(image, image.shape[0],image.shape[1])
    bv.automatan.board.setGrid(0,grid)

    bv.gen_frames(1,5,5)

# Going to try to make this shit work with elliptic curves stack on pokemon.
if __name__ == '__main__':
    width = 100
    height = 100
    initial_seconds = 2
    seconds = 15
    fps = 10

    bv = pokemon_random_gen(width, height)

    bv.gen_frames(initial_seconds,seconds, fps)





