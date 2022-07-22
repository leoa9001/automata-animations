import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path

import CellAutomata as ca

parser = argparse.ArgumentParser()
parser.add_argument(
    '--width', type=int,
    default=160, help='Width of grid.'
)
parser.add_argument(
    '--height', type=int,
    default=90, help='Height of grid.'
)
parser.add_argument(
    '--num-colors', type=int,
    default=3, help='Number of states in the automaton.'
)
parser.add_argument(
    '--neighbour-threshold', type=int,
    default=3, help='Number of neighbours needed for a transition.'
)
parser.add_argument(
    '--seconds', type=int,
    default=10, help='Number of seconds (at 15 FPS).'
)
args = parser.parse_args()

initial_seconds = 1
fps = 15


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

convolution = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]
)


def update(grid):
    new_grid = np.copy(grid)

    color_grids = [grid == i for i in range(args.num_colors)]

    ns = np.arange(args.num_colors)
    for i, j in zip(ns, np.roll(ns, 1)):
        target_mask = color_grids[i]
        neighbour_grid = color_grids[j]
        neighbour_mask = convolve2d(
            neighbour_grid,
            convolution,
            mode='same',
            boundary='wrap'
        ) >= args.neighbour_threshold

        mask = np.logical_and(
            target_mask,
            neighbour_mask
        )
        new_grid[mask] = j
    return new_grid


def dist_pt(p1, p2):
    return np.sum(np.square(p1 - p2))

def closest_color_index(pix,maxind):
    index = 0
    dist = dist_pt(pix, colors[0])
    for i in range(1, maxind):
        dist2 = dist_pt(pix,colors[i])
        if dist2 < dist:
            dist = dist2
            index = i
    return index

def invert_img(img,w,h):
    img2 = np.zeros((w,h,3), dtype = np.uint8)
    for i in range(w):
        for j in range(h):
            img2[i][j] = invert_pixel(img[i][j])
    return img2

def invert_pixel(pix):
    d = pix
    for i in range(3):
        d[i] = (256 - d[i])%256
    return d

def grid_from_image(img, w, h, colors):
    grid = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            grid[i][j] = closest_color_index(img[i][j],colors)
    return grid


def make_image(frame_i, grid, image):
    for i in range(args.num_colors):
        mask = grid == i
        image[mask] = colors[i]


    resize_factor = 8
    out_image = cv2.resize(
        image,
        (resize_factor * args.height, resize_factor * args.width),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(f'frames/{frame_i:04d}.png', out_image)

def gen_frames_rps(grid, width,height, num_colors):
    image = np.zeros((args.width,args.height, 3), dtype=np.uint8)

    initial_frames = initial_seconds * fps
    for frame_i in tqdm(range(initial_frames)):
        make_image(frame_i, grid, image)

    subsequent_frames = args.seconds * fps
    for frame_i in tqdm(range(initial_frames, initial_frames + subsequent_frames)):
        make_image(frame_i, grid, image)
        grid = update(grid)

def gen_frames_gol(grid,width,height):
    cb = ca.CellBoard(width, height)
    cb.setGrid(grid)

    image = np.zeros((args.width, args.height, 3), dtype=np.uint8)

    initial_frames = initial_seconds * fps
    for frame_i in tqdm(range(initial_frames)):
        make_image(frame_i, grid, image)

    subsequent_frames = args.seconds * fps
    for frame_i in tqdm(range(initial_frames, initial_frames + subsequent_frames)):
        make_image(frame_i, grid, image)
        cb.updateBoard()
        grid = cb.getBoard()



if __name__ == '__main__':
    Path('frames').mkdir(exist_ok=True)
    w,h,c = args.width, args.height, args.num_colors

    print("Width: " + str(w) + ", Height: "+str(h)+", numcol: "+str(c))
    print("Seconds: "+ str(args.seconds))
    img = cv2.imread('img-assets/blackdk.png')
    # img2 = invert_img(img,w,h)
    # cv2.imwrite("outputs/blackdkneg.png",img2)
    grid = grid_from_image(img,w,h,c)

   
    gen_frames_rps(grid, w, h, c)

    

    
