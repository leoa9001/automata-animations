Fork from [rps-automata](https://github.com/grey-area/rps-automata).

---
## Guide to usage for automata animations

In a bunch of commits, this repository is pretty far removed from rps-automata. Here is the basic rundown of how to use it:

1. Write code in python using classes BoardGeometry, BoardVisualizer, CellAutomata to generate individual png frames of an animation into a folder called "frames"
2. use `make_gif.sh` or `make_video.sh` as follows to weave the frames into a gif/mp4: `./make_video.sh video_name.mp4 framerate`
3. 

There are three main classes, BoardGeometry (BG), BoardVisualizer (BV), and CellAutomata (CA). In general, CellAutomata is a class that sets rules for Cell Automatan computation: you can set transition rules, set the initial board, set the geometry (e.g. flat cube, torus, etc. via BG). Then, feed it to a BoardVisualizer to fill out functionality for making gifs from the automata.

---
## Old rps-automata readme
Create animations of rock-paper-scissors automata, where each cell transitions
to the next state if at least a given number of its neighbours are in that state.

Requirements are in the ``requirements.txt`` file, so ``pip3 install -r requirements.txt``.

Usage:

``python rps_automata.py --width <grid width> --height <grid height> --num-colours <number of states> --neighbour-threshold <neighbours needed for state transition> --seconds <number of seconds at 15 FPS>``

This will render frames in the ``frames`` directory, which can then be made into a video by
running the ``make_video.sh`` script (requires ffmpeg).

![](example.gif)


