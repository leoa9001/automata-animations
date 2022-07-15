#!/bin/bash

ffmpeg -f image2 -framerate 15 -i frames/%04d.png -loop 0 test4.gif
