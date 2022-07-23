#!/bin/bash

ffmpeg -f image2 -framerate $2 -i frames/%04d.png -loop 0 outputs/$1
