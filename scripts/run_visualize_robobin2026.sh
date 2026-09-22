#!/bin/bash

python tools/visualize_robobin2026.py --split train --num 16 --cols 4
python tools/visualize_robobin2026.py --split valid --num 16 --cols 4
python tools/visualize_robobin2026.py --split test --num 16 --cols 4
