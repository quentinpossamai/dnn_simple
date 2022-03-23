#!/bin/bash
cd ..
python -m src.multiplication\
 --activation relu\
 --batch_size 20\
 --max_epochs 1000\
 --layer 1\
 --lr 1e-3\
 --unit 2
