#!/bin/bash
#SBATCH -J tetris
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 100:00:00

source activate tetris_310

python -B train.py
