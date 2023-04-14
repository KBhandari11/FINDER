# -*- coding: utf-8 -*-
##!CUDA_VISIBLE_DEVICES=0 python train.py
from FINDER import FINDER

def main():
    dqn = FINDER()
    dqn.Train()


if __name__=="__main__":
    main()
