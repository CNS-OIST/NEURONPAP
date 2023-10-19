"""
Author: Joel Nakatani
Overview:
Option parser for command line input.
Uses argparse module.
"""

import argparse
import random


def randomGen(rangeInt):
    return random.randint(0, rangeInt)


def argParser(rangeInt=100):
    parser = argparse.ArgumentParser(
        description="Hippocampal astrocyte model implemented in NEURON"
    )
    parser.add_argument(
        "-r",
        "--mRi",
        dest="Ri",
        action="store_true",
        default=False,
        help="measure input Resistance",
    )  

    parser.add_argument(
        "-s",
        "--singleRun",
        dest="single",
        action="store_true",
        default=False,
        help="Used to run a single simulation",
    )
     
    parser.add_argument(
        "-d",
        "--distance",
        dest="distance",
        action="store_true",
        default=False,
        help="Used to run simulations for various distances"
    )  
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Used to employ debug mode. You can use to run with verbose comments. Mainly used to test if the simulation will construct -> run -> save properly",
    )
    return parser


if __name__ == "__main__":
    parser = argParser()
    args = parser.parse_args()
    print(args)
