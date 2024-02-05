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
        "-o",
        "--optimize",
        dest="Optimize",
        action="store_true",
        default=False,
        help="flag for optimization (works with input Resistance Flag"
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
        "-n",
        "--nonhoc",
        dest="nonhoc",
        action="store_true",
        default=False,
        help="Used to run a single simulation in nonhoc",
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
        "-c",
        "--channels",
        dest="channel",
        action="store_true",
        default=False,
        help="Used to run simulations for various channel counts"
    )  
    parser.add_argument(
        "-v",
        "--video",
        dest="video",
        action="store_true",
        default=False,
        help="Used to run simulations for video"
    )  
    parser.add_argument(
        "-b",
        "--branch",
        dest="branch",
        action="store_true",
        default=False,
        help="Used to run simulations for branch attenuation"
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Used to employ debug mode. You can use to run with verbose comments. Mainly used to test if the simulation will construct -> run -> save properly",
    )
    parser.add_argument(
        "--ko",
        type=float,
        dest="ko",
        default=0.5,
        help="Used to set extracellular potassium of simulation. Default: 0.5 mM.",
    )  # end time arg
    
    parser.add_argument(
        "seed",
        metavar="seed",
        type=int,
        nargs="?",
        default=randomGen(rangeInt),
        help=f"The seed for the simulation. Will use a randomly generated seed from 0-{rangeInt} if none are provided.",
    )  # positional argument that defaults to nothing
    
    return parser


if __name__ == "__main__":
    parser = argParser()
    args = parser.parse_args()
    print(args)
