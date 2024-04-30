"""
Author: Joel Nakatani
Overview:
Option parser for command line input.
Uses argparse module.
"""

import argparse
import random
from distutils.util import strtobool


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
        help="flag for optimization (works with input Resistance Flag",
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
        help="Used to run simulations for various distances",
    )
    parser.add_argument(
        "-c",
        "--channels",
        dest="channel",
        action="store_true",
        default=False,
        help="Used to run simulations for various channel counts",
    )
    parser.add_argument(
        "--spillOver",
        dest="spill",
        action="store_true",
        default=False,
        help="Used to run simulations with spill over, only works for KV Phase plot",
    )
    parser.add_argument(
        "--kComp",
        dest="kcomp",
        action="store_true",
        default=False,
        help="Used to run simulations for various potassium conc. Use --stimGlu to turn on Glutamate Stim.",
    )
    parser.add_argument(
        "--ekComp",
        dest="ekcomp",
        action="store_true",
        default=False,
        help="Used to run simulations for comparison of ek clamp",
    )
    parser.add_argument(
        "--koComp",
        dest="kocomp",
        action="store_true",
        default=False,
        help="Used to run simulations for comparing various KO conditions",
    )
    parser.add_argument(
        "--gluSpill",
        dest="gluspill",
        action="store_true",
        default=False,
        help="Used to run simulations for glutamate spillover.",
    )
    parser.add_argument(
        "-v",
        "--video",
        dest="video",
        action="store_true",
        default=False,
        help="Used to run simulations for video",
    )
    parser.add_argument(
        "-b",
        "--branch",
        dest="branch",
        action="store_true",
        default=False,
        help="Used to run simulations for branch attenuation. Use --stimGlu to turn on Glutamate stimulus.",
    )
    parser.add_argument(
        "-p",
        "--phase",
        dest="phase",
        action="store_true",
        default=False,
        help="Used to run simulations for phase plot",
    )
    parser.add_argument(
        "--vClamp",
        dest="vclamp",
        action="store_true",
        default=False,
        help="Used to run v clamp simulations",
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
        nargs=1,
        help="Used to set extracellular potassium of simulation. Default: 0.5 mM.",
    )  # end time arg

    parser.add_argument(
        "--PAPCount",
        type=int,
        dest="papcount",
        default=1,
        nargs=1,
        help="Used to set number of active PAPs in simulation. Default: 1",
    )  # end time arg
    
    parser.add_argument(
        "--stimCount",
        type=int,
        dest="stimcount",
        default=1,
        nargs=1,
        help="Used to set the number of potassium stimulations in the simulation. Default: 0.5 mM at 100 Hz.",
    )  # end time arg

    parser.add_argument(
        "--ExpVm",
        action="store_true",
        default=False,
        dest="expVm",
        help="Used when optimizing Vm",
    )

    
    parser.add_argument(
        "--ek",
        type=int,
        dest="ek",
        default=None,
        nargs=1,
        help="Used to set the ek; only set when called. Works only during channel comparison experiment and single run",
    )

    parser.add_argument(
        "--stim",
        dest="stim",
        action="store_true",
        default=False,
        help="Called for running experiments with specific stimulus setting. Run along with option --stimGlu --stimK to turn on the specific stimulus. Glu, K are both initially set to False when --stim is called. With --delay you can delay the glutamate stimulus",
    )

    parser.add_argument(
        "--delay",
        type=float,
        dest="delay",
        nargs=1,
        default=0,
        help="Used to set the delay of glutamate stimulus after K stimulus",
    )  # end time arg

    parser.add_argument(
        "--stimGlu",
        dest="glustim",
        action="store_true",
        default=False,
        help="Works with --kComp, --branch to turn on/off glutamate stim. (With option --stimGlu it is turned on."
    )
    parser.add_argument(
        "--stimK",
        dest="kstim",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--GluT",
        type=int,
        dest="glut",
        default=0,
        help="Used to set Glutamate Transporters; 1 True; 0 False",
    )

    parser.add_argument(
        "--NMDAR",
        type=int,
        dest="nmdar",
        default=1,
        help="Used to set NMDAR; 1 True; 0 False",
    )

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
