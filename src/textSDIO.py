"""
Author: Joel Nakatani
Overview:
text stdio.
"""

import sys
from mpi4py import MPI as openMPI

# Global only for read
red = "\033[31;1;4m"
yellow = "\033[33;1;4m"
green = "\033[32;1;4m"
default = "\033[0m"


def eMessage(string, tab=False):
    """Makes Error message"""
    if tab:
        t = "\t"
    else:
        t = ""

    mprint(f"{t}{red}Error{default}: {string}")
    sys.exit(-1)


def wMessage(string, tab=False):
    """Makes Warning message"""
    if tab:
        t = "\t"
    else:
        t = ""
    mprint(f"{t}{yellow}Warning{default}: {string}")


def dprint(string, space=True):
    """Makes Warning message"""
    if space:
        s = " "
    else:
        s = ""
    mprint(f"{s}{green}[D]{default}: {string}")


def InitMessage(string, tab=True):
    """Makes Initialization Message"""
    if tab:
        t = "\t"
    else:
        t = ""
    mprint(f"{t}Initializing {string}...")
    return


def printTime(start, end, message, unit="min", show=True):
    """prints time"""
    if unit == "min":
        end = end / 60
        start = start / 60
    elif unit == "hour":
        end = end / (60**2)
        start = start / (60**2)
    else:
        eMessage("No such unit set for print time")
    tMessage = f"{message} in {'{:.2f}'.format(end - start)} [{unit}]"
    if show:
        print(f"{tMessage} @ rank {openMPI.COMM_WORLD.Get_rank()}")
    return tMessage


def mprint(*args):
    """mpi print"""
    rank = openMPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print(*args)
