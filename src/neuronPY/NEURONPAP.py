"""
To Do:
[x] TDQM
[ ] FInitializeHandler for setK
[x] smarter RMP detection

"""

from mpi4py import MPI
import textSDIO
from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pickle
import os
import pandas as pd
from results.utils.visualize import func as plotRes
from mpl_toolkits import mplot3d
import math
import sys
import time
import tqdm
import copy


def callExperimentMode(**kwargs):
    if "single" in kwargs.keys() and kwargs["single"]:
        # singleRun
        singleRun()
    if "cond" in kwargs.keys() and kwargs["cond"]:
        # measureconductance
        measureCond("IV")

    if "chan" in kwargs.keys() and kwargs["chan"]:
        # multiChannelEffects
        multiChannel()

    if "Ri" in kwargs.keys() and kwargs["Ri"]:
        # Optimal Ri
        # Soma 2.5836550239043317 MOhm
        # PAP 1035.108930679734 MOhm

        print(
            minimize(
                measureRi,
                (10, 30, 5, 0.02, 1),
                method="Nelder-Mead",
                bounds=[(1, None), (10, None), (1, None), (1e-10, None), (1, 50)],
                options={"disp": True},
                tol=0.00001,
            )
        )
    if "Ri" in kwargs.keys() and kwargs["Ri"]:
        # Plot for vatious distnace channel counts
        multiDistance([3, 30, 4.28, 4.3e-4, 1])


if __name__ == "__main__" or parallel:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size > 1:
        parallel = True
    else:
        parallel = False

        #     print(f'rank{rank} initialized')
        # sys.stdout.flush()
    comm.Barrier()

    # arg parse

    if parallel:
        comm.Barrier()
        start = time.time()
        comm.bcast(start, root=0)

    callExperimentMode()

    if parallel:
        comm.Barrier()
        end = time.time()
        comm.bcast(end, root=0)
    if parallel and rank == 0:
        time_took = end - start
        with open(f"timeres{size}.txt", "w") as f:
            f.write(str(time_took))
    # measureRi((2.8e8,50,3.5e7,3))
