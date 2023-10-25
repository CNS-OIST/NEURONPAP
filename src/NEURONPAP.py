"""
To Do:
[ ] FInitializeHandler for setK

"""

from mpi4py import MPI
from scipy.optimize import minimize
import time
from CLIargParser import argParser
from experiments import procedure
from utils import *

def callExperimentMode(**kwargs):
    exp = procedure()
    
    exp.parallel = kwargs['parallel']
    
    if "single" in kwargs.keys() and kwargs["single"]:
        # singleRun
        print('single run')
        exp.singleRun()
    if "cond" in kwargs.keys() and kwargs["cond"]:
        # measureconductance
        exp.measureCond("IV")

    if "chan" in kwargs.keys() and kwargs["chan"]:
        # multiChannelEffects
        exp.multiChannel()

    if "Ri" in kwargs.keys() and kwargs["Ri"]:
        # Optimal Ri
        # Soma 2.5836550239043317 MOhm
        # PAP 1035.108930679734 MOhm

        print(
            minimize(
                exp.measureRi,
                (20,  30,  2,  8e-04,1),
                method="Nelder-Mead",
                bounds=[(1, None), (10, None), (1, None), (1e-10, None), (1, 50)],
                options={"disp": True},
                tol=0.00001,
            )
        )
    if "distance" in kwargs.keys() and kwargs["distance"]:
        # Plot for vatious distnace channel counts
        exp.multiDistance([3, 30, 4.28, 4.3e-4, 1])


    
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
    args = argParser().parse_args()

    if parallel:
        comm.Barrier()
        start = time.time()
        comm.bcast(start, root=0)

    # print(args.__dict__)
    args.__dict__['parallel'] = True
    callExperimentMode(**args.__dict__)

    if parallel:
        comm.Barrier()
        end = time.time()
        comm.bcast(end, root=0)
    if parallel and rank == 0:
        time_took = end - start
        with open(f"timeres{size}.txt", "w") as f:
            f.write(str(time_took))
    # measureRi((2.8e8,50,3.5e7,3))
