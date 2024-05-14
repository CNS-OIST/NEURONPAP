"""
To Do:
[x] FInitializeHandler for setK

"""

from mpi4py import MPI
from scipy.optimize import minimize
import time
from CLIargParser import argParser
from experiments import procedure
from utils import *


def callExperimentMode(**kwargs):
    for k, v in kwargs.items():
        if type(v) == list:
            kwargs[k] = v[0]
    exp = procedure(kwargs["seed"], kwargs["ko"])

    exp.parallel = kwargs["parallel"]
    exp.NMDAR = bool(kwargs["nmdar"])
    exp.GluT = bool(kwargs["glut"])
    exp.PAPCount = kwargs["papcount"]
    exp.stimCount = kwargs["stimcount"]
    exp.ek = kwargs["ek"]

    if "spill" in kwargs.keys() and kwargs["spill"]:
        exp.PAPLen = 2

    if "single" in kwargs.keys() and kwargs["single"]:
        # singleRun
        print("single run")
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
        exp.measureRi()

    if "distance" in kwargs.keys() and kwargs["distance"]:
        # Plot for vatious distnace channel counts
        exp.multiDistance((10, 10, 2, 1.3, 1))

    if "nonhoc" in kwargs.keys() and kwargs["nonhoc"]:
        # Run without hoc
        exp.singleRun(readHoc=False)

    if "channel" in kwargs.keys() and kwargs["channel"]:
        # Plot for various channel counts
        exp.channelComparison()

    if "kcomp" in kwargs.keys() and kwargs["kcomp"]:
        # Plot for various channel counts
        exp.GluStim = kwargs["glustim"]        
        exp.potassiumComparison()

    if "video" in kwargs.keys() and kwargs["video"]:
        # Make video
        exp.plotPAPs()

    if "branch" in kwargs.keys() and kwargs["branch"]:
        # Plot branch atten
        exp.GluStim = kwargs["glustim"]        
        exp.branchAttenuation()

    if "stim" in kwargs.keys() and kwargs["stim"]:
        exp.GluStim = kwargs["glustim"]
        exp.KStim = kwargs["kstim"]
        if "delay" in kwargs.keys() and exp.KStim:
            if kwargs["delay"] > 0:
                exp.stimdelay = kwargs["delay"]
        exp.channelComparison()

    if "phase" in kwargs.keys() and kwargs["phase"]:
        # plot phase plot
        exp.kvPhasePlane()

    if "vclamp" in kwargs.keys() and kwargs["vclamp"]:
        # plot phase plot
        exp.SomaVC()

    if "gluspill" in kwargs.keys() and kwargs["gluspill"]:
        # run before kocomp for automatic peak identification
        exp.glutamateSpillOver()
        
    if "kocomp" in kwargs.keys() and kwargs["kocomp"]:
        exp.KOComp()

    if "ekcomp" in kwargs.keys() and kwargs["ekcomp"]:
        exp.ekComp()
        
    if "expVm" in kwargs.keys() and kwargs["expVm"]:
        # print(
        #     minimize(
        #         exp.optDepolarizationSearch,
        #         (43),
        #         method="Nelder-Mead",
        #         bounds=[(0, None)]
        #     )
        # )
        res = minimize(
            exp.optSpikeSearch,
            (10000,96),
            method="Nelder-Mead",
            bounds=[(0, None),(0,None)]
        )
        print(res)
        

if __name__ == "__main__" or parallel:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    comm.Barrier()
    # arg parse
    args = argParser().parse_args()
    args.__dict__["parallel"] = True
    callExperimentMode(**args.__dict__)
