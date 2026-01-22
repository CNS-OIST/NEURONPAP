"""
Main file to call
astrocyte.hoc
and run simulation programming via
experiments.py

NEURONPAP.py reads command line using CLIargParser.py,
executing desired experimental protocol.
experiments.py runs the actual simulations,
which calls astrocyte.py.
astrocyte.py is the python wrapper for
astrocyte.hoc

by RJ Nakatani
"""

import time
from CLIargParser import argParser
from experiments import procedure
from utils import *
from scipy.optimize import minimize


def callExperimentMode(**kwargs):
    for k, v in kwargs.items():
        if type(v) == list:
            kwargs[k] = v[0]
    exp = procedure(kwargs["seed"], kwargs["ko"])

    exp.parallel = kwargs["parallel"]
    exp.NMDAR = bool(kwargs["nmdar"])
    exp.GABAR = bool(kwargs["gabar"])
    exp.GAP = bool(kwargs["gap"])
    exp.GluT = bool(kwargs["glut"])
    exp.PAPCount = kwargs["papcount"]
    exp.stimCount = kwargs["stimcount"]
    exp.freq = kwargs["freq"]
    exp.ek = kwargs["ek"]
    exp.OE = bool(kwargs["overexpress"])

    if "spill" in kwargs.keys() and kwargs["spill"]:
        exp.PAPLen = 5

    if "nonReadData" in kwargs.keys() and kwargs["nonReadData"]:
        exp.no_read_data = True
    else:
        exp.no_read_data = False

    if "single" in kwargs.keys() and kwargs["single"]:
        # singleRun
        print("single run")
        exp.GluStim = kwargs["glustim"]
        exp.GabaStim = kwargs["gabastim"]

        exp.singleRun()

    if "shell" in kwargs.keys() and kwargs["shell"]:
        exp.distance_analysis()

    if "cond" in kwargs.keys() and kwargs["cond"]:
        # measureconductance
        exp.measureCond("IV")

    if "chan" in kwargs.keys() and kwargs["chan"]:
        # multiChannelEffects
        exp.multiChannel()

    if "freqComp" in kwargs.keys() and kwargs["freqComp"]:
        # multiChannelEffects
        exp.freqComparison()

    if "Ri" in kwargs.keys() and kwargs["Ri"]:
        # Optimal Ri
        # Soma 2.5836550239043317 MOhm
        # PAP 1035.108930679734 MOhm
        exp.measureRi()

    if "distance" in kwargs.keys() and kwargs["distance"]:
        # Plot for vatious distnace channel counts
        exp.multiDistance((10, 10, 2, 1.3, 1))

    if "channel" in kwargs.keys() and kwargs["channel"]:
        # Plot for various channel counts
        exp.GluStim = kwargs["glustim"]
        if exp.GluStim:
            exp.GABAR = False
        exp.GabaStim = kwargs["gabastim"]
        if exp.GabaStim:
            exp.NMDAR = False
            exp.GluT = False
            exp.GABAR = True
        exp.channelComparison()

    if "gabacomp" in kwargs.keys() and kwargs["gabacomp"]:
        # Plot for various channel counts
        exp.GABANMDARCompare()

    if "length" in kwargs.keys() and kwargs["length"]:
        # Plot for various length
        print("compareLen deprecated")

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
        if exp.GluStim:
            exp.NMDAR = True
            exp.GABAR = False
            exp.GluT = True
        exp.GabaStim = kwargs["gabastim"]
        if exp.GabaStim:
            exp.NMDAR = False
            exp.GABAR = True
            exp.GluT = False
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

    if "somaclamp" in kwargs.keys() and kwargs["somaclamp"]:
        exp.SomaCC()
        exp.SomaVC()

    if "gluspill" in kwargs.keys() and kwargs["gluspill"]:
        exp.GluStim = kwargs["glustim"]
        exp.KStim = kwargs["kstim"]
        exp.GabaStim = kwargs["gabastim"]
        if exp.GluStim:
            exp.GluT = True
            exp.GABAR = False
        elif exp.GabaStim:
            exp.GABAR = True
            exp.GluT = False

        # run before kocomp for automatic peak identification
        exp.glutamateSpillOver()

    if "kocomp" in kwargs.keys() and kwargs["kocomp"]:
        exp.KOComp()

    if "ekcomp" in kwargs.keys() and kwargs["ekcomp"]:
        exp.ekComp()

    if "bathcomp" in kwargs.keys() and kwargs["bathcomp"]:
        mprint("running bathExp")
        exp = procedure(4, 0)
        if size == 3:
            exp.bathExperiment()
        elif size == 5:
            exp.bathExperiment(invivo=True)

    if "optVm" in kwargs.keys() and kwargs["optVm"]:
        print(
            minimize(
                exp.optDepolarizationSearch,
                (10, 10, 10),
                method="Nelder-Mead",
                bounds=[(0, None), (0, None), (0, None)],
            )
        )

    if "expVm" in kwargs.keys() and kwargs["expVm"]:
        print(
            minimize(
                exp.optPotassiumSearch, (10), method="Nelder-Mead", bounds=[(0, None)]
            )
        )
        # res = minimize(
        #     exp.optSpikeSearch,
        #     (10000, 96),
        #     method="Nelder-Mead",
        #     bounds=[(0, None), (0, None)],
        # )
        print(res)

    if "expRMP" in kwargs.keys() and kwargs["expRMP"]:
        res = minimize(
            exp.optRMPSearch,
            (100),
            method="Nelder-Mead",
            bounds=[(0, None)],
        )
        print(res)

    if "physiological" in kwargs.keys() and kwargs["physiological"]:
        exp.physiological_stim()


if __name__ == "__main__" or parallel:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    comm.Barrier()
    # arg parse
    args = argParser().parse_args()
    args.__dict__["parallel"] = True
    callExperimentMode(**args.__dict__)
