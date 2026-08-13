"""
Author: Joel Nakatani
Overview:

Parameters:
"""

from neuron import h
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from global_labels import gl
import os

plt.rcParams.update(gl.font)
saveDir = os.path.abspath("../morphResults")
plt.rcParams["savefig.directory"] = saveDir
plt.ioff()


class fitModelCurve:
    plotInterim = False
    plotFinal = True

    class cellDict:
        def __init__(self, cellType):
            """Store synapse configuration (mechanism name, parameters, data file, current reference) for a GABA or NMDA synapse type."""
            self.Type = cellType
            if cellType == "GABA":
                self.Name = "inhSyn"
                self.Parms = ["tau1", "tau2"]
                self.File = "./Data/curTimeGABA.csv"
                self.currName = "_ref_iGaba"

            elif cellType == "NMDA":
                self.Name = "Exp5NMDA"
                self.Parms = ["tau1_0", "tau2_0"]
                self.File = "./Data"
                self.currName = "_ref_iNMDA"

    def __init__(self, cellType):
        """Initialize NEURON, build the cell/synapse configuration for the given synapse type, and set up the soma and synapse."""
        h.load_file("stdgui.hoc")
        self.cellType = self.cellDict(cellType)
        self.createCell()
        self.createSynapse(self.cellType.Name)

    def optimize(self):
        """Fit synapse kinetic parameters by minimizing the loss function with Nelder-Mead, optionally re-plotting the final fit, and write the optimization result to a text file."""
        res = minimize(
            self.lossFunction,
            self.cellType.initParms,
            method="Nelder-Mead",
            bounds=self.defbounds(),
        )
        if self.plotFinal:
            self.plotInterim = True
            self.lossFunction(res.x)
        with open(f"{self.cellType.Name}_res.txt", "w") as f:
            f.write(res)

    def defbounds(self):
        """Build a list of non-negative bounds, one per synapse parameter, for use in the optimizer."""
        boundsList = []
        for parm in self.cellType.Parms:
            boundsList.append((0, None))
        return boundsList

    def lossFunction(self, x):
        """Run the synapse simulation with parameter vector x and return the sum of squared errors between simulated and experimental current traces, optionally plotting the comparison."""
        expData, expTime = np.array(self.readData())
        self.adjustParms(x)
        self.setupSim(max(expTime) + 1, max(expTime))
        self.runSim()
        simData = np.array(self.adjustSimData(expTime))
        if self.plotInterim:
            self.plot(expTime, expData, simData)
        return sum((simData - expData) ** 2)

    def plot(self, t, exp, sim):
        """Clear the current figure and scatter-plot the experimental and simulated current traces against time."""
        plt.cla()
        plt.clf()
        plt.scatter(t, exp)
        plt.scatter(t, sim)
        plt.show()

    def adjustSimData(self, expTime):
        """Extract the simulated current values at the time points matching the experimental data's time samples."""
        simData = []
        for t, current in zip(self.simTime, self.simData):
            if int(t) in expTime:
                print(t)
                simData.append(current)

        return simData

    def setupSim(self, stimStart, tstop):
        """Configure NEURON's time step, stimulus start time and stop time, then reinitialize the simulation."""
        h.dt = 0.1
        self.stim = stimStart
        h.tstop = tstop
        h.finitialize()
        h.frecord_init()

    def runSim(self):
        """Run the NEURON simulation."""
        h.run()

    def readData(self):
        """Load the experimental current-time data from CSV and return the current normalized by its peak value along with the time vector."""
        expData = pd.read_csv(self.cellType.File)
        return expData["i"] / expData["i"].max(), expData["t"]

    def adjustParms(self, x):
        """Assign the parameter values in x to the corresponding synapse parameters on the synapse object."""
        for i, parm in enumerate(x):
            setattr(self.synSoma, self.cellType.Parms[i], parm)

    def createCell(self):
        """Create a single-compartment soma section with a passive membrane mechanism."""
        self.soma = h.Section(name="soma")
        self.soma.insert("pas")

    def createSynapse(self, synName):
        """Attach the synapse mechanism to the soma, wire up a single-spike NetStim/NetCon to trigger it, set up current and time recording vectors, and read the synapse's initial parameter values."""
        self.synSoma = getattr(h, synName)(self.soma(0.5))
        self.stim = h.NetStim()
        self.stim.number = 1
        self.stim.noise = 0
        self.stim.interval = 1
        self.nc = h.NetCon(self.stim, self.synSoma)
        self.simData = h.Vector()
        self.simData.record(getattr(self.synSoma, self.cellType.currName))
        self.simTime = h.Vector()
        self.simTime.record(h._ref_t)
        self.readInitParms()

    def readInitParms(self):
        """Read the current (initial) values of the synapse's fitted parameters from the synapse object."""
        self.cellType.initParms = []
        for parm in self.cellType.Parms:
            initParm = getattr(self.synSoma, parm)
            self.cellType.initParms.append(initParm)


class calibrateChannel:
    def __init__(self):
        """Initialize NEURON and define, for each ion channel under test, its experimental I-V data file and the corresponding NEURON mechanism/point-process name and type."""
        h.load_file("stdgui.hoc")

        self.channelName = [
            "Kir",
            "TWIK",
            "NMDAR",
            # 'GluT'
        ]
        # self.expCurrName = {
        #     'Kir':,
        #     'TWIK':,
        #     'NMDAR':'Data/VClamp40.1Stim.dat',
        #     'GluT':
        # }
        self.expIVName = {
            "Kir": "Data/KirIV.csv",
            "TWIK": "Data/TWIKIV.csv",
            "NMDAR": "Data/NMDARIV.csv",
            "GluT": "Data/GluTrans.csv",
        }
        self.name2NEURON = {
            "Kir": ("kir2", "mm"),
            "TWIK": ("kleak", "mm"),
            "NMDAR": ("setNMDAs", "pm"),
            "GluT": ("setGluTs", "pm"),
        }

    def initModel(self, channel):
        """Load the clamp-check hoc setup, then insert the given channel's NEURON mechanism (or point-process synapse) onto the soma and record its current, plus set up a time recording vector."""
        h.xopen("./neuronHoc/simpleCheck.hoc")
        nrnName, chantype = self.name2NEURON[channel]
        # define record for channel
        setattr(self, channel, h.Vector())

        # Check channel type
        if chantype == "mm":
            h.soma.insert(nrnName)
            getattr(self, channel).record(h.soma(0.5)._ref_ik)
        else:
            getattr(h, nrnName)(h.SectionList([h.soma]))
            if "NMDA" in nrnName:
                listName = "NMDAs"
                currName = "_ref_iNMDA"
            else:
                listName = "GluTs"
                currName = "_ref_iGluT"
            sChannel = list(getattr(h, listName))[-1]
            # print(sChannel)
            getattr(self, channel).record(getattr(sChannel, currName))

        self.time = h.Vector()
        self.time.record(h._ref_t)

    def plotiCurve(self, channel, expSet, mdlSet):
        """Plot experimental and modeled current traces for a channel and save the figure as a PDF."""
        plt.cla()
        plt.clf()
        plt.scatter(*expSet, label="experiment")
        plt.plot(*mdlSet, label="model")
        plt.legend()
        plt.savefig(os.path.join("../results/paperRes/", f"icurve{channel}.pdf"))

    def getModeliCurve(self, channel, run=False, mV=None):
        """Optionally run a voltage-clamp simulation at the given voltage, then return the recorded time and current traces for the channel."""
        if run:
            if mV != None:
                h.clampSwitch(0, mV)
                h.init()
                h.run()

        curr = np.array(getattr(self, channel))
        time = np.array(self.time)
        return time, curr

    def getExpiCurve(self, channel):
        """Load the experimental current-time trace for a channel from its .dat or .csv file and return time and current arrays normalized by the peak absolute current."""
        fName = self.expIVName[channel]
        ext = fName.split(".")[-1]
        time = []
        curr = []
        if ext == "dat":
            f1 = h.File()
            f1.ropen(fName)
            while not f1.eof():
                time.append(f1.scanvar())
                curr.append(f1.scanvar())
            f1.close()
        elif ext == "csv":
            expData = pd.read_csv(fName)
            if "time" in expData.columns:
                time = expData["time"].tolist()
            else:
                print(f"no file {fName}")
                return None, None
            curr = expData["curr"].tolist()
        else:
            print(f"no file {fName}")

        curr = np.array(curr) / max(curr, key=abs)
        return np.array(time), curr

    def plotIVCurve(self, channel, expVolt, expCurr, mdlCurr):
        """Plot the experimental and modeled I-V curves for a channel and save the figure as a PDF."""
        plt.cla()
        plt.clf()
        plt.figure(figsize=gl.figsize_panel)
        plt.subplots_adjust(left=0.2)
        plt.scatter(expVolt, expCurr, label="Experiment", color=gl.exp_color)
        plt.plot(expVolt, mdlCurr, label="Model", color="black")
        plt.legend()
        plt.xlabel(gl.volt)
        plt.ylabel("Normalized Current")
        plt.savefig(os.path.join("../results/paperRes/", f"IVCurve{channel}.pdf"))

    def getModelIVPoint(self, mV, channel):
        """Run a voltage-clamp simulation at the given voltage and return the peak (largest-magnitude) current from the second half of the recorded trace, i.e. after the initial transient."""
        h.clampSwitch(0, mV)
        h.init()
        h.run()
        return max(
            np.array(getattr(self, channel))[int(len(getattr(self, channel)) // 2) :],
            key=abs,
        )

    def getExpIVCurve(self, channel):
        """Load the experimental I-V data for a channel from its .dat or .csv file and return voltage and current arrays normalized by the current at self.index."""
        fName = self.expIVName[channel]
        ext = fName.split(".")[-1]
        volt = []
        curr = []
        if ext == "dat":
            f1 = h.File()
            f1.ropen(fName)
            while not f1.eof():
                volt.append(f1.scanvar())
                curr.append(f1.scanvar())
            f1.close()
        elif ext == "csv":
            expData = pd.read_csv(fName)
            volt = expData["volt"].tolist()
            curr = expData["curr"].tolist()
        else:
            print(f"no file {fName}")

        curr = np.array(curr) / abs(curr[self.index])
        return np.array(volt), curr

    def IVCurve(self, channel, vStep=None):
        """For each experimental voltage step, run the model at that clamp voltage and record the model's I-V response; at the designated vStep also plot the raw current trace comparison; normalize and return the experimental and modeled I-V curves."""
        try:
            expVolt, expCurr = self.getExpIVCurve(channel)
        except FileNotFoundError:
            print(f"no iv data file in Data; skipped {channel}")
            return None
        mdlCurr = []
        for v in expVolt:
            self.initModel(channel)
            mdlCurr.append(self.getModelIVPoint(v, channel))
            print(v)
            if v == vStep:
                expSet = self.getExpiCurve(channel)
                mdlSet = self.getModeliCurve(channel)
                self.plotiCurve(channel, expSet, mdlSet)
        mdlCurr = np.array(mdlCurr) / abs(mdlCurr[self.index])
        return expVolt, expCurr, mdlCurr


if __name__ == "__main__":
    # gabaMdl = fitModelCurve("GABA")
    # gabaMdl.optimize()

    chans = calibrateChannel()
    for channel in chans.channelName:
        chans.index = 0
        print(channel)
        resSet = chans.IVCurve(channel, vStep=-80.05684057174929)
        if resSet is not None:
            chans.plotIVCurve(channel, *resSet)
