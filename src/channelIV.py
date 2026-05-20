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
        h.load_file("stdgui.hoc")
        self.cellType = self.cellDict(cellType)
        self.createCell()
        self.createSynapse(self.cellType.Name)

    def optimize(self):
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
        boundsList = []
        for parm in self.cellType.Parms:
            boundsList.append((0, None))
        return boundsList

    def lossFunction(self, x):
        expData, expTime = np.array(self.readData())
        self.adjustParms(x)
        self.setupSim(max(expTime) + 1, max(expTime))
        self.runSim()
        simData = np.array(self.adjustSimData(expTime))
        if self.plotInterim:
            self.plot(expTime, expData, simData)
        return sum((simData - expData) ** 2)

    def plot(self, t, exp, sim):
        plt.cla()
        plt.clf()
        plt.scatter(t, exp)
        plt.scatter(t, sim)
        plt.show()

    def adjustSimData(self, expTime):
        simData = []
        for t, current in zip(self.simTime, self.simData):
            if int(t) in expTime:
                print(t)
                simData.append(current)

        return simData

    def setupSim(self, stimStart, tstop):
        h.dt = 0.1
        self.stim = stimStart
        h.tstop = tstop
        h.finitialize()
        h.frecord_init()

    def runSim(self):
        h.run()

    def readData(self):
        expData = pd.read_csv(self.cellType.File)
        return expData["i"] / expData["i"].max(), expData["t"]

    def adjustParms(self, x):
        for i, parm in enumerate(x):
            setattr(self.synSoma, self.cellType.Parms[i], parm)

    def createCell(self):
        self.soma = h.Section(name="soma")
        self.soma.insert("pas")

    def createSynapse(self, synName):
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
        self.cellType.initParms = []
        for parm in self.cellType.Parms:
            initParm = getattr(self.synSoma, parm)
            self.cellType.initParms.append(initParm)


class calibrateChannel:
    def __init__(self):
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
            "TWIK": "Data/twikIV.csv",
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
        plt.cla()
        plt.clf()
        plt.scatter(*expSet, label="experiment")
        plt.plot(*mdlSet, label="model")
        plt.legend()
        plt.savefig(os.path.join("../results/paperRes/", f"icurve{channel}.pdf"))

    def getModeliCurve(self, channel, run=False, mV=None):
        if run:
            if mV != None:
                h.clampSwitch(0, mV)
                h.init()
                h.run()

        curr = np.array(getattr(self, channel))
        time = np.array(self.time)
        return time, curr

    def getExpiCurve(self, channel):
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
        plt.cla()
        plt.clf()
        plt.figure(figsize=gl.figsize_panel)
        plt.scatter(expVolt, expCurr, label="Experiment", color=gl.exp_color)
        plt.plot(expVolt, mdlCurr, label="Model", color="black")
        plt.legend()
        plt.xlabel(gl.volt)
        plt.ylabel("Normalized Current")
        plt.savefig(os.path.join("../results/paperRes/", f"IVCurve{channel}.pdf"))

    def getModelIVPoint(self, mV, channel):
        h.clampSwitch(0, mV)
        h.init()
        h.run()
        return max(
            np.array(getattr(self, channel))[int(len(getattr(self, channel)) // 2) :],
            key=abs,
        )

    def getExpIVCurve(self, channel):
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
