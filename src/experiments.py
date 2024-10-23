from mpi4py import MPI
import pickle
from astrocyte import PAPModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import os
import utils
import numpy as np
from utils import *
from textSDIO import *
from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
from math import floor
import glob
from matplotlib.ticker import MaxNLocator
from scipy.stats import f_oneway, ttest_rel
import json
import pandas as pd

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class plotFigures:
    colorDict = {
        "NMDAR": "steelblue",
        "GABAR": "purple",
        "GluT": "lightblue",
        "iK": "orange",
        "Soma": "deepskyblue",
        "PAP": "forestgreen",
        "fluor": "red",
        "Na": "gold",
        "Cl": "chocolate",
    }
    def saveSourceData(self,dataDict):
        with open(
                os.path.join("../results/paperRes", f"SourceData{self.tag}"),
                "w",
        ) as ofile:
            json.dump(dataDict, ofile)
        
        
    def returnColor(self, key):
        for typeName in self.colorDict.keys():
            if typeName in key:
                return self.colorDict[typeName]
        else:
            eMessage(f"Color not found for {key}")

    def plotPAPs(self):
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "readHoc": True,
                "Glu": True,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "naleak": self.leak,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "seed": self.seed,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        if self.stimCount > 1:
            cells.multiSpike(
                number=self.stimCount, freq=self.freq, Ko=self.ko, video=True
            )
        else:
            cells.setK(Ko=self.ko, delay=0)
        cells.run(video=True)

    def plotMorphProperties(self):
        funcArgs = []
        funcArgs.append(
            {
                "ComplexMorph": True,
                "readHoc": True,
                "seed": self.seed,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.plotMorphParms()

    def plotMergeSeries(self, AllCells):
        for attr in ["KoPAP", "vPAP"]:
            plt.cla()
            plt.clf()
            fig, ax = plt.subplots()
            ax.set_xlabel("time (ms)")
            if attr == "vPAP":
                ax.set_ylabel("Voltage (mV)")
                ax.set_ylim((-100, 0))
            else:
                ax.set_ylabel("[K] (mM)")
                ax.set_ylim((0, 15))
            for cells in AllCells:
                for cell in cells:
                    initStep = int(cell.initTstop / cell.dt)
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(getattr(cell, attr))[initStep:],
                        label=f"{cell.Ko:.3f}",
                    )
            ax.legend()
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"{attr}Merge{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                )
            )

    def plotIKSeries(self, AllCells, zoom=False, setyLim=None,setKoylim=False,setekylim=False):
        for cells in AllCells:
            for cell in cells:
                if (
                    not zoom
                    and cell.multiple == self.optNMDAR
                    and cell.GENEDict["kir2"] == self.optKir
                ) or (not zoom and cell.multiple == 0):
                    tmpTag = self.tag
                    self.tag += "_zoom"
                    self.plotIKSeries([[cell]], zoom=True)
                    self.tag = tmpTag

                initStep = int((cell.initTstop - 10) / cell.dt)
                fig, ax = plt.subplots()
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.KoSoma)[initStep:],
                    label="Soma",
                    color=self.returnColor("Soma"),
                )
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.KoPAP)[initStep:],
                    label=f"PAP",
                    color=self.returnColor("PAP"),
                )
                for x in list(range(150, 251, 10))[: self.stimCount]:
                    ax.arrow(
                        x,
                        0.5,
                        0,
                        -0.5,
                        color="black",
                        width=0.001,
                        head_width=0.4,
                        head_length=0.2,
                        length_includes_head=True,
                    )
                if setKoylim:
                    ax.set_ylim((0, 50))
                else:
                    ax.set_ylim((0, 9))
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("extracellular [K] (mM)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()

                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 20))

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"KoCon{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                fig, ax = plt.subplots()
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.NaoPAP)[initStep:],
                    label=f"PAP Na",
                    color=self.returnColor("Na"),
                )
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.CloPAP)[initStep:],
                    label=f"PAP Cl",
                    color=self.returnColor("Cl"),
                )
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("Conc. (mM)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 20))
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"NaCon{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                plt.cla()
                plt.clf()
                fig, ax = plt.subplots()

                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.vPAP)[initStep:],
                    label="PAP Vm",
                    color=self.returnColor("PAP"),
                )
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.ekPAP)[initStep:],
                    label="PAP eK",
                    color=self.returnColor("PAP"),
                    linestyle="--",
                )
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.fluorVPAP)[initStep:],
                    label="PAP fluor",
                    color=self.returnColor("fluor"),
                    linestyle='-.',
                )
                if not zoom:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.vSoma)[initStep:],
                        label="Soma Vm",
                        color=self.returnColor("Soma"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.ekSoma)[initStep:],
                        label="Soma eK",
                        color=self.returnColor("Soma"),
                        linestyle="--",
                    )
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("Voltage (mV)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
                if setekylim:
                    ax.set_ylim((-90,-10))

                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 20))

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"ekPlot{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                plt.cla()
                plt.clf()
                fig, ax = plt.subplots()

                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.iKPAP)[initStep:],
                    label="iK",
                    color=self.returnColor("iK"),
                )
                if hasattr(cell, "iNaPAP"):
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iNaPAP)[initStep:],
                        label="iNa",
                        color=self.returnColor("Na"),
                    )
                if hasattr(cell, "iClPAP"):
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iClPAP)[initStep:],
                        label="iCl",
                        color=self.returnColor("Cl"),
                    )
                if hasattr(cell, "iNMDA") and self.NMDAR:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iNMDA)[initStep:],
                        label="iNMDA",
                        color=self.returnColor("NMDAR"),
                    )
                if hasattr(cell, "iGABA") and self.GABAR:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGABA)[initStep:],
                        label="iGABAa",
                        color=self.returnColor("GABAR"),
                    )
                if hasattr(cell, "iGluT") and self.GluT:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluT)[initStep:],
                        label="iGluT",
                        color=self.returnColor("GluT"),
                    )
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("Currents at PAP (pA)")
                if setyLim != None:
                    ax.set_ylim(setyLim)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if zoom or self.stimCount > 1:
                    ax.legend(loc="lower left")
                else:
                    ax.legend(loc="lower right")

                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 20))

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"ikPlot{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                plt.cla()
                plt.clf()
                fig, ax = plt.subplots()
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.iKSoma)[initStep:],
                    label="ik",
                    color=self.returnColor("iK"),
                )
                if hasattr(cell, "iNaSoma"):
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iNaSoma)[initStep:],
                        label="iNa",
                        color=self.returnColor("Na"),
                    )
                if hasattr(cell, "iClSoma"):
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iClSoma)[initStep:],
                        label="iCl",
                        color=self.returnColor("Cl"),
                    )
                if hasattr(cell, "iGluTSoma") and self.GluT:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluTSoma)[initStep:],
                        label="iGluT",
                        color=self.returnColor("GluT"),
                    )
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("Currents at Soma (pA)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc="lower right")

                ax3 = ax.inset_axes(
                    [0.75, 0.55, 0.2, 0.2]
                )  # Define the position and size of the new subplot
                ax3.plot(
                    list(cell.time)[initStep:],
                    list(cell.vSoma)[initStep:],
                    label="Soma",
                    color=self.returnColor("Soma"),
                )
                ax3.set_ylabel("Voltage")
                ax3.set_ylim((-90, -70))
                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 30))

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"iSomaPlot{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                plt.close("all")

    def plotHeatmap(self, results, tag="", divedend=1,Kir=True):
        plt.cla()
        plt.clf()
        if Kir:
            imArray = np.zeros(
                (
                    int(self.KirMax / self.KirStep) + 1,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                )
            )
        else:
            imArray = np.zeros((5,5))

        for res in results:
            if Kir:
                imArray[
                    int(res[0].GENEDict["kir2"] / self.KirStep),
                    int(res[0].comparecount / self.channelCompareStep),
                ] += (
                    max(res[0].vPAP) - res[0].RMP
                )
            else:
                imArray[
                    int(res[0].PAPLen / 0.3) - 1,
                    int(res[0].comparecount / self.channelCompareStep) - 1,
                ] += (
                    max(res[0].vPAP) - res[0].RMP
                )
                
        cmap = "magma"

        imArray /= divedend
        plt.imshow(
            imArray,
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
        )
        if Kir:
            chanStart = 0
            addChan = 1
        else:
            chanStart = 1
            addChan = 0
        plt.xticks(
            range(0,int(self.channelCompareMax / self.channelCompareStep) + addChan),
            np.arange(chanStart, int(self.channelCompareMax / self.channelCompareStep) + 1, 1)
            * self.channelCompareStep,
        )
        if Kir:
            plt.yticks(
                range(int(self.KirMax / self.KirStep) + 1),
                np.arange(0, int(self.KirMax / self.KirStep) + 1, 1) * self.KirStep,
            )
            plt.ylabel("# of Kir Channels")
        else:
            plt.yticks(
                range(0,5),
                [ f'{i:.2f}' for i in np.arange(0.3,1.6,0.3)],
            )
            plt.ylabel("Affected PAP length (um)")
        if self.NMDAR:
            plt.xlabel("# of NMDAR Channels")
        elif self.GluT:
            plt.xlabel("Multiple of estimated GluT density")
        plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 50, 10), extend="max")
        plt.clim((0, 50))
        plt.savefig(os.path.join("../results/paperRes", f"FullComparison{tag}.pdf"))
        plt.cla()
        plt.clf()
        if Kir:
            imArray = np.zeros(
                (
                    int(self.KirMax / self.KirStep) + 1,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                )
            )

            for res in results:
                imArray[
                    int(res[0].GENEDict["kir2"] / self.KirStep),
                    int(res[0].comparecount / self.channelCompareStep),
                ] += res[0].RMP
            imArray /= divedend
            plt.imshow(
                imArray,
                cmap=cmap,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            plt.xticks(
                range(int(self.channelCompareMax / self.channelCompareStep) + 1),
                np.arange(0, int(self.channelCompareMax / self.channelCompareStep) + 1, 1)
                * self.channelCompareStep,
            )
            plt.yticks(
                range(int(self.KirMax / self.KirStep) + 1),
                np.arange(0, int(self.KirMax / self.KirStep) + 1, 1) * self.KirStep,
            )
            plt.ylabel("# of Kir Channels")
            if self.NMDAR:
                plt.xlabel("# of NMDAR Channels")
            elif self.GluT:
                plt.xlabel("# of GluT Channels")
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(-100, -60, 10), extend="max")
            plt.clim((-100, -60))
            plt.savefig(os.path.join("../results/paperRes", f"FullRMP{tag}.pdf"))
            plt.cla()
            plt.clf()

            imArray = np.zeros(
                (
                    int(self.KirMax / self.KirStep) + 1,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                )
            )

            for res in results:
                imArray[
                    int(res[0].GENEDict["kir2"] / self.KirStep),
                    int(res[0].comparecount / self.channelCompareStep),
                ] += (
                    max(res[0].vSoma) - res[0].RMP
                )
            imArray /= divedend
            plt.imshow(
                imArray,
                cmap=cmap,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            plt.xticks(
                range(int(self.channelCompareMax / self.channelCompareStep) + 1),
                np.arange(0, int(self.channelCompareMax / self.channelCompareStep) + 1, 1)
                * self.channelCompareStep,
            )
            plt.yticks(
                range(int(self.KirMax / self.KirStep) + 1),
                np.arange(0, int(self.KirMax / self.KirStep) + 1, 1) * self.KirStep,
            )
            plt.ylabel("# of Kir Channels")
            if self.NMDAR:
                plt.xlabel("# of NMDAR Channels")
            elif self.GluT:
                plt.xlabel("GluT Channels")
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 10, 1), extend="max")
            plt.clim((0, 10))

            plt.savefig(os.path.join("../results/paperRes", f"FullSoma{tag}.pdf"))


class procedure(plotFigures):
    leak = 3e5
    optKir = 120
    optNMDAR = 12
    optGluT = 1
    channelCompareMax = 50
    channelCompareStep = 5
    KirMax = 400
    KirStep = 40
    seed = int()
    ko = float()
    tag = str()
    OE = False
    NMDAR = True
    GABAR = True
    GluT = False
    GluStim = True
    KStim = True
    stimdelay = 0
    dt = 0.1
    PAPCount = 1
    stimCount = 1
    freq = 100
    ek = None
    PAPLen = 0.3
    peakLen = None

    def __init__(self, seed, ko):
        self.seed = seed
        self.ko = ko
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"

    def addChannelTag(self):
        self.tag = ""
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"

        if self.GluT:
            self.tag += "_Glu"
        if self.NMDAR:
            self.tag += "_NMDAR"
        if not self.GluStim:
            self.tag += "_NoGlu"
        if not self.KStim:
            self.tag += "_NoK"
        if self.stimdelay > 0:
            self.tag += f"_Delay{self.stimdelay}"
        if self.PAPCount > 1:
            self.tag += f"_PAPx{self.PAPCount}"
        if self.stimCount > 1:
            self.tag += f"_multiSpikex{self.stimCount}"
        if self.ek != None:
            self.tag += f"_ek{self.ek}"
        if self.PAPLen > 0.3:
            self.tag += f"_spillover"
        if self.freq != 100:
            self.tag += f"_freq{self.freq}Hz"
        # print(f'{self.GluT=}')
        # print(f'{self.NMDAR=}')

    def nernst(self, k, kin):
        T = 273.16 + 34  # kelvin
        R = 8.3145  # J/K
        z = 1  # k+
        F = 96485.3  # Coulomb
        return (1e3) * R * T / F / z * np.log(k / kin)

    def nernstINV(self, ek, kin):
        T = 273.16 + 34  # kelvin
        R = 8.3145  # J/K
        z = 1  # k+
        F = 96485.3  # Coulomb
        return np.exp(ek * F * z / ((1e3) * R * T)) * kin

    def multiChannel(self, itr=100):
        dList = []
        for i in range(1, itr + 1):
            sim = PAPModel(40, multiple=i, mode=0)
            sim.run()
            dList.append(plot(".") - sim.getRMP())
        with open(os.path.join("intermediaryData", f"dList.pickle"), "wb") as handle:
            pickle.dump(dList, handle, protocol=pickle.HIGHEST_PROTOCOL)
        plt.cla()
        plt.clf()
        plt.scatter(range(1, itr + 1), dList, color="black")
        plt.savefig(os.path.join("../results/paperRes", "patchXDepolar.pdf"))

    def multiDistance(self, x, read=False):
        somaSize, bLen, bWid, PAPWid, bNum = x
        dList = []
        cList = []
        vList = []
        if read:
            with openf(
                os.path.join("intermediaryData", "ballStick.pickle"), "rb"
            ) as handle:
                dList, cList, vList = pickle.load(handle)
        else:
            vSomaList = []
            vPAPList = []
            if self.parallel:
                # Calculate the number of iterations for all parm sets
                iterations = comm.bcast(get_iter(501, 50, 101, 10), root=0)

                # # Adjust the range for the last process
                # Individual list for each rank
                vSoma = []
                vPAP = []
                d = []
                c = []

                comm.Barrier()
                funcArgs = []
                funcArgs.append(
                    {
                        "currentClamp": 20,
                        "bWid": bWid,
                        "somaSize": somaSize,
                        "mode": 0,
                        "bNum": int(bNum),
                        "PAPWid": PAPWid,
                        "Ko": self.ko,
                        "kir2": 0,
                        "seed": self.seed,
                    }
                )
                # make sure that funcParms is in the correct order of whatever iterations spits out
                # results are collected only on rank 0
                results = parallizeFor(
                    iterations,
                    [PAPModel],
                    funcArgs,
                    ["bLen", "multiple"],
                    [["initialize", "run"]],
                    [[{}, {}]],
                )
                comm.Barrier()
                if rank == 0:
                    with open(
                        os.path.join("intermediaryData", "resultsParallel.pickle"), "wb"
                    ) as handle:
                        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    for index, res in enumerate(results):
                        i, j = iterations[index]
                        dList.append(i)
                        cList.append(j)
                        for i, rIndi in enumerate(res):
                            if i == 0:
                                RMP = rIndi.getRMP()
                            vSomaList.append(max(np.array(rIndi.vSoma)))
                            vPAPList.append(max(np.array(rIndi.vPAP)))
                            # if index == 0:
                            #     print(plt.plot(np.array(rIndi.vPAP)))
                            #     plt.show()

            else:
                for i in range(1, 101, 10):
                    for j in range(100, 101, 1):
                        dList.append(i)
                        cList.append(j)
                        sim = PAPModel(
                            multiple=j,
                            bLen=i,
                            currentClamp=1,
                            bWid=bWid,
                            somaSize=somaSize,
                            mode=0,
                            bNum=int(bNum),
                            PAPWid=PAPWid,
                            Ko=self.ko,
                            Glu=True,
                        )
                        sim.run()
                        vSomaList.append(
                            max(np.array(sim.vSoma) - sim.getRMP(), key=abs)
                        )
                        sim = PAPModel(
                            multiple=j,
                            bLen=i,
                            bWid=bWid,
                            currentClamp=1,
                            somaSize=somaSize,
                            mode=0,
                            bNum=int(bNum),
                            PAPWid=PAPWid,
                            Ko=self.ko,
                            Glu=True,
                        )
                        sim.run()
                        vPAPList.append(max(np.array(sim.vPAP) - sim.getRMP(), key=abs))
            vList = [vSomaList, vPAPList]

            if not self.parallel or rank == 0:
                with open(
                    os.path.join("intermediaryData", f"ballStick.pickle"), "wb"
                ) as handle:
                    pickle.dump(
                        [dList, cList, vList], handle, protocol=pickle.HIGHEST_PROTOCOL
                    )

        # Create a figure and a 3D axis
        if not self.parallel or rank == 0:
            for i, v in enumerate(vList):
                fig = plt.figure()
                ax = plt.axes(projection="3d")

                # Create the scatter plot
                ax.scatter3D(dList, cList, v, c=v, cmap="viridis")

                # Set labels and title
                ax.set_xlabel("distance")
                ax.set_ylabel("channel Count")
                if i == 0:
                    name = "soma"
                else:
                    name = "PAP"
                ax.set_zlabel(f"Voltage Change{name}")

                # Show the plot
                j = ""
                while os.path.isfile(f"./3Dplot{name}{j}.pdf"):
                    if j == "":
                        j = 1
                    else:
                        j += 1

                plt.savefig(
                    os.path.join("../results/paperRes", f"./3Dplot{name}{j}.pdf")
                )

    def readIterationRi(self, all_file_names, dName="../results/paperRes"):
        pattern = "RiRes*"
        # Construct the full path
        full_path = os.path.join(dName, pattern)

        # Use glob to find files matching the pattern
        matched_files = glob.glob(full_path)
        extracted_file_names = []

        for file in matched_files:
            file_name = os.path.splitext(os.path.basename(file))[0][len("RiRes") :]
            extracted_file_names.append(file_name)

        extracted_set = set(extracted_file_names)
        all_set = set(all_file_names)

        # Find the elements that are unique to each set
        unique_all = all_set - extracted_set

        return list(unique_all)

    def measureRi(self):
        funcArgs = []
        funcArgs.append(
            {
                "ComplexMorph": True,
                "readHoc": True,
                "Glu": False,
                "kir2": self.optKir,
                "clleak": 0,
                "naleak": self.leak,
                "multiple": 0,
                "dt": 0.1,
                "seed": self.seed,
            }
        )
        if size == 1:
            sim = PAPModel(**funcArgs[-1])
            sim.initialize()
            sim.measureRiAll()
        else:
            comm.Barrier()
            remaining_iteration = None
            if rank == 0:
                sim = PAPModel(**funcArgs[-1])
                iterations = [sec.hname() for sec in h.allsec()]
                remaining_iteration = self.readIterationRi(
                    iterations,
                )
                remaining_iteration = [[sec] for sec in remaining_iteration]

            remaining_iteration = comm.bcast(
                remaining_iteration,
                root=0,
            )
            ccList = comm.bcast(
                ["RiSec"],
                root=0,
            )
            comm.Barrier()
            results = parallizeFor(
                remaining_iteration,
                [PAPModel],
                funcArgs,
                ccList,
                [
                    ["initialize", "measureRiAll"],
                ],
                [[{}, {"parallel": True}]],
            )
            comm.Barrier()
            if rank == 0:
                merged_dict = {}
                for sName in iterations:
                    filePath = os.path.join("../results/paperRes", f"RiRes{sName}.json")
                    if os.path.isfile(filePath):
                        with open(filePath, "r") as rfile:
                            try:
                                RiDict = json.load(rfile)
                            except json.decoder.JSONDecodeError as e:
                                print(filePath)
                        merged_dict.update(RiDict)
                with open(
                    os.path.join("../results/paperRes", "RiRes.json"), "w"
                ) as ofile:
                    json.dump(merged_dict, ofile)

                sim = PAPModel(**funcArgs[-1])
                sim.mapRi(merged_dict)
            comm.Barrier()

    def SomaVC(self):
        vClampList = np.arange(-95, -59, 5)
        vSomaList = []
        funcArgs = []
        plt.cla()
        plt.clf()
        funcArgs.append(
            {
                "somaCheck": True,
                "mode": 2,
                "ComplexMorph": True,
                "readHoc": True,
                "dt": self.dt,
                "naleak": self.leak,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "seed": self.seed,
                "GluTrans": self.optGluT,
            }
        )
        simSoma = PAPModel(**funcArgs[-1])
        simSoma.initialize()

    def branchAttenuation(self, alterDist=False):
        self.addChannelTag()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "readHoc": True,
                "Glu": self.GluStim,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "naleak": self.leak,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
                "seed": self.seed,
                "PAPCount": self.PAPCount,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        if self.stimCount > 1:
            cells.multiSpike(number=self.stimCount, freq=self.freq, Ko=self.ko)
        else:
            cells.setK(Ko=self.ko, delay=0)
        cells.run()
        initStep = int(cells.initTstop / cells.dt)
        cells = cells.copyAttr()

        timeVoltageArray = list()
        for x in cells.branchAtten:
            # print(list(x))
            coord = list(x)[initStep:]
            timeVoltageArray.append(coord)
        timeVoltageArray = np.array(timeVoltageArray).T
        timeVoltageArray -= timeVoltageArray[-1][-1]
        # print(timeVoltageArray)
        # Plot the array using a heatmap
        plt.imshow(timeVoltageArray, cmap="magma", interpolation="none", aspect="auto")
        plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 20, 2), extend="max")
        plt.clim((0, 20))
        plt.xlabel("normalized distance")
        plt.ylabel("Time (ms)")
        plt.xticks(
            range(0, 11, 2), [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )  # float point generated by np.linspace
        plt.yticks(
            range(0, len(list(cells.branchAtten[0])[initStep:]) + 1, 100),
            np.arange(
                0,
                int(cells.time[-1] - cells.time[initStep]) + 1,
                100 * cells.dt,
                dtype=int,
            ),
        )

        # Show the plot
        if not alterDist:
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"branchAtten_{self.tag}_Original.pdf",
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"branchAtten_{self.tag}.pdf",
                )
            )

        # plt.clf()
        # plt.cla()
        # plt.plot(cells.vPAP)
        # plt.show()

    def alteredDist(self):
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "readHoc": True,
                "Glu": True,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "naleak": self.leak,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "seed": self.seed,
            }
        )
        cellComparison = []
        ratioList = [0] + list(np.logspace(-3, 1))
        for ratio in ratioList:
            cells = PAPModel(**funcArgs[-1])
            cells.channelDist(kir2=ratio)
            cells.initialize()
            cells.setK(Ko=self.ko)
            cells.run()
            cellComparison.append(cells.copyAttr())
            print(f"complete{ratio}")
        for i, cell in enumerate(cellComparison):
            plt.plot(cell.time, cell.vSoma, label=f"ratio to PAP:{ratioList[i]}")
            # plt.legend()
            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"RatioComp{self.optKir}_{self.optNMDAR}.pdf"
                )
            )

    def kvPhasePlane(self):
        self.addChannelTag()
        AllCells = []
        for kircount in [self.optKir, 400]:
            for chanCount in [self.optNMDAR, 25]:
                funcArgs = []
                funcArgs.append(
                    {
                        "mode": 0,
                        "ComplexMorph": True,
                        "bNum": 1,
                        "readHoc": True,
                        "Glu": True,
                        "dt": 0.01,
                        "naleak": self.leak,
                        "clleak": 0,
                        "seed": self.seed,
                        "PAPLen": self.PAPLen,
                    }
                )
                if self.GluT:
                    if self.NMDAR:
                        funcArgs[-1]["multiple"] = chanCount
                        funcArgs[-1]["GluTrans"] = 1
                        chanName = "GluT_NMDAR"
                    else:
                        funcArgs[-1]["multiple"] = 0
                        funcArgs[-1]["GluTrans"] = 1
                        chanName = "GluTrans"
                else:
                    if self.NMDAR:
                        funcArgs[-1]["multiple"] = chanCount
                        funcArgs[-1]["GluTrans"] = 0
                        chanName = "NMDAR"
                    else:
                        funcArgs[-1]["multiple"] = 0
                        funcArgs[-1]["GluTrans"] = 0
                        chanName = ""

                iterations = comm.bcast(
                    [
                        (kircount, conc)
                        for conc in np.array([0,0.5,1.0,5.0,10.0])
                    ],
                    root=0,
                )
                ccList = comm.bcast(["kir2", "Ko"], root=0)
                comm.Barrier()
                if self.KStim and self.stimCount == 1:
                    results = parallizeFor(
                        iterations,
                        [PAPModel],
                        funcArgs,
                        ccList,
                        [
                            ["initialize", "setK", "run"],
                        ],
                        [[{}, {}, {}]],
                    )
                elif self.stimCount > 1:
                    results = parallizeFor(
                        iterations,
                        [PAPModel],
                        funcArgs,
                        ccList,
                        [["initialize", "multiSpike", "run"]],
                        [[{}, {"number": self.stimCount, "freq": self.freq}, {}]],
                    )

                else:
                    plt.cla()
                    plt.clf()
                    results = parallizeFor(
                        iterations,
                        [PAPModel, PAPModel],
                        funcArgs,
                        ccList,
                        [["initialize", "run"]],
                        [[{}, {}]],
                    )
                comm.Barrier()
                if rank == 0:
                    # self.plotMergeSeries(results)
                    plt.cla()
                    plt.clf()
                    for j in range(len(results[0])):
                        for i, cell in enumerate(results):
                            cell = cell[j]
                            initStep = int(cell.initTstop / cell.dt) - 200
                            if cell.Ko == 0.5:
                                color = 'r'
                                z = len(results) + 1
                            else:
                                color = cm.summer(i / len(results))
                                z = i
                            plt.plot(
                                list(cell.KoPAP)[initStep:],
                                list(cell.vPAP)[initStep:],
                                label=f"{cell.Ko:.3f}",
                                color=color,
                                zorder=z,
                            )
                            if i == len(results) - 1:
                                x = np.linspace(
                                    min(list(cell.KoPAP)[initStep:]),
                                    max(list(cell.KoPAP)[initStep:]),
                                )
                                plt.plot(
                                    x,
                                    self.nernst(x, cell.kin),
                                    label="eK",
                                    color="black",
                                    linestyle="--",
                                )
                                if (
                                    kircount == self.optKir
                                    and chanCount == self.optNMDAR
                                ):
                                    plt.legend()
                                plt.ylabel("Voltage (mV)")
                                plt.xlabel("[K]o (mM)")
                                if self.PAPLen <= 0.3:
                                    plt.ylim((-90, -50))
                                plt.xlim((2, 14))
                                plt.savefig(
                                    os.path.join(
                                        "../results/paperRes",
                                        f"phasePlane_{kircount}{chanName}{chanCount}{self.tag}.pdf",
                                    )
                                )
                                plt.cla()
                                plt.clf()

                plt.close("all")

    def ekComp(self):
        self.addChannelTag()
        AllCells = []
        funcArgs = []
        koList = []
        for ek in np.arange(-95, -39, 5):
            funcArgs.append(
                {
                    "mode": 0,
                    "ComplexMorph": True,
                    "bNum": 1,
                    "readHoc": True,
                    "Glu": True,
                    "kir2": self.optKir,
                    "clleak": 0,
                    "naleak": self.leak,
                    "dt": self.dt,
                    "seed": self.seed,
                    "stimdelay": 20 * ms,
                }
            )
            if self.NMDAR:
                funcArgs[-1]["multiple"] = self.optNMDAR
            else:
                funcArgs[-1]["multiple"] = 0
            if self.GluT:
                funcArgs[-1]["GluTrans"] = self.optGluT

            ko = self.nernstINV(ek, 80)  # 80 defined in neuron astrocyte.hoc
            koList.append(ko)

            cells = PAPModel(**funcArgs[-1])
            cells.initialize()
            cells.run(koclamp=ko)
            cells = cells.copyAttr()
            cells.ek = ek

            if size > 1:
                AllCells = comm.gather(cells, root=0)
            else:
                AllCells.append([cells])

        ekList = []
        depList = []
        for i, cells in enumerate(AllCells):
            for cell in cells:
                ekList.append(cell.ek)
                initStep = int((cell.initTstop + 10) / cell.dt)
                depList.append(
                    max(list(cell.vPAP)[initStep:]) - list(cell.vPAP)[initStep]
                )  # 3 ms to stablize
                plt.plot(
                    list(cell.time)[initStep:],
                    list(cell.vPAP)[initStep:] - cell.ek,
                    label=f"ek {cell.ek}",
                    color=cm.magma(i / len(AllCells)),
                )

        plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("Vm - ek (mV)")
        plt.savefig(os.path.join("../results/paperRes", "ekDepolarcompTraces.pdf"))
        for cells in AllCells:
            for cell in cells:
                plt.cla()
                plt.clf()
                plt.plot(
                    list(cell.time)[initStep:],
                    list(cell.iKPAP)[initStep:],
                    label=f"iK",
                    color=self.returnColor("iK"),
                )
                plt.plot(
                    list(cell.time)[initStep:],
                    list(cell.iNMDA)[initStep:],
                    label=f"iNMDAR",
                    color=self.returnColor("NMDAR"),
                )
                if hasattr(cell, "iGluT"):
                    plt.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluT)[initStep:],
                        label=f"iGluT",
                        color=self.returnColor("GluT"),
                    )
                plt.legend()
                plt.xlabel("time (ms)")
                plt.ylabel("current (pA)")
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"ekDepolarcompCurrentTraces{cell.ek}.pdf",
                    )
                )

        plt.cla()
        plt.clf()
        plt.scatter(ekList, depList, color="black")
        plt.ylabel("Membrane potential change (mV)")
        plt.xlabel("ek (mV)")
        plt.savefig(os.path.join("../results/paperRes", "ekDepolarcomp.pdf"))

        plt.cla()
        plt.clf()
        plt.scatter(koList, depList, color="black")
        plt.ylabel("Membrane potential change (mV)")
        plt.xlabel("extracellular [K+] (mM)")
        plt.savefig(os.path.join("../results/paperRes", "ekKODepolarcomp.pdf"))
        print(koList)

    def KOComp(self, papCount=10, koCond=6):
        AllCells = []
        # single run
        for i in range(koCond):
            funcArgs = []
            # Order is important
            if i == 0:
                self.GluT = True
                self.NMDAR = True
                controlKir = self.optKir
                self.optKir = controlKir
                controlLeak = self.leak
                tmpdt = self.dt
            elif i < 3:
                self.GluT = True
                self.NMDAR = True
                if i == 1:
                    # Kir OE
                    self.optKir = controlKir * 6.0  # from experiment
                    self.dt *= 0.1
                    self.leak = controlLeak
                else:
                    self.leak = 8455
                    # match findings of Djukic et al. (2007) of -76.3 mV
                    self.dt = tmpdt
                    self.optKir = 0  # from experiment
            else:
                self.leak = controlLeak
                self.dt = tmpdt
                self.optKir = controlKir
                if i == 3:
                    # NMDARKO
                    self.GluT = True
                    self.NMDAR = False
                elif i == 4:
                    # NMDARKO
                    self.NMDAR = True
                    self.GluT = False
                elif i == 5:
                    # NMDARKO
                    self.NMDAR = False
                    self.GluT = False

            funcArgs.append(
                {
                    "mode": 0,
                    "ComplexMorph": True,
                    "bNum": 1,
                    "readHoc": True,
                    "Glu": True,
                    "kir2": self.optKir,
                    "clleak": 0,
                    "naleak": self.leak,
                    "dt": self.dt,
                }
            )
            if i == 1:
                krule = {'kuptake':True}
            elif i == 2:
                # nonspecific K+ block
                # funcArgs[-1]['twik'] = 0
                # funcArgs[-1]['kleak'] = 0
                krule = {'kblock':True}
            else:
                krule = {}
                
            if self.NMDAR:
                funcArgs[-1]["multiple"] = self.optNMDAR
            else:
                funcArgs[-1]["multiple"] = 0
            if self.GluT:
                funcArgs[-1]["GluTrans"] = self.optGluT

            comm.Barrier()
            if self.peakLen == None:
                self.peakLen = 1.47
            else:
                if rank == 0:
                    print(self.peakLen)
            iterations = comm.bcast(
                [(i, j) for j in [0.3, self.peakLen] for i in range(papCount)]
            )
            ccList = ["seed", "PAPLen"]
            # results are collected only on rank 0
            callMethods = [[]]
            callArgs = [[]]
            if self.KStim and self.stimCount == 1:
                callMethods[0] += ["initialize", "setK", "run"]
                callArgs[0] += [krule, {"Ko": self.ko}, {}]
            elif self.stimCount > 1:
                callMethods[0] += ["initialize", "multiSpike", "run"]
                callArgs[0] += [
                    krule,
                    {"number": self.stimCount, "Ko": self.ko, "freq": self.freq},
                    {},
                ]
            else:
                callMethods[0] += ["initialize", "run"]
                callArgs[0] += [{"kblock":kblock}, {}]

            results = parallizeFor(
                iterations, [PAPModel], funcArgs, ccList, callMethods, callArgs
            )

            comm.Barrier()
            if rank == 0:
                cells = results
                AllCells += cells

        if rank == 0:
            resMat = np.zeros((koCond * 2, papCount))
            for cells in AllCells:
                for cell in cells:
                    if cell.multiple > 0:
                        if cell.GENEDict["kir2"] == controlKir:
                            if "GluTrans" in cell.GENEDict.keys():
                                self.NMDAR = True
                                k = 0
                                self.addChannelTag()
                            else:
                                self.NMDAR = True
                                k = 4
                                self.addChannelTag()
                        elif cell.GENEDict["kir2"] > controlKir:
                            # Kir OE
                            k = 1
                            self.addChannelTag()
                        else:
                            # Kir inhibition
                            self.addChannelTag()
                            k = 2
                            plt.plot(cell.time,cell.vPAP)
                            plt.savefig('KO changes.pdf')
                    else:
                        # NMDAR KO
                        if "GluTrans" in cell.GENEDict.keys():
                            k = 3
                            self.NMDAR = False
                            self.GluT = True
                            self.addChannelTag()
                        else:
                            k = 5
                            self.NMDAR = False
                            self.GluT = False
                            self.addChannelTag()

                    if cell.PAPLen > 0.3:
                        k += koCond

                    # remove previous KOComp tag
                    self.tag = self.tag.split("_KOComp")[0]
                    self.tag += "_KOComp"
                    if cell.PAPLen > 0.3:
                        self.tag += f"spillover"
                    self.plotIKSeries([[cell]], setyLim=[-10, 10])
                    resMat[k][cell.seed] = max(cell.vPAP) - cell.RMP

            title = "One-way ANOVA "
            pvalDict = {}
            for i in list(range(0, koCond * 2, koCond)):
                ommit_cond = 1
                stat, pval = f_oneway(
                    *resMat[i : i + koCond - ommit_cond]
                )  # select based on control, OE, KD leave NMDAR out
                if i == koCond:
                    title += f"spillover p-value:{pval:.2E}"
                    key = "spillover"
                else:
                    title += f"constrained p-value:{pval:.2E}"
                    key = "confined"
                print(pval)
                if pval < 0.05:
                    if pval < 0.01:
                        if pval < 0.001:
                            pvalDict[key] = "***"
                        else:
                            pvalDict[key] = "**"
                    else:
                        pvalDict[key] = "*"
                else:
                    pvalDict[key] = "n.s."

            val_means = {"confined": [], "spillover": []}
            val_sd = {"confined": [], "spillover": []}
            category = [
                "Control",
                "Kir OE",
                "Kir Block",
                "NMDAR KO",
                "GluT KO",
                "NMDAR KO\nGluT KO",
            ]
            category = category[:koCond]
            val_test = {}
            for i in range(len(resMat)):
                if i < koCond:  # Number of KO conditions
                    dict_key = "confined"
                    val_test[category[i]] = ttest_rel(resMat[i], resMat[i + koCond])
                else:
                    dict_key = "spillover"
                if i in [0,koCond]:
                    for cond in [1,2]:
                        print(dict_key)
                        print(category[i % 6],category[(i+cond) % 6])
                        print(ttest_rel(resMat[i],resMat[i+cond]))

                val_means[dict_key].append(np.nanmean(resMat[i]))
                val_sd[dict_key].append(np.nanstd(resMat[i]))

            width = 0.25
            multiplier = 0
            x = np.arange(int(len(category) - 2))
            pattern = {"confined": "", "spillover": "|"}
            for k, v in val_means.items():
                offset = width * multiplier
                rects = plt.bar(
                    x + offset,
                    v[0:1] + v[3:],
                    width,
                    yerr=val_sd[k][0:1] + val_sd[k][3:],
                    label=f"{k}",
                    hatch=pattern[k],
                    edgecolor="black",
                )
                # ax.bar_label(rects,padding=3)
                multiplier += 1
            plt.xticks(x + width / len(val_means.keys()), category[0:1] + category[3:])
            plt.legend()
            plt.ylabel("Voltage (mV)")
            plt.ylim(0, 70)
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    "KO_GENE_Comparison.pdf",
                )
            )
            plt.cla()
            plt.clf()

            multiplier = 0
            fig, ax = plt.subplots(layout="constrained")
            fig.suptitle(title)
            color = ["orange", "darkorange", "gold", "orange", "orange", "orange"]
            for k, v in val_means.items():
                offset = width * multiplier
                rects = ax.bar(
                    x[:3] + offset,
                    v[:3],
                    width,
                    yerr=val_sd[k][:3],
                    label=f"{k}|{pvalDict[k]} ",
                    color=color[:3],
                    hatch=pattern[k],
                    edgecolor="black",
                )
                # ax.bar_label(rects,padding=3)
                multiplier += 1

            # get indexes
            iterations = np.concatenate(
                (np.logspace(-0.5, 1, num=19), np.array([self.PAPLen]))
            )
            iterations = np.sort(iterations)
            controlIndex = np.where(self.PAPLen == iterations)[0][
                0
            ]  # get index of PAPLen position in iterations
            try:
                maxIndex = np.where(self.peakLen == iterations)[0][
                    0
                ]  # get index of peakLen position in iterations
            except IndexError:
                maxIndex = (np.abs(iterations - self.peakLen)).argmin()

            ax.axhline(
                val_means["confined"][0],
                linestyle="--",
                c=cm.twilight(controlIndex / len(iterations)),
            )
            ax.axhline(
                val_means["spillover"][0],
                linestyle="--",
                c=cm.twilight(maxIndex / len(iterations)),
            )

            with open(
                os.path.join("../results/paperRes", "ttest_res.json"), "w"
            ) as ofile:
                json.dump(val_test, ofile)
            # for k,v in val_test.items():
            #     if v.pvalue < 0.05:
            #         index = category.index(k)
            ax.set_ylabel("Voltage (mV)")
            ax.set_ylim(0, 70)
            ax.set_xticks(x[:3] + width / len(val_means.keys()), category[:3])
            ax.legend(loc="upper left", ncols=2)
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"KO_maxDepolarComp_avg{papCount}_{self.tag}.pdf",
                )
            )

    def uptakeRatio(self):
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "readHoc": True,
                "Glu": False,
                "kir2": self.optKir,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        if self.NMDAR:
            funcArgs[-1]["multiple"] = self.optNMDAR
        else:
            funcArgs[-1]["multiple"] = 0
        if self.GluT:
            funcArgs[-1]["GluTrans"] = self.optGluT

        cells = PAPModel(**funcArgs[-1])
        cells.setTstop(500)
        cells.initialize()
        cells.setK(Ko=43, delay=0,dur=10)
        cells.run()
        cells = cells.copyAttr()
        initStep = int((cells.initTstop + 10) / cells.dt) + 1
        flux = np.array(list(cells.flux)[initStep:]) * -1
        kbath = np.array(list(cells.kbath)[initStep:]) * -1
        plt.plot(list(cells.time)[initStep:],np.divide(flux,kbath))
        plt.xlabel('time (ms)')
        plt.ylabel('Ratio of\ninflux / diffusion\nfor potassium')
        plt.savefig('fluxRatioOvertime.pdf')
        

            
    def singleRun(self, expOverlay=False,GluTime=True):
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "readHoc": True,
                "Glu": True,
                "GABA":False,
                "kir2": self.optKir,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )

        if self.OE:
            funcArgs[-1]["kir2"] = 400
            
        if funcArgs[-1]['kir2'] > 300: # to compensate for mathematical unstability
            funcArgs[-1]['dt'] *= 0.2
        if self.NMDAR:
            funcArgs[-1]["multiple"] = self.optNMDAR
        else:
            funcArgs[-1]["multiple"] = 0
        if self.GluT:
            funcArgs[-1]["GluTrans"] = self.optGluT

        cells = PAPModel(**funcArgs[-1])
        cells.setTstop(500)
        cells.initialize()
        if self.stimCount > 1:
            cells.multiSpike(number=self.stimCount, freq=self.freq, Ko=self.ko)
        else:
            cells.setK(Ko=self.ko, delay=0)

        if self.ek != None:
            ko = self.nernstINV(self.ek, 80)  # 80 defined in neuron astrocyte.hoc
            cells.run(koclamp=ko)
        else:
            cells.run()

        cells = cells.copyAttr()

        if size > 1:
            AllCells = comm.gather(cells, root=0)
        else:
            AllCells.append([cells])
        if rank == 0:
            setKoylim = True
            # if self.ko > 10:
            #     setKoylim = True
            # else:
            #     setKoylim = False
            self.plotIKSeries(AllCells,setKoylim=setKoylim,setekylim=True,setyLim=[-90,5])
            results = AllCells[0][0]
            # print(max(list(results.vPAP)))
            if expOverlay:
                results = AllCells[0][0]
                plt.plot(
                    list(results.time),
                    np.array(list(results.vPAP)) - results.RMP,
                    label="model",
                )
                plt.plot(
                    list(results.time),
                    np.array(list(results.fluorVPAP)) - results.RMP,
                    label="PAP fluor",
                    color='forestgreen',
                    linestyle='-.',
                )

                df = pd.read_csv("./Data/depolarTime.csv")
                stimIndex = 31
                # calibrate to relative point from stimulus onset
                for c in df.columns:
                    if c == "V":
                        avgV = df[c][:stimIndex].mean()
                        df[c] = df[c] - avgV
                    else:
                        df[c] = df[c] - df[c][stimIndex]
                # Match stim initialization with model
                df["t"] = df["t"] + (results.initTstop + results.stimdelay) * ms
                plt.scatter(df["t"], df["V"], label="experiment", c="black")
                initStep = results.initTstop - 50
                plt.xlim((initStep, 500))
                plt.legend()
                plt.xlabel("Time (ms)")
                plt.ylabel("Membrane potential change (mV)")
                plt.savefig(os.path.join("../results/paperRes", f"Experimental Overlay{self.tag}.pdf"))
            if self.GluT:
                plt.xlim((150,300))
                plt.xlabel('time (ms)')
                plt.ylabel('[Glu]o (mM)')
                plt.plot(list(cells.time),list(cells.GluTGlu))
                plt.savefig(os.path.join("../results/paperRes", f'GlutamateTimecourse{self.tag}.pdf'))
                plt.cla()
                plt.clf()
                plt.plot(list(cells.time),list(cells.GluTC1),label='C1')
                plt.plot(list(cells.time),list(cells.GluTC2),label='C2')
                plt.plot(list(cells.time),list(cells.GluTC3),label='C3')
                plt.plot(list(cells.time),list(cells.GluTC4),label='C4')
                plt.plot(list(cells.time),list(cells.GluTC5),label='C5')
                plt.plot(list(cells.time),list(cells.GluTC6),label='C6')
                plt.legend()
                plt.xlabel('time (ms)')
                plt.ylabel('Ratio of states')
                plt.savefig(os.path.join("../results/paperRes", f'GluTstates{self.tag}.pdf'))

    def channelComparison(self):
        self.addChannelTag()
        # Calculate the number of iterations for all parm sets
        iterations = comm.bcast(
            get_iter(
                self.KirMax,
                self.KirStep,
                self.channelCompareMax,
                self.channelCompareStep,
            ),
            root=0,
        )
        # # Adjust the range for the last process

        comm.Barrier()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "readHoc": True,
                "Glu": self.GluStim,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
            }
        )
        ccList = ["kir2"]
        if self.NMDAR:
            ccList.append("multiple")
            if self.GluT:
                funcArgs[-1]["GluTrans"] = self.optGluT
        else:
            funcArgs[-1]["multiple"] = 0
            if self.GluT:
                ccList.append("GluTrans")
            else:
                iterations = [[i] for i in range(0, self.KirMax + 1, self.KirStep)]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        callMethods = [[]]
        callArgs = [[]]
        if self.KStim and self.stimCount == 1:
            callMethods[0] += ["initialize", "setK", "run"]
            callArgs[0] += [{}, {"Ko": self.ko}, {}]
        elif self.stimCount > 1:
            callMethods[0] += ["initialize", "multiSpike", "run"]
            callArgs[0] += [
                {},
                {"number": self.stimCount, "Ko": self.ko, "freq": self.freq},
                {},
            ]

        else:
            callMethods[0] += ["initialize", "run"]
            callArgs[0] += [{}, {}]

        if self.ek != None:
            self.ko = self.nernstINV(ek, 80)  # 80 defined in neuron astrocyte.hoc
            rIndex = callMethods[0].index("run")
            callArgs[0][rIndex]["ko"] = self.ko

        results = parallizeFor(
            iterations, [PAPModel], funcArgs, ccList, callMethods, callArgs
        )

        comm.Barrier()

        if rank == 0:
            with open(
                os.path.join("intermediaryData", f"resultsParallel{self.tag}.pickle"),
                "wb",
            ) as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.plotHeatmap(results, tag=self.tag)
            self.plotIKSeries(results)
            totResults = []
            path = os.path.join(os.path.abspath("intermediaryData"), "resultsParallel")
            resFiles = glob.glob(path + "*.pickle")
            for res in resFiles:
                with open(os.path.join("intermediaryData", res), "rb") as handle:
                    results = pickle.load(handle)
                totResults += results
            self.plotHeatmap(totResults, divedend=len(resFiles))

    def glutamateSpillOver(self, sampleNum=10):
        self.addChannelTag()
        iterations = np.concatenate(
            (np.logspace(-0.5, 1, num=19), np.array([self.PAPLen]))
        )
        iterations = np.sort(iterations)
        paralleliterations = comm.bcast(
            [(i, j) for i in iterations for j in range(sampleNum)]
        )
        # # Adjust the range for the last process

        comm.Barrier()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "readHoc": True,
                "Glu": True,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
                "kir2": self.optKir,
            }
        )
        ccList = ["PAPLen", "seed"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        if self.KStim and self.stimCount == 1:
            results = parallizeFor(
                paralleliterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["initialize", "setK", "run"]],
                [[{}, {}, {}]],
            )
        elif self.stimCount > 1:
            results = parallizeFor(
                paralleliterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["initialize", "multiSpike", "run"]],
                [[{}, {"number": self.stimCount, "freq": self.freq}, {}]],
            )

        else:
            results = parallizeFor(
                paralleliterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["initialize", "run"]],
                [[{}, {}]],
            )

        comm.Barrier()
        if rank == 0:
            plt.cla()
            plt.clf()
            vList = []
            controlIndex = None
            vListarray = np.zeros((sampleNum, len(iterations)))
            for cells in results:
                for cell in cells:
                    i = np.where(cell.PAPLen == iterations)[0][
                        0
                    ]  # get index of PAPLen position in iterations
                    cindex = i / len(iterations)
                    color = cm.twilight(cindex)
                    initStep = int((cell.initTstop - 10) / cell.dt)
                    plt.plot(
                        list(cell.time)[initStep:],
                        np.array(list(cell.vPAP)[initStep:]) - cell.RMP,
                        color=color,
                    )
                    vListarray[cell.seed][i] = (
                        max(list(cell.vPAP)[initStep:]) - cell.RMP
                    )
            vListarray = vListarray.T
            controlIndex = np.where(self.PAPLen == iterations)[0][
                0
            ]  # get index of PAPLen position in iterations
            controlV = np.nansum(vListarray[controlIndex]) / sampleNum
            plt.xlabel("time (ms)")
            plt.ylabel("Membrane Potential Change (mV)")
            plt.savefig(
                os.path.join("../results/paperRes", f"GlutamateSpillOver{self.tag}.pdf")
            )
            plt.xlim((140, 160))
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"GlutamateSpillOver{self.tag}_zoom.pdf"
                )
            )
            plt.cla()
            plt.clf()

            vList = [
                np.nansum(vListarray[i]) / sampleNum for i in range(len(vListarray))
            ]
            vstdList = [np.nanstd(vListarray[i]) for i in range(len(vListarray))]
            fig, ax = plt.subplots()
            maxY = 65
            ax.errorbar(
                iterations,
                vList,
                yerr=vstdList,
                ecolor="black",
                elinewidth=0.5,
                capsize=5,
                ls="none",
            )
            ax.scatter(
                iterations,
                vList,
                cmap="twilight",
                c=[i / len(iterations) for i in range(len(iterations))],
            )
            # plot control as diamond
            if controlIndex != None:
                ax.scatter(
                    self.PAPLen,
                    controlV,
                    color=cm.twilight(controlIndex / len(iterations)),
                    marker="D",
                    label="Confined",
                    zorder=10,
                )
                # ax.axvline(
                #     self.PAPLen,
                #     ymax=controlV/top,
                #     linestyle='--',
                #     color=cm.twilight(controlIndex/len(iterations)),
                #     zorder=-1,
                # )
            maxIndex = vList.index(max(vList))
            self.peakLen = iterations[maxIndex]
            ax.scatter(
                iterations[maxIndex],
                vList[maxIndex],
                color=cm.twilight(maxIndex / len(iterations)),
                label="Spillover",
                zorder=11,
            )
            ax.axvline(
                self.peakLen,
                ymax=vList[maxIndex] / maxY,
                linestyle="--",
                color=cm.twilight(maxIndex / len(iterations)),
                zorder=-2,
            )
            ax.text(self.peakLen + 0.1, 0.1 * maxY, f"{self.peakLen:.2f} um")
            ax.legend()
            ax.set_ylim((0, maxY))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Affected PAP length (um)")
            ax.set_ylabel("Peak Voltage (mV)")
            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"GlutamateSpillOverMax{self.tag}.pdf"
                )
            )
            cellList = [
                [[cell]]
                for cells in results
                for cell in cells
                if cell.PAPLen in [self.PAPLen, 10, self.peakLen]
            ]
            for cell in cellList:
                self.tag = self.tag.split("_PAPLen")[0]
                self.tag += f"_PAPLen{cell[0][0].PAPLen}"
                self.plotIKSeries(cell,setyLim=[-6,2])

    def potassiumComparison(self):
        self.addChannelTag()
        # Calculate the number of iterations for all parm sets
        iterations = comm.bcast(get_iter(self.KirMax, self.KirStep, 10, 1), root=0)
        # # Adjust the range for the last process

        comm.Barrier()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "readHoc": True,
                "Glu": self.GluStim,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
            }
        )
        if self.NMDAR:
            funcArgs[-1]["multiple"] = self.optNMDAR
        else:
            funcArgs[-1]["multiple"] = 0
        if self.GluT:
            funcArgs[-1]["GluTrans"] = self.optGluT

        ccList = ["kir2", "Ko"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        if self.KStim and self.stimCount == 1:
            results = parallizeFor(
                iterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["initialize", "setK", "run"]],
                [[{}, {}, {}]],
            )
        elif self.stimCount > 1:
            results = parallizeFor(
                iterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["initialize", "multiSpike", "run"]],
                [[{}, {"number": self.stimCount, "freq": self.freq}, {}]],
            )

        else:
            results = parallizeFor(
                iterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["initialize", "run"]],
                [[{}, {}]],
            )

        comm.Barrier()
        if rank == 0:
            plt.cla()
            plt.clf()
            imArray = np.zeros((int(self.KirMax / self.KirStep) + 1, int(10) + 1))
            for res in results:
                imArray[
                    int(res[0].GENEDict["kir2"] / self.KirStep), int(res[0].Ko)
                ] += (max(res[0].vPAP) - res[0].RMP)
            plt.imshow(
                imArray,
                cmap="magma",
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            plt.yticks(
                range(int(self.KirMax / self.KirStep) + 1),
                np.arange(0, int(self.KirMax / self.KirStep) + 1, 1) * self.KirStep,
            )
            plt.ylabel("# of Kir Channels")
            plt.xlabel("extracellular [K] (mM)")
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 20, 2), extend="max")
            plt.clim((0, 20))

            plt.savefig(
                os.path.join("../results/paperRes", f"FullPotassium{self.tag}.pdf")
            )

    def SCeq(self, x, a, l, c):
        return a * np.exp(-x / l) + c

    def spaceConstant(self):
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        funcArgs = []
        funcArgs.append(
            {
                "voltageClamp": -60 * mV,
                "ComplexMorph": True,
                "readHoc": True,
                "Glu": False,
                "kir2": self.optKir,
                "GluTrans": self.optGluT,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        if rank == 0:
            LambdaList, VList, LenList = cells.spaceConstant()
            plt.scatter(LenList, LambdaList, label="Section Space Constant")
            plt.title(f"avg:{sum(LambdaList)/len(LambdaList)}")

            plt.savefig(os.path.join("../results/paperRes", "SpaceConstant.pdf"))
            plt.cla()
            plt.clf()
            plt.scatter(LenList, VList)
            popt, pcov = curve_fit(self.SCeq, LenList, VList)
            plt.plot(
                LenList,
                self.SCeq(np.array(LenList), *popt),
                label=f"{popt[0]}exp(-x/{popt[1]}+{popt[2]}",
            )
            plt.legend()
            plt.savefig(os.path.join("../results/paperRes", "spaceConstantFit.pdf"))

    def optDepolarizationSearch(self, x, optmV=10.0):
        # 28.812 mV for block
        # 10.0 for OE
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "readHoc": True,
                "Glu": False,
                "kir2": self.optKir*int(x),
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
            }
        )
        if funcArgs[-1]['kir2'] > 300:
            funcArgs[-1]['dt'] *= 0.1
        cells = PAPModel(**funcArgs[-1])
        cells.initialize(krule=float(x))
        cells.multiSpike(number=10, freq=100, Ko=0.5)
        cells.run()
        print(x)
        print(abs(max(list(cells.vPAP)) - cells.RMP - optmV))
        return abs(max(list(cells.vPAP)) - cells.RMP - optmV)

    def optPotassiumSearch(self, x, optmV=19.2):
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "readHoc": True,
                "Glu": False,
                "kir2": 400,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
            }
        )
        if funcArgs[-1]['kir2'] > 300:
            funcArgs[-1]['dt'] *= 0.2
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        cells.setK(Ko=float(x))
        cells.run()
        print(x)
        print(abs(max(list(cells.vPAP)) - cells.RMP - optmV))
        return abs(max(list(cells.vPAP)) - cells.RMP - optmV)
    
    def optSpikeSearch(self, x, optmV=19.2):
        freq, number = x
        freq = round(freq)
        number = round(number)
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "readHoc": True,
                "Glu": False,
                "kir2": self.optKir,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt / 100,
                "seed": self.seed,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
           }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        cells.multiSpike(number=number, freq=freq, Ko=self.ko)
        print(x)
        print(abs(max(list(cells.vPAP)) - cells.RMP - optmV))
        return abs(max(list(cells.vPAP)) - cells.RMP - optmV)

    def compareLen(self):
        self.addChannelTag()
        controlLeak = self.leak
        controldt = self.dt
        self.NMDAR = True
        for kir in [120,0,720]:
            self.dt = controldt
            if kir == 0:
                self.leak = 8455
            else:
                if kir == 720:
                    self.dt *= 0.1
                self.leak = controlLeak
            # Calculate the number of iterations for all parm sets 5-25 0.3-1.5
            self.channelCompareMax = 10
            self.channelCompareStep = 2
            iterations = comm.bcast(
                [(i,j*0.001) for i in range(2,self.channelCompareMax+1,self.channelCompareStep) for j in range(300,1501,300)],
                root=0,
            )
            # print(iterations)
            # # Adjust the range for the last process

            comm.Barrier()
            funcArgs = []
            funcArgs.append(
                {
                    "mode": 0,
                    "readHoc": True,
                    "Glu": self.GluStim,
                    "ComplexMorph": True,
                    "naleak": self.leak,
                    "clleak": 0,
                    "dt": self.dt,
                    "seed": self.seed,
                    "stimdelay": self.stimdelay,
                    "PAPCount": self.PAPCount,
                    "GluTrans":self.optGluT,
                    "kir2":kir,
                }
            )
            ccList = ["multiple","PAPLen"]
            # results are collected only on rank 0
            callMethods = [[]]
            callArgs = [[]]
            if self.KStim and self.stimCount == 1:
                callMethods[0] += ["initialize", "setK", "run"]
                callArgs[0] += [{}, {"Ko": self.ko}, {}]
            elif self.stimCount > 1:
                callMethods[0] += ["initialize", "multiSpike", "run"]
                callArgs[0] += [
                    {},
                    {"number": self.stimCount, "Ko": self.ko, "freq": self.freq},
                    {},
                ]

            else:
                callMethods[0] += ["initialize", "run"]
                callArgs[0] += [{}, {}]

            results = parallizeFor(
                iterations, [PAPModel], funcArgs, ccList, callMethods, callArgs
            )

            comm.Barrier()

            if rank == 0:
                self.plotHeatmap(results, tag=f'{self.tag}_Kir{kir}_CompLen',Kir=False)

        

    def optRMPSearch(self, x, optmV=-76.3):
        # x = leak value
        x = int(x)
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "readHoc": True,
                "kir2": 0,
                "clleak": 0,
                "naleak": x,
                "dt": self.dt,
                "seed": self.seed,
                "Glu":False,
                "multiple": 0,
                "GluTrans": 0,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        print(x)
        print(abs(cells.RMP - optmV))
        return abs(cells.RMP - optmV)
    

    def freqComparison(self):
        self.addChannelTag()
        # Calculate the number of iterations for all parm sets
        spikeNumStep = 1
        spikeFreqStep = 20
        spikeNumMin= 1
        spikeNumMax = 11
        spikeFreqMax = 200
        iterations = comm.bcast(
            [
                (i,j)
                for i in range(spikeNumMin, spikeNumMax + 1, spikeNumStep)
                for j in range(0, spikeFreqMax + 1, spikeFreqStep)
            ],
            root=0,
        )
        # print(iterations)
        # # Adjust the range for the last process

        comm.Barrier()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "readHoc": True,
                "Glu": self.GluStim,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "kir2":self.optKir,
                "multiple":self.optNMDAR,
                "GluTrans":self.optGluT,
            }
        )
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        ccList = ['multiSpike']
        callMethods = [[]]
        callArgs = [[]]
        callMethods[0] += ["setTstop","initialize", "multiSpike", "run"]
        callArgs[0] += [
            {'tstop':1000/spikeFreqStep*spikeNumMax+10},
            {},
            {"number": 'parallelItem1', "Ko": self.ko, "freq": 'parallelItem2'},
            {},
        ]

        if self.ek != None:
            self.ko = self.nernstINV(ek, 80)  # 80 defined in neuron astrocyte.hoc
            rIndex = callMethods[0].index("run")
            callArgs[0][rIndex]["ko"] = self.ko

        results = parallizeFor(
            iterations, [PAPModel], funcArgs, ccList, callMethods, callArgs,mode='MethodArgs'
        )

        comm.Barrier()

        if rank == 0:
            with open(
                os.path.join("intermediaryData", f"resultsParallel{self.tag}.pickle"),
                "wb",
            ) as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            plt.cla()
            plt.clf()
            imArray = np.zeros((
                int(spikeNumMax/spikeNumStep) + 1,
                int(spikeFreqMax/spikeFreqStep) + 1
            ))
            for res in results:
                imArray[
                    int(res[0].SpikeNum / spikeNumStep),
                    int(res[0].SpikeFreq / spikeFreqStep),
                ] += (
                    max(res[0].vPAP) - res[0].RMP
                )
            cmap = "magma"
            
            plt.imshow(
                imArray,
                cmap=cmap,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            plt.ylabel('Number of stimuli')
            plt.xlabel('Frequency (Hz)')
            plt.xticks(
                range(int(spikeFreqMax/ spikeFreqStep) + 1),
                np.arange(0, spikeFreqMax + 1, spikeFreqStep)
            )
            plt.yticks(
                range(int(spikeNumMax/spikeNumStep) + 1),
                np.arange(0, spikeNumMax + 1, spikeNumStep)
            )
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 50, 10), extend="max")
            plt.clim((0, 50))
            plt.savefig(os.path.join("../results/paperRes", f"FreqComparison{self.tag}.pdf"))


