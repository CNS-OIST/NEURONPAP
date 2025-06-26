import pickle
from astrocyte import PAPModel
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
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline as spline
import json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

font = {
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 15,
}

plt.rcParams.update(font)

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
        "fluor": "darkorange",
        "Na": "gold",
        "Cl": "chocolate",
        "Ca": "olive",
        "model": "royalblue",
        "local": "white",
        "global": "darkgray",
    }

    def resetTag(self, cell):
        # reset figure tag based on result parms
        self.tag = f"_{cell.seed}_{cell.KoSize}"
        for attr in ["GluTrans", "kir2"]:
            if attr in cell.GENEDict.keys():
                self.tag += f"{attr}{cell.GENEDict[attr]}"
        for attr in ["multiple", "GABACount", "PAPLen", "SpikeNum", "durStim"]:
            if hasattr(cell, attr):
                if attr == "PAPLen":
                    self.tag += f"{attr}{getattr(cell, attr):.2f}"
                else:
                    self.tag += f"{attr}{getattr(cell, attr)}"

        if not cell.Glu:
            self.tag += f"_NoGlu"
        if cell.GABA:
            self.tag += "_GABA"

    def saveSourceData(self, dataDict):
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
        cells.multiSpike(
            number=self.stimCount, freq=self.freq, KoSize=self.ko, video=True
        )
        cells.run(video=True)

    def plotMorphProperties(self):
        funcArgs = []
        funcArgs.append(
            {
                "ComplexMorph": True,
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
                        label=f"{cell.KoSize:.3f}",
                    )
            ax.legend()
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"{attr}Merge{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                )
            )

    def GABANMDARTrace(
        self, AllCells, NMDARCount, GABACount, fName="NMDAR_GABAR_TraceComp"
    ):
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots()

        for cells in AllCells:
            for cell in cells:
                initStep = int((cell.initTstop - 10) / cell.dt)
                if NMDARCount == cell.multiple and cell.GABACount == 0:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.vPAP)[initStep:],
                        label=f"NMDAR Vm({cell.multiple})",
                        color=self.returnColor("NMDAR"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.fluorVPAP)[initStep:],
                        label="NMDAR fluor",
                        color=self.returnColor("NMDAR"),
                        linestyle="-.",
                    )
                if GABACount == cell.GABACount and cell.multiple == 0:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.vPAP)[initStep:],
                        label=f"GABAR Vm({int(cell.GABADensity)})",
                        color=self.returnColor("GABAR"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.fluorVPAP)[initStep:],
                        label="GABAR fluor",
                        color=self.returnColor("GABAR"),
                        linestyle="-.",
                    )
        ax.legend()
        plt.savefig(
            os.path.join(
                "../results/paperRes",
                f"{fName}{self.tag}.pdf",
            )
        )
        if "OPT_" not in fName:
            self.GABANMDARTrace(AllCells, 10, 40, fName=f"OPT_{fName}")

    def plotIKSeries(
        self,
        AllCells,
        zoom=False,
        setyLim=None,
        setKoylim=False,
        setekylim=False,
        showFluor=False,
        initStep=None,
        bath=False,
        tagReset=False,
    ):
        for cells in AllCells:
            for cell in cells:
                if tagReset:
                    self.resetTag(cell)
                if (
                    (
                        not zoom
                        and cell.multiple == self.optNMDAR
                        and cell.GENEDict["kir2"] == self.optKir
                    )
                    or (not zoom and cell.multiple == None)
                    or (not zoom and cell.multiple == 0)
                ):
                    tmpTag = self.tag
                    self.tag += "_zoom"
                    self.plotIKSeries([[cell]], zoom=True)
                    self.tag = tmpTag

                if initStep == None:
                    initStep = int((cell.initTstop - 10) / cell.dt)
                if bath:
                    if max(cell.time) > 2e3:
                        cell.time *= 1e-3
                        second = True
                        startStim = 23
                        endStim = 33
                    else:
                        second = False
                        startStim = int(cell.initTstop)
                        endStim = max(cell.time)
                    print(max(cell.time))
                    fig, (ax, ax2) = plt.subplots(1, 2)
                else:
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
                if not bath:
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
                else:
                    if self.locality == "local":
                        r = patches.Rectangle(
                            xy=(startStim, 19),
                            width=endStim - startStim,
                            height=2,
                            fc="w",
                            ec="k",
                            label=f"{self.locality} stim",
                        )
                        ax.add_patch(r)
                    else:
                        ax.hlines(
                            20,
                            startStim,
                            endStim,
                            color=self.returnColor(self.locality),
                            linewidth=8,
                            label=f"{self.locality} stim",
                        )

                if setKoylim:
                    ax.set_ylim((0, 60))
                else:
                    ax.set_ylim((0, 20))
                if bath:
                    if second:
                        ax.set_xlabel("time (s)")
                    else:
                        ax.set_xlabel("time (ms)")
                else:
                    ax.set_xlabel("time (ms)")

                ax.set_ylabel("extracellular [K] (mM)")
                ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax.legend()

                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 20))

                if not bath:
                    plt.savefig(
                        os.path.join(
                            "../results/paperRes",
                            f"KoCon{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                        )
                    )

                if bath:
                    ax = ax2
                    ax.tick_params(
                        "y", right=True, labelright=True, left=False, labelleft=False
                    )
                else:
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
                if showFluor:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.fluorVPAP)[initStep:],
                        label="PAP fluor",
                        color=self.returnColor("fluor"),
                        linestyle="-.",
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

                if bath:
                    if self.locality == "local":
                        r = patches.Rectangle(
                            xy=(startStim, -54),
                            width=endStim - startStim,
                            height=16 / 5,
                            fc="w",
                            ec="k",
                            label=f"{self.locality} stim",
                        )
                        ax.add_patch(r)
                    else:
                        ax.hlines(
                            -55,
                            startStim,
                            endStim,
                            color=self.returnColor(self.locality),
                            linewidth=8,
                            label=f"{self.locality} stim",
                        )
                    if second:
                        ax.set_xlabel("time (s)")
                    else:
                        ax.set_xlabel("time (ms)")

                else:
                    ax.set_xlabel("time (ms)")

                ax.set_ylabel("Voltage (mV)")
                ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax.legend()
                if setekylim:
                    ax.set_ylim((-90, -10))
                else:
                    ax.set_ylim((-84, -77))

                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 20))

                if not bath:
                    plt.savefig(
                        os.path.join(
                            "../results/paperRes",
                            f"ekPlot{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                        )
                    )
                else:
                    plt.savefig(
                        os.path.join(
                            "../results/paperRes",
                            f"inVivoK{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
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
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.CaiPAP)[initStep:],
                    label=f"PAP Cai",
                    color=self.returnColor("Ca"),
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
                # if hasattr(cell, "iCaPAP"):
                #     ax.plot(
                #         list(cell.time)[initStep:],
                #         list(cell.iCaPAP)[initStep:],
                #         label="iCa",
                #         color=self.returnColor("Ca"),
                #     )
                # if hasattr(cell, "iClPAP"):
                #     ax.plot(
                #         list(cell.time)[initStep:],
                #         list(cell.iClPAP)[initStep:],
                #         label="iCl",
                #         color=self.returnColor("Cl"),
                #     )
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

    def setLabelColors(self, area, Kir=True, x=False, y=False, chanOverride=None):
        stdChannelDict = {
            "Kir": (120, 30),
            "GluT": (14248 * area, 812 * area),
            "GABAR": (np.inf, 0),
            "PAPLen": (
                0.425,
                0.225,
            ),  # 95th percentile of node sizes from Arizono M. Nat Comm. (2020)
        }
        if chanOverride != None and type(chanOverride) == dict:
            for k, v in chanOverride.items():
                stdChannelDict[k] = v
        if y:
            if Kir:
                mean, std = stdChannelDict["Kir"]
                _, labels = plt.yticks()
            else:
                if self.GluT:
                    mean, std = stdChannelDict["GluT"]
                else:
                    mean, std = stdChannelDict["PAPLen"]
            _, labels = plt.yticks()
            for l in labels:
                if abs(float(l.get_text()) - mean) < std:
                    l.set_color("red")

        if x:
            # if self.NMDAR:
            #     mean,std = stdChannelDict['NMDAR']
            if self.GluT:
                mean, std = stdChannelDict["GluT"]
            elif self.GABAR and Kir:
                mean, std = stdChannelDict["GABAR"]
            else:
                mean, std = stdChannelDict["PAPLen"]

            _, labels = plt.xticks()
            for l in labels:
                print(l)
                if abs(float(l.get_text()) - mean) < std:
                    l.set_color("red")

    def plotHeatmap(self, results, tag="", divedend=1, Kir=True, stdLabels=False):
        plt.cla()
        plt.clf()
        if Kir:
            if self.GluT:
                imArray = np.zeros(
                    (
                        int(self.KirMax / self.KirStep) + 1,
                        2 * int(self.channelCompareMax / self.channelCompareStep) + 1,
                    )
                )

            else:
                imArray = np.zeros(
                    (
                        int(self.KirMax / self.KirStep) + 1,
                        int(self.channelCompareMax / self.channelCompareStep) + 1,
                    )
                )
        elif self.GABAR:
            imArray = np.zeros(
                (
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                )
            )
        else:
            imArray = np.zeros((5, 5))

        for res in results:
            if Kir:
                if self.GluT:
                    imArray[
                        int(res[0].GENEDict["kir2"] / self.KirStep),
                        int(
                            (self.channelCompareMax + res[0].comparecount)
                            / self.channelCompareStep
                        ),
                    ] += (
                        max(res[0].vPAP) - res[0].RMP
                    )

                else:
                    # bug when stimulus but not GABAR
                    imArray[
                        int(res[0].GENEDict["kir2"] / self.KirStep),
                        int(res[0].comparecount / self.channelCompareStep),
                    ] += (
                        max(res[0].vPAP) - res[0].RMP
                    )
            elif self.GABAR:
                # if not Kir and GABA i.e. GABA vs. NMDAR do this
                imArray[
                    int(res[0].GABACount / self.channelCompareStep),
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
        if self.GluT:
            addChan = 1
            chanStart = -5
            skip = 2
            plt.xticks(
                range(
                    0,
                    2 * int(self.channelCompareMax / self.channelCompareStep) + addChan,
                    skip,
                ),
                np.arange(
                    chanStart,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                    skip,
                )
                * res[0].PAPGluTCount_std
                + res[0].PAPGluTCount,
            )
        else:
            chanStart = 0
            addChan = 1

        if not self.GluT:
            plt.xticks(
                range(
                    0, int(self.channelCompareMax / self.channelCompareStep) + addChan
                ),
                np.arange(
                    chanStart,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                    1,
                )
                * self.channelCompareStep,
            )
        if Kir:
            plt.yticks(
                range(int(self.KirMax / self.KirStep) + 1),
                np.arange(0, int(self.KirMax / self.KirStep) + 1, 1) * self.KirStep,
            )
            plt.ylabel("# of Kir Channels")
        elif self.GABAR:
            plt.yticks(
                range(
                    0, int(self.channelCompareMax / self.channelCompareStep) + addChan
                ),
                np.arange(
                    chanStart,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                    1,
                )
                * self.channelCompareStep,
            )
            plt.ylabel("# of GABA Channels / um2")
        else:
            plt.yticks(
                range(0, 5),
                [f"{i:.2f}" for i in np.arange(0.3, 3.1, 0.3)],
            )
            plt.ylabel("Affected PAP length (um)")
        if self.NMDAR:
            plt.xlabel("# of NMDAR Channels in PAP")
        elif self.GluT:
            # plt.xlabel("Multiple of estimated GluT density")
            plt.xlabel("# of GLT-1 Channels in PAP")
        elif self.GABAR and Kir:
            # plt.xlabel("# of GABAR channels / um2")
            plt.xlabel("# of GABAR channels in PAP")
        plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 50, 10), extend="max")
        plt.clim((0, 50))
        if stdLabels:
            self.setLabelColors(
                res[0].PAParea,
                Kir=Kir,
                x=True,
                y=True,
                chanOverride={"GluT": (res[0].PAPGluTCount, res[0].PAPGluTCount_std)},
            )

        plt.savefig(os.path.join("../results/paperRes", f"FullComparison{tag}.pdf"))


#        plt.cla()
#        plt.clf()
#        if Kir:
#            imArray = np.zeros(
#                (
#                    int(self.KirMax / self.KirStep) + 1,
#                    int(self.channelCompareMax / self.channelCompareStep) + 1,
#                )
#            )
#
#            for res in results:
#                imArray[
#                    int(res[0].GENEDict["kir2"] / self.KirStep),
#                    int(res[0].comparecount / self.channelCompareStep),
#                ] += res[0].RMP
#            imArray /= divedend
#            plt.imshow(
#                imArray,
#                cmap=cmap,
#                origin="lower",
#                interpolation="nearest",
#                aspect="equal",
#            )
#            plt.xticks(
#                range(int(self.channelCompareMax / self.channelCompareStep) + 1),
#                np.arange(0, int(self.channelCompareMax / self.channelCompareStep) + 1, 1)
#                * self.channelCompareStep,
#            )
#            plt.yticks(
#                range(int(self.KirMax / self.KirStep) + 1),
#                np.arange(0, int(self.KirMax / self.KirStep) + 1, 1) * self.KirStep,
#            )
#            plt.ylabel("# of Kir Channels")
#            if self.NMDAR:
#                plt.xlabel("# of NMDAR Channels")
#            elif self.GluT:
#                plt.xlabel("# of GluT Channels")
#            plt.colorbar(label="Voltage (mV)", ticks=np.arange(-100, -60, 10), extend="max")
#            plt.clim((-100, -60))
#            plt.savefig(os.path.join("../results/paperRes", f"FullRMP{tag}.pdf"))
#            plt.cla()
#            plt.clf()
#           print('Soma Comparison')
#
#            if Kir and self.GluT:
#                imArray = np.zeros(
#                    (
#                        int(self.KirMax / self.KirStep) + 1,
#                        2*int(self.channelCompareMax / self.channelCompareStep) + 1,
#                    )
#                )
#            else:
#                imArray = np.zeros(
#                    (
#                        int(self.KirMax / self.KirStep) + 1,
#                        int(self.channelCompareMax / self.channelCompareStep) + 1,
#                    )
#                )
#
#            for res in results:
#                initStep = int(res[0].initTstop / res[0].dt)
#                if Kir and self.GluT:
#                    imArray[
#                        int(res[0].GENEDict["kir2"] / self.KirStep),
#                        int(self.channelCompareMax + res[0].comparecount / self.channelCompareStep),
#                    ] += (
#                        max(list(res[0].vSoma)[initStep:]) - res[0].RMP
#                    )
#
#                else:
#                    imArray[
#                        int(res[0].GENEDict["kir2"] / self.KirStep),
#                        int(res[0].comparecount / self.channelCompareStep),
#                    ] += (
#                        max(list(res[0].vSoma)[initStep:]) - res[0].RMP
#                    )
#            imArray /= divedend
#            plt.imshow(
#                imArray,
#                cmap=cmap,
#                origin="lower",
#                interpolation="nearest",
#                aspect="equal",
#            )
#            plt.xticks(
#                range(int(self.channelCompareMax / self.channelCompareStep) + 1),
#                np.arange(0, int(self.channelCompareMax / self.channelCompareStep) + 1, 1)
#                * self.channelCompareStep,
#            )
#            plt.yticks(
#                range(int(self.KirMax / self.KirStep) + 1),
#                np.arange(0, int(self.KirMax / self.KirStep) + 1, 1) * self.KirStep,
#            )
#            plt.ylabel("# of Kir Channels")
#            if self.NMDAR:
#                plt.xlabel("# of NMDAR Channels")
#            elif self.GluT:
#                plt.xlabel("GluT Channels")
#            elif self.GABAR and Kir:
#                plt.xlabel("# of GABAR channels / um2")
#
#            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 10, 1), extend="max")
#            plt.clim((0, 10))
#            plt.savefig(os.path.join("../results/paperRes", f"FullSoma{tag}.pdf"))


class procedure(plotFigures):
    leak = 3e5
    optKir = 120
    optNMDAR = 5
    optGABAR = 50
    optGluT = 0  # std * optGluT + mean
    channelCompareMax = 25
    channelCompareStep = 5
    KirMax = 400
    KirStep = 40
    seed = int()
    ko = float()
    tag = str()
    OE = False
    NMDAR = True
    GABAR = True
    GluT = True
    GluStim = True
    GabaStim = False
    KStim = True
    stimdelay = 0
    dt = 0.05
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
        if self.GABAR:
            self.tag += "_GABAR"
        if not self.GluStim:
            self.tag += "_NoGlu"
        if self.GabaStim:
            self.tag += "_GABA"
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
                            KoSize=self.ko,
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
                            KoSize=self.ko,
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
        funcArgs = []
        vClampList = comm.bcast(np.arange(-95, -40, 5), root=0)
        print(vClampList)
        plt.cla()
        plt.clf()
        funcArgs.append(
            {
                "mode": 2,
                "ComplexMorph": True,
                "dt": self.dt,
                "naleak": self.leak,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "seed": self.seed,
                "GluTrans": self.optGluT,
            }
        )
        ccList = ["voltageClamp"]
        results = parallizeFor(
            vClampList,
            [PAPModel],
            funcArgs,
            ccList,
            [["setTstop", "initialize", "run"]],
            [[{"tstop": 260}, {}, {}]],
        )
        if rank == 0:
            for cells in results:
                for cell in cells:
                    print(cell.voltageClamp)
                    plt.plot(cell.time, cell.vSoma, color="black")

            plt.xlabel("time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.xlim((140, 260))
            plt.savefig(os.path.join("../results/paperRes", f"VoltageClampSoma.pdf"))

    def branchAttenuation(self, alterDist=False, replay=False):
        self.addChannelTag()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "Glu": self.GluStim,
                "GABA": self.GabaStim,
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
        if self.GluStim:
            funcArgs[-1]["multiple"] = self.optNMDAR
        else:
            funcArgs[-1]["multiple"] = 0
        if self.GabaStim:
            funcArgs[-1]["GABACount"] = self.optGABAR
        else:
            funcArgs[-1]["GABACount"] = 0

        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        if replay:
            cells.replayK("./Data/invivo_K.csv", isolate=True, setStop=60e3)
            # cells.replayK("./Data/invivo_test.csv", isolate=True)
        else:
            cells.multiSpike(number=self.stimCount, freq=self.freq, KoSize=self.ko)
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
        plt.xticks(
            range(0, 11, 2), [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )  # float point generated by np.linspace
        max_time = len(list(cells.branchAtten[-1])[initStep:]) * cells.dt
        if max_time > 1e3:
            steps_per_time = int(10e3 / cells.dt)  # every 2 s
            cells.dt *= 1e-3
            plt.ylabel("Time (s)")
        else:
            steps_per_time = int(20 / cells.dt)  # every 20 ms
            plt.ylabel("Time (ms)")

        plt.yticks(
            np.arange(
                0,
                len(list(cells.branchAtten[0])[initStep:]) + 1,
                steps_per_time,
                dtype=int,
            ),
            np.round(
                np.arange(
                    0,
                    len(list(cells.branchAtten[0])[initStep:]) + 1,
                    steps_per_time,
                    dtype=int,
                )
                * cells.dt
            ).astype(int),
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
            cells.setK(KoSize=self.ko)
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
        self.KirNMDAPhase()
        # self.duramplenPhase()

    def duramplenPhase(self):
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"
        self.addChannelTag()

        AllCells = []
        for amp in [0.5, 10]:
            funcArgs = []
            funcArgs.append(
                {
                    "mode": 0,
                    "ComplexMorph": True,
                    "bNum": 1,
                    "dt": 0.01,
                    "naleak": self.leak,
                    "clleak": 0,
                    "seed": self.seed,
                    "KoSize": amp,
                }
            )
            if self.GluT:
                if self.NMDAR:
                    funcArgs[-1]["GABACount"] = 0
                    funcArgs[-1]["multiple"] = self.optNMDAR
                    funcArgs[-1]["GluTrans"] = self.optGluT
                    chanName = "GluT_NMDAR"
                elif self.GABAR:
                    funcArgs[-1]["GABACount"] = self.optGABAR
                    funcArgs[-1]["multiple"] = 0
                    funcArgs[-1]["GluTrans"] = self.optGluT
                    chanName = "GABAR"
                else:
                    funcArgs[-1]["GABACount"] = 0
                    funcArgs[-1]["multiple"] = 0
                    funcArgs[-1]["GluTrans"] = self.optGluT
                    chanName = "GluTrans"
            else:
                if self.NMDAR:
                    funcArgs[-1]["GABACount"] = 0
                    funcArgs[-1]["multiple"] = self.optNMDAR
                    funcArgs[-1]["GluTrans"] = None
                    chanName = "NMDAR"
                elif self.GABAR:
                    funcArgs[-1]["GABACount"] = self.optGABAR
                    funcArgs[-1]["multiple"] = 0
                    funcArgs[-1]["GluTrans"] = None
                    chanName = "GABAR"
                else:
                    funcArgs[-1]["GABACount"] = 0
                    funcArgs[-1]["multiple"] = 0
                    funcArgs[-1]["GluTrans"] = None
                    chanName = ""

            if funcArgs[-1]["multiple"] > 0 or funcArgs[-1]["GluTrans"] != None:
                funcArgs[-1]["Glu"] = True
            if funcArgs[-1]["GABACount"] > 0:
                funcArgs[-1]["GABA"] = True

            iterations = comm.bcast(
                [
                    (dur, papLen)
                    for dur in np.arange(0.5, 1, 0.1)
                    for papLen in np.arange(0.3, 0.8, 0.1)
                ],
                root=0,
            )
            ccList = comm.bcast(["dur", "papLen"], root=0)
            comm.Barrier()
            if self.KStim:
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
                        if cell.KoSize == 0.5:
                            color = "r"
                            z = len(results) + 1
                        else:
                            color = cm.summer(i / len(results))
                            z = i
                        plt.plot(
                            list(cell.KoPAP)[initStep:],
                            list(cell.vPAP)[initStep:],
                            label=f"Dur:{cell.dur:.1f},PAPLen:{cell.PAPLen:.1f}",
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
                            plt.legend()
                            plt.ylabel("Voltage (mV)")
                            plt.xlabel("[K]o (mM)")
                            if self.PAPLen <= 0.3:
                                plt.ylim((-90, -50))
                            plt.xlim((2, 14))
                            plt.savefig(
                                os.path.join(
                                    "../results/paperRes",
                                    f"phasePlanePotassium{cell.KoSize:.2f}{self.tag}.pdf",
                                )
                            )
                            plt.cla()
                            plt.clf()

            plt.close("all")

    def KirNMDAPhase(self):
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"
        self.addChannelTag()

        AllCells = []
        for kircount in [400, self.optKir]:
            for chanCount in [self.optNMDAR, 25, self.optGABAR]:
                funcArgs = []
                funcArgs.append(
                    {
                        "mode": 0,
                        "ComplexMorph": True,
                        "bNum": 1,
                        "dt": 0.01,
                        "naleak": self.leak,
                        "clleak": 0,
                        "seed": self.seed,
                        "PAPLen": self.PAPLen,
                    }
                )
                if self.GluT:
                    if self.NMDAR:
                        funcArgs[-1]["GABACount"] = 0
                        funcArgs[-1]["multiple"] = chanCount
                        funcArgs[-1]["GluTrans"] = 0
                        chanName = "GluT_NMDAR"
                    elif self.GABAR:
                        funcArgs[-1]["GABACount"] = chanCount
                        funcArgs[-1]["multiple"] = 0
                        funcArgs[-1]["GluTrans"] = 0
                        chanName = "GABAR"
                    else:
                        funcArgs[-1]["GABACount"] = 0
                        funcArgs[-1]["multiple"] = 0
                        funcArgs[-1]["GluTrans"] = 0
                        chanName = "GluTrans"
                else:
                    if self.NMDAR:
                        funcArgs[-1]["GABACount"] = 0
                        funcArgs[-1]["multiple"] = chanCount
                        funcArgs[-1]["GluTrans"] = None
                        chanName = "NMDAR"
                    elif self.GABAR:
                        funcArgs[-1]["GABACount"] = chanCount
                        funcArgs[-1]["multiple"] = 0
                        funcArgs[-1]["GluTrans"] = None
                        chanName = "GABAR"
                    else:
                        funcArgs[-1]["GABACount"] = 0
                        funcArgs[-1]["multiple"] = 0
                        funcArgs[-1]["GluTrans"] = None
                        chanName = ""

                if funcArgs[-1]["multiple"] > 0 or funcArgs[-1]["GluTrans"] != None:
                    funcArgs[-1]["Glu"] = True
                if funcArgs[-1]["GABACount"] > 0:
                    funcArgs[-1]["GABA"] = True

                iterations = comm.bcast(
                    [(kircount, conc) for conc in np.array([0, 0.5, 1.0, 5.0, 10.0])],
                    root=0,
                )
                ccList = comm.bcast(["kir2", "KoSize"], root=0)
                comm.Barrier()
                if self.KStim:
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
                            if cell.KoSize == 0.5:
                                color = "r"
                                z = len(results) + 1
                            else:
                                color = cm.summer(i / len(results))
                                z = i
                            plt.plot(
                                list(cell.KoPAP)[initStep:],
                                list(cell.vPAP)[initStep:],
                                label=f"{cell.KoSize:.1f}",
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
        plt.xlabel("extracellular [K] (mM)")
        plt.savefig(os.path.join("../results/paperRes", "ekKODepolarcomp.pdf"))
        print(koList)

    def KOComp(self, papCount=10, koCond=6):
        for transmitter in ["GABAR", "NMDAR"]:
            self.NMDAR = False
            self.GABAR = False
            self.GluT = True
            self.GabaStim = False
            self.GluStim = False
            if transmitter == "GABAR":
                self.GabaStim = True
                self.GluStim = False
                self.GABAR = True
            else:
                self.GabaStim = False
                self.GluStim = True
                self.NMDAR = True
            self.runKOComp(transmitter, papCount, koCond)

    def runKOComp(self, transmitter, papCount, koCond):
        AllCells = []
        # single run
        for i in range(koCond):
            funcArgs = []
            # Order is important
            if i == 0:
                self.GluT = True
                setattr(self, transmitter, True)
                controlKir = self.optKir
                self.optKir = controlKir
                controlLeak = self.leak
                tmpdt = self.dt
            elif i < 3:
                self.GluT = True
                setattr(self, transmitter, True)
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
                    setattr(self, transmitter, False)
                elif i == 4:
                    # NMDARKO
                    setattr(self, transmitter, True)
                    self.GluT = False
                elif i == 5:
                    # NMDARKO
                    setattr(self, transmitter, False)
                    self.GluT = False

            funcArgs.append(
                {
                    "mode": 0,
                    "ComplexMorph": True,
                    "bNum": 1,
                    "Glu": self.GluStim,
                    "GABA": self.GabaStim,
                    "kir2": self.optKir,
                    "clleak": 0,
                    "naleak": self.leak,
                    "dt": self.dt,
                }
            )
            if i == 1:
                krule = {"kuptake": True}
            elif i == 2:
                # nonspecific K+ block
                # funcArgs[-1]['twik'] = 0
                # funcArgs[-1]['kleak'] = 0
                krule = {"kblock": True}
            else:
                krule = {}

            if self.NMDAR:
                funcArgs[-1]["multiple"] = self.optNMDAR
            else:
                funcArgs[-1]["multiple"] = 0
            if self.GluT:
                funcArgs[-1]["GluTrans"] = self.optGluT

            if self.GABAR:
                funcArgs[-1]["GABACount"] = self.optGABAR
            else:
                funcArgs[-1]["GABACount"] = 0

            comm.Barrier()
            if self.peakLen == None:
                self.peakLen = 2
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
            callMethods[0] += ["initialize", "multiSpike", "run"]
            callArgs[0] += [
                krule,
                {"number": self.stimCount, "KoSize": self.ko, "freq": self.freq},
                {},
            ]

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
                    if cell.multiple > 0 or hasattr(cell, "GABACount"):
                        if cell.GENEDict["kir2"] == controlKir:
                            if (
                                "GluTrans" in cell.GENEDict.keys()
                                and cell.GENEDict["GluTrans"] != None
                            ):
                                # control
                                setattr(self, transmitter, True)
                                k = 0
                                self.addChannelTag()
                            else:
                                # GluT KO
                                setattr(self, transmitter, True)
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
                            plt.plot(cell.time, cell.vPAP)
                            plt.savefig("KO changes.pdf")
                    else:
                        if (
                            "GluTrans" in cell.GENEDict.keys()
                            and cell.GENEDict["GluTrans"] != None
                        ):
                            # transmitter KO
                            k = 3
                            setattr(self, transmitter, False)
                            self.GluT = True
                            self.addChannelTag()
                        else:
                            # transmitter KO
                            # GluT KO
                            k = 5
                            setattr(self, transmitter, False)
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
                f"{transmitter} KO",
                "GluT KO",
                f"{transmitter} KO\nGluT KO",
            ]
            category = category[:koCond]
            val_test = {}
            for i in range(len(resMat)):
                if i < koCond:  # Number of KO conditions
                    dict_key = "confined"
                    val_test[category[i]] = ttest_rel(resMat[i], resMat[i + koCond])
                else:
                    dict_key = "spillover"
                if i in [0, koCond]:
                    for cond in [1, 2]:
                        print(dict_key)
                        print(category[i % 6], category[(i + cond) % 6])
                        print(ttest_rel(resMat[i], resMat[i + cond]))

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
                    f"KO_GENE_Comparison{transmitter}.pdf",
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
                os.path.join("../results/paperRes", f"ttest_res{transmitter}.json"), "w"
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
        cells.setTstop(260)
        cells.initialize()
        cells.setK(KoSize=100, delay=0, dur=100)
        cells.run()
        cells = cells.copyAttr()
        initStep = int((cells.initTstop + 10) / cells.dt) + 1
        flux = np.array(list(cells.flux)[initStep:]) * -1
        kbath = np.array(list(cells.kbath)[initStep:]) * -1
        kbath[kbath == 0] = np.nan
        _, ax = plt.subplots(figsize=(11, 6))
        ax.plot(list(cells.time)[initStep:], np.divide(flux, kbath))
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Ratio of\ninflux / diffusion\nfor potassium")
        plt.savefig(os.path.join("../results/paperRes", "fluxRatioOvertime.pdf"))

    def singleRun(self, *args, expOverlay=False, GluTime=False, nearSoma=False):
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        if len(args) > 0:
            d = 20
            k = 500
            tau1 = 1.69
            nmdaCount, s, d, tau2 = args
        else:
            k = 500
            nmdaCount = 12
            d = 20
            s = -80
            tau1 = 1.69
            tau2 = 3.97

        AllCells = []
        funcArgs = []
        funcArgs.append(
            {
                "Glu": self.GluStim,
                "GABA": self.GabaStim,
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "kir2": self.optKir,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "KoSize": 3,
                "PAPLen": 0.3,
            }
        )
        if self.OE:
            funcArgs[-1]["kir2"] = 400

        if funcArgs[-1]["kir2"] > 300:  # to compensate for mathematical unstability
            funcArgs[-1]["dt"] *= 0.2

        if self.NMDAR and self.GluStim:
            # funcArgs[-1]["multiple"] = self.optNMDAR
            funcArgs[-1]["multiple"] = nmdaCount
        else:
            funcArgs[-1]["multiple"] = None
        if self.GluT and self.GluStim:
            funcArgs[-1]["GluTrans"] = self.optGluT
        if self.GABAR and self.GabaStim:
            funcArgs[-1]["GABACount"] = self.optGABAR
        else:
            funcArgs[-1]["GABACount"] = 0

        cells = PAPModel(**funcArgs[-1])
        if nearSoma:
            cells.setPAPNearSoma()
        #        cells.setTstop(500)
        cells.initialize()

        if self.GluStim:
            cells.setNMDA_Mgblock(k, d, s)
            cells.setNMDA_TC(tau1, tau2)
        # cells.setSlowing(slow)

        # 6.12418747    5.40398865    0.42320213 -100.
        if size > 2:
            if rank == 2:
                stim = 1
            elif rank == 1:
                stim = 5
            else:
                stim = 10
        else:
            stim = self.stimCount
        cells.multiSpike(number=stim, freq=self.freq, KoSize=self.ko)
        # else:
        #     cells.setK(KoSize=self.ko, delay=0)

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
            self.plotIKSeries(
                AllCells, setKoylim=setKoylim, setekylim=not nearSoma, setyLim=[-5, 5]
            )
            results = AllCells[0][0]
            # print(max(list(results.vPAP)))
            if expOverlay:
                self.plotExpFit(
                    cells, stim=stim, Fname="expOverlay", correctArtifact=True
                )
                # results = AllCells[0][0]
                # fig, ax1 = plt.subplots()
                # ax2 = ax1.twinx()
                # ax1.plot(
                #     list(results.time),
                #     np.array(list(results.vPAP)) - results.RMP,
                #     label="model",
                #     color=self.returnColor('model')
                # )
                # ax2.plot(
                #     list(results.time),
                #     (np.array(list(results.fluorVPAP)) - results.RMP) * -1/10,
                #     linestyle='-.',
                #     color=self.returnColor('fluor'),
                #     label="sim fluor",
                # )
                # df = self.readExpRawData(results)
                # ax2.scatter(
                #     df["t"], df["V"]* -1/10,
                #     label="experiment",
                #     color=self.returnColor('fluor')
                # )
                # initStep = results.initTstop - 50
                # ax1.set_xlim((initStep, 500))
                # ax1.legend(loc='upper left')
                # ax2.legend(loc='upper right')
                # ax1.set_xlabel("Time (ms)")
                # ax1.set_ylabel("Membrane potential change (mV)")
                # ax2.set_ylabel("$\Delta F/F_0$ (%)")
                # ylim_value = 40 #mv
                # ax2.set_ylim((0,ylim_value*-1/10))
                # ax1.set_ylim((0,ylim_value))
                # plt.title(f'{nmdaCount},{d},{s}')
                # for axObj,label in { ax1:'model', ax2:'fluor'}.items():
                #     axObj.tick_params(axis='y',colors=self.returnColor(label))
                #     axObj.yaxis.label.set_color(self.returnColor(label))

                # # r,f = self.calcRiseFall(list(df["t"]),list(df["V"]),'exp')
                # # titleString = 'Exp $T_{1/2}$' + f':{int(r)},{int(f)} ms '
                # # r,f = self.calcRiseFall(
                # #     list(results.time),
                # #     (np.array(list(results.fluorVPAP)) - results.RMP),
                # #     'sim'
                # # )
                # # titleString = titleString + 'Sim $T_{1/2}$' + f':{int(r)},{int(f)} ms'
                # # plt.title(titleString)
                # plt.savefig(os.path.join("../results/paperRes", f"Experimental Overlay{self.tag}.pdf"))

            if self.GluT and GluTime:
                if hasattr(cells, "GluTGlu"):
                    plt.xlim((150, 300))
                    plt.xlabel("time (ms)")
                    plt.ylabel("[Glu]o (mM)")
                    plt.plot(list(cells.time), list(cells.GluTGlu))
                    plt.savefig(
                        os.path.join(
                            "../results/paperRes", f"GlutamateTimecourse{self.tag}.pdf"
                        )
                    )
                plt.cla()
                plt.clf()
                plt.plot(list(cells.time), list(cells.GluTC1), label="C1")
                plt.plot(list(cells.time), list(cells.GluTC2), label="C2")
                plt.plot(list(cells.time), list(cells.GluTC3), label="C3")
                plt.plot(list(cells.time), list(cells.GluTC4), label="C4")
                plt.plot(list(cells.time), list(cells.GluTC5), label="C5")
                plt.plot(list(cells.time), list(cells.GluTC6), label="C6")
                plt.legend()
                plt.xlabel("time (ms)")
                plt.ylabel("Ratio of states")
                plt.savefig(
                    os.path.join("../results/paperRes", f"GluTstates{self.tag}.pdf")
                )

    def bathExperiment(self, runAll=True, invivo=False, isolate=False, gaba=False):
        if runAll:
            if invivo:
                invivoRunConds = [True, False]
            else:
                invivoRunConds = [False]

            allConds = len(invivoRunConds) * 2

            if not size > allConds:
                wMessage(
                    f"bath experiment runAll only when there are more than {allConds} ranks"
                )
            for i, bool_invivo in enumerate(invivoRunConds):
                for j, bool_isolate in enumerate([True, False]):
                    if rank == 2 * i + j:
                        self.bathExperiment(
                            runAll=False,  # for escaping inf loop
                            invivo=bool_invivo,
                            isolate=bool_isolate,
                        )
            if rank == allConds:
                self.bathExperiment(runAll=False, gaba=True)  # for escaping inf loop

        else:
            if gaba:
                self.gababathExperiment()
            else:
                self.kbathExperiment(invivo, isolate)

    def kbathExperiment(self, invivo, isolate):
        # add multispike ek clamp
        self.addChannelTag()
        if invivo:
            self.tag += "_invivoBath"
        else:
            self.tag += "_Bath"
        if isolate:
            self.locality = "local"
            self.tag += "_isolated"
        else:
            self.locality = "global"
        # print(self.tag)
        AllCells = []
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "kir2": 120,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        if self.OE:
            funcArgs[-1]["kir2"] = 400

        if funcArgs[-1]["kir2"] > 300:  # to compensate for mathematical unstability
            funcArgs[-1]["dt"] *= 0.2
        funcArgs[-1]["multiple"] = 0
        funcArgs[-1]["Glu"] = False
        funcArgs[-1]["GABACount"] = 0
        funcArgs[-1]["GABA"] = False
        funcArgs[-1]["GluTrans"] = self.optGluT

        cells = PAPModel(**funcArgs[-1])
        if invivo:
            cells.initialize()
            cells.replayK("./Data/invivo_K.csv", isolate=isolate)
            cells.run()

        else:
            cells.setTstop(500)
            cells.initialize()
            if isolate:
                video = False
            else:
                video = True
            cells.setKBath(8, dur=500, video=video, isolate=isolate)
            cells.run()

        cells = cells.copyAttr()

        # if size > 1:
        #     AllCells = comm.gather(cells, root=0)
        # else:
        AllCells.append([cells])
        setKoylim = True
        # if self.ko > 10:
        #     setKoylim = True
        # else:
        #     setKoylim = False
        self.plotIKSeries(
            AllCells,
            setKoylim=setKoylim,
            setekylim=True,
            setyLim=[-20, 10],
            initStep=0,
            bath=True,
        )
        results = AllCells[0][0]
        # print(max(list(results.vPAP)))

    def gababathExperiment(self):
        # add multispike ek clamp
        self.addChannelTag()
        self.tag += "_gabaBath"
        self.locality = "global"

        # print(self.tag)
        AllCells = []
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "multiple": 0,
                "Glu": False,
                "GABA": True,
                "bNum": 1,
                "kir2": 120,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        if self.OE:
            funcArgs[-1]["kir2"] = 400

        if funcArgs[-1]["kir2"] > 300:  # to compensate for mathematical unstability
            funcArgs[-1]["dt"] *= 0.2
        funcArgs[-1][
            "GABACount"
        ] = 1  # not optGABA as GABABath distributes GABAs with different mechanism compared to default distribution

        cells = PAPModel(**funcArgs[-1])
        cells.setTstop(200)
        cells.GABABath(1, 0)
        cells.run()
        cells = cells.copyAttr()

        # if size > 1:
        #     AllCells = comm.gather(cells, root=0)
        # else:
        AllCells.append([cells])
        setKoylim = True
        # if self.ko > 10:
        #     setKoylim = True
        # else:
        #     setKoylim = False
        self.plotIKSeries(
            AllCells, setKoylim=setKoylim, setekylim=True, setyLim=[-20, 10], bath=True
        )
        results = AllCells[0][0]
        # print(max(list(results.vPAP)))

    def calcRiseFall(self, t, V, label=None, stdout=False):
        t = list(t)
        V = list(V)
        Thalf = []
        if label != None and stdout:
            print(label)
        maxInd = list(V).index(max(V))
        for i, voltageRF in enumerate([V[:maxInd], V[maxInd:]]):
            if i == 0:
                origin = 150
            else:
                origin = t[maxInd]
            for j, v in enumerate(voltageRF):
                if i == 0 and v > max(V) / 2:
                    if stdout:
                        print("rise")
                    break
                elif i == 1 and v < max(V) / 2:
                    if stdout:
                        print("fall")
                    j += maxInd
                    break

            if stdout:
                print(v, max(V))
                print(f"Half T: {abs(t[j] - origin)}")
            Thalf.append(abs(t[j] - origin))

        return Thalf

    def GABANMDARCompare(self):
        self.addChannelTag()
        # Calculate the number of iterations for all parm sets
        iterations = comm.bcast(
            get_iter(
                self.channelCompareMax,
                self.channelCompareStep,
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
                "Glu": True,
                "GABA": True,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "GluTrans": self.optGluT,
                "kir2": self.optKir,
            }
        )
        ccList = ["multiple", "GABACount"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        callMethods = [[]]
        callArgs = [[]]
        if self.KStim:
            callMethods[0] += ["initialize", "multiSpike", "run"]
            callArgs[0] += [
                {},
                {"number": self.stimCount, "KoSize": self.ko, "freq": self.freq},
                {},
            ]

        else:
            callMethods[0] += ["setTstop", "initialize", "run"]
            callArgs[0] += [{}, {}, {}]

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
            # self.plotHeatmap(results, tag=self.tag,Kir=False)
            self.GABANMDARTrace(results, self.channelCompareMax, self.channelCompareMax)

    def channelComparison(self):
        self.addChannelTag()
        if self.GABAR:
            self.channelCompareMax *= 4
            self.channelCompareStep *= 4
        if not (self.GABAR or self.NMDAR) and self.GluT:
            self.channelCompareMax *= 100
            self.channelCompareStep *= 100
            iterations = [
                (i, j)
                for i in range(0, self.KirMax + 1, self.KirStep)
                for j in range(
                    -self.channelCompareMax,
                    self.channelCompareMax + 1,
                    self.channelCompareStep,
                )
            ]
            iterations = comm.bcast(iterations, root=0)
        else:
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
                "Glu": self.GluStim,
                "GABA": self.GabaStim,
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
        if self.GABAR:
            funcArgs[-1]["multiple"] = 0
            ccList.append("GABACount")
        elif self.NMDAR:
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
        callMethods[0] += ["initialize", "multiSpike", "run"]
        callArgs[0] += [
            {},
            {"number": self.stimCount, "KoSize": self.ko, "freq": self.freq},
            {},
        ]

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
            if self.NMDAR and self.GluT:
                self.GluT = False  # force plot priority of NMDA
            self.plotHeatmap(results, Kir=True, tag=self.tag, stdLabels=True)
            self.plotIKSeries(results, setekylim=True, setKoylim=True, setyLim=[-15, 1])
            totResults = []
            path = os.path.join(os.path.abspath("intermediaryData"), "resultsParallel")
            resFiles = glob.glob(path + "*.pickle")
            for res in resFiles:
                with open(os.path.join("intermediaryData", res), "rb") as handle:
                    results = pickle.load(handle)
                totResults += results

    #            self.plotHeatmap(totResults, divedend=len(resFiles))

    def glutamateSpillOver(self, sampleNum=10):
        self.addChannelTag()
        if self.GluStim:
            iterations = np.concatenate(
                (np.logspace(-0.5, 1, num=19), np.array([self.PAPLen]))
            )

        else:
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
                "Glu": self.GluStim,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "kir2": self.optKir,
                "KoSize": self.ko,
                "multiple": 0,
            }
        )
        if not self.KStim:
            funcArgs[-1]["KoSize"] = 0
        if (self.GluStim or self.GabaStim) and self.KStim:
            if self.GluStim:
                # funcArgs[-1]['multiple']=self.optNMDAR
                funcArgs[-1]["GluTrans"] = self.optGluT

            if self.GabaStim:
                funcArgs[-1]["GABACount"] = self.optGABAR
        ccList = ["PAPLen", "seed"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        if self.KStim:
            results = parallizeFor(
                paralleliterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["initialize", "multiSpike", "run"]],
                [
                    [
                        {},
                        {
                            "number": self.stimCount,
                            "freq": self.freq,
                            "KoSize": self.ko,
                        },
                        {},
                    ]
                ],
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
                        np.array(list(cell.time)[initStep:]) * 1e-3,  # ms to s
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
            plt.xlabel("time (s)")
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
            maxY = max(vList) + 5

            self.peakLen = iterations[maxIndex]
            ax.scatter(
                iterations[maxIndex],
                vList[maxIndex],
                color=cm.twilight(maxIndex / len(iterations)),
                label="Spillover",
                zorder=11,
            )
            if self.GluStim and self.KStim:
                ax.axvline(
                    self.peakLen,
                    ymax=vList[maxIndex] / maxY,
                    linestyle="--",
                    color=cm.twilight(maxIndex / len(iterations)),
                    zorder=-2,
                )
                if maxIndex < (len(iterations) - 1):
                    ax.text(
                        self.peakLen * 1.1, 0.1 * maxY, f"{self.peakLen:.2f} \u03bcm"
                    )
                else:
                    ax.text(
                        self.peakLen * 0.9, 0.1 * maxY, f"{self.peakLen:.2f} \u03bcm"
                    )
            ax.legend()
            ax.set_ylim((0, maxY))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Affected PAP length (\u03bcm)")
            self.GluT = False  # just to force setLabelColors to highlight PAPLen
            self.setLabelColors(0, Kir=False, y=True)
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
                self.plotIKSeries(cell, setekylim=True, setKoylim=True, setyLim=[-6, 2])

    def potassiumComparison(self):
        for comparison in ["PAPLen", "durStim", "KoSize"]:
            if comparison == "KoSize":
                compMax = 50
                compStep = 5
                startb = 0
            elif comparison == "PAPLen":
                compMax = 3
                compStep = 0.3
                startb = 0.3
            elif comparison == "durStim":
                compMax = 9
                compStep = 1
                startb = 1

            if comparison != "KoSize":
                if comparison == "durStim":
                    iterations = comm.bcast(
                        get_iter(50, 5, compMax, compStep, startb=startb), root=0
                    )
                    logx = None
                else:
                    logx = np.logspace(1, -1.5, base=0.3, num=10)
                    iterations = comm.bcast(
                        [(i, j) for i in range(0, 51, 5) for j in logx],
                        root=0,
                    )
                self.runAmpLenComparison(
                    comparison, iterations, compMax, compStep, logx=logx
                )

            # Calculate the number of iterations for all parm sets
            iterations = comm.bcast(
                get_iter(self.KirMax, self.KirStep, compMax, compStep, startb=startb),
                root=0,
            )
            # # Adjust the range for the last process
            self.runPotassiumComparison(
                comparison, iterations, maxStep=compMax, intermStep=compStep
            )

    def runAmpLenComparison(
        self, comparison, iterations, maxStep, intermStep, logx=None
    ):
        self.addChannelTag()
        mprint(logx)

        comm.Barrier()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "Glu": self.GluStim,
                "GABA": False,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "kir2": self.optKir,
            }
        )
        # if self.NMDAR:
        #     funcArgs[-1]["multiple"] = self.optNMDAR
        # else:
        self.NMDAR = False
        funcArgs[-1]["multiple"] = 0
        if self.GluStim:
            self.GluT = True
            funcArgs[-1]["GluTrans"] = self.optGluT
        else:
            self.GluT = False
            funcArgs[-1]["GluTrans"] = None

        ccList = ["KoSize", comparison]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            [["initialize", "multiSpike", "run"]],
            [[{}, {"number": self.stimCount, "freq": self.freq}, {}]],
        )

        comm.Barrier()
        if rank == 0:
            plt.cla()
            plt.clf()
            imArray = np.zeros(
                (11, int(len(iterations) / 11))
            )  # int(maxStep / intermStep) + 1))
            for res in results:
                if logx is not None:
                    print(logx)
                    print(getattr(res[0], ccList[1]))
                    index = int(np.where(logx == getattr(res[0], ccList[1]))[0])
                    imArray[int(getattr(res[0], ccList[0]) / 5), index] += (
                        max(res[-1].vPAP) - res[0].RMP
                    )
                else:
                    imArray[
                        int(getattr(res[0], ccList[0]) / 5),
                        int(getattr(res[0], ccList[1]) / intermStep) - 1,
                    ] += (
                        max(res[0].vPAP) - res[0].RMP
                    )

            plt.imshow(
                imArray,
                cmap="magma",
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            # set xlabel
            if comparison == "durStim":
                skip = 2
                dec = 0
                printType = int
                plt.xlabel("stim. duation (ms)")
            else:
                skip = 2
                dec = 1
                printType = float
                plt.xlabel("PAP Length (\u03bcm)")
            if logx is not None:
                logx_label = [
                    f"{val:.1f}" if val < 1 else str(np.round(val).astype(int))
                    for val in logx
                ]
                plt.xticks(np.arange(0, len(logx), 1), logx_label)
            else:
                plt.xticks(
                    np.arange(0, int(maxStep / intermStep) + 1, 1),
                    np.round(
                        np.arange(0, maxStep + intermStep / 2, intermStep), decimals=dec
                    ).astype(printType),
                )
            # set ylabel
            skip = 1
            dec = 0
            printType = int
            maxStep = 50
            intermStep = 5
            plt.ylabel("$\Delta$extracellular [K] (mM)")
            plt.yticks(
                np.arange(0, int(maxStep / intermStep) + 1, 1),
                np.round(
                    np.arange(0, maxStep + intermStep / 2, intermStep), decimals=dec
                ).astype(printType),
            )
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 20, 2), extend="max")
            plt.clim((0, 20))
            if comparison == "PAPLen":
                self.GluT = False  # just to force plot setLabel Colors
            self.setLabelColors(res[0].PAParea, Kir=True, y=True)

            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"FullPotassiumAmp{self.tag}_{comparison}.pdf",
                )
            )
            if comparison == "PAPLen":
                self.plotIKSeries(results, tagReset=True, setKoylim=True)

    def runPotassiumComparison(self, comparison, iterations, maxStep=10, intermStep=1):
        comm.Barrier()
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "Glu": self.GluStim,
                "GABA": False,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "GluTrans": self.optGluT,
            }
        )
        # if self.NMDAR:
        #     funcArgs[-1]["multiple"] = self.optNMDAR
        # else:
        funcArgs[-1]["multiple"] = 0
        if self.GluT:
            funcArgs[-1]["GluTrans"] = self.optGluT

        ccList = ["kir2", comparison]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        if self.KStim:
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
            imArray = np.zeros(
                (int(self.KirMax / self.KirStep) + 1, int(maxStep / intermStep) + 1)
            )
            for res in results:
                imArray[
                    int(res[0].GENEDict["kir2"] / self.KirStep),
                    int(getattr(res[0], comparison) / intermStep),
                ] += (
                    max(res[0].vPAP) - res[0].RMP
                )

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
            if comparison == "KoSize":
                skip = 1
                dec = 0
                printType = int
                plt.xlabel("$\Delta$extracellular [K] (mM)")
            elif comparison == "durStim":
                skip = 2
                dec = 0
                printType = int
                plt.xlabel("stim. duation (ms)")
            else:
                skip = 2
                dec = 1
                printType = float
                plt.xlabel("PAP Length (\u03bcm)")
            plt.xticks(
                np.arange(0, int(maxStep / intermStep) + 1, 1),
                np.round(
                    np.arange(0, maxStep + intermStep / 2, intermStep), decimals=dec
                ).astype(printType),
            )
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 20, 2), extend="max")
            plt.clim((0, 20))
            if comparison == "PAPLen":
                self.GluT = False  # just to force plot setLabel Colors
            self.setLabelColors(res[0].PAParea, Kir=True, y=True)

            plt.savefig(
                os.path.join("../results/paperRes", f"FullPotassium{self.tag}.pdf")
            )

            if comparison == "KoSize":
                self.plotIKSeries(results)

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
            plt.title(f"avg:{sum(LambdaList) / len(LambdaList)}")

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

    def optDepolarizationSearch(self, x, optmV=4.0):
        # 28.812 mV for block
        # 10.0 for OE
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        gluCount, stimAmp, papLen = x
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "Glu": False,
                "kir2": self.optKir,
                "clleak": 0,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "multiple": 0,
                "GluTrans": int(gluCount),
                "PAPLen": papLen,
            }
        )
        if funcArgs[-1]["kir2"] > 300:
            funcArgs[-1]["dt"] *= 0.1
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        cells.multiSpike(number=10, freq=100, KoSize=0.5, amp=stimAmp)
        cells.run()
        print(x)
        print(abs(max(list(cells.vPAP)) - cells.RMP - optmV))
        return abs(max(list(cells.vPAP)) - cells.RMP - optmV)

    def readExpRawData(self, results):
        df = pd.read_csv("./Data/depolarTime.csv")
        stimIndex = 5
        # calibrate to relative point from stimulus onset
        for c in df.columns:
            if c == "V":
                avgV = df[c][:stimIndex].mean()
                df[c] = df[c] - avgV
            else:
                df[c] = df[c] - df[c][stimIndex]
        # Match stim initialization with model
        df["V"] *= 10
        df["t"] = df["t"] + (results.initTstop + results.stimdelay) * ms
        return df

    def fitExpDepolarization(self, x, showFig=False, PAP=True):
        self.addChannelTag()
        AllCells = []
        funcArgs = []
        TWIK = 1
        leak = 3e5
        # maybe bug for result fit check how parms should change with diffrenet leak value
        forcedAccum = None
        if PAP:
            k = 500
            mprint(x)
            NMDAR, s, d, tau2 = x
            tau1 = 1.69
            Kir = 120
            funcArgs.append(
                {
                    "mode": 0,
                    "ComplexMorph": True,
                    "Glu": True,
                    "kir2": Kir,
                    "twik": TWIK,
                    "clleak": 0,
                    "naleak": leak,
                    "dt": self.dt,
                    "seed": self.seed,
                    "multiple": NMDAR,
                    "GluTrans": self.optGluT,
                    "KoSize": 3,
                }
            )
        else:
            mprint(x)
            if len(x) == 4:
                glt, kir, PAPLen, KoSize = x
                tau2 = 5.8
                slowing = 1
                forcedAccum = False
            else:
                glt, kir, PAPLen, KoSize, tau2, slowing = x
                forcedAccum = True
            funcArgs.append(
                {
                    "mode": 0,
                    "Glu": True,
                    "kir2": kir,
                    "twik": TWIK,
                    "clleak": 0,
                    "naleak": leak * 10,
                    "dt": self.dt,
                    "seed": self.seed,
                    "multiple": 0,
                    "GluTrans": glt,
                    "KoSize": KoSize,
                    "PAPLen": PAPLen,
                }
            )

        if funcArgs[-1]["kir2"] >= 200:
            funcArgs[-1]["dt"] *= 0.1
        if rank % 3 == 2:
            stim = 1
        elif rank % 3 == 1:
            stim = 5
        else:
            stim = 10
        cells = PAPModel(**funcArgs[-1])
        cells.setTstop(500)
        if not PAP:
            cells.setGLT_TC(0.61, tau2)
        cells.initialize()

        if PAP:
            cells.setNMDA_Mgblock(k, d, s)
            cells.setNMDA_TC(tau1, tau2)
            # cells.setSlowing(slow)
        else:
            cells.setSlowing(slowing)
        cells.multiSpike(number=stim, freq=100)
        if not PAP:
            cells.setGLT_TC(0.61, 5.8)
            cells.setSlowing(1)  # return to normal after neuro activity

        cells.run()
        cells = cells.copyAttr()

        return self.plotExpFit(
            cells,
            stim=stim,
            PAP=PAP,
            showFig=showFig,
            Fname=f"fit{PAP=}{forcedAccum=}",
            correctArtifact=True,
        )

    def artifactCurve(self, x, a, l, c):
        x = np.array(x)
        x[x < 150] = 150
        return -a * np.exp(-(x - 150) / l) + c

    def fitbaselineCurve(self, expData, tdata):
        popt, pcov = curve_fit(
            self.artifactCurve,
            tdata,
            expData,
        )
        return popt

    def plotExpFit(
        self,
        cells,
        stim=10,
        PAP=True,
        verbose=True,
        showFig=True,
        Fname="FitResult",
        correctArtifact=True,
    ):
        plt.cla()
        plt.clf()
        AllCells = []
        sim_time = None
        # get and tweak results
        tList, fList, stdList = procedure.getExpRes(f"./Data/{stim}stim.csv")

        stdList = [np.nan if val == 0 or val is None else val for val in stdList]
        zeroPoint = tList.index(min(tList, key=abs))
        # print(zeroPoint,fList)
        fList = np.array(fList) - fList[zeroPoint]

        tList = np.array(tList) + int(cells.initTstop + cells.stimdelay)

        fluorTrace = (np.array(list(cells.fluorVPAP)) - cells.RMP) * -1 / 10
        simV = np.array(list(cells.vPAP)) - cells.RMP  # use raw sim data for plot

        # extract corresponding indexes in df and sim
        indexConvert = [
            (i, int(t / cells.dt))
            for i, t in enumerate(tList)
            if 0 <= t < max(cells.time)
        ]

        expT = []
        expF = []
        expSTD = []
        simF = []
        for j, k in indexConvert:
            # print(j,k)
            # print(len(fluorTrace))
            # if max(df["V"]) == df["V"][j]:
            #     maxIndex = j + 1
            expT.append(tList[j])
            expF.append(fList[j])  # f to mV
            expSTD.append(stdList[j])
            # simV.append(voltTrace[k])
            simF.append(fluorTrace[k])  # V to mV
            # print(simV,simF)
        expF = np.array(expF)
        expSTD = np.absolute(expSTD)
        if correctArtifact:
            singleStimBaseline = comm.bcast(expF, root=2)
            singleStimBaselineT = comm.bcast(expT, root=2)
            spl = spline(singleStimBaselineT, singleStimBaseline)
            simF += spl(expT)
        loss = np.absolute(expF - simF)

        stdComp = loss - expSTD
        loss = sum(loss[(stdComp >= 0)] ** 2)

        if len(simV) < len(tList) or np.isnan(simF).any():
            loss = np.inf

        # MLS
        # trueloss = sum((expF - simV)**2)
        # lossRMP = self.optRMPSearch((leak,Kir))
        if not np.isnan(loss) and verbose:
            print(f"Loss:{loss}@rank{rank}")
        loss = comm.gather(loss, root=0)
        if showFig:
            sim_time = cells.time
            expF = comm.gather(expF, root=0)
            expT = comm.gather(expT, root=0)
            expSTD = comm.gather(expSTD, root=0)
            simV = comm.gather(simV, root=0)
            simF = comm.gather(simF, root=0)
            stim = comm.gather(stim, root=0)
            fluorTrace = comm.gather(fluorTrace, root=0)
            AllCells = comm.gather(cells, root=0)
        total = 0
        comm.Barrier()
        if rank == 0 and showFig:
            fig, ax1 = plt.subplots(figsize=(9, 6))
            ax2 = ax1.twinx()

            for l in loss:
                total += l
            color = {10: "tab:blue", 5: "tab:orange", 1: "tab:green"}
            for i, t, f, yerr, sim_v, sim_f in zip(
                stim, expT, expF, expSTD, simV, fluorTrace
            ):
                ax1.plot(
                    sim_time,
                    sim_v,
                    label=f"{i} stim simulation",
                    linestyle="-",
                    color=color[i],
                    alpha=0.5,
                    zorder=i,
                )
                if correctArtifact:
                    sim_f += spl(sim_time)
                    label = f"corrected\n{i} stim simulation"
                else:
                    label = f"{i} stim simulation"
                ax2.plot(
                    sim_time,
                    sim_f,
                    label=label,
                    linestyle="--",
                    color=color[i],
                    zorder=100 + i,
                )
                ax2.errorbar(
                    t, f, yerr=yerr, fmt="none", color=color[i], zorder=200 + i
                )

                ax2.scatter(
                    t, f, label=f"{i} stim experiment", color=color[i], zorder=201 + i
                )
            ax1.set_xlim((100, 500))
            ax1.legend(loc="upper left", edgecolor=self.returnColor("model"))
            ax2.legend(loc="upper right", edgecolor=self.returnColor("fluor"))
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Membrane potential change (mV)")
            ax2.set_ylabel("$\Delta F/F_0$ (%)")
            ylim_value = 80  # mv
            ax2.set_ylim((0, ylim_value * -1 / 10))
            ax1.set_ylim((0, ylim_value))
            for axObj, label in {ax1: "model", ax2: "fluor"}.items():
                axObj.tick_params(axis="y", colors=self.returnColor(label))
                axObj.yaxis.label.set_color(self.returnColor(label))

            if correctArtifact:
                Fname += "_correctedArtifact"

            plt.savefig(f"../results/paperRes/{Fname}.pdf")
            if np.isnan(total):
                total = np.inf
            elif verbose:
                print(f"{total=}")

            self.plotIKSeries(
                [AllCells],
                setKoylim=True,
                setekylim=True,
                setyLim=[-15, 1],
                tagReset=True,
            )

        total = comm.bcast(total, root=0)
        sys.stdout.flush()
        comm.Barrier()
        return total

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
        if funcArgs[-1]["kir2"] > 300:
            funcArgs[-1]["dt"] *= 0.2
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        cells.setK(KoSize=float(x))
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
        cells.multiSpike(number=number, freq=freq, KoSize=self.ko)
        print(x)
        print(abs(max(list(cells.vPAP)) - cells.RMP - optmV))
        return abs(max(list(cells.vPAP)) - cells.RMP - optmV)

    def compareLen(self):
        self.addChannelTag()
        controlLeak = self.leak
        controldt = self.dt
        self.NMDAR = True
        for kir in [120, 0, 720]:
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
                [
                    (i, j * 0.001)
                    for i in range(
                        2, self.channelCompareMax + 1, self.channelCompareStep
                    )
                    for j in range(300, 1501, 300)
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
                    "Glu": self.GluStim,
                    "ComplexMorph": True,
                    "naleak": self.leak,
                    "clleak": 0,
                    "dt": self.dt,
                    "seed": self.seed,
                    "stimdelay": self.stimdelay,
                    "PAPCount": self.PAPCount,
                    "GluTrans": self.optGluT,
                    "kir2": kir,
                }
            )
            ccList = ["multiple", "PAPLen"]
            # results are collected only on rank 0
            callMethods = [[]]
            callArgs = [[]]
            if self.KStim:
                callMethods[0] += ["initialize", "multiSpike", "run"]
                callArgs[0] += [
                    {},
                    {"number": self.stimCount, "KoSize": self.ko, "freq": self.freq},
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
                self.plotHeatmap(results, tag=f"{self.tag}_Kir{kir}_CompLen", Kir=False)

    def optRMPSearch(self, x, optmV=-76.3):
        # x = leak value
        leak, kir = x
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        print(x)
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "kir2": 0,
                "clleak": 0,
                "naleak": leak,
                "dt": self.dt,
                "seed": self.seed,
                "Glu": False,
                "multiple": 0,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        print(abs(cells.RMP - optmV))
        lossKO = (cells.RMP - optmV) ** 2
        print(x)
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "kir2": kir,
                "clleak": 0,
                "naleak": x,
                "dt": self.dt,
                "seed": self.seed,
                "Glu": False,
                "multiple": 0,
            }
        )
        cellsRMP = PAPModel(**funcArgs[-1])
        cellsRMP.initialize()
        print(abs(cellsRMP.RMP + 80))
        loss = (cellsRMP.RMP + 80) ** 2
        return lossKO + loss

    def freqComparison(self):
        self.addChannelTag()
        # Calculate the number of iterations for all parm sets
        spikeNumStep = 1
        spikeFreqStep = 20
        spikeNumMin = 1
        spikeNumMax = 11
        spikeFreqMax = 200
        iterations = comm.bcast(
            [
                (i, j)
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
                "Glu": self.GluStim,
                "ComplexMorph": True,
                "naleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
            }
        )
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        ccList = ["multiSpike"]
        callMethods = [[]]
        callArgs = [[]]
        callMethods[0] += ["setTstop", "initialize", "multiSpike", "run"]
        callArgs[0] += [
            {"tstop": 1000 / spikeFreqStep * spikeNumMax + 10},
            {},
            {"number": "parallelItem1", "KoSize": self.ko, "freq": "parallelItem2"},
            {},
        ]

        if self.ek != None:
            self.ko = self.nernstINV(ek, 80)  # 80 defined in neuron astrocyte.hoc
            rIndex = callMethods[0].index("run")
            callArgs[0][rIndex]["ko"] = self.ko

        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            callMethods,
            callArgs,
            mode="MethodArgs",
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
            imArray = np.zeros(
                (
                    int(spikeNumMax / spikeNumStep) + 1,
                    int(spikeFreqMax / spikeFreqStep) + 1,
                )
            )
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
            plt.ylabel("Number of stimuli")
            plt.xlabel("Frequency (Hz)")
            plt.xticks(
                range(0, int(spikeFreqMax / spikeFreqStep) + 1, 2),
                np.arange(0, spikeFreqMax + 1, spikeFreqStep * 2),
            )
            plt.yticks(
                range(int(spikeNumMax / spikeNumStep) + 1),
                np.arange(0, spikeNumMax + 1, spikeNumStep),
            )
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 50, 10), extend="max")
            plt.clim((0, 50))
            plt.savefig(
                os.path.join("../results/paperRes", f"FreqComparison{self.tag}.pdf")
            )

    @staticmethod
    def extract_and_pair_f_by_gid(csv_file_path):
        df = pd.read_csv(csv_file_path)
        res = []
        t = []

        for target_gid in range(min(df["gId"]), max(df["gId"])):
            # Filter rows by gId
            filtered = df[df["gId"] == target_gid]
            t.append(df[df["gId"] == target_gid]["t"].iloc[0])
            # Reset index and extract 'f' column
            f_values = filtered["f"].reset_index(drop=True)

            # Pair f values into tuples
            f_pairs = [
                (f_values[i], f_values[i + 1]) for i in range(0, len(f_values) - 1, 2)
            ]

            # Handle odd number of values
            if len(f_values) % 2 != 0:
                f_pairs.append((f_values.iloc[-1], None))
            res += f_pairs
        return t, res

    @staticmethod
    def getExpRes(fName):
        t, res = procedure.extract_and_pair_f_by_gid(fName)
        yerr = [abs(m - s) if s is not None else 0 for m, s in res]
        f, _ = list(zip(*res))
        return t, f, yerr

    def plotExpRes(self):
        for fTag in [1, 5, 10]:
            t, f, yerr = procedure.getExpRes(os.path.join("./Data", f"{fTag}stim.csv"))
            plt.scatter(t, f, label=f"{fTag} stim")
            plt.errorbar(t, f, yerr=yerr)
            plt.ylim((0, -5))
        plt.legend()
        plt.savefig()


if __name__ == "__main__":
    if size == 3:
        mprint("exp fit")
        testBools = [True, False]
        for PAP in testBools:
            for forcedAccum in [True, False]:
                exp = procedure(3, 0)
                kwargs = {"PAP": PAP, "showFig": False}
                if PAP:
                    # initParms = (5, -72.5, 10, 19,0.5)
                    initParms = (8, -67.5, 10, 19)
                    bounds = [(1, 10), (-80, -70), (10, 50), (4, 50)]
                else:
                    # initParms = (5, -72.5, 10, 19,0.5)
                    if forcedAccum:
                        initParms = (100, 800, 5, 50, 900, 10000)
                        bounds = [
                            (-1000, 1000),
                            (90, 150),
                            (1, 10),
                            (0.5, 30),
                            (5.8, 800),
                            (0, 10000),
                        ]
                    else:
                        initParms = (100, 800, 5, 50)
                        bounds = [(-1000, 1000), (90, 150), (1, 10), (0.5, 30)]

                exp.fitExpDepolarization(initParms, showFig=True, PAP=PAP)
                #                res = minimize(
                #                    lambda x: exp.fitExpDepolarization(x, **kwargs),
                #                    initParms,
                #                    method="Nelder-Mead",
                #                    bounds=bounds,
                #                )
                #                exp.fitExpDepolarization(res.x, showFig=True, PAP=PAP)
                if PAP:
                    break
    elif size == 5 or size == 2 or size == 4:
        mprint("running bathExp")
        exp = procedure(4, 0)
        if size == 2:
            exp.bathExperiment()
        else:
            exp.bathExperiment(invivo=True)
    else:
        # exp = procedure(6,0)
        # initParms = (5, -71, 10, 19)
        # exp.GABAR=False
        # exp.stimCount = 10
        # exp.singleRun(*initParms)

        exp = procedure(6, 0)
        # exp.gababathExperiment()
        exp.uptakeRatio()
        exp.branchAttenuation(replay=True)

        # print('Nothing set; try MPI -n 2 for kbath; MPI -n 3 for exp fit')
