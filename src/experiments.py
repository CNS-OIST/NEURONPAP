from scipy.optimize import curve_fit
from astrocyte import *
import os
import numpy as np
from utils import *
from textSDIO import *
from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
from math import floor
import glob
from matplotlib.ticker import MaxNLocator
from plot_shape import *
from scipy.stats import f_oneway, ttest_rel
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import shgo
from scipy.interpolate import CubicSpline as spline
import json
import pandas as pd
import inspect
from functools import wraps
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.text as mtext
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, ConnectionPatch


from global_labels import gl

font = {
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 15,
}

plt.rcParams.update(font)
plt.ioff()


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        if "10" in orig_handle:
            color = "tab:blue"
        elif "5" in orig_handle:
            color = "tab:orange"
        else:
            color = "tab:green"
        title = mtext.Text(
            x0,
            y0,
            r"\textbf{" + orig_handle + "}",
            usetex=True,
            color=color,
            **self.text_props,
        )
        handlebox.add_artist(title)
        return title


class plotFigures:
    k_color = "orange"
    gaba_color = "purple"
    colorDict = {
        "NMDAR": "steelblue",
        "GABAR": gaba_color,
        "GABA$_A$R": gaba_color,
        "GluT": "lightblue",
        "iK": k_color,
        "K$^+$": k_color,
        "Soma": "deepskyblue",
        "PAP": "forestgreen",
        "fluor": "black",
        "Na": "gold",
        "Cl": "chocolate",
        "Ca": "olive",
        "model": "darkgray",
        "local": "white",
        "global": "darkgray",
    }

    @staticmethod
    def forceAlpha(color, alpha, bkg=(255, 255, 255, 1)):
        rgba_color = np.array(mcolors.to_rgba(color)) * np.array([255, 255, 255, 1])
        return (rgba_color * alpha + np.array(bkg) * (1 - alpha)) / np.array(
            [255, 255, 255, 1]
        )

    def save_src_Data(plot_func):
        @wraps(plot_func)
        def wrapper(self, *args, **kwargs):
            caller = inspect.stack()[1].function
            if self.global_rw_data or rank == 0:
                AllCells = args[0]
                if type(AllCells) is list:

                    tmpCells = AllCells
                    while type(tmpCells) is list:
                        tmpCells = tmpCells[0]
                else:
                    tmpCells = AllCells
                fName = f"{caller}{self.tag}.pickle"
                # default no overwrite
                fPath = os.path.join("intermediaryData", fName)
                if not os.path.isfile(fPath):
                    with open(fPath, "wb") as handle:
                        pickle.dump(AllCells, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Saved src data file {fName}")
            res = plot_func(self, *args, **kwargs)
            return res

        return wrapper

    @save_src_Data
    def free_figure(self, AllCells):
        # just to force call decorator
        pass

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

    def get_initStep(self, cell, shift=10):
        if hasattr(cell, "cvode") and cell.cvode:
            # get index of initTstop
            tmp_time = np.array(cell.time)
            initStep = np.argmin(abs(tmp_time - (cell.initTstop - shift)))
        else:
            initStep = int((cell.initTstop - shift) / cell.dt)
        return int(initStep)

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

    @save_src_Data
    def GABANMDARTrace(
        self, AllCells, NMDARCount, GABACount, fName="NMDAR_GABAR_TraceComp"
    ):
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots()

        for cells in AllCells:
            for cell in cells:
                initStep = self.get_initStep(cell)
                if NMDARCount == cell.multiple and cell.GABACount == 0:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.vPAP)[initStep:],
                        label=f"NMDAR {gl.vm}({cell.multiple})",
                        color=self.returnColor("NMDAR"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.fluorVPAP)[initStep:],
                        label="NMDAR fluor",
                        color=self.returnColor("NMDAR"),
                        linestyle="-.",
                    )
                if GABACount == cell.GABACount and cell.multiple == None:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.vPAP)[initStep:],
                        label=f"GABA$_A$R {gl.vm}({int(cell.GABADensity)})",
                        color=self.returnColor("GABAR"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.fluorVPAP)[initStep:],
                        label="GABA$_A$R fluor",
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

    @save_src_Data
    def plotIKSeries(
        self,
        AllCells,
        zoom=False,
        setyLim=None,
        setKoylim=False,
        setekylim=False,
        showFluor=False,
        define_initStep=None,
        bath=False,
        tagReset=False,
        panelF=True,
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
                    self.plotIKSeries.__wrapped__(
                        self, [[cell]], zoom=True, setekylim=setekylim
                    )
                    self.tag = tmpTag

                if not define_initStep:
                    # make globally defined
                    initStep = self.get_initStep(cell)
                else:
                    initStep = self.get_initStep(
                        cell, shift=cell.initTstop - define_initStep
                    )
                if bath:
                    if max(cell.time) > 2e3:
                        cell.time *= 1e-3
                        second = True
                        startStim = 23
                        endStim = 33
                    else:
                        second = False
                        startStim = int(cell.initTstop)
                        endStim = max(cell.time) + 10
                    plt.subplots_adjust(wspace=2)
                    fig, (ax, ax2) = plt.subplots(1, 2)
                else:
                    fig, ax = plt.subplots()

                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.KoPAP)[initStep:],
                    label=f"PAP",
                    color=self.returnColor("PAP"),
                )
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.KoSoma)[initStep:],
                    label="Soma",
                    color=self.returnColor("Soma"),
                )
                if setKoylim:
                    _, _, _, ymax = ax.axis()

                    if ymax > 30:
                        ax.set_ylim((0, 60))
                    else:
                        ax.set_ylim(gl.lim_ko)
                else:
                    ax.set_ylim(gl.lim_ko)
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
                    lowerBound, upperBound = gl.lim_ko
                    startBar = (upperBound + lowerBound) / 2
                    lineWidth = 8
                    if self.locality == "local":
                        height_in_inches = lineWidth / 72
                        dpi = fig.dpi
                        height_in_pixels = height_in_inches * dpi
                        print(height_in_pixels)

                        inv = ax.transData.inverted()
                        p1 = inv.transform((0, 0))
                        p2 = inv.transform((0, height_in_pixels))
                        height_data = p2[1] - p1[1]
                        print(height_data)
                        r = patches.Rectangle(
                            xy=(startStim, startBar),
                            width=endStim - startStim,
                            height=height_data,
                            fc="w",
                            ec="k",
                            label=f"Local stim",
                        )
                        ax.add_patch(r)
                    else:
                        ax.hlines(
                            startBar,
                            startStim,
                            endStim,
                            color=self.returnColor(self.locality),
                            linewidth=lineWidth,
                            label=f"Global stim",
                        )

                if bath:
                    if second:
                        ax.set_xlabel(gl.s)
                    else:
                        if zoom:
                            ax.set_xlim(
                                gl.lim_zoom(
                                    initStep - 50,
                                    cell.dt,
                                    time_frame=cell.tstop - cell.initTstop + 50,
                                    cvode=list(cell.time)[initStep],
                                )
                            )

                        ax.set_xlabel(gl.ms)
                else:
                    ax.set_xlabel(gl.ms)

                ax.set_ylabel(gl.ion_o("K"))
                ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax.legend()

                if zoom:
                    ax.set_xlim(
                        gl.lim_zoom(initStep, cell.dt, cvode=list(cell.time)[initStep])
                    )

                if not bath:
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            "../results/paperRes",
                            f"KoCon{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                        )
                    )

                if bath:
                    ax = ax2
                    ax.tick_params(
                        "y", right=False, labelright=False, left=True, labelleft=True
                    )
                else:
                    plt.cla()
                    plt.clf()
                    fig, ax = plt.subplots()

                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.vPAP)[initStep:],
                    label=f"PAP {gl.vm}",
                    color=self.returnColor("PAP"),
                )
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.ekPAP)[initStep:],
                    label=f"PAP {gl.ek_raw}",
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
                        label=f"Soma {gl.vm}",
                        color=self.returnColor("Soma"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.ekSoma)[initStep:],
                        label=f"Soma {gl.ek_raw}",
                        color=self.returnColor("Soma"),
                        linestyle="--",
                    )

                # must set before calculating height for rectangle
                if setekylim:
                    ax.set_ylim(gl.lim_ek)

                if bath:
                    lowerBound, upperBound = gl.lim_ek
                    startBar = (upperBound + lowerBound) / 2
                    lineWidth = 8
                    if self.locality == "local":
                        height_in_inches = lineWidth / 72
                        dpi = fig.dpi
                        height_in_pixels = height_in_inches * dpi

                        inv = ax.transData.inverted()
                        p1 = inv.transform((0, 0))
                        p2 = inv.transform((0, height_in_pixels))
                        height_data = p2[1] - p1[1]
                        r = patches.Rectangle(
                            xy=(startStim, startBar),
                            width=endStim - startStim,
                            height=height_data,
                            fc="w",
                            ec="k",
                            label=f"Local stim",
                        )
                        ax.add_patch(r)
                    else:
                        ax.hlines(
                            startBar,
                            startStim,
                            endStim,
                            color=self.returnColor(self.locality),
                            linewidth=lineWidth,
                            label=f"Global stim",
                        )
                    if second:
                        ax.set_xlabel(gl.s)
                    else:
                        if zoom:
                            ax.set_xlim(
                                gl.lim_zoom(
                                    initStep - 50,
                                    cell.dt,
                                    time_frame=cell.tstop - cell.initTstop + 50,
                                    cvode=list(cell.time)[initStep],
                                )
                            )
                        ax.set_xlabel(gl.ms)

                else:
                    ax.set_xlabel(gl.ms)

                ax.set_ylabel(gl.volt)
                ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax.legend()

                if zoom:
                    ax.set_xlim(
                        gl.lim_zoom(initStep, cell.dt, cvode=list(cell.time)[initStep])
                    )

                plt.tight_layout()
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
                # ax.plot(
                #    list(cell.time)[initStep:],
                #    list(cell.CaiPAP)[initStep:],
                #    label=f"PAP Cai",
                #    color=self.returnColor("Ca"),
                # )
                ax.set_xlabel(gl.ms)
                ax.set_ylabel(gl.free("Conc. (mM)"))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
                if zoom:
                    ax.set_xlim(
                        gl.lim_zoom(initStep, cell.dt, cvode=list(cell.time)[initStep])
                    )
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
                    label=gl.current_ion("K"),
                    color=self.returnColor("iK"),
                )
                if hasattr(cell, "iNaPAP"):
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iNaPAP)[initStep:],
                        label=gl.current_ion("Na"),
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
                        label=gl.current_ion("NMDA"),
                        color=self.returnColor("NMDAR"),
                    )
                if hasattr(cell, "iGABA") and self.GABAR:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGABA)[initStep:],
                        label=gl.current_ion("GABA$_A$R"),
                        color=self.returnColor("GABAR"),
                    )
                if hasattr(cell, "iGluT") and self.GluT:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluT)[initStep:],
                        label=gl.current_ion("GluT"),
                        color=self.returnColor("GluT"),
                    )
                ax.set_xlabel(gl.ms)
                ax.set_ylabel(gl.curr)
                if setyLim != None:
                    ax.set_ylim(setyLim)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if zoom or self.stimCount > 1:
                    ax.legend(loc="lower left")
                else:
                    ax.legend(loc="lower right")

                if zoom:
                    ax.set_xlim(
                        gl.lim_zoom(initStep, cell.dt, cvode=list(cell.time)[initStep])
                    )
                plt.tight_layout()
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
                    label=gl.current_ion("K"),
                    color=self.returnColor("iK"),
                )
                if hasattr(cell, "iNaSoma"):
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iNaSoma)[initStep:],
                        label=gl.current_ion("Na"),
                        color=self.returnColor("Na"),
                    )
                if hasattr(cell, "iClSoma"):
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iClSoma)[initStep:],
                        label=gl.current_ion("Cl"),
                        color=self.returnColor("Cl"),
                    )
                if hasattr(cell, "iGluTSoma") and self.GluT:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluTSoma)[initStep:],
                        label=gl.current_ion("GluT"),
                        color=self.returnColor("GluT"),
                    )
                ax.set_xlabel(gl.ms)
                ax.set_ylabel(gl.curr)
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
                ax3.set_ylabel(gl.volt)
                ax3.set_ylim(gl.lim_curr)
                if zoom:
                    ax.set_xlim(
                        gl.lim_zoom(initStep, cell.dt, cvode=list(cell.time)[initStep])
                    )

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"iSomaPlot{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                if (
                    panelF
                    and cell.GENEDict["kir2"] == 0
                    and not self.NMDAR
                    and not self.GABAR
                ):
                    plt.cla()
                    plt.clf()
                    fig = plt.figure(figsize=(9, 9))

                    ax = fig.add_axes([0.1, 0.52, 0.8, 0.40])
                    ax_inset = fig.add_axes([0.1, 0.15, 0.8, 0.25])
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.vPAP)[initStep:],
                        label=f"PAP {gl.vm}",
                        color=self.returnColor("PAP"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.ekPAP)[initStep:],
                        label=f"PAP {gl.ek_raw}",
                        color=self.returnColor("PAP"),
                        linestyle="--",
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.vSoma)[initStep:],
                        label=f"Soma {gl.vm}",
                        color=self.returnColor("Soma"),
                    )
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.ekSoma)[initStep:],
                        label=f"Soma {gl.ek_raw}",
                        color=self.returnColor("Soma"),
                        linestyle="--",
                    )

                    ax.set_xlabel(gl.ms)
                    ax.legend()
                    ax.set_ylim(gl.lim_ek)
                    ax_inset.plot(
                        list(cell.time)[initStep:],
                        list(cell.vPAP)[initStep:],
                        label=f"PAP {gl.vm}",
                        color=self.returnColor("PAP"),
                    )

                    x0, x1 = 145, 155
                    y0, y1 = gl.lim_ek
                    grey = "0.5"

                    ax_inset.plot(
                        list(cell.time)[initStep:],
                        list(cell.ekPAP)[initStep:],
                        label=f"PAP {gl.ek_raw}",
                        color=self.returnColor("PAP"),
                        linestyle="--",
                    )
                    ax_inset.set_xlim((x0, x1))
                    ax_inset.set_ylim(y0, y1)

                    rect = Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        linewidth=1.5,
                        edgecolor=grey,
                        zorder=2,
                    )
                    ax.add_patch(rect)

                    ax.set_zorder(2)
                    ax_inset.set_zorder(2)

                    for spine in ax_inset.spines.values():
                        spine.set_color(grey)
                        spine.set_zorder(5)

                    ax_inset.tick_params(colors=grey)

                    con1 = ConnectionPatch(
                        xyA=(x0, y0),
                        coordsA=ax.transData,
                        xyB=(x0, y1),
                        coordsB=ax_inset.transData,
                        color=grey,
                        linewidth=1,
                        zorder=3,
                        linestyle="--",
                    )
                    con2 = ConnectionPatch(
                        xyA=(x1, y0),
                        coordsA=ax.transData,
                        xyB=(x1, y1),
                        coordsB=ax_inset.transData,
                        color=grey,
                        linewidth=1,
                        zorder=3,
                        linestyle="--",
                        connectionstyle="arc3,rad=0.1",
                    )

                    fig.add_artist(con1)
                    fig.add_artist(con2)

                    ax.set_ylabel(gl.volt, zorder=4)
                    ax_inset.set_xlabel(gl.ms, color=grey, zorder=4)
                    ax_inset.set_ylabel(gl.volt, color=grey, zorder=4)

                    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                        lbl.set_zorder(5)

                    for lbl in ax_inset.get_xticklabels() + ax_inset.get_yticklabels():
                        lbl.set_zorder(5)

                    plt.savefig(
                        os.path.join(
                            "../results/paperRes",
                            f"panelFPlot{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                        )
                    )

                plt.close("all")

    @save_src_Data
    def mergePlotsIK(self, AllCells, comparison, merge, selected=1, zoom=True):
        AllRes = {}
        AllRecVals = ["vPAP", "KoPAP", "ekPAP"]

        total = []
        if zoom:
            self.mergePlotsIK.__wrapped__(
                self, AllCells, comparison, merge, selected=selected, zoom=False
            )
        for cells in AllCells:
            for cell in cells:
                compVal = getattr(cell, comparison)
                if compVal not in AllRes.keys():
                    AllRes[compVal] = {}
                AllRes[compVal][getattr(cell, merge)] = cell
        for compVal in AllRes.keys():
            for recVal in AllRecVals:
                plt.cla()
                plt.clf()
                if compVal == 16:
                    total = [
                        max(getattr(cell, recVal)) for cell in AllRes[compVal].values()
                    ]
                    total = np.array(total)
                    total = np.unique(total)

                    mprint(total.mean(), total.std())
                for cell in AllRes[compVal].values():
                    alpha = 1
                    if getattr(cell, merge) != selected:
                        alpha = 0.3
                    initStep = self.get_initStep(cell)
                    plt.plot(
                        list(cell.time)[initStep:],
                        list(getattr(cell, recVal))[initStep:],
                        alpha=alpha,
                        label=getattr(cell, merge),
                        zorder=(
                            100
                            if getattr(cell, merge) == selected
                            else getattr(cell, merge)
                        ),
                    )
                if zoom:
                    plt.xlim(
                        gl.lim_zoom(initStep, cell.dt, cvode=list(cell.time)[initStep])
                    )
                plt.legend(title="seed", title_fontsize="x-small", fontsize="xx-small")
                plt.xlabel(gl.ms)
                if recVal == "vPAP":
                    plt.ylabel(gl.volt)
                else:
                    plt.ylabel(recVal)
                plt.ylim(gl.lim_Vmemb)
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"{recVal}_Merged_for_{comparison}={compVal}_over_{merge}_{zoom=}.pdf",
                    )
                )
                plt.close("all")

    def setLabelColors(self, area, Kir=True, x=False, y=False, chanOverride=None):
        stdChannelDict = {
            "Kir": (370 * area + 1 * 4.7e3 * area, 1 * area),
            "GluT": (14248 * area, 812 * area),
            "GABAR": (np.inf, 0),
            "GABA$_A$R": (np.inf, 0),
            "PAPLen": (
                0.425,
                0.225,
            ),  # 95th percentile of node sizes from Arizono M. Nat Comm. (2020)
        }
        print(area)
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
                if abs(float(l.get_text()) - mean) > std:
                    l.set_color("grey")

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
                if abs(float(l.get_text()) - mean) > std:
                    l.set_color("grey")

    def combined_heatmap(self, results, PAPattr, Kir=True):
        caller = inspect.stack()[3].function
        fName = f"{caller}{self.tag}.pickle"
        if "_multiSpikex10" in fName:
            singleFName = "".join(fName.split("_multiSpikex10"))
        else:
            return

        fPath = os.path.join("intermediaryData", singleFName)
        if os.path.isfile(fPath):
            with open(fPath, "rb") as handle:
                AllCells = pickle.load(handle)

        else:
            return

        single_res = AllCells

        imarray_multi = self.createIMArray(results, PAPattr, Kir=Kir)
        imarray_single = self.createIMArray(single_res, PAPattr, Kir=Kir)

        vmin, vmax = gl.clim_volt
        plt.cla()
        plt.clf()
        plt.close("all")

        fig, axes = plt.subplots(1, 2, figsize=(9, 9), sharey=True)

        im1 = axes[0].imshow(
            imarray_single,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            origin="lower",
            interpolation="nearest",
        )
        im2 = axes[1].imshow(
            imarray_multi,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            origin="lower",
            interpolation="nearest",
        )

        res = results[0]

        for ax in axes:
            if self.GluT:
                addChan = 1
                chanStart = -1 * int(self.channelCompareMax / self.channelCompareStep)
                skip = 2
                xlabels = (
                    np.arange(
                        chanStart,
                        int(self.channelCompareMax / self.channelCompareStep) + 1,
                        skip,
                    )
                    * self.channelCompareStep
                    * res[0].PAPGluTCount_std
                    + res[0].PAPGluTCount
                )
                xlabels = [int(val) if val > 0 else 0 for val in xlabels]

                ax.set_xticks(
                    range(
                        0,
                        2 * int(self.channelCompareMax / self.channelCompareStep)
                        + addChan,
                        skip,
                    ),
                    xlabels,
                    rotation=45,
                    ha="right",
                )
            else:
                chanStart = 0
                addChan = 1

            if not self.GluT:
                ax.set_xticks(
                    range(
                        0,
                        int(self.channelCompareMax / self.channelCompareStep) + addChan,
                    ),
                    np.arange(
                        chanStart,
                        int(self.channelCompareMax / self.channelCompareStep) + 1,
                        1,
                    )
                    * self.channelCompareStep,
                    rotation=45,
                    ha="right",
                )
        if self.NMDAR:
            xlabel = gl.chan_num("NMDAR")
        elif self.GluT:
            xlabel = gl.chan_num("GLT-1")
        elif self.GABAR and Kir:
            xlabel = gl.chan_num("GABA$_A$R")
        elif self.GAP:
            xlabel = gl.chan_num("Cx43")
        else:
            xlabel = None
        if Kir:
            ytick_labels = (
                np.arange(
                    -1 * int(self.KirMax / self.KirStep),
                    int(self.KirMax / self.KirStep) + 1,
                    1,
                )
                * self.KirStep
                * res[0].PAPKirCount_std
                + res[0].PAPKirCount
            )
            ytick_labels /= res[0].PAPCount
            ytick_labels[ytick_labels < 0] = 0
            ytick_labels = ytick_labels.astype(int)

            axes[0].set_yticks(
                range(2 * int(self.KirMax / self.KirStep) + 1),
                ytick_labels,
            )
            axes[0].set_ylabel(gl.chan_num("Kir"))
        elif self.GABAR:
            axes[0].set_yticks(
                range(
                    0,
                    int(self.channelCompareMax / self.channelCompareStep) + addChan,
                ),
                np.arange(
                    chanStart,
                    int(self.channelCompareMax / self.channelCompareStep) + 1,
                    1,
                )
                * self.channelCompareStep,
            )
            axes[0].set_ylabel(gl.chan_num("GABA") + f" /{gl.unit_micron_raw}$^2$")
        else:
            axes[0].set_yticks(
                range(0, 5),
                [f"{i:.2f}" for i in np.arange(0.3, 3.1, 0.3)],
            )
            ax[0].set_ylabel(gl.pap_affect)
        _, cbarMax = gl.clim_volt
        fig.colorbar(
            im1,
            label=gl.vm,
            ticks=np.arange(0, cbarMax, 2),
            extend="max",
            ax=axes.ravel().tolist(),
            shrink=0.8 if not self.GluT else 0.5,
        )
        left = axes[0].get_position().x0
        right = axes[1].get_position().x1
        bottom = axes[0].get_position().y0
        top = axes[0].get_position().y1
        fig.text(
            (left + right) / 2,
            bottom - 0.08,
            xlabel,
            fontsize=plt.rcParams["axes.labelsize"],
            ha="center",
            va="top",
        )
        fig.text(
            axes[0].get_position().x0,
            top + 0.01,
            "single stim.",
            fontsize=plt.rcParams["axes.labelsize"],
            ha="left",
            va="bottom",
        )
        fig.text(
            axes[1].get_position().x0,
            top + 0.01,
            "10 stim.",
            ha="left",
            va="bottom",
            fontsize=plt.rcParams["axes.labelsize"],
        )

        plt.savefig(
            os.path.join(
                "../results/paperRes", f"combined_heatmap{PAPattr}{self.tag}.pdf"
            )
        )

    def createIMArray(self, results, PAPattr, Kir=True):
        if Kir:
            if self.GluT:
                imArray = np.zeros(
                    (
                        2 * int(self.KirMax / self.KirStep) + 1,
                        2 * int(self.channelCompareMax / self.channelCompareStep) + 1,
                    )
                )

            else:
                imArray = np.zeros(
                    (
                        2 * int(self.KirMax / self.KirStep) + 1,
                        int(self.channelCompareMax / self.channelCompareStep) + 1,
                    )
                )
        elif self.GABAR or self.GAP:
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
                        int((self.KirMax + res[0].GENEDict["kir2"]) / self.KirStep),
                        int(
                            (self.channelCompareMax + res[0].comparecount)
                            / self.channelCompareStep
                        ),
                    ] += (
                        max(getattr(res[0], PAPattr)) - res[0].RMP
                    )
                elif res[0].comparecount is not None:
                    # bug when stimulus but not GABAR
                    imArray[
                        int((self.KirMax + res[0].GENEDict["kir2"]) / self.KirStep),
                        int(res[0].comparecount / self.channelCompareStep),
                    ] += (
                        max(getattr(res[0], PAPattr)) - res[0].RMP
                    )
            elif self.GABAR:
                # if not Kir and GABA i.e. GABA vs. NMDAR do this
                imArray[
                    int(res[0].GABACount / self.channelCompareStep),
                    int(res[0].comparecount / self.channelCompareStep),
                ] += (
                    max(getattr(res[0], PAPattr)) - res[0].RMP
                )
            else:
                imArray[
                    int(res[0].PAPLen / 0.3) - 1,
                    int(res[0].comparecount / self.channelCompareStep) - 1,
                ] += (
                    max(getattr(res[0], PAPattr)) - res[0].RMP
                )

        return imArray

    @save_src_Data
    def plotHeatmap(self, results, tag="", divedend=1, Kir=True, stdLabels=False):
        plt.cla()
        plt.clf()
        self.plotIKSeries.__wrapped__(
            self,
            results,
            setKoylim=True,
            setekylim=True,
        )

        res = results[0]
        for PAPattr in ["vPAP", "vSoma"]:
            self.combined_heatmap(results, PAPattr, Kir=Kir)
            imArray = self.createIMArray(results, PAPattr, Kir=Kir)
            cmap = "magma"
            plt.cla()
            plt.clf()
            plt.figure(figsize=(7.7, 9))
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
                chanStart = -1 * int(self.channelCompareMax / self.channelCompareStep)
                skip = 2
                xlabels = (
                    np.arange(
                        chanStart,
                        int(self.channelCompareMax / self.channelCompareStep) + 1,
                        skip,
                    )
                    * self.channelCompareStep
                    * res[0].PAPGluTCount_std
                    + res[0].PAPGluTCount
                )
                xlabels = [int(val) if val > 0 else 0 for val in xlabels]

                plt.xticks(
                    range(
                        0,
                        2 * int(self.channelCompareMax / self.channelCompareStep)
                        + addChan,
                        skip,
                    ),
                    xlabels,
                    rotation=45,
                    ha="right",
                )
            else:
                chanStart = 0
                addChan = 1

            if not self.GluT:
                plt.xticks(
                    range(
                        0,
                        int(self.channelCompareMax / self.channelCompareStep) + addChan,
                    ),
                    np.arange(
                        chanStart,
                        int(self.channelCompareMax / self.channelCompareStep) + 1,
                        1,
                    )
                    * self.channelCompareStep,
                    rotation=45,
                    ha="right",
                )
            if Kir:
                ytick_labels = (
                    np.arange(
                        -1 * int(self.KirMax / self.KirStep),
                        int(self.KirMax / self.KirStep) + 1,
                        1,
                    )
                    * self.KirStep
                    * res[0].PAPKirCount_std
                    + res[0].PAPKirCount
                )
                ytick_labels /= res[0].PAPCount
                ytick_labels[ytick_labels < 0] = 0
                ytick_labels = ytick_labels.astype(int)

                plt.yticks(
                    range(2 * int(self.KirMax / self.KirStep) + 1),
                    ytick_labels,
                )
                plt.ylabel(gl.chan_num("Kir"))
            elif self.GABAR:
                plt.yticks(
                    range(
                        0,
                        int(self.channelCompareMax / self.channelCompareStep) + addChan,
                    ),
                    np.arange(
                        chanStart,
                        int(self.channelCompareMax / self.channelCompareStep) + 1,
                        1,
                    )
                    * self.channelCompareStep,
                )
                plt.ylabel(gl.chan_num("GABA") + f" /{gl.unit_micron_raw}$^2$")
            else:
                plt.yticks(
                    range(0, 5),
                    [f"{i:.2f}" for i in np.arange(0.3, 3.1, 0.3)],
                )
                plt.ylabel(gl.pap_affect)
            if self.NMDAR:
                plt.xlabel(gl.chan_num("NMDAR"))
            elif self.GluT:
                # plt.xlabel("Multiple of estimated GluT density")
                plt.xlabel(gl.chan_num("GLT-1"))
            elif self.GABAR and Kir:
                # plt.xlabel("# of GABAR channels / um2")
                plt.xlabel(gl.chan_num("GABA$_A$R"))
            elif self.GAP:
                plt.xlabel(gl.chan_num("Cx43"))
            elif self.NKA:
                plt.xlabel(gl.chan_num("NKA"))
            _, cbarMax = gl.clim_volt
            plt.colorbar(label=gl.vm, ticks=np.arange(0, cbarMax, 2), extend="max")
            plt.clim((0, cbarMax))
            if stdLabels:
                self.setLabelColors(
                    res[0].PAParea,
                    Kir=Kir,
                    x=True,
                    y=True,
                    chanOverride={
                        "GluT": (res[0].PAPGluTCount, res[0].PAPGluTCount_std),
                        "Kir": (res[0].PAPKirCount, res[0].PAPKirCount_std),
                    },
                )

            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"FullComparison{tag}_{PAPattr}.pdf"
                )
            )

    @save_src_Data
    def plot_physiological(self, AllCells, stim, papcounts, models):
        if rank != 0:
            return
            # for cell in AllCells:
            #    print(
            #        cell.PAPCount,
            #        "theta" if not hasattr(cell, "SpikeFreq") else cell.SpikeFreq,
            #    )
        for k, location in enumerate(["vPAP", "vSoma"]):
            index = 0
            for i, p in enumerate(papcounts):
                plt.cla()
                plt.clf()
                plt.close("all")
                fig = plt.figure(figsize= (9,5))
                gs = fig.add_gridspec(nrows=6,ncols=2)
                ax = []
                ax_r = []
                inset = (375,385) 
                inset_y =(-86,-83)
                ax.append(fig.add_subplot(gs[0:2,0]))
                ax.append(fig.add_subplot(gs[2:4,0],sharex=ax[0]))
                ax.append(fig.add_subplot(gs[4:6,0],sharex=ax[0]))
                ax_r.append(fig.add_subplot(gs[4,1]))
                ax_r.append(fig.add_subplot(gs[5,1],sharex=ax_r[0]))
                for j, s in enumerate(stim):
                    if j != len(stim):
                        ax[j].tick_params(labelbottom=False)
                    for m in models:
                        # flattened
                        # Forced may be buggy
                        #
                        cell = AllCells[index + j * len(models)]

                        initStep = self.get_initStep(cell)
                        ax[j].plot(
                            list(cell.time)[initStep:],
                            list(getattr(cell, location))[initStep:],
                            label=m,
                            color=self.returnColor(m),
                        )
                        index += 1
                        if s == 'theta':
                            if "K$^+$" in m:
                                ax_r[0].plot(
                                    list(cell.time)[initStep:],
                                    list(getattr(cell, location))[initStep:],
                                    label=m,
                                    color=self.returnColor(m),
                                )
                                ax_r[0].set_xlim(*inset)
                                ax_r[0].set_ylim(*inset_y)
                                ax_r[0].tick_params(labelbottom=False)
                            elif "GluT" in m:
                                ax_r[1].plot(
                                    list(cell.time)[initStep:],
                                    list(getattr(cell, location))[initStep:],
                                    label=m,
                                    color=self.returnColor(m),
                                )
                                ax_r[1].set_xlim(*inset)
                                ax_r[1].set_ylim(*inset_y)
                                ax_r[1].set_xlabel(gl.ms)

                    if location == "vPAP":
                        ax[j].set_ylim(gl.lim_Vmemb)
                    else:
                        ax[j].set_ylim(gl.lim_VmembSoma)
                    ax[j].set_xlim(right=500)
                index -= len(models) * (len(stim) - 1)
                handle, label = ax[-1].get_legend_handles_labels()
                leg = fig.legend(
                    handle,
                    label,
                    loc="center left",
                    bbox_to_anchor=(0.6, 0.7),
                    fancybox=True,
                    shadow=True,
                    ncol=1,
                )
                ax[-1].set_xlabel(gl.ms)
                # plt.title(f"stim:{s} PAP counts = {p}")
                bottom = ax[-1].get_position().y0
                top = ax[0].get_position().y1

                left = ax[0].get_position().x0
                fig.text(
                    left - 0.07,
                    (bottom + top) / 2,
                    gl.volt,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=plt.rcParams["axes.labelsize"],
                )

                fig.text(
                    left + 0.01,
                    ax[0].get_position().y1 - 0.02,
                    "50 Hz",
                    fontsize=plt.rcParams["axes.labelsize"],
                    ha="left",
                    va="top",
                )
                fig.text(
                    left + 0.01,
                    ax[1].get_position().y1 - 0.02,
                    "100 Hz",
                    ha="left",
                    va="top",
                    fontsize=plt.rcParams["axes.labelsize"],
                )
                fig.text(
                    left + 0.01,
                    ax[2].get_position().y1 - 0.02,
                    "Theta",
                    ha="left",
                    va="top",
                    fontsize=plt.rcParams["axes.labelsize"],
                )
                x0,x1 = inset
                y0,y1 = inset_y
                grey = "0.5"
                rect = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    linewidth=1.5,
                    edgecolor=grey,
                    zorder=2,
                )
                ax[-1].add_patch(rect)


                con1 = ConnectionPatch(
                    xyA=(x1, y0),
                    coordsA=ax[-1].transData,
                    xyB=(x0-1.5, y0),
                    coordsB=ax_r[1].transData,
                    color=grey,
                    linewidth=1,
                    zorder=3,
                    linestyle="--",
                )
                con2 = ConnectionPatch(
                    xyA=(x1, y1),
                    coordsA=ax[-1].transData,
                    xyB=(x0-1, y1),
                    coordsB=ax_r[0].transData,
                    color=grey,
                    linewidth=1,
                    zorder=3,
                    linestyle="--",
                )

                fig.add_artist(con1)
                fig.add_artist(con2)


                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"combined_pap={p}_{location}{self.tag}.pdf",
                    )
                )
        plt.close("all")


class procedure(plotFigures):
    leak = 1.4  # ideal calculated from stable model
    optKir = 0  # std * optkir  + mean
    optNMDAR = 378
    optGABAR = 998
    optGluT = 0  # std * optGluT + mean
    optNKA = 1
    # default NMDAR counts
    channelCompareMax = 500
    channelCompareStep = 100
    # max 390
    seed = int()
    ko = float()
    tag = str()
    OE = False
    NMDAR = True
    GABAR = True
    GAP = False
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

    no_read_data = False
    global_rw_data = False  # default False only for bath exp

    def __init__(self, seed, ko):
        self.seed = seed
        self.ko = ko
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"
        self._KirMax = 5e3
        self._KirStep = self._KirMax / 5

    # no write access to KirMax
    @property
    def KirMax(self):
        return self._KirMax

    @property
    def KirStep(self):
        return self._KirStep

    def addChannelTag(self):
        self.tag = ""
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"

        if self.GluT:
            self.tag += "_Glu"
        if self.NMDAR:
            self.tag += "_NMDAR"
        if self.GABAR:
            self.tag += "_GABAR"
        if self.GAP:
            self.tag += "_GAP"
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
        #
        #

    def read_data(exp_func):
        calling_module_globals = inspect.currentframe().f_back.f_globals
        calling_module_name = calling_module_globals["__name__"]

        @wraps(exp_func)
        def wrapper(self, *args, **kwargs):
            if not self.no_read_data:
                func_name = exp_func.__name__
                if func_name == "free_read_data":
                    func_name = inspect.stack()[1].function

                intermediary_files = os.listdir(os.path.join("intermediaryData"))
                # TODO: rough way to compare tags, think of smarter
                # Mainly aims to keep additional information added during function
                tmptag = self.tag
                self.addChannelTag()
                if len(tmptag) > len(self.tag):
                    self.tag = tmptag

                for f in intermediary_files:
                    if f == f"{func_name}{self.tag}.pickle":
                        if self.global_rw_data:
                            print(f"found intermediary file {f}")
                        else:
                            mprint(f"found intermediary file {f}")
                        sys.stdout.flush()
                        AllCells = [[]]
                        if self.global_rw_data or rank == 0:
                            with open(
                                os.path.join("intermediaryData", f), "rb"
                            ) as handle:
                                AllCells = pickle.load(handle)

                        if not self.global_rw_data:
                            AllCells = comm.bcast(AllCells, root=0)

                        # temporarily override simulation and just output result
                        def parallizeFor_dummy(*args, **kwargs):
                            return AllCells

                        with global_function_override_runtime(
                            "parallizeFor",
                            parallizeFor_dummy,
                            module_name=calling_module_name,
                        ):
                            result = exp_func(self, *args, **kwargs)
                        return result

            else:
                mprint("not reading intermediary data")
            return exp_func(self, *args, **kwargs)

        return wrapper

    @read_data
    def free_read_data(self):
        try:
            res = parallizeFor()
        except TypeError:
            res = None
        return res

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
                    # with open(
                    #    os.path.join("intermediaryData", "resultsParallel.pickle"), "wb"
                    # ) as handle:
                    #    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
                if i == 0:
                    name = "soma"
                else:
                    name = "PAP"
                ax.set_zlabel(gl.d_volt)

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
                "kleak": self.leak,
                "multiple": None,
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
        plt.cla()
        plt.clf()
        funcArgs.append(
            {
                "mode": 2,
                "ComplexMorph": True,
                "dt": self.dt,
                "kleak": self.leak,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "seed": self.seed,
                "GluTrans": self.optGluT,
                "somaCheck": True,
                "voltageClamp": -60,
            }
        )
        if not self.free_read_data():
            cells = PAPModel(**funcArgs[-1])
            cells.setTstop(260)
            cells.initialize(force_print_progress=True)
            cells.run(noclear=True)
            cells.plot_path_attenuation()
            cells = cells.copyAttr()
            results = [[cells]]
            self.free_figure(results)
        else:
            results = self.free_read_data()

            if rank == 0:
                for cells in results:
                    for cell in cells:
                        if hasattr(cell, "paths_toward"):
                            plot_paths(
                                "v",
                                None,
                                None,
                                fname=f"soma_attenuation_{getattr(cell,'voltageClamp')}",
                                precomputed=cell.paths_away,
                            )
                            if hasattr(cell, "paths_toward"):
                                plot_combined(
                                    "v",
                                    None,
                                    None,
                                    None,
                                    fName=f"combined_v_soma_{getattr(cell,'voltageClamp')}",
                                    precomputed_toward=cell.paths_toward,
                                    precomputed_away=cell.paths_away,
                                )

    @read_data
    def SomaCC(self):
        # acutally current injection
        funcArgs = []
        vClampList = comm.bcast(list(np.arange(-40, 461, 50)), root=0)
        plt.cla()
        plt.clf()
        funcArgs.append(
            {
                "mode": 3,
                "ComplexMorph": True,
                "dt": self.dt,
                "kleak": self.leak,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "seed": self.seed,
                "GluTrans": self.optGluT,
                "somaCheck": True,
            }
        )
        ccList = ["currentClamp"]
        results = parallizeFor(
            vClampList,
            [PAPModel],
            funcArgs,
            ccList,
            [["setTstop", "initialize", "run"]],
            [[{"tstop": 260}, {}, {"noclear": True}]],
        )
        if self.free_read_data():
            results = self.free_read_data()

        self.free_figure(results)
        if rank == 0:
            fig = plt.figure(figsize=(9, 5.6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
            ax1 = fig.add_axes([0.1, 0.75, 0.85, 0.2])
            for v in vClampList:
                x = np.linspace(40, 240, 1000)
                holdingpotentials = self.pseudotrace(x, v)
                ax1.plot(x, holdingpotentials, color="grey", label=f"{v}")
            ax1.set_ylabel(gl.curr, color="grey")
            for spine in ax1.spines.values():
                spine.set_visible(False)
            ax1.tick_params(bottom=False, left=True, colors="grey")
            ax1.set_xticks([])

            ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.6])

            for cells in results:
                for cell in cells:
                    ax2.plot(list(cell.time), list(cell.vSoma), color="black")

            ax2.set_xlabel(gl.ms)
            ax2.set_ylabel(gl.volt)
            ax1.set_xlim((40, 240))
            ax2.set_xlim((40, 240))
            plt.savefig(os.path.join("../results/paperRes", f"CurrentClampSoma.pdf"))

    def pseudotrace(self, x, v):
        tmp = []
        for t in x:
            if t < 80 or t > 220:
                tmp.append(0)
            else:
                tmp.append(v)
        return tmp

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
                "kleak": self.leak,
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

        if not self.free_read_data():
            cells = PAPModel(**funcArgs[-1])
            cells.initialize()
            if replay:
                cells.replayK("./Data/invivo_K.csv", isolate=True, setStop=60e3)
                # cells.replayK("./Data/invivo_test.csv", isolate=True)
            else:
                cells.multiSpike(number=self.stimCount, freq=self.freq, KoSize=self.ko)
            cells.run()
            cells = cells.copyAttr()
            AllCells = [[cells]]
            self.free_figure(AllCells)

        else:
            AllCells = self.free_read_data()
            cells = AllCells[0][0]

        initStep = self.get_initStep(cells, shift=0)
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
        if replay:
            plt.colorbar(label=gl.vm, ticks=np.arange(0, 10, 2), extend="max")
            plt.clim((0, 10))
        else:
            plt.colorbar(label=gl.vm, ticks=np.arange(0, 20, 2), extend="max")
            plt.clim(gl.clim_volt)
        plt.xlabel(gl.free("Normalized distance"))
        plt.xticks(
            range(0, 11, 2), [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )  # float point generated by np.linspace

        plt.text(
            0.5,
            len(list(cells.branchAtten[0])[initStep:]) + 2,
            "Soma",
            color="white",
            ha="left",
            va="bottom",
            fontsize=plt.rcParams["axes.labelsize"],
        )
        plt.text(
            9.5,
            len(list(cells.branchAtten[0])[initStep:]) + 2,
            "PAP",
            color="white",
            ha="right",
            va="bottom",
            fontsize=plt.rcParams["axes.labelsize"],
        )

        max_time = len(list(cells.branchAtten[-1])[initStep:]) * cells.dt
        if max_time > 1e3:
            steps_per_time = int(10e3 / cells.dt)  # every 2 s
            cells.dt *= 1e-3
            plt.ylabel(gl.s)
        else:
            steps_per_time = int(20 / cells.dt)  # every 20 ms
            plt.ylabel(gl.ms)

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
                "kleak": self.leak,
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
            plt.plot(cell.time, cell.vSoma, label=f"Ratio to PAP:{ratioList[i]}")
            # plt.legend()
            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"RatioComp{self.optKir}_{self.optNMDAR}.pdf"
                )
            )

    def plot_phase_panel(
        self,
        id,
        data,
        label,
        figObj=None,
        finalize=None,
        legend_label=None,
        final_label=None,
        **plt_args,
    ):
        # get data

        if not figObj:
            fig, axs = plt.subplots(2, 2, figsize=(9, 7.7), sharex=True, sharey=True)
        else:
            fig, axs = figObj

        splt_x, splt_y = id
        axs[splt_x, splt_y].plot(*data, label=label, **plt_args)

        if finalize and final_label:
            plt.tight_layout(rect=[0.18, 0.12, 0.95, 0.90])
            plt.subplots_adjust(left=0.2, right=0.8, wspace=0.2)
            left = axs[0, 0].get_position().x0
            right = axs[0, 1].get_position().x1
            bottom = axs[1, 0].get_position().y0
            top = axs[0, 0].get_position().y1
            x_label, y_label = final_label

            handle, label = axs[splt_x, splt_y].get_legend_handles_labels()
            sortedpair = sorted(
                zip(label, handle),
                key=lambda pair: int(
                    "".join(filter(str.isdigit, pair[0]))
                    if "".join(filter(str.isdigit, pair[0]))
                    else 0
                ),
                reverse=True,
            )
            sorted_label, sorted_handle = zip(*sortedpair)
            if legend_label is None:
                legend_label = gl.delta_ion_o("K", short=True)

            leg = fig.legend(
                sorted_handle,
                sorted_label,
                title=legend_label,
                loc="center left",
                bbox_to_anchor=(0.81, 0.5),
                fancybox=True,
                shadow=True,
                ncol=1,
            )
            leg._legend_box.align = "left"

            fig.text(
                (left + right) / 2,
                bottom - 0.07,
                x_label,
                ha="center",
                va="top",
                fontsize=plt.rcParams["axes.labelsize"],
            )

            fig.text(
                left - 0.09,
                (bottom + top) / 2,
                y_label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=plt.rcParams["axes.labelsize"],
            )

            col1, col2, row1, row2 = finalize

            fig.text(
                axs[0, 0].get_position().x0,
                top + 0.03,
                col1,
                ha="left",
                va="bottom",
                fontsize=plt.rcParams["axes.labelsize"],
            )
            fig.text(
                axs[0, 1].get_position().x0,
                top + 0.03,
                col2,
                ha="left",
                va="bottom",
                fontsize=plt.rcParams["axes.labelsize"],
            )

            fig.text(
                left - 0.16,
                (axs[0, 0].get_position().y0 + axs[0, 0].get_position().y1) / 2,
                row1,
                ha="left",
                va="center",
                rotation=90,
                fontsize=plt.rcParams["axes.labelsize"],
            )
            fig.text(
                left - 0.16,
                (axs[1, 0].get_position().y0 + axs[1, 0].get_position().y1) / 2,
                row2,
                ha="left",
                va="center",
                rotation=90,
                fontsize=plt.rcParams["axes.labelsize"],
            )

        return fig, axs

    def kvPhasePlane(self):
        self.duramplenPhase()
        if self.GluT or self.GABAR or self.NMDAR:
            self.KirNMDAPhase()

    def duramplenPhase(self):
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"
        self.addChannelTag()

        AllCells = []
        KoSteps = np.arange(2, gl.max_ko + 1, 2)
        KoSteps = np.concatenate(([0.5], KoSteps))
        if not self.free_read_data():
            for kircount in [self.KirMax, self.optKir]:
                for PAPLen in [0.3, 5]:

                    funcArgs = []
                    funcArgs.append(
                        {
                            "mode": 0,
                            "ComplexMorph": True,
                            "bNum": 1,
                            "dt": self.dt,
                            "kleak": self.leak,
                            "clleak": 0,
                            "seed": self.seed,
                            "PAPLen": PAPLen,
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
                            funcArgs[-1]["multiple"] = None
                            funcArgs[-1]["GluTrans"] = self.optGluT
                            chanName = "GABAR"
                        else:
                            funcArgs[-1]["GABACount"] = 0
                            funcArgs[-1]["multiple"] = None
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
                            funcArgs[-1]["multiple"] = None
                            funcArgs[-1]["GluTrans"] = None
                            chanName = "GABAR"
                        else:
                            funcArgs[-1]["GABACount"] = 0
                            funcArgs[-1]["multiple"] = None
                            funcArgs[-1]["GluTrans"] = None
                            chanName = ""

                    if (
                        funcArgs[-1]["multiple"] is not None
                        or funcArgs[-1]["GluTrans"] is not None
                    ):
                        funcArgs[-1]["Glu"] = True
                    if funcArgs[-1]["GABACount"] > 0:
                        funcArgs[-1]["GABA"] = True

                    iterations = comm.bcast(
                        [(kircount, amp) for amp in KoSteps],
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
                        results = parallizeFor(
                            iterations,
                            [PAPModel, PAPModel],
                            funcArgs,
                            ccList,
                            [["initialize", "run"]],
                            [[{}, {}]],
                        )
                    if rank == 0:
                        AllCells.append([r[0] for r in results])
                    comm.Barrier()

            self.free_figure(AllCells)

        else:
            AllCells = self.free_read_data()

        if rank == 0:
            plt.cla()
            plt.clf()
            for i, cells in enumerate(AllCells):
                for j in range(len(AllCells[i])):
                    cell = cells[j]
                    id = 0
                    if cell.GENEDict["kir2"] == self.optKir:
                        id += 2
                    if cell.PAPLen > 1:
                        id += 1

                    if cell.KoSize == 0.5:
                        color = "r"
                        z = len(AllCells[i]) + 1
                    else:
                        ko_index = KoSteps == cell.KoSize
                        ko_index = np.argmax(ko_index)
                        color = cm.summer(ko_index / len(AllCells[i]))
                        z = i
                    kw_args = {}
                    if id == 2 and j == len(AllCells[i]) - 1:
                        kw_args["finalize"] = [
                            f"PAP Length 0.3 {gl.unit_micron_raw}",
                            f"5 {gl.unit_micron_raw}",
                            f"Kir Channels\n{int(cell.PAPKirCount_std*self.KirMax+cell.PAPKirCount)}",
                            f"Kir Channels\n{int(cell.PAPKirCount_std*self.optKir+cell.PAPKirCount)}",
                        ]
                        kw_args["final_label"] = [gl.ion_o("K"), gl.volt]
                    initStep = self.get_initStep(cell, shift=0) - 200

                    figobj = self.plot_phase_panel(
                        (0 if id < 2 else 1, id % 2),
                        (list(cell.KoPAP)[initStep:], list(cell.vPAP)[initStep:]),
                        f"{cell.KoSize:.1f}",
                        figObj=None if id == 0 and j == 0 else figobj,
                        color=color,
                        zorder=z,
                    )
                    if j == len(AllCells[i]) - 1:
                        ko_min, ko_max = gl.lim_ko
                        x = np.linspace(ko_min, ko_max)
                        figobj = self.plot_phase_panel(
                            (0 if id < 2 else 1, id % 2),
                            (
                                x,
                                self.nernst(x, cell.kin),
                            ),
                            f"{gl.ek_raw}",
                            figObj=figobj,
                            color="black",
                            linestyle="--",
                            **kw_args,
                        )

            _, axs = figobj
            for ax in axs.flat:
                ax.set_xlim(gl.lim_ko)
                ax.set_ylim(gl.lim_ek)
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"phasePlanePotassium_panel{self.tag}.pdf",
                )
            )
            plt.cla()
            plt.clf()
        plt.close("all")

    @read_data
    def KirNMDAPhase(self):
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"
        self.addChannelTag()

        AllCells = []
        NTChannelComp = [700]
        if self.NMDAR:
            NTChannelComp = [self.optNMDAR] + NTChannelComp
            NT_name = "NMDAR"
        elif self.GABAR:
            NTChannelComp += [self.optGABAR]
            NT_name = "GABAR"
        elif self.GluT:
            NTChannelComp = [0, 100]
            NT_name = "GluT"
        self.tag += NT_name

        if not self.free_read_data():
            for kircount in [self.KirMax, self.optKir]:
                for chanCount in NTChannelComp:
                    funcArgs = []
                    funcArgs.append(
                        {
                            "mode": 0,
                            "ComplexMorph": True,
                            "bNum": 1,
                            "dt": self.dt,
                            "kleak": self.leak,
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
                            funcArgs[-1]["multiple"] = None
                            funcArgs[-1]["GluTrans"] = 0
                            chanName = "GABAR"
                        else:
                            funcArgs[-1]["GABACount"] = 0
                            funcArgs[-1]["multiple"] = None
                            funcArgs[-1]["GluTrans"] = chanCount
                            chanName = "GluTrans"
                    else:
                        if self.NMDAR:
                            funcArgs[-1]["GABACount"] = 0
                            funcArgs[-1]["multiple"] = chanCount
                            funcArgs[-1]["GluTrans"] = None
                            chanName = "NMDAR"
                        elif self.GABAR:
                            funcArgs[-1]["GABACount"] = chanCount
                            funcArgs[-1]["multiple"] = None
                            funcArgs[-1]["GluTrans"] = None
                            chanName = "GABAR"
                        else:
                            funcArgs[-1]["GABACount"] = 0
                            funcArgs[-1]["multiple"] = None
                            funcArgs[-1]["GluTrans"] = None
                            chanName = ""

                    if (
                        funcArgs[-1]["multiple"] is not None
                        or funcArgs[-1]["GluTrans"] is not None
                    ):
                        funcArgs[-1]["Glu"] = True
                    if funcArgs[-1]["GABACount"] > 0:
                        funcArgs[-1]["GABA"] = True

                    KoSteps = np.arange(2, gl.max_ko + 1, 2)
                    KoSteps = np.concatenate(([0.5], KoSteps))

                    iterations = comm.bcast(
                        [(kircount, conc) for conc in KoSteps],
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
                            randomize=False,
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
                    if rank == 0:
                        AllCells.append([r[0] for r in results])
                    comm.Barrier()
            self.free_figure(AllCells)
            # structure AllCells [ chanNum(Kir,NT) [Kosize]]
        else:
            AllCells = self.free_read_data()

        if rank == 0:
            plt.cla()
            plt.clf()
            figobj = None
            for i, cells in enumerate(AllCells):
                for j in range(len(AllCells[i])):
                    cell = cells[j]
                    id = 0
                    if cell.GENEDict["kir2"] == self.optKir:
                        id += 2
                    if (
                        cell.multiple == 0
                        and hasattr(cell, "GABACount")
                        and cell.GABACount == self.optGABAR
                    ):
                        id += 1
                    elif (
                        cell.multiple == 0
                        and "GluTrans" in cell.GENEDict.keys()
                        and cell.GENEDict["GluTrans"] is not None
                        and cell.GENEDict["GluTrans"] > 0
                    ):
                        NTChannelComp = (
                            np.array(NTChannelComp) * cell.PAPGluTCount_std
                            + cell.PAPGluTCount
                        )
                        NTChannelComp = NTChannelComp.astype(int)
                        id += 1
                    elif cell.multiple != self.optNMDAR and not hasattr(
                        cell, "GABACount"
                    ):
                        id += 1

                    initStep = self.get_initStep(cell, shift=0) - 200
                    if cell.KoSize == 0.5:
                        color = "r"
                        z = len(AllCells[i]) + 1
                    else:
                        color = cm.summer(j / len(AllCells[i]))
                        z = i

                    kw_args = {}
                    if id == 2 and j == len(AllCells[i]) - 1:
                        kw_args["finalize"] = [
                            f"{NT_name} Channels\n{NTChannelComp[0]}",
                            NTChannelComp[1],
                            f"Kir Channels\n{int(cell.PAPKirCount_std*self.KirMax+cell.PAPKirCount)}",
                            f"Kir Channels\n{int(cell.PAPKirCount_std*self.optKir+cell.PAPKirCount)}",
                        ]
                        kw_args["final_label"] = [gl.ion_o("K"), gl.volt]
                    initStep = self.get_initStep(cell, shift=0) - 200
                    figobj = self.plot_phase_panel(
                        (0 if id < 2 else 1, id % 2),
                        (list(cell.KoPAP)[initStep:], list(cell.vPAP)[initStep:]),
                        f"{cell.KoSize:.1f}",
                        figObj=figobj,
                        color=color,
                        zorder=z,
                    )
                    if j == len(AllCells[i]) - 1:
                        ko_min, ko_max = gl.lim_ko
                        x = np.linspace(ko_min, ko_max)
                        figobj = self.plot_phase_panel(
                            (0 if id < 2 else 1, id % 2),
                            (
                                x,
                                self.nernst(x, cell.kin),
                            ),
                            f"{gl.ek_raw}",
                            figObj=figobj,
                            linestyle="--",
                            color="black",
                            **kw_args,
                        )

            _, axs = figobj
            for ax in axs.flat:
                ax.set_xlim(gl.lim_ko)
                ax.set_ylim(gl.lim_Vmemb)
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"phasePlaneNT_{self.tag}.pdf",
                )
            )
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
                    "kleak": self.leak,
                    "dt": self.dt,
                    "seed": self.seed,
                    "stimdelay": 20 * ms,
                }
            )
            if self.NMDAR:
                funcArgs[-1]["multiple"] = self.optNMDAR
            else:
                funcArgs[-1]["multiple"] = None
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
                initStep = self.get_initStep(cell)
                depList.append(
                    max(list(cell.vPAP)[initStep:]) - list(cell.vPAP)[initStep]
                )  # 3 ms to stablize
                plt.plot(
                    list(cell.time)[initStep:],
                    list(cell.vPAP)[initStep:] - cell.ek,
                    label=f"{cell.ek}",
                    color=cm.magma(i / len(AllCells)),
                )

        plt.legend()
        plt.xlabel(gl.ms)
        plt.ylabel(gl.free(f"Voltage - {gl.ek_raw} (mV)"))
        plt.savefig(os.path.join("../results/paperRes", "ekDepolarcompTraces.pdf"))
        for cells in AllCells:
            for cell in cells:
                plt.cla()
                plt.clf()
                plt.plot(
                    list(cell.time)[initStep:],
                    list(cell.iKPAP)[initStep:],
                    label=gl.current_ion("K"),
                    color=self.returnColor("iK"),
                )
                plt.plot(
                    list(cell.time)[initStep:],
                    list(cell.iNMDA)[initStep:],
                    label=gl.current_ion("NMDA"),
                    color=self.returnColor("NMDAR"),
                )
                if hasattr(cell, "iGluT"):
                    plt.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluT)[initStep:],
                        label=gl.current_ion("GluT"),
                        color=self.returnColor("GluT"),
                    )
                plt.legend()
                plt.xlabel(gl.ms)
                plt.ylabel(gl.curr)
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"ekDepolarcompCurrentTraces{cell.ek}.pdf",
                    )
                )

        plt.cla()
        plt.clf()
        plt.scatter(ekList, depList, color="black")
        plt.ylabel(gl.d_volt)
        plt.xlabel(gl.ek)
        plt.savefig(os.path.join("../results/paperRes", "ekDepolarcomp.pdf"))

        plt.cla()
        plt.clf()
        plt.scatter(koList, depList, color="black")
        plt.ylabel(gl.d_volt)
        plt.xlabel(gl.ion_o("K"))
        plt.savefig(os.path.join("../results/paperRes", "ekKODepolarcomp.pdf"))

    def KOComp(self, papCount=15, koCond=6):
        for transmitter in ["NMDAR", "GABAR"]:
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
        for i in range(koCond):
            funcArgs = []
            tmpTag = self.tag
            self.addChannelTag()
            # Order is important
            if i == 0:
                self.GluT = True
                setattr(self, transmitter, True)
                self.optKir = 0
                controlLeak = self.leak
                tmpdt = self.dt
            elif i < 3:
                self.GluT = True
                setattr(self, transmitter, True)
                if i == 1:
                    self.tag += "_KirOE"
                    # Kir OE
                    self.optKir = self.KirMax  # from experiment
                    # self.dt *= 0.1
                else:
                    self.tag += "_KirKO"
                    # match findings of Djukic et al. (2007) of -76.3 mV
                    self.dt = tmpdt
                    self.optKir = -2 * self.KirMax  # from experiment
            else:
                self.dt = tmpdt
                self.optKir = 0
                if i == 3:
                    # NMDARKO
                    self.tag += f"_{transmitter}KO"
                    self.GluT = True
                    setattr(self, transmitter, False)
                elif i == 4:
                    # NMDARKO
                    self.tag += "_GluTKO"
                    setattr(self, transmitter, True)
                    self.GluT = False
                elif i == 5:
                    # NMDARKO
                    self.tag += f"_GluTKO_{transmitter}KO"
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
                    "kleak": self.leak,
                    "dt": self.dt,
                }
            )
            #            if i == 1:
            #                krule = {"kuptake": True}
            #            elif i == 2:
            #                # nonspecific K+ block
            #                # funcArgs[-1]['kleak'] = 0
            #                krule = {"kblock": True}
            #            else:
            krule = {}

            if self.NMDAR:
                funcArgs[-1]["multiple"] = self.optNMDAR
            else:
                funcArgs[-1]["multiple"] = None
            if self.GluT:
                funcArgs[-1]["GluTrans"] = self.optGluT

            if self.GABAR:
                funcArgs[-1]["GABACount"] = self.optGABAR
            else:
                funcArgs[-1]["GABACount"] = 0

            comm.Barrier()
            if self.peakLen == None:
                self.peakLen = 5
            else:
                mprint(self.peakLen)
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

            if not self.free_read_data():
                results = parallizeFor(
                    iterations, [PAPModel], funcArgs, ccList, callMethods, callArgs
                )
                self.free_figure(results)
            else:
                results = self.free_read_data()
            self.tag = tmpTag

            comm.Barrier()

            if rank == 0:
                cells = results
                AllCells += cells

        if rank == 0:
            resMat = np.zeros((koCond * 2, papCount))
            for cells in AllCells:
                for cell in cells:
                    if cell.multiple > 0 or (
                        hasattr(cell, "GABACount") and cell.GABACount > 0
                    ):
                        if cell.GENEDict["kir2"] == 0:
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
                        elif cell.GENEDict["kir2"] > 0:
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
                    self.plotIKSeries([[cell]])
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
            pattern = {"confined": "", "spillover": ""}
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
            plt.ylabel(gl.d_volt)
            plt.ylim(gl.lim_d_volt)
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
                c=cm.winter(controlIndex / len(iterations)),
            )
            ax.axhline(
                val_means["spillover"][0],
                linestyle="--",
                c=cm.winter(maxIndex / len(iterations)),
            )

            with open(
                os.path.join("../results/paperRes", f"ttest_res{transmitter}.json"), "w"
            ) as ofile:
                json.dump(val_test, ofile)
            # for k,v in val_test.items():
            #     if v.pvalue < 0.05:
            #         index = category.index(k)
            ax.set_ylabel(gl.d_volt)
            ax.set_ylim(gl.lim_d_volt)
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
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        if self.NMDAR:
            funcArgs[-1]["multiple"] = self.optNMDAR
        else:
            funcArgs[-1]["multiple"] = None
        if self.GluT:
            funcArgs[-1]["GluTrans"] = self.optGluT

        if not self.free_read_data():
            cells = PAPModel(**funcArgs[-1])
            cells.setTstop(160)
            cells.initialize()
            cells.setK(KoSize=100, delay=0, dur=5)
            cells.run()
            cells = cells.copyAttr()
            AllCells = [[cells]]
            self.free_figure(AllCells)
        else:
            AllCells = self.free_read_data()
            cells = AllCells[0][0]
        if rank == 0:
            initStep = self.get_initStep(cells)
            flux = np.array(list(cells.flux)[initStep:])
            kbath = np.array(list(cells.kbath)[initStep:]) * -1
            kbath[kbath == 0] = np.nan
            _, ax = plt.subplots(figsize=(9, 5))
            ax.plot(list(cells.time)[initStep:], np.divide(flux, kbath))
            ax.set_xlabel(gl.ms)
            ax.set_ylabel(gl.free("Ratio of\ninflux / diffusion\nfor potassium"))
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
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "KoSize": 3,
                "PAPLen": 0.3,
            }
        )
        if self.OE:
            funcArgs[-1]["kir2"] = self.KirMax
            self.tag += "_OE"

        if funcArgs[-1]["kir2"] > 3:  # to compensate for mathematical unstability
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

        if not self.free_read_data():
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

        else:
            AllCells = self.free_read_data()
        if rank == 0:
            setKoylim = True
            # if self.ko > 10:
            #     setKoylim = True
            # else:
            #     setKoylim = False
            self.plotIKSeries(AllCells, setKoylim=setKoylim, setekylim=not nearSoma)
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
                    plt.xlabel(gl.ms)
                    plt.ylabel(gl.ion_o("Glu"))
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
                plt.xlabel(gl.ms)
                plt.ylabel(gl.free("Ratio of states"))
                plt.savefig(
                    os.path.join("../results/paperRes", f"GluTstates{self.tag}.pdf")
                )

    def bathExperiment(self, runAll=True, invivo=False, isolate=False, gaba=False):
        self.global_rw_data = True
        # call to set global rw data i.e. each indicidual rank will read and write src_data for plots instead of waiting for all
        if runAll:
            if invivo:
                invivoRunConds = [False, True]
            else:
                invivoRunConds = [False]

            allConds = len(invivoRunConds) * 2

            if not size > allConds:
                wMessage(
                    f"bath experiment runAll only when there are more than {allConds} ranks"
                )
            for i, bool_invivo in enumerate(invivoRunConds):
                for j, bool_isolate in enumerate([False, True]):
                    if rank == 2 * i + j:
                        self.bathExperiment(
                            runAll=False,  # for escaping inf loop
                            invivo=bool_invivo,
                            isolate=bool_isolate,
                        )
            if rank == allConds:
                self.bathExperiment(runAll=False, gaba=True)  # for escaping inf loop

        else:
            # print(f"{gaba=}{invivo=}{isolate=}{rank=}")
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
            self.tag += "_global"
        # print(self.tag)
        AllCells = []
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "kir2": self.optKir,
                "clleak": 0,
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        if self.OE:
            funcArgs[-1]["kir2"] = self.KirMax
            self.tag += "_OE"

        funcArgs[-1]["multiple"] = None
        funcArgs[-1]["Glu"] = False
        funcArgs[-1]["GABACount"] = 0
        funcArgs[-1]["GABA"] = False
        funcArgs[-1]["GluTrans"] = self.optGluT
        funcArgs[-1]["nakpump"] = self.optNKA

        if not self.free_read_data():
            cells = PAPModel(**funcArgs[-1])
            if invivo:
                cells.initialize()
                cells.replayK("./Data/invivo_K.csv", isolate=isolate)
                cells.ko_sim(False)
                cells.run()

            else:
                cells.setTstop(300)
                # if not isolate:
                #    cells.set_ECS(100e4, scale=False)

                cells.initialize()
                tsnap = False
                if not isolate:
                    tsnap = True
                cells.setKBath(
                    10, dur=200, tsnap=tsnap, isolate=isolate, clamp_ki=not isolate
                )
                cells.run()

            cells = cells.copyAttr()

            # if size > 1:
            #     AllCells = comm.gather(cells, root=0)
            # else:
            AllCells.append([cells])
        else:
            AllCells = self.free_read_data()
        setKoylim = True
        # if self.ko > 10:
        #     setKoylim = True
        # else:
        #     setKoylim = False
        self.plotIKSeries(
            AllCells,
            setKoylim=setKoylim,
            setekylim=True,
            define_initStep=50,
            bath=True,
        )
        # results = AllCells[0][0]
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
                "multiple": None,
                "Glu": False,
                "GABA": True,
                "bNum": 1,
                "kir2": 0,
                "clleak": 0,
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        if self.OE:
            funcArgs[-1]["kir2"] = 5
            self.tag += "_OE"

        if funcArgs[-1]["kir2"] > 5:  # to compensate for mathematical unstability
            funcArgs[-1]["dt"] *= 0.2
        funcArgs[-1][
            "GABACount"
        ] = 1  # not optGABA as GABABath distributes GABAs with different mechanism compared to default distribution

        if not self.free_read_data():
            cells = PAPModel(**funcArgs[-1])
            cells.setTstop(200)
            cells.GABABath(1, 0)
            cells.run()
            cells = cells.copyAttr()

            # if size > 1:
            #     AllCells = comm.gather(cells, root=0)
            # else:
            AllCells.append([cells])
        else:
            AllCells = self.free_read_data()
        setKoylim = True
        # if self.ko > 10:
        #     setKoylim = True
        # else:
        #     setKoylim = False
        self.plotIKSeries(AllCells, setKoylim=setKoylim, setekylim=True, bath=True)
        # results = AllCells[0][0]
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
                "kleak": self.leak,
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

    @read_data
    def channelComparison(self):
        self.addChannelTag()
        if self.GABAR:
            self.channelCompareMax *= 2
            self.channelCompareStep *= 2
        if self.GAP and not self.GABAR:
            self.channelCompareMax /= 2
            self.channelCompareStep /= 2
        if not (self.GABAR or self.NMDAR) and self.GluT:
            self.channelCompareMax = 5
            self.channelCompareStep = int(self.channelCompareMax / 5)
            iterations = [
                (i, j)
                for i in np.arange(
                    -self.KirMax, self.KirMax + self.KirStep / 2, self.KirStep
                )
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
                    starta=-self.KirMax,
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
                "kleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
            }
        )
        ccList = ["kir2"]
        if self.GABAR:
            funcArgs[-1]["dt"] *= 0.2
            funcArgs[-1]["multiple"] = None
            funcArgs[-1]["GABA"] = True
            ccList.append("GABACount")
        elif self.NMDAR:
            ccList.append("multiple")
            if self.GluT:
                funcArgs[-1]["GluTrans"] = self.optGluT
            funcArgs[-1]["Glu"] = True
        else:
            funcArgs[-1]["multiple"] = None
            if self.GluT:
                ccList.append("GluTrans")
                funcArgs[-1]["Glu"] = True
            elif self.GAP:
                ccList.append("gapCount")
                funcArgs[-1]["GABA"] = False
                funcArgs[-1]["Glu"] = False
            elif self.NKA:
                ccList.append("nakpump")
                funcArgs[-1]["GABA"] = False
                funcArgs[-1]["Glu"] = False

            else:
                iterations = [
                    i
                    for i in np.arange(
                        -self.KirMax, self.KirMax + self.KirStep / 2, self.KirStep
                    )
                ]
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
            if self.NMDAR and self.GluT:
                self.GluT = False  # force plot priority of NMDA
            self.plotHeatmap(results, Kir=True, tag=self.tag, stdLabels=True)
            # self.plotIKSeries(results, setekylim=True, setKoylim=True, setyLim=[-15, 1])

    @read_data
    def shift_PAP_location(self):
        self.addChannelTag()
        gapCounts = [0, 50, 100]
        shift_range = np.arange(0, 1, 0.1)
        iterations = [(i, j) for i in shift_range for j in gapCounts]
        iterations = comm.bcast(iterations, root=0)
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "Glu": self.GluStim,
                "GABA": self.GabaStim,
                "ComplexMorph": True,
                "kleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
            }
        )
        ccList = ["shift_PAP", "gapCount"]
        callMethods = [[]]
        callArgs = [[]]
        callMethods[0] += ["initialize", "multiSpike", "run"]
        callArgs[0] += [
            {},
            {"number": self.stimCount, "KoSize": self.ko, "freq": self.freq},
            {},
        ]
        results = parallizeFor(
            iterations, [PAPModel], funcArgs, ccList, callMethods, callArgs
        )

        self.free_figure(results)

        comm.Barrier()

        if rank == 0:
            imArray = np.zeros((len(shift_range), len(gapCounts)))
            gapCounts = np.array(gapCounts)
            for cells in results:
                for cell in cells:
                    imArray[
                        np.where(shift_range == cell.shift_PAP)[0][0],
                        np.where(gapCounts == cell.gapcount)[0][0],
                    ] += (
                        max(np.array(cell.vPAP)) - cell.RMP
                    )

            cmap = "magma"
            plt.cla()
            plt.clf()
            plt.figure(figsize=(5.1, 9))
            plt.imshow(
                imArray,
                cmap=cmap,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            maxv = 30
            plt.colorbar(label=gl.vm, ticks=np.arange(0, maxv, 2), extend="max")
            plt.clim((5, maxv))
            plt.xlabel(gl.chan_num("Cx43"))
            plt.ylabel(gl.free(f"PAP distance from GJ {gl.unit_micron}"))
            plt.xticks(
                range(len(gapCounts)),
                gapCounts,
            )
            plt.yticks(range(len(shift_range)), [f"{x:.2f}" for x in shift_range])

            plt.savefig(os.path.join("../results/paperRes", f"PAP_shift{self.tag}.pdf"))

    #            self.plotHeatmap(totResults, divedend=len(resFiles))
    #
    @read_data
    def glutamateSpillOver(self, sampleNum=10):
        self.addChannelTag()
        if self.GluStim:
            iterations = np.concatenate(
                (np.logspace(0, 1, num=9), np.array([self.PAPLen]))
            )

        else:
            iterations = np.concatenate(
                (np.logspace(0, 1, num=9), np.array([self.PAPLen]))
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
                "kleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "kir2": self.optKir,
                "KoSize": self.ko,
                "multiple": None,
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
            self.free_figure(results)
            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_axes([0.1, 0.52, 0.8, 0.40])
            ax_inset = fig.add_axes([0.1, 0.15, 0.8, 0.25])
            for cells in results:
                for cell in cells:
                    i = np.where(cell.PAPLen == iterations)[0][
                        0
                    ]  # get index of PAPLen position in iterations
                    cindex = i / len(iterations)
                    color = cm.winter(cindex)
                    initStep = self.get_initStep(cell)
                    ax.plot(
                        np.array(list(cell.time)[initStep:]) * 1e-3,  # ms to s
                        np.array(list(cell.vPAP)[initStep:]) - cell.RMP,
                        color=color,
                    )
                    ax_inset.plot(
                        np.array(list(cell.time)[initStep:]),  # ms
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
            ax.set_xlabel(gl.s)
            ax.set_ylabel(gl.d_volt)
            x0 = 145 * 1e-3
            x1 = 165 * 1e-3
            y0 = -0.01
            y1 = 3.01
            ax.set_ylim((y0, y1))

            ax_inset.set_xlim((x0 * 1e3, x1 * 1e3))
            ax_inset.set_ylim((y0, y1))
            ax_inset.set_xlabel(gl.ms)
            ax_inset.set_ylabel(gl.d_volt)

            grey = "0.5"
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=1.5,
                edgecolor=grey,
                zorder=2,
            )
            ax.add_patch(rect)
            con1 = ConnectionPatch(
                xyA=(x0, y0),
                coordsA=ax.transData,
                xyB=(x0 * 1e3, y1),
                coordsB=ax_inset.transData,
                color=grey,
                linewidth=1,
                zorder=3,
                linestyle="--",
            )
            con2 = ConnectionPatch(
                xyA=(x1, y0),
                coordsA=ax.transData,
                xyB=(x1 * 1e3, y1),
                coordsB=ax_inset.transData,
                color=grey,
                linewidth=1,
                zorder=3,
                linestyle="--",
                connectionstyle="arc3,rad=0.1",
            )

            fig.add_artist(con1)
            fig.add_artist(con2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig(
                os.path.join("../results/paperRes", f"GlutamateSpillOver{self.tag}.pdf")
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
                cmap="winter",
                s=10,
                edgecolor="black",
                linewidth=1,
                c=[i / len(iterations) for i in range(len(iterations))],
            )
            # plot control as diamond
            if controlIndex != None:
                ax.scatter(
                    self.PAPLen,
                    controlV,
                    color=cm.winter(controlIndex / len(iterations)),
                    marker="D",
                    edgecolor="black",
                    label="Control",
                    zorder=10,
                )
                # ax.axvline(
                #     self.PAPLen,
                #     ymax=controlV/top,
                #     linestyle='--',
                #     color=cm.winter(controlIndex/len(iterations)),
                #     zorder=-1,
                # )
            maxIndex = vList.index(max(vList))

            # self.peakLen = iterations[maxIndex]
            ax.scatter(
                iterations[maxIndex],
                vList[maxIndex],
                color=cm.winter(maxIndex / len(iterations)),
                label="Spillover",
                zorder=-1,
            )
            if self.GluStim and self.KStim:
                if self.peakLen is not None:
                    ax.axvline(
                        self.PAPLen,
                        ymax=vList[0] / maxY,
                        linestyle="--",
                        color=cm.winter(maxIndex / len(iterations)),
                        zorder=-2,
                    )
            ax.legend()
            ax.set_ylim((y0, y1))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.set_xlabel(gl.pap_affect)
            ax.set_ylabel(gl.free("Peak Voltage Change (mV)"))
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
                self.plotIKSeries(
                    cell,
                    setekylim=True,
                    setKoylim=True,
                )

    def potassiumComparison(self):
        self.KoCompMax = gl.max_ko
        self.KoCompStep = 2
        for comparison in ["seed", "PAPLen", "KoSize", "durStim"]:
            if comparison == "KoSize":
                compMax = self.KoCompMax
                compStep = self.KoCompStep
                startb = 0
            elif comparison == "PAPLen":
                compMax = 3
                compStep = 0.3
                startb = 0.3
            elif comparison == "durStim":
                compMax = 9
                compStep = 1
                startb = 1
            elif comparison == "seed":
                compMax = 15
                compStep = 1
                startb = 1

            if comparison != "KoSize":
                if comparison == "PAPLen":
                    logx = np.logspace(1, -1.5, base=0.3, num=10)
                    iterations = comm.bcast(
                        [
                            (i, j)
                            for i in range(0, self.KoCompMax + 1, self.KoCompStep)
                            for j in logx
                        ],
                        root=0,
                    )
                else:
                    iterations = comm.bcast(
                        get_iter(
                            self.KoCompMax,
                            self.KoCompStep,
                            compMax,
                            compStep,
                            startb=startb,
                        ),
                        root=0,
                    )
                    logx = None

                self.addChannelTag()
                self.tag += f"_{comparison}"
                self.runAmpLenComparison(
                    comparison, iterations, compMax, compStep, logx=logx
                )

            # Calculate the number of iterations for all parm sets
            iterations = comm.bcast(
                get_iter(
                    self.KirMax,
                    self.KirStep,
                    compMax,
                    compStep,
                    starta=-self.KirMax,
                    startb=startb,
                ),
                root=0,
            )
            # # Adjust the range for the last process
            if comparison != "seed":
                self.addChannelTag()
                self.tag += f"_{comparison}"
                self.runPotassiumComparison(
                    comparison, iterations, maxStep=compMax, intermStep=compStep
                )

    @read_data
    def runAmpLenComparison(
        self, comparison, iterations, maxStep, intermStep, logx=None
    ):
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "Glu": self.GluStim,
                "GABA": False,
                "ComplexMorph": True,
                "kleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "kir2": self.optKir,
            }
        )
        # if self.NMDAR:
        #     funcArgs[-1]["multiple"] = self.optNMDAR
        # else:
        self.NMDAR = False
        funcArgs[-1]["multiple"] = None
        if self.GluStim:
            self.GluT = True
            funcArgs[-1]["GluTrans"] = self.optGluT
        else:
            self.GluT = False
            funcArgs[-1]["GluTrans"] = None

        if comparison != "seed":
            funcArgs[-1]["seed"] = self.seed

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
        self.free_figure(results)
        if rank == 0:
            plt.cla()
            plt.clf()
            imArray = np.zeros(
                (
                    int(self.KoCompMax / self.KoCompStep + 1),
                    int(len(iterations) / (self.KoCompMax / self.KoCompStep + 1)),
                )
            )  # int(maxStep / intermStep) + 1))
            for res in results:
                if logx is not None:
                    index = int(np.where(logx == getattr(res[0], ccList[1]))[0])
                    imArray[
                        int(getattr(res[0], ccList[0]) / self.KoCompStep), index
                    ] += (max(res[-1].vPAP) - res[0].RMP)
                else:
                    imArray[
                        int(getattr(res[0], ccList[0]) / self.KoCompStep),
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
                skip = 1
                dec = 0
                printType = int
                plt.xlabel(gl.durstim)
            elif comparison == "seed":
                skip = 1
                dec = 0
                printType = int
                plt.xlabel(gl.seed_num)
            else:
                skip = 0
                dec = 1
                printType = float
                plt.xlabel(gl.pap_len)
            if logx is not None:
                logx_label = [
                    f"{val:.1f}" if val < 10 else str(np.round(val).astype(int))
                    for val in logx
                ]
                plt.xticks(np.arange(0, len(logx), 1), logx_label)
            else:
                plt.xticks(
                    np.arange(0, int(maxStep / intermStep) + (1 - skip), 1),
                    np.round(
                        np.arange(skip, maxStep + intermStep / 2, intermStep),
                        decimals=dec,
                    ).astype(printType),
                )
            # set ylabel
            skip = 1
            dec = 0

            printType = int
            maxStep = self.KoCompMax
            intermStep = self.KoCompStep
            plt.ylabel(gl.delta_ion_o("K"))
            plt.yticks(
                np.arange(0, int(maxStep / intermStep) + 1, 1),
                np.round(
                    np.arange(0, maxStep + intermStep / 2, intermStep), decimals=dec
                ).astype(printType),
            )
            plt.colorbar(label=gl.vm, ticks=np.arange(0, 20, 2), extend="max")
            plt.clim(gl.clim_volt)
            if comparison == "PAPLen":
                self.GluT = False  # just to force plot setLabel Colors
            self.setLabelColors(
                res[0].PAParea,
                Kir=True,
                y=True,
                x=True if comparison == "PAPLen" else False,
                chanOverride={"Kir": (res[0].PAPKirCount, res[0].PAPKirCount_std)},
            )

            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"FullPotassiumAmp{self.tag}_{comparison}.pdf",
                )
            )
            if comparison == "PAPLen":
                self.plotIKSeries(results, tagReset=True, setKoylim=True)
            elif comparison == "seed":
                self.mergePlotsIK(results, "KoSize", "seed", selected=1)

    @read_data
    def runPotassiumComparison(self, comparison, iterations, maxStep=10, intermStep=1):
        # reset tag to current status
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "Glu": self.GluStim,
                "GABA": False,
                "ComplexMorph": True,
                "kleak": self.leak,
                "clleak": 0,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "GluTrans": self.optGluT,
            }
        )

        # if self.NMDAR:
        #     funcArgs[-1]["multiple"] = self.optNMDAR
        # else:
        funcArgs[-1]["multiple"] = None
        if self.GluT:
            funcArgs[-1]["GluTrans"] = self.optGluT

        if comparison != "seed":
            funcArgs[-1]["seed"] = self.seed
        ccList = ["kir2", comparison]
        #        if comparison == "PAPLen":
        #            funcArgs[-1]["dt"] = self.dt / 2
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
        AllCells = results
        self.free_figure(AllCells)
        results = AllCells
        if rank == 0:
            plt.cla()
            plt.clf()
            if comparison == "durStim":
                adjust = 1
            else:
                adjust = 0
            imArray = np.zeros(
                (
                    2 * int(self.KirMax / self.KirStep) + 1,
                    int(maxStep / intermStep) + (1 - adjust),
                )
            )
            for res in results:
                arrayValue = max(res[0].vPAP) - res[0].RMP
                originalValue = imArray[
                    int((self.KirMax + res[0].GENEDict["kir2"]) / self.KirStep),
                    int(getattr(res[0], comparison) / intermStep) - adjust,
                ]
                if originalValue == 0:
                    imArray[
                        int((self.KirMax + res[0].GENEDict["kir2"]) / self.KirStep),
                        int(getattr(res[0], comparison) / intermStep) - adjust,
                    ] = arrayValue
            #                else:
            #                    mprint(
            #                        originalValue,
            #                        arrayValue,
            #                        res[0].GENEDict["kir2"],
            #                        getattr(res[0], comparison),
            #                        adjust,
            #                        int((self.KirMax + res[0].GENEDict["kir2"]) / self.KirStep),
            #                        int(getattr(res[0], comparison) / intermStep) - adjust,
            #                    )

            plt.imshow(
                imArray,
                cmap="magma",
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            ylabels = (
                np.arange(
                    -1 * int(self.KirMax / self.KirStep),
                    int(self.KirMax / self.KirStep) + 1,
                    1,
                ).astype(float)
                * self.KirStep
                * res[0].PAPKirCount_std
                + res[0].PAPKirCount
            )
            if comparison == "PAPLen":
                ylabels = ylabels / float(res[0].PAPLen)
            ylabels = [int(val) if val > 0 else 0 for val in ylabels]

            plt.yticks(
                range(2 * int(self.KirMax / self.KirStep) + 1),
                ylabels,
            )
            if comparison == "PAPLen":
                plt.ylabel(gl.chan_num("Kir") + "/ PAP length (\u03bcm$^{-1}$)")
            else:
                plt.ylabel(gl.chan_num("Kir"))
            if comparison == "KoSize":
                skip = 1
                dec = 0
                printType = int
                plt.xlabel(gl.delta_ion_o("K"))
            elif comparison == "durStim":
                skip = 2
                dec = 0
                printType = int
                plt.xlabel(gl.durstim)
            elif comparison == "seed":
                skip = 1
                dec = 0
                printType = int
                plt.xlabel(gl.seed_num)
            else:
                skip = 2
                dec = 1
                printType = float
                plt.xlabel(gl.pap_len)
            if comparison == "durStim":
                plt.xticks(
                    np.arange(0, int(maxStep / intermStep), 1),
                    np.round(
                        np.arange(1, maxStep + intermStep / 2, intermStep), decimals=dec
                    ).astype(printType),
                )
            else:
                plt.xticks(
                    np.arange(0, int(maxStep / intermStep) + 1, 1),
                    np.round(
                        np.arange(0, maxStep + intermStep / 2, intermStep), decimals=dec
                    ).astype(printType),
                )
            plt.colorbar(label=gl.vm, ticks=np.arange(0, 20, 2), extend="max")
            plt.clim(gl.clim_volt)
            if comparison == "PAPLen":
                self.GluT = False  # just to force plot setLabel Colors
                chanOverride = {
                    "Kir": (
                        res[0].PAPKirCount / res[0].PAPLen,
                        res[0].PAPKirCount_std / res[0].PAPLen,
                    )
                }

            else:
                chanOverride = {
                    "Kir": (
                        res[0].PAPKirCount,
                        res[0].PAPKirCount_std,
                    )
                }
            self.setLabelColors(
                res[0].PAParea,
                Kir=True,
                y=True,
                x=True if comparison == "PAPLen" else False,
                chanOverride=chanOverride,
            )

            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"FullPotassium{self.tag}_{comparison}.pdf"
                )
            )

            if comparison == "KoSize" or comparison == "PAPLen":
                self.plotIKSeries(results, tagReset=True, setKoylim=True)

    def SCeq(self, x, a, l, c):
        return a * np.exp(-x / l) + c

    @read_data
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
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        ccList = ["g_pas"]
        iterations = [x for x in np.geomspace(2.60, 0.69, num=15)]
        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            [
                ["initialize", "setDualPatch", "run", "getDualPatch_lambda"],
            ],
            [[{}, {}, {}]],
        )
        comm.Barrier()
        self.free_figure(results)

        if rank == 0:
            sensitivity = []
            test_conductance = []
            tolerance = 0.1
            inject_x = 0.01
            for cells in results:
                for cell in cells:
                    sensitivity.append(cell.spaceConstant)
                    test_conductance.append(cell.g_pas)
                    if abs(3.6 - sensitivity[-1]) < tolerance:
                        maxVal = max(cell.soma_atten) - cell.RMP
                        inject_site = cell.soma_L * inject_x
                        plt.scatter(
                            cell.lenUnits * np.arange(len(cell.soma_atten))
                            - inject_site,
                            (np.array(cell.soma_atten) - cell.RMP) / maxVal,
                        )
                        plt.axvline(
                            -0.08,
                            ymin=0,
                            ymax=1 / 1.05,
                            color="lightgrey",
                            linestyle="--",
                        )
                        plt.axhline(
                            1 / np.e,
                            xmin=0,
                            xmax=1,
                            label="1/e",
                            color="grey",
                            linestyle="--",
                        )
                        plt.xlabel(gl.free("Distance ($\mu$m)"))
                        plt.ylabel(gl.free("V(d)/V(0)"))
                        plt.title(
                            gl.free(f"Rm ($\Omega$ cm$^2$) = {int(1/cell.g_pas)}")
                        )
                        plt.savefig("dual_patch.pdf")
            plt.cla()
            plt.clf()
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(test_conductance, sensitivity)
            ax.set_xlabel(gl.free("1/Rm ($\Omega^{-1}$ cm$^{-2}$)"))
            ax.set_ylabel(gl.free("Projected $\lambda$"))
            ax.axhline(y=3.6, xmin=0, xmax=1, label="3.6", color="grey", linestyle="--")

            inset = inset_axes(ax, width="35%", height="35%", loc="upper right")
            inset.plot(test_conductance, sensitivity)
            inset.axhline(
                3.6, xmin=0, xmax=1, label="1/e", color="grey", linestyle="--"
            )
            inset.set_ylim(0, 5)
            inset.set_xlim(2, 2.60)
            mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.5")

            plt.tight_layout()
            plt.savefig("dual_patch_sensitivity.pdf")

    def optDepolarizationSearch(self, x, optmV=20.0):
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
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "multiple": None,
            }
        )
        print(x)
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        cells.setTstop(tstop=151)
        cells.multiSpike(number=1, freq=100, KoSize=x[0])
        cells.run()
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

    def physiological_stim(
        self,
    ):
        self.tag += "_physiological"
        AllCells = []
        funcArgs = []
        models = ["K$^+$ Model", "GluT Model", "GABA$_A$R Model", "NMDAR Model"]
        stim = [50, 100, "theta"]
        papcounts = [1, 10]
        for s in stim:
            for p in papcounts:
                for m in models:
                    funcArgs.append(
                        {
                            "mode": 0,
                            "ComplexMorph": True,
                            "kir2": 0,
                            "clleak": 0,
                            "kleak": self.leak,
                            "dt": self.dt,
                            "seed": self.seed,
                            "KoSize": 0.5,
                        }
                    )
                    funcArgs[-1]["PAPCount"] = p
                    funcArgs[-1]["stimtype"] = s
                    if m == "GluTModel":
                        funcArgs[-1]["Glu"] = True
                        funcArgs[-1]["GluTrans"] = self.optGluT
                    elif m == "GABA$_A$R Model":
                        funcArgs[-1]["GABA"] = True
                        funcArgs[-1]["GABACount"] = self.optGABAR
                    elif m == "NMDAR Model":
                        funcArgs[-1]["multiple"] = self.optNMDAR
                        funcArgs[-1]["Glu"] = True
                        funcArgs[-1]["GluTrans"] = self.optGluT

        # distribute different sims
        iterations_per_process = len(funcArgs) // size
        if len(funcArgs) % size == 0:
            remaining_iterations = 0
            minimum = rank * iterations_per_process
            maximum = (rank + 1) * iterations_per_process
        elif rank >= size - len(funcArgs) % size:
            remaining_iterations = 1
            minimum = rank * (iterations_per_process + remaining_iterations) - (
                size - len(funcArgs) % size
            )
            maximum = (rank + 1) * (iterations_per_process + remaining_iterations) - (
                size - len(funcArgs) % size
            )
        else:
            remaining_iterations = 0
            minimum = rank * iterations_per_process
            maximum = (rank + 1) * iterations_per_process

        if not self.free_read_data():
            for index in range(minimum, maximum):
                # print(f"started sim {funcArgs[index]}")
                stype = funcArgs[index].pop("stimtype")
                cells = PAPModel(**funcArgs[index])
                cells.setTstop(500)

                if stype == "theta":
                    # initialize() multiSpike() run() all packaged in TBS()
                    cells.TBS()
                else:
                    cells.initialize()
                    cells.multiSpike(number=int(3 * stype / 10), freq=stype)
                    cells.run()
                AllCells.append(cells.copyAttr())
            print(f"ran_sim {rank}")
            comm.Barrier()
            AllCells = comm.gather(AllCells, root=0)
            # flatten list
            if rank == 0:
                AllCells = [item for cell_list in AllCells for item in cell_list]
        else:
            AllCells = self.free_read_data()

        self.plot_physiological(AllCells, stim, papcounts, models)

    def fitExpDepolarization(
        self,
        x,
        showFig=False,
        PAP=True,
        use_tau=False,
        autosave=True,
        skipsave=False,
    ):

        # print(f"{rank}:{x}")
        # unify x value
        x = comm.bcast(x, root=0)
        comm.Barrier()

        self.foundFitExperiment = False
        if autosave:
            self.global_rw_data = True
        self.addChannelTag()
        AllCells = []
        funcArgs = []
        leak = self.leak
        # maybe bug for result fit check how parms should change with diffrenet leak value
        forcedAccum = None
        if PAP:
            if self.GABAR:
                self.tag += "_PAP_GABAR"
                k = 500
                mprint(x)
                if use_tau:
                    GABAR, s, d, tau2 = x
                else:
                    GABAR = int(x[0])
                tau1 = 1.69
                Kir = 0
                funcArgs.append(
                    {
                        "mode": 0,
                        "ComplexMorph": True,
                        "Glu": False,
                        "GABA": True,
                        "GABACount": GABAR,
                        "kir2": Kir,
                        "clleak": 0,
                        "kleak": leak,
                        "dt": self.dt,
                        "seed": self.seed,
                        "multiple": None,
                        "GluTrans": None,
                        "KoSize": 0.5,
                    }
                )

            else:
                self.tag += "_PAP_NMDAR"
                k = 500
                mprint(x)
                if use_tau:
                    NMDAR, s, d, tau2 = x
                else:
                    NMDAR = int(x[0])
                tau1 = 1.69
                Kir = 0
                funcArgs.append(
                    {
                        "mode": 0,
                        "ComplexMorph": True,
                        "Glu": True,
                        "kir2": Kir,
                        "clleak": 0,
                        "kleak": leak,
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
                if use_tau:
                    glt, kir, PAPLen, KoSize = x
                else:
                    glt, kir = x
                    glt = int(glt)
                    kir = int(kir)
                tau2 = 5.8
                slowing = 1
                forcedAccum = False
            else:
                self.tag += "_forced_accum"
                if use_tau:
                    glt, kir, PAPLen, KoSize, tau2, slowing = x
                else:
                    glt, kir = x
                forcedAccum = True
                if not use_tau:
                    tau2 = None
                    slowing = None

            if not use_tau:
                PAPLen = 0.3
                KoSize = 0.5

            funcArgs.append(
                {
                    "mode": 0,
                    "Glu": True,
                    "kir2": kir,
                    "clleak": 0,
                    "kleak": leak,
                    "dt": self.dt,
                    "seed": self.seed,
                    "multiple": None,
                    "GluTrans": glt,
                    "KoSize": KoSize,
                    "PAPLen": PAPLen,
                }
            )

        if rank % 3 == 2:
            stim = 1
        elif rank % 3 == 1:
            stim = 5
        else:
            stim = 10

        # match rank to iteration
        rankDict = {}
        stimList = comm.gather(stim, root=0)
        if rank == 0:
            for i, s in enumerate(stimList):
                rankDict[s] = i
        rankDict = comm.bcast(rankDict, root=0)
        comm.Barrier()

        if not autosave or not self.free_read_data():
            cells = PAPModel(**funcArgs[-1])
            cells.setTstop(500)
            if not PAP:
                cells.setGLT_TC(0.61, tau2)
                # cells.NaKpumpOn(False)
            cells.initialize()

            if use_tau:
                if PAP:
                    cells.setNMDA_Mgblock(k, d, s)
                    cells.setNMDA_TC(tau1, tau2)
                else:
                    cells.setSlowing(slowing)
            cells.multiSpike(number=stim, freq=100)
            if not PAP and use_tau:
                cells.setGLT_TC(0.61, 5.8)
                cells.setSlowing(1)  # return to normal after neuro activity

            cells.run()
            cells = cells.copyAttr()
            print(f"ran_sim {rank}")
            comm.Barrier()
            AllCells = comm.gather(cells, root=0)
            if rank == 0 and autosave and not skipsave:
                self.free_figure(AllCells)
        else:
            if autosave:
                self.foundFitExperiment = True
                AllCells = self.free_read_data()
                cells = AllCells[rank]
            # if rank == 0:
            #    self.plotIKSeries([[cells]])

        loss = self.plotExpFit(
            cells,
            stim=stim,
            PAP=PAP,
            showFig=showFig,
            Fname=f"fit{PAP=}{forcedAccum=}{self.GABAR=}",
            correctArtifact=False,
            rankDict=rankDict,
        )
        # print(loss)
        return loss

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
        normalize=False,
        split=False,
        rankDict={10: 0, 5: 1, 1: 2},
    ):
        plt.close("all")
        if split:
            for i in range(2):
                plt.figure(i)
                plt.cla()
                plt.clf()
        else:
            plt.cla()
            plt.clf()
        AllCells = []
        sim_time = None
        if hasattr(cells, "GABACount"):
            print(f"{cells.GABACount=}")
        else:
            print(f"{cells.multiple=}")
        # get and tweak results
        tList, fList, stdList = procedure.getExpRes(f"./Data/{stim}stim.csv")

        stdList = [np.nan if val == 0 or val is None else val for val in stdList]
        zeroPoint = tList.index(min(tList, key=abs))
        # print(zeroPoint,fList)
        fList = np.array(fList) - fList[zeroPoint]

        tList = np.array(tList) + int(cells.initTstop + cells.stimdelay)
        # print(tList)

        fluorTrace = (np.array(list(cells.fluorVPAP)) - cells.RMP) * -1 / 10
        simV = np.array(list(cells.vPAP)) - cells.RMP  # use raw sim data for plot

        # extract corresponding indexes in df and sim
        indexConvert = [
            (i, np.argmin(abs(np.array(cells.time) - t))) for i, t in enumerate(tList)
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
            singleStimBaseline = comm.bcast(expF, root=rankDict[1])
            singleStimBaselineT = comm.bcast(expT, root=rankDict[1])
            spl = spline(singleStimBaselineT, singleStimBaseline)
            simF += spl(expT)
        loss = np.absolute(expF - simF)
        if normalize:
            loss /= max(expF)

        stdComp = loss - expSTD
        loss = sum(loss[(stdComp >= 0)] ** 2)

        if len(simV) < len(tList) or np.isnan(simF).any():
            loss = np.inf

        # MLS
        # trueloss = sum((expF - simV)**2)
        # lossRMP = self.optRMPSearch((leak,Kir))
        comm.Barrier()
        if not np.isnan(loss) and verbose:
            print(f"Loss:{loss}@rank{rank}")
        loss = comm.gather(loss, root=0)
        sim_time = comm.gather(cells.time, root=0)
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
        if rank == 0:
            for l in loss:
                total += l
            if showFig:
                if split:
                    color = {10: "tab:green", 5: "tab:orange", 1: "tab:blue"}
                    for i, t, f, yerr, sim_v, sim_f, sim_t in zip(
                        stim, expT, expF, expSTD, simV, fluorTrace, sim_time
                    ):
                        plt.figure(0)
                        plt.plot(
                            sim_t,
                            sim_v,
                            label=f"{i} stim simulation",
                            linestyle="-",
                            color=color[i],
                            zorder=i,
                        )
                        if correctArtifact:
                            sim_f += spl(sim_t)
                            label = f"\n{i} stim simulation"
                        else:
                            label = f"{i} stim simulation"
                        plt.figure(1)
                        plt.plot(
                            sim_t,
                            sim_f,
                            label=label,
                            linestyle="--",
                            color=color[i],
                            zorder=100 + i,
                        )
                        plt.errorbar(
                            t, f, yerr=yerr, fmt="none", color=color[i], zorder=200 + i
                        )

                        plt.scatter(
                            t,
                            f,
                            label=f"{i} stim experiment",
                            color=color[i],
                            zorder=201 + i,
                        )
                    if correctArtifact:
                        Fname += "_correctedArtifact"
                    for i in range(2):
                        plt.figure(i)
                        plt.xlim((100, 500))
                        plt.legend()
                        plt.xlabel(gl.ms)
                        if i == 0:
                            plt.ylabel(gl.volt)
                            plt.ylim(gl.lim_Vmemb)
                            Fname += "memb_potential"
                        else:
                            _, ymax = gl.clim_volt
                            plt.ylabel(gl.fluor)
                            plt.ylim((0, ymax * -1 / 10))
                            Fname = "fluor"
                        plt.savefig(f"../results/paperRes/{Fname}.pdf")
                    if np.isnan(total):
                        total = np.inf
                    elif verbose:
                        print(f"{total=}")

                else:
                    fig, ax1 = plt.subplots(figsize=(9, 6))
                    ax2 = ax1.twinx()
                    color = {10: "tab:blue", 5: "tab:orange", 1: "tab:green"}
                    plotObjects = []
                    plotObjects_ax2 = []
                    for i, t, f, yerr, sim_v, sim_f, sim_t in zip(
                        stim, expT, expF, expSTD, simV, fluorTrace, sim_time
                    ):
                        [tmp] = ax2.plot(
                            sim_t,
                            sim_v,
                            label=f"{gl.vm} simulation",
                            linestyle="-",
                            color=plotFigures.forceAlpha(color[i], 0.3),
                            zorder=rankDict[i],
                        )
                        plotObjects_ax2.append(tmp)
                        if correctArtifact:
                            sim_f += spl(sim_t)
                            label = f"{i} stim simulation"
                        else:
                            label = f"{i} stim simulation"
                        [tmp] = ax1.plot(
                            sim_t,
                            sim_f,
                            label=label,
                            linestyle="--",
                            color=color[i],
                            zorder=500 + rankDict[i],
                        )
                        plotObjects.append(tmp)
                        ax1.errorbar(
                            t, f, yerr=yerr, fmt="none", color=color[i], zorder=200 + i
                        )

                        tmp = ax1.scatter(
                            t,
                            f,
                            label=f"{i} stim experiment",
                            color=color[i],
                            zorder=201 + rankDict[i],
                        )
                        plotObjects.append(tmp)
                    ax1.set_zorder(ax2.get_zorder() + 1)
                    ax1.patch.set_visible(False)
                    ax2.set_xlim((100, 500))
                    legend = ax2.legend(
                        # [
                        #    "10 stim",
                        #    plotObjects_ax2[0],
                        #    "5 stim",
                        #    plotObjects_ax2[1],
                        #    "1 stim",
                        #    plotObjects_ax2[2],
                        # ],
                        # ["", "Simulation"] * 3,
                        title=r"$\bf{Membrane\ potential}$",
                        title_fontsize="large",
                        loc="upper right",
                        edgecolor=self.returnColor("model"),
                        # handler_map={str: LegendTitle({"fontsize": 16})},
                    )
                    legend.get_title().set_color(self.returnColor("model"))
                    plt.setp(legend.texts, color=self.returnColor("model"))
                    ax1.legend(
                        [
                            "1 stim",
                            plotObjects[4],
                            plotObjects[5],
                            "5 stim",
                            plotObjects[2],
                            plotObjects[3],
                            "10 stim",
                            plotObjects[0],
                            plotObjects[1],
                        ],
                        ["", "Simulation", "Experiment"] * 3,
                        title=r"$\bf{Fluorescence}$",
                        title_fontsize="large",
                        loc="upper left",
                        edgecolor=self.returnColor("fluor"),
                        handler_map={str: LegendTitle({"fontsize": 16})},
                    )
                    ax1.set_xlabel(gl.ms)
                    ax2.set_ylabel(gl.d_volt)
                    ax1.set_ylabel(gl.fluor)
                    _, ylim_value = gl.lim_d_volt  # mv
                    ax1.set_ylim((0, ylim_value * -1 / 10))
                    ax2.set_ylim((0, ylim_value))
                    for axObj, label in {ax2: "model", ax1: "fluor"}.items():
                        axObj.tick_params(axis="y", colors=self.returnColor(label))
                        axObj.yaxis.label.set_color(self.returnColor(label))

                    if correctArtifact:
                        Fname += "_correctedArtifact"

                    plt.savefig(f"../results/paperRes/{Fname}.pdf")
                    if np.isnan(total):
                        total = np.inf
                    elif verbose:
                        print(f"{total=}")

                    self.plotIKSeries.__wrapped__(
                        self,
                        [AllCells],
                        setKoylim=True,
                        setekylim=True,
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
                "kir2": 0,
                "clleak": 0,
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
            }
        )
        if funcArgs[-1]["kir2"] > 5:
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
                "kleak": self.leak,
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
        for kir in [-5, 0, 5]:
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
                    "kleak": self.leak,
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

    def flatten_list(self, nested_data):
        for item in nested_data:
            if isinstance(item, list):
                yield from self.flatten_list(item)
            else:
                yield item

    def split_and_remove(self, key, *args):
        original_len = len(args)
        new_list = []
        flat_list = list(self.flatten_list(args))

        for s in flat_list:
            new_list += s.split(key)

        for i, s in enumerate(new_list):
            new_list[i] = s.replace(key, "")

        new_len = len(new_list)
        if original_len == new_len:
            return new_list
        else:
            return self.split_and_remove(key, *new_list)

    def combine_distance_analysis_plots(self):
        if rank != 0:
            return
        curr_tag = self.tag
        # split tag with longer string first
        for splitter in ["_NoGlu", "_Glu", "_GABAR", "_GABA", "_NMDAR"]:
            curr_tag = self.split_and_remove(splitter, curr_tag)

        found_files = 0
        for chan in ["_GABAR_NoGlu_GABA", "_Glu", "_NMDAR"]:
            tmp_tag = ["distance_analysis"] + curr_tag[:-1] + [chan, curr_tag[-1]]
            tmp_tag = "".join(tmp_tag)
            path = os.path.join("intermediaryData", f"{tmp_tag}.pickle")
            if os.path.isfile(path):
                print("loading", path)
                with open(path, "rb") as handle:
                    AllCells = pickle.load(handle)

                if found_files == 0:
                    plt.cla()
                    plt.clf()
                    ax2 = None
                    ax = plt.figure(figsize=(9, 4.5)).gca()
                ax, ax2 = self.plot_shell(AllCells, ax, ax2=ax2)
                for ax_obj in [ax, ax2]:
                    lines = ax_obj.get_lines()
                    path = path.replace("_Glu", "_GluT")
                    color = self.returnColor(path)
                    lines[found_files].set_color(color)

                found_files += 1
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            labels=["GABA$_A$R", "GLT-1", "NMDAR"],
        )
        plt.tight_layout()

        if found_files == 3:
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"combined_minAmplitude_{tmp_tag}shellcomp.pdf",
                )
            )
        else:
            plt.cla()
            plt.clf()
            plt.close("all")

    def plot_shell(self, results, ax, ax2=None):
        resList = []
        shell_num = []
        for cells in results:
            for cell in cells:
                VCI = list(cell.VClampI)
                resList.append(min(VCI) - VCI[-1])
                shell_num.append(cell.shell)

        shell_num, resList = zip(*[(i, j) for i, j in sorted(zip(shell_num, resList))])
        resList = np.array(resList) * 1000  # nA to pA
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(shell_num, resList)
        ax.set_xlabel(gl.free("Shell number"))
        ax.set_ylabel(gl.free("Electrode " + gl.curr))
        ax.set_ylim(gl.lim_min_amp)

        if ax2 is None:
            ax2 = ax.inset_axes(
                [0.55, 0.25, 0.4, 0.4]
            )  # Define the position and size of the new subplot
        ax2.plot(
            shell_num,
            resList,
        )
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_xlim((2, 9))
        ax2.set_ylim((-20, 1))
        return ax, ax2

    @read_data
    def distance_analysis(self, shell_range=10):
        self.addChannelTag()
        syn_count = 50
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
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "KoSize": 3,
                "PAPLen": 0.3,
            }
        )
        if self.OE:
            funcArgs[-1]["kir2"] = self.KirMax
            self.tag += "_OE"

        if self.NMDAR and self.GluStim:
            funcArgs[-1]["multiple"] = self.optNMDAR * syn_count
        else:
            funcArgs[-1]["multiple"] = None
        if self.GluT and self.GluStim:
            funcArgs[-1]["GluTrans"] = self.optGluT * syn_count
        if self.GABAR and self.GabaStim:
            funcArgs[-1]["GABACount"] = self.optGABAR * syn_count
            # GABA alread calculates per section
        else:
            funcArgs[-1]["GABACount"] = 0

        ccList = ["shell"]
        iterations = [i for i in range(1, shell_range)]
        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            [
                [
                    "define_shell",
                    "select_shell",
                    "record_VClampI",
                    "initialize",
                    "multiSpike",
                    "run",
                ]
            ],
            [
                [
                    {"total_shell": shell_range, "synapse": syn_count},
                    {},
                    {},
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

        comm.Barrier()
        self.free_figure(results)

        self.combine_distance_analysis_plots()
        comm.Barrier()

        if rank == 0:
            plt.cla()
            plt.clf()
            ax = plt.figure(figsize=(9, 5.625)).gca()
            self.plot_shell(results, ax)
            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"minAmplitude_shellcomp{self.tag}.pdf"
                )
            )

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
                "kir2": None,
                "clleak": 0,
                "kleak": leak,
                "dt": self.dt,
                "seed": self.seed,
                "Glu": False,
                "multiple": None,
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
                "kleak": x,
                "dt": self.dt,
                "seed": self.seed,
                "Glu": False,
                "multiple": None,
            }
        )
        cellsRMP = PAPModel(**funcArgs[-1])
        cellsRMP.initialize()
        print(abs(cellsRMP.RMP + 80))
        loss = (cellsRMP.RMP + 80) ** 2
        return lossKO + loss

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
                "kir2": None,
                "clleak": 0,
                "kleak": leak,
                "dt": self.dt,
                "seed": self.seed,
                "Glu": False,
                "multiple": None,
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
                "kleak": x,
                "dt": self.dt,
                "seed": self.seed,
                "Glu": False,
                "multiple": None,
            }
        )
        cellsRMP = PAPModel(**funcArgs[-1])
        cellsRMP.initialize()
        print(abs(cellsRMP.RMP + 80))
        loss = (cellsRMP.RMP + 80) ** 2
        return lossKO + loss

    @read_data
    def freqComparison(self):
        self.addChannelTag()
        # Calculate the number of iterations for all parm sets
        spikeNumStep = 2
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
                "kleak": self.leak,
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
        self.free_figure(results)

        if rank == 0:
            # with open(
            #    os.path.join("intermediaryData", f"resultsParallel{self.tag}.pickle"),
            #    "wb",
            # ) as handle:
            #    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
            plt.ylabel(gl.free("Number of stimuli"))
            plt.xlabel(gl.free("Frequency (Hz)"))
            plt.xticks(
                range(0, int(spikeFreqMax / spikeFreqStep) + 1, 2),
                np.arange(0, spikeFreqMax + 1, spikeFreqStep * 2),
            )
            plt.yticks(
                range(int(spikeNumMax / spikeNumStep) + 1),
                np.arange(0, spikeNumMax + 1, spikeNumStep),
            )
            plt.colorbar(label=gl.vm, ticks=np.arange(0, 30, 5), extend="max")
            plt.clim(gl.clim_volt)
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
        seed = size
        testBools = [True, False]
        use_tau = False
        for PAP in testBools:
            for forcedAccum in testBools:
                exp = procedure(seed, 0)
                kwargs = {"PAP": PAP, "showFig": False}
                if PAP:

                    # initParms = (5, -72.5, 10, 19,0.5)
                    bounds = [(1, 1000), (-80, 100), (0, 10), (3, 20)]
                    if forcedAccum:
                        initParms = (exp.optGABAR, -67, 10, 19)
                        exp.GABAR = True
                    else:
                        initParms = (exp.optNMDAR, 100, 0.1, 3.97)
                        exp.GABAR = False
                    if not use_tau:
                        initParms = tuple([initParms[0]])
                        bounds = [bounds[0]]

                else:
                    # initParms = (5, -72.5, 10, 19,0.5)
                    #        glt, kir, PAPLen, KoSize, tau2, slowing
                    if forcedAccum:
                        use_tau = True
                    initParms = (
                        exp.optGluT,
                        exp.optKir,
                        1.93825396e00,
                        4.56907947e01,
                    )
                    bounds = [(0, 1e5), (0, 1e5), (0.3, 50), (0.5, 50)]

                    if forcedAccum and use_tau:
                        initParms += (
                            6.79258427e02,
                            9.98155414e03,
                        )
                        initParms = list(initParms)
                        initParms = [
                            5.38257517e-04,
                            9.78283818e-04,
                            1.93825396e00,
                            4.56907947e01,
                            6.79258427e02,
                            9.98155414e03,
                        ]
                        # initParms[0] = 0
                        # initParms[1] = 1
                        initParms = tuple(initParms)

                        bounds += [
                            (5.8, 10000),
                            (0, 10000),
                        ]

                    if not use_tau:
                        initParms = tuple(initParms[:2])
                        bounds = list(bounds[:2])

                kwargs = {
                    "use_tau": use_tau,
                    "autosave": False,
                    "PAP": PAP,
                    "showFig": False,
                }

                exp.fitExpDepolarization(
                    initParms,
                    showFig=True,
                    PAP=PAP,
                    skipsave=False,
                    use_tau=use_tau,
                )
                if not exp.foundFitExperiment:
                    print(kwargs["use_tau"])
                    res = minimize(
                        lambda x: exp.fitExpDepolarization(x, **kwargs),
                        initParms,
                        method="Nelder-Mead",
                        bounds=bounds,
                        tol=1,
                    )
                    mprint(res.x)
                    if res.success:
                        kwargs["showFig"] = True
                        kwargs["skipsave"] = False
                        exp.fitExpDepolarization(res.x, **kwargs)
                exp.foundfitExperiment = False

    elif size == 5 or size == 2 or size == 4:
        mprint("running bathExp")
        exp = procedure(4, 0)
        if size == 2:
            exp.bathExperiment()
        else:
            exp.bathExperiment(invivo=True)
    else:
        exp = procedure(6, 0)
        exp.uptakeRatio()

        # print('nothing set; try MPI -n 2 for kbath; MPI -n 3 for exp fit')
