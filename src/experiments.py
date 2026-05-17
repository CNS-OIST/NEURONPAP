from scipy.optimize import curve_fit
from astrocyte import *
import os
import numpy as np
from utils import *
from textSDIO import *
from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
import glob
from matplotlib.ticker import MaxNLocator
from plot_shape import *
from scipy.stats import f,t,f_oneway, ttest_rel,pearsonr,ttest_1samp,linregress
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import shgo
from scipy.interpolate import CubicSpline as spline
from scipy.signal import find_peaks
import json
import pandas as pd
import inspect
from functools import wraps
import sys
from math import ceil,floor,log
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.text as mtext
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset 
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from copy import copy,deepcopy
from importlib import reload
import statsmodels.api as sm
import statsmodels.stats.multitest as smm 

from matplotlib.patches import Arc
from matplotlib.patches import Wedge
from itertools import combinations
from scipy.ndimage import gaussian_filter


from global_labels import gl
import faulthandler
faulthandler.enable()
#from memory_profiler import profile

plt.rcParams.update(gl.font)
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
    nmda_color = "steelblue"
    glt_color = "lightblue"
    colorDict = {
        "NMDAR": nmda_color,
        "NMDA": nmda_color,
        "GABA": gaba_color,
        "GABAR": gaba_color,
        "GABA$_A$R": gaba_color,
        "GluT": glt_color,
        "GLT-1": glt_color,
        "iK": k_color,
        "K$^+$": k_color,
        "Soma": "deepskyblue",
        "PAP": "forestgreen",
        "Primary Branch":"lightblue",
        "fluor": "black",
        "Na": "gold",
        "Cl": "chocolate",
        "Ca": "olive",
        "model": "darkgray",
        "local": "darkgray",
        "global": "crimson",
    }

    @staticmethod
    def forceAlpha(color, alpha, bkg=(255, 255, 255, 1)):
        rgba_color = np.array(mcolors.to_rgba(color)) * np.array([255, 255, 255, 1])
        return (rgba_color * alpha + np.array(bkg) * (1 - alpha)) / np.array(
            [255, 255, 255, 1]
        )

    @staticmethod
    def p_to_stars(p):
        if p < 1e-4:
            return '****'
        elif p < 1e-3:
            return '***'
        elif p < 1e-2:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'


    @staticmethod
    def adjust_pvals(pvals, method='holm'):
        pvals = np.array(pvals)
        m = len(pvals)

        if method == 'bonferroni':
            return np.minimum(pvals * m, 1.0)

        elif method == 'holm':
            order = np.argsort(pvals)
            adjusted = np.empty(m)
            for count, idx in enumerate(order):
                adjusted[idx] = min((m - count) * pvals[idx], 1.0)
            # enforce monotonicity
            for i in range(m - 1):
                adjusted[order[i+1]] = max(adjusted[order[i+1]], adjusted[order[i]])
            return adjusted

        else:
            raise ValueError("method must be 'holm' or 'bonferroni'")



    @staticmethod
    def add_sig_from_category_pdict(ax, heights, pval_dict,
                                alpha=0.05,
                                y_offset=0.05, line_height=0.02,
                                text_offset=0.01, lw=1.5, fontsize=11):
        """
        - Anchors all brackets above the GLOBAL highest bar
        - Stacks brackets with no overlap across all categories
        """

        heights = np.asarray(heights)

        global_top = np.max(heights)

        all_p = [v for values in pval_dict.values() for v in values]
        adj_p = plotFigures.adjust_pvals(all_p)
        all_comparisons = {m:[] for m in pval_dict.keys()} 
        for m,p_vals in pval_dict.items():
            for (i,j),p in zip(list(combinations(range(len(heights)),2)),adj_p):
                if p < alpha:
                    all_comparisons[m].append((i,j,p))

        levels = []
        def get_level(i, j):
            level = 0
            while True:
                conflict = False
                for (li, lj, llevel) in levels:
                    if (max(i, li) <= min(j, lj)) and (llevel == level):
                        conflict = True
                        break
                if not conflict:
                    return level
                level += 1

        # draw
        for m,list_ids in enumerate(all_comparisons.values()):
            if list_ids == []:
                continue
            else:
                i,j,p = list_ids
            level = get_level(i, j)
            levels.append((i, j, level))

            y = global_top + y_offset + level * (line_height + y_offset)

            x1, x2 = i + (m-3/2)*0.2,j + (m-3/2)*0.2  

            ax.plot([x1, x1, x2, x2],
                    [y, y + line_height, y + line_height, y],
                    lw=lw, c='black')

            stars = plotFigures.p_to_stars(p)
            if stars:
                ax.text((x1 + x2) / 2,
                        y + line_height + text_offset,
                        stars,
                        ha='center', va='bottom',
                        fontsize=fontsize)

        # adjust ylim
        if levels:
            max_level = max(l for _, _, l in levels)
            ax.set_ylim(top=global_top + y_offset + (max_level + 2) * (line_height + y_offset))

    def get_papLen_color_from_value(self,value):
        vmid = 6.5454
        if self.peakLen is None:
            self.peakLen = 100
        if not hasattr(self,'paplen_cm'):
            if vmid < self.peakLen:
                vmin = self.PAPLen
                vmax = self.peakLen
                pos_mid = (vmid - vmin) / (vmax - vmin)
                colors = np.array([
                    mcolors.to_rgb(self.returnColor('PAP')),
                    mcolors.to_rgb(self.returnColor('Soma')),
                    mcolors.to_rgb(self.returnColor('global'))     
                ])
                N = 256
                x = np.array([0,pos_mid,1.0])
                x_log = np.log10(x + 1e-6)

                xi = np.linspace(0, 1, N)
                xi_log = np.log10(xi + 1e-6)

                x_log = (x_log - x_log.min()) / (x_log.max() - x_log.min())
                xi_log = (xi_log - xi_log.min()) / (xi_log.max() - xi_log.min())

                cmap_array = np.zeros((N, 3))
                for i in range(3):
                    cmap_array[:, i] = np.interp(xi_log, x_log, colors[:, i])

                cm = mcolors.LinearSegmentedColormap.from_list("custom_gb", cmap_array)
                self.paplen_cm = cm
            else:
                colors = [self.returnColor('PAP'), self.returnColor('Soma')]  # Green to Blue
                cmap_name = "green_to_blue"
                cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
        else:
            cm = self.paplen_cm

        if not hasattr(self,'paplen_norm'):
            norm = mcolors.Normalize(vmin=self.PAPLen, vmax=self.peakLen)
            self.paplen_norm = norm
        else:
            norm = self.paplen_norm
        rgba = cm(norm(value))
        return mcolors.to_hex(rgba)

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
                if not os.path.isfile(fPath) or (hasattr(self,'override_src') and self.override_src):
                    if isinstance(AllCells, LazySharedObject):
                        AllCells.dump_to_file(fPath)
                    else:
                        with open(fPath, "wb") as handle:
                            pickle.dump(AllCells, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Saved src data file {fName}")
            res = plot_func(self, *args, **kwargs)
            return res

        return wrapper

    @save_src_Data
    def free_figure(self, AllCells):
        pass
        #if size != 1:
        #    del AllCells
        # just to force call decorator

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

    def returnColor(self, key, words=False):
        for typeName in self.colorDict.keys():
            if typeName in key:
                if not words:
                    return self.colorDict[typeName]
                else:
                    rgb = mcolors.to_rgb(self.colorDict[typeName])
                    hsv = mcolors.rgb_to_hsv(rgb)
                    new_hsv = hsv.copy()
                    new_hsv[2] = hsv[2] * 0.9
                    new_rgb = mcolors.hsv_to_rgb(new_hsv)
                    return new_rgb
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
        plt.tight_layout()
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
        panelF=False,
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
                        if 'gabaBath' in self.tag:
                            endStim = startStim + 1
                        else:
                            endStim = max(cell.time) + 10
                    fig, (ax, ax2) = plt.subplots(1, 2, figsize=gl.figsize_panel)
                    fig.subplots_adjust(wspace=0.5)
                else:
                    fig, ax = plt.subplots(figsize=gl.figsize_ikPlots)

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
                if not bath:
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
                    if 'gabaBath' in self.tag:
                        lowerBound,upperBound = (-95,-20)
                        ax.set_ylim((lowerBound,upperBound))


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
                            label=f"Local\nstim.",
                        )
                        ax.add_patch(r)
                    else:
                        ax.hlines(
                            startBar,
                            startStim,
                            endStim,
                            color=self.returnColor(self.locality),
                            linewidth=lineWidth,
                            label=f"Global\nstim.",
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

                if not bath:
                    plt.tight_layout()
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
                plt.tight_layout()
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
                        label=gl.current_ion("NMDAR"),
                        color=self.returnColor("NMDAR"),
                    )
                if hasattr(cell, "iGABA") and self.GABAR:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGABA)[initStep:],
                        label=gl.current_ion("GABA_AR"),
                        color=self.returnColor("GABAR"),
                    )
                if hasattr(cell, "iGluT") and self.GluT:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluT)[initStep:],
                        label=gl.current_ion("GLT"),
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
                        label=gl.current_ion("GLT"),
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

                plt.tight_layout()
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
                    fig = plt.figure(figsize=gl.figsize_panel)

                    ax = fig.add_axes([0.2, 0.52, 0.7, 0.40])
                    ax_inset = fig.add_axes([0.2, 0.15, 0.7, 0.25])
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

                    ax.legend()
                    ax.set_ylim(gl.lim_ek)
                    ax_inset.plot(
                        list(cell.time)[initStep:],
                        list(cell.vPAP)[initStep:],
                        label=f"PAP {gl.vm}",
                        color=self.returnColor("PAP"),
                    )

                    x0, x1 = 145, 155
                    y0, y1 = gl.lim_ek_zoom
                    grey = "0.5"
                    light_grey = grey

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
                        color=light_grey,
                        linewidth=1,
                        zorder=3,
                        linestyle="--",
                    )
                    con2 = ConnectionPatch(
                        xyA=(x1, y0),
                        coordsA=ax.transData,
                        xyB=(x1, y1),
                        coordsB=ax_inset.transData,
                        color=light_grey,
                        linewidth=1,
                        zorder=3,
                        linestyle="--",
                        connectionstyle="angle,angleA=96,angleB=-1,rad=30",
                    )

                    fig.add_artist(con1)
                    fig.add_artist(con2)

                    ax.set_xlabel(gl.ms, zorder=10)
                    ax.set_ylabel(gl.volt, zorder=4)
                    ax_inset.set_xlabel(gl.ms, color=grey, zorder=4)
                    ax_inset.set_ylabel(gl.volt, color=grey, zorder=4)

                    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                        lbl.set_zorder(5)

                    for lbl in ax_inset.get_xticklabels() + ax_inset.get_yticklabels():
                        lbl.set_zorder(5)
                    left = ax.get_position().x0
                    top = ax.get_position().y1

                    if int(cell.KoSize) > 0:
                        format_ko = int(cell.KoSize)
                    else:
                        format_ko = f"{cell.KoSize:.1f}"
                    fig.text(
                        left + 0.01,
                        top + 0.01,
                        f'{format_ko}{gl.mM_raw} {gl.delta_ion_o("K",short=True,unit=False)}',
                        fontsize=plt.rcParams["axes.labelsize"],
                        ha="left",
                        va="bottom",
                    )

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
                plt.figure(figsize=gl.figsize_panel_long)
                total = [
                    max(getattr(cell, recVal)) for cell in AllRes[compVal].values()
                ]
                total = np.array(total)

                mprint(f"{compVal=}",f"{recVal=}",total.mean(), total.std())
                for cell in AllRes[compVal].values():
                    alpha = 1
                    if getattr(cell, merge) != selected:
                        alpha = 0.3
                    initStep = self.get_initStep(cell,shift=5)
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

                plt.xlabel(gl.ms)
                plt.xlim((148,156))
                if recVal == "vPAP":
                    plt.ylabel(gl.volt)
                else:
                    plt.ylabel(recVal)
                plt.ylim((-90,-60))
                plt.title(
                    label=f"mean={total.mean()},std={total.std()},med={np.median(total)}"
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"{recVal}_Merged_for_{comparison}={compVal}_over_{merge}_{zoom=}.pdf",
                    )
                )
                plt.close("all")

    def plot_combined_cvk(
        self,
        AllCells,
    ):
        for cells in AllCells:
            for cell in cells:
                initStep = self.get_initStep(cell)
                fig = plt.figure(figsize=gl.figsize_panel_long)
                fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.15)
                gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.5)
                ax_volt = fig.add_subplot(gs[0, 0])
                ax_curr = fig.add_subplot(gs[0, 1])
                ax_curr_inset = ax_curr.inset_axes([0.55, 0.1, 0.33, 0.25])
                ax_ko = fig.add_subplot(gs[0, 2])
                ax_ko.plot(
                    list(cell.time)[initStep:],
                    list(cell.KoPAP)[initStep:],
                    label=f"PAP",
                    color=self.returnColor("PAP"),
                )
                ax_ko.plot(
                    list(cell.time)[initStep:],
                    list(cell.KoSoma)[initStep:],
                    label="Soma",
                    color=self.returnColor("Soma"),
                )
                ax_ko.set_ylim(gl.lim_cvk_ko)
                ax_ko.set_ylabel(gl.ion_o("K"))
                ax_ko.set_xlabel(gl.ms)
                ax_ko.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax_volt.set_xlabel(gl.ms)
                ax_ko.legend()

                ax_volt.plot(
                    list(cell.time)[initStep:],
                    list(cell.vPAP)[initStep:],
                    label=f"PAP {gl.vm}",
                    color=self.returnColor("PAP"),
                )
                ax_volt.plot(
                    list(cell.time)[initStep:],
                    list(cell.ekPAP)[initStep:],
                    label=f"PAP {gl.ek_raw}",
                    color=self.returnColor("PAP"),
                    linestyle="--",
                )
                ax_volt.plot(
                    list(cell.time)[initStep:],
                    list(cell.vSoma)[initStep:],
                    label=f"Soma {gl.vm}",
                    color=self.returnColor("Soma"),
                )
                ax_volt.plot(
                    list(cell.time)[initStep:],
                    list(cell.ekSoma)[initStep:],
                    label=f"Soma {gl.ek_raw}",
                    color=self.returnColor("Soma"),
                    linestyle="--",
                )

                ax_volt.set_ylim(gl.lim_cvk_volt)
                ax_volt.set_ylabel(gl.volt)
                ax_volt.set_xlabel(gl.ms)
                ax_volt.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax_volt.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax_volt.legend()


                for ax in [ax_curr,ax_curr_inset]:
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
                    if hasattr(cell, "iGluT") and self.GluT:
                        ax.plot(
                            list(cell.time)[initStep:],
                            list(cell.iGluT)[initStep:],
                            label=gl.current_ion("GLT"),
                            color=self.returnColor("GluT"),
                    )
                if hasattr(cell, "iNMDA") and self.NMDAR:
                    ax_curr.plot(
                        list(cell.time)[initStep:],
                        list(cell.iNMDA)[initStep:],
                        label=gl.current_ion("NMDAR"),
                        color=self.returnColor("NMDAR"),
                    )
                if hasattr(cell, "iGABA") and self.GABAR:
                    ax_curr.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGABA)[initStep:],
                        label=gl.current_ion("GABAR"),
                        color=self.returnColor("GABAR"),
                    )

                mark_inset(ax_curr, ax_curr_inset, loc1=1, loc2=2, fc="none", ec="lightgray",lw=0.5,linestyle='--',zorder=0)
                ax_curr_inset.set_xlim(right=300)
                ax_curr_inset.set_ylim(gl.lim_curr_inset)
                ax_curr.set_xlabel(gl.ms)
                ax_curr.set_ylabel(gl.curr)
                ax_curr.set_ylim(gl.lim_curr)
                ax_curr.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax_curr.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
                ax_curr.legend()

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"combined_cvk{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}_{cell.SpikeNum}.pdf",
                    )
                )
                plt.close("all")

    def setLabelColors(
        self, area, Kir=True, x=False, y=False, chanOverride=None, labelObj=None
    ):
        stdChannelDict = {
            "Kir": (370 * area + 1 * 4.7e3 * area, 1 * area),
            "GluT": (14248 * area, 812 * area),
            "GABAR": (self.optGABAR, 10),
            "GABA$_A$R": (self.optGABAR, 10),
            "NMDAR": (self.optNMDAR, 50),
            "NKA": (1, 0.5),
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
                if labelObj is not None:
                    _, labels = labelObj
                else:
                    _, labels = plt.yticks()
            else:
                if self.GluT:
                    mean, std = stdChannelDict["GluT"]
                else:
                    mean, std = stdChannelDict["PAPLen"]
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
            elif self.NKA and Kir:
                mean, std = stdChannelDict["NKA"]
            elif self.NMDAR and Kir:
                mean,std = stdChannelDict['NMDAR']
            else:
                mean, std = stdChannelDict["PAPLen"]
            if labelObj is not None:
                labels, _ = labelObj
            else:
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

        fig, axes = plt.subplots(1, 2, figsize=gl.figsize_panel, sharey=True)
        if self.NKA:
            fig.subplots_adjust(left=0.2, right=0.95, bottom=0, top=1)
        else:
            fig.subplots_adjust(left=0.2, right=0.9, bottom=0.05, top=1)

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
                skip = 1
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
                    ha="center",
                    va="top",
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
                    ha="center",
                    va="top",
                )
            if self.NKA:
                for labels in [ax.get_xticklabels(), ax.get_yticklabels()]:
                    for label in labels:
                        label.set_fontsize(8)
        if self.NMDAR:
            xlabel = gl.chan_num("NMDAR")
        elif self.GluT:
            xlabel = gl.chan_num("GLT-1")
        elif self.GABAR and Kir:
            xlabel = gl.chan_num("GABA$_A$R")
        elif self.GAP:
            xlabel = gl.chan_num("Cx43")
        elif self.NKA:
            xlabel = gl.density_num("NKA")
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
        chanOverride = {
            "Kir": (
                int(res[0].PAPKirCount),
                ceil(res[0].PAPKirCount_std),
            ),
            "GluT": (
                int(res[0].PAPGluTCount),
                ceil(res[0].PAPGluTCount_std),
            ),
        }
        self.setLabelColors(
            results[0][0].PAParea,
            Kir=True,
            y=True,
            x=True,
            labelObj=(axes[0].get_xticklabels(), axes[0].get_yticklabels()),
            chanOverride=chanOverride,
        )
        self.setLabelColors(
            results[0][0].PAParea,
            Kir=True,
            y=False,
            x=True,
            labelObj=(axes[1].get_xticklabels(), axes[1].get_yticklabels()),
            chanOverride=chanOverride,
        )

        _, cbarMax = gl.clim_volt
        fig.colorbar(
            im1,
            label=gl.d_volt_short,
            ticks=np.arange(0, cbarMax, 2),
            extend="max",
            ax=axes.ravel().tolist(),
            shrink=0.5 if not self.NKA else 0.3,
        )
        left = axes[0].get_position().x0
        right = axes[1].get_position().x1
        bottom = axes[0].get_position().y0
        top = axes[0].get_position().y1
        fig.text(
            (left + right) / 2,
            bottom - 0.15,
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

    def combined_somaPAP_heatmap(self, results, Kir=True):
        vmin, vmax = gl.clim_volt
        plt.cla()
        plt.clf()
        plt.close("all")

        fig, axes = plt.subplots(1, 2, figsize=gl.figsize_panel, sharey=True)
        if self.NKA:
            fig.subplots_adjust(left=0.2, right=0.95, bottom=0, top=1)
        else:
            fig.subplots_adjust(left=0.2, right=0.9, bottom=0.05, top=1)

        imarray_pap = self.createIMArray(results, "vPAP", Kir=Kir)
        imarray_soma = self.createIMArray(results, "vSoma", Kir=Kir)


        im1 = axes[0].imshow(
            imarray_pap,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            origin="lower",
            interpolation="nearest",
        )
        im2 = axes[1].imshow(
            imarray_soma,
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
                skip = 1
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
                    ha="center",
                    va="top",
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
                    ha="center",
                    va="top",
                )
            if self.NKA:
                for labels in [ax.get_xticklabels(), ax.get_yticklabels()]:
                    for label in labels:
                        label.set_fontsize(8)
        if self.NMDAR:
            xlabel = gl.chan_num("NMDAR")
        elif self.GluT:
            xlabel = gl.chan_num("GLT-1")
        elif self.GABAR and Kir:
            xlabel = gl.chan_num("GABA$_A$R")
        elif self.GAP:
            xlabel = gl.chan_num("Cx43")
        elif self.NKA:
            xlabel = gl.density_num("NKA")
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
        chanOverride = {
            "Kir": (
                int(res[0].PAPKirCount),
                ceil(res[0].PAPKirCount_std),
            ),
            "GluT": (
                int(res[0].PAPGluTCount),
                ceil(res[0].PAPGluTCount_std),
            ),
        }
        self.setLabelColors(
            results[0][0].PAParea,
            Kir=True,
            y=True,
            x=True,
            labelObj=(axes[0].get_xticklabels(), axes[0].get_yticklabels()),
            chanOverride=chanOverride,
        )
        self.setLabelColors(
            results[0][0].PAParea,
            Kir=True,
            y=False,
            x=True,
            labelObj=(axes[1].get_xticklabels(), axes[1].get_yticklabels()),
            chanOverride=chanOverride,
        )

        _, cbarMax = gl.clim_volt
        fig.colorbar(
            im1,
            label=gl.d_volt_short,
            ticks=np.arange(0, cbarMax, 2),
            extend="max",
            ax=axes.ravel().tolist(),
            shrink=0.5 if not self.NKA else 0.3,
        )
        left = axes[0].get_position().x0
        right = axes[1].get_position().x1
        bottom = axes[0].get_position().y0
        top = axes[0].get_position().y1
        fig.text(
            (left + right) / 2,
            bottom - 0.15,
            xlabel,
            fontsize=plt.rcParams["axes.labelsize"],
            ha="center",
            va="top",
        )
        fig.text(
            axes[0].get_position().x0,
            top + 0.01,
            f"PAP {gl.vm}",
            fontsize=plt.rcParams["axes.labelsize"],
            ha="left",
            va="bottom",
        )
        fig.text(
            axes[1].get_position().x0,
            top + 0.01,
            f"Soma {gl.vm}",
            ha="left",
            va="bottom",
            fontsize=plt.rcParams["axes.labelsize"],
        )

        plt.savefig(
            os.path.join(
                "../results/paperRes", f"combined_heatmap_somaPAPComp_{self.tag}.pdf"
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
        if self.NMDAR:
            self.combined_somaPAP_heatmap(results,Kir=Kir)
        for PAPattr in ["vPAP", "vSoma"]:
            self.combined_heatmap(results, PAPattr, Kir=Kir)
            imArray = self.createIMArray(results, PAPattr, Kir=Kir)
            cmap = "magma"
            plt.cla()
            plt.clf()
            plt.figure(figsize=gl.figsize_panel)
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
                plt.xlabel(gl.density_num("NKA"))
            _, cbarMax = gl.clim_volt
            plt.colorbar(
                label=gl.d_volt_short,
                ticks=np.arange(0, cbarMax, 2),
                extend="max",
                shrink=0.9,
            )
            plt.clim((0, cbarMax))
            if stdLabels:
                self.setLabelColors(
                    res[0].PAParea,
                    Kir=Kir,
                    x=True,
                    y=False,
                    chanOverride={
                        "GluT": (res[0].PAPGluTCount, res[0].PAPGluTCount_std),
                        "Kir": (res[0].PAPKirCount, res[0].PAPKirCount_std),
                    },
                )

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"FullComparison{tag}_{PAPattr}.pdf"
                )
            )

    @save_src_Data
    def plot_physiological(self, AllCells, stim, papcounts, models,syn_count=25):
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
                fig = plt.figure(figsize=gl.figsize_halfh)
                gs = fig.add_gridspec(nrows=6, ncols=2, hspace=0.3)
                ax = []
                ax_r = []
                inset = (375, 385)
                inset_y = (-90, -85)
                ax.append(fig.add_subplot(gs[0:2, 0]))
                ax.append(fig.add_subplot(gs[2:4, 0], sharex=ax[0]))
                ax.append(fig.add_subplot(gs[4:6, 0], sharex=ax[0]))
                if location == 'vPAP':
                    ax_r.append(fig.add_subplot(gs[4, 1]))
                    ax_r.append(fig.add_subplot(gs[5, 1], sharex=ax_r[0]))
                ax_summary = fig.add_subplot(gs[:3, 1])
                summary_res = {}
                all_res = {}
                if location == 'vPAP' and p > 1:
                    all_pvalues = {} 
                    heights = []
                    anova_p = {}
                for j, s in enumerate(stim):
                    summary_res[s] = []
                    all_res[s] = []
                    if j != len(stim) - 1:
                        ax[j].tick_params(labelbottom=False)
                    for m in models:
                        # flattened
                        # Forced may be buggy
                        #
                        cell = AllCells[index + j * len(models)]

                        initStep = self.get_initStep(cell)
                        calc_max = self.get_initStep(cell, shift=-100)
                        if location == 'vPAP' and p > 1 and hasattr(cell,'recordAllPAP'):
                            items = []
                            for i,vPAP_record in enumerate(getattr(cell, location)): 
                                if max(list(vPAP_record)[calc_max:]) - cell.RMP[i] < 80:
                                    items.append(max(list(vPAP_record)[calc_max:]) - cell.RMP[i])
                                if i == 0:
                                    ax[j].plot(
                                        list(cell.time)[initStep:],
                                        list(vPAP_record)[initStep:],
                                        label=m,
                                        color=self.returnColor(m),
                                    )
                                    index += 1
                                    if s == "theta" and location == 'vPAP':
                                        if "K$^+$" in m:
                                            ax_r[0].plot(
                                                list(cell.time)[initStep:],
                                                list(vPAP_record)[initStep:],
                                                label=m,
                                                color=self.returnColor(m),
                                            )
                                            ax_r[0].set_xlim(*inset)
                                            ax_r[0].set_ylim(*inset_y)
                                            ax_r[0].tick_params(labelbottom=False)
                                        elif "GLT-1" in m:
                                            ax_r[1].plot(
                                                list(cell.time)[initStep:],
                                                list(vPAP_record)[initStep:],
                                                label="GLT-1",
                                                color=self.returnColor(m),
                                            )
                                            ax_r[1].set_xlim(*inset)
                                            ax_r[1].set_ylim(*inset_y)
                                            ax_r[1].set_xlabel(gl.ms)


                            mean = np.mean(items)
                            std = np.std(items)
                            summary_res[s].append((mean,std))
                            all_res[s].append(items)

                        else:
                            if type(cell.RMP) == list:
                                cell.RMP = cell.RMP[0]
                            summary_res[s].append(
                                max(list(getattr(cell, location))[calc_max:]) - cell.RMP
                            )
                            ax[j].plot(
                                list(cell.time)[initStep:],
                                list(getattr(cell, location))[initStep:],
                                label=m,
                                color=self.returnColor(m),
                            )
                            index += 1
                            if s == "theta" and location =='vPAP':
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
                                elif "GLT-1" in m:
                                    ax_r[1].plot(
                                        list(cell.time)[initStep:],
                                        list(getattr(cell, location))[initStep:],
                                        label="GLT-1",
                                        color=self.returnColor(m),
                                    )
                                    ax_r[1].set_xlim(*inset)
                                    ax_r[1].set_ylim(*inset_y)
                                    ax_r[1].set_xlabel(gl.ms)

                    if location == "vPAP":
                        ax[j].set_ylim(gl.lim_ek_zoom)
                    else:
                        ax[j].set_ylim(gl.lim_VmembSoma)
                    ax[j].set_xlim(right=500)

                index -= len(models) * (len(stim) - 1)

                groups = []
                for k in summary_res.keys():
                    label = (
                        f"{k} {gl.unit_hz_raw}" if type(k) is int else k.capitalize()
                    )
                    groups.append(label)

                values = np.array(list(summary_res.values()))
                


                n_groups = len(groups)
                n_bars = values.shape[1]
                bar_colors = [self.returnColor(m) for m in models]

                x = np.arange(n_groups)
                width = 0.2
                if location == 'vPAP' and p > 1 and hasattr(cell,'recordAllPAP'):
                    for i in range(n_bars):
                        mean,std = zip(*values[:,i])
                        heights.append(np.max(mean + std))
                        xcoords = x + (i - (n_bars - 1) / 2) * width
                        ax_summary.bar(
                            xcoords,
                            mean,
                            yerr=std,
                            width=width,
                            color=bar_colors[i],
                            edgecolor="black",
                        )
                        
                        def ttest_from_summary(mean1, std1, n1, mean2, std2, n2):
                            # Welch’s t-test
                            se = np.sqrt(std1**2 / n1 + std2**2 / n2)
                            t_stat = (mean1 - mean2) / se

                            # degrees of freedom (Welch–Satterthwaite)
                            df = (std1**2 / n1 + std2**2 / n2)**2 / (
                                (std1**2 / n1)**2 / (n1 - 1) +
                                (std2**2 / n2)**2 / (n2 - 1)
                            )

                            # two-tailed p-value
                            p = 2 * (1 - t.cdf(abs(t_stat), df))

                            return p

                        all_pvalues[models[i]] = [ttest_from_summary(*values[:,i][j],syn_count,*values[:,i][k],syn_count) for j,k in list(combinations(range(len(mean)),2))]
                    
                        def anova_from_summary(means, stds, ns):
                            means = np.array(means, dtype=float)
                            stds  = np.array(stds,  dtype=float)
                            ns    = np.array(ns,    dtype=float)

                            k = len(means)
                            N = np.sum(ns)

                            # grand mean
                            grand_mean = np.sum(ns * means) / N

                            # between-group sum of squares
                            ss_between = np.sum(ns * (means - grand_mean)**2)

                            # within-group sum of squares
                            ss_within = np.sum((ns - 1) * stds**2)

                            # degrees of freedom
                            df_between = k - 1
                            df_within  = N - k

                            # mean squares
                            ms_between = ss_between / df_between
                            ms_within  = ss_within / df_within

                            # F statistic
                            F = ms_between / ms_within

                            # p-value
                            p = 1 - f.cdf(F, df_between, df_within)

                            return p

                        anova_p[models[i]]=  anova_from_summary(mean,std,[syn_count]*len(mean))
                        



                else:
                    for i in range(n_bars):
                        ax_summary.bar(
                            x + (i - (n_bars - 1) / 2) * width,
                            values[:, i],
                            width=width,
                            color=bar_colors[i],
                            edgecolor="black",
                        )

                ax_summary.set_xticks(x)
                ax_summary.set_xticklabels(groups)
                ax_summary.set_ylabel(gl.d_volt)
                if location == 'vPAP':
                    ax_summary.set_ylim(gl.lim_d_volt)
                else:
                    ax_summary.set_ylim((0,1))
                handle, label = ax[-1].get_legend_handles_labels()

                if location == 'vPAP' and p > 1:
                    for i,(l,anova) in enumerate(zip(label,anova_p.values())):
                        label[i] = f': {self.p_to_stars(anova)}'


                    ax_summary.legend(
                        loc='upper center',
                        handles=handle,
                        bbox_to_anchor=(0.5, -0.11),
                        title='ANOVA',
                        columnspacing=0.8,
                        handletextpad=0.4,
                        borderaxespad=0.2,
                        labelspacing=0.3,
                        ncol=len(handle),
                        labels=label,
                    )

                    self.add_sig_from_category_pdict(ax_summary,heights,all_pvalues)

                for i in range(len(ax_r)):
                    ax_r[i].yaxis.set_major_locator(MaxNLocator(integer=True,nbins=3))
                ax[-1].set_xlabel(gl.ms)
                # plt.title(f"stim:{s} PAP counts = {p}")
                bottom = ax[-1].get_position().y0
                top = ax[0].get_position().y1

                left = ax[0].get_position().x0
                right = ax_summary.get_position().x1
                dx = (right-left)/(len(models)-1)
                label_pos = ['left','center','center','right']
                for i,m in enumerate(models):
                    fig.text(
                        dx*i+left,
                        top + 0.015,
                        m,
                        color=self.returnColor(m),
                        fontsize=plt.rcParams["axes.labelsize"],
                        ha=label_pos[i],
                        va="bottom",
                        fontweight="bold",
                    )

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
                    f"50 {gl.unit_hz_raw}",
                    fontsize=plt.rcParams["axes.labelsize"],
                    ha="left",
                    va="top",
                )
                fig.text(
                    left + 0.01,
                    ax[1].get_position().y1 - 0.02,
                    f"100 {gl.unit_hz_raw}",
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
                if location == 'vPAP':
                    x0, x1 = inset
                    y0, y1 = inset_y
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

                    con1 = ConnectionPatch(
                        xyA=(x1, y0),
                        coordsA=ax[-1].transData,
                        xyB=(x0 - 1.5, y0),
                        coordsB=ax_r[1].transData,
                        color=grey,
                        linewidth=1,
                        zorder=3,
                        linestyle="--",
                    )
                    con2 = ConnectionPatch(
                        xyA=(x1, y1),
                        coordsA=ax[-1].transData,
                        xyB=(x0 - 1, y1),
                        coordsB=ax_r[0].transData,
                        color=grey,
                        linewidth=1,
                        zorder=3,
                        linestyle="--",
                    )


                    ax[-1].add_patch(rect)
                    fig.add_artist(con1)
                    fig.add_artist(con2)

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"combined_pap={p}_{location}{self.tag}.pdf",
                    )
                )
        plt.close("all")

    def plot_fluor_comparison(self,AllCells):
        fig = plt.figure(figsize=gl.figsize_panel)
        fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.15)
        gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.5)
        ax_volt = fig.add_subplot(gs[0])
        ax_ko = fig.add_subplot(gs[1],sharey=ax_volt)
 
        for cells in AllCells:
            for cell in cells:
                if cell.seed == 1:
                    initStep = self.get_initStep(cell)
                    dF = (max(list(cell.fluorVPAP)[initStep:]) - cell.RMP) * -1/10
                    ax_ko.scatter(max(list(cell.KoPAP)[initStep:]),dF,color='black')
                    ax_volt.scatter(max(list(cell.vPAP)[initStep:]),dF,color='black')

        ax_volt.invert_yaxis()
        ax_volt.set_ylabel(gl.fluor)
        ax_ko.set_ylabel(gl.fluor)
        ax_volt.set_xlabel(gl.volt)
        ax_ko.set_xlabel(gl.ion_o('K'))
        ax_volt.set_ylim(0,-2.5)
        plt.savefig(os.path.join("../results/paperRes",f"fluor_comparison{self.tag}.pdf"))


class procedure(plotFigures):
    alpha = 0.05
    leak = 1.15  # ideal calculated from stable model
    optKir = 0  # std * optkir  + mean
    optNMDAR = 344
    optGABAR = 1320
    optGluT = 0  # std * optGluT + mean
    optNKA = 1
    maxNKA = 10
    spillOverLen = 1.78832533e+01
    spillOverSlowing = 6.03081751e+02
    OEpump = 150
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
    NKA = False
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
    PAPLenMax = 10 
    peakLen = None
    pb_seed_max = 7
    kdifl = False

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
        if self.NKA:
            self.tag += "_NKA"
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
        if hasattr(self,'kdifl') and self.kdifl:
            self.tag += '_intra_diff'
        # print(f'{self.GluT=}')
        # print(f'{self.NMDAR=}')
        #
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
                    if func_name == "find_missing_iter":
                        func_name = inspect.stack()[2].function


                intermediary_files = os.listdir(os.path.join("intermediaryData"))
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
                            mprint(f"found intermediary file {f}\r")
                        sys.stdout.flush()
                        AllCells = [[]]
                        if self.global_rw_data or rank == 0:
                            with open(
                                os.path.join("intermediaryData", f), "rb"
                            ) as handle:
                                AllCells = pickle.load(handle)

                        if not self.global_rw_data:
                            AllCells = load_interm_data(AllCells,root=0)

                        # temporarily override simulation and just output result
                        def parallizeFor_dummy(*args, AllCells=AllCells,**kwargs):
                            if len(args) > 0 and not self.global_rw_data:
                                AllCells = self.find_missing_iter(AllCells,*args,**kwargs)
                            return AllCells

                        with global_function_override_runtime(
                            "parallizeFor",
                            parallizeFor_dummy,
                            module_name=calling_module_name,
                        ):
                            result = exp_func(self, *args, **kwargs)
                            #if not self.global_rw_data:
                            #    release_pickle(AllCells,win)
                        return result

            else:
                mprint("not reading intermediary data")
            return exp_func(self, *args, **kwargs)

        return wrapper

    def find_missing_iter(
        self,
        AllCells,
        iterations,
        functions,
        functionArgs,
        functionParms,
        callmethods,
        methodArgs,
        mode="InitArgs",
        randomize=True,
    ):
        missing_iter = deepcopy(iterations) 
        if AllCells is not None:
            for cells in AllCells:
                for cell in cells:
                    cell_iter = []
                    for cell_attr in functionParms:
                        if hasattr(cell,cell_attr):
                            cell_iter.append(getattr(cell, cell_attr))
                        elif cell_attr in cell.GENEDict.keys():
                            cell_iter.append(cell.GENEDict[cell_attr])
                        else:
                            break

                    cell_iter = tuple(cell_iter)
                    if len(cell_iter) == len(functionParms):
                        if len(cell_iter) == 1:
                            cell_iter = cell_iter[0]

                        if cell_iter in iterations and cell_iter in missing_iter:
                            missing_iter.remove(cell_iter)

        if len(missing_iter) > 0:
            mprint(f'Found missing iters {missing_iter}; rerun')
            self.override_src = True
            import utils
            results = utils.parallizeFor(
                    missing_iter,
                    functions,
                    functionArgs,
                    functionParms,
                    callmethods,
                    methodArgs,
                    mode=mode,
                    randomize=randomize,
            )

            if AllCells is None:
                AllCells = results
            else:
                if rank == 0:
                    AllCells = AllCells.load()
                    AllCells += results
                    # this only saves result as backup and not actually accesible to other functions
                    #self.free_figure(AllCells)
                else:
                    AllCells = None


            comm.barrier()
            return AllCells
        else:
            return AllCells


    def match_attr(self,a,b,functionParms):
        for a_cells in a:
            for a_cell in a_cells:
                for b_cells in b:
                    for b_cell in b_cells:
                        if getattr(a_cell,functionParms[0]) == getattr(b_cell,functionParms[0]):
                            if getattr(a_cell,functionParms[1]) == getattr(b_cell,functionParms[1]):
                                for attr in b_cell.__dict__:
                                    if not hasattr(a_cell,attr):
                                        setattr(a_cell,attr,deepcopy(getattr(b_cell,attr)))




    def Add_callMethods(
        self,
        AllCells,
        iterations,
        functions,
        functionArgs,
        functionParms,
        callmethods,
        methodArgs,
        mode="InitArgs",
        randomize=True,
    ):
        # specialized call for only non simulation run
                #
                #
        for method in callmethods:
            if method in ['run','setK','setKBath','replayK','replayKBath']:
                wMessage(f'cannot run call method {method}')
                return
                

        from utils import parallizeFor as pf
        results = pf(
                iterations,
                functions,
                functionArgs,
                functionParms,
                callmethods,
                methodArgs,
                mode=mode,
                randomize=randomize,
        )
        results = comm.bcast(results, root=0)

        self.match_attr(AllCells,results)
                
           
        self.override_src = True
        return AllCells


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
        plt.tight_layout()
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

                ax.scatter3D(dList, cList, v, c=v, cmap="viridis")
                if i == 0:
                    name = "soma"
                else:
                    name = "PAP"
                ax.set_zlabel(gl.d_volt)

                j = ""
                while os.path.isfile(f"./3Dplot{name}{j}.pdf"):
                    if j == "":
                        j = 1
                    else:
                        j += 1

                plt.tight_layout()
                plt.savefig(
                    os.path.join("../results/paperRes", f"./3Dplot{name}{j}.pdf")
                )

    def readIterationRi(self, all_file_names, dName="../results/paperRes"):
        pattern = "RiRes*"
        full_path = os.path.join(dName, pattern)

        # Use glob to find files matching the pattern
        matched_files = glob.glob(full_path)
        extracted_file_names = []

        for file in matched_files:
            file_name = os.path.splitext(os.path.basename(file))[0][len("RiRes") :]
            extracted_file_names.append(file_name)

        extracted_set = set(extracted_file_names)
        all_set = set(all_file_names)

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
            fig = plt.figure(figsize=gl.figsize_panel)
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
            ax1 = fig.add_axes([0.15, 0.75, 0.75, 0.2])
            for v in vClampList:
                x = np.linspace(100, 240, 1000)
                holdingpotentials = self.pseudotrace(x, v)
                ax1.plot(x, holdingpotentials, color="grey", label=f"{v}")
            ax1.set_ylabel(gl.curr, color="grey")
            for spine in ax1.spines.values():
                spine.set_visible(False)
            ax1.tick_params(bottom=False, left=True, colors="grey")
            ax1.set_xticks([])

            ax2 = fig.add_axes([0.15, 0.15, 0.75, 0.6])

            for cells in results:
                for cell in cells:
                    ax2.plot(list(cell.time), list(cell.vSoma), color="black")

            ax2.set_xlabel(gl.ms)
            ax2.set_ylabel(gl.volt)
            ax1.set_xlim((100, 240))
            ax2.set_xlim((100, 240))
            fig.subplots_adjust(left=0.2, right=1, bottom=0.1, top=1)
            plt.tight_layout()
            plt.savefig(os.path.join("../results/paperRes", f"CurrentClampSoma.pdf"))

    def pseudotrace(self, x, v,bb=(130,220)):
        min_b,max_b = bb
        tmp = []
        for t in x:
            if t < min_b or t > max_b:
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
        plt.figure(figsize=gl.figsize_panel)
        if hasattr(cells,'cvode') and cells.cvode:
            plt.subplots_adjust(left=0.2)
            ny, nx = timeVoltageArray.shape
            y_edges = np.array(cells.time)[initStep:] - list(cells.time)[initStep]

            x_vals = np.arange(nx)          # linear x-axis

            X, Y = np.meshgrid(x_vals, y_edges)

            plt.pcolormesh(X, Y, timeVoltageArray, shading='auto',cmap='magma',edgecolor='none',linewidth=0,rasterized=True)
            plt.gca().invert_yaxis()
            plt.ylabel(gl.ms)
        else:
            plt.imshow(timeVoltageArray, cmap="magma", interpolation="none", aspect="auto")
        if replay:
            plt.colorbar(label=gl.d_volt_short, ticks=np.arange(0, 10, 2), extend="max")
            plt.clim((0, 10))
        else:
            plt.colorbar(label=gl.d_volt_short, ticks=np.arange(0, 20, 2), extend="max")
            plt.clim(gl.clim_volt)
        plt.xlabel(gl.free("Normalized distance"))
        plt.xticks(
            range(0, 11, 5), [0, 0.5, 1.0]
        )  # float point generated by np.linspace
        if hasattr(cells,'cvode') and cells.cvode:
            plt.text(
                0.1,
                100,
                "Soma",
                color="white",
                ha="left",
                va="top",
                fontsize=plt.rcParams["axes.labelsize"],
            )
            plt.text(
                9.9,
                100,
                "PAP",
                color="white",
                ha="right",
                va="top",
                fontsize=plt.rcParams["axes.labelsize"],
            )


        else:
            plt.text(
                0.1,
                len(list(cells.branchAtten[0])[initStep:]) + 2,
                "Soma",
                color="white",
                ha="left",
                va="bottom",
                fontsize=plt.rcParams["axes.labelsize"],
            )
            plt.text(
                9.9,
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
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"branchAtten_{self.tag}_Original.pdf",
                )
            )
        else:
            plt.tight_layout()
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
            plt.tight_layout()
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
            fig, axs = plt.subplots(
                2, 2, figsize=gl.figsize_panel, sharex=True, sharey=True
            )
        else:
            fig, axs = figObj

        splt_x, splt_y = id
        axs[splt_x, splt_y].plot(*data, label=label, **plt_args)

        if finalize and final_label:
            # plt.tight_layout(rect=[0.2, 0.0, 0.8, 0.9])
            right_edge= 0.70
            fig.subplots_adjust(left=0.2, right=right_edge, wspace=0.15, hspace=0.1)
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
                bbox_to_anchor=(right_edge+0.01, 0.5),
                fancybox=True,
                shadow=True,
                ncol=1,
            )
            leg._legend_box.align = "left"

            fig.text(
                (left + right) / 2,
                bottom - 0.06,
                x_label,
                ha="center",
                va="top",
                fontsize=plt.rcParams["axes.labelsize"],
            )

            fig.text(
                left - 0.12,
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
                left - 0.19,
                (axs[0, 0].get_position().y0 + axs[0, 0].get_position().y1) / 2,
                row1,
                ha="left",
                va="center",
                rotation=90,
                fontsize=plt.rcParams["axes.labelsize"],
            )
            fig.text(
                left - 0.19,
                (axs[1, 0].get_position().y0 + axs[1, 0].get_position().y1) / 2,
                row2,
                ha="left",
                va="center",
                rotation=90,
                fontsize=plt.rcParams["axes.labelsize"],
            )

        return fig, axs

    def kvPhasePlane(self):
        #self.duramplenPhase()
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
                            f"PAP Length\n0.3 {gl.unit_micron_raw}",
                            f"5 {gl.unit_micron_raw}",
                            f"Kir Channels {int(cell.PAPKirCount_std*self.KirMax+cell.PAPKirCount)}",
                            f"Kir Channels {int(cell.PAPKirCount_std*self.optKir+cell.PAPKirCount)}",
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
                            zorder=-1,
                            **kw_args,
                        )

            _, axs = figobj
            for ax in axs.flat:
                ax.set_xlim(gl.lim_ko)
                ax.set_ylim(gl.lim_Vmemb)
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
            NT_name = "GABA$_A$R"
        elif self.NKA:
            NTChannelComp = [self.optNKA, self.maxNKA]
            NT_name = "NKA"
        elif self.GluT:
            NTChannelComp = [0, 100]
            NT_name = "GluT"
        else:
            wMessage("No NT Channels Selected;skipped")
            return
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
                            if self.NKA:
                                funcArgs[-1]["nakpump"] = chanCount
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

                    if self.NKA:
                        for key in funcArgs[-1].keys():
                            if key in ["GABA", "Glu"]:
                                funcArgs[-1][key] = False

                    KoSteps = np.arange(2, gl.max_depo_ko + 1, 2)
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
                        "nakpump" in cell.GENEDict.keys()
                        and cell.GENEDict["nakpump"] > 1
                    ):
                        id += 1
                    elif (
                        cell.multiple == 0
                        and "GluTrans" in cell.GENEDict.keys()
                        and cell.GENEDict["GluTrans"] is not None
                        and cell.GENEDict["GluTrans"] > 0
                        and "nakpump" not in cell.GENEDict.keys()
                    ):
                        if type(NTChannelComp[0]) is int:
                            tmp_NTChannelComp = (
                                np.array(NTChannelComp) * cell.PAPGluTCount_std
                                + cell.PAPGluTCount
                            )
                            tmp_NTChannelComp = tmp_NTChannelComp.astype(int)
                            for i, val in enumerate(tmp_NTChannelComp):
                                ret_string = "{:.2e}".format(val)
                                a, b = ret_string.split("e")
                                b = int(b)
                                NTChannelComp[i] = f"{a} x10$^{b}$"
                        id += 1
                    elif (
                        cell.multiple != self.optNMDAR
                        and not hasattr(cell, "GABACount")
                        and cell.multiple != 0
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
                            f"Kir Channels {int(cell.PAPKirCount_std*self.KirMax+cell.PAPKirCount)}",
                            f"Kir Channels {int(cell.PAPKirCount_std*self.optKir+cell.PAPKirCount)}",
                        ]
                        if self.NKA:
                            kw_args["finalize"][0] = (
                                f"{NT_name} Current\n{NTChannelComp[0]} "
                                + gl.unit_curr_density_raw
                            )
                            kw_args["finalize"][1] = (
                                f"{NT_name} Current\n{NTChannelComp[1]} "
                                + gl.unit_curr_density_raw
                            )

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
                            linestyle="--",
                            color="black",
                            zorder=-1,
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
        plt.tight_layout()
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
                    label=gl.current_ion("NMDAR"),
                    color=self.returnColor("NMDAR"),
                )
                if hasattr(cell, "iGluT"):
                    plt.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluT)[initStep:],
                        label=gl.current_ion("GLT"),
                        color=self.returnColor("GluT"),
                    )
                plt.legend()
                plt.xlabel(gl.ms)
                plt.ylabel(gl.curr)
                plt.tight_layout()
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
        plt.tight_layout()
        plt.savefig(os.path.join("../results/paperRes", "ekDepolarcomp.pdf"))

        plt.cla()
        plt.clf()
        plt.scatter(koList, depList, color="black")
        plt.ylabel(gl.d_volt)
        plt.xlabel(gl.ion_o("K"))
        plt.tight_layout()
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
                            # plt.plot(cell.time, cell.vPAP)
                            # plt.tight_layout()
                            # plt.savefig("KO changes.pdf")
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
                    # self.plotIKSeries([[cell]])
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
            plt.cla()
            plt.clf()
            fig = plt.figure(figsize=gl.figsize_panel)
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
                c=self.get_papLen_color_from_value(self.PAPLen),
            )
            ax.axhline(
                val_means["spillover"][0],
                linestyle="--",
                c=self.get_papLen_color_from_value(self.peakLen),
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
            plt.tight_layout()
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
            cells.setK(KoSize=22, delay=0, dur=1)
            #cells.setSlowing(float('inf'))
            cells.run()
            cells = cells.copyAttr()
            AllCells = [[cells]]
            self.free_figure(AllCells)
        else:
            AllCells = self.free_read_data()
            cells = AllCells[0][0]
        if rank == 0:
            initStep = self.get_initStep(cells,shift=-0.1)
            flux = np.array(list(cells.flux)[initStep:])
            kbath = np.array(list(cells.kbath)[initStep:]) * -1
            kbath[kbath == 0] = np.nan

            fig = plt.figure(figsize=gl.figsize_panel_long)
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 3], hspace=0.2)
            ax_ko = fig.add_subplot(gs[0])
            ax_ko.plot(list(cells.time)[initStep:], list(cells.KoPAP)[initStep:], color="gray")
            ax_ko.axhline(3.5,color='gray',linestyle='--',lw=1)
            ax_ko.set_ylabel(gl.ion_o("K",short=True),color="grey")
            for spine in ax_ko.spines.values():
                spine.set_visible(False)
            ax_ko.tick_params(bottom=False, left=True, colors="grey")
            ax_ko.set_xticks([])


            ax = fig.add_subplot(gs[1])
            #print(flux,kbath)
            ax.plot(list(cells.time)[initStep:], np.divide(kbath, flux+kbath),color='black')
            ax.set_xlabel(gl.ms)
            max_t = max(list(cells.time))
            min_t = list(cells.time)[initStep]
            ax.set_xlim(left=min_t, right=max_t)
            ax_ko.set_xlim(left=min_t,right=max_t)
            ax.set_ylabel(gl.free(f"Ratio of\ndiffusional loss / {gl.delta_ion_o('K',short=True)}"))

            left = ax_ko.get_position().x0
            bottom  = ax_ko.get_position().y0
            fig.text(
                left + 0.01,
                bottom + 0.01,
                "baseline",
                fontsize=plt.rcParams["axes.labelsize"],
                color='gray',
                ha="left",
                va="bottom",
            )

            plt.savefig(os.path.join("../results/paperRes", "fluxRatioOvertime.pdf"))

    def singleRun(
        self, *args, expOverlay=False, GluTime=False, nearSoma=False,
    ):
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
        if hasattr(self,'kdifl') and self.kdifl:
            funcArgs[-1]['nakpump'] = self.OEpump
            funcArgs[-1]['dt'] = self.dt/5
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

            if hasattr(self,'kdifl') and self.kdifl:
                cells.set_diff_ki(True)
                cells.durStim = 5
            if nearSoma:
                cells.setPAPNearSoma()
            #        cells.setTstop(500)
            cells.initialize()

            if self.GluStim:
                cells.setNMDA_Mgblock(k, d, s)
                cells.setNMDA_TC(tau1, tau2)
            # cells.setSlowing(slow)

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
            self.plotIKSeries(
                AllCells, panelF=True, setKoylim=setKoylim, setekylim=not nearSoma
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
                    plt.xlabel(gl.ms)
                    plt.ylabel(gl.ion_o("Glu"))
                    plt.plot(list(cells.time), list(cells.GluTGlu))
                    plt.tight_layout()
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
                plt.tight_layout()
                plt.savefig(
                    os.path.join("../results/paperRes", f"GluTstates{self.tag}.pdf")
                )

    def bathExperiment(self, runAll=True, invivo=False, isolate=False, gaba=False,soma=False):
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
                    if bool_isolate and bool_invivo and rank > allConds:
                        self.bathExperiment(
                            runAll=False,
                            invivo=True,
                            isolate=True,
                            soma=True
                        )
            if rank == allConds:
                self.bathExperiment(runAll=False, gaba=True)  # for escaping inf loop

        else:
            # print(f"{gaba=}{invivo=}{isolate=}{rank=}")
            if gaba:
                self.gababathExperiment()
            else:
                self.kbathExperiment(invivo, isolate,soma=soma)

    def kbathExperiment(self, invivo, isolate,soma=False):
        # add multispike ek clamp
        self.addChannelTag()
        if invivo:
            self.tag += "_invivoBath"
        else:
            self.tag += "_Bath"
        if isolate:
            self.locality = "local"
            self.tag += "_isolated"
            if soma and invivo:
                self.tag += '_soma'
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

        if invivo:
            funcArgs[-1]['dt'] = 1000

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
                if soma:
                    cells.setPAP2Soma()
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
        self.free_figure(AllCells)
        if invivo:
            for cells in AllCells:
                for cell in cells:
                    print(max(list(cell.vPAP)),max(list(cell.vSoma)))
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
                "kir2": self.optKir,
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
        ] = self.optGABAR/3 # avg num of sections in a PAP  

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
            self.channelCompareMax *= 2.6 
            self.channelCompareStep *= 2.6 
        elif self.GAP:
            self.channelCompareMax /= 2
            self.channelCompareStep /= 2
        elif self.NKA:
            self.channelCompareMax = self.maxNKA
            self.channelCompareStep = 1
        if not (self.GABAR or self.NMDAR) and self.GluT:
            self.channelCompareMax = 4
            self.channelCompareStep = int(self.channelCompareMax / 2)
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
            funcArgs[-1]["dt"] /= 50 
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
            plt.figure(figsize=gl.figsize_panel)
            plt.imshow(
                imArray,
                cmap=cmap,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            maxv = 30
            plt.colorbar(
                label=gl.d_volt_short, ticks=np.arange(0, maxv, 2), extend="max"
            )
            plt.clim((5, maxv))
            plt.xlabel(gl.chan_num("Cx43"))
            plt.ylabel(gl.free(f"PAP distance from GJ {gl.unit_micron}"))
            plt.xticks(
                range(len(gapCounts)),
                gapCounts,
            )
            plt.yticks(range(len(shift_range)), [f"{x:.2f}" for x in shift_range])

            plt.tight_layout()
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
            fig = plt.figure(figsize=gl.figsize_panel_long)
            gs = fig.add_gridspec(nrows=4, ncols=2, hspace=0.3)
            ax = fig.add_subplot(gs[0:2, 0])
            ax_inset = fig.add_subplot(gs[3, 0])
            ax_peak = fig.add_subplot(gs[2:, 1])
            ax_cable = fig.add_subplot(gs[0,1],sharex=ax_peak)


            self.get_papLen_color_from_value(0)
            length = self.peakLen 
            height = length/5 
            r = height / 2 

            x = np.linspace(0,self.peakLen,1000).reshape(1,-1)

            ax_cable.imshow(
                x,
                extent=[0, length, -height/2, height/2],
                aspect='auto',
                cmap=self.paplen_cm,
                origin='lower',
                norm=self.paplen_norm,
            )

            ax_cable.plot([0, length], [r, r], color='black', lw=1.5)
            ax_cable.plot([0, length], [-r, -r], color='black', lw=1.5)
            right_color = self.paplen_cm(1.0)
            right_cap = Wedge((length, 0), r, -90, 90,
                            facecolor=right_color, edgecolor='black', lw=1.5)

            ax_cable.plot([0, length], [r, r], color='black', lw=1.5)
            ax_cable.plot([0, length], [-r, -r], color='black', lw=1.5)

            right_arc = Arc((length, 0), height, height, angle=0,
                            theta1=-90, theta2=90, lw=1.5, color='black')

            ax_cable.add_patch(right_cap)
            ax_cable.add_patch(right_arc)
            rect = Rectangle(
                (0, -height/2),
                length,
                height,
                linewidth=1.5,
                edgecolor='black',
                facecolor='none'
            )
            ax_cable.add_patch(rect)

            ax_cable.set_yticks([])
            ax_cable.set_xticks(np.linspace(0, length, 5))

            for spine in ax_cable.spines.values():
                spine.set_visible(False)

            ax_cable.set_xscale('log')

            for cells in results:
                for cell in cells:
                    color = self.get_papLen_color_from_value(cell.PAPLen)
                    initStep = self.get_initStep(cell)
                    ax.plot(
                        np.array(list(cell.time)[initStep:]),  # ms to s
                        np.array(list(cell.vPAP)[initStep:]) - cell.RMP,
                        color=color,
                    )
                    ax_inset.plot(
                        np.array(list(cell.time)[initStep:]),  # ms
                        np.array(list(cell.vPAP)[initStep:]) - cell.RMP,
                        color=color,
                    )
                    i = np.where(cell.PAPLen == iterations)[0][0]
                    vListarray[cell.seed][i] = (
                        max(list(cell.vPAP)[initStep:]) - cell.RMP
                    )
            vListarray = vListarray.T
            controlIndex = np.where(self.PAPLen == iterations)[0][
                0
            ]  # get index of PAPLen position in iterations
            controlV = np.nansum(vListarray[controlIndex]) / sampleNum
            ax.set_xlabel(gl.ms)
            ax.set_ylabel(gl.d_volt_short)
            x0 = 145
            x1 = 155
            y0 = -0.01
            y1 = 3.01
            ax.set_ylim((y0, y1))

            ax_inset.set_xlim((x0, x1))
            ax_inset.set_ylim((y0, y1))
            ax_inset.set_xlabel(gl.ms)
            ax_inset.set_ylabel(gl.d_volt_short)

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
                connectionstyle="angle,angleA=110,angleB=-5,rad=20",
            )

            fig.add_artist(con1)
            fig.add_artist(con2)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
            ax_inset.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))

            ax = ax_peak
            vList = [
                np.nansum(vListarray[i]) / sampleNum for i in range(len(vListarray))
            ]
            vstdList = [np.nanstd(vListarray[i]) for i in range(len(vListarray))]
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
                edgecolor="black",
                linewidth=1,
                c=[self.get_papLen_color_from_value(i) for i in iterations],
            )
            # plot control as diamond
            if controlIndex != None:
                ax.scatter(
                    self.PAPLen,
                    controlV,
                    color=self.get_papLen_color_from_value(self.PAPLen),
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
            # ax.scatter(
            #    iterations[maxIndex],
            #    vList[maxIndex],
            #    color=cm.winter(maxIndex / len(iterations)),
            #    label="Spillover",
            #    zorder=-1,
            # )
            #if self.GluStim and self.KStim:
                #if self.peakLen is not None:
                #    ax.axvline(
                #        self.PAPLen,
                #        ymax=vList[0] / maxY,
                #        linestyle="--",
                #        color=cm.winter(maxIndex / len(iterations)),
                #        zorder=-2,
                #    )
            ax.legend()
            ax.set_ylim((y0, y1))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0,)))
            
            xmax = x.max() 
            ticks = ax.get_xticks()
            ticks = ticks[ticks <= xmax]
            ticks = np.append(ticks, xmax)

            ax.set_xticks(ticks)

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

            ax.set_xlim(left=0.2,right=xmax)

            ax.set_xlabel(gl.pap_affect)
            ax.set_ylabel(gl.d_volt_short)

            left = ax_cable.get_position().x0
            bottom = ax_cable.get_position().y0
            fig.text(
                left + 0.01,
                bottom - 0.1,
                f'Length Color Code {gl.unit_micron_bold}',
                fontsize=plt.rcParams["axes.labelsize"],
                color='black',
                ha="left",
                va="top",
                fontweight="bold",
            )


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


    def plot_seed_map(self,start,end,skip_seed=None,save=False,no_gap=False):
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
                "multiple": None,
            }
        )
        if no_gap:
            funcArgs[-1]['gapCount'] = 0

        ccList = ['seed']
        if skip_seed is not None:
            iterations = [ (i) for i in np.arange(start,end+1) if i not in skip_seed]
        else:
            iterations = [ (i) for i in np.arange(start,end+1) ]


        # remove overleapping morphology memory
        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            [["get_PAPName","savePAPProp","calcPAPRi"]],
            [[{},{},{'cleanMorph':True,'direct':save}]],
            randomize=False,
        )
        tmp_skip_cell = []
        if self.pb_seed_max > 0:
            pb_seeds = list(range(self.pb_seed_max))
        else:
            pb_seeds = [self.seed]

        if save:
            if rank == 0 and size > 1:
                funcArgs[-1]['seed'] = self.seed
                tmpCell = PAPModel(**funcArgs[-1])
                tmpCell.setPAPNearSoma(onSoma=True)
                tmpCell.get_PAPName()
                tmpCell.savePAPProp()
                tmpCell.calcPAPRi(direct=save)
                interm_results = [tmpCell.copyAttr()] 

            else:
                for remove_seed in self.skipPB:
                    pb_seeds.remove(remove_seed) 

                if len(pb_seeds) <= size - 1:
                    if rank < len(pb_seeds) + 1:
                        rank_seeds = [pb_seeds[rank - 1]]
                    else:
                        rank_seeds = []
                        interm_results = []

                else:
                    job_per_thread = len(pb_seeds) // size
                    remainder = len(pb_seeds) % size
                    rank_seeds = []
                    rank_seeds += pb_seeds[1+job_per_thread * rank: 1+job_per_thread* (rank + 1)]
                    if remainder > 0:
                        remainder_seeds = pb_seeds[job_per_thread*size :]
                        if rank + 1 < len(remainder_seeds):
                            rank_seeds.append(remainder_seeds[rank - 1])
                
                for job_seed in rank_seeds:
                    funcArgs[-1]['seed'] = job_seed 
                    tmpCell = PAPModel(**funcArgs[-1])
                    tmpCell.setPAPNearSoma(onSoma=False)
                    tmpCell.get_PAPName()
                    tmpCell.savePAPProp()
                    tmpCell.calcPAPRi(direct=save)
                    interm_results = [tmpCell.copyAttr()] 
            
            interm_results = comm.gather(interm_results, root=0)

            if rank == 0:
                results += interm_results
                self.free_figure(results)

        if rank == 0:

            col_dict = {0: self.returnColor('local'), 1: self.returnColor('Primary Branch'), 2: self.returnColor("Soma"), 3: self.returnColor('PAP')}
            colors = [col_dict[key] for key in sorted(col_dict.keys())]

            ref_names = []
            for cells in results:
                for cell in cells:
                    if cell.PAP_name not in ref_names:
                        ref_names.append(cell.PAP_name)
                    elif 'dendrite' not in cell.PAP_name:
                        tmp_skip_cell.append(cell.seed)
                    else:
                        self.skipPB.append(cell.seed)

            if save:
                plot_3d_morphology(
                    rangevar='PAP',
                    color_names = [cell.PAP_name for cells in results for cell in cells],
                    colormap_name = colors,
                    add_colorbar=False,
                    clim=(0,3)
                )
                plt.savefig(
                    os.path.join(
                        '../morphResults/',
                        'Plot_PAPs.pdf'
                    )
                )
                plt.close('all')
        tmp_skip_cell = comm.bcast(tmp_skip_cell,root=0)
        return tmp_skip_cell


    def compareIKSizes2KoSizes(self):
        count = 0
        PAP_AllProperties = None 
        replace_props = ['PAPLen','KoSize']
        while PAP_AllProperties is None and 'seed' not in self.tag:
            if count > len(replace_props):
                eMessage('Could not find pap_props')
            self.tag = self.tag.replace(replace_props[count],"seed_point_stim")
            PAP_AllProperties = self.find_pap_props()
            count += 1

        
        PAP_AllProperties.release()

        self.compareIKSize()
        self.compareIKSize_pbSoma()
        self.plot_IKSizes()


    def plot_IKSizes(self,point_stim=True):
        IKSize_PAP = self.find_run_comp('compareIKSize')
        IKSize_Soma = self.find_run_comp('compareIKSize_pbSoma')
        ampLen = self.find_run_comp('runAmpLenComparison')
        PAP_AllProperties = self.find_pap_props()
        if rank == 0:
            #plt.cla()
            #plt.clf()
           #nameList = {}
            #for i,data in enumerate([IKSize_PAP,IKSize_Soma]):
            #    for cells in data:
            #        for cell in cells:
            #            plt.plot(list(cell.time),list(cell.vPAP))
            #plt.savefig('traces.pdf')
            #plt.cla()
            #plt.clf()
            #
            #for i,data in enumerate([IKSize_PAP,IKSize_Soma]):
            #    if i == 0:
            #        color = self.returnColor('PAP')
            #    else:
            #        color = self.returnColor('Primary Branch')
            #    
            #    for cells in data:
            #        for cell in cells:
            #            if hasattr(cell,'PAP_name') and cell.PAP_name == 'soma':
            #                color = self.returnColor('Soma')

            #            if not hasattr(cell,'fit_maxResponse'):
            #                initStep = self.get_initStep(cell)
            #                data = -(np.array(list(cell.vPAP)[initStep:]) - cell.RMP)
            #                peaks,_ = find_peaks(data)
            #                last_peak_index = peaks[-1]
            #                last_peak_value = list(cell.vPAP)[last_peak_index]
            #                plt.scatter(cell.KoSize,last_peak_value-cell.RMP,color=color)
            #            else:
            #                plt.scatter(cell.KoSize,cell.fit_maxResponse-cell.fit_minResponse,color=color)

            #plt.ylabel(gl.d_volt)
            #plt.xlabel(gl.delta_ion_o('K'))
            #plt.savefig(
            #    os.path.join(
            #        "../results/paperRes",
            #        f"KoSize_depo{self.tag}.pdf",
            #    )
            #)

            #plt.cla()
            #plt.clf()
            #for cells in ampLen:
            #    for cell in cells:
            #        def find_seed(seed):
            #            for refs in IKSize_PAP:
            #                for ref in refs:
            #                    if ref.seed == seed:
            #                        return ref
            #            return None 

            #        iksize = find_seed(cell.seed)
            #        if iksize is not None and cell.KoSize == self.KoCompMax:
            #            plt.scatter(iksize.fit_maxResponse-iksize.fit_minResponse,max(list(cell.vPAP))-cell.RMP,color=self.returnColor('PAP'))
            #res_column_soma = self.read_run_comp("run_comp_soma")
           

            ## only specific use case
            #def find_by_name(name):
            #    save_name = []
            #    for refs in PAP_AllProperties:
            #        for ref in refs:
            #            if name in ref.PAP_name:
            #                save_name.append(ref)

            #    if len(save_name) > 0:
            #        return save_name
            #    else:
            #        return None 

            #ri_pb_seed = [ri.seed for ri in find_by_name('dendrite')]
            #
            #for j in ri_pb_seed:
            #    def getbySeed(cell):
            #        if cell.seed == j:
            #            return max(cell.vPAP) - cell.RMP
            #        else:
            #            return None
            #  
            #    res_column_pb = self.read_run_comp("run_comp_pb",func=getbySeed)
            #    def find_seed():
            #        for refs in IKSize_Soma:
            #            for ref in refs:
            #                if ref.seed == j:
            #                    return ref
            #        return None 

            #    iksize = find_seed()
            #    if iksize is not None:
            #        plt.scatter(iksize.fit_maxResponse-iksize.fit_minResponse,res_column_pb[-1],color=self.returnColor('Primary Branch'))

            #for refs in IKSize_Soma:
            #    for ref in refs:
            #        if ref.PAP_name == 'soma':
            #            plt.scatter(iksize.fit_maxResponse-iksize.fit_minResponse,res_column_soma[-1],color=self.returnColor('Soma'))

            #plt.savefig(
            #    os.path.join(
            #        "../results/paperRes",
            #        f"Response_potassiumRequirement{self.tag}.pdf",
            #    )
            #)


            plt.cla()
            plt.clf()
            plt.figure(figsize=gl.figsize_panel)
            plt.subplots_adjust(left=0.2,bottom=0.2)
            def find_by_name(name):
                save_name = []
                for refs in PAP_AllProperties:
                    for ref in refs:
                        if name in ref.PAP_name:
                            save_name.append(ref)

                if len(save_name) > 0:
                    return save_name
                else:
                    return None 

            for i,data in enumerate([IKSize_PAP,IKSize_Soma]):
                if i == 0:
                    color = self.returnColor('PAP')
                else:
                    color = self.returnColor('Primary Branch')
                
                for cells in data:
                    for cell in cells:
                        if hasattr(cell,'PAP_name') and cell.PAP_name == 'soma':
                            color = self.returnColor('Soma')
                        def findbyNameSeed(comp_cell):
                            for refs in PAP_AllProperties:
                                for ref in refs:
                                    if not hasattr(comp_cell,'PAP_name'):
                                        if hasattr(ref,'PAP_name') and 'Glia' in ref.PAP_name and ref.seed == comp_cell.seed:
                                            return ref
                                    else:
                                        if hasattr(ref,'PAP_name')  and ref.PAP_name == comp_cell.PAP_name:
                                            return ref
                            return None

                        prp_cell = findbyNameSeed(cell)
                        if prp_cell is not None:
                            if hasattr(cell,'fit_maxResponse'):
                                plt.scatter(cell.fit_maxResponse-cell.fit_minResponse,prp_cell.PAP_Ri,color=color)
                            else:
                                plt.scatter((list(cell.vPAP)[-1]- cell.RMP),prp_cell.PAP_Ri,color=color)

            plt.ylabel(gl.ri)
            plt.xlabel(gl.d_volt)
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"KoSize_Ri{self.tag}.pdf",
                )
            )

            plt.cla()
            plt.clf()
            x =[]
            y=[]
            Gms=[]
            Gas=[]
            compare_ko = 10 
            index_ko = int(compare_ko/self.KoCompStep - 1)

            find_label = []
            for cells in ampLen:
                for cell in cells:
                    def find_seed(seed):
                        for refs in IKSize_PAP:
                            for ref in refs:
                                if ref.seed == seed:
                                    return ref
                        return None 

                    iksize = find_seed(cell.seed)
                    prp_cell = findbyNameSeed(cell)
                    if iksize is not None and cell.KoSize == compare_ko:
                        Rm = 26.7*np.log((iksize.KoSize+iksize.Ko)/iksize.Ko)/0.05
                        Ri =prp_cell.PAP_Ri
                        Ra = 1/(1/Ri-1/Rm)
                        Gm = 1/Rm
                        Ga = 1/Ra
                        a = prp_cell.PAP_properties[-1]['diam']/2 * 1e-4
                        L = prp_cell.PAP_properties[-1]['L']/prp_cell.PAP_properties[-1]['nseg']*1e-4
                        val = np.sqrt(Ga/Gm * L*2/a)


                        plt.scatter(val ,max(list(cell.vPAP))-cell.RMP,color=self.returnColor('PAP'))
                        if prp_cell.PAP_name in find_label:
                            plt.annotate(
                                prp_cell.PAP_name,
                                (val,max(list(cell.vPAP))-cell.RMP),
                                xytext=(1.2*val,1.5*(max(list(cell.vPAP))-cell.RMP)),
                                arrowprops=dict(arrowstyle='-')
                            )

                        x.append(val)
                        y.append(max(list(cell.vPAP))-cell.RMP)
                        Gms.append(Gm)
                        Gas.append(Ga)

            def abs_vol(cell):
                return max(list(cell.vPAP))

            res_column_soma = self.read_run_comp("run_comp_soma")
           

            # only specific use case
            def find_by_name(name):
                save_name = []
                for refs in PAP_AllProperties:
                    for ref in refs:
                        if name in ref.PAP_name:
                            save_name.append(ref)

                if len(save_name) > 0:
                    return save_name
                else:
                    return None 

            ri_pb_seed = [ri.seed for ri in find_by_name('dendrite')]
            
            for j in ri_pb_seed:
                def getbySeed(cell):
                    if cell.seed == j:
                        return max(list(cell.vPAP)) - cell.RMP
                    else:
                        return None
                def findbyNameSeed(comp_cell_seed,name):
                    for refs in PAP_AllProperties:
                        for ref in refs:
                            if hasattr(ref,'PAP_name') and name in ref.PAP_name and ref.seed == comp_cell_seed:
                                return ref
                    return None

                prp_cell = findbyNameSeed(j,'dendrite')
              
                res_column_pb = self.read_run_comp("run_comp_pb",func=getbySeed)
                def find_seed():
                    for refs in IKSize_Soma:
                        for ref in refs:
                            if ref.seed == j:
                                return ref
                    return None 

                iksize = find_seed()
                if iksize is not None:
                    Rm = 26.7*np.log((iksize.KoSize+iksize.Ko)/iksize.Ko)/0.05
                    Ri =prp_cell.PAP_Ri
                    Ra = 1/(1/Ri-1/Rm)
                    Gm = 1/Rm
                    Ga = 1/Ra
                    a = prp_cell.PAP_properties[-1]['diam']/2 * 1e-4
                    L = prp_cell.PAP_properties[-1]['L']/prp_cell.PAP_properties[-1]['nseg']*1e-4
                    val = np.sqrt(Ga/Gm * L*2/a)
                    Gms.append(Gm)
                    Gas.append(Ga)



                    plt.scatter(val,res_column_pb[index_ko],color=self.returnColor('Primary Branch'))
                    if prp_cell.PAP_name in find_label:
                        plt.annotate(
                            prp_cell.PAP_name,
                            (val,res_column_pb[index_ko]),
                            xytext=(0.8*val,res_column_pb[index_ko]*0.4),
                            arrowprops=dict(arrowstyle='-'),
                            ha='right',
                            va='center',
                        )

                    x.append(val)
                    y.append(res_column_pb[index_ko])
 
            for refs in IKSize_Soma:
                for ref in refs:
                    if ref.PAP_name == 'soma':
                        prp_cell = findbyNameSeed(ref.seed,'soma')
                        Rm = 26.7*np.log((ref.KoSize+ref.Ko)/ref.Ko)/0.05
                        Ri =prp_cell.PAP_Ri
                        Ra = 1/(1/Ri-1/Rm)
                        Gm = 1/Rm
                        Ga = 1/Ra
                        a = prp_cell.PAP_properties[-1]['diam']/2 * 1e-4
                        L = prp_cell.PAP_properties[-1]['L']/prp_cell.PAP_properties[-1]['nseg']*1e-4
                        val = np.sqrt(Ga/Gm * L*2/a)
                        Gms.append(Gm)
                        Gas.append(Ga)


                        plt.scatter(val,res_column_soma[index_ko],color=self.returnColor('Soma'))
                        x.append(val)
                        y.append(res_column_pb[index_ko])
    


            correlation_coefficient, p_value = pearsonr(x,y)
            if p_value < 0.05:
                plt.title(f'Total={p_value:.2e} R$^2$={correlation_coefficient**2:.1e}')
                res = linregress(x,y)
                x = sorted(x)
                plt.plot(x,res.slope*np.array(x) + res.intercept,linestyle='--',color=self.returnColor('local'),zorder=-1)


            plt.ylabel(gl.d_volt)
            plt.ylim(gl.clim_volt)
            plt.xlabel(r'$\Gamma _K$')
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"KoSize_Ra{self.tag}.pdf",
                )
            )

            plt.cla()
            plt.clf()
            plt.scatter(Gms,Gas)
            plt.xscale('log')
            plt.savefig('Gm_Ga.pdf')




        IKSize_PAP.release()
        IKSize_Soma.release()
        ampLen.release()
        PAP_AllProperties.release()




  
    @read_data
    def compareIKSize(self):
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
        ccList = ["seed"]
        PAP_AllProperties = self.find_pap_props()
        replace_props = ['KoSize','PAPLen']
        count = 0
        while PAP_AllProperties is None:
            if count > len(replace_props):
                eMessage('Could not find pap_props')
            self.tag = self.tag.replace(replace_props[count],"seed")
            PAP_AllProperties = self.find_pap_props()
            count += 1

        iterations = [cell.seed for cells in PAP_AllProperties for cell in cells if 'Glia' in cell.PAP_name]
        run_func = [["initialize", "fitIKSize"]]
        run_func_args = [[{},{}]]


        if hasattr(self,'kdifl') and self.kdifl:
            run_func[0] = ['set_kdfl_iter'] + run_func[0]
            run_func_args[0] = [{}] + run_func_args[0]  
            funcArgs[-1]['dt'] = self.dt/5
            funcArgs[-1]['nakpump'] = self.OEpump


        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            run_func,
            run_func_args,
        )
        self.free_figure(results)
        PAP_AllProperties.release()

    @read_data
    def compareIKSize_pbSoma(self):
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
        ccList = ["seed"]
        PAP_AllProperties = self.find_pap_props()
        replace = ['KoSize','PAPLen']
        count = 0
        while PAP_AllProperties is None:
            if count > len(replace):
                eMessage('Could not find pap_props')
            self.tag = self.tag.replace(replace[count],"seed")
            PAP_AllProperties = self.find_pap_props()
            count += 1

        iterations = [(cell.seed) for cells in PAP_AllProperties for cell in cells if 'dendrite' in cell.PAP_name]
        run_func = [["setPAPNearSoma","get_PAPName","initialize","fitIKSize"]]
        run_func_args = [[{},{},{},{}]]

        if hasattr(self,'kdifl') and self.kdifl:
            run_func[0] = [run_func[0][0]] +['set_kdfl_iter'] + run_func[0][1:]
            run_func_args[0] = [run_func_args[0][0]] + [{}] + run_func_args[0][1:] 
            funcArgs[-1]['dt'] = self.dt/5
            funcArgs[-1]['nakpump'] = self.OEpump


        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            run_func,
            run_func_args,
        )
        if self.free_read_data() is None:
            if rank == 0:
                mprint('Generating Soma Data')
                funcArgs[-1]['seed'] = self.seed
                tmpCell = PAPModel(**funcArgs[-1])
                tmpCell.setPAPNearSoma(onSoma=True)
                tmpCell.get_PAPName()
                tmpCell.initialize(force_print_progress=True)
                tmpCell.fitIKSize()
                results += [[tmpCell.copyAttr()]] 
                self.free_figure(results)
        else:
            PAP_AllProperties.release()

        

    def potassiumComparison(self,nearSoma=False):
        self.KoCompMax = gl.max_ko
        self.KoCompStep = 5 
        for comparison in ["seed", "PAPLen"]: #, "KoSize"]: #,  "durStim"]:
            if comparison == "KoSize":
                compMax = self.KoCompMax
                compStep = self.KoCompStep
                startb = 0
            elif comparison == "PAPLen":
                # update when you want to extend
                compMax = self.PAPLenMax 
                compStep = self.PAPLen 
                startb = self.PAPLen 
                comp_prev = self.PAPLen 
            elif comparison == "durStim":
                compMax = 9
                compStep = 1
                startb = 1
            elif comparison == "seed":
                compMax = 26
                if self.kdifl:
                    compMax = 6
                compStep = 1
                startb = 1


            if comparison != "KoSize":
                if comparison == "PAPLen":
                    # Think of a way to extend from previous
                    #AllCells = self.find_run_comp('runAmpLen')
                    logx = np.logspace(log(comp_prev,compStep), log(compMax,compStep), base=compStep, num=10)
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
                self.skipPB = [3,4,5]
                if comparison in ['seed','PAPLen']:
                    iter_bath = [i for i in range(0,self.KoCompMax + 1,self.KoCompStep)]
                   #self.test_soma_response(iter_bath)
                    #iter_seed = [i for i in range(startb,compMax + 1,compStep)]
                    #self.run_comp_ri(iter_seed)

                    if self.GAP and '_no_gap' not in self.tag:
                        tmptag = self.tag
                        self.tag += '_no_gap'

                    self.rect_off = False
                    if self.rect_off:
                        tmp_tag = self.tag
                        self.tag += 'rect_off'

                    point_stim = False
                    if comparison == 'seed':
                        point_stim = True
                        if point_stim:
                            self.tag += '_point_stim'
                    self.run_comp_bath(iter_bath)
                    self.run_comp_soma(iter_bath,point_stim=point_stim)
                    self.run_comp_pb(iter_bath,point_stim=point_stim)


                if comparison == 'seed':
                    all_skip_seed = []
                    PAP_AllProperties = self.find_pap_props()
                    self.pap_win = True
                    if PAP_AllProperties is None:
                        skip_seed = self.plot_seed_map(startb,compMax)
                        tmp_compMax = compMax
                        all_skip_seed += skip_seed
                        while tmp_compMax - len(all_skip_seed) < (compMax-startb):
                            tmp_compMax += len(skip_seed) * compStep
                            skip_seed = self.plot_seed_map(startb,tmp_compMax,skip_seed=all_skip_seed)
                            all_skip_seed += skip_seed
                            mprint(f'Found total uniques seeds;{tmp_compMax - len(all_skip_seed)}')
                        mprint(f'Redundant seed;{all_skip_seed}')
                        logx = [
                            j
                            for j in range(startb,tmp_compMax+1,compStep)
                            if j not in all_skip_seed
                        ]

                        self.plot_seed_map(startb,tmp_compMax,skip_seed=all_skip_seed,save=True,no_gap=self.GAP)

                        # name for keyword argument when you cannot automatically sequentially inform the values
                    else:
                        logx = [cell.seed for cells in PAP_AllProperties for cell in cells if 'Glia' in cell.PAP_name]
                        logx = list(set(logx))
                        if len(logx) > compMax:
                            logx = logx[:compMax]
                            self.cutoff_PAP_properties = compMax - 1
                        names = [cell.PAP_name for cells in PAP_AllProperties for cell in cells]
                        nv = NeuronMorphologyVisualizer(
                            '../morphResults',
                        )
                        nv.load_morphology('Geometry/GeometryAstrocyteCA1.hoc')
                        for name in names:
                            nv.plot_local(name)




                    # change to np array for later
                    logx = np.array(logx)

                    iterations = [
                        (i,j)
                        for i in range(0,self.KoCompMax+1,self.KoCompStep)
                        for j in logx 
                    ]

                    if hasattr(self,'pap_win') and self.pap_win and PAP_AllProperties is not None:
                        PAP_AllProperties.release()

                    self.run_comp_pb(iter_bath,point_stim=point_stim,sec_range=True)
                    self.plot_range_pb()

 
                self.runAmpLenComparison(
                    comparison, iterations, compMax, compStep, logx=logx,point_stim=point_stim
                )
                if comparison == "seed" and self.rect_off:
                    self.rect_off = False
                    self.tag = tmp_tag


            # Calculate the number of iterations for all parm sets
            iterations = comm.bcast(
                get_iter(
                    self.optKir,
                    self.KirStep,
                    compMax,
                    compStep,
                    starta=-self.KirMax,
                    startb=startb,
                ),
                root=0,
            )
            # # Adjust the range for the last process
            if comparison == "KoSize":
                self.addChannelTag()
                self.tag += f"_{comparison}"
                self.runPotassiumComparison(
                    comparison, iterations, maxStep=compMax, intermStep=compStep
                )


    def run_comp_bath(self,iterations):
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
        ccList = ["KoSize"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        run_func = [["initialize", "setKBath_iter"]]
        run_func_args = [[{'force_print_progress':True},{}]]
        if hasattr(self,'kdifl') and self.kdifl:
            run_func[0] = ['set_kdfl_iter'] + run_func[0]
            run_func_args[0] = [{}] + run_func_args[0] 
            funcArgs[-1]['dt'] = self.dt/5
            funcArgs[-1]['nakpump'] = self.OEpump

        if hasattr(self,'rect_off') and self.rect_off:
            run_func[0] = ['kir_rect_off'] + run_func[0]
            run_func_args[0] = [{}] + run_func_args[0]  

        AllCells = self.find_run_comp("run_comp_bath")
        if AllCells is None:
            results = parallizeFor(
                iterations,
                [PAPModel],
                funcArgs,
                ccList,
                run_func,
                run_func_args,
            )
            self.free_figure(results)
            AllCells = results
        else:
            AllCells.release()

    def run_comp_ri(self,iterations):
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
                "KoSize":self.KoCompMax
            }
        )
        ccList = ["seed"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        AllCells = self.find_run_comp("run_comp_ri")
        if AllCells is None:
            results = parallizeFor(
                iterations,
                [PAPModel],
                funcArgs,
                ccList,
                [["calcPAPRi","calcPAPRi"]],
                [[{}, {'all':True} ]],
            )
            self.free_figure(results)
            AllCells = results
        else:
            AllCells.release()


        if rank == 0:
            for cells in AllCells:
                for cell in cells:
                    plt.scatter(cell.PAP_Ri[0],sum(cell.PAP_Ri[1:])/len(cell.PAP_Ri[1:]),color='black')
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    "comp_ri.pdf"
                )
            )




    def test_soma_response(self,iterations,mode='branch'):
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
        ccList = ["KoSize"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        run_func = [["setPAP2Soma","initialize", "setK", "run"]]
        if mode == 'branch':
            run_func[0] = ['set_pb'] + run_func[0]
            iterations = [ (i,j) for i in iterations for j in np.arange(0.1,5.1,0.5)]
            ccList.append('pbLen')

        AllCells = self.find_run_comp("test_soma_response")
        if AllCells is None:
            results = parallizeFor(
                iterations,
                [PAPModel],
                funcArgs,
                ccList,
                run_func,
                [[{},{},{"force_print_progress":True}, {"dur":50}, {}]],
            )
            self.free_figure(results)
            AllCells = self.find_run_comp("test_soma_response")
        if rank == 0:
            plt.figure(figsize=gl.figsize_panel)
            plt.subplots_adjust(left=0.2)
            for cells in AllCells:
                for cell in cells:
                    plt.scatter(cell.pbLen,max(list(cell.vPAP)),color='black')

            plt.xlabel(f'Primary branch diameter {gl.unit_micron}')
            plt.ylabel(gl.volt)
            plt.ylim(gl.lim_ek_zoom)
            plt.savefig(
                os.path.join(
                    '../results/paperRes',
                    'pblen_vPAP.pdf'
                )
            )
        AllCells.release()


    def run_comp_soma(self,iterations,point_stim=False):
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
                "GluTrans": self.optGluT
            }
        )
        run_func = [["setPAP2Soma","initialize","setK","run"]]
        if point_stim:
            run_func[0][2] = "setKPoint"

        run_func_args = [[{},  {"force_print_progress":True}, {"dur":50},  {}]]
        if hasattr(self,'kdifl') and self.kdifl:
            run_func[0] = [run_func[0][0]] +['set_kdfl_iter'] + run_func[0][1:]
            run_func_args[0] = [run_func_args[0][0]] + [{}] + run_func_args[0][1:] 
            funcArgs[-1]['dt'] = self.dt/5
            funcArgs[-1]['nakpump'] = self.OEpump

        if hasattr(self,'rect_off') and self.rect_off:
            run_func[0] = ['kir_rect_off'] + run_func[0]
            run_func_args[0] = [{}] + run_func_args[0]  
        if self.GAP:
            # turn gaps off with gap flag for runAmpLen
            funcArgs[-1]['gapCount'] = 0


        ccList = ["KoSize"]
        # make sure that funcParms is in the correct order of whatever iterations spits out
        AllCells = self.find_run_comp("run_comp_soma")
        if AllCells is None:
            results = parallizeFor(
                iterations,
                [PAPModel],
                funcArgs,
                ccList,
                run_func,
                run_func_args
            )
            self.free_figure(results)
        else:
            AllCells.release()


    def plot_range_pb(self):
        plt.cla()
        plt.clf()
        plt.figure(figsize=gl.figsize_panel)
        plt.subplots_adjust(left=0.2,bottom=0.2)

        i = 0
        lens = [l for l in np.arange(0,1.01,0.1) if l not in [0,1]]

        for range_len in lens:
            if range_len in [0,1]:
                continue
            def getbyRange(cell):
                if cell.sec_range == range_len:
                    return max(cell.vPAP) - cell.RMP
                else:
                    return None
            if i == 0:
                res_column_pb = self.read_run_comp("run_comp_pb",func=getbyRange)
                i += 1
            else:
                tmp_col = self.read_run_comp("run_comp_pb",func=getbyRange)
                if not column_in_array(tmp_col,res_column_pb):
                    res_column_pb = np.column_stack((res_column_pb,tmp_col))

        self.tag = self.tag.replace('_sec_range','') 
        res_column_soma = self.read_run_comp("run_comp_soma")
        res_column = self.read_run_comp("run_comp_bath")
        base = 3.5
        seed=2
        PAP_AllProperties = self.find_pap_props()
        for refs in PAP_AllProperties:
            for ref in refs:
                if 'dendrite' in ref.PAP_name and ref.seed == seed:
                    rangesec=ref.PAP_name
                    L = ref.PAP_properties[0]['L']

        for i,col in enumerate(res_column_pb.T):
            if np.nan not in col and 0 not in col[1:] and all(col > 0):
                plt.plot(
                    base+np.arange(0,self.KoCompMax + 1,self.KoCompStep),
                    col,
                    color=self.get_papLen_color_from_value(lens[i]*L)
                )


        for i,col in enumerate([res_column]):
            if i == 0:
                color = self.returnColor('global')
            else: 
                color = self.returnColor('Soma')
            if np.nan not in col and 0 not in col[1:] and all(col > 0):
                plt.plot(base+np.arange(0,self.KoCompMax + 1,self.KoCompStep),col,color=color)

 

        x = np.linspace(0,self.KoCompMax)
        plt.plot(base + x,self.nernst(x+base,120)+85,linestyle='--',color='black') 
 
        custom_handles = [
            Line2D([0], [0], color=self.returnColor('Soma'), label='Comparable to Soma'),
            Line2D([0], [0], color=self.returnColor('global'), label='Bath Application'),
            Line2D([0], [0], color='black', linestyle='--',label=gl.ek_raw),
        ]
        plt.legend(handles=custom_handles)
        plt.xlabel(gl.ion_o('K',short=True))
        plt.ylabel(gl.d_volt)
        plt.ylim(gl.lim_d_volt)
        plt.savefig(
            os.path.join(
                "../results/paperRes",
                f'K_v_rangeplot{self.tag}.pdf'
            )
        )
        plt.xlabel(f"{gl.ion_o('K',short=True)}")
        plt.xscale('log')
        xmax = self.KoCompMax
        ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0,)))
        ticks = ax.get_xticks()
        ticks = ticks[ticks <= xmax]
        ticks = np.append(ticks, xmax)

        ax.set_xticks(ticks)

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(gl.lim_log_ko)

        plt.savefig(
            os.path.join(
                "../results/paperRes",
                f'log_K_v_rangeplot{self.tag}.pdf'
            )
        )


        self.plot_morphology_range(rangesec)



    def plot_morphology_range(self,rangesec):
        # range sec
        range_secs = [(2,rangesec,None,0.1)]
        funcArgs = []
        funcArgs.append(
            {
                "mode": 0,
                "ComplexMorph": True,
                "Glu": True,
                "dt": self.dt,
                "stimdelay": self.stimdelay,
                "clleak": 0,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                'seed':1,
                'PAPLen':9.8
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.savePAPProp(name=True)
        totalL = 0
        color_names = {}
        sorted_list = sorted(cells.PAP_properties, key=lambda d: d["distance"],reverse=True)
        for cell in sorted_list:
            totalL += cell['L']
            color_names[cell['name']] = totalL

        plt.close('all')
        plt.colormaps.register(self.paplen_cm)


        if rank == 0:
            plot_3d_morphology(
                rangevar='PAP',
                color_names = color_names,
                colormap_name = self.paplen_cm.name,
                norm=self.paplen_norm,
                add_null=True,
                add_colorbar=False,
                rangesec=(rangesec,0.1)
            )
            plt.ylim((0,50))
            plt.xlim((-10,30))
            ax = plt.gca()
            ax.set_zlim((30,50))
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            zticks = ax.get_zticks()

            ax.set_xticks(xticks[1:-1])
            ax.set_yticks(yticks[1:-1])
            ax.set_zticks(zticks[1:-1])
            xlim = ax.get_xlim3d()
            ylim = ax.get_ylim3d()
            zlim = ax.get_zlim3d()

            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            zr = zlim[1] - zlim[0]

            ax.set_box_aspect((xr, yr, zr))
            plt.savefig(os.path.join("../morphResults",f"rangesec_plot_morph{self.tag}.pdf"))




    def run_comp_pb(self,iterations,point_stim=False,sec_range=False):
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
                "GluTrans": self.optGluT,
            }
        )
        run_func = [["setPAPNearSoma","get_PAPName","initialize","setK","run"]]
        if point_stim:
            run_func[0][3] = "setKPoint"
        run_func_args = [[{}, {},{"force_print_progress":True}, {"dur":50}, {} ]]

        if hasattr(self,'kdifl') and self.kdifl:
            run_func[0] = [run_func[0][0]] +['set_kdfl_iter'] + run_func[0][1:]
            run_func_args[0] = [run_func_args[0][0]] + [{}] + run_func_args[0][1:] 
            funcArgs[-1]['dt'] = self.dt/5
            funcArgs[-1]['nakpump'] = self.OEpump

        if hasattr(self,'rect_off') and self.rect_off:
            run_func[0] = ['kir_rect_off'] + run_func[0]
            run_func_args[0] = [{}] + run_func_args[0]  
        if self.GAP:
            # Turn gaps on with gap flag for runAmpLen
            funcArgs[-1]['gapCount'] = 0


        if point_stim and sec_range:
            ccList = ["KoSize","sec_range"]
            self.tag += '_sec_range'
            funcArgs[-1]['seed']=2
            iterations = comm.bcast([
                (i,j)
                for i in iterations
                for j in np.arange(0,1.01,0.1)
                if j not in [0,1]
            ],root=0)
        else:
            ccList = ["KoSize","seed"]
            # make sure that funcParms is in the correct order of whatever iterations spits out
            iterations = comm.bcast([
                (i,j)
                for i in iterations
                for j in range(self.pb_seed_max)
                if j not in self.skipPB
            ],root=0)
        AllCells = self.find_run_comp("run_comp_pb")
        tmp_AllCells =self.find_missing_iter(
            AllCells,
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            run_func,
            run_func_args,
            randomize=False
        )
        self.free_figure(tmp_AllCells)
        #if AllCells is None:
        #    results = parallizeFor(
        #        iterations,
        #        [PAPModel],
        #        funcArgs,
        #        ccList,
        #        run_func,
        #        run_func_args
        #    )
        #    self.free_figure(results)

    def find_pap_props(self):
        intermediary_files = os.listdir(os.path.join("intermediaryData"))
        # Mainly aims to keep additional information added during function
        fileName = f"plot_seed_map{self.tag}.pickle"
        if 'PAPLen' in self.tag:
            fileName = fileName.replace('PAPLen','PAPLen_point_stim')
        fileName = fileName.replace('PAPLen','seed')
        fileName = fileName.replace('_intra_diff','')
        fileName = fileName.replace('_Glu','')
        #print(fileName)
        

        for f in intermediary_files:
            if  fileName in f:
                print(f"found plot_seed file {f}")
                sys.stdout.flush()
                AllCells = [[]]
                with open(
                    os.path.join("intermediaryData", f), "rb"
                ) as handle:
                    AllCells = pickle.load(handle)

                    #for cells in AllCells:
                    #    for cell in cells:
                    #        cell.PAP_Ri = cell.PAP_Ri / (min(cell.vPAP) - cell.v_init) * (min(list(cell.vPAP)[self.get_initStep(cell):]) - cell.RMP)
                    #        print(cell.PAP_name,(min(cell.vPAP) - cell.v_init) / (min(list(cell.vPAP)[self.get_initStep(cell):]) - cell.RMP))
                if hasattr(self,'cutoff_PAP_properties'):
                    tmpCells = AllCells[:self.cutoff_PAP_properties]
                    for cells in AllCells:
                        for cell in cells:
                            if 'dendrite' in cell.PAP_name or 'soma' in cell.PAP_name:
                                tmpCells.append([cell])
                    AllCells = tmpCells
                return load_interm_data(AllCells)
    
        return None



    def find_run_comp(self,func_name,single_load=False):
        intermediary_files = os.listdir(os.path.join("intermediaryData"))
        # Mainly aims to keep additional information added during function
        tmptag = self.tag
        #self.addChannelTag()
        #if len(tmptag) > len(self.tag):
        #    self.tag = tmptag

        for f in intermediary_files:
            #print(f)
            #print(f'{func_name}{self.tag}.pickle')
            if f == f"{func_name}{self.tag}.pickle":
                print(f"found intermediary file {f}")
                sys.stdout.flush()
                AllCells = [[]]
                with open(
                    os.path.join("intermediaryData", f), "rb"
                ) as handle:
                    AllCells = pickle.load(handle)

                if single_load:
                    return AllCells
                else:
                    return load_interm_data(AllCells)
    
        return None

    def read_run_comp(self,func_name,func=None):
        AllCells = self.find_run_comp(func_name)
        #self.plotIKSeries.__wrapped__(self,AllCells)
        if AllCells is not None:
            column_res = [] 
            for cells in AllCells:
                for cell in cells:
                    if int(cell.KoSize/self.KoCompStep) >= len(column_res):
                        while len(column_res) < int(cell.KoSize/self.KoCompStep) + 1:
                            column_res.append(None)

                    if func is None:
                        column_res[int(cell.KoSize/self.KoCompStep)] = max(cell.vPAP) - cell.RMP
                    else:
                        if func(cell) is not None:
                            column_res[int(cell.KoSize/self.KoCompStep)] = func(cell) 
                

            AllCells.release()
            return np.array(column_res)

        else:
            return None

    @read_data
    def runFitCaliburation(
        self,
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
                "seed":self.seed,
                "PAPLen":self.spillOverLen,
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

        ccList = ["KoSize"]


        run_func_args = [[{}, {'slow':self.spillOverSlowing},{"number": 10, "freq": self.freq}, {"slow":1},{}]]

        run_func = [["savePAPProp","initialize", "setSlow_iter","multiSpike", "setSlow_iter","run"]]
        run_func_args[0] = [{}] + run_func_args[0]
        self.KoCompMax = gl.max_ko
        self.KoCompStep = 5 
        iterations = [
            (i)
            for i in range(0,self.KoCompMax+1,self.KoCompStep)
        ]
 
        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            run_func,
            run_func_args,
        )

        comm.Barrier()
        self.free_figure(results)
        if rank == 0:
            self.plot_fluor_comparison(results)

 


    @read_data
    def runAmpLenComparison(
        self, comparison, iterations, maxStep, intermStep, logx=None, add_bath=True,plot_total_p=True,point_stim=False
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


        run_func_args = [[{}, {"number": self.stimCount, "freq": self.freq,"point":point_stim}, {}]]

        if comparison == "seed" or comparison == 'PAPLen':
            run_func = [["savePAPProp","initialize", "multiSpike", "run"]]
            run_func_args[0] = [{}] + run_func_args[0]
        else:
            run_func = [["initialize", "multiSpike", "run"]]

        if hasattr(self,'kdifl') and self.kdifl:
            run_func[0] = ['set_kdfl_iter'] + run_func[0]
            run_func_args[0] = [{}] + run_func_args[0] 
            funcArgs[-1]['dt'] = self.dt/5
            funcArgs[-1]['nakpump'] = self.OEpump


        if hasattr(self,'rect_off') and self.rect_off:
            run_func[0] = ['kir_rect_off'] + run_func[0]
            run_func_args[0] = [{}] + run_func_args[0]  
            
        if self.GAP:
            funcArgs[-1]['gapCount'] = 0


        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ccList,
            run_func,
            run_func_args,
        )

        comm.Barrier()
        self.free_figure(results)
        if add_bath and comparison in ["seed","PAPLen"]:
            res_column = self.read_run_comp("run_comp_bath")
            res_column_soma = self.read_run_comp("run_comp_soma")
            def getiKdiff(cell):
                return -(min(list(cell.iKPAP))- max(list(cell.iKPAP)))
            ik_soma = self.read_run_comp("run_comp_soma",func=getiKdiff)
 
            PAP_AllProperties = self.find_pap_props()
            if PAP_AllProperties is not None:
                self.pap_win = True
            # only specific use case
            #print(self.tag)
            def find_by_name(name):
                save_name = []
                for refs in PAP_AllProperties:
                    for ref in refs:
                        if name in ref.PAP_name:
                            save_name.append(ref)

                if len(save_name) > 0:
                    return save_name
                else:
                    return None 
            property_soma = find_by_name('soma')[0].PAP_properties
            ri_soma = [find_by_name('soma')[0].PAP_Ri] 
            self.plot_RiPlots(PAP_AllProperties,[find_by_name('soma')],[find_by_name('dendrite')])
            res_column_pb = None
            if comparison == 'seed':
                property_pb  = find_by_name('dendrite')
                ri_pb = [[ri.PAP_Ri for ri in property_pb]]
                ri_pb_seed = [ri.seed for ri in property_pb]
            
                for i,j in enumerate(ri_pb_seed):
                    def getbySeed(cell):
                        if cell.seed == j:
                            return max(cell.vPAP) - cell.RMP
                        else:
                            return None
                    def getiKdiff(cell):
                        if cell.seed == j:
                            return -(min(list(cell.iKPAP))- max(list(cell.iKPAP)))
                        else:
                            return None
    
                
                    if i == 0:
                        ik_column_pb = self.read_run_comp("run_comp_pb",func=getiKdiff)
                        res_column_pb = self.read_run_comp("run_comp_pb",func=getbySeed)
                    else:
                        tmp_col = self.read_run_comp("run_comp_pb",func=getbySeed)
                        if not column_in_array(tmp_col,res_column_pb):
                            res_column_pb = np.column_stack((res_column_pb,tmp_col))
                        tmp_col = self.read_run_comp("run_comp_pb",func=getiKdiff)
                        if not column_in_array(tmp_col,ik_column_pb):
                            ik_column_pb = np.column_stack((ik_column_pb,tmp_col))


        if rank == 0:
            if comparison == 'seed':
                self.plot_fluor_comparison(results)
            plt.cla()
            plt.clf()
            plt.figure(figsize=gl.figsize_panel)
            total_corr = {'x':[],'y':[],'p-value':[]} 
            search_closest = ['area']
            inset_props = ['mol','area','ecs','kir_count','Ra','r_ratio','IKClamp']
            inset_pb = ['PAP_Ri']
            log_property = []
            skip_props = ['iK','nseg','diff_tau','ecs','area','Ra','diam','adj_diam','distance','mol','kir_count','L','RMP','Rk','r_ratio']
            if add_bath and comparison in ["seed"]:
                pap_props = ['PAP_Ri','mol','Ra','r_ratio','Rk','iK','RMP']
                if PAP_AllProperties is not None:
                    pap_props += list(PAP_AllProperties[0][0].PAP_properties[-1].keys()) 
                    ref_paps = [find_by_name('Glia')]
                elif hasattr(results[0][0],'PAP_properties'):
                    pap_props += list(results[0][0].PAP_properties[-1].keys()) 
                    ref_paps = results
                else:
                    ref_paps = None
                for p in skip_props:
                    pap_props.remove(p)
                for property in pap_props:
                    plt.cla()
                    plt.clf()

                    if property in inset_props or property in inset_pb:
                        fig = plt.figure(figsize=gl.figsize_panel)
                        gs = fig.add_gridspec(nrows=2,ncols=1,hspace=0.2)
                        ax = fig.add_subplot(gs[0])
                        ax_inset = fig.add_subplot(gs[1])
                        fig.subplots_adjust(bottom=0.2)


                    else:
                        plt.figure(figsize=gl.figsize_panel)
                        plt.subplots_adjust(bottom=0.15)
                        ax = plt.gca()

                    pap_corr = {'x':[],'y':[],'seed':[]}
                    for cells in results:
                        for cell in cells:
                            def find_seed(seed):
                                for refs in ref_paps:
                                    for ref in refs:
                                        if ref.seed == seed:
                                            return ref
                                return None 

                            prp_cell = find_seed(cell.seed)
                            if prp_cell is not None:
                                if property in ['adj_diam','area','kir_count']:
                                    val = 0
                                    # total of all sections in pap
                                    for p in prp_cell.PAP_properties:
                                        val += p[property]
                                elif property == 'PAP_Ri':
                                    val = prp_cell.PAP_Ri
                                elif property == 'iK':
                                    val = min(list(cell.iKPAP))
                                elif property == 'RMP':
                                    val = cell.RMP
                                elif property == 'mol':
                                    val = cylindrical_shell_volume(
                                        prp_cell.PAP_properties[-1]['diam'],
                                        prp_cell.PAP_properties[-1]['ecs'],
                                        prp_cell.PAP_properties[-1]['L']
                                    )
                                    val *= self.KoCompMax

                                elif property =='Ra' and hasattr(prp_cell,'PAP_Ri'):
                                    val = prp_cell.PAP_Ri - 1/(22 * prp_cell.PAP_properties[-1]['kir_count'])*1e6 
                                elif (property =='r_ratio' or property == 'Rk') and hasattr(prp_cell,'PAP_Ri'):
                                    val = (22 * prp_cell.PAP_properties[-1]['kir_count'])/1e6 
                                else:
                                    val = prp_cell.PAP_properties[-1][property]
                            else:
                                continue
                            if cell.KoSize == 10:
                                if property == 'r_ratio' or property == 'Rk':
                                    val *= 1/(1+np.exp(98.892+max(list(cell.vPAP)))/10.89) 
                                    val += (prp_cell.PAP_properties[-1]['area']*1e-8*1.4*1e6) 
                                    if property == 'r_ratio':
                                        val *= prp_cell.PAP_Ri
                                    else:
                                        val = 1/val
                                        val = (max(list(cell.vPAP))-cell.RMP)/-(min(list(cell.iKPAP))- max(list(cell.iKPAP)))
                                ax.scatter(val,max(list(cell.vPAP))-cell.RMP,color=self.returnColor('PAP'))
                                if property in inset_props:
                                    ax_inset.scatter(val,max(list(cell.vPAP))-cell.RMP,color=self.returnColor('PAP'))
                                pap_corr['x'].append(val)
                                pap_corr['y'].append(max(list(cell.vPAP))-cell.RMP)
                                pap_corr['seed'].append(cell.seed)

                    if len(total_corr['x']) > 0:
                        if property not in ['kir_count']:
                            tmp_corr = deepcopy(pap_corr['x'])
                            total_corr['x'].append(np.array(tmp_corr))
                    else:
                        total_corr['x'] = [np.array(pap_corr['x'])]
                    y_ols = np.array(deepcopy(pap_corr['y']))

                    if not (np.amax(pap_corr['x']) == np.amin(pap_corr['x']) and len(pap_corr['x']) > 1):
                        x = np.linspace(min(pap_corr['x']),max(pap_corr['x']))
                        if property in log_property:
                            correlation_coefficient, p_value = pearsonr(np.log(pap_corr['x']), pap_corr['y'])
                            res = linregress(np.log(pap_corr['x']),pap_corr['y'])
                            if p_value < self.alpha:
                                ax.plot(x,res.slope*np.log(x) + res.intercept,linestyle='--',color=self.returnColor('PAP'),zorder=-1)
                                if property in inset_props:
                                    ax_inset.plot(x,res.slope*np.log(x) + res.intercept,linestyle='--',color=self.returnColor("PAP"),zorder=-1)

                        else:
                            correlation_coefficient, p_value = pearsonr(pap_corr['x'], pap_corr['y'])
                            res = linregress(pap_corr['x'],pap_corr['y'])
                            if p_value < self.alpha:
                                ax.plot(x,res.slope*x + res.intercept,linestyle='--',color=self.returnColor('PAP'),zorder=-1)
                                if property in inset_props:
                                    ax_inset.plot(x,res.slope*x + res.intercept,linestyle='--',color=self.returnColor("PAP"),zorder=-1)

                        #total_corr['p-value'].append(p_value)
                        if p_value < self.alpha:
                            ax.set_title(f'PAP={p_value:.1e}')
                        else:
                            ax.set_title('')


                    ax_main = ax

                    if property in inset_pb:
                        axes = [ax_main,ax_inset]

                    else:
                        axes = [ax_main]

                    index_ko = int(10/self.KoCompStep - 1)

                    for ax in axes:
                        large_sec = {'x':[],'y':[]} 
                        if ri_soma is not None and property == 'PAP_Ri':
                            t_stat, p_value = ttest_1samp(pap_corr['x'], ri_soma[0])
                            ax.scatter(ri_soma[0],res_column_soma[index_ko],color=self.returnColor('Soma'))
                            pap_corr['x'].append(ri_soma[0])
                            pap_corr['y'].append(res_column_soma[index_ko])
                            pap_corr['seed'].append(None)
                        elif property == 'iK':
                            AllCells = self.find_run_comp('run_comp_soma',single_load=True)
                            for cells in AllCells:
                                for cell in cells:
                                    val = min(list(cell.iKPAP))
                                    if cell.KoSize == 10:
                                        ax.scatter(val,res_column_soma[index_ko],color=self.returnColor('Soma'))
                        elif property == 'RMP':
                            AllCells = self.find_run_comp('run_comp_soma',single_load=True)
                            for cells in AllCells:
                                for cell in cells:
                                    val = cell.RMP
                                    if cell.KoSize == 10:
                                        ax.scatter(val,res_column_soma[index_ko],color=self.returnColor('Soma'))

                        elif property_soma is not None:
                            if property == 'mol':
                                val = cylindrical_shell_volume(
                                    property_soma[-1]['diam'],
                                    property_soma[-1]['ecs'],
                                    property_soma[-1]['L']
                                )
                                val *= self.KoCompMax
                            elif property =='Ra' and ri_soma is not None:
                                val = ri_soma[0] - 1/(22 * property_soma[-1]['kir_count'])*1e6 
                            elif (property =='r_ratio' or property == 'Rk') and ri_soma is not None:
                                val = ((22 * property_soma[-1]['kir_count'])/1e6)
                                val *= 1/(1+np.exp(98.892+res_column_soma[index_ko]-85)/10.89) 
                                val += (property_soma[-1]['area']*1e-8 * 1.4 *1e6)
                                if property == 'r_ratio':
                                    val *= ri_soma[0]
                                else:
                                    val = 1/ val
                                    val = (res_column_soma[index_ko])/ik_soma[-1]
                            else:
                                val = property_soma[-1][property]
                            ax.scatter(val,res_column_soma[index_ko],color=self.returnColor('Soma'))
                            pap_corr['x'].append(val)
                            pap_corr['y'].append(res_column_soma[index_ko])
                            pap_corr['seed'].append(None)
                        large_sec['x'].append(pap_corr['x'][-1])
                        large_sec['y'].append(pap_corr['y'][-1])

                        if comparison == 'seed':
                            for i,j in enumerate(ri_pb_seed):
                                if ri_pb is not None and property == 'PAP_Ri':
                                    ax.scatter(ri_pb[0][i],res_column_pb[index_ko][i],color=self.returnColor("Primary Branch"))
                                    pap_corr['x'].append(ri_pb[0][i])
                                    pap_corr['y'].append(res_column_pb[index_ko][i])
                                    pap_corr['seed'].append(i)
                                elif property == 'iK':
                                    AllCells = self.find_run_comp("run_comp_pb",single_load=True)
                                    for cells in AllCells:
                                        for cell in cells:
                                            val = min(list(cell.iKPAP))
                                            if cell.KoSize == 10 and cell.seed == j:
                                                ax.scatter(val,res_column_pb[index_ko][i],color=self.returnColor('Primary Branch'))
                                elif property == 'RMP':
                                    AllCells = self.find_run_comp("run_comp_pb",single_load=True)
                                    for cells in AllCells:
                                        for cell in cells:
                                            val = cell.RMP
                                            if cell.KoSize == 10 and cell.seed == j:
                                                ax.scatter(val,res_column_pb[index_ko][i],color=self.returnColor('Primary Branch'))


                                elif property_pb is not None:
                                    for pb in property_pb:
                                        if pb.seed == j:
                                            if property == 'mol':
                                                val = cylindrical_shell_volume(
                                                    pb.PAP_properties[-1]['diam'],
                                                    pb.PAP_properties[-1]['ecs'],
                                                    pb.PAP_properties[-1]['L']
                                                )
                                                val *= self.KoCompMax
                                            elif property =='Ra' and ri_pb is not None:
                                                val = ri_pb[0][i] - 1/(22 * pb.PAP_properties[-1]['kir_count'])*1e6 
                                            elif (property =='r_ratio' or property == 'Rk') and ri_pb is not None:
                                                val =  (22 * pb.PAP_properties[-1]['kir_count'])/1e6
                                                val *= 1/(1+np.exp(98.892+res_column_pb[index_ko][i]-85)/10.89) 
                                                val += (pb.PAP_properties[-1]['area']*1e-8 * 1.4 *1e6)
                                                if property == 'r_ratio':
                                                    val *= ri_pb[0][i]
                                                else:
                                                    val = 1/ val
                                                    val = res_column_pb[index_ko][i]/ik_column_pb[-1][i]
                                            else:
                                                val = pb.PAP_properties[-1][property]
    
                                            ax.scatter(val,res_column_pb[index_ko][i],color=self.returnColor('Primary Branch'))
                                            pap_corr['x'].append(val)
                                            pap_corr['y'].append(res_column_pb[index_ko][i])
                                            pap_corr['seed'].append(i)
                                large_sec['x'].append(pap_corr['x'][-1])
                                large_sec['y'].append(pap_corr['y'][-1])

                        if not (np.amax(large_sec['x']) == np.amin(large_sec['x']) and len(large_sec['x']) > 1):
                            x = np.linspace(min(large_sec['x']),max(large_sec['x']))
                            if property in log_property:
                                correlation_coefficient, p_value = pearsonr(np.log(large_sec['x']), large_sec['y'])
                                res = linregress(np.log(large_sec['x']),large_sec['y'])
                                if p_value < 0.05:
                                    ax.plot(x,res.slope*np.log(x) + res.intercept,linestyle='--',color=self.returnColor("Soma"),zorder=-1)

                            else:
                                correlation_coefficient, p_value = pearsonr(large_sec['x'], large_sec['y'])
                                res = linregress(large_sec['x'],large_sec['y'])
                                if p_value < 0.05:
                                    ax.plot(x,res.slope*x + res.intercept,linestyle='--',color=self.returnColor("Soma"),zorder=-1)
                            total_corr['p-value'].append(p_value)
                            if p_value < self.alpha:
                                ax.set_title(ax.get_title() + f'Other={p_value:.1e}')


                    ax = ax_main
                    if not (np.amax(pap_corr['x']) == np.amin(pap_corr['x']) and len(pap_corr['x']) > 1) and plot_total_p:
                        x = np.linspace(min(pap_corr['x']),max(pap_corr['x']))
                        if property in log_property:
                            correlation_coefficient, p_value = pearsonr(np.log(pap_corr['x']), pap_corr['y'])
                            res = linregress(np.log(pap_corr['x']),pap_corr['y'])
                            if p_value < 0.05:
                                ax.plot(x,res.slope*np.log(x) + res.intercept,linestyle='--',color=self.returnColor('local'),zorder=-1)

                        else:
                            correlation_coefficient, p_value = pearsonr(pap_corr['x'], pap_corr['y'])
                            res = linregress(pap_corr['x'],pap_corr['y'])
                            if p_value < 0.05:
                                ax.plot(x,res.slope*x + res.intercept,linestyle='--',color=self.returnColor('local'),zorder=-1)
                        total_corr['p-value'].append(p_value)
                        if p_value < self.alpha:
                            ax.set_title(ax.get_title() + f'Total={p_value:.1e}')
                            if property == 'PAP_Ri':
                                ax.set_title(ax.get_title() + f'R$^2$={correlation_coefficient**2:.1e}')

                        


                    ax.set_ylabel(gl.d_volt)
                    ax.set_ylim(gl.clim_volt)
                    if property in inset_props or property in inset_pb:
                        ax.set_ylabel(gl.d_volt_short)
                        ax_inset.set_ylabel(gl.d_volt_short)
                        ax_inset.set_ylim(gl.clim_volt)
                    if not (np.amax(pap_corr['x']) == np.amin(pap_corr['x']) and len(pap_corr['x']) > 1):
                        if property == search_closest:
                            tmp_min = np.inf
                            max_x = max(pap_corr['x'])
                            for x,y,seed in zip(pap_corr['x'],pap_corr['y'],pap_corr['seed']):
                                if x == max_x:
                                    mprint(f'found max seed {seed}')
                                lin_y = res.slope*x + res.intercept
                                if tmp_min > abs(y-lin_y) and seed is not None:
                                    tmp_min = abs(y-lin_y)
                                    mprint(f'Found closer seed {seed}')
                        if property in log_property:
                            ax.set_xscale('log')
                            fit_label = f'{res.slope:.1f}ln(x)+{res.intercept:.1f}'
                        else:
                            fit_label = f'{res.slope:.1f}x+{res.intercept:.1f}'

                        custom_handles = [
                            Line2D([0], [0], marker='o', color=self.returnColor('PAP'), label='PAP',
                                markerfacecolor=self.returnColor('PAP'), linestyle='None',markersize=10),
                            Line2D([0], [0], marker='o', color=self.returnColor('Soma'), label='Soma',
                                markerfacecolor=self.returnColor('Soma'), linestyle='None',markersize=10),
                            Line2D([0], [0], marker='o', color=self.returnColor('Primary Branch'), label='Primary Branch',
                                markerfacecolor=self.returnColor('Primary Branch'), linestyle='None',markersize=10),
                            Line2D([0], [0], color=self.returnColor('PAP'), label='PAP fit',linestyle='--'),
                            Line2D([0], [0], color=self.returnColor('Soma'), label='Soma-branch fit',linestyle='--'),
                            Line2D([0], [0], color=self.returnColor('local'), label='Total fit',linestyle='--'),
                        ]

                        if property not in inset_props and property not in inset_pb:
                            ax.legend(handles=custom_handles, loc='best')

                    if property in inset_props or property in inset_pb:
                        ax_inset.set_title('')
                        ax = ax_inset

                    if property == 'area':
                            
                        if property in log_property:
                            ax.set_xlabel(f'Area {gl.unit_um_squared}')
                        else:
                            ax.set_xlabel(f'Area {gl.unit_um_squared}')
                    elif property == 'ecs':
                        if property in log_property:
                            ax.set_xlabel(f"l$_{ECS}$ {gl.unit_micron}")
                        else:
                            ax.set_xlabel(f'ECS {gl.unit_micron}')
                    elif property == 'PAP_Ri':
                        ax.set_xlabel(gl.ri)
                    else:
                        ax.set_xlabel(property)


                    plt.savefig(os.path.join("../results/paperRes",f"{property}_peakV_{comparison}{self.tag}.pdf"))


                print("Stat Results:")
                print(pap_props)
                print("OLS Results:")
                x_ols = np.column_stack(total_corr['x'])
                x_ols = sm.add_constant(x_ols) 
                model = sm.OLS(y_ols, x_ols).fit()
                print(model.params)
                print("Adjusted p-value Results:")
                reject, p_adjusted, alphacSidak, alphacBonf = smm.multipletests(total_corr['p-value'],method='fdr_bh')
                print(f"Original p-values: {total_corr['p-value']}")
                print(f"Adjusted p-values: {p_adjusted}")
                print(f"Rejection results (at alpha=0.05): {reject}")
                print(f"Adjusted alpha level: {alphacBonf}")



            plt.cla()
            plt.clf()
            plt.figure(figsize=gl.figsize_panel)
            plt.subplots_adjust(bottom=0.15)
            imArray = np.zeros(
                (
                    int(self.KoCompMax / self.KoCompStep + 1),
                    int(len(iterations) / (self.KoCompMax / self.KoCompStep + 1)),
                )
            )  # int(maxStep / intermStep) + 1))
            for res in results:
                if logx is not None:
                    try:
                        index = int(np.where(logx == getattr(res[0], ccList[1]))[0])
                    except TypeError:
                        imArray = np.concatenate(
                            (
                                imArray,
                                np.zeros(
                                    (
                                        int(self.KoCompMax / self.KoCompStep + 1),
                                        1
                                    )
                                )
                            ),
                            axis=1
                        )
                        index = int(np.where(logx == max(logx))[0]) + 1
                        logx = np.append(logx,np.array(getattr(res[0], ccList[1])))

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

            comp_res = {} 
            if add_bath and comparison in ["seed","PAPLen"]:
                if res_column_soma is not None:
                    comp_res['soma'] = res_column_soma

                if res_column_pb is not None:
                    comp_res['pb'] = res_column_pb

                if res_column is not None:
                    comp_res['bath'] = res_column

                for res_val in comp_res.values(): 
                    if res_val.ndim < 2:
                        res_val = res_val[:,np.newaxis]

                    imArray = np.concatenate((imArray,res_val),axis=1)
                if comparison == "seed":
                    skip = 1
                    dec = 0
                    printType = int
                else:
                    skip = 0
                    dec = 1
                    printType = float

                if logx is None:
                    xlabels = np.round(
                        np.arange(skip, maxStep + intermStep / 2, intermStep),
                        decimals=dec,
                    ).astype(printType)
                else:
                    xlabels = np.round(
                        logx,
                        decimals=dec,
                    ).astype(printType)

                for key in comp_res.keys():
                    if key == 'pb':
                        for i in range(self.pb_seed_max):
                            if hasattr(self,'skipPB') and i not in self.skipPB:
                                xlabels = np.append(
                                    xlabels,
                                    f'branch {i}' 
                                )

                    else:
                        xlabels = np.append(
                            xlabels,
                            key 
                        )

                base = 3.5
                x = np.linspace(0,self.KoCompMax)
                plt.plot(base + x,self.nernst(x+base,res[0].kin)-res[0].RMP,linestyle='--',color='black') 
  
                if comparison == 'seed':
                    for i,col in enumerate(imArray.T):
                        label = xlabels[i]
                        if label not in ['soma','bath'] and 'branch' not in label:
                            color=self.returnColor('PAP')
                        else:
                            if label in ['soma']:
                                color =self.returnColor('Soma')
                            elif 'branch' in label:
                                color=self.returnColor('Primary Branch')
                            else:
                                color=self.returnColor('global')
                        if np.nan not in col and 0 not in col[1:] and all(col > 0) and not (label != 'bath' and any(col >60)) :
                            plt.plot(base+np.arange(0,self.KoCompMax + 1,self.KoCompStep),col,color=color)
                            #if label not in ['soma','bath'] and 'branch' not in label:
                            #    print(base+np.arange(0,self.KoCompMax + 1,self.KoCompStep),col)

                    custom_handles = [
                        Line2D([0], [0], color=self.returnColor('PAP'), label='PAP'),
                        Line2D([0], [0], color=self.returnColor('Soma'), label='Soma'),
                        Line2D([0], [0], color=self.returnColor('Primary Branch'), label='Primary Branch'),
                        Line2D([0], [0], color=self.returnColor('global'), label='Bath'),
                        Line2D([0], [0], color='black', linestyle='--',label=gl.ek_raw),
                    ]
                    plt.legend(handles=custom_handles)

                
                elif comparison == 'PAPLen':
                    papLens = []
                    for label in xlabels:
                        try:
                            float(label)
                            papLens.append(float(label))
                        except ValueError:
                            continue
                    for i,col in enumerate(imArray.T):
                        label = xlabels[i]
                        if label not in ['soma','bath'] and 'branch' not in label:
                            color=self.get_papLen_color_from_value(papLens[i])
                        else:
                            if label in ['soma']:
                                color =self.returnColor('Soma')
                                continue
                            elif 'branch' in label:
                                color=self.returnColor('Primary Branch')
                                continue
                            else:
                                color=self.returnColor('global')
                        if np.nan not in col or 0 not in col[1:] or not all(col > 0):
                            plt.plot(base+np.arange(0,self.KoCompMax + 1,self.KoCompStep),col,color=color,lw=0.3,alpha=0.5)

                    custom_handles = [
                        Line2D([0], [0], color=self.returnColor('PAP'), label='PAP Size'),
                        Line2D([0], [0], color=self.returnColor('Soma'), label='Comparable to Soma'),
                        Line2D([0], [0], color=self.returnColor('global'), label='Bath Application'),
                        Line2D([0], [0], color='black', linestyle='--',label=gl.ek_raw),
                    ]
                    plt.legend(handles=custom_handles)
 

                plt.xlabel(gl.ion_o('K',short=True))
                plt.ylabel(gl.d_volt)
                plt.ylim(gl.lim_d_volt)



                comp_lin = plt.axhline(20,color='gray',linestyle='-.',lw=1)
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f'K_v_plot{self.tag}.pdf'
                    )
                )
                comp_lin.remove()
                plt.draw()
                plt.xlabel(f"{gl.ion_o('K',short=True)}")
                plt.xscale('log')
                xmax = self.KoCompMax
                ax = plt.gca()
                ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0,)))
                ticks = ax.get_xticks()
                ticks = ticks[ticks <= xmax]
                ticks = np.append(ticks, xmax)

                ax.set_xticks(ticks)

                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.set_xlim(gl.lim_log_ko)

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f'log_K_v_plot{self.tag}.pdf'
                    )
                )

                plt.cla()
                plt.clf()
                plt.figure(figsize=gl.figsize_panel)

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
                if add_bath and comparison == "seed" and len(comp_res) > 0:
                    xlabels = np.round(
                        np.arange(skip, maxStep + 1 + intermStep / 2, intermStep * 2),
                        decimals=dec,
                    ).astype(printType)
                    xticks = np.arange(0, int(maxStep / intermStep) + (1 - skip) + 1, 2)

                    for key in comp_res.keys():
                        xticks = np.append(
                            xticks,
                            np.max(xticks)+1,
                        )
                        xlabels = np.append(
                            xlabels,
                            key 
                        )
                    plt.xticks(xticks,xlabels)
                    for i,_ in enumerate(comp_res.keys()):
                        ticks = plt.gca().get_xticklabels()
                        ticks[-(i+1)].set_rotation(90)
 
                else:
                    plt.xticks(
                        np.arange(0, int(maxStep / intermStep) + (1 - skip), 2),
                        np.round(
                            np.arange(skip, maxStep + intermStep / 2, intermStep * 2),
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
            plt.colorbar(
                label=gl.d_volt_short,
                ticks=np.arange(0, 20, 2),
                extend="max",
                shrink=0.7,
            )
            plt.clim(gl.clim_volt)
            if comparison == "PAPLen":
                self.GluT = False  # just to force plot setLabel Colors
            self.setLabelColors(
                res[0].PAParea,
                Kir=True,
                y=False,
                x=True if comparison == "PAPLen" else False,
                chanOverride={"Kir": (res[0].PAPKirCount, res[0].PAPKirCount_std)},
            )

            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"FullPotassiumAmp{self.tag}_{comparison}.pdf",
                )
            )
        plt.close('all')
        if hasattr(self,'pap_win') and PAP_AllProperties is not None:
            PAP_AllProperties.release()

            #if comparison == "PAPLen":
            #    self.plotIKSeries(results, tagReset=True, setKoylim=True)
            if comparison == "seed":
                self.mergePlotsIK.__wrapped__(self,results, "KoSize", "seed", selected=1)
        

    def plot_RiPlots(self,*AllCells_List):
        plt.cla()
        plt.clf()
        plt.figure(figsize=gl.figsize_panel)
        plt.subplots_adjust(left=0.2)
        for AllCells in AllCells_List:
            for cells in AllCells:
                for cell in cells:
                    if hasattr(cell,"PAP_Ri") and hasattr(cell,'PAP_name') and hasattr(cell,'time'):
                            if 'dendrite' in cell.PAP_name:
                                cell_label = 'Primary Branch'
                                color = self.returnColor('Primary Branch')
                            elif 'soma' == cell.PAP_name:
                                cell_label = 'Soma'
                                color = self.returnColor('Soma')
                            else:
                                cell_label = 'PAP'
                                color = self.returnColor('PAP')
                            plt.plot(cell.time,np.array(cell.vPAP) - cell.RMP,label=cell_label,color=color)

                    else:
                        wMessage('No PAP_Ri detected')
        # TODO:
        # Integrate to global labels
        plt.xlim((145,160))
        plt.xlabel(gl.ms)
        plt.ylim((-0.6,0.01))
        plt.ylabel(gl.d_volt)
        #plt.legend()
        plt.savefig(
            os.path.join(
                "../results/paperRes",
                f'PAP_Ri_Raw{self.tag}.pdf'
            )
        )



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
            if comparison == 'KoSize':
                plt.figure(figsize=gl.figsize_panel)
                plt.subplots_adjust(left=0.16)
                #if res_column_soma is not None:
                #    comp_res['soma'] = res_column_soma

                #if res_column is not None:
                #    comp_res['bath'] = res_column

                for i,col in enumerate(imArray):
                    color=self.returnColor('PAP')
                    base=3.5
                    plt.plot(base+np.arange(0,self.KoCompMax + 1,self.KoCompStep),col,color=color)
                    custom_handles = [
                        Line2D([0], [0], color=self.returnColor('PAP'), label='PAP'),
                    ]
                plt.xlabel(gl.ion_o('K',short=True))
                plt.ylabel(gl.d_volt_short)



                plt.legend(handles=custom_handles)
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f'K_v_kir_plot{self.tag}.pdf'
                    )
                )


                plt.xlabel(f"{gl.ion_o('K',short=True)}")
                plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
                plt.xscale('log')
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f'log_K_v_kir_plot{self.tag}.pdf'
                    )
                )




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
            plt.colorbar(label=gl.d_volt_short, ticks=np.arange(0, 20, 2), extend="max")
            plt.clim(gl.clim_volt)
            if comparison == "PAPLen":
                self.GluT = False  # just to force plot setLabel Colors
                chanOverride = {
                    "Kir": (
                        int(res[0].PAPKirCount / res[0].PAPLen),
                        ceil(res[0].PAPKirCount_std / res[0].PAPLen),
                    )
                }

            else:
                chanOverride = {
                    "Kir": (
                        int(res[0].PAPKirCount),
                        ceil(res[0].PAPKirCount_std),
                    )
                }
            self.setLabelColors(
                res[0].PAParea,
                Kir=True,
                y=False,
                x=True if comparison == "PAPLen" else False,
                chanOverride=chanOverride,
            )

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    "../results/paperRes", f"FullPotassium{self.tag}_{comparison}.pdf"
                )
            )

            #if comparison == "KoSize" or comparison == "PAPLen":
            #    self.plotIKSeries(results, tagReset=True, setKoylim=True)

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
                "GluTrans": self.optGluT,
                "kir2":-1e7,
                "clleak": 0,
                "kleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
            }
        )
        ccList = ["kleak"]
        iterations = [x for x in np.geomspace(0.69, 5,num=15)]
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
                    test_conductance.append(cell.GENEDict['kleak'])
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
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                "../morphResults",
                                "dual_patch.pdf"
                            )
                        )
            plt.cla()
            plt.clf()
            fig, ax = plt.subplots(figsize=gl.figsize_panel)
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
            plt.savefig(
                os.path.join(
                    "../morpResults",
                    "dual_patch_sensitivity.pdf"
                )
            )

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
        #print(abs(max(list(cells.vPAP)) - cells.RMP - optmV))
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
        models = ["K$^+$ Model", "GLT-1 Model", "GABA$_A$R Model", "NMDAR Model"]
        stim = [50, 100, "theta"]
        papcounts = [1, 40]
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
                    if p > 1:
                        funcArgs[-1]["recordAllPAP"] = True
                    if m == "GLT-1 Model":
                        funcArgs[-1]["Glu"] = True
                        funcArgs[-1]["multiple"] = None
                        funcArgs[-1]["GluTrans"] = self.optGluT
                    elif m == "GABA$_A$R Model":
                        funcArgs[-1]["Glu"] = False
                        funcArgs[-1]["GABA"] = True
                        funcArgs[-1]["GABACount"] = self.optGABAR
                        funcArgs[-1]['dt'] / = 50 
                    elif m == "NMDAR Model":
                        funcArgs[-1]["multiple"] = self.optNMDAR
                        funcArgs[-1]["Glu"] = True
                        funcArgs[-1]["GluTrans"] = self.optGluT

                    if hasattr(self,'kdifl') and self.kdifl:
                        funcArgs[-1]['dt'] / = 20
                        funcArgs[-1]['nakpump'] = self.OEpump

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
                if hasattr(self,'kdifl') and self.kdifl:
                    cells.set_diff_ki(True)
                if funcArgs[index]['PAPCount'] > 1:
                    cells.setSlowing(100)

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

        self.plot_physiological(AllCells, stim, papcounts, models,syn_count=papcounts[1])


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
            KoSize = 22
            if len(x) == 3:
                if use_tau:
                    glt, kir, PAPLen = x
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
                    glt, kir, PAPLen, tau2, slowing = x
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
        voltageOn=False,
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
            self.plot_combined_cvk([AllCells])
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
                        plt.tight_layout()
                        plt.savefig(f"../results/paperRes/{Fname}.pdf")
                    if np.isnan(total):
                        total = np.inf
                    elif verbose:
                        print(f"{total=}")

                else:
                    fig, ax1 = plt.subplots(figsize=gl.figsize_panel)
                    color = {10: "tab:blue", 5: "tab:orange", 1: "tab:green"}
                    plotObjects = []
                    plotObjects_ax2 = []
                    if voltageOn:
                        fig.subplots_adjust(left=0.15, right=0.85, top=0.9)
                        ax2 = ax1.twinx()
                    else:
                        fig.subplots_adjust(left=0.2,right=0.95,top=0.9)

                    for i, t, f, yerr, sim_v, sim_f, sim_t in zip(
                        stim, expT, expF, expSTD, simV, fluorTrace, sim_time
                    ):
                        if voltageOn:
                            [tmp] = ax2.plot(
                                sim_t,
                                sim_v,
                                label=f"Sim.",
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

                    ax1.patch.set_visible(False)
                    if voltageOn:
                        ax1.set_zorder(ax2.get_zorder() + 1)
                        ax2.set_xlim((100, 500))
                        leg = ax2.legend(
                            [
                                plotObjects_ax2[0],
                            ],
                            ["Sim"],
                            title=gl.vm,
                            title_fontsize=10,
                            loc="upper right",
                            edgecolor=self.returnColor("model"),
                            handler_map={str: LegendTitle({"fontsize": 10})},
                            handlelength=1.5,
                            handletextpad=0.5,
                            borderpad=0.5,
                        )
                        leg.get_title().set_color(self.returnColor("model"))
                        for legend in leg.get_lines():
                            legend.set_color(self.returnColor("model"))

                        plt.setp(leg.texts, color=self.returnColor("model"))
                    leg = ax1.legend(
                        [
                            plotObjects[0],
                            plotObjects[1],
                        ],
                        ["Sim.", "Exp."],
                        title=r"$\bf{Fluorescence}$",
                        title_fontsize=10,
                        loc="upper left",
                        edgecolor=self.returnColor("fluor"),
                        handler_map={str: LegendTitle({"fontsize": 10})},
                        ncol=3,
                        columnspacing=0.5,
                        handlelength=1.5,
                        handletextpad=0.5,
                        borderpad=0.5,
                    )
                    for legend in leg.get_lines():
                        legend.set_color("black")

                    handles, labels = ax1.get_legend_handles_labels()

                    copied_handles = [copy(h) for h in handles]

                    for h in leg.legend_handles:
                        try:
                            h.set_facecolor("white")
                            h.set_edgecolor("black")
                        except AttributeError:
                            pass

                    ax1.set_xlabel(gl.ms)
                    ax1.set_ylabel(gl.fluor)
                    if voltageOn:
                        _, ylim_value = gl.lim_d_volt  # mv
                        ax1.set_ylim((0, ylim_value * -1 / 10))
                    else:
                        ax1.set_ylim(gl.lim_fluor)
                    if voltageOn:
                        ax2.set_ylabel(gl.d_volt)
                        ax2.set_ylim((0, ylim_value))
                        ax_objs ={ax2: "model", ax1: "fluor"} 
                    else:
                        ax_objs = {ax1:'fluor'}
                    for axObj, label in ax_objs.items():
                        axObj.tick_params(axis="y", colors=self.returnColor(label))
                        axObj.yaxis.label.set_color(self.returnColor(label))

                    if correctArtifact:
                        Fname += "_correctedArtifact"

                    top = ax1.get_position().y1
                    right = ax1.get_position().x1
                    left = ax1.get_position().x0
                    center = (ax1.get_position().y1 + ax1.get_position().y0) / 2
                    color_pos = [
                        ("left", "tab:green", "1 stim."),
                        ("center", "tab:orange", "5 stim."),
                        ("right", "tab:blue", "10 stim."),
                    ]
                    for pos, color, xlabel in color_pos:
                        fig.text(
                            locals()[pos],
                            top + 0.015,
                            xlabel,
                            color=color,
                            fontsize=plt.rcParams["axes.labelsize"],
                            ha=pos,
                            va="bottom",
                            fontweight="bold",
                        )
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


    def plotExpFit_combined(self):
        if rank != 0:
            return
        intermediary_files = os.listdir(os.path.join("intermediaryData"))
        All_fits = {}

        key_order = ['K$^+$','Accumulation','GABA$_A$R','NMDAR']
        def determine_tag(tag):
            if 'PAP' in tag:
                if 'GABAR' in tag:
                    return 'GABA$_A$R'
                else:
                    return 'NMDAR'
            else:
                if 'forced' in tag:
                    return 'Accumulation'
                else:
                    return 'K$^+$'
        def determine_id(tag):
            if 'GABA' in tag or 'NMDA' in tag:
                if 'GABA' in tag:
                    return (1,0) 
                else:
                    return (1,1) 
            else:
                if 'Accum' in tag:
                    return (0,1) 
                else:
                    return (0,0) 



        for f in intermediary_files:
            if "fitExpDepolarization" in f and str(self.seed) in f:
                tag = determine_tag(f)
                print(f)
                with open(os.path.join("intermediaryData", f), "rb") as handle:
                    AllCells = pickle.load(handle)
                All_fits[tag]= AllCells

        color = {10: "tab:green", 5: "tab:orange", 1: "tab:blue"}
        All_fits = {k: All_fits[k] for k in key_order if k in All_fits}
        fig = plt.figure(figsize=gl.figsize_halfh)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.15)
        gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.5,hspace=0.5)
        plotObjects_ax2 = []

        for i, (tag, all_cells) in enumerate(All_fits.items()):
            if i == 0:
                sharex = None
                sharey = None
            else:
                sharex = original_ax
                sharey = original_ax
            x,y = determine_id(tag)
            ax = fig.add_subplot(gs[x,y],sharex=sharex,sharey=sharey)
            if i == 0:
                original_ax = ax

            ax.set_title(tag,loc='left')
            for cells in all_cells:
                initStep = self.get_initStep(cells)
                [tmp] = ax.plot(
                    list(cells.time)[initStep:],
                    list(cells.vPAP)[initStep:],
                    linestyle="-",
                    color=color[cells.SpikeNum]
                )
                plotObjects_ax2.append(tmp)
                ax.set_ylabel(gl.volt)
                ax.set_xlabel(gl.ms)


        top = original_ax.get_position().y1
        right = ax.get_position().x1
        left = original_ax.get_position().x0
        center = (left + right) / 2
        color_pos = [
            ("left", "tab:green", "1 stim."),
            ("center", "tab:orange", "5 stim."),
            ("right", "tab:blue", "10 stim."),
        ]
        for pos, color, xlabel in color_pos:
            fig.text(
                locals()[pos],
                top + 0.045,
                xlabel,
                color=color,
                fontsize=plt.rcParams["axes.labelsize"],
                ha=pos,
                va="bottom",
                fontweight="bold",
            ) 


        Fname = 'Combined_fit_exp_voltage'
        plt.savefig(f"../results/paperRes/{Fname}.pdf")


  


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

    def plot_GluT_experiment(self, ax, cell):
        if os.path.isfile(os.path.join("Data", "glut_somato.csv")):
            df = pd.read_csv(os.path.join("Data", "glut_somato.csv"))
            ax.axhline(min(df["i"]), 0, 1, linestyle="--", color="grey")
        else:
            ax.axhline(-129.14, 0, 1, linestyle="--", color="grey")


    def combine_distance_analysis_plots(self):
        if rank != 0:
            return
        curr_tag = self.tag
        # split tag with longer string first
        for splitter in ["_NoGlu", "_Glu", "_GABAR", "_GABA", "_NMDAR"]:
            curr_tag = self.split_and_remove(splitter, curr_tag)

        found_files = 0
        found_cells = []
        mask = [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
        ]
        for i, chan in enumerate(["_Glu", "_NMDAR", "_GABAR_NoGlu_GABA"]):
            tmp_tag = ["distance_analysis"] + curr_tag[:-1] + [chan, curr_tag[-1]]
            tmp_tag = "".join(tmp_tag)
            path = os.path.join("intermediaryData", f"{tmp_tag}.pickle")
            if os.path.isfile(path):
                print("loading", path)
                with open(path, "rb") as handle:
                    AllCells = pickle.load(handle)
                found_cells.append(AllCells)

                if found_files == 0:
                    plt.cla()
                    plt.clf()
                    ax2 = None
                    ax = plt.figure(figsize=gl.figsize_panel).gca()
                ax, ax2 = self.plot_shell(
                    AllCells,
                    ax,
                    ax2=ax2,
                    mask=[shell for cell, shell in mask if cell == i],
                )
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
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"combined_minAmplitude_{tmp_tag}shellcomp.pdf",
                )
            )
            plt.cla()
            plt.clf()
            plt.close()
            fig = plt.figure(figsize=gl.figsize_distCurr)
            emptyrow = 1
            cutoff = 9
            gs = fig.add_gridspec(
                nrows=cutoff + emptyrow + 1, ncols=6, hspace=0.5, wspace=0.8
            )
            axes = []
            total_current = [0] * 3
            ax_inset_select = []
            shell_blowup = 3
            ratio_plot = True
            for i, all_cells in enumerate(found_cells):
                cell_id = i
                if i != 0:
                    sharey = axes[0][0]
                else:
                    sharey = None
                axes.append(
                    [
                        fig.add_subplot(
                            gs[0, 2 * cell_id : 2 * (cell_id + 1)], sharey=sharey
                        )
                    ]
                )

                for i in range(1, cutoff + 1):
                    if i != 1:
                        sharey = axes[0][1]
                    else:
                        sharey = None

                    if (cell_id, i) in mask:
                        axes[cell_id].append([])

                    else:
                        axes[cell_id].append(
                            fig.add_subplot(
                                gs[i + emptyrow, 2 * cell_id : 2 * (cell_id + 1)],
                                sharex=axes[cell_id][0],
                                sharey=sharey,
                            )
                        )

                if cell_id == 0:
                    if ratio_plot:
                        ratio_traces = {}

                    ax_inset_select.append([fig.add_subplot(gs[1:3, 3:5])])
                    ax_inset_select.append(
                        [
                            fig.add_subplot(
                                gs[4:6, 3:5],
                                sharey=ax_inset_select[0][0],
                                sharex=ax_inset_select[0][0],
                            )
                        ]
                    )
                    for i in range(2):
                        ax_inset_select[i].append(ax_inset_select[i][0].twinx())

                    for i in range(2):
                        ax_inset_select[i][0].set_ylabel(
                            gl.clampI + " " + gl.unit_pA, labelpad=-13
                        )
                        ax_inset_select[i][1].set_ylabel(
                            gl.sigma_glt + " " + gl.unit_pA,
                            labelpad=-10,
                            color=self.returnColor("GluT", words=True),
                        )
                    # share twin x
                    ax1 = ax_inset_select[1][1]
                    ax2 = ax_inset_select[0][1]
                    ax1.sharey(ax2)
                    for ax_insets in ax_inset_select:
                        for ax_inset in ax_insets:
                            for spine in ax_inset.spines.values():
                                spine.set_visible(True)
                                spine.set_linewidth(1.5)
                            ax_inset.set_facecolor("white")

                for cells in all_cells:
                    for cell in cells:
                        axes[cell_id][0].set_xlim((145, 170))
                        initStep = self.get_initStep(cell, shift=50)
                        avg = list(cell.VClampI)[initStep]
                        shell_id = cell.shell
                        if shell_id > cutoff:
                            if shell_id == cell.total_shell:
                                grey = "0.5"
                                for loc in ["top", "right", "left", "bottom"]:
                                    axes[cell_id][0].spines[loc].set_visible(False)
                                    if loc == "left":
                                        # axes[cell_id][0].yaxis.set_ticks_position('left')
                                        if cell_id != 0:
                                            for label in axes[cell_id][
                                                0
                                            ].get_yticklabels():
                                                label.set_visible(False)
                                        for label in axes[cell_id][0].get_xticklabels():
                                            label.set_visible(False)
                                axes[cell_id][0].tick_params(
                                    axis="y", color=grey, labelcolor=grey
                                )
                                axes[cell_id][0].tick_params(
                                    axis="x", color=grey, labelcolor=grey
                                )

                                for spine in axes[cell_id][0].spines.values():
                                    spine.set_color(grey)

                                axes[cell_id][0].plot(
                                    list(cell.time)[initStep:],
                                    np.array(cell.VClampI)[initStep:] - avg,
                                    color="grey",
                                )
                                if cell_id == 0:
                                    axes[0][0].set_ylabel(
                                        gl.clampI + gl.unit_pA,
                                        color="grey",
                                        linespacing=0.8,
                                        labelpad=-1,
                                    )
                                    self.plot_GluT_experiment(axes[0][0], cell)
                            break
                        if (cell_id, shell_id) in mask:
                            continue

                        if (cell_id, shell_id) not in mask:
                            total_current[cell_id] += (
                                min(np.array(list(cell.VClampI)[initStep:]))
                                - list(cell.VClampI)[initStep]
                            )
                        axes[cell_id][shell_id].spines["right"].set_visible(False)
                        axes[cell_id][shell_id].spines["top"].set_visible(False)
                        # axes[cell_id][shell_id].set_ylim(gl.lim_currSoma)

                        current = "fake_current"
                        if cell_id == 0:
                            current = "iGluT"
                        elif cell_id == 1:
                            current = "iNMDA"
                        elif cell_id == 2:
                            current = "iGABA"
                        else:
                            wMessage(f"no current set for {cell_id=}")

                        if hasattr(cell, current) and (cell_id, shell_id) not in mask:
                            min_amp = -1 * np.min(getattr(cell, current))
                            axes[cell_id][shell_id].plot(
                                list(cell.time)[initStep:],
                                (np.array(getattr(cell, current))[initStep:]) / min_amp,
                                color=self.returnColor(current[1:]),
                            )
                            axes[cell_id][shell_id].plot(
                                list(cell.time)[initStep:],
                                (np.array(cell.VClampI)[initStep:] - avg) / min_amp,
                                color="black",
                            )

                            if ratio_plot:
                                if cell_id == 0:
                                    ratio_traces[shell_id] = (
                                        list(cell.time)[initStep:],
                                        (
                                            (np.array(cell.VClampI)[initStep:] - avg)
                                            / np.array(
                                                getattr(cell, current)[initStep:]
                                            )
                                        ),
                                    )

                            if cell_id == 0 and shell_id in [1, shell_blowup]:
                                ax = axes[cell_id][shell_id]
                                xmin, xmax = ax.get_xlim()
                                if shell_id == 1:
                                    ax_inset = ax_inset_select[0]

                                else:
                                    ax_inset = ax_inset_select[1]

                                ax_inset[0].set_ylim(gl.lim_min_amp_bs)
                                ax_inset[1].set_ylim(
                                    np.array(ax_inset[0].get_ylim()) * 10
                                )
                                for i in range(2):
                                    ax_inset[i].set_xlim(xmin + 5, xmax - 15)
                                    # ax_inset[i].xaxis.set_major_locator(
                                    #    MaxNLocator(integer=True)
                                    # )
                                ax_inset[1].plot(
                                    list(cell.time)[initStep:],
                                    np.array(getattr(cell, current))[initStep:],
                                    color=self.returnColor(current[1:]),
                                )
                                ax_inset[1].tick_params(
                                    axis="y", colors=self.returnColor(current[1:])
                                )
                                for label in ax_inset[1].get_yticklabels():
                                    label.set_color(
                                        self.returnColor(current[1:], words=True)
                                    )

                                ax_inset[1].spines["right"].set_color(
                                    self.returnColor(current[1:])
                                )
                                ax_inset[1].spines["right"].set_linewidth(2)
                                ax_inset[0].plot(
                                    list(cell.time)[initStep:],
                                    (np.array(cell.VClampI)[initStep:] - avg),
                                    color="black",
                                )
                                x1 = ax.get_xlim()[1]
                                y0 = ax_inset[0].get_ylim()[0]
                                y1 = ax_inset[0].get_ylim()[1]

                                origin = (-0.5, -0.25)
                                len_x = 2.1
                                len_y = 1.35

                                con1 = ConnectionPatch(
                                    xyA=(x1, 1),
                                    coordsA=ax.transData,  # top-left of zoom box
                                    xyB=(origin[0], origin[1] + len_y),
                                    coordsB=ax_inset[0].transAxes,  # top-left of inset
                                    color="grey",
                                    linestyle="--",
                                )
                                x1 = ax.get_xlim()[1]

                                con2 = ConnectionPatch(
                                    xyA=(x1, -1),
                                    coordsA=ax.transData,  # bottom-right of zoom box
                                    xyB=origin,
                                    coordsB=ax_inset[
                                        0
                                    ].transAxes,  # bottom-right of inset
                                    color="grey",
                                    linestyle="--",
                                )

                                fig.add_artist(con1)
                                fig.add_artist(con2)

                                rect = Rectangle(
                                    origin,
                                    len_x,
                                    len_y,
                                    transform=ax_inset[0].transAxes,
                                    facecolor="white",
                                    edgecolor="black",
                                    fill=True,
                                    lw=1.5,
                                    clip_on=False,
                                )
                                rect.set_path_effects(
                                    [
                                        pe.SimplePatchShadow(offset=(4, -4), alpha=0.4),
                                        pe.Normal(),
                                    ]
                                )
                                ax_inset[0].add_patch(rect)

                            # if cell_id == 1 and hasattr(cell, "iGluT"):
                            #    axes[cell_id][1].plot(
                            #        list(cell.time)[initStep:],
                            #        list(cell.iGluT)[initStep:],
                            #        color=self.returnColor("GluT"),
                            #    )
                            # if cell_id == 0:
                            #    axes[cell_id][1].set_ylabel(gl.curr)
                            # else:
                            #    for label in axes[cell_id][1].get_yticklabels():
                            #        label.set_visible(False)
                            # for loc in ["top", "right"]:
                            #    axes[cell_id][1].spines[loc].set_visible(False)
                            # axes[cell_id][1].set_xlabel(gl.ms)

                        if cell_id == 0:

                            def get_origin_y(ax):
                                x0 = ax.get_xlim()[0]

                                _, fig_y = fig.transFigure.inverted().transform(
                                    ax.transData.transform((x0, 0))
                                )
                                return fig_y

                            bottom = (
                                axes[cell_id][shell_id].get_position().y1
                                + axes[cell_id][shell_id].get_position().y0
                            ) / 2
                            left = axes[cell_id][shell_id].get_position().x0
                            right = axes[cell_id][shell_id].get_position().x1
                            fig.text(
                                left - 0.07,
                                bottom + 0.01,
                                f"Shell {shell_id}",
                                color="grey",
                                ha="center",
                                va="center",
                                rotation=90,
                                fontsize=plt.rcParams["axes.labelsize"],
                            )

                        if shell_id != cutoff:
                            axes[cell_id][shell_id].spines["bottom"].set_visible(False)
                            axes[cell_id][shell_id].spines["left"].set_visible(False)
                            axes[cell_id][shell_id].tick_params(axis="y", left=False)
                            axes[cell_id][shell_id].tick_params(axis="x", bottom=False)

                            for label in axes[cell_id][shell_id].get_xticklabels():
                                label.set_visible(False)
                            for label in axes[cell_id][shell_id].get_yticklabels():
                                label.set_visible(False)

                        else:
                            # axes[cell_id][shell_id].spines["bottom"].set_visible(
                            #    False
                            # )
                            # axes[cell_id][shell_id].spines["left"].set_visible(
                            #    False
                            # )
                            # axes[cell_id][shell_id].tick_params(
                            #    axis="y", left=False
                            # )
                            # axes[cell_id][shell_id].tick_params(
                            #    axis="x", bottom=False
                            # )

                            # if cell_id != 0:
                            #    for label in axes[cell_id][shell_id].get_xticklabels():
                            #        label.set_visible(False)
                            # for label in axes[cell_id][shell_id].get_yticklabels():
                            #    label.set_visible(False)
                            axes[cell_id][shell_id].set_xlabel(gl.ms)

            xlim_raw = axes[0][-1].get_xlim()
            ylim_raw = axes[0][-1].get_ylim()

            top = axes[0][0].get_position().y1
            # bottom = axes[0][-1][1].get_position().y0
            nt_pos = [axes[0][0], axes[1][0], axes[2][0]]
            nt_labels = ["GLT-1", "NMDAR", "GABA$_A$R"]
            i = 0
            for label, ax in zip(nt_labels, nt_pos):
                pos = ax.get_position().x0
                fig.text(
                    pos + 0.01,
                    top + 0.01,
                    label,
                    color=self.returnColor(label),
                    fontsize=plt.rcParams["axes.labelsize"],
                    ha="left",
                    va="bottom",
                    fontweight="bold",
                )

                i += 1

            left = axes[0][0].get_position().x0
            center = (axes[0][1].get_position().y1 + axes[0][-1].get_position().y0) / 2
            fig.text(
                left - 0.11,
                center,
                gl.free("Normalized Current\t$I/\min(I_{channel})$"),
                fontsize=plt.rcParams["axes.labelsize"],
                ha="center",
                va="center",
                rotation=90,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"combined_currPlots_{tmp_tag}shellcomp.pdf",
                )
            )
            if ratio_plot:
                plt.cla()
                plt.clf()
                sorted_dict = dict(sorted(ratio_traces.items()))
                solid_colors = PAPModel.gen_colors(cell.total_shell)
                plt.figure(figsize=gl.figsize_distCurr_panel)
                for id, (t, ratio) in sorted_dict.items():
                    t = np.array(t)
                    plt.scatter(
                        id,
                        max(ratio[(t > 150) & (t < 151)]),
                        label=id,
                        color=solid_colors[id - 1],
                        edgecolor=solid_colors[cell.total_shell - id - 1],
                    )
                plt.ylim((0, 0.06))
                plt.xlabel(gl.shell_num)
                plt.ylabel(gl.free(f"{gl.clampI}/{gl.sigma_glt}"))
                # plt.xlim((150, 151))
                plt.gca().xaxis.set_major_locator(
                    MaxNLocator(nbins="auto", integer=True)
                )
                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"combined_ratioPlots{tmp_tag}shellcomp.pdf",
                    )
                )

        else:
            plt.cla()
            plt.clf()
            plt.close("all")

    def plot_shell(self, results, ax, ax2=None, mask=None):
        resList = []
        shell_num = []
        for cells in results:
            for cell in cells:
                VCI = list(cell.VClampI)
                total_shell = cell.total_shell
                if mask is not None and len(mask) > 0:
                    if cell.shell not in mask:
                        resList.append(min(VCI) - VCI[-1])
                        shell_num.append(cell.shell)
                else:
                    resList.append(min(VCI) - VCI[-1])
                    shell_num.append(cell.shell)

        shell_num, resList = zip(
            *[(i, j) for i, j in sorted(zip(shell_num, resList)) if i < total_shell]
        )
        resList = np.array(resList)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(shell_num, resList)
        ax.set_xlabel(gl.shell_num)
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
        ax2.set_xlim((5, 9))
        ax2.set_ylim((-0.1, 0.01))
        return ax, ax2

    @read_data
    def distance_analysis(self, shell_range=10):
        self.addChannelTag()
        syn_count = 4e1
        # clustered input into n sections
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

        # 1 PAP is 3 sections(synapses) for seed 1
        PAPLen = funcArgs[-1]["PAPLen"]
        if self.NMDAR and self.GluStim:
            funcArgs[-1]["multiple"] = self.optNMDAR * syn_count
        else:
            funcArgs[-1]["multiple"] = None
        if self.GluT and self.GluStim:
            funcArgs[-1]["GluTrans"] = self.optGluT
            # value matches avg/std count per synapse
        if self.GABAR and self.GabaStim:
            funcArgs[-1]["GABACount"] = (
                self.optGABAR * syn_count
            )  # GABA alread calculates per section
            funcArgs[-1]['dt'] /= 50
        else:
            funcArgs[-1]["GABACount"] = 0

        ccList = ["shell"]
        iterations = [i for i in range(1, shell_range + 1)]
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
                    {"force_print_progress": True},
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
            ax = plt.figure(figsize=gl.figsize_panel).gca()
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
            plt.xlabel(gl.hz)
            plt.xticks(
                range(0, int(spikeFreqMax / spikeFreqStep) + 1, 2),
                np.arange(0, spikeFreqMax + 1, spikeFreqStep * 2),
            )
            plt.yticks(
                range(int(spikeNumMax / spikeNumStep) + 1),
                np.arange(0, spikeNumMax + 1, spikeNumStep),
            )
            plt.colorbar(label=gl.d_volt_short, ticks=np.arange(0, 30, 5), extend="max")
            plt.clim(gl.clim_volt)
            plt.tight_layout()
            plt.savefig(
                os.path.join("../results/paperRes", f"FreqComparison{self.tag}.pdf")
            )

    def fit_fluor(self):
        iterations = [2,4,6,8,10,20,100,500] 
        self.interval_spacing = iterations
        # Calculate the number of iterations each process will handle
        iterations_per_process = len(iterations) // size

        # Adjust the range for the last process
        if len(iterations) % size == 0:
            remaining_iterations = 0
            minimum = rank * iterations_per_process
            maximum = (rank + 1) * iterations_per_process
        elif rank >= size - len(iterations) % size:
            remaining_iterations = 1
            minimum = rank * (iterations_per_process + remaining_iterations) - (
                size - len(iterations) % size
            )
            maximum = (rank + 1) * (iterations_per_process + remaining_iterations) - (
                size - len(iterations) % size
            )
        else:
            remaining_iterations = 0
            minimum = rank * iterations_per_process
            maximum = (rank + 1) * iterations_per_process

        results = []
        h.load_file("stdgui.hoc")
        soma = h.Section(name='soma')
        soma.L = 20      # length in um
        soma.diam = 20   # diameter in um
        soma.cm = 1      # membrane capacitance
        soma.Ra = 100    # axial resistance

        soma.insert('GEVI')
        vclamp = h.VClamp(soma(0.5))

        t = h.Vector().record(h._ref_t)
        v = h.Vector().record(soma(0.5)._ref_dF_GEVI)


        for index in range(minimum, maximum):
            interval = iterations[index]

            v_init = -65
            vclamp.dur[0] = 10   # duration of first phase (ms)
            vclamp.amp[0] = v_init   # target voltage (mV)

            vclamp.dur[1] = interval 
            vclamp.amp[1] = v_init + 100   # target voltage (mV)
            vclamp.dur[2] = 10
            vclamp.amp[2] = v_init   # target voltage (mV)

            h.finitialize(v_init)
            h.continuerun(20 + interval)
            results.append((interval,list(t).copy(),np.array(v)-v_init))
            print(max(list(t)),max(list(v)))
        results = comm.gather(results, root=0)

        comm.Barrier()
        if rank == 0:
            self.plot_exp_comparison_fluor(results)

    def plot_exp_comparison_fluor(self,results):
        max_v = 0
        df = pd.read_csv(os.path.join("./Data","archlight_delay.csv"))
        df['std'] -= df['fraction']

        fig,ax = plt.subplots(figsize=gl.figsize_panel)
        fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.15)
        ax_inset = ax.inset_axes([0.15,0.35,0.3,0.3]) 
        ax.errorbar(range(len(df)),df['fraction'],yerr=df['std'],label='Exp.',fmt='-o',ecolor='lightgray',color='lightgray',zorder=-1)

        for cells in results:
            for cell in cells:
                interval,t,v = cell
                if max(v) > max_v:
                    max_v = max(v)

        x = []
        y = []
        for cells in results:
            for cell in cells:
                interval,t,v = cell
                x.append(self.interval_spacing.index(interval))
                y.append(max(v)/max_v*100)
                if interval == 500:
                    ax_inset.plot(t,v,color='black')
                    ax_inset.axhline(max(v),linestyle='--',color='black')
                    ax_inset.set_xlim(0,510)
                    df = pd.read_csv(os.path.join("./Data","archlight_500_trace.csv"))
                    df['t'] -= df['t'].iloc[0]
                    df['t'] += 10
                    ax_inset.scatter(df['t'],df['v'],color='lightgray')
                    ax_inset.set_title('Fluor. Response\n(500 ms)')
                    ax_inset.set_xlabel(gl.ms)
        ax.plot(x,y,label='Sim.',color='black')
        ax.set_xticks(range(len(self.interval_spacing)),labels=self.interval_spacing)
        ax.set_ylabel(gl.fluor_frac)
        ax.set_xlabel(f'Voltage clamp duration {gl.unit_ms}')
        ax.legend()

        plt.savefig(os.path.join("../results/paperRes",'fluor_relation.pdf'))


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
        plt.tight_layout()


if __name__ == "__main__":
    if size == 3:
        mprint("exp fit")
        seed = size
        testBools = [True, False]
        use_tau = False
        for PAP in testBools:
            for forcedAccum in testBools:
                exp = procedure(1, 0)
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
                    )
  
                    bounds = [(0, 1e5), (-1e5, 1e5), (0.3, 50)]

                    if forcedAccum and use_tau:
                        initParms += (exp.spillOverLen,6.31084066, exp.spillOverSlowing)
                        initParms = list(initParms)
                        # initParms += [
                        #    5.11574074e-01,
                        #    5.00000000e01,
                        #    1.02314815e02,
                        #    9.89814815e02,
                        # ]
                        # initParms = [
                        #    5.38257517e-04,
                        #    9.78283818e-04,
                        #    1.93825396e00,
                        #    4.56907947e01,
                        #    6.79258427e02,
                        #    9.98155414e03,
                        # ]
                        # initParms[0] = 0
                        # initParms[1] = 1
                        initParms = tuple(initParms)

                        bounds += [
                            (5.8, 10000),
                            (0, 1e10),
                        ]
                    else:
                        initParms += (0.3,)

                    if not use_tau:
                        initParms = tuple(initParms[:2])
                        bounds = list(bounds[:2])

                kwargs = {
                    "use_tau": use_tau,
                    "autosave": False,
                    "PAP": PAP,
                    "showFig": False,
                }

                skipsave = False
                exp.fitExpDepolarization(
                    initParms,
                    showFig=True,
                    PAP=PAP,
                    skipsave=skipsave,
                    use_tau=use_tau,
                )
                if not exp.foundFitExperiment and skipsave:
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
                        kwargs["autosave"] = True
                        exp.fitExpDepolarization(res.x, **kwargs)
                exp.foundfitExperiment = False
        exp.plotExpFit_combined()
    elif size in [6,5,2, 4]:
        mprint("running bathExp")
        exp = procedure(1, 0)
        exp.dt *=2
        if size == 2:
            exp.bathExperiment()
        else:
            exp.bathExperiment(invivo=True)

    elif size > 6:
        exp = procedure(1, 0)
        exp.runFitCaliburation()
        exp.fit_fluor()
    else:
        exp = procedure(1, 0)
        exp.uptakeRatio()
        #exp.measureRi()

        # print('nothing set; try MPI -n 2 for kbath; MPI -n 3 for exp fit')
