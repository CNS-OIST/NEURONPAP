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
from scipy.stats import f_oneway,ttest_rel
import json

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class plotFigures:
    colorDict = {
        'NMDAR':'steelblue',
        'GluT':'lightblue',
        'iK':'orange',
        'Soma':'deepskyblue',
        'PAP':'forestgreen',
        'Na':'gold',
        'Cl':'chocolate'
    }

    def returnColor(self,key):
        for typeName in self.colorDict.keys():
            if typeName in key:
                return self.colorDict[typeName]
        else:
            eMessage(f'Color not found for {key}')
                
    def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
        """ 
        Annotate barplot with p-values.

        :param num1: number of left bar to put bracket over
        :param num2: number of right bar to put bracket over
        :param data: string to write or number for generating asterixes
        :param center: centers of all bars (like plt.bar() input)
        :param height: heights of all bars (like plt.bar() input)
        :param yerr: yerrs of all bars (like plt.bar() input)
        :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
        :param barh: bar height in axes coordinates (0 to 1)
        :param fs: font size
        :param maxasterix: maximum number of asterixes to write (for very small p-values)
        """

        if type(data) is str:
            text = data
        else:
            # * is p < 0.05
            # ** is p < 0.005
            # *** is p < 0.0005
            # etc.
            text = ''
            p = .05

            while data < p:
                text += '*'
                p /= 10.

                if maxasterix and len(text) == maxasterix:
                    break

            if len(text) == 0:
                text = 'n. s.'

        lx, ly = center[num1], height[num1]
        rx, ry = center[num2], height[num2]

        if yerr:
            ly += yerr[num1]
            ry += yerr[num2]

        ax_y0, ax_y1 = plt.gca().get_ylim()
        dh *= (ax_y1 - ax_y0)
        barh *= (ax_y1 - ax_y0)

        y = max(ly, ry) + dh

        barx = [lx, lx, rx, rx]
        bary = [y, y+barh, y+barh, y]
        mid = ((lx+rx)/2, y+barh)

        plt.plot(barx, bary, c='black')

        kwargs = dict(ha='center', va='bottom')
        if fs is not None:
            kwargs['fontsize'] = fs

        plt.text(*mid, text, **kwargs)
            
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
                "clleak": self.leak,
                "kir2": self.optKir,
                "multiple": self.optNMDAR,
                "seed": self.seed,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        cells.setK(Ko=self.ko)
        cells.run(video=True)

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

    def plotIKSeries(self, AllCells, zoom=False):
        for cells in AllCells:
            for cell in cells:
                if (
                    not zoom
                    and cell.multiple == self.optNMDAR
                    and cell.GENEDict["kir2"] == self.optKir
                ) or (
                    not zoom
                    and cell.multiple == 0
                ):
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
                # ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                # ax.plot(list(cell.time), list(cell.KiSoma), label="Soma Ki")
                ax.plot(
                    list(cell.time)[initStep:],
                    list(cell.KoPAP)[initStep:],
                    label=f"PAP",
                    color=self.returnColor("PAP"),
                )
                for x in list(range(150,251,10))[:self.stimCount]:
                    ax.arrow(
                        x,
                        0.5,
                        0,
                        -0.5,
                        color='black',
                        width=0.001,
                        head_width=0.4,
                        head_length=0.2,
                        length_includes_head=True                        
                    )
                ax.set_ylim((0, 9))
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("extracellular [K] (mM)")
                ax.legend()

                # ax2 = ax.inset_axes([0.7, 0.3, 0.3, 0.3])  # Define the position and size of the new subplot
                # ax2.plot(list(cell.time)[initStep:],
                #          list(cell.vSoma)[initStep:],
                #          label="Soma")
                # ax2.plot(list(cell.time)[initStep:],
                #          list(cell.vPAP)[initStep:],
                #          label=f"PAP Ko")
                # ax2.set_ylabel('Voltage')
                # ax2.set_xlabel('time')
                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 20))

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"KoCon{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                fig, ax = plt.subplots()
                # ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                # ax.plot(list(cell.time), list(cell.KiSoma), label="Soma Ki")
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
                    linestyle='--',
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
                        linestyle='--',
                    )
                # ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("Voltage (mV)")
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()

                # ax2 = ax.inset_axes([0.7, 0.4, 0.3, 0.3])  # Define the position and size of the new subplot
                # ax2.plot(list(cell.time)[initStep:], list(cell.ekPAP)[initStep:])
                # # ax2.plot(list(cell.time), list(cell.enaPAP))
                # plt.legend()
                # ax2.set_ylabel('e_rev')
                # ax2.set_xlabel('time')
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
                # ax.plot(list(cell.time)[initStep:], list(cell.iKSoma)[initStep:], label="ik Soma")
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
                if hasattr(cell, "iGluT") and self.GluT:
                    ax.plot(
                        list(cell.time)[initStep:],
                        list(cell.iGluT)[initStep:],
                        label="iGluT",
                        color=self.returnColor("GluT"),
                    )
                # if hasattr(cell, "iMemPAP"):
                #     ax.plot(list(cell.time), list(cell.iMemPAP), label="iMem PAP")
                # if hasattr(cell, "iMemSoma"):
                #     ax.plot(list(cell.time), list(cell.iMemSoma), label="iMem Soma")
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("Currents at PAP (pA)")
                # ax.set_ylim([-1e-3,1e-3])
                if zoom or self.stimCount > 1:
                    ax.legend(loc="lower left")
                else:
                    ax.legend(loc="lower right")

                # ax2 = ax.inset_axes([0.75,0.2, 0.2, 0.2])  # Define the position and size of the new subplot
                # if cell.Glu:
                #     ax2.plot(list(cell.time)[initStep:], list(cell.iNMDA)[initStep:], label="iNMDA",color='purple')
                #     ax2.set_ylabel('Currents (pA)')
                # else:
                #     ax2.plot(list(cell.time)[initStep:], list(cell.iKPAP)[initStep:], label="ik PAP")
                #     ax2.set_ylabel('Currents (pA)')

                # ax3 = ax.inset_axes([0.75, 0.55, 0.2, 0.2])  # Define the position and size of the new subplot
                # ax3.plot(list(cell.time)[initStep:], list(cell.vPAP)[initStep:], label="PAP")
                # ax3.set_ylabel('Voltage')
                # ax2.set_xlabel('time')
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
                # if hasattr(cell, "iMemPAP"):
                #     ax.plot(list(cell.time), list(cell.iMemPAP), label="iMem PAP")
                # if hasattr(cell, "iMemSoma"):
                #     ax.plot(list(cell.time), list(cell.iMemSoma), label="iMem Soma")
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("Currents at Soma (pA)")
                # ax.set_ylim([-1e-3,1e-3])
                ax.legend(loc="lower right")

                # ax2 = ax.inset_axes([0.75,0.2, 0.2, 0.2])  # Define the position and size of the new subplot
                # if cell.Glu:
                #     ax2.plot(list(cell.time)[initStep:], list(cell.iNMDA)[initStep:], label="iNMDA",color='purple')
                #     ax2.set_ylabel('Currents (pA)')
                # else:
                #     ax2.plot(list(cell.time)[initStep:], list(cell.iKPAP)[initStep:], label="ik PAP")
                #     ax2.set_ylabel('Currents (pA)')

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
                # ax2.set_xlabel('time')
                if zoom:
                    ax.set_xlim((initStep * cell.dt, initStep * cell.dt + 30))

                plt.savefig(
                    os.path.join(
                        "../results/paperRes",
                        f"iSomaPlot{cell.GENEDict['kir2']}_{cell.comparecount}{self.tag}.pdf",
                    )
                )

                plt.close("all")

    def plotHeatmap(self, results, tag="", divedend=1):
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
                max(res[0].vPAP) - res[0].RMP
            )
        cmap = 'magma'
        if self.GluT:
            cmap = 'virdis'
        
        
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
        plt.ylabel("# of Kir Channel")
        if self.NMDAR:
            plt.xlabel("# of NMDAR Channel")
        elif self.GluT:
            plt.xlabel("GluT Channel")
        plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 50, 10), extend="max")
        plt.clim((0, 50))
        plt.savefig(os.path.join("../results/paperRes", f"FullComparison{tag}.pdf"))
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
        plt.ylabel("Kir Channel")
        if self.NMDAR:
            plt.xlabel("NMDAR Channel")
        elif self.GluT:
            plt.xlabel("GluT Channel")
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
        plt.ylabel("# of Kir Channel")
        if self.NMDAR:
            plt.xlabel("# of NMDAR Channel")
        elif self.GluT:
            plt.xlabel("GluT Channel")
        plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 10, 1), extend="max")
        plt.clim((0, 10))

        plt.savefig(os.path.join("../results/paperRes", f"FullSoma{tag}.pdf"))
                
        

class procedure(plotFigures):
    leak = 3e5
    optKir = 120
    optNMDAR = 10
    optGluT = 1
    channelCompareMax = 50
    channelCompareStep = 5
    KirMax = 400
    KirStep = 40
    seed = int()
    ko = float()
    tag = str()
    NMDAR = True
    GluT = False
    GluStim = True
    KStim = True
    stimdelay = 0
    dt = 0.1
    PAPCount = 1
    stimCount = 1
    freq = 100
    ek = None
    PAPLen=0.3
    peakLen = None

    def __init__(self, seed, ko):
        self.seed = seed
        self.ko = ko
        self.tag = "_" + str(self.seed) + f"_{self.ko:.3f}"

    def addChannelTag(self):
        self.tag = ''
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
        plt.scatter(range(1, itr + 1), dList,color='black')
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
                            #     plt.show()n

                # vSomaList = comm.gather(vSoma, root = 0)
                # vPAPList = comm.gather(vPAP, root = 0)
                # dList = comm.gather(d, root = 0)
                # cList = comm.gather(c, root = 0)

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
            # plt.plot(np.array(range(len(vList[-1])))*PAPModel.dt,vList[-1],label=f'{current} pA')
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

    def measureRi(self, x):
        somaSize, bLen, bWid, PAPWid, bNum = x
        # Make a list for tested currents
        cList = np.arange(-20, 21, 2)
        vSomaList = []
        vPAPList = []
        funcArgs = []
        for current in cList:
            funcArgs.append(
                {
                    "currentClamp": current,
                    "bLen": bLen,
                    "bWid": bWid,
                    "somaSize": somaSize,
                    "mode": 3,
                    "bNum": int(bNum),
                    "PAPWid": PAPWid,
                    "readHoc": True,
                    "somaCheck": True,
                    "Glu": False,
                }
            )
            simSoma = PAPModel(**funcArgs[-1])
            simSoma.initialize()
            simSoma.run()
            vSomaList.append(list(simSoma.vSoma))

            funcArgs[-1]["somaCheck"] = False
            simPAP = PAPModel(**funcArgs[-1])
            simPAP.initialize()
            simPAP.run()
            vPAPList.append(list(simPAP.vPAP))
            # plt.plot(np.array(range(len(vSomaList[-1])))*PAPModel.dt,vSomaList[-1],label=f'{current} pA')
        vList, somaC = remove_nan_values([v[-1] for v in vSomaList], cList)
        if len(vList) > 1:
            somapopt, pcov = curve_fit(eq, somaC, vList)
            print(f"{abs(somapopt[0])} MOhm")
        else:
            somapopt = [float("inf")]
        # x = np.linspace(-600,600)
        # plt.plot(eq(x,*somapopt),x)
        # plt.legend()
        # plt.show()
        vList, PAPC = remove_nan_values([v[-1] for v in vPAPList], cList)
        if len(vList) > 1:
            PAPpopt, pcov = curve_fit(eq, PAPC, vList)
            print(f"{abs(PAPpopt[0])} MOhm")
        else:
            PAPpopt = [float("inf")]

        return (
            abs(somapopt[0] - 2.6) * 0.1 / 2.6 + abs(PAPpopt[0] - 1050) * 0.9 / 1050
        )  # soma input resistance score

    def SomaVC(self):
        vClampList = np.arange(-95, -59, 5)
        vSomaList = []
        funcArgs = []
        plt.cla()
        plt.clf()
        for v in vClampList:
            funcArgs.append(
                {
                    "somaCheck": True,
                    "mode": 2,
                    "ComplexMorph": True,
                    "readHoc": True,
                    "dt": self.dt,
                    "naleak": self.leak,
                    "clleak": self.leak,
                    "kir2": self.optKir,
                    "multiple": self.optNMDAR,
                    "seed": self.seed,
                    "GluTrans": self.optGluT,
                }
            )
            simSoma = PAPModel(**funcArgs[-1])
            simSoma.initialize()
            simSoma.run()
            vSomaList.append(list(simSoma.vSoma))
            initStep = int((simSoma.initTstop - 10) / simSoma.dt)
            plt.plot(
                list(simSoma.time)[initStep:],
                list(simSoma.vSoma)[initStep:],
                label=f"{v} mV",
                color="black",
            )
        # plt.legend()
        plt.savefig(os.path.join("../results/paperRes", f"VoltageClampAstrocyte.pdf"))

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
                "clleak": self.leak,
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
        # cells.setK(Ko=ko)
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
        plt.imshow(
            timeVoltageArray, cmap="magma", interpolation="none", aspect="auto"
        )
        plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 20, 2), extend="max")
        plt.clim((0, 20))
        plt.xlabel("normalized distance")
        plt.ylabel("Time (ms)")
        # print(list(range(0,len(list(cells.branchAtten[0])[initStep:])+1,100)))
        plt.xticks(
            range(0, 11, 2), [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )  # float point generated by np.linspace
        plt.yticks(
            range(0, len(list(cells.branchAtten[0])[initStep:]) + 1, 100),
            np.arange(
                0,
                int(cells.time[-1] - cells.time[initStep]) + 1,
                100 * cells.dt,
                dtype=int                
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
                "clleak": self.leak,
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
        for kircount in [self.optKir,400]:
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
                        "clleak": self.leak,
                        "seed": self.seed,
                        "PAPLen":self.PAPLen,
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
                        for conc in np.append(np.array(0), np.logspace(-2, 1, 9))
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
                            plt.plot(
                                list(cell.KoPAP)[initStep:],
                                list(cell.vPAP)[initStep:],
                                label=f"{cell.Ko:.3f}",
                                color=cm.summer(i/len(results))
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
                                if kircount == self.optKir and chanCount == self.optNMDAR:
                                    plt.legend()
                                plt.ylabel("Voltage (mV)")
                                plt.xlabel("[K]o (mM)")
                                if self.PAPLen <= 0.3:
                                    plt.ylim((-90,-50))
                                plt.xlim((2,14))
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
                    # 'currentClamp':0,
                    # 'voltageClamp':20,
                    "mode": 0,
                    "ComplexMorph": True,
                    "bNum": 1,
                    "readHoc": True,
                    "Glu": True,
                    "kir2": self.optKir,
                    "clleak": self.leak,
                    "naleak": self.leak,
                    "dt": self.dt,
                    "seed": self.seed,
                    "stimdelay": 20 * ms
                    # "readHoc":readHoc
                }
            )
            if self.NMDAR:
                funcArgs[-1][
                    "multiple"
                ] = (
                    self.optNMDAR
                )  # Maximum conductance of model is equal to 50 single channels
            else:
                funcArgs[-1][
                    "multiple"
                ] = 0  # Maximum conductance of model is equal to 50 single channels
            if self.GluT:
                funcArgs[-1][
                    "GluTrans"
                ] = (
                    self.optGluT
                )  # Maximum conductance of model is equal to 50 single channels

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
        plt.scatter(ekList, depList,color='black')
        plt.ylabel("Relative Amplitude (mV)")
        plt.xlabel("ek (mV)")
        plt.savefig(os.path.join("../results/paperRes", "ekDepolarcomp.pdf"))

        plt.cla()
        plt.clf()
        plt.scatter(koList, depList,color='black')
        plt.ylabel("Relative Amplitude (mV)")
        plt.xlabel("extracellular [K+] (mM)")
        plt.savefig(os.path.join("../results/paperRes", "ekKODepolarcomp.pdf"))
        print(koList)

    def KOComp(self, papCount=10,koCond=4):
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
                tmpdt = self.dt
            elif i < 3:
                self.GluT = True
                self.NMDAR = True
                if i == 1:
                    # Kir OE
                    self.optKir = controlKir + 30
                    self.dt *= 0.1
                else:
                    self.dt = tmpdt
                    self.optKir = controlKir - 30
            else:
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
                    # 'currentClamp':0,
                    # 'voltageClamp':20,
                    "mode": 0,
                    "ComplexMorph": True,
                    "bNum": 1,
                    "readHoc": True,
                    "Glu": True,
                    "kir2": self.optKir,
                    "clleak": self.leak,
                    "naleak": self.leak,
                    "dt": self.dt,
                    # "readHoc":readHoc
                }
            )
            if self.NMDAR:
                funcArgs[-1][
                    "multiple"
                ] = (
                    self.optNMDAR
                )  # Maximum conductance of model is equal to 50 single channels
            else:
                funcArgs[-1][
                    "multiple"
                ] = 0  # Maximum conductance of model is equal to 50 single channels
            if self.GluT:
                funcArgs[-1][
                    "GluTrans"
                ] = (
                    self.optGluT
                )  # Maximum conductance of model is equal to 50 single channels

            comm.Barrier()
            if self.peakLen == None:
                self.peakLen = 2
            iterations = comm.bcast([(i,j) for j in [0.3, self.peakLen] for i in range(papCount)])
            ccList = ["seed","PAPLen"]
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
                cells = results
                AllCells += cells

        if rank == 0:
            resMat = np.zeros((koCond*2, papCount))
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
                            # GluT and NMDAR WT
                            k = 1
                            self.addChannelTag()
                        else:
                            # GluT KO and NMDAR WT
                            self.addChannelTag()
                            k = 2
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
                    self.tag = self.tag.split('_KOComp')[0]
                    self.tag += '_KOComp'
                    if cell.PAPLen > 0.3:
                        self.tag += f'spillover'
                    self.plotIKSeries([[cell]])
                    resMat[k][cell.seed] = max(cell.vPAP) - cell.RMP

            title = 'One-way ANOVA '
            for i in list(range(0,koCond*2,koCond)):
                ommit_cond = 1
                stat,pval = f_oneway(*resMat[i:i+koCond-ommit_cond]) # select based on control, OE, KD leave NMDAR out
                if i == koCond:
                    title += f'spillover p-value:{pval:.2E}'
                else:
                    title += f'constrained p-value:{pval:.2E}'
                        
            val_means = {
                'confined':[],
                'spillover':[]
            }
            val_sd = {
                'confined':[],
                'spillover':[]
            }
            category = ["Control", "Kir OE", "Kir KD", "NMDAR KO", "GluT KO", "NMDAR KO\nGluT KO"]
            category = category[:koCond]
            val_test = {}
            for i in range(len(resMat)):
                if i < koCond: # Number of KO conditions
                    dict_key = 'confined'
                    val_test[category[i]] = ttest_rel(resMat[i],resMat[i+koCond])
                else:
                    dict_key = 'spillover'
                    
                val_means[dict_key].append(np.nanmean(resMat[i]))
                val_sd[dict_key].append(np.nanstd(resMat[i]))

            width = 0.25
            multiplier = 0
            x = np.arange(len(category))
            fig, ax = plt.subplots(layout='constrained')
            fig.suptitle(title)
            color = ['orange','darkorange','gold','orange','orange','orange']
            color = color[:koCond]
            pattern = {'confined':'','spillover':'/'}
            for k,v in val_means.items():
                offset = width * multiplier
                rects = ax.bar(
                    x + offset,
                    v,
                    width,
                    yerr=val_sd[k],
                    label=k,
                    color=color,
                    hatch=pattern[k],
                    edgecolor = "black",
                )
                # ax.bar_label(rects,padding=3)
                multiplier+=1
            ax.axhline(val_means['confined'][0],linestyle='--')

            with open("ttest_res.json","w") as ofile:
                json.dump(val_test,ofile)
            # for k,v in val_test.items():
            #     if v.pvalue < 0.05:
            #         index = category.index(k)
            ax.set_ylabel("Voltage (mV)")
            ax.set_ylim(0,30)
            ax.set_xticks(x+ width/len(val_means.keys()),category)
            ax.legend(loc='upper left', ncols = 2)
            plt.savefig(
                os.path.join(
                    "../results/paperRes",
                    f"KO_maxDepolarComp_avg{papCount}_{self.tag}.pdf",
                )
            )
            
    def singleRun(self):
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        funcArgs.append(
            {
                # 'currentClamp':0,
                # 'voltageClamp':20,
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "readHoc": True,
                "Glu": True,
                "kir2": self.optKir,
                "clleak": self.leak,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed
                # "readHoc":readHoc
            }
        )
        if self.NMDAR:
            funcArgs[-1][
                "multiple"
            ] = (
                self.optNMDAR
            )  # Maximum conductance of model is equal to 50 single channels
        else:
            funcArgs[-1][
                "multiple"
            ] = 0  # Maximum conductance of model is equal to 50 single channels
        if self.GluT:
            funcArgs[-1][
                "GluTrans"
            ] = (
                self.optGluT
            )  # Maximum conductance of model is equal to 50 single channels

        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        # cells.setK(Ko=ko,mode='step',dur=100,delay = i*20*ms)
        # cells.setK(Ko=self.ko)
        # cells.setK(Ko=ko)
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
            self.plotIKSeries(AllCells)
            plt.plot(list(cells.time),list(cells.GluTGlu))
            plt.savefig('GlutamateTimecourse.pdf')


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
                "clleak": self.leak,
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
                iterations = [[i] for i in range(0,self.KirMax+1,self.KirStep)]
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

    def glutamateSpillOver(self):
        self.addChannelTag()
        iterations = np.concatenate((np.logspace(-1,1,num=19),np.array([self.PAPLen])))
        iterations = np.sort(iterations)
        iterations = comm.bcast([[i] for i in iterations])
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
                "clleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
                "kir2": self.optKir
            }
        )
        ccList = ["PAPLen"]
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
            vList = []
            sizeList = []
            controlIndex = None            
            for i,cells in enumerate(results):
                for cell in cells:
                    initStep = int((cell.initTstop - 10) / cell.dt)
                    plt.plot(
                        list(cell.time)[initStep:],
                        np.array(list(cell.vPAP)[initStep:])-cell.RMP,
                        label=f'{cell.PAPLen:.2f} um',
                        color= cm.BrBG(i/len(results)),
                    )
                    if cell.PAPLen == self.PAPLen:
                        controlV = max(list(cell.vPAP)[initStep:])-cell.RMP
                        controlIndex = i
                    vList.append(max(list(cell.vPAP)[initStep:])-cell.RMP)
                    sizeList.append(cell.PAPLen)
            plt.xlabel('time (ms)')
            plt.ylabel('Voltage (mV)')
            plt.savefig(
                os.path.join("../results/paperRes", f"GlutamateSpillOver{self.tag}.pdf")
            )
            plt.xlim((140,160))
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig(
                os.path.join("../results/paperRes", f"GlutamateSpillOver{self.tag}_zoom.pdf")
            )
            plt.cla()
            plt.clf()
            
            plt.scatter(
                sizeList,
                vList,
                cmap='BrBG',
                c=range(len(results))
            )
            # plot control as diamond
            if controlIndex != None:
                plt.scatter(
                    self.PAPLen,
                    controlV,
                    color=cm.BrBG(controlIndex/len(results)),
                    marker='D',
                    label='Confined',
                )
            maxIndex = vList.index(max(vList))
            plt.scatter(
                sizeList[maxIndex],
                vList[maxIndex],
                color=cm.BrBG(maxIndex/len(results)),
                marker='D',
                label='Spillover',
            )
            plt.legend()
            plt.ylim((0,22))
            plt.xlabel('Affected PAP length (um)')
            plt.ylabel('Peak Voltage (mV)')
            plt.savefig(
                os.path.join("../results/paperRes", f"GlutamateSpillOverMax{self.tag}.pdf")
            )
            self.peakLen = sizeList[int(np.argmax(vList))]
            cellList = [[[cell]] for cells in results for cell in cells if cell.PAPLen in [self.PAPLen,10,self.peakLen]]
            for cell in cellList:
                self.tag = self.tag.split('_PAPLen')[0]
                self.tag += f'_PAPLen{cell[0][0].PAPLen}'
                self.plotIKSeries(cell)

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
                "clleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "stimdelay": self.stimdelay,
                "PAPCount": self.PAPCount,
                "multiple": self.optNMDAR,
                "GluTrans": self.optGluT,
            }
        )
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
            plt.ylabel("# of Kir Channel")
            plt.xlabel("extracellular [K] (mM)")
            plt.colorbar(label="Voltage (mV)", ticks=np.arange(0, 20, 2), extend="max")
            plt.clim((0, 20))

            plt.savefig(
                os.path.join("../results/paperRes", f"FullPotassium{self.tag}.pdf")
            )

    def SCeq(self,x,a,l,c):
        return a*np.exp(-x/l) + c
    
    def spaceConstant(self):
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        funcArgs.append(
            {
                # 'currentClamp':0,
                "voltageClamp":-60*mV,
                "ComplexMorph": True,
                "readHoc": True,
                "Glu": False,
                "kir2": self.optKir,
                "GluTrans":self.optGluT,
                "clleak": self.leak,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed
                # "readHoc":readHoc
            }
        )
        cells = PAPModel(**funcArgs[-1])
        if rank == 0:
            LambdaList, VList, LenList = cells.spaceConstant()
            plt.scatter(LenList,LambdaList,label='Section Space Constant')
            plt.title(f'avg:{sum(LambdaList)/len(LambdaList)}')
            
            plt.savefig('SpaceConstant.pdf')
            plt.cla()
            plt.clf()
            plt.scatter(LenList,VList)
            popt,pcov = curve_fit(self.SCeq,LenList,VList)
            plt.plot(LenList,self.SCeq(np.array(LenList),*popt),label=f'{popt[0]}exp(-x/{popt[1]}+{popt[2]}')
            plt.legend()
            plt.savefig('spaceConstantFit.pdf')
            


    def optDepolarizationSearch(self,Ko,optmV=19.2):
        # add multispike ek clamp
        self.addChannelTag()
        # print(self.tag)
        AllCells = []
        # single run
        funcArgs = []
        funcArgs.append(
            {
                # 'currentClamp':0,
                "mode": 0,
                "ComplexMorph": True,
                "bNum": 1,
                "readHoc": True,
                "Glu": False,
                "kir2": self.optKir,
                "clleak": self.leak,
                "naleak": self.leak,
                "dt": self.dt,
                "seed": self.seed,
                "multiple":self.optNMDAR,
                "GluTrans":self.optGluT
                # "readHoc":readHoc
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        print(f"trial:{Ko}")
        cells.setK(Ko=float(Ko))
        cells.run()
        return abs(max(list(cells.vPAP))-cells.RMP - optmV)
