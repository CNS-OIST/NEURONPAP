from mpi4py import MPI
import pickle
from astrocyte import PAPModel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import utils
import numpy as np
from utils import *
from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class procedure():
    
    def measureCond(self,fName):
        if not os.path.isfile(os.path.join('intermediaryData',f"{fName}.pickle")):
            voltList = np.arange(0, 100, 5)
            IV = {}
            IVslow = {}
            IVfast = {}
            for volt in voltList:
                print(volt)
                iS, iF = PAPModel(volt)
                IVslow[volt] = iS
                IVfast[volt] = iF
                print(iS, iF)
            IV["slow"] = IVslow
            IV["fast"] = IVfast
            with open(os.path.join('intermediaryData',f"{fName}.pickle"), "wb") as handle:
                pickle.dump(IV, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join('intermediaryData',f"{fName}.pickle"), "rb") as handle:
                IV = pickle.load(handle)

        for mode in IV.keys():
            print(mode)
            plt.cla()
            plt.clf()
            I = list(IV[mode].values())
            V = list(IV[mode].keys())
            popt, pcov = curve_fit(eq, V, I)
            x = np.linspace(-40, 100, 50)
            print(popt)
            plt.plot(V, I, label="model")
            plt.plot(x, eq(x, *popt), label=f"{popt[0]}x+{popt[1]}")
            plt.legend()
            plt.savefigf(os.path.join('../results/codeSortTest',"{fName}{mode}.pdf"))


    def multiChannel(self,itr=100):
        dList = []
        for i in range(1, itr + 1):
            sim = PAPModel(40, multiple=i, mode=0)
            sim.run()
            dList.append(plot(".") - sim.getRMP())
        with open(os.path.join('intermediaryData',f"dList.pickle"), "wb") as handle:
            pickle.dump(dList, handle, protocol=pickle.HIGHEST_PROTOCOL)
        plt.cla()
        plt.clf()
        plt.scatter(range(1, itr + 1), dList)
        plt.savefig(os.path.join('../results/codeSortTest',"patchXDepolar.pdf"))


    def multiDistance(self,x, read=False):
        somaSize, bLen, bWid, PAPWid, bNum = x
        dList = []
        cList = []
        vList = []
        if read:
            with openf(os.path.join('intermediaryData',"ballStick.pickle"), "rb") as handle:
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
                        "Ko": 5,
                        "kir2": 1e8,
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
                    with open(os.path.join('intermediaryData',"resultsParallel.pickle"), "wb") as handle:
                        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    for index, res in enumerate(results):
                        i, j = iterations[index]
                        dList.append(i)
                        cList.append(j)
                        for i, rIndi in enumerate(res):
                            if i == 0:
                                RMP =  rIndi.getRMP()
                            vSomaList.append(
                                max(np.array(rIndi.vSoma))
                            )
                            vPAPList.append(
                                max(np.array(rIndi.vPAP))
                            )
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
                            Ko=8.5,
                            Glu=True,
                        )
                        sim.run()
                        vSomaList.append(max(np.array(sim.vSoma) - sim.getRMP(), key=abs))
                        sim = PAPModel(
                            multiple=j,
                            bLen=i,
                            bWid=bWid,
                            currentClamp=1,
                            somaSize=somaSize,
                            mode=0,
                            bNum=int(bNum),
                            PAPWid=PAPWid,
                            Ko=8.5,
                            Glu=True,
                        )
                        sim.run()
                        vPAPList.append(max(np.array(sim.vPAP) - sim.getRMP(), key=abs))
            # plt.plot(np.array(range(len(vList[-1])))*PAPModel.dt,vList[-1],label=f'{current} pA')
            vList = [vSomaList, vPAPList]

            if not self.parallel or rank == 0:
                with open(os.path.join('intermediaryData',f"ballStick.pickle"), "wb") as handle:
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

                plt.savefig(f"./3Dplot{name}{j}.pdf")


    def measureRi(self,x):
        somaSize, bLen, bWid, PAPWid, bNum = x
        # Make a list for tested currents
        cList = np.arange(-80, 81, 20)
        vSomaList = []
        vPAPList = []
        funcArgs = []
        for current in cList:
            funcArgs.append(
                {
                    'currentClamp':current,
                    'bLen':bLen,
                    'bWid':bWid,
                    'somaSize':somaSize,
                    'mode':3,
                    'bNum':int(bNum),
                    'PAPWid':PAPWid,
                    'somaCheck':True,
                }
            )
            simSoma = PAPModel(**funcArgs[-1])
            simSoma.initialize()
            simSoma.run()
            vSomaList.append(list(simSoma.vSoma))

            funcArgs[-1]['somaCheck'] = False
            simPAP = PAPModel(**funcArgs[-1])
            simPAP.initialize()
            simPAP.run()
            vPAPList.append(list(simPAP.vPAP))
            # plt.plot(np.array(range(len(vSomaList[-1])))*PAPModel.dt,vSomaList[-1],label=f'{current} pA')

        vList, somaC = remove_nan_values([v[-1] for v in vSomaList], cList)
        if len(vList) > 1:
            somapopt, pcov = curve_fit(eq, somaC,vList)
            print(f"{abs(somapopt[0])} MOhm")
        else:
            somapopt = [float("inf")]
        # x = np.linspace(-600,600)
        # plt.plot(eq(x,*somapopt),x)
        # plt.legend()
        # plt.show()
        vList, PAPC = remove_nan_values([v[-1] for v in vPAPList], cList)
        if len(vList) > 1:
            PAPpopt, pcov = curve_fit(eq, PAPC,vList)
            print(f"{abs(PAPpopt[0])} MOhm")
        else:
            PAPpopt = [float("inf")]

        return abs(1 / somapopt[0] - 2.6)*0.1/2.6 + abs(
            1 / PAPpopt[0] - 1050
        )*0.9/1050  # soma input resistance score


    def singleRun(self,readHoc=True):

        for i in range(1):
            # single run
            funcArgs = []
            funcArgs.append(
                {
                    "bWid": 2,
                    "somaSize": 10,
                    "mode": 0,
                    "bLen": 10,
                    "PAPWid": 1.3,
                    "kir2":100,
                    "Glu":True,
                    "multiple":5,
                    "readHoc":readHoc
                }
            )
            cells = PAPModel(**funcArgs[-1])
            cells.initialize()
            cells.setK(Ko=8.5,mode='step',dur=50)
            cells.run()
            # initStep = int(cells.initTstop / cells.dt)
        initStep=0
        cells = cells.copyAttr()
        AllCells = comm.gather(cells, root=0)
        if rank == 0:
            for cell in AllCells:
                fig, ax = plt.subplots()
                
                ax.plot(list(cell.time), list(cell.KoPAP), label="PAP Ko")
                ax.plot(list(cell.time), list(cell.KoSoma), label="Soma Ko")
                ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                ax.plot(list(cell.time), list(cell.KiSoma), label="Soma Ki")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('[K] (mM)')
                ax.legend()

                ax2 = ax.inset_axes([0.2, 0.6, 0.3, 0.3])  # Define the position and size of the new subplot
                ax2.plot(list(cells.time)[initStep:],
                         list(cell.vPAP)[initStep:],
                         label="PAP")
                ax2.plot(list(cells.time)[initStep:],
                         list(cell.vSoma)[initStep:],
                         label="Soma")
                ax2.set_ylabel('Vm')
                ax2.set_xlabel('time')
                ax2.legend()
                
                plt.savefig(os.path.join('../results/codeSortTest',"KoCon.pdf"))

                plt.cla()
                plt.clf()
                fig, ax = plt.subplots()
                
                ax.plot(list(cell.time), list(cell.KoPAP), label="PAP Ko")
                ax.plot(list(cell.time), list(cell.KoSoma), label="Soma Ko")
                ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('[K] (mM)')
                ax.legend()

                ax2 = ax.inset_axes([0.2, 0.6, 0.3, 0.3])  # Define the position and size of the new subplot
                ax2.plot(list(cells.time), list(cell.ekPAP))
                ax2.set_ylabel('ek')
                ax2.set_xlabel('time')
                
                plt.savefig(os.path.join('../results/codeSortTest','ekPlot.pdf'))

                plt.cla()
                plt.clf()
                fig, ax = plt.subplots()
                
                ax.plot(list(cell.time), list(cell.iKPAP), label="ik PAP")
                ax.plot(list(cell.time), list(cell.iKSoma), label="ik Soma")
                ax.plot(list(cell.time), list(cell.iNMDA), label="iNMDA")
                ax.plot(list(cell.time), list(cell.iMemPAP), label="iMem PAP")
                ax.plot(list(cell.time), list(cell.iMemSoma), label="iMem Soma")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('Currents (nA)')
                ax.legend()

                ax2 = ax.inset_axes([0.6, 0.2, 0.3, 0.3])  # Define the position and size of the new subplot
                ax2.plot(list(cells.time), list(cell.vPAP),color='orange')
                ax2.set_ylabel('Vm')
                ax2.set_xlabel('time')
                
                plt.savefig(os.path.join('../results/codeSortTest','ikPlot.pdf'))

