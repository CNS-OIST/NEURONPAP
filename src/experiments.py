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
    leak =3e5
    optKir = 30
    optNMDAR = 30
    NMDAMax = 30
    NMDAStep = 1
    KirMax = 30
    KirStep = 1

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
        plt.savefig(os.path.join('../results/fullMorph',"patchXDepolar.pdf"))


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
                        "kir2": 0,
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
        cList = np.arange(-20, 21, 2)
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
                    'readHoc':True,
                    'somaCheck':True,
                    'Glu':False                    
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

        return abs(somapopt[0] - 2.6)*0.1/2.6 + abs(
            PAPpopt[0] - 1050
        )*0.9/1050  # soma input resistance score

    def branchAttenuation(self):
        funcArgs = []
        funcArgs.append(
            {
                'mode':0,
                'ComplexMorph':True,
                'readHoc':True,
                'Glu':True,
                'dt':0.1,
                'NMDAdelay':0.01,
                'naleak':self.leak,
                'clleak':self.leak,
                'kir2':self.optKir,
                'multiple':self.optNMDAR,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize()
        cells.setK(Ko=0.5,delay = 0)
        # cells.setK(Ko=ko)
        cells.run()
        initStep = int(cells.initTstop / cells.dt)
        cells = cells.copyAttr()

        timeVoltageArray = list()
        for x in cells.branchAtten:
            coord = list(x)[initStep:]
            timeVoltageArray.append(coord)
        timeVoltageArray = np.array(timeVoltageArray).T
        print(timeVoltageArray)
        # Plot the array using a heatmap
        plt.imshow(timeVoltageArray,
                   cmap='viridis',
                   interpolation='none',
                   aspect='auto'
                   )
        plt.colorbar()

        # Show the plot
        plt.savefig("branchAtten.pdf")

        # plt.clf()
        # plt.cla()
        # plt.plot(cells.vPAP)
        # plt.show()
            
        
    def plotPAPs(self):
        funcArgs = []
        funcArgs.append(
            {
                'mode':0,
                'ComplexMorph':True,
                'readHoc':True,
                'Glu':True,
                'dt':0.1,
                'NMDAdelay':0.01,
                'naleak':self.leak,
                'clleak':self.leak,
                'kir2':self.optKir,
                'multiple':self.optNMDAR,
            }
        )
        cells = PAPModel(**funcArgs[-1])
        cells.initialize(video=True)
        cells.setK(Ko=0.5)
        cells.run(video=True)
        

    def singleRun(self,readHoc=True):
        koConc = list(range(2,6))
        AllCells = []        
        for g,kirCount in enumerate(range(0,10,1)):
            AllCells.append(list())
            for i,ko in enumerate(koConc):
                # single run
                funcArgs = []
                funcArgs.append(
                    {
                        # 'currentClamp':0,
                        # 'voltageClamp':20,
                        'mode':0,
                        'ComplexMorph':True,
                        'bNum':1,
                        'readHoc':True,
                        'Glu':True,
                        "bWid": 2.160,
                        "somaSize": 7.597,
                        "bLen": 8.237,
                        "PAPWid": 1.21,
                        "multiple":self.optNMDAR, # Maximum conductance of model is equal to 50 single channels
                        "NMDAdelay":5*i*ms,
                        "kir2":kirCount,
                        'dt':0.1
                        # "readHoc":readHoc
                    }
                )
                cells = PAPModel(**funcArgs[-1])
                cells.initialize()
                # cells.setK(Ko=ko,mo2de='step',dur=100,delay = i*20*ms)
                cells.setK(Ko=ko,delay = i*5*ms)
                # cells.setK(Ko=ko)
                cells.setK(Ko=ko,delay=10*ms)
                cells.run()
                initStep = int(cells.initTstop / cells.dt)
                cells = cells.copyAttr()

                if size < 2:
                    AllCells[g].append(cells)
                if size > 1:
                    AllCells = comm.gather(cells, root=0)
                if rank == 0:
                    self.plotIKSeries(AllCells)
                break
            break

    def plotIKSeries(self,AllCells):
        for cells in AllCells:
            for cell in cells:
                initStep = int(cell.initTstop / cell.dt)
                fig, ax = plt.subplots()
                ax.plot(list(cell.time)[initStep:], list(cell.KoSoma)[initStep:], label="Soma Ko")
                # ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                # ax.plot(list(cell.time), list(cell.KiSoma), label="Soma Ki")
                ax.plot(list(cell.time)[initStep:], list(cell.KoPAP)[initStep:], label=f"PAP Ko")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('[K] (mM)')
                ax.legend()

                ax2 = ax.inset_axes([0.7, 0.3, 0.3, 0.3])  # Define the position and size of the new subplot
                ax2.plot(list(cell.time)[initStep:],
                         list(cell.vSoma)[initStep:],
                         label="Soma")
                ax2.plot(list(cell.time)[initStep:],
                         list(cell.vPAP)[initStep:],
                         label=f"PAP Ko")
                ax2.set_ylabel('Vm')
                ax2.set_xlabel('time')

                plt.savefig(os.path.join('../results/fullMorph',f"KoCon{cell.GENEDict['kir2']}_{cell.multiple}.pdf"))

                fig, ax = plt.subplots()
                # ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                # ax.plot(list(cell.time), list(cell.KiSoma), label="Soma Ki")
                ax.plot(list(cell.time)[initStep:], list(cell.NaoPAP)[initStep:], label=f"PAP Ko")
                ax.plot(list(cell.time)[initStep:], list(cell.CloPAP)[initStep:], label=f"PAP Ko")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('Conc. (mM)')
                ax.legend()
                plt.savefig(os.path.join('../results/fullMorph',f"NaCon{cell.GENEDict['kir2']}_{cell.multiple}.pdf"))

                plt.cla()
                plt.clf()
                fig, ax = plt.subplots()

                ax.plot(list(cell.time)[initStep:], list(cell.KoPAP)[initStep:], label="PAP Ko")
                ax.plot(list(cell.time)[initStep:], list(cell.KoSoma)[initStep:], label="Soma Ko")
                # ax.plot(list(cell.time), list(cell.KiPAP), label="PAP Ki")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('[K] (mM)')
                ax.legend()

                ax2 = ax.inset_axes([0.7, 0.4, 0.3, 0.3])  # Define the position and size of the new subplot
                ax2.plot(list(cell.time)[initStep:], list(cell.ekPAP)[initStep:])
                # ax2.plot(list(cell.time), list(cell.enaPAP))
                plt.legend()
                ax2.set_ylabel('e_rev')
                ax2.set_xlabel('time')

                plt.savefig(os.path.join('../results/fullMorph',f"ekPlot{cell.GENEDict['kir2']}_{cell.multiple}.pdf"))

                plt.cla()
                plt.clf()
                fig, ax = plt.subplots()

                ax.plot(list(cell.time)[initStep:], list(cell.iKPAP)[initStep:], label="ik PAP")
                ax.plot(list(cell.time)[initStep:], list(cell.iKSoma)[initStep:], label="ik Soma")
                if hasattr(cell,"iNaPAP"):
                    ax.plot(list(cell.time)[initStep:], list(cell.iNaPAP)[initStep:], label="iNa PAP",color="orange")
                if hasattr(cell,"iClPAP"):
                    ax.plot(list(cell.time)[initStep:], list(cell.iClPAP)[initStep:], label="iCl PAP",color="green")
                if cell.Glu:
                    ax.plot(list(cell.time)[initStep:], list(cell.iNMDA)[initStep:], label="iNMDA",color='purple')
                # if hasattr(cell, "iMemPAP"):
                #     ax.plot(list(cell.time), list(cell.iMemPAP), label="iMem PAP")
                # if hasattr(cell, "iMemSoma"):
                #     ax.plot(list(cell.time), list(cell.iMemSoma), label="iMem Soma")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel('Currents (nA)')
                ax.legend(loc="lower center")

                ax2 = ax.inset_axes([0.75,0.2, 0.2, 0.2])  # Define the position and size of the new subplot
                if cell.Glu:
                    ax2.plot(list(cell.time)[initStep:], list(cell.iNMDA)[initStep:], label="iNMDA",color='purple')
                    ax2.set_ylabel('Currents (nA)')
                else:
                    ax2.plot(list(cell.time)[initStep:], list(cell.vPAP)[initStep:], label="PAP")
                    ax2.set_ylabel('Vm')
                # ax2.set_xlabel('time')
                ax3 = ax.inset_axes([0.75, 0.55, 0.2, 0.2])  # Define the position and size of the new subplot
                ax3.plot(list(cell.time)[initStep:], list(cell.iKPAP)[initStep:], label="ik PAP")
                ax3.set_ylabel('Currents (nA)')



                plt.savefig(os.path.join('../results/fullMorph',f"ikPlot{cell.GENEDict['kir2']}_{cell.multiple}.pdf"))
                plt.close('all')


    def channelComparison(self):        
        # Calculate the number of iterations for all parm sets
        iterations = comm.bcast(get_iter(self.KirMax,
                                         self.KirStep,
                                         self.NMDAMax,
                                         self.NMDAStep
                                         ),
                                root=0
                                )
        # # Adjust the range for the last process

        comm.Barrier()
        funcArgs = []
        funcArgs.append(
            {
                'mode':0,
                'readHoc':True,
                'Glu':True,
                'ComplexMorph':True,
                'NMDAdelay':0.01,
                'naleak':self.leak,
                'clleak':self.leak,
                'dt':0.1
            }
        )
        # make sure that funcParms is in the correct order of whatever iterations spits out
        # results are collected only on rank 0
        results = parallizeFor(
            iterations,
            [PAPModel],
            funcArgs,
            ["kir2", "multiple"],
            [["initialize", "setK","run"]],
            [[{}, {"Ko":0.5},{}]]
        )
        comm.Barrier()
        
        if rank == 0:
            seedtag = results[0][0].seed
            with open(
                    os.path.join(
                        'intermediaryData',
                        f"resultsParallel{seedtag}.pickle"
                    ), "wb") as handle:
                pickle.dump(results,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            self.plotHeatmap(results,tag=seedtag)
            totResults = []
            path = os.path.join(
                os.path.abspath('intermediaryData'),
                "resultsParallel"
                )
            resFiles = glob.glob(path+"*.pickle")
            for res in resFiles:
                with open(
                        os.path.join(
                            'intermediaryData',
                            res
                        ), "rb") as handle:
                    results = pickle.load(handle)
                totResults += results
            self.plotIKSeries(results)
            self.plotHeatmap(totResults,divedend=len(resFiles))

    def plotHeatmap(self,results,tag="",divedend=1):
        imArray = np.zeros((int(self.KirMax/self.KirStep) + 1,
                            int(self.NMDAMax/self.NMDAStep) + 1))

        for res in results:
            imArray[int(res[0].GENEDict['kir2']/self.KirStep),
                    int(res[0].multiple/self.NMDAStep)] += max(res[0].vPAP) - res[0].RMP
        imArray /= divedend
        plt.imshow(imArray,
                   cmap='viridis',
                   origin='lower',
                   interpolation='nearest',
                   aspect='equal'
                   )
        plt.xticks(range(int(self.NMDAMax/self.NMDAStep) + 1),
                   np.arange(0,int(self.NMDAMax/self.NMDAStep) + 1,1)*self.NMDAStep)
        plt.yticks(range(int(self.KirMax/
                             self.KirStep) + 1),np.arange(0,int(self.KirMax/self.KirStep) + 1,1)*self.KirStep)
        plt.ylabel('Kir Channel')
        plt.xlabel('NMDAR Channel')
        plt.colorbar(label = 'values',ticks=np.arange(0,50,10),extend='max')
        plt.clim((0,50))
        plt.savefig(f'FullComparison{tag}.pdf')
        plt.cla()
        plt.clf()

        imArray = np.zeros((int(self.KirMax/self.KirStep) + 1,
                            int(self.NMDAMax/self.NMDAStep) + 1))        

        for res in results:
            imArray[int(res[0].GENEDict['kir2']/self.KirStep),int(res[0].multiple/self.NMDAStep)] += res[0].RMP
        imArray /= divedend
        plt.imshow(imArray,
                   cmap='viridis',
                   origin='lower',
                   interpolation='nearest',
                   aspect='equal'
                   )
        plt.xticks(range(int(self.NMDAMax/self.NMDAStep) + 1),np.arange(0,int(self.NMDAMax/self.NMDAStep) + 1,1)*self.NMDAStep)
        plt.yticks(range(int(self.KirMax/self.KirStep) + 1),np.arange(0,int(self.KirMax/self.KirStep) + 1,1)*self.KirStep)
        plt.ylabel('Kir Channel')
        plt.xlabel('NMDAR Channel')
        plt.colorbar(label = 'values',ticks=np.arange(-90,-20,10),extend='max')
        plt.clim((-90,-20))
        plt.savefig(f'FullRMP{tag}.pdf')
        plt.cla()
        plt.clf()

        imArray = np.zeros((int(self.KirMax/self.KirStep) + 1,
                            int(self.NMDAMax/self.NMDAStep) + 1))

        for res in results:
            imArray[int(res[0].GENEDict['kir2']/self.KirStep),int(res[0].multiple/self.NMDAStep)] += max(res[0].vSoma) - res[0].RMP
        imArray /= divedend
        plt.imshow(imArray,
                   cmap='viridis',
                   origin='lower',
                   interpolation='nearest',
                   aspect='equal'
                   )
        plt.xticks(range(int(self.NMDAMax/self.NMDAStep) + 1),np.arange(0,int(self.NMDAMax/self.NMDAStep) + 1,1)*self.NMDAStep)
        plt.yticks(range(int(self.KirMax/self.KirStep) + 1),np.arange(0,int(self.KirMax/self.KirStep) + 1,1)*self.KirStep)
        plt.ylabel('Kir Channel')
        plt.xlabel('NMDAR Channel')
        plt.colorbar(label = 'values',ticks=np.arange(0,10,1),extend='max')
        plt.clim((0,10))
        plt.savefig(f'FullSoma{tag}.pdf')



