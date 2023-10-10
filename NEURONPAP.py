"""
To Do:
[x] TDQM

"""

from mpi4py import MPI
from neuron import h, load_mechanisms
from neuron.units import mM,mV
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pickle
import os
import pandas as pd
from results.utils.visualize import func as plotRes
from mpl_toolkits import mplot3d
import math
import sys
import time
import tqdm


comm = MPI.COMM_WORLD
size = comm.Get_size()
if size > 1:
    parallel = True
else:
    parallel = False
sys.stdout.flush()

if parallel:
    rank = comm.Get_rank()
    print(f'rank{rank} initialized')
    tag_end = 23
    comm.bcast(tag_end,root=0)

comm.Barrier()


class GENEManipulation():
    compartment = object()
    GENE = dict()

    def kir4Change(self,multiple):
        for seg in self.compartment:
            seg.kir4.Pkir = seg.kir4.Pkir/multiple
            if seg.kir4.Pkir > 1.0:
                seg.kir4.Pkir = 1.0

    def kir2Change(self,multiple):
        for seg in self.compartment:
            seg.kir2.gkbar = seg.kir2.gkbar*multiple
    
class GENExpression(GENEManipulation):
    compartment = object()
    GENE = dict()

    def __init__(self,compartment,GENE):
        if GENE == None:
            return
        self.compartment = compartment

        if type(GENE) == dict:
            self.GENE = GENE
        else:
            print(f'Error: GENE{GENE} should be dict type')
            return
        self.checkExpressionStatement()
        for gName,xfold in GENE.items():
            self.changeExpression(gName,xfoldXpression=xfold)

    def changeExpression(self,gName,xfoldXpression= None):
        if gName in self.GENE.keys():
            if xfoldXpression == None:
                xfoldXpression = self.GENE[gName]
            if hasattr(self,f'{gName}Change'): # check if method is implemented
                exec(f'self.{gName}Change({xfoldXpression})')
            else:
                print(f'No {gName} skipped')
        

    def checkExpressionStatement(self):
        for gName, multiple in self.GENE.items():
            if type(gName) != str and type(multiple) != float:
                print(f'Error: GENE{GENE} should be name and float')
                return

class PAPModel():
    tstop = 1000
    dt = 0.01 #0.00001
    celsius = 37
    v_init = -85
    pap = object()
    somaSize = 10 # Soma Size
    bLen = 30 # Branch Size
    bWid = 3
    papWid = 0.02
    branches = []

    # NMDA parms
    Tau2_0 = float()
    Tau3_0 = float()
    A2 = float()
    A3 = float()
    B2 =float()
    B3 = float()
    DELTA = float()
    SynWeight = float()

    # K Parms
    defaultKo = 2.5
    

    def __init__(self,
                 papWid=0.02,
                 bWid=3,
                 bNum=1,
                 bLen=30,
                 voltageClamp=40,
                 somaSize=10,
                 currentClamp=2,
                 multiple=1,
                 mode=2,
                 somaCheck=True,
                 Glu=False,
                 initialKo=2.5,
                 **kwargs
                 ):

        # Load NEURON GUI and parameters
        from neuron import h
        h.load_file("stdgui.hoc")
        h.load_file("params.hoc")
        # print('loaded files')

        # Set simulation parameters
        h.tstop = self.tstop
        h.dt = self.dt
        h.celsius = self.celsius
        h.v_init = self.v_init
        # print('set sim parms')

        # set morphology parameters
        self.somaSize = somaSize
        self.bLen = bLen
        self.bWid = bWid
        self.bNum = bNum
        self.papWid = papWid

        # Build Morphology
        self.morph()
        print('built morphology')
        sys.stdout.flush()

        # Clamp settings
        if mode > 2:
            #Step Current
            if somaCheck:
                ic = h.IClamp(self.soma(0.5))
            else:
                ic = h.IClamp(self.pap(0.5))
            ic.dur = self.tstop
            ic.delay = 0 # ms starts with glutamate
            ic.amp = currentClamp * 0.001  # nA current injection (1 pA)
        elif mode > 1:
            vc = h.VClamp(self.soma(0.5))
            vc.dur[0] = h.tstop
            vc.amp[0] = voltageClamp  # mV depolarization (dV = 20)

        elif mode > 0:
            #Impulse Current
            ic = h.IClamp(self.pap(0.5))
            ic.dur = 0.002
            ic.delay = 10 # ms starts with glutamate
            ic.amp = currentClamp * 0.001  # nA current injection (1 pA)
        print('clamp experiment setup')
        sys.stdout.flush()


        self.readParameters()
        self.NMDAs = []
        self.NCs = []
        for i in range(multiple):
            # Create the synaptic NMDA conductance
            stim = h.NetStim(self.pap(0.5))
            stim.interval = 1
            stim.number = 1
            stim.start = 10
            stim.noise = 0
            self.NMDAs.append(self.nmda())
            if Glu:
                self.NCs.append(h.NetCon(stim, self.NMDAs[-1])) # Must be in outer later with python address allocated
                self.NCs[-1].weight[0] = self.SynWeight
                self.NCs[-1].delay = 0

            print('placed NMDAR')
            sys.stdout.flush()
        # h.init()
        # print('initialized')
        # sys.stdout.flush()

        
        for sec in h.allsec():
            if sec == self.pap:
                # print('set pap Ko')
                self.setK(sec, initialKo)
            else:
                self.setK(sec,self.defaultKo)
            GENExpression(sec,kwargs)
        # print('set GENE manipulation')
        self.record(sNMDA = self.NMDAs[-1])

        

    def run(self,printRes=False):
        h.init()
        h.run()
        if printRes:
            self.printRec()
        self.cleanMorphology()
        print('ran simulation')
        sys.stdout.flush()

    def cleanMorphology(self):
        for sec in h.allsec():
            h.delete_section(sec=sec)
        self.branches = []

    def astroMem(self,compartment):
        # add astrocyte properties
        compartment.Ra = 100
        compartment.cm = 0.8
        compartment.insert('pas')
        compartment.e_pas = -85
        compartment.g_pas = 1/11150
        self.channels(compartment)

    def channels(self,compartment):
        # insert relevant channels
        compartment.insert('kir2')
        compartment.insert('twik')
        compartment.insert('K_acc')
        compartment.insert('kleak')
        compartment.insert('kdifl')
        h.ki0_k_ion = 70 * mM # Tweaked for RMP -85
        h.ko0_k_ion = 2.5 * mM # Global concentration
        compartment.ki = 70 * mM
        compartment.ko = 2.5 * mM
        compartment.ek = -90 * mV
        # compartment.ki = 130 * mM
        # compartment.ko = 8.5 * mM # STEPHEN F. 1988 for seizure induction

    def morph(self,isolate=False,printTopology=False):
        # Access the PAP object
        self.pap = h.Section(name="PAP")

        # Set astrocyte leaf membrane parameters
        self.pap.L = 0.3
        self.pap.diam = self.papWid
        self.pap.nseg = 1
        self.astroMem(self.pap)

        # h.psection(sec=self.pap)
        if not isolate:
            # create Soma
            self.soma = h.Section(name='soma')
            self.soma.diam = self.somaSize
            self.soma.L = self.somaSize
            # self.soma.diam = 10 # Agulhon 2008 Neuron
            # self.soma.L = 10
            self.astroMem(self.soma)
            # h.psection(sec=self.soma)
            
            for i in range(self.bNum):
                # Create the branch (not included in the original hoc file)
                self.branches.append(h.Section(name=f'branch{i}'))
                self.branches[-1].L = self.bLen
                self.branches[-1].diam = self.bWid
                self.branches[-1].nseg = 10
                self.astroMem(self.branches[-1])
                # h.psection(sec=self.branches[-1])
                #Connect
                self.branches[-1].connect(self.soma(0.5))
                if i == 0:
                    self.pap.connect(self.branches[-1])
        if printTopology:
            h.topology()

            
    def readParameters(self,fDir='./results/optimize'):        
        # Load optimization parameters        
        if not parallel or rank == 0:
            wFile = h.File()
            wFile.ropen(f"{fDir}/optT2.dat")
            self.Tau2_0 = wFile.scanvar()
            self.A2 = wFile.scanvar()
            self.B2 = wFile.scanvar()
            wFile.close()

            wFile = h.File()
            wFile.ropen(f"{fDir}/optT3.dat")
            self.Tau3_0 = wFile.scanvar()
            self.A3 = wFile.scanvar()
            self.B3 = wFile.scanvar()
            wFile.close()

            self.DELTA = 0 # Lalo 2006 J. Neuroscience

            wFile = h.File()
            wFile.ropen(f"{fDir}/optW.dat")
            self.SynWeight = wFile.scanvar()
            wFile.close()

        if parallel:
            comm.bcast(self.Tau2_0,root=0)
            comm.bcast(self.A2,root=0)
            comm.bcast(self.B2,root=0)
            comm.bcast(self.Tau3_0,root=0)
            comm.bcast(self.A2,root=0)
            comm.bcast(self.B3,root=0)
            comm.bcast(self.DELTA,root=0)
            comm.bcast(self.SynWeight,root=0)

            

    def nmda(self):        
        sNMDA = h.Exp5NMDA(self.pap(0.5))
        sNMDA.tau2_0 = self.Tau2_0
        sNMDA.a2 = self.A2
        sNMDA.b2 = self.B2
        sNMDA.tau3_0 = self.Tau3_0
        sNMDA.a3 = self.A3
        sNMDA.b3 = self.B3
        sNMDA.delta = self.DELTA

        return sNMDA

    def record(self,sNMDA=None):
        #Save Stuff
        if sNMDA != None:
            self.iNMDA = h.Vector()
            self.iNMDA.record(sNMDA._ref_i)

            self.iFile = h.File("iFile.dat")
            self.iFile.wopen("iFile.dat")

        self.iMem = h.Vector()
        self.iMem.record(self.pap(0.5)._ref_i_pas)

        self.iFileMem = h.File("iFileMem.dat")
        self.iFileMem.wopen("iFileMem.dat")

        self.vPAP = h.Vector()
        self.vPAP.record(self.pap(0.5)._ref_v)

        self.vFile = h.File("vFile.dat")
        self.vFile.wopen("vFile.dat")

        if hasattr(self,"soma"):
            self.vSoma = h.Vector()
            self.vSoma.record(self.soma(0.5)._ref_v)

            self.vFileSoma = h.File("vFileSoma.dat")
            self.vFileSoma.wopen("vFileSoma.dat")

            self.iKSoma = h.Vector()
            self.iKSoma.record(self.soma(0.5)._ref_ik)

            self.iKFileSoma = h.File("iKFileSoma.dat")
            self.iKFileSoma.wopen("iKFileSoma.dat")

        self.time = h.Vector()
        self.time.record(h._ref_t)

        self.tFile = h.File("tFile.dat")
        self.tFile.wopen("tFile.dat")

    def setK(self,compartment,initialKo):
        h.ki0_k_ion = 110 * mM # Global concentration for astrocytes from Savtchenko
        # h.ko0_k_ion = initialKo * mM # Global concentration
        compartment.ki = h.ki0_k_ion
        compartment.ko = initialKo * mM
        compartment.ek = -90 * mV
        # print('Potassium Parms')

    def printRec(self):
        if hasattr(self,"iNMDA"):
            self.iNMDA.printf(self.iFile)
            self.iFile.close()
        
        self.iMem.printf(self.iFileMem)
        self.iFileMem.close()

        
        self.vPAP.printf(self.vFile)
        self.vFile.close()

        if hasattr(self,"soma"):
            self.vSoma.printf(self.vFileSoma)
            self.vFileSoma.close()
            
            self.iKSoma.printf(self.iKFileSoma)
            self.iKFileSoma.close()

        self.time.printf(self.tFile)
        self.tFile.close()
        
        if hasattr(self,"iNMDA"):
            return list(self.iNMDA)[-1], max(self.iNMDA)

def eq(x,a,b):
    return a*x + b

def measureCond(fName):
    if not os.path.isfile(f'{fName}.pickle'):
        voltList = np.arange(0,100,5)
        IV = {}
        IVslow = {}
        IVfast = {}
        for volt in voltList:
            print(volt)
            iS,iF = PAPModel(volt)
            IVslow[volt] = iS
            IVfast[volt] = iF
            print(iS,iF)
        IV['slow'] = IVslow
        IV['fast'] =IVfast
        with open(f'{fName}.pickle', 'wb') as handle:
            pickle.dump(IV, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{fName}.pickle', 'rb') as handle:
            IV = pickle.load(handle)

    for mode in IV.keys():
        print(mode)
        plt.cla()
        plt.clf()
        I = list(IV[mode].values())
        V = list(IV[mode].keys())
        popt, pcov = curve_fit(eq,V,I)
        x = np.linspace(-40, 100, 50)
        print(popt)
        plt.plot(V,I,label='model')
        plt.plot(x,eq(x,*popt),label=f'{popt[0]}x+{popt[1]}')
        plt.legend()
        plt.savefig(f"{fName}{mode}.pdf")
        
def loadFile(fName):
    if os.path.isfile(fName):
        return pd.read_csv(fName,header=None)
    else:
        return None

def plot(dir,zoom=False,ext=False):
    i = loadFile(os.path.join(dir,'iFile.dat'))
    v = loadFile(os.path.join(dir,'vFile.dat'))
    mem = loadFile(os.path.join(dir,'iFileMem.dat'))
    t = loadFile(os.path.join(dir,'tFile.dat'))
    plt.plot(t,i)
    plt.ylabel('pA')
    if max(i.iloc[:, 0]) > 1:
        plt.ylim(0,25)
    plt.xlabel('ms')
    if zoom:
        plt.xlim(9.99,10.025)
    elif ext and max(t) >= 10000:
        plt.xlim(11,10000)
    plt.savefig(os.path.join(dir,'results.pdf'))
    plt.cla()
    plt.clf()
    if v is not None:
        plt.plot(t,v)
        plt.ylabel('mV')
        plt.xlabel('ms')
        print(max(v.iloc[:,0]))
        plt.ylim(-90,0)
        # if zoom or max(v.iloc[:, 0]) > -20: 
        #     plt.xlim(9.99,10.1)
        # elif ext and max(t) >= 10000:
        #     plt.xlim(11,10000)
            
        plt.savefig(os.path.join(dir,'resultsV.pdf'))
        plt.cla()
        plt.clf()
    if mem is not None:
        plt.plot(t,mem)
        plt.ylabel('pA')
        plt.xlabel('ms')
        # plt.ylim(-90,0)
        if zoom:
            plt.xlim(9.99,10.025)
        elif ext and max(t) >= 10000:
            plt.xlim(11,10000)

        plt.savefig(os.path.join(dir,'resultsIMem.pdf'))
    return max(v.iloc[:,0])

def multiChannel(itr=100):
    dList = []
    for i in range(1,itr + 1):
        PAPModel(40,multiple=i,mode=0)
        dList.append(plot(".") + 85)
    with open(f'dList.pickle', 'wb') as handle:
        pickle.dump(dList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plt.cla()
    plt.clf()
    plt.scatter(range(1,itr + 1),dList)
    plt.savefig('patchXDepolar.pdf')

def get_iter(distance,dist_steps,chan,chan_steps):
    iterations = []
    for i in range(1,distance,dist_steps):
        for j in range(1,chan,chan_steps):
            iterations.append((i,j))

    return iterations

def parallizeFor(iterations,functions,functionArgs,functionParms,callmethods):
    tmpsize = size - 1 # Exclude root
    # Set up tqdm
    if rank == 0:
        pbar = tqdm.tqdm(total = len(iterations))
    # Calculate the number of iterations each process will handle
    iterations_per_process = len(iterations) // tmpsize

    # Adjust the range for the last process
    if rank == size - 1:
        remaining_iterations = len(iterations) % tmpsize
    else:
        remaining_iterations = 0

    # results list for each rank
    results = [ [] for i in range(len(iterations)) ]

    comm.Barrier()
    if rank == 0:
        remaining = len(iterations)
        while remaining > 0:
            print(remaining)
            sys.stdout.flush()
            s = MPI.Status()
            comm.Probe(status=s)
            if s.tag == compute_tag:
                update_msg = comm.recv(tag=compute_tag)  
                pbar.update(1)
                remaining -= 1
                print(f'recieved end {remaining}')
                sys.stdout.flush()
    else:
        for index in range((rank-1) * iterations_per_process, rank * iterations_per_process + remaining_iterations):
            parmSet = iterations[index]
            print(f'Thread {rank} is performing set {parmSet}')
            for k,func in enumerate(functions):
                for l,parameterName in enumerate(functionParms):
                    functionArgs[k][parameterName] = parmSet[l]
                tmpInstance = functions[k](**functionArgs[k])
                results[index].append(getattr(tmpInstance,callmethods[k])())
            # update tqdm
            comm.send(update_msg,dest=0,tag=compute_tag)

    comm.Barrier()
    return comm.gather(results, root = 0)

    
def multiDistance(x,read=False):
    somaSize, bLen, bWid, papWid, bNum = x
    dList = []
    cList = []
    vList = []
    if read:
        with open(f'ballStick.pickle', 'rb') as handle:
            dList,cList,vList = pickle.load(handle)

    else:
        vSomaList = []
        vPAPList = []
        if parallel:
            # Calculate the number of iterations each process will handle
            iterations = comm.bcast(get_iter(101,100,301,10),root=0)
            # iterations_per_process = len(iterations) // size

            # # Adjust the range for the last process
            # if rank == size - 1:
            #     remaining_iterations = len(iterations) % size
            # else:
            #     remaining_iterations = 0

            # Individual list for each rank
            vSoma = []
            vPAP = []
            d = []
            c = []

            comm.Barrier()
            funcArgs = []
            funcArgs.append({
                'currentClamp':20,
                'bWid':bWid,
                'somaSize':somaSize,
                'mode':0,
                'bNum':int(bNum),
                'papWid':papWid,
                'Glu':True,
                'kir2':2
            })
            funcArgs.append({
                'currentClamp':20,
                'bWid':bWid,
                'somaSize':somaSize,
                'mode':0,
                'bNum':int(bNum),
                'somaCheck':False,
                'papWid':papWid,
                'Glu':True,
                'kir2':2
            })
            # make sure that funcParms is in the correct order of whatever iterations spits out
            results = parallizeFor(iterations,[PAPModel,PAPModel],funcArgs,['bLen','multiple'],['run','run'])
            # for index in range(rank * iterations_per_process, (rank + 1) * iterations_per_process + remaining_iterations):
            #     i,j = iterations[index]
            #     print(f'Thread {rank} is performing distance {i}, channel count {j}')
                # vSoma.append(max(
                #     np.array(PAPModel(currentClamp=20,
                #                       multiple=j,
                #                       bLen=i,
                #                       bWid=bWid,
                #                       somaSize=somaSize,
                #                       mode=0,
                #                       bNum=int(bNum),
                #                       papWid=papWid,
                #                       Glu=True).vSoma) + 85,
                #     key=abs
                # ))
                # vPAP.append(max(
                #     np.array(PAPModel(currentClamp=20,
                #                       multiple=j,
                #                       bLen=i,
                #                       bWid=bWid,
                #                       somaSize=somaSize,
                #                       mode=0,
                #                       bNum=int(bNum),
                #                       somaCheck=False,
                #                       papWid=papWid,
                #                       Glu=True).vPAP) + 85,
                #     key=abs
                # ))
                # d.append(i)
                # c.append(j)
            comm.Barrier()
            for index,res in enumerate(results):
                i,j = iterations[index]
                d.append(i)
                c.append(j)
                for i,rIndi in enumerate(res):
                    if i == 0:
                        vSoma.append(max(np.array(rIndi.vSoma) + 85, key=abs))
                    else:
                        vPAP.append(max(np.array(rIndi.vPAP) + 85, key=abs))
                
            vSomaList = comm.gather(vSoma, root = 0)
            vPAPList = comm.gather(vPAP, root = 0)
            dList = comm.gather(d, root = 0)
            cList = comm.gather(c, root = 0)

        else:
            for i in range(1,101,10):
                for j in range(100,101,1):
                    dList.append(i)
                    cList.append(j)
                    sim = PAPModel(multiple=j,
                                   bLen=i,
                                   currentClamp=1,
                                   bWid=bWid,
                                   somaSize=somaSize,
                                   mode=0,
                                   bNum=int(bNum),
                                   papWid=papWid,
                                   Glu=True)
                    sim.run()
                    vSomaList.append(max(
                        np.array(sim.vSoma) + 85,
                        key=abs
                    ))
                    sim = PAPModel(multiple=j,
                                   bLen=i,
                                   bWid=bWid,
                                   currentClamp=1,
                                   somaSize=somaSize,
                                   mode=0,
                                   bNum=int(bNum),
                                   somaCheck=False,
                                   papWid=papWid,
                                   Glu=True)
                    sim.run()
                    vPAPList.append(max(
                        np.array(sim.vPAP) + 85,
                        key=abs
                    ))
        # plt.plot(np.array(range(len(vList[-1])))*PAPModel.dt,vList[-1],label=f'{current} pA')
        vList = [vSomaList,vPAPList]

        if not parallel or rank == 0:            
            with open(f'ballStick.pickle', 'wb') as handle:
                pickle.dump([dList,cList,vList], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create a figure and a 3D axis
    if not parallel or rank == 0:
        for i,v in enumerate(vList):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            # Create the scatter plot
            ax.scatter3D(dList, cList, v, c=v, cmap='viridis')

            # Set labels and title
            ax.set_xlabel('distance')
            ax.set_ylabel('channel Count')
            if i == 0:
                name = 'soma'
            else:
                name = 'pap'
            ax.set_zlabel(f'Voltage Change{name}')

            # Show the plot
            j=""
            while os.path.isfile(f'./3Dplot{name}{j}.pdf'):
                if j == "":
                    j = 1
                else:
                    j += 1
                
            plt.savefig(f'./3Dplot{name}{j}.pdf')


def find_nan_inf_index(lst):
    for i, value in enumerate(lst):
        if math.isnan(value) or math.isinf(value):
            return i
    return -1  # Return -1 if no NaN or inf value is found

def remove_nan_values(lst,lst2):
    index = find_nan_inf_index(lst)
    while index != -1:
        del lst[index]
        lst2 = np.delete(lst2,index)
        index = find_nan_inf_index(lst)
    return lst,lst2

def measureRi(x):
    somaSize, bLen, bWid, papWid, bNum = x
    # Make a list for tested currents
    cList = np.arange(-800,801,200)
    vSomaList = []
    vPAPList = []
    for current in cList:
        vSomaList.append(PAPModel(currentClamp=current, bLen=bLen, bWid=bWid, somaSize=somaSize, mode=2,bNum=int(bNum),papWid=papWid).vSoma)
        vPAPList.append(PAPModel(currentClamp=current, bLen=bLen, bWid=bWid, somaSize=somaSize, mode=2,bNum=int(bNum),somaCheck=False,papWid=papWid).vPAP)
        # plt.plot(np.array(range(len(vList[-1])))*PAPModel.dt,vList[-1],label=f'{current} pA')

    
    vList,somaC = remove_nan_values([ v[-1] for v in vSomaList],cList)
    somapopt, pcov = curve_fit(eq,vList,somaC)
    # x = np.linspace(-600,600)
    # plt.plot(eq(x,*popt),x)
    # plt.legend()
    # plt.show()
    print(f'{abs(1/somapopt[0])} MOhm')
    vList,papC = remove_nan_values([ v[-1] for v in vPAPList],cList)
    pappopt, pcov = curve_fit(eq,vList,papC)
    print(f'{abs(1/pappopt[0])} MOhm')

    return abs(1/somapopt[0] - 2.6) + abs(1/pappopt[0] - 1050) # soma input resistance score

if __name__ == "__main__" or parallel:
    #measureCond('IV')
    #multiChannel()
    # Soma 2.5836550239043317 MOhm
    # PAP 1035.108930679734 MOhm
    if parallel:
        comm.Barrier()
        start = time.time()
        comm.bcast(start,root=0)
    # measureRi([3,  30,  4.28,  4.3e-4,  1])
    multiDistance([3,  30,  4.28,  4.3e-4,  1])
    # cell = PAPModel(currentClamp=0, bLen=20, bWid=4.28, somaSize=3, mode=2,bNum=1,papWid=4/3e-4)
    # cell.run()
    if parallel:
        comm.Barrier()
        end = time.time()
        comm.bcast(end,root=0)
    if parallel and rank == 0:
        time_took = end - start
        with open(f'timeres{size}.txt','w') as f:
            f.write(str(time_took))
    # measureRi((2.8e8,50,3.5e7,3))
    # print(minimize(measureRi,(10,30,5,0.02,1),method='Nelder-Mead',bounds=[(1,None),(10,None),(1,None),(0.000001,None),(1,50)],options={'disp':True},tol=0.01))
