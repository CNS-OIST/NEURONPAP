from neuron import h, load_mechanisms
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
import os
import pandas as pd
from results.utils.visualize import func as plotRes

class PAPModel():
    tstop = 100
    dt = 0.00001
    celsius = 37
    v_init = -85
    pap = object()

    def __init__(self,bLen=30,voltageClamp=40,multiple=1,mode=-1):
        # Load NEURON GUI and parameters
        h.load_file("stdgui.hoc")
        h.load_file("params.hoc")

        # Set simulation parameters
        h.tstop = self.tstop
        h.dt = self.dt
        h.celsius = self.celsius
        h.v_init = self.v_init

        # Build Morphology
        self.morph(bLen=bLen)

        # Clamp settings
        if mode > 0:
            vc = h.VClamp(0.5)
            vc.dur[0] = h.tstop
            vc.amp[0] = voltageClamp  # mV depolarization (dV = 20)

        elif mode == 0:
            ic = h.IClamp(0.5)
            ic.dur = 0.002
            ic.delay = 10 # ms starts with glutamate
            ic.amp = 2 * 0.001  # nA current injection (1 pA)

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
            
            nc = h.NetCon(stim, self.NMDAs[-1])        
            nc.weight[0] = self.SynWeight
            nc.delay = 0

            self.NCs.append(nc) # Must be in outer later with python address allocated
        self.record(sNMDA = self.NMDAs[-1])
        h.init()
        h.run()
        self.printRec()

    def astroMem(self,compartment):
        compartment.Ra = 100
        compartment.cm = 0.8
        compartment.insert('pas')
        compartment.e_pas = -85
        compartment.g_pas = 1/11150

    def morph(self,isolate=False,bLen=None):
        # Access the PAP object
        self.pap = h.Section(name="PAP")

        # Set astrocyte leaf membrane parameters
        self.pap.L = 0.3
        self.pap.diam = 0.02
        self.pap.nseg = 1
        self.astroMem(self.pap)

        h.psection(sec=self.pap)
        if not isolate:
            # Create the branch (not included in the original hoc file)
            self.branch = h.Section(name='branch')
            if bLen != None and type(bLen) == int:
                self.branch.L = bLen
            else:
                self.branch.L = 30
            self.branch.diam = 1
            self.branch.nseg = 10
            self.astroMem(self.branch)
            h.psection(sec=self.branch)


            # create Soma
            self.soma = h.Section(name='soma')
            self.soma.diam = 10 # Agulhon 2008 Neuron
            self.soma.L = 10
            self.astroMem(self.soma)
            h.psection(sec=self.soma)


            #Connect
            self.branch.connect(self.soma(0.5))
            self.pap.connect(self.branch)
            h.topology()
            
    def readParameters(self,fDir='./results/optimize'):
        # Load optimization parameters
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


        self.time = h.Vector()
        self.time.record(h._ref_t)

        self.tFile = h.File("tFile.dat")
        self.tFile.wopen("tFile.dat")

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

if __name__ == "__main__":
    #measureCond('IV')
    #multiChannel()
    dList = []
    cList = []
    vList = []
    for i in range(1,101):
        for j in range(1,51):
            PAPModel(bLen=1,multiple=j)
            vMax = plotRes(".")
            dList.append(i)
            cList.append(j)
            vList.append(vMax)
            with open(f'ballStick.pickle', 'wb') as handle:
                pickle.dump([dList,cList,vList], handle, protocol=pickle.HIGHEST_PROTOCOL)
