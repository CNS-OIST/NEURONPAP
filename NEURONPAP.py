from neuron import h, load_mechanisms
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
import os
import pandas as pd

def main(voltageClamp,multiple=None,mode=1):
    # Load NEURON GUI and parameters
    h.load_file("stdgui.hoc")
    h.load_file("params.hoc")

    # Set simulation parameters
    h.tstop = 520
    h.dt = 0.0001
    h.celsius = 37
    h.v_init = -85

    # Create the branch (not included in the original hoc file)
    # branch = h.Section()

    # Access the PAP object
    pap = h.Section(name="PAP")

    # Set astrocyte leaf membrane parameters
    pap.L = 0.3
    pap.diam = 0.02
    pap.nseg = 1
    pap.Ra = 100
    pap.cm = 0.8
    pap.insert('pas')
    pap.e_pas = -85
    pap.g_pas = 1/11150

    h.psection()


    # Function to switch between voltage clamp and current clamp
    # def clampSwitch(mode):
    #     if mode > 1:
    #         vRec = h.Vector(570)
    #         tRec = h.Vector(570)
    #         vFile = h.File()
    #         tFile = h.File()

    #         vFile.ropen("./Data/Exp10-Clarke08-Fig10/vNaSpike.dat")
    #         tFile.ropen("./Data/Exp10-Clarke08-Fig10/tNaSpike.dat")

    #         vRec.scanf(vFile)
    #         tRec.scanf(tFile)
    #         vRec.play(vc.amp[0], tRec)
    #     elif mode > 0:
    #         vc = h.VClamp(0.5)
    #         vc.dur[0] = h.tstop
    #         vc.amp[0] = 40  # mV depolarization (dV = 20)
    #         return vc
    #     else:
    #         ic = h.IClamp(0.5)
    #         ic.dur = 2 * h.dt
    #         ic.delay = 10 - ic.dur  # ms starts with glutamate
    #         ic.amp = 1.5 * 0.001  # nA current injection (1 pA)
    #         return ic

    # print(clampSwitch(1))  # Use different mode
    # Outside function because of bug
    if mode > 0:
        vc = h.VClamp(0.5)
        vc.dur[0] = h.tstop
        vc.amp[0] = voltageClamp  # mV depolarization (dV = 20)

    elif mode == 0:
        ic = h.IClamp(0.5)
        ic.dur = 0.002
        ic.delay = 10 # ms starts with glutamate
        ic.amp = 2 * 0.001  # nA current injection (1 pA)

    # Load optimization parameters
    wFile = h.File()
    wFile.ropen("./results/optimize/optT2.dat")
    Tau2_0 = wFile.scanvar()
    A2 = wFile.scanvar()
    B2 = wFile.scanvar()
    wFile.close()

    wFile = h.File()
    wFile.ropen("./results/optimize/optT3.dat")
    Tau3_0 = wFile.scanvar()
    A3 = wFile.scanvar()
    B3 = wFile.scanvar()
    wFile.close()

    wFile = h.File()
    wFile.ropen("./results/optimize/optDelta.dat")
    DELTA = wFile.scanvar()
    wFile.close()
    DELTA = 0

    wFile = h.File()
    wFile.ropen("./results/optimize/optW.dat")
    SynWeight = wFile.scanvar()
    wFile.close()

    print(SynWeight)

    # Create the synaptic NMDA conductance
    stim = h.NetStim(0.5)
    stim.interval = 1
    stim.number = 1
    stim.start = 10
    stim.noise = 0

    if multiple is not None and type(multiple) == int:
        NMDAs = []
        NCs = []
        for i in range(multiple):
            NMDAs.append(h.Exp5NMDA(0.5))
            NCs.append( h.NetCon(stim, NMDAs[i]))
            NCs[i].weight[0] = SynWeight
            NCs[i].delay = 0
            NMDAs[i].tau2_0 = Tau2_0
            NMDAs[i].a2 = A2
            NMDAs[i].b2 = B2
            NMDAs[i].tau3_0 = Tau3_0
            NMDAs[i].a3 = A3
            NMDAs[i].b3 = B3
        sNMDA = NMDAs[i]

    else:
        sNMDA = h.Exp5NMDA(0.5)
        nc = h.NetCon(stim, sNMDA)
        nc.weight[0] = SynWeight
        nc.delay = 0
        sNMDA.tau2_0 = Tau2_0
        sNMDA.a2 = A2
        sNMDA.b2 = B2
        sNMDA.tau3_0 = Tau3_0
        sNMDA.a3 = A3
        sNMDA.b3 = B3

    #Save Stuff
    iNMDA = h.Vector()
    iNMDA.record(sNMDA._ref_i)

    iFile = h.File("iFile.dat")
    iFile.wopen("iFile.dat")

    iMem = h.Vector()
    iMem.record(pap(0.5)._ref_i_pas)

    iFileMem = h.File("iFileMem.dat")
    iFileMem.wopen("iFileMem.dat")

    vSoma = h.Vector()
    vSoma.record(pap(0.5)._ref_v)

    vFile = h.File("vFile.dat")
    vFile.wopen("vFile.dat")

    time = h.Vector()
    time.record(h._ref_t)

    tFile = h.File("tFile.dat")
    tFile.wopen("tFile.dat")

    h.init()
    h.run()

    iNMDA.printf(iFile)
    iFile.close()

    iMem.printf(iFileMem)
    iFileMem.close()


    vSoma.printf(vFile)
    vFile.close()

    time.printf(tFile)
    tFile.close()
    return list(iNMDA)[-1], max(iNMDA)

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
            iS,iF = main(volt)
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

if __name__ == "__main__":
    #measureCond('IV')
    itr = 100
    dList = []
    for i in range(1,itr + 1):
        main(40,multiple=i,mode=0)
        dList.append(plot(".") + 85)
    with open(f'dList.pickle', 'wb') as handle:
        pickle.dump(dList, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.cla()
    plt.clf()
    plt.scatter(range(1,itr + 1),dList)
    plt.savefig('patchXDepolar.pdf')
    
