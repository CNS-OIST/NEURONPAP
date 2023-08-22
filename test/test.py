"""
Author: Joel Nakatani
Overview:

Parameters:
"""

from neuron import h
from neuron.units import mM,mV
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pickle
import os

class cell():
    soma = object()
    tstop = 200
    dt = 0.01
    celsius = 37
    v_init = -85
    
    def __init__(self):
        # Load NEURON GUI and parameters
        h.load_file("stdgui.hoc")

        # Set simulation parameters
        h.tstop = self.tstop
        h.dt = self.dt
        h.celsius = self.celsius
        h.v_init = self.v_init

        # Build Morphology
        self.morphology()
        self.astroMem(self.soma)

    def run(self,clamp=0,mode=0):
        # Clamp settings
        if mode > 1:
            #Step Current
            if not hasattr(self,'ic'):
                self.ic = h.IClamp(self.soma(0.5))
            self.ic.dur = h.tstop
            self.ic.delay = 0 # ms starts with glutamate
            self.ic.amp = clamp * 0.001  # nA current injection (1 pA)
        elif mode > 0:
            if not hasattr(self,'vc'):
                self.vc = h.VClamp(self.soma(0.5))
            self.vc.dur[0] = h.tstop
            self.vc.amp[0] = clamp  # mV depolarization (dV = 20)

        h.run()

    def morphology(self):
        # create morphology
        self.soma = h.Section(name='soma')
        self.soma.diam = 10
        self.soma.L = 10
        self.soma.nseg = 1 

    def astroMem(self,compartment):
        # add astrocyte properties
        compartment.Ra = 100
        compartment.cm = 0.8
        compartment.insert('pas')
        compartment.e_pas = -85
        compartment.g_pas = 1/11150

    def channels(self,compartment,**kwargs):
        channelLibrary = ['kir4','twik','K_acc','kleak','kdifl']
        
        switch = set(kwargs.keys()) & set(channelLibrary)
        for k,v in kwargs.items():
            if k in switch and v:
                compartment.insert(k)
        # insert relevant channels
        default = set(channelLibrary) - set(switch)
        for channel in default:
            compartment.insert(channel)
        # compartment.ki = 130 * mM
        # compartment.ko = 8.5 * mM # STEPHEN F. 1988 for seizure induction


    def record(self,toFile=False):
        #Save Stuff
        self.vSoma = h.Vector()
        self.vSoma.record(self.soma(0.5)._ref_v)
        self.iSoma = h.Vector()
        self.iSoma.record(self.soma(0.5)._ref_i_pas)
        self.iKSoma = h.Vector()
        self.iKSoma.record(self.soma(0.5)._ref_ik)
        self.KconcSoma = h.Vector()
        self.KconcSoma.record(self.soma(0.5)._ref_ko)
        self.time = h.Vector()
        self.time.record(h._ref_t)
        
        if toFile and hasattr(self,"soma"):
            self.vFileSoma = h.File("vFileSoma.dat")
            self.vFileSoma.wopen("vFileSoma.dat")
            self.tFile = h.File("tFile.dat")
            self.tFile.wopen("tFile.dat")
        
    def setK(self,initialKo,initialKi=70):
        # initialKi tweaked for maintaining RMP of -85
        h.init()
        h.ki0_k_ion = initialKi * mM # Global concentration for astrocytes from Savtchenko
        h.ko0_k_ion = initialKo * mM # Global concentration
        self.soma.ki = initialKi * mM
        self.soma.ko = initialKo * mM
        # self.soma.ek = -90 * mV

    def overexpressionKir(self, multiple):
        # captures reducing depolarization
        for seg in self.soma:
            seg.kir4.Pkir = seg.kir4.Pkir/multiple # assumption that Pk change is smaller than change of U(max)
            if seg.kir4.Pkir > 1.0:
                seg.kir4.Pkir = 1.0

    def overexpressionTwik(self, multiple):
        # captures reducing depolarization
        for seg in self.soma:
            seg.twik.PBkp = seg.twik.PBkp * multiple
            if seg.twik.PBkp > 1.0:
                seg.twik.PBkp = 1.0


def testKoConc():
    astrocyte = cell()
    astrocyte.channels(astrocyte.soma)
    astrocyte.record()
    for Kconc in [2.5,5,8.5]:
        astrocyte.setK(Kconc)
        print(astrocyte.soma.psection())
        astrocyte.run()
        # plt.plot(astrocyte.time,astrocyte.vSoma)
        print(astrocyte.soma.psection())
        plt.plot(astrocyte.time,astrocyte.vSoma)
        print(max(np.array(astrocyte.vSoma) + 85))
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')
    fname = f'Depolarization_Ko{Kconc}_ALL.pdf'
    dirName = os.path.join(".") #,"results","SOMA","SOMATIC_KIROE") # To Do make title selection more smarter
    plt.savefig(os.path.join(dirName,fname))

                
def testOVERExpression(**kwargs):
    astrocytes = []
    for multiple in np.arange(1,500,50):
        astrocytes.append(cell())
        astrocytes[-1].channels(astrocytes[-1].soma,
                                **kwargs
                                )
                           
        astrocytes[-1].record()

        for Kconc in [2.5,5,8.5]:
            astrocytes[-1].setK(Kconc)
            print(astrocytes[-1].soma.psection())
            if 'kir4' in kwargs.keys() and kwargs['kir4']:
                astrocytes[-1].overexpressionKir(multiple)
            astrocytes[-1].run()
            # plt.plot(astrocytes[-1].time,astrocytes[-1].vSoma)
            print(astrocytes[-1].soma.psection())
            plt.plot(astrocytes[-1].time,astrocytes[-1].vSoma)
            print(max(np.array(astrocytes[-1].vSoma) + 85))
            plt.xlabel('time (ms)')
            plt.ylabel('voltage (mV)')
            fname = f'Depolarization_Ko{Kconc}_KIROE_{multiple}.pdf'
            dirName = os.path.join(".","results","SOMA","SOMATIC_KIROE") # To Do make title selection more smarter
            plt.savefig(os.path.join(dirName,fname))

    SSVoltage = [ astrocyte.vSoma[-1] for astrocyte in astrocytes ] # Shoule be all 8.5 mM
    plt.cla()
    plt.clf()
    plt.plot(np.arange(1,500,50),SSVoltage)
    plt.xlabel('Relative Overexpression')
    plt.ylabel('depolarization at 8.5 mM K+')
    fname = f'Depolarization_KIROE_ALL.pdf'
    plt.savefig(os.path.join(dirName,fname))

def testKDExpression(**kwargs):
    astrocytes = []
    for multiple in np.arange(1,500,50):
        astrocytes.append(cell())
        astrocytes[-1].channels(astrocytes[-1].soma,
                                **kwargs
                                )
                           
        astrocytes[-1].record()

        for Kconc in [2.5,5,8.5]:
            astrocytes[-1].setK(Kconc)
            print(astrocytes[-1].soma.psection())
            if 'kir4' in kwargs.keys() and kwargs['kir4']:
                astrocytes[-1].overexpressionKir(1/multiple)
            if 'twik' in kwargs.keys() and kwargs['twik']:
                astrocytes[-1].overexpressionTwik(1/multiple)
            astrocytes[-1].run()
            # plt.plot(astrocytes[-1].time,astrocytes[-1].vSoma)
            print(astrocytes[-1].soma.psection())
            plt.plot(astrocytes[-1].time,astrocytes[-1].vSoma)
            print(max(np.array(astrocytes[-1].vSoma) + 85))
            plt.xlabel('time (ms)')
            plt.ylabel('voltage (mV)')
            fname = f'Depolarization_Ko{Kconc}_KIRKD_{multiple}.pdf'
            dirName = os.path.join(".","results","SOMA","SOMATIC_TWIKKO") # To Do make title selection more smarter
            plt.savefig(os.path.join(dirName,fname))

    SSVoltage = [ astrocyte.vSoma[-1] for astrocyte in astrocytes ] # Shoule be all 8.5 mM
    plt.cla()
    plt.clf()
    plt.plot(np.arange(1,500,50),SSVoltage)
    plt.xlabel('Relative KD')
    plt.ylabel('depolarization at 8.5 mM K+')
    fname = f'Depolarization_KIROE_ALL.pdf'
    plt.savefig(os.path.join(dirName,fname))
    
        

def eq(x,a,b):
    return a * x + b

def measureCond(fName,twikExpr,kirExpr,gmax=True):
    vStep = 5
    voltList = np.arange(-80,100,vStep)
    IV = {}
    astrocyte = cell()
    astrocyte.channels(astrocyte.soma,
                       twik=True,
                       kir4=True,
                       K_acc=True,
                       kleak=True,
                       kdifl=True
                       )
    astrocyte.record()
    astrocyte.setK(50,initialKi=130)
    if not twikExpr:
        astrocyte.overexpressionTwik(1/100)
    if not kirExpr:
        astrocyte.overexpressionKir(1/100)
        

    
    for volt in voltList:
        print(volt)
        astrocyte.run(clamp=volt,mode=1)
        IV[volt] = astrocyte.iKSoma[-1]
    with open(f'{fName}.pickle', 'wb') as handle:
        pickle.dump(IV, handle, protocol=pickle.HIGHEST_PROTOCOL)
    I = list(IV.values())
    V = list(IV.keys())
    # plt.plot(V,I,label=f'{fName}')
    if gmax:
        # print(np.array(list(zip(I,V))))
        gmaxValue = max(np.gradient(np.array(I)))/vStep
        b = I[0] - gmaxValue * V[0]
        popt = [gmaxValue,b]
        print(gmaxValue)
        # gVals  = [ i / (v + 85) for i,v in zip(I,V)] 
        # gmaxValue = abs(max(gVals,key=abs))
        
    else:
        popt, pcov = curve_fit(eq,V,I)
    x = np.linspace(-150, 50, 50)
    # print(popt)
    # plt.plot(x,eq(x,*popt),label=f'gmax slope:{popt[0]:.2f}x+{popt[1]:.2f}')
    # plt.legend()
    # plt.xlabel('voltage (mv)')
    # plt.ylabel('current (nA)')
    # plt.savefig(f"{fName}.pdf")
    # plt.cla()
    # plt.clf()
    relG = np.gradient(np.array(I))/max(np.gradient(np.array(I)))
    plt.scatter(V,relG,label=f'{fName}')
    # plt.plot(V,gVals/gmaxValue,label=f'{fName}')
    plt.legend()
    plt.xlabel('voltage (mv)')
    plt.ylabel('relative conductance (g/gmax)')
    plt.savefig(f"{fName}_relG.pdf")
    # plt.cla()
    # plt.clf()

    

if __name__ == "__main__":
    # testKDExpression(
    #     twik=True,
    #     kir4=False,
    #     K_acc=False,
    #     kleak=False,
    #     kdifl=False
    # )
    # astrocyte = cell()
    # astrocyte.channels(astrocyte.soma,
    #                    twik=True,
    #                    kir4=True,
    #                    K_acc=False,
    #                    kleak=False,
    #                    kdifl=False
    #                    )
    # astrocyte.record()
    # astrocyte.setK(50,initialKi=130)
    # astrocyte.run(clamp=30,mode=1)
    # plt.plot(astrocyte.time,astrocyte.iKSoma)
    # plt.show()
    measureCond('Control',True,True)
    measureCond('TWIKKO',False,True)
    # measureCond('KIRKO',True,False)
    # testKoConc()
