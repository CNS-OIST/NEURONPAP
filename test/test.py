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

class cell():
    soma = object()
    tstop = 1000
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
            ic = h.IClamp(self.soma(0.5))
            ic.dur = h.tstop
            ic.delay = 0 # ms starts with glutamate
            ic.amp = clamp * 0.001  # nA current injection (1 pA)
        elif mode > 0:
            vc = h.VClamp(self.soma(0.5))
            vc.dur[0] = h.tstop
            vc.amp[0] = clamp  # mV depolarization (dV = 20)

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
        self.soma.ek = -90 * mV

    def overexpressionKir(self, multiple):
        # captures reducing depolarization
        for seg in self.soma:
            seg.kir4.Pkir = multiple * seg.kir4.Pkir
        
def testKoConc():
    astrocyte = cell()
    astrocyte.channels(astrocyte.soma,twik=False)
    astrocyte.record()

    for Kconc in [2.5,5,8.5]:
        astrocyte.setK(Kconc)
        print(astrocyte.soma.psection())
        # astrocyte.overexpressionKir(30)
        astrocyte.run()
        # plt.plot(astrocyte.time,astrocyte.vSoma)
        print(astrocyte.soma.psection())
        plt.plot(astrocyte.time,astrocyte.vSoma)
        print(max(np.array(astrocyte.vSoma) + 85))
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')
        plt.savefig(f'voltagteKo{Kconc}.pdf')

def eq(x,a,b):
    return a * x + b

def measureCond(fName,twikExpr):
    voltList = np.arange(-150,50,5)
    IV = {}
    astrocyte = cell()
    astrocyte.channels(astrocyte.soma,
                       twik=twikExpr,
                       K_acc=False,
                       kleak=False,
                       kdifl=False
                       )
    astrocyte.record()
    astrocyte.setK(5,initialKi=130)
    
    for volt in voltList:
        print(volt)
        astrocyte.run(clamp=volt,mode=1)
        IV[volt] = astrocyte.iKSoma[-1]
    with open(f'{fName}.pickle', 'wb') as handle:
        pickle.dump(IV, handle, protocol=pickle.HIGHEST_PROTOCOL)
    I = list(IV.values())
    V = list(IV.keys())
    popt, pcov = curve_fit(eq,V,I)
    x = np.linspace(-150, 50, 50)
    print(popt)
    plt.plot(V,I,label='model')
    plt.plot(x,eq(x,*popt),label=f'{popt[0]}x+{popt[1]}')
    plt.legend()
    plt.savefig(f"{fName}.pdf")

    

if __name__ == "__main__":
    measureCond('Control',True)
    measureCond('TWIKKO',False)



































