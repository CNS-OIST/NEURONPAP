"""
Author: Joel Nakatani
Overview:

Parameters:
"""

from neuron import h
from neuron.units import mM,mV
import matplotlib.pyplot as plt
import numpy as np

class cell():
    soma = object()
    tstop = 500
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

    def run(self): #,clamp=2,mode=0):
        # Clamp settings
        # if mode > 0:
        #     #Step Current
        #     ic = h.IClamp(self.soma(0.5))
        #     ic.dur = h.tstop
        #     ic.delay = 0 # ms starts with glutamate
        #     ic.amp = clamp * 0.001  # nA current injection (1 pA)
        # else:
        #     vc = h.VClamp(self.soma(0.5))
        #     vc.dur[0] = h.tstop
        #     vc.amp[0] = clamp  # mV depolarization (dV = 20)

        h.init()
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
        # compartment.insert('pas')
        # compartment.e_pas = -85
        # compartment.g_pas = 1/11150
        self.channels(compartment)

    def channels(self,compartment):
        # insert relevant channels
        compartment.insert('kir4')
        compartment.insert('twik')
        compartment.insert('K_acc')
        compartment.insert('kleak')
        # compartment.ki = 130 * mM
        # compartment.ko = 8.5 * mM # STEPHEN F. 1988 for seizure induction


    def record(self,toFile=False):
        #Save Stuff
        self.vSoma = h.Vector()
        self.vSoma.record(self.soma(0.5)._ref_v)
        self.iKSoma = h.Vector()
        self.iKSoma.record(self.soma(0.5)._ref_ik)
        self.time = h.Vector()
        self.time.record(h._ref_t)
        
        if toFile and hasattr(self,"soma"):
            self.vFileSoma = h.File("vFileSoma.dat")
            self.vFileSoma.wopen("vFileSoma.dat")
            self.tFile = h.File("tFile.dat")
            self.tFile.wopen("tFile.dat")
        
    def setK(self,initialKo):
        h.ki0_k_ion = 110 * mM # Global concentration for astrocytes from Savtchenko
        # h.ko0_k_ion = initialKo * mM # Global concentration
        self.soma.ki = h.ki0_k_ion
        self.soma.ko = initialKo * mM
        self.soma.ek = -90 * mV
        
def main():
    astrocyte = cell()
    astrocyte.record()

    for Kconc in [8.5]:
        astrocyte.setK(Kconc)
        print(astrocyte.soma.psection())
        astrocyte.run()
        # plt.plot(astrocyte.time,astrocyte.vSoma)
        print(astrocyte.soma.psection())
        plt.plot(astrocyte.time,astrocyte.vSoma)
        print(max(np.array(astrocyte.vSoma) + 85))
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')
        plt.savefig(f'voltagteKo{Kconc}.pdf')

if __name__ == "__main__":
    main()
    
