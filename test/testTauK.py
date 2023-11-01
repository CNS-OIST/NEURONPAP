# Load NEURON GUI and parameters
from neuron import h
import matplotlib.pyplot as plt

h.load_file("stdgui.hoc")
h.load_file("../src/neuronHoc/params.hoc")
h.load_file("stdgui.hoc")

for dt in range(1,3):
    for tau in range(10,20,5):
        h('xopen("../src/neuronHoc/astrocyte.hoc")')
        KConc = h.Vector()
        KConc.record(h.PAP(0.5)._ref_ko)
        time = h.Vector()
        time.record(h._ref_t)
        
        dt = dt * 0.01
        h.dt = dt
        
        for sec in h.allsec():
            for seg in sec:
                seg.k_acc.tauk = tau

                
        h.finitialize(-85)
        h.continuerun(3)
        
        h.setK(8.5,2.5)
        h.psection(sec=h.PAP)

        h.continuerun(100)
        plt.plot(list(time),list(KConc),label=f'tau: {tau} dt:{dt}')
plt.legend()
plt.show()
