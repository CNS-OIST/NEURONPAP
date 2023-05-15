from neuron import h, load_mechanisms
import numpy as np

def main(voltageClamp):
    # Load NEURON GUI and parameters
    h.load_file("stdgui.hoc")
    h.load_file("params.hoc")

    # Set simulation parameters
    h.tstop = 10000
    h.dt = 0.001
    h.celsius = 23
    h.v_init = -65

    # Create the branch (not included in the original hoc file)
    # branch = h.Section()

    # Access the PAP object
    pap = h.Section(name="PAP")

    # Set astrocyte leaf membrane parameters
    pap.L = 0.3
    pap.diam = 0.02
    pap.nseg = 10
    pap.insert('pas')
    pap.e_pas = -65

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
    vc = h.VClamp(0.5)
    vc.dur[0] = h.tstop
    vc.amp[0] = voltageClamp  # mV depolarization (dV = 20)


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

    wFile = h.File()
    wFile.ropen("./results/optimize/optW.dat")
    SynWeight = wFile.scanvar()
    wFile.close()

    print(SynWeight)

    # Create the synaptic NMDA conductance
    sNMDA = h.Exp5NMDA(0.5)
    stim = h.NetStim(0.5)
    stim.interval = 1
    stim.number = 1
    stim.start = 10
    stim.noise = 0

    nc = h.NetCon(stim, sNMDA)
    nc.weight[0] = SynWeight
    nc.delay = 0
    sNMDA.tau2_0 = Tau2_0
    sNMDA.a2 = A2
    sNMDA.b2 = B2
    sNMDA.tau3_0 = Tau3_0
    sNMDA.a3 = A3
    sNMDA.b3 = B3
    sNMDA.Mg = 0  # To turn off external Magnesium

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
    return list(iNMDA)[-1]

if __name__ == "__main__":
    voltList = np.arange(-65,100,5)
    IV = {}
    for volt in voltList:
        print(volt)
        i = main(volt)
        IV[volt] = i
        print(i)
    I = IV.values()
    V = IV.keys()
    A = np.vstack([V, np.ones(len(V))]).T
    m,c = np.linalg.lstsq(A,I, rcond = None)[0]
    print(m,c)
