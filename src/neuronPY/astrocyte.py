from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
import sys

class PAPModel(ResultsPAPModel):
    tstop = 1000
    initTstop = 500
    dt = 0.01  # not enabled
    celsius = 37
    v_init = -85
    somaSize = 10  # Soma Size
    bLen = 30  # Branch Size
    bWid = 3
    papWid = 0.02
    branches = []

    # NMDA parms
    multiple = int()
    readParms = False
    Tau2_0 = float()
    Tau3_0 = float()
    A2 = float()
    A3 = float()
    B2 = float()
    B3 = float()
    DELTA = float()
    SynWeight = 9.603338159338435e-09

    # K Parms
    defaultKo = 2.5

    def __init__(
        self,
        papWid=0.02,
        bWid=3,
        bNum=1,
        bLen=30,
        voltageClamp=40,
        somaSize=10,
        currentClamp=2,
        multiple=1,
        mode=2,
        somaCheck=False,
        Glu=False,
        initialKo=2.5,
        **kwargs,
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
        self.multiple = multiple

        # set K parms
        self.initialKo = initialKo

        # Build Morphology
        self.morph()
        # print('built morphology')
        # sys.stdout.flush()

        # set clamp parms
        self.mode = mode
        self.voltageClamp = voltageClamp
        self.currentClamp = currentClamp
        self.somaCheck = somaCheck

        if not parallel:
            self.readParameters()  # readfile in parallel causes errors
        if not hasattr(self, "NMDAs"):
            self.NMDAs = []
            self.NCs = []

        for i in range(self.multiple - len(self.NMDAs)):
            # Create the synaptic NMDA conductance
            stim = h.NetStim(self.pap(0.5))
            stim.interval = 1
            stim.number = 1
            stim.start = 1 * ms
            stim.noise = 0
            self.NMDAs.append(self.nmda())
            if Glu:
                self.NCs.append(
                    h.NetCon(stim, self.NMDAs[-1])
                )  # Must be in outer later with python address allocated
                self.NCs[-1].weight[0] = self.SynWeight
                self.NCs[-1].delay = 0

            # print('placed NMDAR')
            # sys.stdout.flush()
        # print('initialized')
        # sys.stdout.flush()

        for sec in h.allsec():
            if not hasattr(self, "GENEDict"):
                GENExpression(sec, kwargs)
                self.GENEDict = kwargs
        # print('set GENE manipulation')
        self.record(sNMDA=self.NMDAs[-1])

    def initialize(self, saveState=False):
        # print('initializing')
        # sys.stdout.flush()
        h.ki0_k_ion = 70 * mM  # Global concentration for astrocytes from Savtchenko
        self.setK()
        h.continuerun(self.initTstop * ms)
        if saveState:
            s = h.SaveState()
            s.save()
            with open(f"initializedState{rank}.dat", "wb") as f:
                s.fwrite(f)

    def run(self, printRes=False):
        # Clamp settings
        if self.mode > 2:
            # Step Current
            if self.somaCheck:
                ic = h.IClamp(self.soma(0.5))
            else:
                ic = h.IClamp(self.pap(0.5))
            ic.dur = self.tstop
            ic.delay = 0 * ms  # ms starts with glutamate
            ic.amp = self.currentClamp * 0.001  # nA current injection (1 pA)
        elif self.mode > 1:
            # voltage clamp
            vc = h.VClamp(self.soma(0.5))
            vc.dur[0] = h.tstop
            vc.amp[0] = self.voltageClamp  # mV depolarization (dV = 20)

        elif self.mode > 0:
            # Impulse Current
            ic = h.IClamp(self.pap(0.5))
            ic.dur = 0.002
            ic.delay = 10  # ms starts with glutamate
            ic.amp = self.currentClamp * 0.001  # nA current injection (1 pA)
        # print('clamp experiment setup')
        # sys.stdout.flush()

        h.continuerun((self.tstop) * ms)
        if printRes:
            self.printRec()
        self.cleanMorphology()
        # print('ran simulation')
        # sys.stdout.flush()

    def getRMP(self):
        self.initialize()
        self.run()
        RMP = sum(list(self.vSoma)) / len(list(self.vSoma))
        self.RMP = RMP
        return RMP

    def cleanMorphology(self):
        for sec in h.allsec():
            h.delete_section(sec=sec)
        self.branches = []
        self.soma = None
        self.pap = None
        delattr(self, "GENEDict")

    def astroMem(self, compartment):
        # add astrocyte properties
        compartment.Ra = 100
        compartment.cm = 0.8
        compartment.insert("pas")
        compartment.e_pas = -85
        compartment.g_pas = 1 / 11150
        self.channels(compartment)

    def channels(self, compartment):
        # insert relevant channels
        compartment.insert("kir2")
        compartment.insert("twik")
        compartment.insert("K_acc")
        compartment.insert("kleak")
        compartment.insert("kdifl")

    def morph(self, isolate=False, printTopology=False):
        # Access the PAP object
        if not hasattr(self, "pap"):
            self.pap = h.Section(name="PAP")
            self.astroMem(self.pap)

        # Set astrocyte leaf membrane parameters
        self.pap.L = 0.3
        self.pap.diam = self.papWid
        self.pap.nseg = 1

        # h.psection(sec=self.pap)
        if not isolate:
            # create Soma
            if not hasattr(self, "soma"):
                self.soma = h.Section(name="soma")
                self.astroMem(self.soma)
            self.soma.diam = self.somaSize
            self.soma.L = self.somaSize
            # h.psection(sec=self.soma)
            sl = h.SectionList()  # section shows up again bug
            branchCount = len(
                [branch for branch in self.branches if branch in list(sl)]
            )
            for i in range(self.bNum - branchCount):
                # Create the branch (not included in the original hoc file)
                self.branches.append(h.Section(name=f"branch{i}"))
                self.branches[-1].L = self.bLen
                self.branches[-1].diam = self.bWid
                self.branches[-1].nseg = 10
                self.astroMem(self.branches[-1])
                # h.psection(sec=self.branches[-1])

                # Connect
                self.branches[-1].connect(self.soma(0.5))
                if i == 0:
                    self.pap.connect(self.branches[-1])
        if printTopology:
            h.topology()

    def readParameters(self, fDir="./results/optimize"):
        self.readParms = True
        # Load optimization parameters
        lines = MPIReadlines(f"{fDir}/optT2.dat")
        self.Tau2_0 = lines[0]
        self.A2 = lines[1]
        self.B2 = lines[2]

        lines = MPIReadlines(f"{fDir}/optT3.dat")
        self.Tau3_0 = lines[0]
        self.A3 = lines[1]
        self.B3 = lines[2]

        self.DELTA = 0  # Lalo 2006 J. Neuroscience

        lines = MPIReadlines(f"{fDir}/optW.dat")
        self.SynWeight = lines[0]

    def nmda(self):
        sNMDA = h.Exp5NMDA(self.pap(0.5))
        if self.readParms:
            # load files if parameters are read
            sNMDA.tau2_0 = self.Tau2_0
            sNMDA.a2 = self.A2
            sNMDA.b2 = self.B2
            sNMDA.tau3_0 = self.Tau3_0
            sNMDA.a3 = self.A3
            sNMDA.b3 = self.B3
            sNMDA.delta = self.DELTA

        return sNMDA

    def record(self, sNMDA=None, toFile=False):
        # Save Stuff
        if sNMDA != None:
            self.iNMDA = h.Vector()
            self.iNMDA.record(sNMDA._ref_i)
            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        self.iMem = h.Vector()
        self.iMem.record(self.pap(0.5)._ref_i_pas)
        if toFile:
            self.iFileMem = h.File("iFileMem.dat")
            self.iFileMem.wopen("iFileMem.dat")

        self.vPAP = h.Vector()
        self.vPAP.record(self.pap(0.5)._ref_v)

        if toFile:
            self.vFile = h.File("vFile.dat")
            self.vFile.wopen("vFile.dat")

        self.KoPAP = h.Vector()
        self.KoPAP.record(self.pap(0.5)._ref_ko)

        if toFile:
            self.KoFile = h.File("KoFile.dat")
            self.KoFile.wopen("KoFile.dat")

        if hasattr(self, "soma"):
            self.vSoma = h.Vector()
            self.vSoma.record(self.soma(0.5)._ref_v)

            if toFile:
                self.vFileSoma = h.File("vFileSoma.dat")
                self.vFileSoma.wopen("vFileSoma.dat")

            self.iKSoma = h.Vector()
            self.iKSoma.record(self.soma(0.5)._ref_ik)

            if toFile:
                self.iKFileSoma = h.File("iKFileSoma.dat")
                self.iKFileSoma.wopen("iKFileSoma.dat")

            self.KoSoma = h.Vector()
            self.KoSoma.record(self.soma(0.5)._ref_ko)

            if toFile:
                self.KoFileSoma = h.File("KoFileSoma.dat")
                self.KoFileSoma.wopen("KoFileSoma.dat")

        self.time = h.Vector()
        self.time.record(h._ref_t)
        if toFile:
            self.tFile = h.File("tFile.dat")
            self.tFile.wopen("tFile.dat")

    def setK(self, initialKo=None):
        if initialKo == None:
            initialKo = self.initialKo
        h.init()  # quite important
        for sec in h.allsec():
            # h.psection(sec=sec)
            if sec == self.pap:
                for seg in sec:
                    seg.ko = initialKo * mM
                # print('set pap Ko')
            else:
                for seg in sec:
                    seg.ko = self.defaultKo * mM

            sec.ek = -90 * mV
        # print('Potassium Parms')

    def printRec(self):
        if hasattr(self, "iNMDA"):
            self.iNMDA.printf(self.iFile)
            self.iFile.close()

        self.iMem.printf(self.iFileMem)
        self.iFileMem.close()

        self.vPAP.printf(self.vFile)
        self.vFile.close()

        if hasattr(self, "soma"):
            self.vSoma.printf(self.vFileSoma)
            self.vFileSoma.close()

            self.iKSoma.printf(self.iKFileSoma)
            self.iKFileSoma.close()

        self.time.printf(self.tFile)
        self.tFile.close()

        if hasattr(self, "iNMDA"):
            return list(self.iNMDA)[-1], max(self.iNMDA)
