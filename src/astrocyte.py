from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
import sys
from classResults import ResultsPAPModel
from utils import *
from geneManip import GENExpression

class PAPModel(ResultsPAPModel):
    tstop = 200 * ms
    initTstop = 50 * ms
    dt = 0.001 * ms
    celsius = 37
    v_init = -85 * mV
    somaSize = 10  # Soma Size
    bLen = 30  # Branch Size
    bWid = 3
    PAPWid = 0.02
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
            readHoc=True,
            PAPWid=0.02,
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
            Ko=2.5,
            **kwargs,
    ):
        # Load NEURON GUI and parameters
        from neuron import h

        h.load_file("stdgui.hoc")
        h.load_file("./neuronHoc/params.hoc")
        # print('loaded files')

        # Set simulation parameters
        h.tstop = self.tstop
        h.dt = self.dt
        h.celsius = self.celsius
        h.v_init = self.v_init
        # print('set sim parms')

        # set NMDA
        self.multiple = multiple
        

        # set clamp parms
        self.mode = mode
        self.voltageClamp = voltageClamp
        self.currentClamp = currentClamp
        self.somaCheck = somaCheck

        self.readHoc = readHoc

        # set morphology parameters
        self.somaSize = somaSize
        self.bLen = bLen
        self.bWid = bWid
        self.bNum = bNum
        self.PAPWid = PAPWid

        if readHoc:
            # subsitute
            # [x] morphology
            # [x] NMDA setting up
            # [x] membrane properties
            
            h.load_file("stdgui.hoc")
            h('xopen("./neuronHoc/astrocyte.hoc")')
            # set morphology parameters
            h.soma.L = self.somaSize
            h.branch.L = self.bLen
            h.branch.diam = self.bWid 
            h.PAP.L = self.PAPWid

            # set K parms
            self.Ko = 2.5

            # match sections to self
            self.PAP = h.PAP
            self.soma = h.soma
            self.branch = h.branch


        else:
            # set K parms
            self.Ko = Ko

            # Build Morphology
            self.morph()
            # print('built morphology')
            # sys.stdout.flush()

        # NMDA setup
        self.Glu = Glu
        for sec in h.allsec():
            if not hasattr(self, "GENEDict"):
                GENExpression(sec, kwargs)
                self.GENEDict = kwargs
        # print('set GENE manipulation')

    def initNMDAs(self):
        if self.readParms:
            self.readParameters()  # readfile in parallel causes errors
            
        if not hasattr(self, "NMDAs"):
            self.NMDAs = []
            self.NCs = []
            
    def setNMDAs(self):
        self.initNMDAs()
        if self.readHoc:
            self.NMDAs.append(h.sNMDA)
            self.NCs.append(h.nc)
            h.stim.start = self.initTstop * ms + h.dt
            h.nc.weight[0] = self.SynWeight
            if self.Glu:
                h.sNMDA.multiple = self.multiple
            else:
                h.sNMDA.multiple = 0
        else:            
            # Create the synaptic NMDA conductance
            stim = h.NetStim(self.PAP(0.5))
            stim.interval = 1
            stim.number = 1
            stim.start = (self.initTstop + 1) * ms
            stim.noise = 0

            # print(range(self.multiple - len(self.NMDAs)))
            self.NMDAs.append(self.nmda())
            if self.Glu:
                self.NCs.append(
                    h.NetCon(stim, self.NMDAs[-1])
                )  # Must be in outer later with python address allocated
                self.NCs[-1].weight[0] = self.SynWeight
                self.NCs[-1].delay = 0
        
                

        # print('placed NMDAR')
        # sys.stdout.flush()
        # print('initialized')
        # sys.stdout.flush()

    def initialize(self, saveState=False):
        # print('initializing')
        # sys.stdout.flush()
        if not self.readHoc:
            h.ki0_k_ion = 70 * mM  # Global concentration for astrocytes from Savtchenko
            self.setK()
        self.setNMDAs()
        self.record(sNMDA=self.NMDAs[-1])
        h.finitialize(self.v_init)
        h.fcurrent()

        
        h.continuerun(self.initTstop * ms)
        self.RMP = sum(list(self.vPAP))/len(list(self.vPAP)) # consider RMP for local or global
        if saveState:
            s = h.SaveState()
            s.save()
            with open(f"initializedState{rank}.dat", "wb") as f:
                s.fwrite(f)

    def run(self, printRes=False):
        # Clamp settings
        if self.readHoc:
            if self.mode > 2:
                if self.somaCheck:
                    h.clampSwitch(3,self.currentClamp)
                else:
                    h.clampSwitch(2,self.currentClamp)
                    
            elif self.mode >1:
                h.clampSwitch(1,self.voltageClamp)

            elif self.mode>0:
                h.clampSwitch(0,self.currentClamp)
                    

        else:
            if self.mode > 2:
                # Step Current
                if self.somaCheck:
                    ic = h.IClamp(self.soma(0.5))
                else:
                    ic = h.IClamp(self.PAP(0.5))
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
                ic = h.IClamp(self.PAP(0.5))
                ic.dur = 0.002
                ic.delay = 10  # ms starts with glutamate
                ic.amp = self.currentClamp * 0.001  # nA current injection (1 pA)
            # print('clamp experiment setup')
            # sys.stdout.flush()
            
        try:
            h.continuerun((self.tstop) * ms)
        except RuntimeError as e:
            if "hocobj_call" in str(e):
                print("skip run")
                vPAP = ['nan']
                vSoma = ['nan']
                
        
        if printRes:
            self.printRec()
        self.cleanMorphology()
        # print('ran simulation')
        # sys.stdout.flush()

    def getRMP(self):
        # decapreated
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
        self.PAP = None
        if hasattr(self,"GENEDict"):
            delattr(self, "GENEDict")

    def astroMem(self, compartment):
        # add astrocyte properties
        compartment.Ra = 100
        compartment.cm = 0.8
        compartment.insert("pas")
        compartment.e_pas = self.v_init
        compartment.g_pas = 1 / 11150
        self.channels(compartment)

    def channels(self, compartment):
        # insert relevant channels
        compartment.insert("kir2")
        compartment.insert("twik")
        compartment.insert("k_acc")
        compartment.insert("kdifl")

    def morph(self, isolate=False, printTopology=False):
        # Access the PAP object
        if not hasattr(self, "PAP"):
            self.PAP = h.Section(name="PAP")
            self.astroMem(self.PAP)

        # Set astrocyte leaf membrane parameters
        self.PAP.L = 0.3
        self.PAP.diam = self.PAPWid
        self.PAP.nseg = 1

        # h.psection(sec=self.PAP)
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
                    self.PAP.connect(self.branches[-1])
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
        sNMDA = h.Exp5NMDA(self.PAP(0.5))

        if self.readParms:
            # load files if parameters are read
            sNMDA.tau2_0 = self.Tau2_0
            sNMDA.a2 = self.A2
            sNMDA.b2 = self.B2
            sNMDA.tau3_0 = self.Tau3_0
            sNMDA.a3 = self.A3
            sNMDA.b3 = self.B3
            sNMDA.delta = self.DELTA
            sNMDA.multiple = self.multiple

        return sNMDA

    def record(self, sNMDA=None, toFile=False):
        h.frecord_init()
        # Save Stuff
        if sNMDA != None:
            self.iNMDA = h.Vector()
            self.iNMDA.record(sNMDA._ref_i)
            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        self.iMemPAP = h.Vector()
        self.iMemPAP.record(self.PAP(0.5)._ref_i_pas)
        if toFile:
            self.iFileMem = h.File("iFileMem.dat")
            self.iFileMem.wopen("iFileMem.dat")

        self.vPAP = h.Vector()
        self.vPAP.record(self.PAP(0.5)._ref_v)

        if toFile:
            self.vFile = h.File("vFile.dat")
            self.vFile.wopen("vFile.dat")

        self.ekPAP = h.Vector()
        self.ekPAP.record(self.PAP(0.5)._ref_ek)

        if toFile:
            self.ekFile = h.File("ekFile.dat")
            self.ekFile.wopen("ekFile.dat")

        self.KoPAP = h.Vector()
        self.KoPAP.record(self.PAP(0.5)._ref_ko)
        self.KiPAP = h.Vector()
        self.KiPAP.record(self.PAP(0.5)._ref_ki)

        
        if toFile:
            self.KoFile = h.File("KoFile.dat")
            self.KoFile.wopen("KoFile.dat")
            self.KiFile = h.File("KiFile.dat")
            self.KiFile.wopen("KiFile.dat")

        self.iKPAP = h.Vector()
        self.iKPAP.record(self.PAP(0.5)._ref_ik)


        if toFile:
            self.iKFilePAP = h.File("iKFilePAP.dat")
            self.iKFilePAP.wopen("iKFilePAP.dat")

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

            self.iMemSoma = h.Vector()
            self.iMemSoma.record(self.soma(0.5)._ref_i_pas)


            self.KoSoma = h.Vector()
            self.KoSoma.record(self.soma(0.5)._ref_ko)
            self.KiSoma = h.Vector()
            self.KiSoma.record(self.soma(0.5)._ref_ki)

            if toFile:
                self.KoFileSoma = h.File("KoFileSoma.dat")
                self.KoFileSoma.wopen("KoFileSoma.dat")
                self.KiFileSoma = h.File("KiFileSoma.dat")
                self.KiFileSoma.wopen("KiFileSoma.dat")

        self.time = h.Vector()
        self.time.record(h._ref_t)
        if toFile:
            self.tFile = h.File("tFile.dat")
            self.tFile.wopen("tFile.dat")

    def setK(self, Ko=None,restKo=2.5,mode='pulse',dur=500):
        if Ko == None:
            Ko = self.Ko

        if self.readHoc:
            if mode == 'pulse':
                h.setK(Ko,restKo)
            if mode == 'step':
                h.setK(Ko,Ko)
                h.continuerun(dur * ms + h.t)
                h.setK(Ko,restKo)

        else:
            if mode == 'pulse':
                h.init()  # quite important
                for sec in h.allsec():
                    # h.psection(sec=sec)
                    if sec == self.PAP:
                        for seg in sec:
                            seg.ko = Ko * mM
                        # print('set PAP Ko')
                    else:
                        for seg in sec:
                            seg.ko = self.defaultKo * mM

                    sec.ek = -90 * mV
            # print('Potassium Parms')

    def printRec(self):
        #  need to update to fit new record function
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
