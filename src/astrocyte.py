from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
import sys, subprocess, os
from classResults import ResultsPAPModel
from utils import *
from geneManip import GENExpression
import matplotlib.pyplot as plt
import plotly
import json
import math


class PAPModel(ResultsPAPModel):
    tstop = 260 * ms
    # parameters for three compartment model
    somaSize = 10  # Soma Size
    bLen = 30  # Branch Size
    bWid = 3
    PAPWid = 0.02
    branches = []
    branchAtten = []
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

    # K Parms
    defaultKo = 2.5

    GENEDict = None

    # video option
    varMorph = ["v", "ko"]

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
        mode=0,
        somaCheck=False,
        ComplexMorph=True,
        Glu=False,
        Ko=2.5,
        stimdelay=0,
        initTstop=150,
        dt=0.001,
        seed=0,
        PAPCount=1,
        PAPLen=0.3,
        RiSec=None,
        v_init = -85,
        **kwargs,
    ):
        # Load NEURON GUI and parameters
        from neuron import h

        h.load_file("stdgui.hoc")
        # h.load_file("./neuronHoc/params.hoc")
        # print('loaded files')
        # sys.stdout.flush()

        # set simulation parameters
        self.initTstop = initTstop * ms
        self.dt = dt

        h.dt = self.dt
        h.tstop = self.tstop
        self.v_init = v_init * mV
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
        self.branchAtten = []
        self.ComplexMorph = ComplexMorph
        self.PAPLen = PAPLen
        if PAPLen > 0.3:
            self.multiple = int(
                self.multiple * (math.pi * (1 - math.e ** (1 - PAPLen / 0.3)) + 1)
            )
            # Density decreases in gaussian manner with units of PAPLen
        self.RiSec = str(RiSec)

        if self.readHoc:
            # subsitute
            # morphology
            # NMDA setting up
            # membrane properties
            # using the HOC astrocyte.hoc library instead of
            # setting up within python 
            # print("read hoc")

            h.load_file("stdgui.hoc")
            h('{xopen("./neuronHoc/astrocyte.hoc")}')
            # print("read hoc")
            # set morphology parameters
            if not self.ComplexMorph:
                h.soma.L = self.somaSize
                h.branch.L = self.bLen
                h.branch.diam = self.bWid
                h.PAP.L = self.PAPWid

            # # set K parms
            # self.Ko = 2.5
            self.Ko = Ko

            # match sections to self
            # print("Match section")
            # sys.stdout.flush()

            self.seed = seed
            h.setSeed(seed)
            # print(self.PAPLen)
            for i in range(PAPCount):
                h.PAP = h.get_randomfinalSection(h.soma)
                self.PAP = h.PAP.sec
                h.slPAP = h.get_parent_sections(self.PAP, self.PAPLen, sec=self.PAP)
                if i == 0:
                    self.PAPs = [h.slPAP]
                else:
                    self.PAPs.append(h.slPAP)
            self.soma = h.soma
            if not self.ComplexMorph:
                self.branch = h.branch

            self.PAParea = 0
            for sec in self.flattenPAP():
                self.PAParea += h.area(0.5, sec=sec) * sec.nseg
            self.PAParea /= len(self.PAPs)
            # print(self.PAParea)
            self.somaArea = h.area(0.5, sec=self.soma)

            # self.allarea = 0
            # for sec in h.allsec():
            #     self.allarea += h.area(0.5, sec=sec) * sec.nseg
            # print(self.allarea)

        else:
            # set K parms
            self.Ko = Ko

            # Build Morphology
            self.morph()
            # print('built morphology')
            # sys.stdout.flush()

        # NMDA setup
        self.Glu = Glu
        self.stimdelay = stimdelay

        # GENE expression setup
        self.GENEobj = GENExpression(h.allsec(), self.PAPs, kwargs)
        self.GENEDict = kwargs
        # print('set GENE manipulation')
        # sys.stdout.flush()

        # print(self.GENEDict)
        if self.multiple == 0 and "GluTrans" in self.GENEDict.keys():
            self.comparecount = self.GENEDict["GluTrans"]
        else:
            # print('NMDAR selected')
            self.comparecount = self.multiple
            # print(self.multiple)

    def channelDist(self, **channelDict):
        # change the density of a certain channel by xfold
        for channel, xfold in channelDict.items():
            # print(channel,xfold)
            self.GENEobj.alterDistribution(channel, ratioToPAP=xfold)

    def koClamp(self, ko=None):
        h.koclamp(ko)

    def multiSpike(self, number=None, freq=None, Ko=None, koclamp=False, video=False):
        ISI = 1 / freq * 1e3  # change to ms
        if Ko == None:
            Ko = self.Ko
        if hasattr(h, "stim"):
            h.stim.number = number
            h.stim.interval = ISI * ms
        if koclamp:
            while h.t < ISI * number:
                self.koClamp(Ko)
                h.fadvance()
        else:
            for i in range(number):
                currTime = h.t
                self.setK(Ko=Ko)
                if video:
                    self.makeVideo(
                        self.varMorph,
                        stop=ISI * ms + currTime,
                        interval=ISI / 2 / self.dt,  # sample at half ISI ms interval
                    )
                else:
                    h.continuerun(ISI * ms + currTime)

    def setkin(self, kin):
        if self.readHoc:
            h.setkin(kin)

    def getkin(self):
        self.kin = h.getkin()

    def initNMDAs(self):
        if self.readParms:
            self.readParameters()  # readfile in parallel causes errors

        if not hasattr(self, "NMDAs"):
            self.NMDAs = []
            if not hasattr(self, "NCs"):
                self.NCs = []
            if not hasattr(h, "slPAP"):
                h.slPAP = self.flattenPAP()
            h("slPAP {setNMDAs(slPAP)}")
            # h.setNMDAs(self.flattenPAP())

    def setNMDAs(self):
        self.initNMDAs()
        if self.readHoc:
            self.NMDAs = list(h.NMDAs)
            # print(self.NMDAs)
            # sys.stdout.flush()
            self.NCs += list(h.ncNMDAList)
            for i, sNMDA in enumerate(self.NMDAs):
                if self.Glu and self.multiple != 0:
                    # distribute the total num of NMDA equally among all patches with remainder clustered at tip
                    totNMDA = len(self.NMDAs)
                    if i > (totNMDA - self.multiple % totNMDA):
                        sNMDA.multiple = 1 + self.multiple // totNMDA
                    else:
                        sNMDA.multiple = self.multiple // totNMDA
                else:
                    sNMDA.multiple = 0
                    for nc in list(h.ncNMDAList):
                        nc.active(False)

    def initGluTs(self):
        if not hasattr(self, "GluTs"):
            self.GluTs = []
            if not hasattr(self, "NCs"):
                self.NCs = []
            if not hasattr(h, "slPAP"):
                h.slPAP = self.flattenPAP()
            h("slPAP {setGluTs(slPAP)}")
            # using 
            # h.setGluTs(self.flattenPAP())
            # causes error

    def setGluTs(self):
        self.initGluTs()
        if self.readHoc:
            self.GluTs = list(h.GluTs)
            # print(self.NMDAs)
            self.NCs += list(h.ncGluList)
            # h.stim.number = 1
            if "GluTrans" in self.GENEDict.keys():
                # print(self.GENEDict)
                # print(len(self.GluTs))
                for sGluT in self.GluTs:
                    if self.GENEDict["GluTrans"] > 0:
                        sGluT.multiple = self.GENEDict[
                            "GluTrans"
                        ]  # Manipulation at this stage not in geneManip.py
                    else:
                        sGluT.multiple = 0
                        # print(sGluT.multiple)
                        # print(sGluT.has_loc())
                        for nc in list(h.ncGluList):
                            nc.active(False)

            if not self.Glu:
                for nc in list(h.ncGluList):
                    nc.active(False)

    def setStimStart(self):
        h.stim.number = 1
        h.stim.interval = 10 * ms
        h.stim.start = (self.initTstop + self.stimdelay) * ms  # Mutual Setup

    def setTstop(self, tstop):
        h.tstop = tstop
        self.tstop = tstop

    def checkNetCons(self):
        print(self.NCs)
        for nc in self.NCs:
            print(f"{nc}:{nc.active()}")
            print(nc.weight[0])
            print(nc.syn())

    def initialize(self, saveState=False, video=False,kblock=False):
        # print('initializing')
        # sys.stdout.flush()
        if not self.readHoc:
            h.ki0_k_ion = 70 * mM  # Global concentration for astrocytes from Savtchenko
            self.setK()

        # print('placing GluChannel')
        # sys.stdout.flush()
        self.setNMDAs()
        # print('placed NMDAR')
        # sys.stdout.flush()
        self.setGluTs()
        # print('placed GluT')
        # sys.stdout.flush()
        self.setStimStart()
        # print('setStim')
        # sys.stdout.flush()
        # self.checkNetCons()
        if self.Glu:
            PAPGluT = [
                s.syn()
                for s in list(h.ncGluList)
                if s.postseg().sec in self.flattenPAP()
            ]
            # print(PAPGluT)
            if len(PAPGluT) == 1:
                self.record(sNMDA=self.NMDAs[-1], sGluT=PAPGluT[0])
            else:
                self.record(sNMDA=self.NMDAs[-1])
        else:
            self.record()
        # print('setup Record')
        # sys.stdout.flush()
        h.finitialize(self.v_init)
        h.fcurrent()
        self.getkin()
        # print('initialized')
        # sys.stdout.flush()
        if video:
            self.makeVideo(self.varMorph, stop=self.initTstop)
        else:
            h.continuerun(self.initTstop * ms)
        # print(list(self.KoPAP)[-1])
        self.RMP = list(self.vPAP)[
            -1
        ]  # consider last timepoint in initialization as RMP
        # print(self.RMP)
        if saveState:
            s = h.SaveState()
            s.save()
            with open(f"initializedState{rank}.dat", "wb") as f:
                s.fwrite(f)

        if kblock:
            h.kbath_off()

    def run(self, printRes=False, video=False, koclamp=None):
        # print('running')
        # sys.stdout.flush()
        # Clamp settings
        if self.readHoc:
            if self.mode > 2:
                if self.somaCheck:
                    h.clampSwitch(3, self.currentClamp)
                    h.ic.delay = h.t
                else:
                    h.clampSwitch(2, self.currentClamp)
                    h.ic.delay = h.t

            elif self.mode > 1:
                h.clampSwitch(1, self.voltageClamp)

            elif self.mode > 0:
                h.clampSwitch(0, self.currentClamp)

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
        # print('about to run')
        # sys.stdout.flush()
        if video:
            self.plot_topology()
            self.plot_topology(zoom=True)
            self.makeVideo(self.varMorph, zoom=True)
        elif koclamp != None:
            while h.t < h.tstop:
                self.koClamp(koclamp)
                h.fadvance()
            # self.setK(Ko=initKO,mode='step',dur=self.tstop)
        else:
            try:
                h.continuerun((self.tstop) * ms)
            except RuntimeError as e:
                if "hocobj_call" in str(e):
                    if self.dt < 1e-6:
                        print("skip run")
                        vPAP = ["nan"]
                        vSoma = ["nan"]
                    else:
                        self.dt = self.dt / 10
                        h.t = self.dt
                        self.run(printRes=printRes, video=video, koclamp=koclamp)

        # print('finish run')
        if printRes:
            self.printRec()
        self.cleanMorphology()
        # print('ran simulation')
        # sys.stdout.flush()

    def getRMP(self):
        # decapreated
        # just call self.RMP
        print("function getRMP is decapreated")
        self.initialize()
        self.run()
        RMP = sum(list(self.vSoma)) / len(list(self.vSoma))
        self.RMP = RMP
        return RMP

    def makeVideo(self, var, interval=5, stop=None, zoom=False):
        if not hasattr(self, "frames"):
            self.frames = []
        if stop == None:
            stop = self.tstop
        while h.t < stop:
            if int(h.t / self.dt) % interval == 0:
                for v in var:
                    fname = f"astro{v}_{int(h.t/self.dt)}.psf"
                    self.plotWholecellVariable(
                        v, os.path.join("../morphResults/video", fname)
                    )
                    if zoom:
                        fname = f"pap{v}_{int(h.t/self.dt)}.psf"
                        self.plotWholecellVariable(
                            v, os.path.join("../morphResults/video", fname), zoom=zoom
                        )
            h.fadvance()
        for v in var:
            subprocess.call(
                f"convert -delay 2 -loop 0 ../morphResults/video/astro{v}*.psf ../morphResults/video/{v}Morph_{self.seed}_{self.Ko}.gif",
                shell=True,
            )
            if zoom:
                subprocess.call(
                    f"convert -delay 2 -loop 0 ../morphResults/video/pap{v}*.psf ../morphResults/video/{v}PAPMorph_{self.seed}_{self.Ko}.gif",
                    shell=True,
                )
        return

    def flattenPAP(self):
        # flatten self.PAPs to section list
        flattenPap = []
        for pap in self.PAPs:
            flattenPap += pap
        return h.SectionList(flattenPap)

    def plotWholecellVariable(self, var, frameName, zoom=False):
        if self.readHoc:
            if zoom:
                ps = h.plotPAP_varMorph(var, frameName, self.flattenPAP())
            else:
                ps = h.plot_varMorph(var, frameName)

    def plot_topology(self, zoom=False):
        if rank == 0:
            if self.readHoc:
                if zoom:
                    ps = h.plotPAP_topology(
                        os.path.join(
                            "../morphResults/", f"astrocyte_PAPtopology_{self.seed}.psf"
                        ),
                        self.flattenPAP(),
                    )
                else:
                    ps = h.plot_topology(
                        os.path.join(
                            "../morphResults/", f"astrocyte_topology_{self.seed}.psf"
                        ),
                        self.flattenPAP(),
                    )
            else:
                ps = h.PlotShape()
                ps.color_all(1)
                ps.color_list(self.flattenPAP(), 2)
                ps.printfile(f"{fname}.psf")

    def cleanMorphology(self):
        # Used for removing morphology
        # use when morphology is built within
        # python interface
        # print('cleaning')

        for sec in h.allsec():
            h.delete_section(sec=sec)
        # print('Remove attr')
        self.branches = []
        delattr(self, "soma")
        delattr(self, "PAP")
        # if hasattr(self,"GENEDict"):
        #     delattr(self, "GENEDict")

    def astroMem(self, compartment):
        # add astrocyte properties
        # used only when model constructed within
        # python interface
        compartment.Ra = 100
        compartment.cm = 0.8
        compartment.insert("pas")
        compartment.e_pas = self.v_init
        compartment.g_pas = 1 / 11150
        self.channels(compartment)

    def channels(self, compartment):
        # insert relevant channels
        # used only when model constructed within
        # python interface
        compartment.insert("kir2")
        compartment.insert("twik")
        compartment.insert("k_acc")
        compartment.insert("kdifl")

    def plotMorphParms(self):
        if self.readHoc:
            h.plot_varMorph("diam", "DiamMap.psf")
            h.plot_varMorph("nseg", "nsegMap.psf")

    def morph(self, isolate=False, printTopology=False):
        print("function morph is depracated")
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
        # used only when model constructed within
        # python interface and NMDAR parameters are read from file
        # fails during parallel
        
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

    def record(self, sNMDA=None, sGluT=None, toFile=False):
        # Function recording each individual aspect of astrocyte variable
        h.frecord_init()
        # Save Stuff
        if sNMDA != None:
            self.iNMDA = h.Vector()
            self.iNMDA.record(sNMDA._ref_iNMDA)
            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        if sGluT != None:
            self.iGluT = h.Vector()
            self.iGluT.record(sGluT._ref_iGluT)
            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")
            self.GluTGlu = h.Vector()
            self.GluTGlu.record(sGluT._ref_Gluout)
            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        if hasattr(self.PAP(0.5), "pas"):
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

        self.flux = h.Vector()
        self.flux.record(self.PAP(0.5)._ref_flux_k_acc)

        self.kbath = h.Vector()
        self.kbath.record(self.PAP(0.5)._ref_kbath_k_acc)

        if hasattr(self.PAP(0.5), "_ref_ena"):
            self.enaPAP = h.Vector()
            self.enaPAP.record(self.PAP(0.5)._ref_ena)

        if hasattr(self.PAP(0.5), "_ref_ecl"):
            self.eclPAP = h.Vector()
            self.eclPAP.record(self.PAP(0.5)._ref_ecl)

        if toFile:
            self.ekFile = h.File("ekFile.dat")
            self.ekFile.wopen("ekFile.dat")

        self.KoPAP = h.Vector()
        self.KoPAP.record(self.PAP(0.5)._ref_ko)
        self.NaoPAP = h.Vector()
        self.NaoPAP.record(self.PAP(0.5)._ref_nao)
        self.CloPAP = h.Vector()
        self.CloPAP.record(self.PAP(0.5)._ref_clo)
        self.KiPAP = h.Vector()
        self.KiPAP.record(self.PAP(0.5)._ref_ki)

        if toFile:
            self.KoFile = h.File("KoFile.dat")
            self.KoFile.wopen("KoFile.dat")
            self.KiFile = h.File("KiFile.dat")
            self.KiFile.wopen("KiFile.dat")

        self.iKPAP = h.Vector()
        self.iKPAP.record(self.PAP(0.5)._ref_ik)

        if hasattr(self.PAP(0.5), "_ref_icl"):
            self.iNaPAP = h.Vector()
            self.iNaPAP.record(self.PAP(0.5)._ref_ina)

        if hasattr(self.PAP(0.5), "_ref_icl"):
            self.iClPAP = h.Vector()
            self.iClPAP.record(self.PAP(0.5)._ref_icl)

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

            self.iNaSoma = h.Vector()
            self.iNaSoma.record(self.soma(0.5)._ref_ina)

            self.iClSoma = h.Vector()
            self.iClSoma.record(self.soma(0.5)._ref_icl)

            if hasattr(self, "GluTs"):
                self.iGluTSoma = h.Vector()
                somaGluT = [s for s in self.GluTs if s.get_segment().sec == self.soma][
                    0
                ]
                self.iGluTSoma.record(somaGluT._ref_iGluT)

            self.ekSoma = h.Vector()
            self.ekSoma.record(self.soma(0.5)._ref_ek)

            if toFile:
                self.iKFileSoma = h.File("iKFileSoma.dat")
                self.iKFileSoma.wopen("iKFileSoma.dat")

            if hasattr(self.soma(0.5), "pas"):
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

        if hasattr(self, "branch"):
            for i in range(10):
                self.branchAtten.append(h.Vector())
                self.branchAtten[-1].record(self.branch(i / 10.0)._ref_v)

        else:
            # print(equiDistSec)
            self.equiDistSec = self.getEquiDistSec(list(self.getPath(self.PAP)))

            for sec in self.equiDistSec:
                self.branchAtten.append(h.Vector())
                self.branchAtten[-1].record(sec._ref_v)

        self.time = h.Vector()
        self.time.record(h._ref_t)
        if toFile:
            self.tFile = h.File("tFile.dat")
            self.tFile.wopen("tFile.dat")

    def getEquiDistSec(self, path, cutLen=10):
        totLen = h.distance(self.PAP(1), sec=self.soma)
        pathLenList = [totLen * i / cutLen for i in range(1, cutLen)]
        j = 0
        i = 0
        path.reverse()  # Flip from soma to PAP
        # print(pathLenList)
        equiDistSec = [self.soma(0.5)]
        secRight = 0
        while secRight < totLen:
            currSec = path[i]
            secLeft = h.distance(currSec(0), sec=self.soma)
            secRight = h.distance(currSec(1), sec=self.soma)
            # print(secLeft,secRight)
            # print(j)
            # print(pathLenList[j])
            if secLeft < pathLenList[j] and secRight > pathLenList[j]:
                x = (pathLenList[j] - secLeft) / (secRight - secLeft)
                equiDistSec.append(currSec(x))
                j += 1
            if j < cutLen - 1:
                if secRight < pathLenList[j]:
                    i += 1
            else:
                break
        equiDistSec.append(self.PAP(0.5))
        return equiDistSec

    def getPath(self, section):
        # get section list from PAP to soma
        currentSection = h.SectionRef(sec=section)
        sl = h.SectionList()
        while currentSection.has_parent():
            sl.append(currentSection.sec)
            currentSection = h.SectionRef(sec=currentSection.parent)
        return sl
    

    def setK(self, Ko=None, restKo=2.5, mode="step", dur=0.5, delay=0):
        if Ko == None:
            Ko = self.Ko
            # print(f'set Ko to {Ko}\n')
        else:
            self.Ko = Ko
        # print(list(self.PAPs))
        if self.readHoc:
            if mode == "pulse":
                h.continuerun(delay * ms + h.t)
                # print("setting Ko to pulse mode")
                papk = self.getPAPK()
                h.setK(self.flattenPAP(), Ko, papk + Ko, 1)
                self.KoPAP[-1] = papk + Ko
                h.fcurrent()
                h.fadvance()  # change to 1 ms?
                h.setK(self.flattenPAP(), 0, restKo, 0)
            if mode == "step":
                h.continuerun(delay * ms + h.t)
                papk = self.getPAPK()
                h.setK(self.flattenPAP(), Ko, Ko + papk, 2)
                h.fcurrent()
                h.continuerun(dur * ms + h.t)
                # papk = self.getPAPK()
                h.setK(self.flattenPAP(), 0, restKo, 0)
            self.Ko = Ko

    def LambdaEq(self, ra, rm, d):
        return (rm * d / ra / 4) ** 0.5  # um

    def spaceConstant(self):
        # h.clampSwitch(4, self.voltageClamp)
        # h.run()
        self.initialize()
        Rm_Rd = [
            (seg.sec.Ra, 1 / seg.sec.g_pas, seg.sec.diam) for seg in self.equiDistSec
        ]
        LambdaList = [self.LambdaEq(*vals) for vals in Rm_Rd]
        LenList = [h.distance(self.soma(0.5), seg) for seg in self.equiDistSec]
        h.clampSwitch(4, self.voltageClamp)
        self.run()
        VList = [list(v)[-1] for v in self.branchAtten]
        return LambdaList, VList, LenList

    def getSecbyName(self, secname):
        for sec in h.allsec():
            if sec.hname() == secname:
                return sec
        else:
            return None

    def measureRiAll(self, parallel=False):
        if self.readHoc:
            if parallel:
                if self.RiSec != None:
                    RiSec = self.getSecbyName(self.RiSec)
                    RiSec.insert("inputRes")
                    h.measure_input_resistance(sec=RiSec)
                    self.RiDict = {str(RiSec): float(RiSec.Ri_inputRes)}
                    if len(self.RiDict) > 0:
                        self.saveRiDict()
            else:
                h.getAllRi()
                h.plot_varMorph("Ri", "RiMap.psf")

    def mapRi(self, sectionDict):
        for k, v in sectionDict.items():
            RiSec = self.getSecbyName(k)
            RiSec.insert("inputRes")
            RiSec.Ri_inputRes = v
        h.plot_varMorph("Ri_inputRes", "RiMap.psf")

    def saveRiDict(self):
        for sName, v in self.RiDict.items():
            with open(
                os.path.join("../results/paperRes", f"RiRes{sName}.json"), "w"
            ) as ofile:
                json.dump(self.RiDict, ofile)

    def getPAPK(self):
        if self.readHoc:
            return h.getPAPK(self.PAP, sec=self.PAP)

    def printRec(self):
        # used to write each recorded aspect into .dat file
        #  need to update to fit new record function
        # fails under parallel
        #
        # Just call self.attr instead
        
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
