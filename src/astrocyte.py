from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
import sys, subprocess, os
from classResults import ResultsPAPModel
from utils import *
from geneManip import GENExpression
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
from textSDIO import *


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
    defaultKo = 3

    GENEDict = None

    # video option
    varMorph = ["v", "ko"]

    # Morphology
    Node = False

    def __init__(
        self,
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
        GABA=False,
        GABACount = None,
        Ko=3,
        KoSize=0.5,
        stimdelay=0,
        durStim=0.5,
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


        # set morphology parameters
        self.somaSize = somaSize
        self.bLen = bLen
        self.bWid = bWid
        self.bNum = bNum
        self.PAPWid = PAPWid
        self.branchAtten = []
        self.ComplexMorph = ComplexMorph
        self.PAPLen = PAPLen
        self.RiSec = str(RiSec)
        self.getPeriphery = True

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
        self.KoSize = KoSize

        self.Ko = Ko
        h.set_kobase(self.Ko)

        self.durStim = durStim

        # match sections to self
        # print("Match section")
        # sys.stdout.flush()

        self.seed = seed
        h.setSeed(seed)
        # print(self.PAPLen)
        if self.Node:
            PAPCount = 3
        for i in range(PAPCount):
            if self.getPeriphery:
                h.PAP = h.get_randomfinalSection(h.soma)
                self.PAP = h.PAP.sec
                # print(self.PAP)
            elif self.Node:
                h.PAP = h.get_NodeSection(h.soma,0)
                self.PAP = h.PAP.sec
            else:
                # get chunk of section as PAP
                h.PAP = h.get_randomSection(h.soma,0.3)
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
        if len(list(self.flattenPAP())) > 0 and len(self.PAPs) > 0:
            self.PAParea /= len(list(self.flattenPAP()))
            self.PAParea /= len(self.PAPs)
        # print(self.PAParea)
        self.somaArea = h.area(0.5, sec=self.soma)

        # self.allarea = 0
        # for sec in h.allsec():
        #     self.allarea += h.area(0.5, sec=sec) * sec.nseg
        # print(self.allarea)

        # NMDA setup
        self.Glu = Glu
        self.stimdelay = stimdelay

        # GABA setup
        self.GABA = GABA
        if GABA:
            if GABACount == None:
                self.GABACount = 50
            else:
                self.GABACount = GABACount

        # if 'kir2' in kwargs.keys():
        #     if kwargs['kir2'] > 200:
        #         self.dt = 0.05
        #         h.dt = self.dt


        # GENE expression setup
        self.GENEobj = GENExpression(h.allsec(), self.PAPs, kwargs)
        self.GENEDict = kwargs
        # print('set GENE manipulation')
        # sys.stdout.flush()

        # print(self.GENEDict)
        if self.multiple == 0:
            if "GluTrans" in self.GENEDict.keys() and self.GENEDict["GluTrans"] != None:
                # Can be buggy
                self.comparecount = self.GENEDict["GluTrans"]
            elif GABA:
                self.comparecount = self.GABACount
            else:
                self.comparecount = self.multiple
            
        else:
            # print('NMDAR selected')
            self.comparecount = self.multiple
            # print(self.multiple)

        # scale the number at the end
        if PAPLen > 0.3:
            self.multiple = int(
                self.multiple * (math.pi * (1 - math.e ** (1 - PAPLen / 0.3)) + 1)
            )
            # Density decreases in gaussian manner with units of PAPLen


    def channelDist(self, **channelDict):
        # change the density of a certain channel by xfold
        for channel, xfold in channelDict.items():
            # print(channel,xfold)
            self.GENEobj.alterDistribution(channel, ratioToPAP=xfold)

    def cleanMorphology(self):
        # Used for removing morphology
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

    def koClamp(self, ko=None):
        h.koclamp(ko)

    def setGEVI(self,tON,tOFF):
        h.setGEVI(tON,tOFF)

    def multiSpike(self, number=None, freq=None, KoSize=None, koclamp=False, video=False,dur = None):
        self.SpikeFreq = freq
        self.SpikeNum = number
        # print(self.SpikeFreq,self.SpikeNum)
        # sys.stdout.flush()
        if freq == 0:
            if number > 0:
                number = 1
            ISI = 0
        else:
            ISI = 1 / freq * 1e3  # change to ms
        if number == 1:
            freq = 0
            ISI = 0
            
        if KoSize == None:
            KoSize = self.KoSize
        if dur == None:
            dur = self.durStim
        if hasattr(h, "stim") and number > 0:
            h(f'stim.number = {number}')
            h(f'stim.interval = {ISI}')
            # print(h.stim.number,h.stim.interval)
        if koclamp:
            while h.t < ISI * number:
                self.koClamp(self.Ko)
                h.fadvance()
        else:
            for i in range(number):
                currTime = h.t
                self.setK(KoSize=KoSize,dur=dur)
                if video:
                    self.makeVideo(
                        self.varMorph,
                        stop=ISI * ms + currTime,
                        interval=ISI / 2 / self.dt,  # sample at half ISI ms interval
                        zoom=True,
                    )
                else:
                    h.continuerun(ISI * ms + currTime)

    def setkin(self, kin):
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

    def setNMDA_TC(self,tau1,tau2):
        if not hasattr(self,'NMDAs'):
            wMessage('NO NMDAs defined for setNMDA_TC')
            return
        if len(self.NMDAs) > 0:
            for sNMDA in self.NMDAs:
                sNMDA.tau1_0 = tau1
                sNMDA.tau2_0 = tau2

    def setNMDA_Mgblock(self,K0,delta,shift):
        if not hasattr(self,'NMDAs'):
            wMessage('NO NMDAs defined for setNMDA_Mgblock')
            return
        if len(self.NMDAs) > 0:
            for sNMDA in self.NMDAs:
                sNMDA.K0 = K0
                sNMDA.delta = delta
                sNMDA.shift = shift

    def initGABAas(self):
        if not hasattr(self, "GABAas"):
            self.GABAas = []
            if not hasattr(self, "NCs"):
                self.NCs = []
            if not hasattr(h, "slPAP"):
                h.slPAP = self.flattenPAP()
            h("slPAP {setGABAas(slPAP)}")
            # h.setGABAas(self.flattenPAP())

    def setGABAas(self):
        self.initGABAas()
        self.GABAas = list(h.GABAas)
        # print(self.GABAas)
        # sys.stdout.flush()
        self.NCs += list(h.ncGABAaList)
        for i, sGABAa in enumerate(self.GABAas):
            if self.GABA:
                # distribute the total num of GABAa equally among all patches with remainder clustered at tip
                totGABAa = len(self.GABAas)
                if i > (totGABAa - self.GABACount % totGABAa):
                    sGABAa.multiple = 1 + self.GABACount // totGABAa
                    sGABAa.isOn = 1
                else:
                    sGABAa.multiple = self.GABACount // totGABAa
                    sGABAa.isOn = 1
            else:
                sGABAa.multiple = 0
                for nc in list(h.ncGABAaList):
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
        self.GluTs = list(h.GluTs)
        # print(self.NMDAs)
        self.NCs += list(h.ncGluList)
        self.getGLTCountPAP()
        if "GluTrans" in self.GENEDict.keys() and self.Glu:
            # print(self.GENEDict)
            # print(len(self.GluTs))
            for sGluT in self.GluTs:
                if self.GENEDict["GluTrans"] != None:
                    sGluT.multiple = self.GENEDict[
                        "GluTrans"
                    ]  # Manipulation at this stage not in geneManip.py
                else:
                    # sGluT.multiple = 0
                    # print(sGluT.multiple)
                    # print(sGluT.has_loc())
                    for nc in list(h.ncGluList):
                        nc.active(False)

        else:
            for nc in list(h.ncGluList):
                nc.active(False)

    def getGLTCountPAP(self):
        self.PAPGluTCount = 0
        self.PAPGluTCount_std = 0
        
        for nc in list(h.ncGluList):
            sGLT = nc.syn()
            self.PAPGluTCount += int(sGLT.count)
            self.PAPGluTCount_std += int(sGLT.count_std)
        # print(self.PAPGluTCount,self.PAPGluTCount_std)
    

    def setStimStart(self):
        h('objref stim')
        h('stim = new NetStim(.5)')
        h(f'stim.start = {(self.initTstop + self.stimdelay) * ms}')
        h('stim.noise = 0')
        h('stim.number = 1')
        h('stim.interval = 0')

    def setTstop(self, tstop=500):
        h.tstop = tstop
        self.tstop = tstop

    def checkNetCons(self):
        print(self.NCs)
        for nc in self.NCs:
            print(f"{nc}:{nc.active()}")
            print(nc.weight[0])
            print(nc.syn())

    def initialize(self, saveState=False, video=False,kblock=False,kuptake=False,krule=None):
        # print('initializing')
        # sys.stdout.flush()
        if kblock:
            h.kbath_off()
        elif kuptake:
            h.kbath_on()
        elif type(krule) == float:
            if krule == 0:
                krule = 1/math.exp(700)
            h.kbath_rule(krule)

        self.setStimStart()
        # print('setStim')
        # sys.stdout.flush()
        # print('placing GluChannel')
        # sys.stdout.flush()
        self.setNMDAs()
        # print('placed NMDAR')
        # sys.stdout.flush()
        self.setGluTs()
        # print('placed GluT')
        # sys.stdout.flush()
        self.setGABAas()
        # print('placed GluT')
        # sys.stdout.flush()
        # self.checkNetCons()
        PAPGluT = [
            s.syn()
            for s in list(h.ncGluList)
            if s.postseg().sec in self.flattenPAP()
        ]
        # print(PAPGluT)
        recordDictArgs = {}
        if len(PAPGluT) > 0:
            recordDictArgs['sGluT'] = PAPGluT[-1]
        if self.Glu and len(self.NMDAs) > 0:
            recordDictArgs['sNMDA'] = self.NMDAs[-1]
        if self.GABA and len(self.GABAas) > 0:
            recordDictArgs['sGABA'] = self.GABAas[-1]
        self.record(**recordDictArgs)
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
        # print(list(self.KoSizePAP)[-1])
        self.RMP = list(self.vPAP)[
            -1
        ]  # consider last timepoint in initialization as RMP
        # print(self.RMP)
        if saveState:
            s = h.SaveState()
            s.save()
            with open(f"initializedState{rank}.dat", "wb") as f:
                s.fwrite(f)


    def run(self, printRes=False, video=False, koclamp=None):
        # print('running')
        # sys.stdout.flush()
        # Clamp settings
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
        if zoom:
            ps = h.plotPAP_varMorph(var, frameName, self.flattenPAP())
        else:
            ps = h.plot_varMorph(var, frameName)

    def plot_topology(self, zoom=False):
        if rank == 0:
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


    def plotMorphParms(self):
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

    def record(self, sNMDA=None, sGABA=None,sGluT=None, toFile=False):
        # Function recording each individual aspect of astrocyte variable
        h.frecord_init()
        # Save Stuff
        if sNMDA != None:
            self.iNMDA = h.Vector()
            # self.iNMDA.record(sNMDA._ref_iNMDA)
            self.iNMDA.record(sNMDA._ref_iNMDA)

            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        if sGABA != None:
            self.iGABA = h.Vector()
            self.iGABA.record(sGABA._ref_iGaba)
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
            self.GluTC1 = h.Vector()
            self.GluTC1.record(sGluT._ref_C1)
            self.GluTC2 = h.Vector()
            self.GluTC2.record(sGluT._ref_C2)
            self.GluTC3 = h.Vector()
            self.GluTC3.record(sGluT._ref_C3)
            self.GluTC4 = h.Vector()
            self.GluTC4.record(sGluT._ref_C4)
            self.GluTC4 = h.Vector()
            self.GluTC4.record(sGluT._ref_C4)
            self.GluTC5 = h.Vector()
            self.GluTC5.record(sGluT._ref_C5)
            self.GluTC6 = h.Vector()
            self.GluTC6.record(sGluT._ref_C6)
            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        if hasattr(self.PAP(0.5), "pas"):
            self.iMemPAP = h.Vector()
            self.iMemPAP.record(self.PAP(0.5)._ref_i_pas)
        if toFile:
            self.iFileMem = h.File("iFileMem.dat")
            self.iFileMem.wopen("iFileMem.dat")

        if hasattr(self.PAP(0.5), "ncx"):
            self.iNCXPAP = h.Vector()
            self.iNCXPAP.record(self.PAP(0.5)._ref_incx_ncx)
            
        self.vPAP = h.Vector()
        self.vPAP.record(self.PAP(0.5)._ref_v)
        
        self.fluorVPAP = h.Vector()
        self.fluorVPAP.record(self.PAP(0.5)._ref_dF_GEVI)

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
        self.CaiPAP = h.Vector()
        self.CaiPAP.record(self.PAP(0.5)._ref_cai)
        self.KiPAP = h.Vector()
        self.KiPAP.record(self.PAP(0.5)._ref_ki)

        if toFile:
            self.KoFile = h.File("KoFile.dat")
            self.KoFile.wopen("KoFile.dat")
            self.KiFile = h.File("KiFile.dat")
            self.KiFile.wopen("KiFile.dat")

        self.iKPAP = h.Vector()
        self.iKPAP.record(self.PAP(0.5)._ref_ik)

        if hasattr(self.PAP(0.5), "_ref_ica"):
            self.iCaPAP = h.Vector()
            self.iCaPAP.record(self.PAP(0.5)._ref_ica)
            
        if hasattr(self.PAP(0.5), "_ref_ina"):
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
                somaGluT = [s for s in self.GluTs if s.get_segment().sec == self.soma][0]
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
    
    def setSlowing(self,slow):
        h.setSlowing(slow)

    def setK(self, KoSize=None, mode="step", dur=0.5, delay=0):
        if dur == 0:
            return
        restKo = self.Ko
        if KoSize == None:
            KoSize = self.KoSize
            # print(f'set Ko to {Ko}\n')
        if mode == "pulse":
            h.continuerun(delay * ms + h.t)
            # print("setting KoSize to pulse mode")
            papk = self.getPAPK()
            h.setK(self.flattenPAP(), KoSize, papk + KoSize, 1)
            self.KoPAP[-1] = papk + KoSize
            h.fcurrent()
            h.fadvance()  # change to 1 ms?
            h.setK(self.flattenPAP(), 0, restKo, 0)
        if mode == "step":
            h.continuerun(delay * ms + h.t)
            papk = self.getPAPK()
            h.setK(self.flattenPAP(), KoSize, KoSize + papk, 2)
            h.fcurrent()
            h.continuerun(dur * ms + h.t)
            # papk = self.getPAPK()
            h.setK(self.flattenPAP(), 0, restKo, 0)
        self.KoSize = KoSize

    def setKBath(self, Ko,dur=100, delay=0,isolate=False,video=False):
        h.continuerun(delay * ms + h.t)
        papk = self.getPAPK()
        if isolate:
            h.setK(self.flattenPAP(), Ko-papk, Ko,2)
        else:
            h.setK(h.getWholetree(), Ko-papk, Ko,2)
        h.fcurrent()
        if video:
            self.makeVideo(
                self.varMorph,
                stop=dur * ms + h.t,
                interval= 100 / self.dt,  # sample at 100 ms interval
                zoom=False,
            )

        else:
            h.continuerun(dur * ms + h.t)
        self.Ko = Ko
        
    def GABABath(self,number,freq,KoSize,video=False):
        self.setStimStart()
        h.slPAP = h.SectionList(self.soma.wholetree())
        self.setGABAas()
        self.KoSize = 0

        recordDictArgs = {}
        recordDictArgs['sGABA'] = self.GABAas[-1]
        self.record(**recordDictArgs)
        h.finitialize(self.v_init)
        h.fcurrent()
        self.getkin()
        h.fcurrent()
        h.continuerun(self.initTstop*ms)
        
        self.multiSpike(number=number,freq=freq,KoSize=KoSize,video=video)
        
    def setKClearance(self,mode):
        if type(mode) == bool:
            if not mode:
                h.kbath_off()
            else:
                h.kbath_on()
        elif type(mode) == float or type(mode) == int:
            h.kbath_rule(mode)
            

    def replayK(self,fileName,isolate=False,video=False):
        df = pd.read_csv(fileName)
        baselineK = self.getPAPK()
        df['k'] += baselineK
        df['t'] += h.t
        # self.dt = 10*math.floor(math.log(max(df['t']),10) - 2)
        # h.dt = self.dt
        print(f'{self.dt=}')
        # get Max
        maxT = max(df['t'])
        maxT -= maxT % self.dt
        # remove remainder
        maxT += self.dt * 6
        # add 6 timesteps
        self.setTstop(maxT)
        
        for i,(_,row) in enumerate(df.iterrows()):
            t = int(row['t'])
            t -= (t % self.dt)
            k = row['k']
            
            # print(t,k)
            if video:
                self.makeVideo(
                    self.varMorph,
                    stop=t*ms + h.t,
                    interval= 10000 / self.dt,  # sample at 10 ms interval
                    zoom=True,
                )

            else:
                h.continuerun(t*ms)
            h.fcurrent()
            papK = self.getPAPK()
            if isolate:
                h.setK(self.flattenPAP(),k-papK,k,2)
            else:
                h.setK(h.getWholetree(),k-papK,k,2)
            h.fcurrent()
            
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
