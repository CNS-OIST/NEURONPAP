import os

os.environ["NEURON_MODULE_OPTIONS"] = "-nogui"
from neuron import h, load_mechanisms
from neuron.units import mM, mV, ms
import sys
from classResults import ResultsPAPModel
from utils import *
from geneManip import GENExpression
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import numpy as np
from textSDIO import *
from plot_shape import *
from global_labels import gl
import random
from importlib import reload
from scipy.optimize import minimize


class PAPModel(ResultsPAPModel):
    tstop = 260 * ms
    # parameters for three compartment model
    somaSize = 10  # Soma Size
    pbLen = None  # Branch diam
    bWid = 3
    PAPWid = 0.02
    branches = []
    branchAtten = []
    # NMDA parms
    multiple = int()
    readParms = False
    record_single_synapse = True
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
        pbLen=2,
        voltageClamp=40,
        somaSize=10,
        currentClamp=2,
        multiple=None,
        mode=0,
        somaCheck=False,
        ComplexMorph=True,
        Glu=False,
        GABA=False,
        GABACount=None,
        gapCount=None,
        KoSize=0.5,
        stimdelay=0,
        durStim=0.5,
        initTstop=150,
        dt=0.001,
        seed=0,
        PAPCount=1,
        PAPLen=0.3,
        RiSec=None,
        v_init=-85,
        g_pas=0.69,
        shell=0,
        shift_PAP=0.7,
        getPeriphery=True,
        sec_range=None,
        **kwargs,
    ):
        from neuron import h

        h.nrn_load_dll("nrniv.so")

        h.load_file("stdgui.hoc")
        # h.load_file("./neuronHoc/params.hoc")
        # print('loaded files')
        # cells.plot
        # sys.stdout.flush()

        # set simulation parameters
        self.initTstop = initTstop * ms
        self.dt = dt

        h.dt = self.dt
        h.tstop = self.tstop
        self.v_init = v_init * mV
        h.v_init = self.v_init

        # set shell layer
        self.shell = shell
        # print('set sim parms')
        # Set gap count
        self.shift_PAP = shift_PAP
        self.gapcount = gapCount

        # set NMDA
        self.multiple = multiple

        # set clamp parms
        self.mode = mode
        self.voltageClamp = voltageClamp
        self.currentClamp = currentClamp
        self.somaCheck = somaCheck

        # set morphology parameters
        self.somaSize = somaSize
        self.pbLen = pbLen
        self.bWid = bWid
        self.bNum = bNum
        self.PAPWid = PAPWid
        self.branchAtten = []
        self.ComplexMorph = ComplexMorph
        self.PAPLen = PAPLen
        self.RiSec = str(RiSec)
        self.getPeriphery = getPeriphery

        # only used for dual patch protocol
        self.g_pas = g_pas

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
            h.branch.L = self.pbLen
            h.branch.diam = self.bWid
            h.PAP.L = self.PAPWid

        # # set K parms
        self.KoSize = KoSize

        self.Ko = float(h.ko0)
        h.set_kobase(self.Ko)

        self.durStim = durStim

        # match sections to self
        # print("Match section")
        # sys.stdout.flush()

        self.seed = seed
        random.seed(seed)
        h.setSeed(seed)
        # print(self.PAPLen)
        self.PAPCount = PAPCount
        if self.Node:
            PAPCount = 3
        for i in range(PAPCount):
            if self.getPeriphery:
                h.PAP = h.get_randomfinalSection(h.soma)
                h.PAP = h.shiftPAP(h.PAP, self.shift_PAP, sec=h.PAP.sec)
                self.PAP = h.PAP.sec
                # print(self.PAP)
            elif self.Node:
                h.PAP = h.get_NodeSection(h.soma, 0)
                self.PAP = h.PAP.sec
            else:
                # get chunk of section as PAP
                h.PAP = h.get_randomSection(h.soma, 0.3)
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
        self.somaArea = 0
        for seg in h.soma:
            self.somaArea += seg.area()

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

        # GENE expression setup
        self.GENEobj = GENExpression(h.allsec(), self.PAPs, kwargs)
        self.GENEDict = kwargs
        # print('set GENE manipulation')
        # sys.stdout.flush()

        # print(self.GENEDict)
        if self.multiple is None:
            if "GluTrans" in self.GENEDict.keys() and self.GENEDict["GluTrans"] != None:
                # Can be buggy
                self.comparecount = self.GENEDict["GluTrans"]
            elif GABA:
                self.comparecount = self.GABACount
            elif self.gapcount is not None:
                # 0 should be also included
                self.comparecount = self.gapcount
            elif "nakpump" in self.GENEDict.keys() and self.GENEDict["nakpump"] != None:
                self.comparecount = self.GENEDict["nakpump"]
            else:
                self.comparecount = self.multiple

            self.multiple = 0
        else:
            # print('NMDAR selected')
            self.comparecount = self.multiple
            # print(self.multiple)

        # print(self.comparecount)
        # scale the number at the end
        if PAPLen > 0.3:
            self.multiple = int(
                self.multiple * (math.pi * (1 - math.e ** (1 - PAPLen / 0.3)) + 1)
            )
            # Density decreases in gaussian manner with units of PAPLen
        if sec_range is not None:
            self.sec_range = sec_range

    def get_PAPName(self):
        self.PAP_name = str(self.PAP)
        return str(self.PAP)

    def kir_rect_off(self):
        h.kir_rect_off(1)

    def setIKSize2KoSize(self, *val, analytical=True):
        if not hasattr(self, "IKSize"):
            wMessage("IKSize not defined")
        elif hasattr(self, "PAP_properties") and analytical:
            self.PAP_properties[-1]["kir_count"] = 0
            vhalfl = self.PAP.vhalfl_kir2
            kl = self.PAP.kl_kir2
            v = self.RMP
            linf = 1 / (1 + exp((v - vhalfl) / kl))
            A = 0.09534626
            I = (
                50 * A * sqrt(self.KoSize) * prp_cell.PAP_properties[-1]["kir_count"]
            ) * (v - np.log(self.KoSize / 120))
        else:
            self.KoSize = val[0]

    def fitIKSize(self, goalSize=-0.05):
        self.goal_IK = goalSize
        curr_index = len(list(self.iKPAP))
        self.stable_current = list(self.iKPAP)[-1]
        if self.stable_current < 0 or self.stable_current - goalSize < 0:
            print(f"stable current too small {self.stable_current}")
        self.Ko = list(self.KoPAP)[-1]

        res = minimize(self.getKoSize4IKSize, (1), method="Nelder-Mead")
        print(res.x)
        self.KoSize = res.x[0]
        currTime = h.t
        curr_index = len(list(self.vPAP))
        h.continuerun(currTime + 10)
        self.setKPoint(dur=10)
        self.fit_maxResponse = max(list(self.vPAP)[curr_index:])
        self.fit_minResponse = min(list(self.vPAP)[curr_index:])

    def getKoSize4IKSize(self, x):
        # make segment specific
        Ko = x[0] + self.Ko - self.getPAPK()
        # print(x[0], self.getPAPK(), Ko)
        curr_index = len(list(self.iKPAP))
        # print(list(self.iKPAP)[curr_index - 1] * self.PAP(0.5).area() / 1e8)
        if self.PAP.L > 1:
            dur = 10
        else:
            dur = 5

        self.setKPoint(KoSize=Ko, dur=dur)
        # mA/cm2 -> mA
        current = (
            max(
                np.array(list(self.iKPAP)[curr_index:]) - self.stable_current,
                key=abs,
            )
            * self.PAP(0.5).area()
            / 1e8
        )
        # mA -> nA
        current *= 1e6
        if x[0] > 0:
            print(x[0], self.getPAPK(), Ko, current)
        self.setKPoint(KoSize=self.Ko - self.getPAPK(), dur=1)
        h.continuerun(h.t + 9)
        # print(current)
        # print((self.goal_IK - current) ** 2)
        if self.PAP.L > 1:
            factor = 1e6
        else:
            factor = 1e2

        return ((self.goal_IK - current) * factor) ** 2

    def savePAPProp(self, name=False):
        self.PAP_properties = []
        h.finitialize()
        for i, pap in enumerate(self.flattenPAP()):
            self.PAP_properties.append({})
            self.PAP_properties[-1]["L"] = pap.L
            self.PAP_properties[-1]["diam"] = pap.diam
            self.PAP_properties[-1]["nseg"] = pap.nseg
            self.PAP_properties[-1]["area"] = 0
            self.PAP_properties[-1]["kir_count"] = 0
            for seg in pap:
                self.PAP_properties[-1]["area"] += seg.area()
                self.PAP_properties[-1]["kir_count"] += pap.count_kir2
            self.PAP_properties[-1]["local_kir_count"] = pap(0.5).count_kir2
            self.PAP_properties[-1]["ecs"] = pap.fhspace_k_acc
            self.PAP_properties[-1]["adj_diam"] = h.adjacent_total_diam(sec=pap)
            self.PAP_properties[-1]["distance"] = h.distance(pap(0.5))
            self.PAP_properties[-1]["distance"] = h.distance(pap(0.5))
            self.PAP_properties[-1]["diff_tau"] = pap.tauk_0_k_acc
            if name:
                self.PAP_properties[-1]["name"] = str(pap)

    def setPAPNearSoma(self, onSoma=False, onPB=True, diam=1, dist_radius=2):
        if onSoma:
            return self.setPAP2Soma()
        elif onPB:
            self.PAP = h.random_pb()
            self.PAP = self.PAP.sec
            h.convertPB2slPAP(sec=self.PAP)
            self.PAPs = [h.slPAP]
        else:
            h.PAP = h.get_midAstrocytePAP(diam, dist_radius)
            self.PAP = h.PAP.sec
            h.slPAP = h.get_parent_sections(self.PAP, self.PAPLen, sec=self.PAP)
            self.PAPs = [h.slPAP]

        # wMessage(f'Did not find PAP candidate with {diam=} in {radius=}')
        #
        #

    def nernstK(self, ko, kin):
        R = 8.314  # Gas constant J/(mol*K)
        F = 96485  # Faraday constant C/mol
        T = h.celsius + 273.15  # Convert to Kelvin

        return (R * T) / (F) * np.log(ko / kin)

    def set_kdfl_iter(self):
        self.set_diff_ki(True)

    def set_diff_ki(self, on):
        if on:
            if str(self.PAP) == "soma":
                h.set_ki_sim(h.all_pb())
            else:
                if "Glia" in str(self.PAP):
                    h.set_ki_sim(h.all_child(self.PAP, 2, sec=self.PAP))
                h.set_ki_sim(h.getPath(self.PAP, sec=self.PAP))

    def set_gapBath(self, on):
        if on:
            h.set_gapv(self.nernstK(10, 120) * 1000)
        else:
            h.set_gapv(self.v_init)

    def plot_path_attenuation(self, parmName="voltageClamp", origin=None):
        if not origin:
            origin = self.soma

        h.load_file("./neuronHoc/paths.hoc")
        h.get_paths_away_from_soma(origin, sec=origin)
        self.paths_away = plot_paths(
            "v",
            origin,
            h.paths_away,
            fname=f"soma_attenuation_{getattr(self,parmName)}",
        )
        self.paths_toward, _ = plot_combined(
            "v",
            origin,
            h.paths_away,
            h.path_toward,
            fName=f"combined_v_{origin=}_{getattr(self,parmName)}",
        )

    def setDualPatch(self):
        h.tstop = 50
        # set g_pas
        for sec in h.allsec():
            for seg in sec:
                setattr(seg, "g_pas", self.g_pas)

        h.clampSwitch(5, self.voltageClamp)
        self.lenUnits = h.lenUnits
        self.soma_L = getattr(self.soma, "L")
        # wMessage(f'Did not find PAP candidate with {diam=} in {radius=}')

    def getDualPatch_lambda(self):
        # get v atten
        tmpV = []
        for seg in self.soma:
            tmpV.append(getattr(seg, "v"))
        self.soma_atten = tmpV
        maxVal = max(tmpV) - self.RMP
        self.spaceConstant = self._find_interpolate(
            (np.array(tmpV) - self.RMP) / maxVal, 1 / np.e
        )

    def _find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        if value - array[idx] > 0:
            return idx, idx + 1
        else:
            return idx - 1, idx

    def _find_interpolate(self, array, value):
        id_start, id_end = self._find_nearest(array, value)
        b = array[id_end] - (array[id_end] - array[id_start]) * id_start
        return (value - b) * h.lenUnits / (array[id_end] - array[id_start])

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
        if hasattr(self, "soma"):
            delattr(self, "soma")
        if hasattr(self, "PAP"):
            delattr(self, "PAP")
        # if hasattr(self,"GENEDict"):
        #     delattr(self, "GENEDict")

    def koClamp(self, ko=None):
        h.koclamp(ko)

    def ko_sim(self, on):
        if on:
            h.ko_sim_on()
        else:
            h.ko_sim_off()

    def setGEVI(self, tON, tOFF):
        h.setGEVI(tON, tOFF)

    def multiSpike(
        self,
        number=None,
        freq=None,
        KoSize=None,
        koclamp=False,
        video=False,
        dur=0.5,
        amp=None,
        delay=0,
        point=True,
    ):
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
            h(f"stim.number = {number}")
            h(f"stim.interval = {ISI}")
            currTime = int(h.t / self.dt) * self.dt
            h(f"stim.start = {currTime + delay}")
            # print(h.stim.number,h.stim.interval)
            #
        if self.cvode:
            maxCVODEstep = h.cvode.maxstep()
            h.cvode.maxstep(0.1)

        if amp is not None:
            for nc in self.NCs:
                nc.weight[0] = amp
        if koclamp:
            self.koClamp(self.Ko)
            h.continuerun(ISI * number)

        else:
            currTime = int(h.t / self.dt) * self.dt
            for i in range(number):
                if point:
                    setting_kfunc = self.setKPoint
                else:
                    setting_kfunc = self.setK
                setting_kfunc(KoSize=KoSize, dur=dur, delay=delay if i == 0 else 0)
                if video:
                    self.makeVideo(
                        self.varMorph,
                        stop=ISI * ms + currTime,
                        frame_num=self.tstop
                        / ISI
                        * 2,  # sample at half ISI ms interval
                        zoom=True,
                    )
                else:
                    h.continuerun(ISI * ms * (i + 1) + currTime)
        if self.cvode:
            h.cvode.maxstep(maxCVODEstep)

    def TBS(
        self,
        KoSize=None,
        video=False,
        dur=0.5,
        amp=None,
        initvoltageClamp=True,
        delay=0,
    ):
        if KoSize == None:
            KoSize = self.KoSize
        if dur == None:
            dur = self.durStim

        # custom initialization
        #
        # initSpike at current T
        totalT = [int(self.initTstop)]
        while totalT[-1] < self.tstop:
            for _ in range(3):
                # 4 spikes of 100 Hz
                totalT.append(totalT[-1] + 10)

            # rest of 5 Hz
            totalT.append(totalT[-1] + 200)

        self.initialize(video=video, voltageClamp=initvoltageClamp, TBS=totalT)
        if delay > 0:
            h.continuerun(delay + h.t - self.dt)

        if amp is not None:
            for nc in self.NCs:
                nc.weight[0] = amp
        else:
            for t in totalT:
                self.setK(KoSize=KoSize, dur=dur)
                if video:
                    self.makeVideo(
                        self.varMorph,
                        stop=t,
                        frame_num=self.tstop / 10,  # sample at half 10 ms interval
                        zoom=True,
                    )
                else:
                    h.continuerun(t)

    def setkin(self, kin):
        h.setkin(kin)

    def getkin(self):
        self.kin = h.getkin()

    def setGap(self):
        for _, sGap in enumerate(list(h.gaplist)):
            if self.gapcount is not None:
                sGap.multiple = self.gapcount

    def checkGap(self):
        for _, sGap in enumerate(h.gaplist):
            print(self.gapcount, sGap.multiple, sGap.g)

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
            if self.Glu and self.multiple != 0 and self.multiple != None:
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

    def setNMDA_TC(self, *args):
        names = ["tau1_0", "tau2_0"]
        if not hasattr(self, "NMDAs"):
            wMessage("NO NMDAs defined for setNMDA_TC")
            return
        if len(self.NMDAs) > 0:
            for sNMDA in self.NMDAs:
                for i, parm in enumerate(args):
                    if parm:
                        setattr(sNMDA, names[i], parm)

    def setNMDA_Mgblock(self, *args):
        names = ["K0", "delta", "shift"]
        if not hasattr(self, "NMDAs"):
            wMessage("NO NMDAs defined for setNMDA_Mgblock")
            return
        if len(self.NMDAs) > 0:
            for sNMDA in self.NMDAs:
                for i, parm in enumerate(args):
                    if parm:
                        setattr(sNMDA, names[i], parm)

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

            # if hasattr(self, "shell_synapse"):
            #    for nc in list(h.ncGluList):
            #        nc.active(False)

            #    import random

            #    self.shell_synapse_glut = []
            #    random.seed(self.seed)
            #    for nc in random.sample(list(h.ncGluList), k=self.shell_synapse):
            #        nc.active(True)
            #        self.shell_synapse_glut.append(nc)

    def setGLT_TC(self, *args):
        names = ["tau1", "tau2"]
        if "GluTrans" not in self.GENEDict.keys() or self.GENEDict["GluTrans"] == None:
            wMessage("NO GluTs defined for setNMDA_TC")
            return
        if len(h.ncGluList) > 0:
            for s in list(h.ncGluList):
                sGLT = s.syn()
                for i, parm in enumerate(args):
                    if parm:
                        setattr(sGLT, names[i], parm)

    def getGLTCountPAP(self):
        self.PAPGluTCount = 0
        self.PAPGluTCount_std = 0

        for nc in list(h.ncGluList):
            sGLT = nc.syn()
            self.PAPGluTCount += float(sGLT.count)
            self.PAPGluTCount_std += float(sGLT.count_std)
        self.PAPGluTCount = self.PAPGluTCount
        self.PAPGluTCount_std = self.PAPGluTCount_std

    def getKirCountPAP(self):
        self.PAPKirCount = 0
        self.PAPKirCount_std = 0

        for pap in self.PAPs:
            for sec in pap:
                for seg in sec:
                    self.PAPKirCount += float(seg.kir2.count)
                    self.PAPKirCount_std += float(seg.kir2.count_std)
        self.PAPKirCount = self.PAPKirCount
        self.PAPKirCount_std = self.PAPKirCount_std

    def setStimStart(self):
        h("objref stim")
        h("stim = new NetStim(.5)")
        h(f"stim.start = {(self.initTstop+self.stimdelay) * ms}")
        h("stim.noise = 0")
        h("stim.number = 1")
        h("stim.interval = 0")

    def setVecStim(self, time):
        # rewrite stim object
        h("objref stimTime")
        h("stimTime = new Vector()")
        h.stimTime.from_python(time)
        h("objref stim")
        h("stim = new VecStim(.5)")
        h("stim.play(stimTime)")

    def NaKpumpOn(self, state):
        if state:
            h.setNak_pump(1)
        else:
            h.setNak_pump(0)

    def setTstop(self, tstop=500):
        h.tstop = tstop
        self.tstop = tstop

    def checkNetCons(self):
        print(self.NCs)
        for nc in self.NCs:
            print(f"{nc}:{nc.active()}")
            print(nc.weight[0])
            print(nc.syn())

    def set_cvode(self, force_print_progress=False):
        self.cvode = False
        if hasattr(h, "cvode"):
            self.cvode = True
            if size < 2 or force_print_progress:
                h.print_progress(self.tstop)

    def initialize(
        self,
        saveState=False,
        video=False,
        kblock=False,
        kuptake=False,
        krule=None,
        voltageClamp=False,
        TBS=None,
        force_print_progress=False,
    ):
        self.set_cvode(force_print_progress=force_print_progress)
        if hasattr(h, "cvode"):
            voltageClamp = False
        # print('initializing')
        # sys.stdout.flush()
        if kblock:
            h.kbath_off()
        elif kuptake:
            h.kbath_on()
        elif type(krule) == float:
            if krule == 0:
                krule = 1 / math.exp(700)
            h.kbath_rule(krule)

        if TBS:
            self.setVecStim(TBS)
        else:
            self.setStimStart()
        # print('setStim')
        # sys.stdout.flush()
        # print('placing GluChannel')
        # sys.stdout.flush()

        self.setGap()
        # print('placed GAP')
        # sys.stdout.flush()
        self.setNMDAs()
        # print('placed NMDAR')
        # sys.stdout.flush()
        self.setGluTs()
        # print('placed GluT')
        # sys.stdout.flush()
        self.setGABAas()
        # sys.stdout.flush()
        # print('placed GluT')
        # self.checkNetCons()
        PAPGluT = [
            s.syn() for s in list(h.ncGluList) if s.postseg().sec in self.flattenPAP()
        ]
        if self.shell > 0:
            self.record_single_synapse = False
            PAPGluT = [
                s.syn() for s in list(h.ncGluList) if s.postseg().sec in list(h.slPAP)
            ]
            synapse_factor = self.shell_synapse / len(PAPGluT)
            if synapse_factor > 1:
                for s in list(h.ncGluList):
                    s.syn().density *= synapse_factor
            else:
                sample_ncs = random.sample(list(h.ncGluList), k=int(self.shell_synapse))
                for s in list(h.ncGluList):
                    if s in sample_ncs:
                        s.active(True)
                    else:
                        s.active(False)
                PAPGluT = [s for s in PAPGluT if s not in sample_ncs]

            for s in PAPGluT:
                s.density *= 2
                if self.shell == 1:
                    s.density *= 1.4  # Radelescu PCB (2022)
                if self.shell > 2:
                    s.density *= 2.3  # Radelescu PCB (2022)
            for i, ncs in enumerate([h.ncNMDAList, h.ncGABAaList]):
                synapse_factor = self.shell_synapse / len(list(ncs))
                if synapse_factor > 1:
                    for s in list(ncs):
                        s.syn().multiple *= synapse_factor
                else:
                    sample_ncs = random.sample(list(ncs), k=int(self.shell_synapse))
                    for s in list(ncs):  # sample_ncs:
                        if s in sample_ncs:
                            s.active(True)
                        else:
                            s.active(False)
                    if i == 1:
                        attr = "GABAas"
                    else:
                        attr = "NMDAs"
                    setattr(self, attr, [s.syn() for s in sample_ncs])

        recordDictArgs = {}
        if len(PAPGluT) > 0:
            recordDictArgs["sGluT"] = PAPGluT
        if self.Glu and len(self.NMDAs) > 0:
            recordDictArgs["sNMDA"] = self.NMDAs
        if self.GABA and len(self.GABAas) > 0:
            recordDictArgs["sGABA"] = self.GABAas
        self.record(**recordDictArgs)
        # print('setup Record')
        # sys.stdout.flush()
        #  voltage clamp for initialization
        if voltageClamp:
            h.tstop = self.initTstop / 10
            h.clampSwitch(1, -90)
            h.tstop = self.tstop

        # setup clamp protocols
        if self.mode > 2:
            if self.somaCheck:
                h.clampSwitch(3, self.currentClamp)
                self.initTstop -= 20
                h.ic.dur = self.tstop - self.initTstop - 40
            else:
                h.clampSwitch(2, self.currentClamp)
            h.ic.delay = self.initTstop

        else:
            if self.mode > 1 and self.cvode:
                h(f'fih = new FInitializeHandler("delaysec({self.initTstop})")')
                h.clampSwitch(5, self.voltageClamp)
            elif self.mode > 0:
                h.clampSwitch(0, self.currentClamp)
                h.ic.delay = self.initTstop + 10

        h.finitialize(self.v_init)
        h.fcurrent()

        # set gleak so RMP is v_init
        h.set_gleakNa(self.v_init)

        self.getKirCountPAP()
        self.getGLTCountPAP()
        self.getkin()
        # print('initialized')
        # sys.stdout.flush()
        h.fadvance()
        # self.checkGap()
        if video:
            self.makeVideo(self.varMorph, stop=self.initTstop)
        else:
            if voltageClamp:
                # for equilibriation purposes
                self.setK(dur=(self.initTstop / 10 - h.t))

            if self.cvode:
                h.continuerun(self.initTstop / 2)
                h.set_vgap(list(self.vSoma)[-1])
                h.continuerun(self.initTstop)
                if self.mode > 0 and self.mode != 2:
                    h.cvode.active(False)
            else:
                while h.t < self.initTstop:
                    if len(list(self.vPAP)) > 0 and np.isnan(list(self.vPAP)[-1]):
                        print("Encountered Nan in initialization")
                        print(list(self.vPAP))
                        sys.exit(-1)
                    h.fadvance()
        # print(list(self.KoSizePAP)[-1])
        self.RMP = list(self.vPAP)[
            -1
        ]  # consider last timepoint in initialization as RMP
        print(f"RMP:{self.RMP}")
        # print(f"EK: {list(self.ekSoma)[-1]}")
        # cvode.active(False)
        # self.ko_sim(False)
        if saveState:
            s = h.SaveState()
            s.save()
            with open(f"initializedState{rank}.dat", "wb") as f:
                s.fwrite(f)

    @staticmethod
    def gen_colors(total_shell):
        colors = [
            (
                255 * (1 - n / (total_shell - 1)),
                255 * (1 - n / (total_shell - 1)),
                255 * (1 - n / (total_shell - 1)),
                1,
            )
            for n in range(0, total_shell - 1)
        ]
        solid_colors = []
        bkg = (255, 255, 255)
        for r, g, b, a in colors:
            rgb = np.array((r, g, b)) * a + (1 - a) * np.array(bkg)
            rgb /= 255
            solid_colors.append(mcolors.to_rgba(tuple(rgb) + (1,)))
        return solid_colors

    def run(self, printRes=False, video=False, koclamp=None, noclear=False):
        # print('running')
        # sys.stdout.flush()
        # Clamp settings
        if not self.cvode and self.mode == 2:
            h.clampSwitch(1, self.voltageClamp)

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
        if not noclear:
            self.cleanMorphology()
        # print('ran simulation')
        # sys.stdout.flush()
        #
        #

    def getRMP(self):
        # decapreated
        # just call self.RMP
        print("function getRMP is decapreated")
        self.initialize()
        self.run()
        RMP = sum(list(self.vSoma)) / len(list(self.vSoma))
        self.RMP = RMP
        return RMP

    def makeVideo(self, var, frame_num=200, stop=None, zoom=False):
        if not hasattr(self, "frames"):
            self.frames = []
        if stop == None:
            stop = self.tstop
        if zoom:
            pap = self.flattenPAP()
        else:
            pap = None

        if type(var) is not list:
            var = [var]

        for v in var:
            if v == "v":
                clim = gl.lim_Vmemb
            elif v == "ko":
                clim = gl.lim_ko
            else:
                clim = None
            outfile = os.path.join(
                "../morphResults/",
                f"morph{v}_{zoom=}_{self.seed=}_{self.PAPCount=}_{self.tstop=}",
            )
            if hasattr(self, "SpikeFreq"):
                outfile += f"_{self.SpikeFreq}Hz_{self.SpikeNum=}"
            outfile += ".mp4"
            animate_morphology(
                tstop=stop,
                rangevar=v,
                outfile=outfile,
                zoom=pap,
                clim=clim,
                frame_num=(frame_num),
            )
        return

    def flattenPAP(self):
        # flatten self.PAPs to section list
        flattenPap = []
        for pap in self.PAPs:
            flattenPap += pap
        return h.SectionList(flattenPap)

    def plot_morph_iter(self):
        for name in ["v", "ko"]:
            self.plotWholecellVariable(
                name, f"{name}_{self.KoSize}_{rank}.pdf", zoom=False
            )

    def plotWholecellVariable(self, var, frameName, zoom=False):
        if "v" in var:
            clim = gl.lim_Vmemb
        elif "ko" in var:
            clim = gl.lim_ko
        else:
            clim = None
        if zoom:
            plot_3d_morphology(rangevar=var, zoom=self.flattenPAP(), clim=clim)
        else:
            # ps = h.plot_varMorph(var, frameName)
            plot_3d_morphology(rangevar=var, clim=clim)
        plt.savefig(frameName)

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
        if rank == 0:
            plot_3d_morphology(rangevar="diam")
            plt.savefig("DiamMap.pdf")
            plt.cla()
            plt.clf()
            plot_3d_morphology(rangevar="nseg")
            plt.savefig("nsegMap.pdf")

        # h.plot_varMorph("diam", "DiamMap.psf")
        # h.plot_varMorph("nseg", "nsegMap.psf")

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
                self.branches[-1].L = self.pbLen
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

    def set_list_record(self, attr, rec_attr, list_syn):
        setattr(self, attr, [])
        for s in list_syn:
            getattr(self, attr).append(h.Vector())
            getattr(self, attr)[-1].record(getattr(s, rec_attr))

    def record(self, sNMDA=None, sGABA=None, sGluT=None, toFile=False):
        # Function recording each individual aspect of astrocyte variable
        h.frecord_init()
        # Save Stuff
        if sNMDA != None:
            if self.record_single_synapse:
                if len(list(sNMDA)) > 1:
                    sNMDA = sNMDA[-1]
                self.iNMDA = h.Vector()
                # self.iNMDA.record(sNMDA._ref_iNMDA)
                self.iNMDA.record(sNMDA._ref_iNMDA_N2C)
            else:
                self.set_list_record("iNMDA", "_ref_iNMDA_N2C", sNMDA)

            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        if sGABA != None:
            if self.record_single_synapse:
                if len(list(sGABA)) > 1:
                    sGABA = sGABA[-1]
                self.iGABA = h.Vector()
                self.iGABA.record(sGABA._ref_iGaba)
            else:
                self.set_list_record("iGABA", "_ref_iGaba", sGABA)

            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")

        if sGluT != None:
            if self.record_single_synapse:
                if type(sGluT) is list:
                    sGluT = sGluT[-1]
                elif len(list(sGluT)) > 1:
                    sGluT = list(sGluT)[-1]
                self.iGluT = h.Vector()
                self.iGluT.record(sGluT._ref_iGluT)
            else:
                self.set_list_record("iGluT", "_ref_iGluT", sGluT)
            if toFile:
                self.iFile = h.File("iFile.dat")
                self.iFile.wopen("iFile.dat")
            if type(sGluT) is list:
                sGluT = sGluT[-1]
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
        self.flux.record(self.PAP(0.5)._ref_flux_change_k_acc)

        self.kbath = h.Vector()
        self.kbath.record(self.PAP(0.5)._ref_kbath_change_k_acc)

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
        # self.CaiPAP = h.Vector()
        # self.CaiPAP.record(self.PAP(0.5)._ref_cai)
        self.KiPAP = h.Vector()
        self.KiPAP.record(self.PAP(0.5)._ref_ki)

        if toFile:
            self.KoFile = h.File("KoFile.dat")
            self.KoFile.wopen("KoFile.dat")
            self.KiFile = h.File("KiFile.dat")
            self.KiFile.wopen("KiFile.dat")

        self.iKPAP = h.Vector()
        self.iKPAP.record(self.PAP(0.5)._ref_ik)

        # if hasattr(self.PAP(0.5), "_ref_ica"):
        #    self.iCaPAP = h.Vector()
        #    self.iCaPAP.record(self.PAP(0.5)._ref_ica)

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
        if self.PAP == self.soma:
            return []
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

    def setSlowing(self, slow, changeBaseline=None):
        if slow:
            h.setSlowing(self.flattenPAP(), slow)
        if changeBaseline is not None:
            h.changeBaseline(self.flattenPAP(), changeBaseline)

    def setPAP2Soma(self):
        self.PAP = h.soma
        h.convertSoma2slPAP()
        self.PAPs = [h.slPAP]
        self.soma = h.soma

    def setK(self, KoSize=None, mode="step", dur=0.5, delay=0):
        if dur == 0 or KoSize == 0:
            return
        restKo = self.Ko
        if KoSize == None:
            KoSize = self.KoSize
            # print(f"set Ko to {KoSize}\n")
        if mode == "pulse":
            h.continuerun(delay * ms + h.t)
            # print("setting KoSize to pulse mode")
            papk = self.getPAPK()
            h.setK(self.flattenPAP(), KoSize, papk + KoSize, 1)
            self.KoPAP[-1] = papk + KoSize
            h.fcurrent()
            h.continuerun(self.dt + h.t)
            h.setK(self.flattenPAP(), 0, restKo, 0)
        if mode == "step":
            # stop at one timestep before for cvode
            h.continuerun(delay * ms + h.t)
            if hasattr(h, "cvode"):
                h.cvode.active(False)
                h.dt = self.dt
            papk = self.getPAPK()
            h.setK(self.flattenPAP(), KoSize, KoSize + papk, 2)
            h.fcurrent()
            h.continuerun(dur * ms + h.t)
            # papk = self.getPAPK()
            h.setK(self.flattenPAP(), 0, restKo, 0)
            if hasattr(h, "cvode"):
                h.cvode.active(True)

        self.KoSize = KoSize

    def setKPoint(self, KoSize=None, mode="step", dur=0.5, delay=0, sec_range=None):
        if (
            sec_range is None
            and hasattr(self, "sec_range")
            and self.sec_range is not None
        ):
            sec_range = self.sec_range
        if dur == 0 or KoSize == 0:
            return
        restKo = self.Ko
        if KoSize == None:
            KoSize = self.KoSize
            # print(f"set Ko to {KoSize}\n")
        if mode == "pulse":
            h.continuerun(delay * ms + h.t)
            # print("setting KoSize to pulse mode")
            papk = self.getPAPK()
            if sec_range is None:
                h.setK_point(self.flattenPAP(), KoSize, papk + KoSize, 1)
            else:
                h.setK_range(self.flattenPAP(), KoSize, papk + KoSize, 1, sec_range)

            self.KoPAP[-1] = papk + KoSize
            h.fcurrent()
            h.continuerun(self.dt + h.t)
            h.setK(self.flattenPAP(), 0, restKo, 0)
        if mode == "step":
            # stop at one timestep before for cvode
            h.continuerun(delay * ms + h.t)
            if hasattr(h, "cvode"):
                h.cvode.active(False)
                h.dt = self.dt
            papk = self.getPAPK()
            if sec_range is None:
                h.setK_point(self.flattenPAP(), KoSize, KoSize + papk, 2)
            else:
                h.setK_range(self.flattenPAP(), KoSize, KoSize + papk, 2, sec_range)
            h.fcurrent()
            h.continuerun(dur * ms + h.t)
            # papk = self.getPAPK()
            h.setK(self.flattenPAP(), 0, restKo, 0)
            if hasattr(h, "cvode"):
                h.cvode.active(True)

        self.KoSize = KoSize

    def set_ECS(self, angs, scale=True):
        if scale:
            h.setECS(angs, 1)
        else:
            h.setECS(angs, 0)

    def clamp_ki(self, clamp):
        if clamp:
            h.set_gap_k(0)
            # under the assumption the astrocyte network equilibriates ki
            h.ki_clamp(1)

    def define_shell(self, total_shell=5, synapse=10000):
        self.total_shell = total_shell
        self.shell_synapse = synapse
        removed = h.define_shell(total_shell, synapse)
        self.removed = list(removed)

        # [print(len(list(i))) for i in h.shell_compartments]
        h.clampSwitch(1, -40)
        if self.shell > 0 and rank == 0 and hasattr(self, "total_shell"):
            plt.cla()
            plt.clf()
            solid_colors = PAPModel.gen_colors(self.total_shell)
            plot_3d_morphology(
                rangevar="num_shell",
                add_shell=self.total_shell,
                colormap_name=solid_colors,
                add_null=True,
            )
            plt.savefig(
                os.path.join(
                    "../morphResults/", f"defined_shell_{self.total_shell}.pdf"
                )
            )

    # only for rank 0

    def record_VClampI(self):
        self.VClampI = h.Vector()
        vc = h.electrodeList[-1]
        self.VClampI.record(vc._ref_i)

    def select_shell(self, scale=False):
        # clean up all previous synapses
        i = self.shell
        if i == 0:
            return
        h.set_all_pp(0)
        for channel in ["NMDAs", "GABAas", "GluTs"]:
            if hasattr(self, channel):
                delattr(self, channel)

        if scale:
            scale_multiple = 1 + self.removed[i] / len(list(h.slPAP))
        else:
            scale_multiple = 1

        PAPSecLen = 0
        for pap in list(self.PAPs[0]):
            if pap.L > 1:
                PAPSecLen += int(pap.L)
            else:
                PAPSecLen += 1
        mask = 0
        if self.multiple == 0:
            if self.GABA:
                mask = 4
        else:
            mask = 5

        if self.shell == self.total_shell:
            scale_multiple *= self.total_shell - mask - 1
            self.shell_synapse *= self.total_shell - mask - 1
        mode = 0

        if self.multiple == 0:
            if self.GABA:
                self.GABACount *= scale_multiple
                self.GABACount /= PAPSecLen
                mode = 1
        else:
            self.multiple *= scale_multiple
            self.multiple /= PAPSecLen
            mode = 2
        if "GluTrans" in self.GENEDict.keys() and self.GENEDict["GluTrans"] != None:
            # self.GENEDict["GluTrans"] *= scale_multiple
            mode = 0

        h.select_shell(i, mode)
        self.PAParea = 0
        for s in h.slPAP:
            for sec in s:
                self.PAParea += sec.area()

    def set_pb(self):
        h.pb(self.pbLen)

    def setKBath_iter(self, **kwargs):
        self.setKBath(self.KoSize, **kwargs)

    def setKBath(
        self,
        Ko,
        dur=100,
        delay=0,
        isolate=False,
        tsnap=False,
        video=False,
        clamp_ki=False,
    ):
        if h.t + delay < h.tstop:
            h.continuerun(delay * ms + h.t)
        if hasattr(h, "cvode"):
            h.dt = self.dt
        if not isolate:
            self.set_gapBath(True)
        papk = self.getPAPK()
        if hasattr(h, "cvode"):
            h.cvode.active(False)
            h.dt = self.dt
        if isolate:
            h.setK(self.flattenPAP(), Ko, papk + Ko, 2)
        else:
            h.setK(h.getWholetree(), Ko, papk + Ko, 2)

        if clamp_ki:
            if hasattr(self, "Dk_kdifl"):
                self.clamp_ki(True)

        h.psection(sec=self.soma)
        h.psection(sec=self.PAP)
        h.fcurrent()
        if video:
            self.makeVideo(
                self.varMorph,
                stop=dur * ms + h.t,
                frame_num=2,
                zoom=False,
            )

        else:
            h.continuerun(min(dur * ms + h.t, h.tstop))
            if tsnap and rank == 0:
                plot_3d_morphology(rangevar="v", clim=gl.lim_ek)
                plt.savefig(os.path.join("../morphResults", f"kbath_v.pdf"))

        self.Ko = Ko
        if isolate:
            h.setK(self.flattenPAP(), Ko - papk, Ko, 0)
        else:
            h.setK(h.getWholetree(), Ko - papk, Ko, 0)
        if hasattr(h, "cvode"):
            h.cvode.active(True)

    def GABABath(self, number, freq, video=False):
        self.set_cvode()
        self.setStimStart()
        # all section besides soma
        h.slPAP = h.SectionList(
            [sec for sec in self.soma.wholetree() if sec != self.soma]
        )
        # by default GABA count is uniformly distributed among all PAP sl.
        # since the whole astrocyte is manipulated as PAP we convert to all slPAP having the initially defined GABACount
        numSecs = len(list(h.slPAP))
        self.GABACount *= numSecs
        self.setGap()
        self.setGluTs()
        self.setGABAas()
        self.KoSize = 0

        self.record()
        h.finitialize(self.v_init)
        h.fcurrent()
        self.getkin()
        h.set_gleakNa(self.v_init)
        h.fcurrent()
        h.continuerun(self.initTstop * ms)

        self.multiSpike(number=number, freq=freq, video=video)

    def setKClearance(self, mode):
        if type(mode) == bool:
            if not mode:
                h.kbath_off()
            else:
                h.kbath_on()
        elif type(mode) == float or type(mode) == int:
            h.kbath_rule(mode)

    def replayK(self, fileName, isolate=False, video=False, setStop=None):
        df = pd.read_csv(fileName)
        baselineK = self.getPAPK()
        df["k"] += baselineK
        df["t"] += h.t
        # self.dt = 10*math.floor(math.log(max(df['t']),10) - 2)
        # h.dt = self.dt
        if setStop != None:
            self.setTstop(setStop)
        else:
            # get Max
            maxT = max(df["t"])
            maxT -= maxT % self.dt
            # remove remainder
            maxT += self.dt * 6
            # add 6 timesteps
            self.setTstop(maxT)

        for i, (_, row) in enumerate(df.iterrows()):
            t = int(row["t"])
            t -= t % self.dt
            k = row["k"]
            if setStop != None:
                if t >= setStop:
                    if h.t >= setStop:
                        break
                    else:
                        t = setStop

            # print(t,k)
            if video:
                self.makeVideo(
                    self.varMorph,
                    stop=t * ms + h.t,
                    frame_num=self.tstop / 10,  # sample at 10 ms interval
                    zoom=True,
                )

            else:
                h.continuerun(t * ms)
            h.fcurrent()
            papK = self.getPAPK()
            if isolate:
                h.setK(self.flattenPAP(), k - papK, k, 2)
            else:
                h.setK(h.getWholetree(), k - papK, k, 2)
            h.fcurrent()

    def LambdaEq(self, ra, rm, d):
        return (rm * d / ra / 4) ** 0.5  # um

    def spaceConstant(self):
        # TODO: double check if this is correct
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

    def calcPAPRi(self, direct=True, all=False, cleanMorph=False):
        if all:
            measure_pap = self.flattenPAP()
        else:
            measure_pap = [self.PAP]
        for pap_sec in measure_pap:
            RiSec = self.getSecbyName(str(pap_sec))
            RiSec.insert("inputRes")
            if direct:
                dur = 50
                self.setTstop(self.initTstop + dur)
                h.measure_input_resistance_direct(
                    self.initTstop, self.initTstop + dur, sec=RiSec
                )
                self.initialize()
                # get current vector length, get minimum for time after
                init_len = len(list(self.vPAP))
                self.run(noclear=True)
                RiSec.Ri_inputRes = (min(list(self.vPAP)[init_len:]) - self.RMP) / -0.01
            else:
                # self.initialize()
                h.measure_input_resistance(sec=RiSec)
                print(self.PAP_name, RiSec.Ri_inputRes)
            if not hasattr(self, "PAP_Ri"):
                self.PAP_Ri = float(RiSec.Ri_inputRes)
            else:
                if type(self.PAP_Ri) == float:
                    self.PAP_Ri = tuple((self.PAP_Ri,))
                self.PAP_Ri += (float(RiSec.Ri_inputRes),)
            if direct and cleanMorph:
                self.cleanMorphology()

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
            # h.plot_varMorph("Ri", "RiMap.psf")
            if rank == 0:
                plot_3d_morphology(rangevar="Ri")
                plt.savefig("RiMap.pdf")

    def mapRi(self, sectionDict):
        for k, v in sectionDict.items():
            RiSec = self.getSecbyName(k)
            RiSec.insert("inputRes")
            RiSec.Ri_inputRes = v
        # h.plot_varMorph("Ri_inputRes", "RiMap.psf")
        if rank == 0:
            plot_3d_morphology(rangevar="Ri_inputRes")
            plt.savefig("RiMap.pdf")

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
