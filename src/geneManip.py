"""
Class that manipulates channel densities.
Any channel desired for manipulation must have a
[channelname]Change method within GENEManipulation class.

[channelname] should match the suffix of NEURON MODL
"""

from textSDIO import *
import math


class GENEManipulation:
    compartment = object()
    GENE = dict()
    PAPs = object()

    # specific functions to manipulate the expression of a GENE
    # i.e. manipulate Kir by increasing conductance

    def kir4Change(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.kir4.Pkir = seg.kir4.Pkir / multiple
                if seg.kir4.Pkir > 1.0:
                    seg.kir4.Pkir = 1.0

    def kir2Change(self, multiple):
        # get PAP area
        PAParea = 0
        for pap in self.PAPs:
            for sec in pap:
                for seg in sec:
                    PAParea += seg.area()
                    # set channel count to uniform density
        PAParea /= len(self.PAPs)
        if PAParea == 0:
            # Uniform distribution to standard pap area
            PAParea = 0.3 * 0.05 ** 2 * math.pi
        for sec in self.compartments:
            for seg in sec:
                # print(sec)
                # print(seg.area()/PAParea)
                seg.kir2.gkbar = seg.kir2.gkbar * multiple * seg.area() / PAParea

    def twikChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.twik.PBkp = seg.twik.PBkp * multiple

    def GluTransChange(self, multiple):
        pass

    # No longer membrane mechanism
    # for sec in self.compartments:
    #     for seg in sec:
    #         seg.GluTrans.count = multiple * seg.GluTrans.count_std

    def kleakChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.kleak.gleak = seg.kleak.gleak * multiple

    def naleakChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.naleak.gleak = seg.naleak.gleak * multiple

    def clleakChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.clleak.gleak = seg.clleak.gleak * multiple

    def kpumpChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.kpump.Kp = seg.kpump.Kp * multiple

    def kir2DistChange(self, multiple):
        # print('Checking Compartments')
        for sec in self.compartments:
            if sec not in list(self.PAPs):
                for seg in sec:
                    # print(f'changing to {multiple}')
                    seg.kir2.gkbar = seg.kir2.gkbar * multiple
            # else:
            #     print('found PAP')


class GENExpression(GENEManipulation):
    compartments = object()
    GENE = dict()
    PAPs = object()

    # class that actually calls and manipulates the
    # necessary functions to alter expressions and what not

    def __init__(self, compartments, PAPs, GENE):
        if GENE == None:
            return
        self.compartments = compartments
        self.PAPs = PAPs

        if type(GENE) == dict:
            self.GENE = GENE
        else:
            eMessage(f"GENE{GENE} should be dict type")
        self.checkExpressionStatement()
        for gName, xfold in GENE.items():
            self.changeExpression(gName, xfoldXpression=xfold)

    def changeExpression(self, gName, xfoldXpression=None):
        if hasattr(self, f"{gName}Change"):  # check if method is implemented
            if xfoldXpression == None:
                xfoldXpression = self.GENE[gName]
            exec(f"self.{gName}Change({xfoldXpression})")
        else:
            wMessage(f"No {gName} skipped")

    def alterDistribution(self, gName, ratioToPAP=1):
        # print(gName)
        # print(self.GENE.keys())
        if gName in self.GENE.keys():
            self.changeExpression(f"{gName}Dist", xfoldXpression=ratioToPAP)

    def checkExpressionStatement(self):
        for gName, multiple in self.GENE.items():
            if type(gName) != str and type(multiple) != float:
                eMessage(f"GENE{GENE} should be name and float")
