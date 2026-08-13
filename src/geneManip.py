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
        """Scale the kir4 channel's 'multiple' parameter by the given factor across all compartments and segments."""
        for sec in self.compartments:
            for seg in sec:
                seg.kir4.multiple = multiple

    def kir2Change(self, multiple):
        """Scale the kir2 channel's 'multiple' parameter by the given factor across all compartments and segments."""
        for sec in self.compartments:
            for seg in sec:
                seg.kir2.multiple = multiple

    def twikChange(self, multiple):
        """Scale the twik channel's PBkp conductance by the given factor across all compartments and segments."""
        for sec in self.compartments:
            for seg in sec:
                seg.twik.PBkp = seg.twik.PBkp * multiple

    def GluTransChange(self, multiple):
        """No-op placeholder; the glutamate transporter is no longer implemented as a membrane mechanism, so there is nothing to scale here."""
        pass

    # No longer membrane mechanism
    # for sec in self.compartments:
    #     for seg in sec:
    #         seg.GluTrans.count = multiple * seg.GluTrans.count_std
    #
    def kleakChange(self, multiple):
        """Scale the potassium leak channel's conductance (kleak.gleak) by the given factor across all compartments and segments."""
        for sec in self.compartments:
            for seg in sec:
                seg.kleak.gleak = seg.kleak.gleak * multiple

    def naleakChange(self, multiple):
        """Scale the sodium leak channel's conductance (naleak.gleak) by the given factor across all compartments and segments."""
        for sec in self.compartments:
            for seg in sec:
                seg.naleak.gleak = seg.naleak.gleak * multiple

    def clleakChange(self, multiple):
        """Scale the chloride leak channel's conductance (clleak.gleak) by the given factor across all compartments and segments."""
        for sec in self.compartments:
            for seg in sec:
                seg.clleak.gleak = seg.clleak.gleak * multiple

    def nakpumpChange(self, multiple):
        """Scale the Na/K pump's total pump rate by the given factor across all compartments and segments."""
        for sec in self.compartments:
            for seg in sec:
                seg.nakpump.totalpump = seg.nakpump.totalpump * multiple

    def kir2DistChange(self, multiple):
        """Scale the kir2 channel's gkbar conductance by the given factor across all compartments and segments, excluding those that belong to the PAPs."""
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
        """Store compartments and PAPs, validate that GENE is a dict, and apply each non-None fold-expression value in GENE via changeExpression."""
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
            if xfold is not None:
                self.changeExpression(gName, xfoldXpression=xfold)

    def changeExpression(self, gName, xfoldXpression=None):
        """Call the gene's '<gName>Change' method with the given (or stored GENE) fold-expression value if it is implemented, otherwise log that the gene was skipped."""
        if hasattr(self, f"{gName}Change"):  # check if method is implemented
            if xfoldXpression == None:
                xfoldXpression = self.GENE[gName]
            getattr(self, f"{gName}Change")(xfoldXpression)
        else:
            wMessage(f"No {gName} skipped")

    def alterDistribution(self, gName, ratioToPAP=1):
        """If the given gene is present in GENE, call its '<gName>Dist' change method to set its distribution ratio relative to the PAP."""
        # print(gName)
        # print(self.GENE.keys())
        if gName in self.GENE.keys():
            self.changeExpression(f"{gName}Dist", xfoldXpression=ratioToPAP)

    def checkExpressionStatement(self):
        """Validate that each entry in GENE has the expected name/float types, logging an error message for any entry that fails the check."""
        for gName, multiple in self.GENE.items():
            if type(gName) != str and type(multiple) != float:
                eMessage(f"GENE{GENE} should be name and float")
