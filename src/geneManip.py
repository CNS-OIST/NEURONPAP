from textSDIO import *

class GENEManipulation:
    compartment = object()
    GENE = dict()

    # specific functions to manipulate the expression of a GENE
    # i.e. manipulate Kir by increasing conductance

    def kir4Change(self, multiple):
        for seg in self.compartment:
            seg.kir4.Pkir = seg.kir4.Pkir / multiple
            if seg.kir4.Pkir > 1.0:
                seg.kir4.Pkir = 1.0

    def kir2Change(self, multiple):
        for seg in self.compartment:
            seg.kir2.gkbar = seg.kir2.gkbar * multiple


class GENExpression(GENEManipulation):
    compartment = object()
    GENE = dict()

    # class that actually calls and manipulates the
    # necessary functions to alter expressions and what not

    def __init__(self, compartment, GENE):
        if GENE == None:
            return
        self.compartment = compartment

        if type(GENE) == dict:
            self.GENE = GENE
        else:
            eMessage(f"GENE{GENE} should be dict type")
        self.checkExpressionStatement()
        for gName, xfold in GENE.items():
            self.changeExpression(gName, xfoldXpression=xfold)

    def changeExpression(self, gName, xfoldXpression=None):
        if gName in self.GENE.keys():
            if xfoldXpression == None:
                xfoldXpression = self.GENE[gName]
            if hasattr(self, f"{gName}Change"):  # check if method is implemented
                exec(f"self.{gName}Change({xfoldXpression})")
            else:
                wMessage(f"No {gName} skipped")

    def checkExpressionStatement(self):
        for gName, multiple in self.GENE.items():
            if type(gName) != str and type(multiple) != float:
                eMessage(f"GENE{GENE} should be name and float")
