from textSDIO import *

class GENEManipulation:
    compartment = object()
    GENE = dict()

    # specific functions to manipulate the expression of a GENE
    # i.e. manipulate Kir by increasing conductance

    def kir4Change(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.kir4.Pkir = seg.kir4.Pkir / multiple
                if seg.kir4.Pkir > 1.0:
                    seg.kir4.Pkir = 1.0

    def kir2Change(self, multiple):
        # get PAP
        PAP = [sec for sec in self.compartments if str(sec) == "PAP"][0]
        # get PAP area
        PAParea = 0
        for seg in PAP:
            PAParea += seg.area()
        # set channel count to uniform density
        for sec in self.compartments:
            for seg in sec:
                # print(sec)
                # print(seg.area()/PAParea)
                seg.kir2.gkbar = seg.kir2.gkbar * multiple * seg.area() / PAParea

    def twikChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.twik.PBkp = seg.twik.PBkp * multiple

    def kleakChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.kleak.gleak = seg.kleak.gleak * multiple

    def naleakChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.naleak.gleak = seg.naleak.gleak * multiple

    def kpumpChange(self, multiple):
        for sec in self.compartments:
            for seg in sec:
                seg.kpump.Kp = seg.kpump.Kp * multiple


class GENExpression(GENEManipulation):
    compartments = object()
    GENE = dict()

    # class that actually calls and manipulates the
    # necessary functions to alter expressions and what not

    def __init__(self, compartments, GENE):
        if GENE == None:
            return
        self.compartments = compartments

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
