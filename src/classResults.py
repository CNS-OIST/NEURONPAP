import copy
import numpy as np

class ResultsPAPModel():
    # class to copy data of simulations
    # necessary for MPI copy as NEURON components cannot be copied

    dont_copy = [
        "PAP",
        "soma",
        "branches",
        "branch",
        "NMDAs",
        "NCs"
        ]
    currents = [
        "iKPAP",
        "iClPAP",
        "iNaPAP",
        "iKSoma"
        ]

    def copyAttr(self):
        # print("copying to result class")
        newInstance = ResultsPAPModel()
        newInstance.__dict__ = {
            attr: copy.deepcopy(self.__dict__[attr])
            # print(attr)
            for attr in self.__dict__
            if attr not in self.dont_copy
        }
        # total current calculate mA/cm2 to nA
        for attr in newInstance.__dict__:
            if attr in self.currents:
                newInstance.__dict__[attr] = np.array(
                    newInstance.__dict__[attr]
                ) * self.PAParea * 0.01
        return newInstance

    def getRMP(self):
        RMP = list(self.vSoma)[-1]
        return RMP
