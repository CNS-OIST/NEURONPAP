import copy
import numpy as np
from neuron import h


class ResultsPAPModel:
    # class to copy data of simulations
    # necessary for MPI copy as NEURON components cannot be copied

    dont_copy = [
        "PAP",
        "PAPs",
        "soma",
        "branches",
        "branch",
        "NMDAs",
        "GluTs",
        "GABAas",
        "NCs",
        "GENEobj",
        "equiDistSec",
    ]
    memcurrents = [
        "iKPAP",
        "iClPAP",
        "iNaPAP",
        "iMemPAP",
        "iKSoma",
        "iClSoma",
        "iNaSoma",
        "iMemSoma",
        "iNCXPAP"
        # "iNMDA"
    ]
    ppcurrents = ["iGluT", "iNMDA", "iGluTSoma", "iGABA"]

    def copyAttr(self):
        # print("copying to result class")
        newInstance = ResultsPAPModel()
        # print(self.__dict__)
        newInstance.__dict__ = {
            attr: copy.deepcopy(self.__dict__[attr])
            for attr in self.__dict__
            if attr not in self.dont_copy
        }
        # total current calculate mA/cm2 to nA
        for attr in newInstance.__dict__:
            if attr in self.memcurrents:
                if "PAP" in attr:
                    area = self.PAParea
                else:
                    area = self.somaArea
                newInstance.__dict__[attr] = (
                    np.array(newInstance.__dict__[attr]) * area * 0.01
                )

        # nA to pA
        currList = self.memcurrents + self.ppcurrents
        for attr in newInstance.__dict__:
            if attr in currList:
                newInstance.__dict__[attr] = np.array(newInstance.__dict__[attr]) * 1e3

        # Convert Vector to list
        # for attr in newInstance.__dict__:
        #     if isinstance(newInstance.__dict__[attr],h.Vector):
        #         newInstance.__dict__[attr] = list(newInstance.__dict__[attr])
        return newInstance

    def getRMP(self):
        RMP = list(self.vSoma)[-1]
        return RMP
